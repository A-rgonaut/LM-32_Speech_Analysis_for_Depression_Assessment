import torch
import torch.nn as nn
from tqdm import tqdm
import os
from sklearn.model_selection import ParameterGrid

from .config import SSLConfig
from .data_loader import DataLoader
from .model import SSLModel
from ..src_utils  import EarlyStopping, get_metrics

class Trainer():
    def __init__(self, model : SSLModel, train_loader : DataLoader, val_loader: DataLoader, config: SSLConfig):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        tot_loss, correct_predictions = 0, 0
        for batch in tqdm(self.train_loader):
            inputs = batch['features'].to(self.device)
            label = batch['label'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model({'features': inputs, 'attention_mask': attention_mask})
            loss = self.criterion(outputs.squeeze(1), label)
            tot_loss += loss.item()
            preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            correct_predictions += torch.sum(preds == label)
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        avg_loss = tot_loss / len(self.train_loader)
        accuracy = correct_predictions.double() / len(self.train_loader.dataset)
        return avg_loss, accuracy

    def validate_epoch(self):
        self.model.eval()

        total_loss = 0
        predictions, targets = [], []
    
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                inputs = batch['features'].to(self.device)  
                label = batch['label'].to(self.device)    
                attention_mask = batch['attention_mask'].to(self.device)     
                outputs = self.model({'features': inputs, 'attention_mask': attention_mask})

                loss = self.criterion(outputs.squeeze(1), label)
                total_loss += loss.item()

                preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()

                predictions.extend(preds.cpu().numpy())
                targets.extend(label.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = get_metrics(targets, predictions, 'f1_macro', 'f1_depression', 'accuracy')

        return avg_loss, metrics['accuracy'], metrics['f1_macro'], metrics['f1_depression']

    def train(self, experiment):
        early_stopping = EarlyStopping(
            patience = self.config.early_stopping_patience,
            min_delta = self.config.early_stopping_min_delta,
            mode = self.config.early_stopping_mode
        )

        best_model_weights, best_epoch, best_val_f1 = None, -1, -float('inf')
        experiment.set_model_graph(self.model)

        for epoch in range(self.config.epochs): 
            train_loss, train_acc = self.train_epoch()
            experiment.log_metric("train/loss", train_loss, step=epoch)
            experiment.log_metric("train/accuracy", train_acc, step=epoch)

            val_loss, val_acc, val_f1_macro, val_f1_depression = self.validate_epoch()
            experiment.log_metric("val/loss", val_loss, step=epoch)
            experiment.log_metric("val/accuracy", val_acc, step=epoch)
            experiment.log_metric("val/f1_macro", val_f1_macro, step=epoch)
            experiment.log_metric("val/f1_depression", val_f1_depression, step=epoch)

            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | " + \
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1_macro:.4f} | Val F1 Depression: {val_f1_depression:.4f}")

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                best_epoch = epoch
                best_model_weights = self.model.state_dict().copy()
                print(f"New best F1: {best_val_f1:.4f} (model saved)")

            if early_stopping(val_f1_macro):
                print(f"Early stopping activated after {epoch+1} epochs.")
                break

        if best_model_weights is None:
            best_model_weights = self.model.state_dict().copy()
        
        if not os.path.exists(os.path.dirname(self.config.model_save_path)):
            os.makedirs(os.path.dirname(self.config.model_save_path))
        torch.save(best_model_weights, self.config.model_save_path)

        return early_stopping.best_score, best_epoch

    # TODO da sistemare, non funziona
    def train_grid(self, experiment):
        results = []
        grid_params = ParameterGrid(self.config.grid_params)

        for params in grid_params:
            # Aggiorna il modello e gli iperparametri
            self.model.init_weights()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params['learning_rate'])

            # Esegui l'addestramento
            best_val_f1, best_epoch = self.train(self.config.grid_params, experiment)
            result = {
                'params': params,
                'best_val_f1': best_val_f1,
                'best_epoch': best_epoch
            }
            results.append(result)
            experiment.log_parameters(params)
            experiment.log_metrics({"best_val_f1": best_val_f1, "best_epoch": best_epoch})
        
        return results