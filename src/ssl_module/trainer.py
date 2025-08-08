import torch
import torch.nn as nn
from tqdm import tqdm
import os
from transformers import get_linear_schedule_with_warmup

from .config import SSLConfig
from .data_loader import DataLoader
from .model import SSLModel
from ..utils  import EarlyStopping, get_metrics

class Trainer():
    def __init__(self, model : SSLModel, train_loader : DataLoader, val_loader: DataLoader, config: SSLConfig):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss() 
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        tot_loss = 0
        accumulation_steps = self.config.gradient_accumulation_steps 
        self.optimizer.zero_grad()
        for i, batch in enumerate(tqdm(self.train_loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['label']

            outputs = self.model(batch)
            if torch.isnan(outputs).any():
                print("Skipping batch because of nan")
                # clear memory
                del batch
                del labels
                del outputs
                torch.cuda.empty_cache()
                continue

            loss = self.criterion(outputs, labels)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            tot_loss += loss.item() * accumulation_steps 

            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        avg_loss = tot_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()

        total_loss = 0
        predictions, targets = [], []
    
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch['label']
                outputs = self.model(batch)

                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()

                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        metrics = get_metrics(targets, predictions, 'f1_macro', 'f1_depression', 'accuracy')

        return avg_loss, metrics['accuracy'], metrics['f1_macro'], metrics['f1_depression']

    def train(self, experiment, model_filename):
        experiment.set_model_graph(self.model)

        early_stopping = EarlyStopping(
            patience = self.config.early_stopping_patience,
            min_delta = self.config.early_stopping_min_delta,
            mode = self.config.early_stopping_mode
        )

        accumulation_steps = self.config.gradient_accumulation_steps 
        num_training_steps = int(len(self.train_loader) * self.config.epochs) // accumulation_steps
        num_warmup_steps = int(num_training_steps * 0.10)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        best_epoch, best_val_f1 = -1, -float('inf')

        for epoch in range(self.config.epochs): 
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1_macro, val_f1_depression = self.validate_epoch()
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | " + \
                  f"Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1_macro:.4f} | Val F1 Depression: {val_f1_depression:.4f}")

            experiment.log_metric("train/loss", train_loss, step=epoch)
            experiment.log_metric("val/loss", val_loss, step=epoch)
            experiment.log_metric("val/accuracy", val_acc, step=epoch)
            experiment.log_metric("val/f1_macro", val_f1_macro, step=epoch)
            experiment.log_metric("val/f1_depression", val_f1_depression, step=epoch)

            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                best_epoch = epoch + 1
                print(f"New best F1: {best_val_f1:.4f}. Saving model...")
                self.save_model(model_filename)

            if early_stopping(val_f1_macro):
                print(f"Early stopping activated after {epoch+1} epochs.")
                break

        print(f"\nTraining finished for this fold. Best model saved to {model_filename} (Epoch: {best_epoch}, F1 Macro: {best_val_f1:.4f})")

    def save_model(self, model_filename):
        if not os.path.exists(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)
        model_save_path = os.path.join(self.config.model_save_dir, model_filename)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
