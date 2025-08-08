import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

from .config import CNNConfig
from .data_loader import DataLoader
from .model import CNNModel
from ..utils  import EarlyStopping, get_metrics

class Trainer():
    def __init__(self, model : CNNModel, train_loader : DataLoader, val_loader: DataLoader, config: CNNConfig):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()
        self.model.to(self.device)
        self.eval_strategy = config.eval_strategy

    def train_epoch(self):
        self.model.train()
        tot_loss = 0
        for batch in tqdm(self.train_loader, desc="Training Epoch"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['label']
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss = self.criterion(outputs, labels)
            tot_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        avg_loss = tot_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self):
        self.model.eval()

        total_loss = 0
        session_targets = {}
        session_scores = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating Epoch"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                audio_ids = batch.pop('audio_id')
                labels = batch['label']

                outputs = self.model(batch)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                scores = torch.sigmoid(outputs)

                for idx in range(len(audio_ids)):
                    session_id = audio_ids[idx].item()
                    score = scores[idx].item()
                    target = labels[idx].item()

                    if session_id not in session_scores:
                        session_scores[session_id] = []
                    
                    session_scores[session_id].append(score)
                    session_targets[session_id] = target

        final_predictions, final_targets, final_scores = [], [], []
        
        for session_id in session_scores:
            if self.eval_strategy == 'average':
                avg_score = np.mean(session_scores[session_id])
                predicted_label = 1 if avg_score > 0.5 else 0
                final_scores.append(avg_score)
            elif self.eval_strategy == 'majority':
                segment_predictions = [1 if score > 0.5 else 0 for score in session_scores[session_id]]
                predicted_label = max(set(segment_predictions), key=segment_predictions.count)
                final_scores.append(np.mean(session_scores[session_id]))

            final_predictions.append(predicted_label)
            final_targets.append(session_targets[session_id])

        avg_loss = total_loss / len(self.val_loader)
        
        metrics = get_metrics(final_targets, final_predictions, 'f1_macro', 'f1_depression', 'accuracy', y_score=final_scores)

        return avg_loss, metrics['accuracy'], metrics['f1_macro'], metrics['f1_depression']

    def train(self, experiment, model_filename):
        experiment.set_model_graph(self.model)
        
        early_stopping = EarlyStopping(
            patience = self.config.early_stopping_patience,
            min_delta = self.config.early_stopping_min_delta,
            mode = self.config.early_stopping_mode
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
                print(f"New best F1 Macro: {best_val_f1:.4f}. Saving model...")
                self.save_model(model_filename)

            if early_stopping(val_f1_macro):
                print(f"Early stopping activated after {epoch+1} epochs.")
                break

        print(f"\nTraining finished. Best model saved to {model_filename} (Epoch: {best_epoch}, F1 Macro: {best_val_f1:.4f})")
    
    def save_model(self, model_filename):
        if not os.path.exists(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)
        model_save_path = os.path.join(self.config.model_save_dir, model_filename)
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")