import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from .cnn_module.config import CNNConfig
from .utils import EarlyStopping, get_metrics, get_predictions, aggregate_predictions

class Trainer():
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=0.01)
        self.model_name = 'ssl'
        self.scheduler_step = ''
        if isinstance(config, CNNConfig):
            self.model_name = 'cnn'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss() 
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        if hasattr(self.model, 'ssl_model') and self.model.num_layers_to_unfreeze == 0:
            self.model.ssl_model.eval()
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
                if self.scheduler_step == 'batch':
                    self.scheduler.step()
                self.optimizer.zero_grad()

        avg_loss = tot_loss / len(self.train_loader)
        return avg_loss
    
    def validate_epoch(self):
        self.model.eval()

        session_scores, session_targets, avg_loss = get_predictions(
            self.val_loader, self.model, "Validation", self.device, self.criterion
        )

        final_predictions, final_targets, final_scores = aggregate_predictions(
            session_scores, session_targets, self.config.eval_strategy
        )
        
        metrics = get_metrics(final_targets, final_predictions, 'f1_macro', 'f1_depression', 'accuracy', y_score=final_scores)

        return avg_loss, metrics['accuracy'], metrics['f1_macro'], metrics['f1_depression']

    def train(self, experiment, model_filename):
        experiment.set_model_graph(self.model)

        early_stopping = EarlyStopping(
            patience = self.config.early_stopping_patience,
            min_delta = self.config.early_stopping_min_delta,
            mode = self.config.early_stopping_mode
        )

        if self.model_name == 'cnn':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
            self.scheduler_step = 'epoch'
        else:
            num_training_steps = int(len(self.train_loader) * self.config.epochs) // self.config.gradient_accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(num_training_steps * 0.1),
                num_training_steps=num_training_steps,
            )
            self.scheduler_step = 'batch'

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

            if self.scheduler_step == 'epoch': self.scheduler.step(val_f1_macro)

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