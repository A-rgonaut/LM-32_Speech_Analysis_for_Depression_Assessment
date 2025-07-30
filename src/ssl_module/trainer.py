
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils  
import torch
import src_utils
import torch.nn as nn
from tqdm import tqdm
from model import SSLModel
from config import SSLConfig
from data_loader import DataLoader
from collections import defaultdict

class Trainer():
    
    def __init__(self, model : SSLModel, train_loader : DataLoader, val_loader: DataLoader, config: SSLConfig):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_epoch(self):

        self.model.train()

        tot_loss = 0.0
        y_pred, y_true = [], []

        for batch in tqdm(self.train_loader):
            input = batch['input_values'].to(self.device)
            label = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(input)
            loss = self.criterion(output, label)
            loss.backward()

            self.optimizer.step()

            tot_loss += loss.item()
            preds = (output > 0.5).float()

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(label.cpu().numpy())

        print(src_utils.get_metrics(y_true, y_pred, 'precision', 'recall', 'f1_score', 'accuracy', 'roc_auc'))

        return tot_loss / len(self.train_loader)

    def validate_epoch(self, seg_threshold = 0.5, tie_positive = True):
        
        self.model.eval()
        
        total_loss = 0.0
        have_audio_id = False
        counts_pos = defaultdict(int)
        counts_tot = defaultdict(int)
        true_lbl = {}

        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                inputs = batch['input_values'].to(self.device)  # [B, C, T]
                labels = batch['label'].to(self.device)         # [B, 1] float
                outputs = self.model(inputs)                    # [B, 1] prob (Sigmoid già nel modello)

                # segment loss
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                total_loss += loss.item()

                # majority: raccogli predizioni segmentali
                # converto prob -> 0/1
                seg_preds = (outputs.squeeze(-1) >= seg_threshold).long()  # [B]
                seg_tgts = (labels.squeeze() >= 0.5).long()                 # [B]

                if 'audio_id' in batch:
                    have_audio_id = True
                    audio_ids = batch['audio_id']  # list/tensor CPU
                    # assicurati di poter iterare
                    if torch.is_tensor(audio_ids):
                        audio_ids = audio_ids.tolist()

                    for aid, p, t in zip(audio_ids, seg_preds.cpu().tolist(), seg_tgts.cpu().tolist()):
                        counts_tot[aid] += 1
                        counts_pos[aid] += int(p)
                        # memorizza la vera etichetta (tutti uguali per quell'audio)
                        true_lbl[aid] = int(t)

        avg_loss = total_loss / len(self.val_loader)

        # Calcolo majority solo se abbiamo audio_id
        if have_audio_id:
            audio_preds = {}

            for aid in counts_tot:
                n_pos = counts_pos[aid]
                n_neg = counts_tot[aid] - n_pos

                # TODO Questo condizione non fa la stessa cosa? A meno che sia scritta male #
                if tie_positive:
                    pred = 1 if n_pos >= n_neg else 0
                else:
                    pred = 1 if n_pos > n_neg else 0

                audio_preds[aid] = pred

            # metriche audio-level di base (accuracy)
            y_true = []
            y_pred = []

            for aid in sorted(audio_preds):
                y_true.append(true_lbl[aid])
                y_pred.append(audio_preds[aid])

            y_true = torch.tensor(y_true)
            y_pred = torch.tensor(y_pred)
            audio_acc = (y_true == y_pred).float().mean().item()
            print(src_utils.get_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy(), 'precision', 'recall', 'f1_score', 'accuracy', 'roc_auc'))

            print(f'Validation majority-vote audio accuracy: {audio_acc:.4f} (audio_count={len(y_pred)})')
        else:
            print("Validation majority-vote skipped (val_dataset creato senza return_audio_id=True).")

        return avg_loss

    def train(self, experiment):
       
        early_stopping = utils.EarlyStopping(
            patience = self.config.early_stopping_patience,
            min_delta = self.config.early_stopping_min_delta,
            mode = self.config.early_stopping_mode
            )
        
        best_val_loss = float('inf')

        self.model.to(self.device)
        experiment.set_model_graph(self.model)

        for epoch in range(self.config.epochs): 
            train_loss = self.train_epoch()
            dev_loss = self.validate_epoch()

            experiment.log_metric("train/loss", train_loss, step = epoch)
            experiment.log_metric("val/loss", dev_loss, step = epoch)

            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {dev_loss}')
            # Early stopping
            if dev_loss < best_val_loss:
                best_val_loss = dev_loss
                best_epoch = epoch
                best_model = self.model.state_dict()
                print(f'Saving model at epoch {epoch+1} with loss {dev_loss:.4f}')

            if early_stopping(dev_loss):
                print(f'Early stopping at epoch {epoch+1}')
                self.model.load_state_dict(best_model)
                break

        # salvo il modello migliore
        torch.save(self.model.state_dict(), 'best_SSL_model.pth')

        return best_val_loss, best_epoch

    def train_grid(self):
        history = self.model.fit(
            self.data['train'],
            validation_data=self.data['val'],
            epochs=self.config.epochs,
            callbacks=[]  # aggiungi EarlyStopping o ModelCheckpoint
        )
        return history
