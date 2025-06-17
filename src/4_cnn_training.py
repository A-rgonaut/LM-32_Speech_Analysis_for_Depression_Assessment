#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils import load_labels_from_dataset, get_audio_paths


# In[ ]:


# --- Parametri di Configurazione dal Paper ---

# Parametri audio e di segmentazione (Pag. 8, colonna sx)
SR = 16000
SEGMENT_MS = 250  # Durata del segmento in ms
HOP_MS = 50       # Spostamento (shift) tra segmenti in ms

# Parametri di training (Tabella IV)
BATCH_SIZE = 200
LEARNING_RATE = 0.01
NUM_EPOCHS = 100          # Il paper usa 100 iterazioni, che interpretiamo come epoche
DROPOUT_RATE = 0.25

# Parametri di Early Stopping (Pag. 8, menzionato nel testo)
EARLY_STOPPING_PATIENCE = 5 # "five epochs with no improvement"

# Altri parametri
MODEL_SAVE_PATH = "parkinson_detection_cnn_best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {DEVICE}")

# Impostazione del seed per la riproducibilità
seed_value = 42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# In[ ]:


class CNNMLP(nn.Module):
    """
    Implementazione dell'architettura CNN+MLP descritta nella Tabella III del paper.
    La rete accetta in input segmenti di waveform audio (1D).
    """
    def __init__(self, dropout_rate=0.25, num_classes=2):
        super(CNNMLP, self).__init__()
        
        # Blocco Convoluzionale 1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Blocco Convoluzionale 2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=1),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Blocco Convoluzionale 3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Il blocco MLP viene inizializzato dinamicamente nel forward pass
        # per adattarsi alla dimensione dell'output delle CNN.
        self.flatten = nn.Flatten()
        self.mlp_block = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=1, out_features=128), # Inizializzato dinamicamente
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        self._mlp_initialized = False

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x_flattened = self.flatten(x)
        
        # Inizializzazione dinamica del primo layer MLP
        if not self._mlp_initialized:
            in_features = x_flattened.shape[1]
            self.mlp_block[1] = nn.Linear(in_features, 128).to(x.device)
            print(f"MLP inizializzato dinamicamente con {in_features} feature di input.")
            self._mlp_initialized = True
            
        output = self.mlp_block(x_flattened)
        return output


# In[ ]:


class AudioSegmentDataset(Dataset):
    """
    Dataset PyTorch per caricare file audio e suddividerli in segmenti
    secondo le specifiche del paper.
    """
    def __init__(self, file_paths, labels, sr, segment_ms, hop_ms):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.segment_length = int(sr * (segment_ms / 1000.0))
        self.hop_length = int(sr * (hop_ms / 1000.0))
        
        self.segments = []
        self.segment_labels = []
        
        print("Creazione dei segmenti dal dataset...")
        for i, file_path in enumerate(tqdm(self.file_paths, desc="Segmentazione Audio")):
            label = self.labels[i]
            try:
                waveform, _ = librosa.load(file_path, sr=self.sr)
                
                # Normalizzazione per evitare problemi con audio a basso volume
                if np.max(np.abs(waveform)) > 0:
                    waveform = waveform / np.max(np.abs(waveform))
                
                start = 0
                while start + self.segment_length <= len(waveform):
                    segment = waveform[start : start + self.segment_length]
                    self.segments.append(segment)
                    self.segment_labels.append(label)
                    start += self.hop_length
            except Exception as e:
                print(f"Errore durante l'elaborazione del file {file_path}: {e}")
        
        print(f"Creati {len(self.segments)} segmenti totali.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        label = self.segment_labels[idx]
        # Aggiunge la dimensione del canale (1) richiesta da Conv1d
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return segment_tensor, label_tensor


# In[ ]:


class EarlyStopping:
    """Attiva l'early stopping se la metrica di validazione non migliora."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False

    def __call__(self, current_score):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# In[ ]:


def train_epoch(model, data_loader, loss_fn, optimizer, device):
    """Esegue un'epoca di training."""
    model.train()
    total_loss, correct_predictions = 0, 0
    
    for batch_segments, batch_labels in tqdm(data_loader, desc="Training Epoch", leave=False):
        batch_segments = batch_segments.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_segments)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct_predictions += torch.sum(preds == batch_labels)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return avg_loss, accuracy.item()


# In[ ]:


def evaluate_model(model, file_paths, labels, device, sr, segment_ms, hop_ms):
    """
    Valuta il modello su file audio completi, mediando le predizioni dei segmenti,
    come descritto nel paper. Ritorna accuratezza, sensitività e specificità.
    """
    model.eval()
    all_targets = []
    all_predictions = []
    
    segment_length = int(sr * (segment_ms / 1000.0))
    hop_length = int(sr * (hop_ms / 1000.0))

    with torch.no_grad():
        for file_path, label in tqdm(zip(file_paths, labels), desc="Valutazione", total=len(file_paths), leave=False):
            try:
                waveform, _ = librosa.load(file_path, sr=sr)
                if np.max(np.abs(waveform)) > 0:
                    waveform = waveform / np.max(np.abs(waveform))
                
                segments = []
                start = 0
                while start + segment_length <= len(waveform):
                    segment = waveform[start : start + segment_length]
                    segments.append(segment)
                    start += hop_length
                
                if not segments: continue
                
                segments_tensor = torch.tensor(np.array(segments), dtype=torch.float32).unsqueeze(1).to(device)
                
                segment_outputs = model(segments_tensor)
                segment_probs = torch.softmax(segment_outputs, dim=1)
                
                avg_probs = torch.mean(segment_probs, dim=0)
                final_prediction = torch.argmax(avg_probs).item()
                
                all_predictions.append(final_prediction)
                all_targets.append(label)
            except Exception as e:
                print(f"Errore durante la valutazione del file {file_path}: {e}")

    if not all_targets:
        return 0.0, 0.0, 0.0, np.array([])
    
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    
    if len(cm.ravel()) == 4:
        tn, fp, fn, tp = cm.ravel()
    else: # Gestisce il caso di predizioni tutte di una classe
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(all_targets)) == 1: # Se ci sono solo etichette 0 o 1
             if np.unique(all_targets)[0] == 0: tn = cm[0,0]
             else: tp = cm[0,0]
        else: # Se ci sono entrambe le etichette ma la matrice è 1x1
             tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return accuracy, sensitivity, specificity, cm, all_targets, all_predictions


# In[ ]:


dataset_name = "datasets/DAIC-WOZ-Cleaned"

train_df = pd.read_csv(os.path.join(dataset_name, 'train_split_Depression_AVEC2017.csv'))
dev_df = pd.read_csv(os.path.join(dataset_name, 'dev_split_Depression_AVEC2017.csv'))
test_df = pd.read_csv(os.path.join(dataset_name, 'full_test_split.csv'))

y_train = load_labels_from_dataset(train_df)
y_dev = load_labels_from_dataset(dev_df) 
y_test = load_labels_from_dataset(test_df)

train_paths = get_audio_paths(train_df, dataset_name)
dev_paths = get_audio_paths(dev_df, dataset_name)
test_paths = get_audio_paths(test_df, dataset_name)

print(f"File di training: {len(train_paths)}, di validazione: {len(dev_paths)}, di test: {len(test_paths)}")

# Creazione Dataset e DataLoader
train_dataset = AudioSegmentDataset(train_paths, y_train, sr=SR, segment_ms=SEGMENT_MS, hop_ms=HOP_MS)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

# Inizializzazione Modello, Ottimizzatore e Loss
model = CNNMLP(dropout_rate=DROPOUT_RATE, num_classes=2).to(DEVICE)
# L'ottimizzatore nel paper è SGD
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

print(f"\nModello inizializzato con {sum(p.numel() for p in model.parameters() if p.requires_grad)} parametri allenabili.")


# In[ ]:


# Ciclo di Training
print("\n=== Inizio Training ===")
best_val_accuracy = -1

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Training -> Loss: {train_loss:.4f}, Accuracy (sui segmenti): {train_acc:.4f}")

    # Validazione
    val_acc, val_sens, val_spec, _, _, _ = evaluate_model(model, dev_paths, y_dev, DEVICE, SR, SEGMENT_MS, HOP_MS)
    print(f"Validation -> Accuracy: {val_acc:.4f}, Sensitivity: {val_sens:.4f}, Specificity: {val_spec:.4f}")
    
    # Salvataggio del miglior modello basato sull'accuratezza di validazione
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Nuova migliore accuratezza: {best_val_accuracy:.4f}. Modello salvato in '{MODEL_SAVE_PATH}'.")

    # Early stopping
    early_stopper(val_acc)
    if early_stopper.early_stop:
        print("Early stopping attivato.")
        break

print("\n=== Training Completato ===")


# In[ ]:


print("\n=== Valutazione Finale sul Test Set ===")

# Carica il miglior modello salvato
best_model = CNNMLP(dropout_rate=DROPOUT_RATE, num_classes=2).to(DEVICE)
best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
print(f"Miglior modello caricato da '{MODEL_SAVE_PATH}'.")

# Valutazione
test_acc, test_sens, test_spec, test_cm, test_targets, test_preds = evaluate_model(
    best_model, test_paths, y_test, DEVICE, SR, SEGMENT_MS, HOP_MS
)

print("\n--- Risultati sul Test Set ---")
print(f"Accuracy: {test_acc:.4f}")
print(f"Sensitivity: {test_sens:.4f}")
print(f"Specificity: {test_spec:.4f}")

print("\nMatrice di Confusione:")
print(test_cm)

print("\nReport di Classificazione Dettagliato:")
print(classification_report(test_targets, test_preds, target_names=['Non-Depressed (0)', 'Depressed (1)']))

