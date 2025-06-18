import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import AudioDepressionDatasetSSL, collate_fn
from src.models import DepressionClassifier
from src.training import (
    train_epoch, 
    eval_model,
    EarlyStopping, 
    load_labels_from_dataset, 
    get_audio_paths,
    print_model_summary
)

def main():
    # --- 1. CONFIGURAZIONE DELL'ESPERIMENTO ---
    print("--- Configurazione dell'esperimento Transformer ---")
    SEED = 42
    DATASET_NAME = "datasets/DAIC-WOZ-Cleaned"
    MODEL_NAME_HF = "facebook/wav2vec2-base" # Nome del modello da Hugging Face
    MODEL_SAVE_PATH = "transformer_best.pth"
    
    # Iperparametri 
    BATCH_SIZE = 2  
    LEARNING_RATE = 3e-5 
    NUM_EPOCHS = 20
    NUM_WORKERS = 4
    
    # Parametri specifici del dataset/modello
    SEGMENT_LENGTH_SECONDS = 4
    MAX_SEGMENTS = 20
    NUM_CLASSES = 2
    
    # Impostazione del seed per la riproducibilità
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Selezione del device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. CARICAMENTO E PREPARAZIONE DEI DATI ---
    print("\n--- Caricamento dati ---")
    train_df = pd.read_csv(os.path.join(DATASET_NAME, 'train_split_Depression_AVEC2017.csv'))
    dev_df = pd.read_csv(os.path.join(DATASET_NAME, 'dev_split_Depression_AVEC2017.csv'))

    y_train = load_labels_from_dataset(train_df)
    y_dev = load_labels_from_dataset(dev_df) 

    train_paths = get_audio_paths(train_df, DATASET_NAME)
    dev_paths = get_audio_paths(dev_df, DATASET_NAME)

    # Creazione dei Dataset
    train_dataset = AudioDepressionDatasetSSL(
        audio_paths=train_paths, 
        labels=y_train, 
        model_name=MODEL_NAME_HF, 
        segment_length_seconds=SEGMENT_LENGTH_SECONDS, 
        max_segments=MAX_SEGMENTS
    )
    dev_dataset = AudioDepressionDatasetSSL(
        audio_paths=dev_paths, 
        labels=y_dev, 
        model_name=MODEL_NAME_HF, 
        segment_length_seconds=SEGMENT_LENGTH_SECONDS, 
        max_segments=MAX_SEGMENTS
    )

    # Creazione dei DataLoader (con il collate_fn personalizzato!)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    print(f"Train files: {len(train_dataset)}, Dev files: {len(dev_dataset)}")

    # --- 3. INIZIALIZZAZIONE MODELLO E COMPONENTI DI TRAINING ---
    print("\n--- Inizializzazione modello ---")
    model = DepressionClassifier(model_name=MODEL_NAME_HF, num_classes=NUM_CLASSES).to(device)
    print_model_summary(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss() 
    early_stopping = EarlyStopping(patience=5, min_delta=0.01, mode='max')
    
    # Scheduler del learning rate
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # --- 4. TRAINING LOOP ---
    print("\n--- Inizio Training ---")
    best_val_f1 = -1.0

    for epoch in range(NUM_EPOCHS):
        # Training
        train_loss, train_acc = train_epoch(model, train_dataloader, criterion, optimizer, scheduler, device, epoch, NUM_EPOCHS)
        
        # Validation
        val_loss, val_acc, val_f1 = eval_model(model, dev_dataloader, criterion, device)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> Nuovo miglior F1-score: {best_val_f1:.4f}. Modello salvato in '{MODEL_SAVE_PATH}'")

        if early_stopping(val_f1):
            print(f"Early stopping attivato dopo {epoch+1} epoche.")
            break
            
    print("\n--- Training Completato ---")

    # --- 5. TEST FINALE SUL MIGLIOR MODELLO ---
    print("\n--- Inizio Test Finale ---")
    test_df = pd.read_csv(os.path.join(DATASET_NAME, 'full_test_split.csv'))
    y_test = load_labels_from_dataset(test_df)
    test_paths = get_audio_paths(test_df, DATASET_NAME)
    
    test_dataset = AudioDepressionDatasetSSL(
        audio_paths=test_paths, 
        labels=y_test, 
        model_name=MODEL_NAME_HF, 
        segment_length_seconds=SEGMENT_LENGTH_SECONDS, 
        max_segments=MAX_SEGMENTS
    )
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # Carica il miglior modello e valuta
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc, test_f1 = eval_model(model, test_dataloader, criterion, device)

    print("\n=== Risultati del Test ===")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.4f}")
    print(f"  Test F1:   {test_f1:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    main()