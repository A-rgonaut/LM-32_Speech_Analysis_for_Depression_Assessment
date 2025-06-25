import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
from src.data import AudioDepressionDataset
from src.models import CNNMLP

from src.training import (
    eval_model_by_file_aggregation,
    load_labels_from_dataset, 
    get_audio_paths
)

def main():
    DATASET_NAME = "datasets/DAIC-WOZ-Cleaned"
    MODEL_SAVE_PATH = "cnn_best.pth"
    BATCH_SIZE = 512
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 1 else 0

    # Selezione del device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("\n--- Inizio Test Finale ---")
    test_df = pd.read_csv(os.path.join(DATASET_NAME, 'full_test_split.csv'))
    y_test = load_labels_from_dataset(test_df)
    test_paths = get_audio_paths(test_df, DATASET_NAME)
    
    # Per il test, dobbiamo aggregare per file, quindi `return_filename=True`
    test_dataset = AudioDepressionDataset(audio_paths=test_paths, labels=y_test, return_filename=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Carica il miglior modello e valuta
    model = CNNMLP().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    accuracy, f1, sensitivity, specificity = eval_model_by_file_aggregation(model, test_dataloader, device)

    print("\n=== Risultati del Test (aggregati per file) ===")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print("---------------------------------------------")

if __name__ == "__main__":
    main()