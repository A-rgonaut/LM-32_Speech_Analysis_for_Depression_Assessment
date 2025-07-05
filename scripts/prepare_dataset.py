import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    SEED = 42
    MASTER_CSV_PATH = "datasets/dataset.csv"
    OUTPUT_DIR = "datasets/splits"
    USE_TONE_CHANGE = False
    
    # Proporzioni per la suddivisione
    TEST_SIZE = 0.20  # 20% per il test set
    VALIDATION_SIZE = 0.15 # 15% del totale per la validazione (circa 18.75% del rimanente)

    print("--- Inizio preparazione dataset ---")

    # Crea la directory di output se non esiste
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Carica il dataset master
    master_df = pd.read_csv(MASTER_CSV_PATH)
    master_df['Participant_ID'] = master_df['Participant_ID'].astype(str)

    # --- Filtra solo i dati originali per fare lo split ---
    original_df = master_df[master_df['augmentation'] == 'none'].copy()
    print(f"Dataset originale (senza augmentation): {len(original_df)} campioni")

    # --- Prima suddivisione: Train+Validation vs Test (solo dati originali) ---
    print(f"Suddivisione in Train/Validation ({1-TEST_SIZE:.0%}) e Test ({TEST_SIZE:.0%})")
    
    train_val_df, test_df = train_test_split(
        original_df, 
        test_size=TEST_SIZE, 
        random_state=SEED,
        stratify=original_df['PHQ8_Binary']  
    )

    # --- Seconda suddivisione: Train vs Validation ---
    val_split_size = VALIDATION_SIZE / (1 - TEST_SIZE)
    print(f"Suddivisione in Train e Validation (validation size: {val_split_size:.2%})")
    
    train_df, validation_df = train_test_split(
        train_val_df,
        test_size=val_split_size,
        random_state=SEED,
        stratify=train_val_df['PHQ8_Binary']
    )

    # --- Opzionale: Aggiungi dati aumentati al training set ---    
    if USE_TONE_CHANGE:
        print("\nAggiunta dati aumentati al training set...")
        train_participant_ids = set(train_df['Participant_ID'])
        augmented_data = master_df[
            (master_df['Participant_ID'].isin(train_participant_ids)) & 
            (master_df['augmentation'] == 'tone_change')
        ].copy()
        
        print(f"Dati aumentati trovati: {len(augmented_data)} campioni")
        train_df = pd.concat([train_df, augmented_data], ignore_index=True)
        print(f"Training set finale: {len(train_df)} campioni (originali + aumentati)")

    # Salvataggio dei file CSV
    train_csv_path = os.path.join(OUTPUT_DIR, "train_split.csv")
    val_csv_path = os.path.join(OUTPUT_DIR, "validation_split.csv")
    test_csv_path = os.path.join(OUTPUT_DIR, "test_split.csv")

    train_df.to_csv(train_csv_path, index=False)
    validation_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"\nFile CSV salvati in '{OUTPUT_DIR}':")
    print(f"  - {os.path.basename(train_csv_path)}")
    print(f"  - {os.path.basename(val_csv_path)}")
    print(f"  - {os.path.basename(test_csv_path)}")
    print("\n--- Preparazione completata ---")

if __name__ == "__main__":
    main()