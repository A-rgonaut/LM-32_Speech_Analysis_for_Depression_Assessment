import sys
import os

from src.ssl_module.config import SSLConfig
from src.preprocessor import E1_DAIC
from src.src_utils import get_splits, filter_edaic_samples
from src.ssl_module.feature_extractor import FeatureExtractor

def main():
    print("Starting feature extraction process...")
    
    # 1. Carica la configurazione e prepara i dati
    config = SSLConfig()
    preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
    splits = preprocessor.get_dataset_splits()
    
    if not config.edaic_aug:
        print("Filtering out E-DAIC augmentation samples.")
        splits = filter_edaic_samples(splits)

    train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(splits)
    
    # 2. Inizializza l'estrattore di feature
    extractor = FeatureExtractor(config)
    
    # 3. Estrai e salva le feature per ogni split
    extractor.extract_and_save(train_paths, "train")
    extractor.extract_and_save(dev_paths, "dev")
    extractor.extract_and_save(test_paths, "test")
    
    print("Feature extraction process completed successfully.")

if __name__ == "__main__":
    main()