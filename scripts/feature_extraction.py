from src.ssl_module.config import SSLConfig
from src.preprocessor import E1_DAIC
from src.utils import get_splits, filter_edaic_samples, set_seed
from src.ssl_module.feature_extractor import FeatureExtractor

def main():
    print("Starting feature extraction process...")
    
    config = SSLConfig()
    set_seed(config.seed)  
    preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
    splits = preprocessor.get_dataset_splits()
    
    if not config.edaic_aug:
        print("Filtering out E-DAIC augmentation samples.")
        splits = filter_edaic_samples(splits)

    train_paths, _, test_paths, _, dev_paths, _ = get_splits(splits)
    
    extractor = FeatureExtractor(config)
    
    extractor.extract_and_save(train_paths, "train")
    extractor.extract_and_save(dev_paths, "dev")
    extractor.extract_and_save(test_paths, "test")
    
    print("Feature extraction process completed successfully.")

if __name__ == "__main__":
    main()