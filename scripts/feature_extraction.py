import copy
import os
from src.config_loader import load_config
from src.preprocessor import E1_DAIC
from src.utils import get_splits, set_seed
from src.ssl_module.feature_extractor import FeatureExtractor

def main():
    print("Starting feature extraction process...")
    
    config = load_config()
    set_seed(config.seed)  
    preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
    splits = preprocessor.get_dataset_splits()

    train_paths, _, test_paths, _, dev_paths, _ = get_splits(splits)
    
    model_list = config.ssl_model_names if isinstance(config.ssl_model_names, list) else [config.ssl_model_names]

    for model_name in model_list:
        print(f"Processing Model: {model_name}")

        run_config = copy.deepcopy(config)
        run_config.ssl_model_name = model_name
        
        ssl_model_name_path = model_name.replace('/', '-')
        run_config.feature_path = os.path.join("features/", ssl_model_name_path)
        
        extractor = FeatureExtractor(run_config)
        
        extractor.extract_and_save(train_paths, "train")
        extractor.extract_and_save(dev_paths, "dev")
        extractor.extract_and_save(test_paths, "test")
    
    print("\nFeature extraction process completed successfully for all specified models.")

if __name__ == "__main__":
    main()