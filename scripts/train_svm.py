import numpy as np
import os
import json

from src.utils import clear_cache, set_seed
from src.svm_module.config import SVMConfig
from src.svm_module.data_loader import DataLoader
from src.svm_module.model import SVMModel

def main():
    set_seed(42)
    config = SVMConfig()
    data_loader = DataLoader(config)
    all_train_X = []
    final_train_y = None

    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    # Train models for individual features
    feature_types = ['articulation', 'phonation', 'prosody']
    for feature_type in feature_types:
        print(f"Finding best params for feature: {feature_type}")
        train_X, train_y, _, _, _, _ = data_loader.load_data(feature_type)

        train_X, train_y = np.array(train_X), np.array(train_y)
        
        all_train_X.append(train_X)
        
        if final_train_y is None:
            final_train_y = train_y

        model = SVMModel(config)
        best_params = model.find_best_params(train_X, train_y)
        
        # Save the best parameters
        params_filename = f'svm_params_{feature_type}.json'
        params_path = os.path.join(config.model_save_dir, params_filename)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Best parameters for '{feature_type}' saved to {params_path}")
        clear_cache()

    # Train model for combined features
    print("Finding best params for combined features (articulation + phonation + prosody)")
    combined_train_X = np.hstack(all_train_X)

    model = SVMModel(config)
    best_params_combined = model.find_best_params(combined_train_X, final_train_y)
    # Save the combined model
    params_filename_combined = 'svm_params_combined.json'
    params_path_combined = os.path.join(config.model_save_dir, params_filename_combined)
    with open(params_path_combined, 'w') as f:
        json.dump(best_params_combined, f, indent=4)
    print(f"Best parameters for 'combined' saved to {params_path_combined}")
    
if __name__ == '__main__':
    main()