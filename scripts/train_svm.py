import numpy as np
import os
import json

from src.utils import clear_cache, set_seed
from src.svm_module.config import SVMConfig
from src.svm_module.data_loader import DataLoader
from src.svm_module.model import SVMModel

def train_svm_for_feature(feature_type, config: SVMConfig, data_loader: DataLoader):
    print(f"Training SVM for feature: {feature_type}")
    
    if feature_type == 'combined':
        all_train_X, train_y = [], None
        for f_type in ['articulation', 'phonation', 'prosody']:
            tr_X, tr_y, *_ = data_loader.load_data(f_type)
            all_train_X.append(np.array(tr_X))
            if train_y is None:
                train_y = np.array(tr_y)
        train_X = np.hstack(all_train_X)
    else:
        train_X, train_y, *_ = data_loader.load_data(feature_type)
        train_X, train_y = np.array(train_X), np.array(train_y)
    
    model = SVMModel(config)
    best_params = model.tune_and_train(train_X, train_y)

    params_path = os.path.join(config.model_save_dir, f'svm_params_{feature_type}.json')
    os.makedirs(config.model_save_dir, exist_ok=True)
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Params saved to {params_path}")

    model.save_model(feature_type)
    print(f"Final model saved to {config.model_save_dir}/svm_model_{feature_type}.pkl")

    clear_cache()

def main():
    set_seed(42)
    config = SVMConfig()
    data_loader = DataLoader(config)

    for feature_type in ['articulation', 'phonation', 'prosody', 'combined']:
        train_svm_for_feature(feature_type, config, data_loader)
    
if __name__ == '__main__':
    main()