import numpy as np

from src.src_utils import clear_cache
from src.svm_module.config import SVMConfig
from src.svm_module.data_loader import DataLoader
from src.svm_module.model import SVMModel

def main():
    config = SVMConfig()
    data_loader = DataLoader(config)
    all_train_X, all_dev_X = [], []
    final_train_y, final_dev_y = None, None

    # Train models for individual features
    feature_types = ['articulation', 'phonation', 'prosody']
    for feature_type in feature_types:
        print(f"Training model for feature: {feature_type}")
        train_X, train_y, _, _, dev_X, dev_y = data_loader.load_data(feature_type)
        
        all_train_X.append(train_X)
        all_dev_X.append(dev_X)
        
        if final_train_y is None:
            final_train_y, final_dev_y = train_y, dev_y

        model = SVMModel(config)
        model.train(train_X, train_y, dev_X, dev_y)
        
        # Save the model with a name based on the feature
        model_path = config.model_save_path.replace('.pkl', f'_{feature_type}.pkl')
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        clear_cache()

    # Train model for combined features
    print("Training model for combined features (articulation + phonation + prosody)")

    # Concatenate features horizontally
    combined_train_X = np.hstack(all_train_X)
    combined_dev_X = np.hstack(all_dev_X)

    model = SVMModel(config)
    model.train(combined_train_X, final_train_y, combined_dev_X, final_dev_y)

    # Save the combined model
    combined_model_path = config.model_save_path.replace('.pkl', '_combined.pkl')
    model.save_model(combined_model_path)
    print(f"Combined model saved to {combined_model_path}")

if __name__ == '__main__':
    main()