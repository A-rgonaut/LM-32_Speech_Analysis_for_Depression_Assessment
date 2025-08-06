import os
import numpy as np
import json

from src.svm_module.config import SVMConfig
from src.svm_module.data_loader import DataLoader
from src.svm_module.evaluator import Evaluator
from src.svm_module.model import SVMModel
from src.utils import set_seed

def main():
    set_seed(42)  
    config = SVMConfig()
    data_loader = DataLoader(config)

    feature_types_to_test = ['articulation', 'phonation', 'prosody', 'combined']

    for feature_type in feature_types_to_test:
        print(f"Testing model for feature: {feature_type}")
        params_path = os.path.join(config.model_save_dir, f'svm_params_{feature_type}.json')
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print(f"Best parameters for '{feature_type}' loaded from '{params_path}'")

        if feature_type == 'combined':
            feature_list = ['articulation', 'phonation', 'prosody']
            all_train_X, all_dev_X, all_test_X = [], [], []
            train_y, dev_y, test_y = None, None, None
            for f_type in feature_list:
                tr_X, tr_y, te_X, te_y, de_X, de_y = data_loader.load_data(f_type)
                all_train_X.append(np.array(tr_X))
                all_dev_X.append(np.array(de_X))
                all_test_X.append(np.array(te_X))
                if train_y is None:
                    train_y, dev_y, test_y = np.array(tr_y), np.array(de_y), np.array(te_y)
            train_X, dev_X, test_X = np.hstack(all_train_X), np.hstack(all_dev_X), np.hstack(all_test_X)
        else:
            train_X, train_y, test_X, test_y, dev_X, dev_y = data_loader.load_data(feature_type)
            train_X, train_y = np.array(train_X), np.array(train_y)
            dev_X, dev_y = np.array(dev_X), np.array(dev_y)
            test_X, test_y = np.array(test_X), np.array(test_y)

        model = SVMModel(config)
        kfold_results_df = model.train_and_evaluate_kfold(train_X, train_y, dev_X, dev_y, best_params)
        dev_results_path = os.path.join(config.result_dir, f'kfold_dev_results_{feature_type}.csv')
        kfold_results_df.to_csv(dev_results_path)
        print(f"K-Fold dev set results saved to {dev_results_path}")

        print("Training final model on the entire training set...")
        model.train(train_X, train_y, best_params)
        print("Final model trained.")

        model_filename = f'svm_model_{feature_type}.pkl'
        model_path = os.path.join(config.model_save_dir, model_filename)
        model.save_model()
        print(f"Final model saved to {model_path}")

        print("Evaluating final model on Test set")
        evaluator = Evaluator(model.model, test_X, test_y, config)
        evaluator.evaluate(feature_type)

if __name__ == '__main__':
    main()