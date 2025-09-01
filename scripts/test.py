import numpy as np
import copy
import os
import json

from src.config_loader import load_config
from src.evaluator import Evaluator
from src.utils import set_seed

def main():
    config = load_config()
    set_seed(config.seed)
    if config.active_model in ['cnn', 'ssl']:
        params_filename = f'best_params_{config.active_model}.json'
        path = config.model_save_dir
        if config.active_model == 'ssl':
            path = path.split("ssl")[0] + "ssl"
        params_path = os.path.join(path, params_filename)

        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                best_params = json.load(f)
            for key, value in best_params.items():
                setattr(config, key, value)
                print(key,value)
                
    if config.active_model == 'svm':
        from src.svm_module.data_loader import DataLoader
    elif config.active_model == 'cnn':
        from src.cnn_module.data_loader import DataLoader
    elif config.active_model == 'ssl':
        from src.ssl_module.data_loader import DataLoader

    data_loader = DataLoader(config)

    if config.active_model == 'svm':
        for feature_type in config.feature_types:
            print(f"Testing model for feature: {feature_type}")

            if feature_type == 'combined':
                feature_list = ['articulation', 'phonation', 'prosody']
                all_train_X, all_dev_X, all_test_X = [], [], []
                train_y, dev_y, test_y = None, None, None
                for f_type in feature_list:
                    tr_X, tr_y, _, te_X, te_y, _, de_X, de_y, _ = data_loader.load_data(f_type)
                    all_train_X.append(np.array(tr_X))
                    all_dev_X.append(np.array(de_X))
                    all_test_X.append(np.array(te_X))
                    if train_y is None:
                        train_y, dev_y, test_y = np.array(tr_y), np.array(de_y), np.array(te_y)
                train_X, dev_X, test_X = np.hstack(all_train_X), np.hstack(all_dev_X), np.hstack(all_test_X)
            else:
                train_X, train_y, _, test_X, test_y, _, dev_X, dev_y, _ = data_loader.load_data(feature_type)
                train_X, train_y = np.array(train_X), np.array(train_y)
                dev_X, dev_y = np.array(dev_X), np.array(dev_y)
                test_X, test_y = np.array(test_X), np.array(test_y)
                
            evaluator = Evaluator(config, (dev_X, dev_y))
            evaluator.evaluate('dev', feature_type)

            evaluator = Evaluator(config, (test_X, test_y))
            evaluator.evaluate('test', feature_type)
    else:
        if config.active_model == 'ssl' and hasattr(config, 'run_layer_sweep') and config.run_layer_sweep:
            model_list = config.ssl_model_names if isinstance(config.ssl_model_names, list) else [config.ssl_model_names]
            layer_list = config.layers_to_use if isinstance(config.layers_to_use, list) else [config.layers_to_use]

            base_model_save_dir = config.model_save_dir.split("ssl/")[0] + "ssl"
            base_result_dir = config.result_dir.split("ssl/")[0] + "ssl"

            for model_name in model_list:
                for layer in layer_list:
                    print(f"\nEvaluating Model: {model_name}, Layer: {layer}")

                    run_config = copy.deepcopy(config)
                    run_config.ssl_model_name = model_name
                    run_config.layer_to_use = layer
                    run_config.use_all_layers = False
                    for key, value in best_params.items():
                        setattr(run_config, key, value)

                    ssl_model_name_path = model_name.replace('/', '-')
                    run_config.feature_path = f"features/{ssl_model_name_path}"
                    run_config.model_save_dir = os.path.join(base_model_save_dir, ssl_model_name_path, f"layer{layer}")
                    run_config.result_dir = os.path.join(base_result_dir, ssl_model_name_path, f"layer{layer}")

                    ssl_model_name_path = model_name.replace('/', '-')
                    run_config.feature_path = f"features/{ssl_model_name_path}"

                    data_loader = DataLoader(run_config)
                    test_loader = data_loader.get_data_loader('test')
                    dev_loader = data_loader.get_data_loader('dev')

                    evaluator = Evaluator(run_config, dev_loader)
                    evaluator.evaluate('dev')

                    evaluator = Evaluator(run_config, test_loader)
                    evaluator.evaluate('test')
        else:
            test_loader = data_loader.get_data_loader('test')
            dev_loader = data_loader.get_data_loader('dev')
            
            evaluator = Evaluator(config, dev_loader)
            evaluator.evaluate('dev')

            evaluator = Evaluator(config, test_loader)
            evaluator.evaluate('test')

if __name__ == "__main__":
    main()