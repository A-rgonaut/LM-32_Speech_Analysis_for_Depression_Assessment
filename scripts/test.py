import numpy as np

from src.config_loader import load_config
from src.evaluator import Evaluator
from src.utils import set_seed

def main():
    config = load_config()
    set_seed(config.seed)
    if config.active_model == 'svm':
        from src.svm_module.data_loader import DataLoader
    elif config.active_model == 'cnn':
        from src.cnn_module.data_loader import DataLoader
    elif config.active_model == 'ssl':
        from src.ssl_module.data_loader import DataLoader
    elif config.active_model == 'ssl2':
            from src.ssl_module_2.data_loader import DataLoader

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
        test_loader = data_loader.get_data_loader('test')
        dev_loader = data_loader.get_data_loader('dev')
        
        evaluator = Evaluator(config, dev_loader)
        evaluator.evaluate('dev')

        evaluator = Evaluator(config, test_loader)
        evaluator.evaluate('test')

if __name__ == "__main__":
    main()