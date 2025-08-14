import numpy as np

from src.svm_module.config import SVMConfig
from src.svm_module.data_loader import DataLoader
from src.evaluator import Evaluator
from src.svm_module.model import SVMModel
from src.utils import set_seed

def main():
    config = SVMConfig()
    set_seed(config.seed)  
    data_loader = DataLoader(config)

    feature_types_to_test = ['articulation', 'phonation', 'prosody', 'combined']

    for feature_type in feature_types_to_test:
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

        model = SVMModel(config)
        model.load_model(feature_type)

        evaluator = Evaluator(config, 'svm', (dev_X, dev_y), model.model)
        evaluator.evaluate('dev', feature_type)

        evaluator = Evaluator(config, 'svm', (test_X, test_y), model.model)
        evaluator.evaluate('test', feature_type)

if __name__ == '__main__':
    main()