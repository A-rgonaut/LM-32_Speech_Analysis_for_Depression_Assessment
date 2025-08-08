from src.cnn_module.config import CNNConfig
from src.cnn_module.data_loader import DataLoader
from src.cnn_module.evaluator import Evaluator
from src.utils import set_seed

def main():
    config = CNNConfig()
    set_seed(config.seed)
    data_loader = DataLoader(config)
    test_loader = data_loader.get_data_loader('test')
    dev_loader = data_loader.get_data_loader('dev')
    
    evaluator = Evaluator(dev_loader, config)
    evaluator.evaluate('dev')
    
    evaluator = Evaluator(test_loader, config)
    evaluator.evaluate('test')

if __name__ == "__main__":
    main()