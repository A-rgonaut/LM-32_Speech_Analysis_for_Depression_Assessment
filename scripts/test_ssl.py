from src.ssl_module.config import SSLConfig
from src.ssl_module.data_loader import DataLoader
from src.evaluator import Evaluator
from src.utils import set_seed

def main():
    config = SSLConfig()
    set_seed(config.seed) 
    data_loader = DataLoader(config)
    test_loader = data_loader.get_data_loader('test')
    dev_loader = data_loader.get_data_loader('dev')
    
    evaluator = Evaluator(dev_loader, config, 'ssl')
    evaluator.evaluate('dev')
    
    evaluator = Evaluator(test_loader, config, 'ssl')
    evaluator.evaluate('test')

if __name__ == "__main__":
    main()