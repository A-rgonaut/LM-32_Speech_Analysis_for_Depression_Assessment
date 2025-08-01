import torch

from src.ssl_module.config import SSLConfig
from src.ssl_module.data_loader import DataLoader
from src.ssl_module.model import SSLModel
from src.ssl_module.evaluator import Evaluator
from src.src_utils import set_seed

def main():
    set_seed(42) 
    config = SSLConfig()
    data_loader = DataLoader(config)
    _, test_loader, _ = data_loader.load_data()

    model = SSLModel(config)
    model.load_state_dict(torch.load(config.model_save_path))
    evaluator = Evaluator(model, test_loader)
    evaluator.evaluate()

if __name__ == "__main__":
    main()