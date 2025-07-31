import torch

from src.ssl_module.config import SSLConfig
from src.ssl_module.data_loader import DataLoader
from src.ssl_module.model import SSLModel
from src.ssl_module.evaluator import Evaluator

def main():
    config = SSLConfig()
    data_loader = DataLoader(config)
    _, test_loader, _ = data_loader.load_data()

    model = SSLModel(config)
    model.load_state_dict(torch.load(config.model_save_path))
    evaluator = Evaluator(model, test_loader, config.eval_strategy)
    evaluator.evaluate()

if __name__ == "__main__":
    main()