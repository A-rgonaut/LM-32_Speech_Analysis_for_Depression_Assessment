import torch

from src.cnn_module.config import CNNConfig
from src.cnn_module.data_loader import DataLoader
from src.cnn_module.model import CNNModel
from src.cnn_module.evaluator import Evaluator

config = CNNConfig()
data_loader = DataLoader(config)
_, test_loader, _ = data_loader.load_data()

model = CNNModel(config)
model.load_state_dict(torch.load(config.model_save_path))
evaluator = Evaluator(model, test_loader, config.eval_strategy)
evaluator.evaluate()