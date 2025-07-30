import os
from dotenv import load_dotenv
from comet_ml import Experiment

from src.src_utils import clear_cache
from src.cnn_module.config import CNNConfig
from src.cnn_module.data_loader import DataLoader
from src.cnn_module.model import CNNModel
from src.cnn_module.trainer import Trainer

clear_cache()
load_dotenv()

experiment = Experiment(
    api_key = os.getenv("COMET_API_KEY"),
    project_name = os.getenv("COMET_PROJECT_NAME"),
    workspace = os.getenv("COMET_WORKSPACE")
)

config = CNNConfig()
data_loader = DataLoader(config)
train_loader, test_loader, dev_loader = data_loader.load_data()

model = CNNModel(config)

trainer = Trainer(model, train_loader, dev_loader, config)
trainer.train(experiment)