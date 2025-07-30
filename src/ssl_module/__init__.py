
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src_utils
from dotenv import load_dotenv
from comet_ml import Experiment
from config import SSLConfig
from data_loader import DataLoader
from model import SSLModel
from trainer import Trainer
from evaluator import Evaluator

src_utils.clear_cache()

load_dotenv()

experiment = Experiment(
    api_key = os.getenv("COMET_API_KEY"),
    project_name = os.getenv("COMET_PROJECT_NAME"),
    workspace = os.getenv("COMET_WORKSPACE")
    )

config = SSLConfig()
data_loader = DataLoader(config)
train_loader, test_loader, dev_loader = data_loader.load_data()

model = SSLModel(config)

trainer = Trainer(model, train_loader, dev_loader, config)
trainer.train(experiment)

#evaluator = Evaluator(model, data['test'])
#evaluator.evaluate()
