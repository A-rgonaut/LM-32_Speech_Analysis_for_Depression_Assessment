import os
from dotenv import load_dotenv
from comet_ml import Experiment

from src.ssl_module.config import SSLConfig
from src.ssl_module.data_loader import DataLoader
from src.ssl_module.model import SSLModel
from src.ssl_module.trainer import Trainer
from src.src_utils import set_seed

def main():
    set_seed(42)
    load_dotenv()

    experiment = Experiment(
        api_key = os.getenv("COMET_API_KEY"),
        project_name = os.getenv("COMET_PROJECT_NAME"),
        workspace = os.getenv("COMET_WORKSPACE")
    )

    config = SSLConfig()
    data_loader = DataLoader(config)
    train_loader, _, dev_loader = data_loader.load_data()

    model = SSLModel(config)

    trainer = Trainer(model, train_loader, dev_loader, config)
    trainer.train(experiment)

if __name__ == "__main__":
    main()