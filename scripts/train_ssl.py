import os
from dotenv import load_dotenv
from comet_ml import Experiment
import numpy as np
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.ssl_module.config import SSLConfig
from src.ssl_module.data_loader import DataLoader
from src.ssl_module.model import SSLModel
from src.ssl_module.trainer import Trainer
from src.utils import set_seed, clear_cache

def main():
    load_dotenv()
    experiment = Experiment(
        api_key = os.getenv("COMET_API_KEY"),
        project_name = os.getenv("COMET_PROJECT_NAME"),
        workspace = os.getenv("COMET_WORKSPACE")
    )

    config = SSLConfig()
    set_seed(config.seed)

    data_loader = DataLoader(config)
    train_dataset = data_loader.get_dataset('train')

    indices = np.arange(len(train_dataset))
    labels = train_dataset.labels
    groups = [train_dataset.path_to_id[p] for p in train_dataset.audio_paths]

    if config.k_folds == 1:
        kfold = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=config.seed)
    else:
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices, labels, groups)):
        print(f"\nTraining Fold {fold + 1}/{config.k_folds}")

        val_subset = Subset(train_dataset, val_idx)
        val_loader = data_loader.get_data_loader('dev', val_subset)

        train_subset = Subset(train_dataset, train_idx)
        train_loader = data_loader.get_data_loader('train', train_subset)

        model = SSLModel(config)
        if fold == 0: 
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"SSLModel created.")
            print(f"  Total parameters: {total_params/1e6:.2f}M")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")

        trainer = Trainer(model, train_loader, val_loader, config)
        with experiment.context_manager(f"fold_{fold+1}"):
            trainer.train(experiment, f'ssl_model_fold_{fold + 1}.pth')
        clear_cache()

    print("\nK-Fold training complete. All models have been saved.")

if __name__ == "__main__":
    main()