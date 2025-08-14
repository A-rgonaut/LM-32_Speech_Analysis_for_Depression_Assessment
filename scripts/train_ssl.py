import os
from dotenv import load_dotenv
from comet_ml import Experiment
from sklearn.model_selection import StratifiedGroupKFold

from src.ssl_module.config import SSLConfig
from src.ssl_module.data_loader import DataLoader, Dataset
from src.ssl_module.model import SSLModel
from src.trainer import Trainer
from src.utils import set_seed, clear_cache

def main():
    config = SSLConfig()
    set_seed(config.seed)
    experiment = None
    if config.use_comet:
        load_dotenv()
        experiment = Experiment(
            api_key = os.getenv("COMET_API_KEY"),
            project_name = os.getenv("COMET_PROJECT_NAME"),
            workspace = os.getenv("COMET_WORKSPACE")
        )

    data_loader = DataLoader(config)

    train_val_df, _, _ = data_loader.splits_df
    
    participant_ids = train_val_df["Participant_ID"].tolist()
    labels = train_val_df['PHQ_Binary'].tolist()
    groups = participant_ids

    if config.k_folds == 1:
        validation_split_fraction = 0.1
        n_splits_for_single_run = int(1 / validation_split_fraction)
        kfold = StratifiedGroupKFold(n_splits=n_splits_for_single_run, shuffle=True, random_state=config.seed)
    else:
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    kfold_splitter = kfold.split(participant_ids, labels, groups)

    for fold in range(config.k_folds):
        print(f"\nTraining Fold {fold + 1}/{config.k_folds}")

        train_idx, val_idx = next(kfold_splitter)

        train_pids_fold = [participant_ids[i] for i in train_idx]
        val_pids_fold = [participant_ids[i] for i in val_idx]

        train_info_fold = {pid: data_loader.participant_info[pid] for pid in train_pids_fold}
        val_info_fold = {pid: data_loader.participant_info[pid] for pid in val_pids_fold}

        train_dataset_fold = Dataset(config, train_info_fold, is_train=True)
        val_dataset_fold = Dataset(config, val_info_fold)

        train_loader = data_loader.get_data_loader('train', train_dataset_fold)
        val_loader = data_loader.get_data_loader('dev', val_dataset_fold)

        model = SSLModel(config)
        if fold == 0: 
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"SSLModel created.")
            print(f"  Total parameters: {total_params/1e6:.2f}M")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")

        trainer = Trainer(model, train_loader, val_loader, config)
        if experiment:
            with experiment.context_manager(f"fold_{fold+1}"):
                trainer.train(experiment, f'ssl_model_fold_{fold + 1}.pth')
        else:
            trainer.train(None, f'ssl_model_fold_{fold + 1}.pth')
        clear_cache()
        
    print("\nK-Fold training complete. All models have been saved.")

if __name__ == "__main__":
    main()