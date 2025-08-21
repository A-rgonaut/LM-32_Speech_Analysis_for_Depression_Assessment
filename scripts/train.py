import os
import random
import torch
import numpy as np
import json
from dotenv import load_dotenv
from comet_ml import Experiment
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Subset

from src.config_loader import load_config
from src.trainer import Trainer
from src.utils import set_seed, clear_cache

def _train_svm(config):
    from src.svm_module.data_loader import DataLoader
    from src.svm_module.model import SVMModel

    data_loader = DataLoader(config)

    for feature_type in config.feature_types:
        print(f"\nTraining SVM for feature type: {feature_type}")
        
        if feature_type == 'combined':
            all_train_X, train_y, train_groups = [], None, []
            for f_type in ['articulation', 'phonation', 'prosody']:
                tr_X, tr_y, tr_groups, *_ = data_loader.load_data(f_type)
                all_train_X.append(np.array(tr_X))
                if train_y is None:
                    train_y = np.array(tr_y)
                    train_groups = np.array(tr_groups)
            train_X = np.hstack(all_train_X)
        else:
            train_X, train_y, train_groups, *_ = data_loader.load_data(feature_type)
            train_X, train_y, train_groups = np.array(train_X), np.array(train_y), np.array(train_groups)

        model = SVMModel(config)
        best_params = model.tune_and_train(train_X, train_y, train_groups)

        params_path = os.path.join(config.model_save_dir, f'svm_params_{feature_type}.json')
        os.makedirs(config.model_save_dir, exist_ok=True)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Params saved to {params_path}")

        model.save_model(feature_type)
        print(f"Final model saved to {config.model_save_dir}/svm_model_{feature_type}.pkl")
        clear_cache()

def _train_pytorch_segment_cv(config, experiment):
    model_name = config.active_model

    if model_name == 'cnn':
        from src.cnn_module.data_loader import DataLoader
        from src.cnn_module.model import CNNModel as Model
    elif model_name == 'ssl2':
        from src.ssl_module_2.data_loader import DataLoader
        from src.ssl_module_2.model import SSLModel2 as Model

    data_loader = DataLoader(config)
    train_dataset = data_loader.get_dataset('train')

    segment_indices_for_split = np.arange(len(train_dataset.segments))
    segment_labels = [train_dataset.id_to_label[seg[0]] for seg in train_dataset.segments] # seg[0] is the audio_id
    segment_groups = [seg[0] for seg in train_dataset.segments]
    
    if config.k_folds == 1:
        validation_split_fraction = 0.2
        n_splits_for_single_run = int(1 / validation_split_fraction)
        kfold = StratifiedGroupKFold(n_splits=n_splits_for_single_run, shuffle=True, random_state=config.seed)
    else:
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
        
    kfold_splitter = kfold.split(segment_indices_for_split, segment_labels, segment_groups)

    for fold in range(config.k_folds):
        print(f"\nTraining Fold {fold + 1}/{config.k_folds}")
        train_segment_idx, val_segment_idx = next(kfold_splitter)

        val_subset = Subset(train_dataset, val_segment_idx)
        val_loader = data_loader.get_data_loader('dev', val_subset)
        
        final_train_idx = train_segment_idx
        if config.balance_train_set:
            train_segments_in_fold = [train_dataset.segments[i] for i in train_segment_idx]
            depressed_segments = [train_segment_idx[i] for i, seg in enumerate(train_segments_in_fold) if train_dataset.id_to_label[seg[0]] == 1]
            non_depressed_segments = [train_segment_idx[i] for i, seg in enumerate(train_segments_in_fold) if train_dataset.id_to_label[seg[0]] == 0]
            
            random.seed(config.seed + fold)
            if len(depressed_segments) < len(non_depressed_segments):
                balanced_dep_idx = depressed_segments
                balanced_non_dep_idx = random.sample(non_depressed_segments, len(depressed_segments))
            else:
                balanced_dep_idx = random.sample(depressed_segments, len(non_depressed_segments))
                balanced_non_dep_idx = non_depressed_segments
            
            balanced_train_idx = balanced_dep_idx + balanced_non_dep_idx
            random.shuffle(balanced_train_idx)
            final_train_idx = balanced_train_idx
            
            print(f"Fold {fold+1} training set balanced: {len(balanced_dep_idx)} depressed vs {len(balanced_non_dep_idx)} non-depressed segments.")

        train_subset = Subset(train_dataset, final_train_idx)
        train_loader = data_loader.get_data_loader('train', train_subset)
        
        model = Model(config)
        if fold == 0: 
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{model.__class__.__name__} created.")
            print(f"  Total parameters: {total_params/1e6:.2f}M")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
        
        trainer = Trainer(model, train_loader, val_loader, config)
        model_filename = f'{model_name}_model_fold_{fold + 1}.pth'
        
        if experiment:
            with experiment.context_manager(f"fold_{fold+1}"):
                trainer.train(experiment, model_filename)
        else:
            trainer.train(None, model_filename)
        clear_cache()

    print(f"\nK-Fold training complete for {model_name.upper()}. All models have been saved.")

def _train_pytorch_participant_cv(config, experiment):
    model_name = config.active_model
    
    from src.ssl_module.data_loader import DataLoader, Dataset
    from src.ssl_module.model import SSLModel as Model

    data_loader = DataLoader(config)
    train_val_df, _, _ = data_loader.splits_df
    
    participant_ids = train_val_df["Participant_ID"].tolist()
    labels = train_val_df['PHQ_Binary'].tolist()
    groups = participant_ids

    if config.k_folds == 1:
        validation_split_fraction = 0.2
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

        model = Model(config)
        if fold == 0: 
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{model.__class__.__name__} created.")
            print(f"  Total parameters: {total_params/1e6:.2f}M")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")

        trainer = Trainer(model, train_loader, val_loader, config)
        model_filename = f'{model_name}_model_fold_{fold + 1}.pth'
        
        if experiment:
            with experiment.context_manager(f"fold_{fold+1}"):
                trainer.train(experiment, model_filename)
        else:
            trainer.train(None, model_filename)
        clear_cache()
        
    print(f"\nK-Fold training complete for {model_name.upper()}. All models have been saved.")

def main():
    config = load_config()
    #torch.autograd.set_detect_anomaly(True)
    set_seed(config.seed)
    experiment = None
    if config.use_comet:
        load_dotenv()
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE")
        )

    if config.active_model == 'svm':
        _train_svm(config)
    elif config.active_model in ['cnn', 'ssl2']:
        _train_pytorch_segment_cv(config, experiment)
    elif config.active_model == 'ssl':
        _train_pytorch_participant_cv(config, experiment)

if __name__ == "__main__":
    main()