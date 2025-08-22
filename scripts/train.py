import os
import random
import numpy as np
import itertools
import copy
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
    fold_best_f1s, temp_paths = [], []

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
        model_filename = f'temp_{model_name}_model_fold_{fold + 1}.pth'
        best_fold_f1 = 0
        
        if experiment:
            with experiment.context_manager(f"fold_{fold+1}"):
                best_fold_f1 = trainer.train(experiment, model_filename)
        else:
            best_fold_f1 = trainer.train(None, model_filename)
        
        full_temp_path = os.path.join(config.model_save_dir, model_filename)
        temp_paths.append(full_temp_path)
        fold_best_f1s.append(best_fold_f1)
        clear_cache()

    mean_f1 = np.mean(fold_best_f1s)
    print(f"\nK-Fold training complete for {model_name.upper()}. Average F1-macro: {mean_f1:.4f}")
    return mean_f1, temp_paths

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
    fold_best_f1s, temp_paths = [], []

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
        model_filename = f'temp_{model_name}_model_fold_{fold + 1}.pth'
        best_fold_f1 = 0

        if experiment:
            with experiment.context_manager(f"fold_{fold+1}"):
                best_fold_f1 = trainer.train(experiment, model_filename)
        else:
            best_fold_f1 = trainer.train(None, model_filename)
            
        fold_best_f1s.append(best_fold_f1)
        full_temp_path = os.path.join(config.model_save_dir, model_filename)
        temp_paths.append(full_temp_path)
        clear_cache()
        
    mean_f1 = np.mean(fold_best_f1s)
    print(f"\nK-Fold training complete for {model_name.upper()}. Average F1-macro: {mean_f1:.4f}")
    return mean_f1, temp_paths

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
    elif config.active_model == 'cnn':
        if config.hyperparameter_search_mode:
            print("Starting Hyperparameter Search for CNN model...")
            best_score = -1
            best_params = {}

            param_combinations = list(itertools.product(
                config.dropout_rate,
                config.learning_rate,
                config.batch_size,
                config.segmentation_strategy,
                config.max_utt_seconds,
                config.overlap_seconds
            ))

            for i, params in enumerate(param_combinations):
                run_config = copy.deepcopy(config)
                dropout, lr, batch_size, seg_strat, max_utt, overlap = params
                if overlap >= max_utt:
                    continue

                run_config.dropout_rate = dropout
                run_config.learning_rate = lr
                run_config.batch_size = batch_size
                run_config.segmentation_strategy = seg_strat
                run_config.max_utt_seconds = max_utt
                run_config.overlap_seconds = overlap

                current_params = {
                    'dropout_rate': dropout,
                    'learning_rate': lr,
                    'batch_size': batch_size,
                    'segmentation_strategy': seg_strat,
                    'max_utt_seconds': max_utt,
                    'overlap_seconds': overlap
                }

                print(f"\nRun {i+1}/{len(param_combinations)}")
                print(f"Testing params: {current_params}")

                avg_f1, temp_paths = _train_pytorch_segment_cv(run_config, experiment)
                print(f"Result for params {current_params}: Average F1 = {avg_f1:.4f}")

                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = current_params
                    print(f"New best score found: {best_score:.4f}")
                    if best_model_paths:
                        for p in best_model_paths:
                            if os.path.exists(p):
                                os.remove(p)
                    
                    print("Saving new best models...")
                    final_paths = []
                    for temp_p in temp_paths:
                        final_p = temp_p.replace('temp_', '')
                        os.rename(temp_p, final_p)
                        final_paths.append(final_p)
                    best_model_paths = final_paths
                else:
                    for temp_p in temp_paths:
                        if os.path.exists(temp_p):
                            os.remove(temp_p)

            print("\nHyperparameter Search Finished")
            print(f"Best F1 score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            params_path = os.path.join(config.model_save_dir, 'best_params_cnn.json')
            os.makedirs(os.path.dirname(params_path), exist_ok=True)
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"Best CNN parameters saved to {params_path}")
        else:
            _train_pytorch_segment_cv(config, experiment)
    elif config.active_model == 'ssl':
        if config.hyperparameter_search_mode:
            print("Starting Hyperparameter Search for SSL model...")
            best_score = -1
            best_params = {}
            best_model_paths = []

            all_param_combinations = []
            base_params = list(itertools.product(
                config.seq_model_type, config.seq_hidden_size, config.learning_rate,
                config.dropout_rate, config.seq_num_layers, config.batch_size,
                config.chunk_segments, config.chunk_overlap_segments
            ))

            for (seq_type, hidden_size, lr, dropout, num_layers, batch_size, chunk_seg, chunk_over) in base_params:
                if chunk_over >= chunk_seg:
                    continue
                if seq_type == 'transformer':
                    for nhead in config.transformer_nhead:
                        all_param_combinations.append((seq_type, hidden_size, lr, dropout, num_layers, batch_size, chunk_seg, chunk_over, nhead))
                elif seq_type == 'bilstm':
                    all_param_combinations.append((seq_type, hidden_size, lr, dropout, num_layers, batch_size, chunk_seg, chunk_over, None))

            for i, params in enumerate(all_param_combinations):
                run_config = copy.deepcopy(config)
                seq_type, hidden_size, lr, dropout, num_layers, batch_size, chunk_seg, chunk_over, nhead = params

                run_config.use_all_layers = True
                run_config.seq_model_type = seq_type
                run_config.seq_hidden_size = hidden_size
                run_config.learning_rate = lr
                run_config.dropout_rate = dropout
                run_config.seq_num_layers = num_layers
                run_config.batch_size = batch_size
                run_config.chunk_segments = chunk_seg
                run_config.chunk_overlap_segments = chunk_over
                if nhead is not None: run_config.transformer_nhead = nhead
                
                current_params = {
                    'seq_model_type': seq_type, 'seq_hidden_size': hidden_size, 'learning_rate': lr,
                    'dropout_rate': dropout, 'seq_num_layers': num_layers, 'batch_size': batch_size,
                    'chunk_segments': chunk_seg, 'chunk_overlap_segments': chunk_over,
                    'transformer_nhead': nhead
                }

                print(f"\nRun {i+1}/{len(all_param_combinations)}")
                print(f"Testing params: {current_params}")

                avg_f1, temp_paths = _train_pytorch_participant_cv(run_config, experiment)
                print(f"Result for params {current_params}: Average F1 = {avg_f1:.4f}")

                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_params = current_params
                    print(f"New best score found: {best_score:.4f}")
                    if best_model_paths:
                        for p in best_model_paths:
                            if os.path.exists(p):
                                os.remove(p)
                    
                    print("Saving new best models...")
                    final_paths = []
                    for temp_p in temp_paths:
                        final_p = temp_p.replace('temp_', '')
                        os.rename(temp_p, final_p)
                        final_paths.append(final_p)
                    best_model_paths = final_paths
                else:
                    for temp_p in temp_paths:
                        if os.path.exists(temp_p):
                            os.remove(temp_p)

            print("\nHyperparameter Search Finished")
            print(f"Best F1 score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            params_path = os.path.join(config.model_save_dir, 'best_params_ssl.json')
            os.makedirs(os.path.dirname(params_path), exist_ok=True)
            with open(params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"Best SSL architecture parameters saved to {params_path}")
        else:
            params_path = os.path.join(config.model_save_dir, 'best_params_ssl.json')
            if not os.path.exists(params_path):
                print("Please run the script with 'hyperparameter_search_mode: true' first.")
                return

            with open(params_path, 'r') as f:
                best_params = json.load(f)

            final_config = copy.deepcopy(config)
            for key, value in best_params.items():
                setattr(final_config, key, value)
            
            _train_pytorch_participant_cv(final_config, experiment)

if __name__ == "__main__":
    main()