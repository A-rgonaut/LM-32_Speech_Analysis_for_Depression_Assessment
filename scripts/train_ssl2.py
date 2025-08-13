import os
from dotenv import load_dotenv
from comet_ml import Experiment
import numpy as np
import random
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedGroupKFold

from src.ssl_module_2.config import SSLConfig
from src.ssl_module_2.data_loader import DataLoader
from src.ssl_module_2.model import SSLModel
from src.trainer import Trainer
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

    segment_indices_for_split = np.arange(len(train_dataset.segments))
    segment_labels = [train_dataset.id_to_label[seg[0]] for seg in train_dataset.segments] # seg[0] is the audio_id
    segment_groups = [seg[0] for seg in train_dataset.segments]
    
    if config.k_folds == 1:
        validation_split_fraction = 0.1
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
        if config.balance_segments:
            train_segments_in_fold = [train_dataset.segments[i] for i in train_segment_idx]
        
            depressed_segments = []
            non_depressed_segments = []
            for i, seg in enumerate(train_segments_in_fold):
                audio_id = seg[0]
                if train_dataset.id_to_label[audio_id] == 1:
                    depressed_segments.append(train_segment_idx[i])
                else:
                    non_depressed_segments.append(train_segment_idx[i])
            
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
        model = SSLModel(config)
        if fold == 0: 
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"SSLModel created.")
            print(f"  Total parameters: {total_params/1e6:.2f}M")
            print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
        
        trainer = Trainer(model, train_loader, val_loader, config)
        with experiment.context_manager(f"fold_{fold+1}"):
            trainer.train(experiment, f'cnn_model_fold_{fold + 1}.pth')
        clear_cache()

    print("\nK-Fold training complete. All models have been saved.")

if __name__ == "__main__":
    main()