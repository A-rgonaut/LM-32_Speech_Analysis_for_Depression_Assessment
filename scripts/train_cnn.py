import os
from dotenv import load_dotenv
from comet_ml import Experiment
import numpy as np
import random
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.cnn_module.config import CNNConfig
from src.cnn_module.data_loader import DataLoader
from src.cnn_module.model import CNNModel
from src.cnn_module.trainer import Trainer
from src.utils import set_seed, clear_cache

def main():
    load_dotenv()
    experiment = Experiment(
        api_key = os.getenv("COMET_API_KEY"),
        project_name = os.getenv("COMET_PROJECT_NAME"),
        workspace = os.getenv("COMET_WORKSPACE")
    )

    config = CNNConfig()
    set_seed(config.seed)

    data_loader = DataLoader(config)
    train_dataset = data_loader.get_dataset('train')

    segment_indices_for_split = np.arange(len(train_dataset.segments))
    segment_labels = [train_dataset.id_to_label[seg[0]] for seg in train_dataset.segments] # seg[0] is the audio_id
    segment_groups = [seg[0] for seg in train_dataset.segments]
    
    if config.k_folds == 1:
        kfold = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=config.seed)
    else:
        kfold = StratifiedGroupKFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)

    for fold, (train_segment_idx, val_segment_idx) in enumerate(kfold.split(segment_indices_for_split, segment_labels, segment_groups)):
        print(f"\nTraining Fold {fold + 1}/{config.k_folds}")

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
        model = CNNModel(config)
        
        trainer = Trainer(model, train_loader, val_loader, config)
        with experiment.context_manager(f"fold_{fold+1}"):
            trainer.train(experiment, f'cnn_model_fold_{fold + 1}.pth')
        clear_cache()

    print("\nK-Fold training complete. All models have been saved.")

if __name__ == "__main__":
    main()