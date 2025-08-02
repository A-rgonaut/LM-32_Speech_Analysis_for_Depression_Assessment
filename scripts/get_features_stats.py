import sys
import os
import torch
import pandas as pd
from tqdm import tqdm

from src.ssl_module.config import SSLConfig
from src.preprocessor import E1_DAIC
from src.src_utils import get_splits

def main():
    config = SSLConfig()
    preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
    splits = preprocessor.get_dataset_splits()
    train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(splits)

    all_splits = {
        "train": (train_paths, train_labels),
        "dev": (dev_paths, dev_labels),
        "test": (test_paths, test_labels)
    }

    layer_to_check = config.layer_to_extract

    for split_name, (paths, labels) in all_splits.items():
        depressed_segments = 0
        not_depressed_segments = 0
        
        print(f"\n--- Calculating Segment Stats for {split_name.upper()} SPLIT ---")
        
        for path, label in tqdm(zip(paths, labels), total=len(paths), desc=f"Processing {split_name}"):
            feature_filename = os.path.basename(path).replace('_AUDIO.wav', '.pt')
            feature_path = os.path.join(config.feature_path, split_name, feature_filename)
            
            if not os.path.exists(feature_path):
                continue
            
            try:
                # Carica gli hidden states e prendi il layer che ti interessa
                hidden_states = torch.load(feature_path, map_location='cpu')
                num_segments = hidden_states[layer_to_check].shape[0]
                
                if label == 1:
                    depressed_segments += num_segments
                else:
                    not_depressed_segments += num_segments
            except Exception as e:
                print(f"Could not process file {feature_path}: {e}")

        total_segments = depressed_segments + not_depressed_segments
        if total_segments == 0:
            print(f"--- No segments found for {split_name} split ---")
            continue

        print(f"Depressed segments:     {depressed_segments} ({depressed_segments/total_segments:.2%})")
        print(f"Not Depressed segments: {not_depressed_segments} ({not_depressed_segments/total_segments:.2%})")
        print(f"Total segments:         {total_segments}")
        print("------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()