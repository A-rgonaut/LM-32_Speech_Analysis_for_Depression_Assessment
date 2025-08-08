import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from .config import SSLConfig
from ..utils import get_splits, filter_edaic_samples
from ..audio_utils import load_audio, segment_audio_by_transcript, segment_audio_sliding_window
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, config : SSLConfig = SSLConfig(), 
                 subdialogue_windows: list = None, is_train: bool = False):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.subdialogue_windows = subdialogue_windows
        if not config.use_preextracted_features:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name, do_normalize=False)
        self.sample_rate = config.sample_rate
        self.segment_samples = int(config.max_utt_seconds * config.sample_rate)
        self.max_segments = config.max_segments if is_train else None
        self.path_to_id = {path: os.path.basename(path).replace('_AUDIO.wav', '') for path in audio_paths}

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        if self.config.use_preextracted_features:
            feature_filename = os.path.basename(audio_path).split('.')[0]  # Remove file extension
            feature_path = os.path.join(self.config.feature_path, f'{feature_filename}_layer_{self.config.layer_to_use}.pt')

            segment_features = torch.load(feature_path, map_location='cpu', weights_only=True) 

            if self.subdialogue_windows is not None:
                start_idx, end_idx = self.subdialogue_windows[idx]
                segment_features = segment_features[start_idx:end_idx]

            num_segments = segment_features.shape[0]

            if self.max_segments and num_segments > self.max_segments:
                segment_features = segment_features[:self.max_segments]
                num_segments = self.max_segments
            
            return {
                'segment_features': segment_features, # (num_segments, hidden_size)
                'label': torch.tensor(label, dtype=torch.float32),
                'num_segments': num_segments
            }
        else:
            audio = load_audio(audio_path, self.sample_rate)

            segments = []
            if self.config.segmentation_strategy == 'transcript':
                transcript_path = audio_path.replace("_AUDIO.wav", "_Transcript.csv")
                transcript_df = pd.read_csv(transcript_path)
                segments = segment_audio_by_transcript(audio, transcript_df)
            else: # 'fixed_length'
                segments = segment_audio_sliding_window(audio)
            
            if self.subdialogue_windows is not None:
                start_idx, end_idx = self.subdialogue_windows[idx]
                segments = segments[start_idx:end_idx]

            # Limit the number of segments if specified
            if self.max_segments and len(segments) > self.max_segments:
                segments = segments[:self.max_segments]

            all_input_values = []
            all_attention_masks = []
            
            for segment in segments:
                features = self.feature_extractor(
                    segment, 
                    sampling_rate=self.sample_rate,
                    max_length=self.segment_samples,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    return_attention_mask=True,
                )
                # features['input_values'] has shape [1, N]
                all_input_values.append(features['input_values'].squeeze(0)) 
                all_attention_masks.append(features['attention_mask'].squeeze(0))
            
            stacked_input_values = torch.stack(all_input_values)
            stacked_attention_masks = torch.stack(all_attention_masks)
            
            return {
                'input_values': stacked_input_values,  # (num_segments, seq_len)
                'attention_mask_segment': stacked_attention_masks,  # (num_segments,)
                'label': torch.tensor(label, dtype=torch.float32),
                'num_segments': len(segments)
            }

def collate_fn(batch):
    """
    This function is used because different audio files can have a different number of segments.
    For example:
    - Audio 1: 30 seconds → 3 segments of 10s
    - Audio 2: 50 seconds → 5 segments of 10s

    To create a uniform batch, we need to pad to the maximum number of segments.
    """
    # Find the maximum number of segments in the batch
    max_segments = max([item['num_segments'] for item in batch])
    
    batch_features = []
    batch_input_values = []
    batch_labels = []
    batch_audio_masks = []
    batch_segment_masks = []

    is_preextracted = 'segment_features' in batch[0]
    
    for item in batch:
        num_segments = item['num_segments']

        # Audio-level mask (for padding entire segments)
        audio_mask = torch.zeros(max_segments, dtype=torch.bool)
        audio_mask[num_segments:] = True # True where segments are padding
        
        batch_audio_masks.append(audio_mask)
        batch_labels.append(item['label'])

        # Pad if necessary
        if num_segments < max_segments:
            if is_preextracted:
                # Pad pre-extracted features
                features = item['segment_features']
                padding_shape = (max_segments - num_segments, features.shape[1])
                padding = torch.zeros(padding_shape, dtype=features.dtype)
                padded_features = torch.cat([features, padding], dim=0)
                batch_features.append(padded_features)
            else:
                # Pad input_values
                input_values = item['input_values']
                padding_shape_values = (max_segments - num_segments, input_values.shape[1])
                padding_values = torch.zeros(padding_shape_values, dtype=input_values.dtype)
                input_values = torch.cat([input_values, padding_values], dim=0)
                
                # Pad segment-level attention mask
                attention_mask_segment = item['attention_mask_segment']
                padding_shape_mask = (max_segments - num_segments, attention_mask_segment.shape[1])
                padding_mask = torch.zeros(padding_shape_mask, dtype=attention_mask_segment.dtype)
                attention_mask_segment = torch.cat([attention_mask_segment, padding_mask], dim=0)
                
                batch_input_values.append(input_values)
                batch_segment_masks.append(attention_mask_segment)
        else:
            if is_preextracted:
                batch_features.append(item['segment_features'])
            else:
                batch_input_values.append(item['input_values'])
                batch_segment_masks.append(item['attention_mask_segment'])
    
    if is_preextracted:
        return {
            'segment_features': torch.stack(batch_features),
            'attention_mask_audio': torch.stack(batch_audio_masks),
            'label': torch.stack(batch_labels)
        }
    else:
        return {
            'input_values': torch.stack(batch_input_values),
            'attention_mask_segment': torch.stack(batch_segment_masks),
            'attention_mask_audio': torch.stack(batch_audio_masks),
            'label': torch.stack(batch_labels)
        }

class DataLoader():
    def __init__(self, config : SSLConfig = SSLConfig()):
        self.config = config
        self.batch_size = config.batch_size
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits = filter_edaic_samples(self.splits)
        self._load_metadata()

    def _load_metadata(self):
        self.metadata = {}
        metadata_dir = self.config.feature_path
        print(f"Loading metadata from: {metadata_dir}")

        for split in ['train', 'dev', 'test']:
            metadata_path = os.path.join(metadata_dir, f'{split}_metadata.csv')
            if os.path.exists(metadata_path):
                df = pd.read_csv(metadata_path)
                self.metadata.update(pd.Series(df.num_segments.values, index=df.filename).to_dict())

    def _apply_subdialogue_shuffling(self, original_paths, original_labels):
        print("Applying sub-dialogue shuffling for data augmentation...")
        
        # Algoritmo 1: Passi 1-4
        N_pos = sum(1 for label in original_labels if label == 1)
        N_neg = len(original_labels) - N_pos
        M_pos = self.config.subdialogue_M_pos
        M_neg = round(N_pos * M_pos / N_neg) if N_neg > 0 else 0
        
        el = self.config.subdialogue_len_low
        eh = self.config.subdialogue_len_high

        new_paths, new_labels, new_windows = [], [], []
        path_to_label = {path: label for path, label in zip(original_paths, original_labels)}
        
        # Algoritmo 1: Passi 6-17
        for path in tqdm(path_to_label.keys(), desc="Generating Sub-dialogues"):
            label = path_to_label[path]
            filename = os.path.basename(path).replace('.wav', '.pt')
            
            total_segments = self.metadata[filename]
            if total_segments < 2:
                continue

            M = M_pos if label == 1 else M_neg
            
            for _ in range(M):
                e_len_fraction = np.random.uniform(el, eh)
                num_sub_segments = int(e_len_fraction * total_segments)

                if total_segments - num_sub_segments <= 0:
                    start_segment_idx = 0
                else:
                    start_segment_idx = np.random.randint(0, total_segments - num_sub_segments + 1)
                
                end_segment_idx = start_segment_idx + num_sub_segments
                
                new_paths.append(path)
                new_labels.append(label)
                new_windows.append((start_segment_idx, end_segment_idx))

        print(f"Augmentation complete. Original samples: {len(original_paths)}, New samples: {len(new_paths)}")
        return new_paths, new_labels, new_windows

    def get_dataset(self, split: str):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits)
        aug_train_windows = None
        if self.config.use_subdialogue_shuffling and split == 'train':
            train_paths, train_labels, aug_train_windows = self._apply_subdialogue_shuffling(train_paths, train_labels)

        if split == 'train':
            dataset = Dataset(audio_paths=train_paths, labels=train_labels, config=self.config, subdialogue_windows=aug_train_windows)
        elif split == 'dev':
            dataset = Dataset(audio_paths=dev_paths, labels=dev_labels, config=self.config)
        elif split == 'test':
            dataset = Dataset(audio_paths=test_paths, labels=test_labels, config=self.config)
        
        return dataset

    def get_data_loader(self, split: str, dataset: TorchDataset = None):
        if dataset is None:
            dataset = self.get_dataset(split)
            
        return TorchDataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn,
            pin_memory=True
        )