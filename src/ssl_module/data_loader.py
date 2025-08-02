import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from tqdm import tqdm
#from torch_audiomentations import Compose, PitchShift

from .config import SSLConfig
from ..src_utils import get_splits, filter_edaic_samples
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, feature_paths: str, labels: str, config: SSLConfig = SSLConfig(), segment_indices: list = None):
        self.config = config
        self.feature_paths = feature_paths
        self.labels = labels
        self.segment_indices = segment_indices
        self.layer_to_extract = 8

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        label = self.labels[idx]

        # Carica tutti gli hidden states pre-calcolati per un audio
        # hidden_states è una tupla di tensori: (num_layers, num_segments, num_frames, hidden_size)
        hidden_states = torch.load(feature_path, map_location='cpu')
        
        # Estrai solo l'hidden state che ti interessa
        # Shape: (num_segments, num_frames, hidden_size)
        segment_features = hidden_states[self.layer_to_extract]

        # Se stai usando sub-dialogue shuffling, seleziona solo i segmenti pertinenti
        if self.segment_indices and self.segment_indices[idx] is not None:
            start_idx, end_idx = self.segment_indices[idx]
            segment_features = segment_features[start_idx:end_idx+1]

        num_segments = segment_features.shape[0]

        return {
            'features': segment_features, 
            'label': torch.tensor(label, dtype=torch.float32),
            'num_segments': num_segments
        }
    
def collate_fn_features(batch):
    """Collate function per le feature pre-calcolate."""
    max_segments = max([item['num_segments'] for item in batch])
    
    batch_features = []
    batch_labels = []
    batch_masks = []
    
    # Assumiamo che num_frames e hidden_size siano costanti
    # Prendiamo le dimensioni dal primo elemento
    _, num_frames, hidden_size = batch[0]['features'].shape

    for item in batch:
        features = item['features']
        num_segments = item['num_segments']

        mask = torch.zeros(max_segments, dtype=torch.bool)
        mask[num_segments:] = True
        batch_masks.append(mask)

        # Pad se necessario
        if num_segments < max_segments:
            padding_shape = (max_segments - num_segments, num_frames, hidden_size)
            padding = torch.zeros(padding_shape, dtype=features.dtype)
            features = torch.cat([features, padding], dim=0)
        
        batch_features.append(features)
        batch_labels.append(item['label'])

    return {
        'features': torch.stack(batch_features),
        'label': torch.stack(batch_labels),
        'attention_mask': torch.stack(batch_masks)
    }

class DataLoader():
    def __init__(self, config : SSLConfig = SSLConfig()):
        self.config = config
        self.batch_size = config.batch_size
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits = filter_edaic_samples(self.splits)

    def _get_feature_paths(self, original_audio_paths):
        """Converte i percorsi audio nei percorsi delle feature pre-calcolate."""
        feature_dir = self.config.feature_path
        feature_paths = []
        for audio_path in original_audio_paths:
            # Es: .../300_AUDIO.wav -> 300_AUDIO.pt
            filename = os.path.basename(audio_path).replace('.wav', '.pt')
            # Es: features/wav2vec_base/train/300_AUDIO.pt
            split_name = os.path.basename(os.path.dirname(audio_path)) # train, dev, o test
            feature_paths.append(os.path.join(feature_dir, split_name, filename))
        return feature_paths

    def _apply_subdialogue_shuffling(self, original_paths, original_labels):
        print("Applying sub-dialogue shuffling for data augmentation...")
        
        # Algoritmo 1: Passi 1-4
        N_pos = sum(1 for label in original_labels if label == 1)
        N_neg = len(original_labels) - N_pos
        M_pos = self.config.subdialogue_M_pos
        M_neg = round(N_pos * M_pos / N_neg) if N_neg > 0 else 0
        
        el = self.config.subdialogue_len_low
        eh = self.config.subdialogue_len_high

        new_paths, new_labels, new_segment_indices = [], [], []

        for path, label in tqdm(zip(original_paths, original_labels), total=len(original_paths), desc="Generating Sub-dialogues"):
            # Carica le feature per ottenere il numero totale di segmenti
            feature_path = self._get_feature_paths([path])[0]
            if not os.path.exists(feature_path): continue
            
            # Carichiamo solo il primo layer per avere la shape
            num_segments = torch.load(feature_path, map_location='cpu')[0].shape[0]
            
            if num_segments < 2: continue

            M = M_pos if label == 1 else M_neg
            
            for _ in range(M):
                e_len_fraction = np.random.uniform(el, eh)
                length = round(e_len_fraction * num_segments)
                length = max(1, min(length, num_segments)) 
                upper_bound = num_segments - length
                if upper_bound <= 0:
                    start_idx = 0
                else:
                    start_idx = np.random.randint(0, upper_bound)
                end_idx = start_idx + length
                
                new_paths.append(path)
                new_labels.append(label)
                new_segment_indices.append((start_idx, end_idx))

        print(f"Augmentation complete. Original samples: {len(original_paths)}, New samples: {len(new_paths)}")
        return new_paths, new_labels, new_segment_indices

                
    def __get_generators(self):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits)

        segment_indices_train = None
        if self.config.use_subdialogue_shuffling:
            train_paths, train_labels, segment_indices_train = self._apply_subdialogue_shuffling(train_paths, train_labels)

        train_feature_paths = self._get_feature_paths(train_paths)
        test_feature_paths = self._get_feature_paths(test_paths)
        dev_feature_paths = self._get_feature_paths(dev_paths)
            
        train_dataset = Dataset(
            feature_paths=train_feature_paths,
            labels=train_labels,
            config=self.config,
            segment_indices=segment_indices_train
        )

        test_dataset = Dataset(
            feature_paths=test_feature_paths,
            labels=test_labels,
            config=self.config
        )
        
        dev_dataset = Dataset(
            feature_paths=dev_feature_paths,
            labels=dev_labels,
            config=self.config
        )

        return train_dataset, test_dataset, dev_dataset

    def load_data(self):
        train_dataset, test_dataset, dev_dataset = self.__get_generators()

        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn_features,
            pin_memory=True
        )

        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            num_workers=os.cpu_count(),
            collate_fn=collate_fn_features,
            pin_memory=True
        )

        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size=self.batch_size, 
            num_workers=os.cpu_count(),
            collate_fn=collate_fn_features,
            pin_memory=True
        )

        return train_loader, test_loader, dev_loader