import os
import random
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, Sampler
from transformers import AutoFeatureExtractor

from .config import SSLConfig
from ..src_utils import get_splits, filter_edaic_samples
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, config : SSLConfig = SSLConfig()):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name, do_normalize=False)
        self.sample_rate = config.sample_rate
        self.segment_samples = int(config.max_utt_seconds * config.sample_rate)
        self.max_segments = config.max_segments

    def _segment_audio(self, audio):
        """Segments the audio into fixed-length chunks."""
        segments = []
        
        # If the audio is shorter than the segment length, pad with zeros
        if len(audio) < self.segment_samples:
            padded_audio = np.zeros(self.segment_samples)
            padded_audio[:len(audio)] = audio
            segments.append(padded_audio)
        else:
            # Divide in segments of fixed length
            for i in range(0, len(audio), self.segment_samples):
                segment = audio[i:i + self.segment_samples]

                # If the last segment is too short, pad with zeros
                if len(segment) < self.segment_samples:
                    padded_segment = np.zeros(self.segment_samples)
                    padded_segment[:len(segment)] = segment
                    segment = padded_segment
                
                segments.append(segment)

                # Limit the number of segments if specified
                if self.max_segments and len(segments) >= self.max_segments:
                    break
        
        return np.array(segments)


    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        segments = self._segment_audio(audio)
        
        segment_features = []
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
            segment_features.append(features.input_values[0])
        
        segment_features = torch.stack(segment_features)  # (num_segments, seq_len)
        
        return {
            'input_values': segment_features, 
            'label': torch.tensor(label, dtype=torch.long),
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
    
    batch_input_values = []
    batch_labels = []
    batch_masks = []
    
    for item in batch:
        input_values = item['input_values']
        num_segments = item['num_segments']

        mask = torch.zeros(max_segments, dtype=torch.bool)
        mask[num_segments:] = True
        batch_masks.append(mask)

        # Pad if necessary
        if num_segments < max_segments:
            padding_shape = (max_segments - num_segments, input_values.shape[1])
            padding = torch.zeros(padding_shape, dtype=input_values.dtype)
            input_values = torch.cat([input_values, padding], dim=0)
        
        batch_input_values.append(input_values)
        batch_labels.append(item['label'])

    return {
        'input_values': torch.stack(batch_input_values),
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

    def __get_generators(self):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits)

        train_dataset = Dataset(
            audio_paths = train_paths,
            labels = train_labels,
            config = self.config
        )
        
        test_dataset = Dataset(
            audio_paths = test_paths,
            labels = test_labels,
            config = self.config
        )
        
        dev_dataset = Dataset(
            audio_paths = dev_paths,
            labels = dev_labels,
            config = self.config
        )

        return train_dataset, test_dataset, dev_dataset

    def load_data(self):
        train_dataset, test_dataset, dev_dataset = self.__get_generators()

        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn,
            #pin_memory=True
        )

        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count(),
            collate_fn=collate_fn,
            #pin_memory=True
        )

        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count(),
            collate_fn=collate_fn,
            #pin_memory=True
        )

        return train_loader, test_loader, dev_loader