
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import src_utils
import torchaudio
import numpy as np
from config import SSLConfig
from preprocessor import E1_DAIC
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

class Dataset(TorchDataset):

    def __init__(self, audio_paths : str, labels : str, return_audio_id : bool = True, config : SSLConfig = SSLConfig()):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = config.sample_rate
        self.segment_indices = []
        self.segment_ms = config.segment_ms
        self.segment_samples = int(config.sample_rate * (config.segment_ms / 1000.0))
        self.hop_ms = config.hop_ms
        self.hop_samples = int(config.sample_rate * (config.hop_ms / 1000.0))
        self.return_audio_id = return_audio_id

        # Precompute segment indices for each audio file
        for id, audio_path in enumerate(self.audio_paths):
            num_frames = torchaudio.info(audio_path).num_frames
            starts = np.arange(0, num_frames - self.segment_samples + 1, self.hop_samples)
            self.segment_indices.extend([(id, int(start)) for start in starts])

    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):

        id, start_sample = self.segment_indices[idx]
        file_path = self.audio_paths[id]
        label = self.labels[id]

        waveform_segment, _ = torchaudio.load(file_path,
                                              frame_offset = start_sample,
                                              num_frames = self.segment_samples)
        
        item = {'input_values': waveform_segment, 
                'label': torch.tensor([label], dtype = torch.float32)}
        
        if self.return_audio_id:
            item['audio_id'] = id

        return item

class DataLoader():
    
    def __init__(self, config : SSLConfig = SSLConfig()):
        self.config = config
        self.batch_size = config.batch_size
        self.preprocessor = E1_DAIC(config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.split_dataset()

    def __get_generators(self):

        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = src_utils.get_splits(self.splits)

        train_dataset = Dataset(
            audio_paths = train_paths,
            labels = train_labels,
            return_audio_id = True,
            config = self.config
        )
        
        test_dataset = Dataset(
            audio_paths = test_paths,
            labels = test_labels,
            return_audio_id = True,
            config = self.config
        )
        
        dev_dataset = Dataset(
            audio_paths = dev_paths,
            labels = dev_labels,
            return_audio_id = True,
            config = self.config
        )

        return train_dataset, test_dataset, dev_dataset

    def load_data(self):
        
        train_dataset, test_dataset, dev_dataset = self.__get_generators()

        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())
        
        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())
        
        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())

        return train_loader, test_loader, dev_loader