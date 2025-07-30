import os
import sys
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

from .config import CNNConfig
from ..src_utils import get_splits, filter_edaic_samples
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, return_audio_id : bool = True, config : CNNConfig = CNNConfig(), balance_segments : bool = False):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = config.sample_rate
        self.segment_samples = config.segment_samples
        self.hop_samples = config.hop_samples
        self.return_audio_id = return_audio_id

        self.segment_indices = []
        self.__precompute_segments()

        if balance_segments:
            self.__balance_segments()

    def __balance_segments(self):
        print("\n--- Bilanciamento dei segmenti in corso (under-sampling)... ---")
        
        # Separa gli indici dei segmenti per classe
        segments_class_0 = []
        segments_class_1 = []

        # Se non usiamo l'augmentazione E-DAIC, bilanciamo solo i segmenti DAIC
        if not self.config.edaic_aug:
            daic_segment_indices = []
            for segment_index in self.segment_indices:
                audio_id = segment_index[0]
                participant_id = os.path.basename(self.audio_paths[audio_id]).split('_AUDIO.wav')[0]
                if int(participant_id) < 600:
                    daic_segment_indices.append(segment_index)
            
            target_segments = daic_segment_indices
        else:
            target_segments = self.segment_indices

        for segment_index in target_segments:
            audio_id = segment_index[0]
            if self.labels[audio_id] == 0:
                segments_class_0.append(segment_index)
            else:
                segments_class_1.append(segment_index)

        n_0, n_1 = len(segments_class_0), len(segments_class_1)
        print(f"Segmenti originali: Classe 0: {n_0}, Classe 1: {n_1}")

        # Esegui l'under-sampling
        if n_0 > n_1:
            # Sotto-campiona la classe 0
            segments_class_0 = random.sample(segments_class_0, n_1)
        elif n_1 > n_0:
            # Sotto-campiona la classe 1
            segments_class_1 = random.sample(segments_class_1, n_0)
        
        # Combina e mescola i segmenti bilanciati
        self.segment_indices = segments_class_0 + segments_class_1
        random.shuffle(self.segment_indices)
        
        print(f"Segmenti bilanciati: Classe 0: {len(segments_class_0)}, Classe 1: {len(segments_class_1)}")
        print(f"Numero totale di segmenti dopo il bilanciamento: {len(self.segment_indices)}\n")
    
    def __precompute_segments(self):
        for id, audio_path in enumerate(self.audio_paths):
            num_frames = torchaudio.info(audio_path).num_frames
            starts = np.arange(0, num_frames - self.segment_samples + 1, self.hop_samples)
            self.segment_indices.extend([(id, int(start), self.segment_samples) for start in starts])

    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        id, start_sample, num_frames = self.segment_indices[idx]
        file_path = self.audio_paths[id]
        label = self.labels[id]
        
        waveform_segment, _ = torchaudio.load(file_path,
                                            frame_offset = start_sample,
                                            num_frames = num_frames)
        input_values = waveform_segment

        item = {'input_values': input_values, 
                'label': torch.tensor(label, dtype = torch.float32)}
        
        if self.return_audio_id:
            item['audio_id'] = id

        return item

class DataLoader():
    def __init__(self, config : CNNConfig = CNNConfig()):
        self.config = config
        self.batch_size = config.batch_size
        self.balance_segments = config.balance_segments
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        splits = self.preprocessor.get_dataset_splits()
        self.splits = filter_edaic_samples(splits)

    def __get_generators(self):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits)

        train_dataset = Dataset(
            audio_paths = train_paths,
            labels = train_labels,
            return_audio_id = False,
            balance_segments = self.balance_segments,
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
            return_audio_id = False,
            config = self.config
        )

        return train_dataset, test_dataset, dev_dataset

    def load_data(self):
        train_dataset, test_dataset, dev_dataset = self.__get_generators()

        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size,
            shuffle=True, 
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