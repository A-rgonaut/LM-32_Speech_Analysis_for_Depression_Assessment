import os
import random
import torch
import torchaudio
import numpy as np
from collections import Counter
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, Sampler

from .config import CNNConfig
from ..src_utils import get_splits, filter_edaic_samples
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, return_audio_id : bool = True, config : CNNConfig = CNNConfig(),
                  balance_segments : bool = False):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = config.sample_rate
        self.segment_samples = config.segment_samples
        self.hop_samples = config.hop_samples
        self.return_audio_id = return_audio_id
        self.balance_segments = balance_segments

        self.segment_indices = []
        self.__precompute_segments()
        self.__print_segment_balance(audio_paths, labels, self.segment_samples, self.hop_samples)

    def __print_segment_balance(self, audio_paths, labels, segment_samples, hop_samples):
        class_counts = Counter(labels)
        print("Numero audio per classe:", class_counts)

        seg_counts = {0: 0, 1: 0}
        total_durations = {0: 0.0, 1: 0.0}
        for i, path in enumerate(audio_paths):
            label = labels[i]
            info = torchaudio.info(path)
            if self.random_crop_to_shortest:
                num_frames_to_segment = self.crop_len_samples
            else:
                num_frames_to_segment = info.num_frames

            duration = num_frames_to_segment / info.sample_rate
            total_durations[label] += duration
            n_segments = max(1, (num_frames_to_segment - segment_samples) // hop_samples + 1)
            seg_counts[label] += n_segments

        print("Numero segmenti per classe:", seg_counts)
        print("Durata totale (s) per classe:", total_durations)

    def __precompute_segments(self):
        all_segments = []

        for id, audio_path in enumerate(self.audio_paths):
            num_frames = torchaudio.info(audio_path).num_frames
            starts = np.arange(0, num_frames - self.segment_samples + 1, self.hop_samples)
            all_segments.extend([(id, int(start), self.segment_samples) for start in starts])

        if self.balance_segments:
            depressed_segments = []
            non_depressed_segments = []
            for seg in all_segments:
                audio_id = seg[0]
                if self.labels[audio_id] == 1:
                    depressed_segments.append(seg)
                else:
                    non_depressed_segments.append(seg)
            
            min_segments = min(len(depressed_segments), len(non_depressed_segments))
            
            # Campionamento senza rimpiazzo
            depressed_segments = random.sample(depressed_segments, min_segments)
            non_depressed_segments = random.sample(non_depressed_segments, min_segments)
            
            self.segment_indices = depressed_segments + non_depressed_segments
            random.shuffle(self.segment_indices)
            print(f"Training set balanced: {len(depressed_segments)} depressed vs {len(non_depressed_segments)} non-depressed segments.")
        else:
            self.segment_indices = all_segments


    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        id, start_sample, num_frames = self.segment_indices[idx]
        file_path = self.audio_paths[id]
        label = self.labels[id]
        
        waveform_segment, _ = torchaudio.load(file_path,
                                            frame_offset = start_sample,
                                            num_frames = num_frames)
        #mean = torch.mean(waveform_segment, dim=1, keepdim=True)
        #std_dev = torch.std(waveform_segment, dim=1, keepdim=True)
        #input_values = (waveform_segment - mean) / (std_dev + 1e-12)
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
        self.splits = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits = filter_edaic_samples(self.splits)

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

        if self.balance_segments:
            train_batch_sampler = BalancedBatchSampler(train_dataset, self.batch_size)
            train_loader = TorchDataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=os.cpu_count(),
                pin_memory=True
            )
        else:
            train_loader = TorchDataLoader(
                train_dataset, 
                batch_size = self.batch_size,
                shuffle=True,
                num_workers=os.cpu_count(),
                pin_memory=True
            )
        
        '''
        # Testare se serve effettivamente anche bilanciare il batch perfettamente
        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            pin_memory=True
        )
        '''
            
        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())
        
        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())

        return train_loader, test_loader, dev_loader
    

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Assicurati che il batch size sia pari per un bilanciamento 50/50
        if self.batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced sampling.")

        # Ottieni gli indici dei segmenti per ogni classe
        self.depressed_indices = []
        self.non_depressed_indices = []
        for i, segment_info in enumerate(self.dataset.segment_indices):
            audio_id = segment_info[0]
            label = self.dataset.labels[audio_id]
            if label == 1:
                self.depressed_indices.append(i)
            else:
                self.non_depressed_indices.append(i)
        
        self.num_batches = min(len(self.depressed_indices), len(self.non_depressed_indices)) // (self.batch_size // 2)

    def __iter__(self):
        # Mischia gli indici all'inizio di ogni epoca
        random.shuffle(self.depressed_indices)
        random.shuffle(self.non_depressed_indices)
        
        samples_per_class = self.batch_size // 2
        for i in range(self.num_batches):
            start_dep = i * samples_per_class
            start_non_dep = i * samples_per_class
            
            dep_batch = self.depressed_indices[start_dep : start_dep + samples_per_class]
            non_dep_batch = self.non_depressed_indices[start_non_dep : start_non_dep + samples_per_class]
            
            batch = dep_batch + non_dep_batch
            random.shuffle(batch) # Mischia i campioni all'interno del batch
            yield batch

    def __len__(self):
        return self.num_batches