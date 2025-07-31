import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader

from .config import SSLConfig
from ..src_utils import get_splits, filter_edaic_samples
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths, labels, config=SSLConfig(), balance_segments=False):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = config.sample_rate
        self.max_utt_seconds = config.max_utt_seconds
        self.min_utt_seconds = config.min_utt_seconds
        self.utt_overlap = config.utt_overlap

        self.interviews = self.__precompute_interviews()

    def __precompute_interviews(self):
        interviews = []
        for idx, audio_path in enumerate(self.audio_paths):
            transcript_path = audio_path.replace('_AUDIO.wav', '_Transcript.csv')
            df = pd.read_csv(transcript_path)
            utterances = []
            for _, row in df.iterrows():
                start = float(row['Start_Time'])
                end = float(row['End_Time'])
                duration = end - start
                if duration < self.config.min_utt_seconds:
                    continue  # Remove short utterances
                elif duration <= self.config.max_utt_seconds:
                    utterances.append((start, end))
                else:
                    # Split long utterance into segments of max max_utt_seconds with overlap
                    seg_start = start
                    while seg_start < end:
                        seg_end = min(seg_start + self.config.max_utt_seconds, end)
                        if seg_end - seg_start >= self.config.min_utt_seconds:
                            utterances.append((seg_start, seg_end))
                        seg_start += self.config.max_utt_seconds - self.config.utt_overlap
            if utterances:
                interviews.append({
                    'audio_path': audio_path,
                    'utterances': utterances,
                    'label': self.labels[idx]
                })
        return interviews

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        interview = self.interviews[idx]
        audio, sr = torchaudio.load(interview['audio_path'])
        utterance_tensors = []
        for start, end in interview['utterances']:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            utt = audio[:, start_sample:end_sample]
            utt_len = end_sample - start_sample
            pad_len = int(self.config.max_utt_seconds * sr) - utt_len
            if pad_len > 0:
                utt = torch.nn.functional.pad(utt, (0, pad_len))
            utterance_tensors.append(utt)
        utterances_tensor = torch.stack(utterance_tensors)  # (T, C, L)
        item = {
            'utterances': utterances_tensor,  # shape: (T, C, L)
            'label': torch.tensor(interview['label'], dtype=torch.float32)
        }
        return item

class DataLoader():
    def __init__(self, config : SSLConfig = SSLConfig()):
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
            balance_segments = self.balance_segments,
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
            pin_memory=True
        )
            
        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())
        
        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())

        return train_loader, test_loader, dev_loader