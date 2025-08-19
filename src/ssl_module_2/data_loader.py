import os
import torch
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from transformers import AutoFeatureExtractor

from ..utils import get_splits, filter_edaic_samples
from ..audio_utils import load_audio, segment_audio_by_transcript, segment_audio_sliding_window
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, config):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = config.sample_rate
        self.max_utt_seconds = config.max_utt_seconds
        self.min_utt_seconds = config.min_utt_seconds
        self.overlap_seconds = config.overlap_seconds
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.ssl_model_name, do_normalize=True)
        self.id_to_label = {os.path.basename(path).replace('_AUDIO.wav', ''): label for path, label in zip(audio_paths, labels)}
        self.id_to_path = {os.path.basename(path).replace('_AUDIO.wav', ''): path for path in audio_paths}

        self.segments = []
        self.__precompute_segments()

    def __precompute_segments(self):
        for audio_path in self.audio_paths:
            audio_id = os.path.basename(audio_path).replace('_AUDIO.wav', '')
            audio = load_audio(audio_path)
            if self.config.segmentation_strategy == 'fixed_length':
                audio_segments_indices = segment_audio_sliding_window(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    max_utt_seconds=self.max_utt_seconds,
                    overlap_seconds=self.overlap_seconds,
                    min_utt_seconds=self.min_utt_seconds,
                    return_indices=True
                )
            elif self.config.segmentation_strategy == 'transcript':
                transcript_path = audio_path.replace('_AUDIO.wav', '_Transcript.csv')
                transcript_df = pd.read_csv(transcript_path)
                audio_segments_indices = segment_audio_by_transcript(
                    audio=audio,
                    transcript_df=transcript_df,
                    sample_rate=self.sample_rate,
                    max_utt_seconds=self.max_utt_seconds,
                    overlap_seconds=self.overlap_seconds,
                    min_utt_seconds=self.min_utt_seconds,
                    return_indices=True
                )
            self.segments.extend([(audio_id, seg[0], seg[1]) for seg in audio_segments_indices])

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        id, start_sample, end_sample = self.segments[idx]
        label = self.id_to_label[id]
        audio_path = self.id_to_path[id]

        segment = load_audio(
            audio_path=audio_path,
            sample_rate=self.sample_rate,
            offset_samples=start_sample,
            duration_samples=end_sample - start_sample
        
        )
        target_length = int(self.max_utt_seconds * self.sample_rate)

        features = self.feature_extractor(
            [segment], 
            sampling_rate=self.config.sample_rate,
            max_length=target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
        )

        return {
            'input_values': features['input_values'].squeeze(0),
            'attention_mask_segment': features['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32),
            'audio_id': torch.tensor(int(id), dtype=torch.long)
        }

class DataLoader():
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits = filter_edaic_samples(self.splits)
    
    def get_dataset(self, split: str):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits)

        if split == 'train':
            dataset = Dataset(audio_paths=train_paths, labels=train_labels, config=self.config)
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
            pin_memory=True
        )