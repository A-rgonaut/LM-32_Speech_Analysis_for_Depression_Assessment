import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from transformers import AutoFeatureExtractor
from tqdm import tqdm
#from torch_audiomentations import Compose, PitchShift

from .config import SSLConfig
from ..src_utils import get_splits, filter_edaic_samples
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, config : SSLConfig = SSLConfig(), time_windows: list = None):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.time_windows = time_windows
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name, do_normalize=False)
        self.sample_rate = config.sample_rate
        self.segment_samples = int(config.max_utt_seconds * config.sample_rate)
        self.max_segments = config.max_segments
        #self.augment = Compose([
        #    PitchShift(min_semitones=-2, max_semitones=2, p=0.4) # probability of applying pitch shift 40%
        #])

    def segment_audio_by_transcript(self, audio, transcript_df):
        """Segment audio based on transcript, grouping 5 utterances together."""
        segments = []
        num_utterances = 5

        for i in range(0, len(transcript_df), num_utterances):
            chunk = transcript_df.iloc[i:i + num_utterances]
            if chunk.empty:
                continue

            start_time = chunk['Start_Time'].iloc[0]
            end_time = chunk['End_Time'].iloc[-1]

            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            segment = audio[start_sample:end_sample]
            segments.append(segment)

            if self.max_segments and len(segments) >= self.max_segments:
                break

        return segments

    def _segment_audio_fixed_length(self, audio):
        """Segments the audio into fixed-length chunks."""
        segments = []

        # Divide in segments of fixed length.
        for i in range(0, len(audio), self.segment_samples):
            segment = audio[i:i + self.segment_samples]
            
            # Pad the last segment if it's shorter
            if len(segment) < self.segment_samples:
                padding = torch.zeros(self.segment_samples - len(segment), dtype=audio.dtype)
                segment = torch.cat([segment, padding], dim=0)

            segments.append(segment)

            # Limit the number of segments if specified
            if self.max_segments and len(segments) >= self.max_segments:
                break
        
        return torch.stack(segments)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        audio = audio.squeeze()
        audio = torch.tensor(audio, dtype=torch.float32)
        transcript_path = audio_path.replace("_AUDIO.wav", "_Transcript.csv")
        transcript_df = pd.read_csv(transcript_path)

        if self.time_windows is not None:
            start_time, end_time = self.time_windows[idx]
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            audio = audio[start_sample:end_sample]
            #audio = self.augment(samples=audio, sample_rate=self.sample_rate)
            valid_utterances = transcript_df[
                (transcript_df['Start_Time'] >= start_time) &
                (transcript_df['End_Time'] <= end_time)
            ].copy()
            valid_utterances['Start_Time'] -= start_time
            valid_utterances['End_Time'] -= start_time
            segments = self.segment_audio_by_transcript(audio, valid_utterances)
        else: 
            segments = self.segment_audio_by_transcript(audio, transcript_df)
        #segments = self._segment_audio_fixed_length(audio)
        
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
def _print_segment_stats(dataset: Dataset, split_name: str):
    """Stampa le statistiche sul numero di segmenti per audio depressi e non depressi,
    basandosi sulla segmentazione reale del Dataset."""
    depressed_segments = 0
    not_depressed_segments = 0

    print(f"Calculating segment stats for {split_name} split...")
    for i in range(len(dataset)):
        audio_path = dataset.audio_paths[i]
        audio, _ = librosa.load(audio_path, sr=dataset.sample_rate)
        transcript_path = audio_path.replace("_AUDIO.wav", "_Transcript.csv")
        transcript_df = pd.read_csv(transcript_path)

        if dataset.time_windows is not None:
            start_time, end_time = dataset.time_windows[i]
            start_sample = int(start_time * dataset.sample_rate)
            end_sample = int(end_time * dataset.sample_rate)
            audio = audio[start_sample:end_sample]
            valid_utterances = transcript_df[
                (transcript_df['Start_Time'] >= start_time) &
                (transcript_df['End_Time'] <= end_time)
            ].copy()
            valid_utterances['Start_Time'] -= start_time
            valid_utterances['End_Time'] -= start_time
            segments = dataset.segment_audio_by_transcript(audio, valid_utterances)
        else:
            segments = dataset.segment_audio_by_transcript(audio, transcript_df)
        
        num_segments = len(segments)

        if dataset.labels[i] == 1:
            depressed_segments += num_segments
        else:
            not_depressed_segments += num_segments
    
    total_segments = depressed_segments + not_depressed_segments
    if total_segments == 0:
        print(f"--- No segments found for {split_name} split ---")
        return

    print(f"\n--- Segment Stats for {split_name.upper()} SPLIT (Transcript-Based) ---")
    print(f"Depressed segments:     {depressed_segments} ({depressed_segments/total_segments:.2%})")
    print(f"Not Depressed segments: {not_depressed_segments} ({not_depressed_segments/total_segments:.2%})")
    print(f"Total segments:         {total_segments}")
    print("------------------------------------------------------------------\n")

class DataLoader():
    def __init__(self, config : SSLConfig = SSLConfig()):
        self.config = config
        self.batch_size = config.batch_size
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits = filter_edaic_samples(self.splits)

    def _apply_subdialogue_shuffling(self, original_paths, original_labels):
        print("Applying sub-dialogue shuffling for data augmentation...")
        
        # Algoritmo 1: Passi 1-4
        N_pos = sum(1 for label in original_labels if label == 1)
        N_neg = len(original_labels) - N_pos
        M_pos = self.config.subdialogue_M_pos
        M_neg = round(N_pos * M_pos / N_neg) if N_neg > 0 else 0
        
        el = self.config.subdialogue_len_low
        eh = self.config.subdialogue_len_high

        new_paths, new_labels, new_time_windows = [], [], []

        # Raggruppa per percorso per caricare ogni file audio e trascrizione una sola volta
        path_to_data = {}
        for path, label in zip(original_paths, original_labels):
            if path not in path_to_data:
                path_to_data[path] = {'label': label, 'transcript': None}
        
        # Algoritmo 1: Passi 6-17
        for path in tqdm(path_to_data.keys(), desc="Generating Sub-dialogues"):
            label = path_to_data[path]['label']
            transcript_path = path.replace("_AUDIO.wav", "_Transcript.csv")
            
            transcript_df = pd.read_csv(transcript_path)

            T = len(transcript_df)
            if T < 2: # Non si può creare un sotto-dialogo significativo
                continue

            M = M_pos if label == 1 else M_neg
            
            for _ in range(M):
                # Sample e (lunghezza del sub-dialogo come frazione)
                e_len_fraction = np.random.uniform(el, eh)
                
                # Calcola il numero di enunciati nel sub-dialogo
                length = round(e_len_fraction * T)
                # Assicura che la lunghezza sia in un range valido [1, T]
                length = max(1, min(length, T)) 
                d = length - 1
                upper_bound = T - d
                if upper_bound <= 0:
                    s = 0
                else:
                    s = np.random.randint(0, T - d)
                e_idx = s + d
                
                # Estrai start_time e end_time dal dataframe della trascrizione
                start_time = transcript_df.iloc[s]['Start_Time']
                end_time = transcript_df.iloc[e_idx]['End_Time']
                
                new_paths.append(path)
                new_labels.append(label)
                new_time_windows.append((start_time, end_time))

        print(f"Augmentation complete. Original samples: {len(original_paths)}, New samples: {len(new_paths)}")
        return new_paths, new_labels, new_time_windows
    
    def __get_generators(self):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits)

        if self.config.use_subdialogue_shuffling:
            aug_train_paths, aug_train_labels, aug_train_time_windows = self._apply_subdialogue_shuffling(train_paths, train_labels)
            
            train_dataset = Dataset(
                audio_paths = aug_train_paths,
                labels = aug_train_labels,
                config = self.config,
                time_windows = aug_train_time_windows # Passa le finestre temporali
            )
        else:
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
        _print_segment_stats(train_dataset, "train")
        _print_segment_stats(test_dataset, "test")
        _print_segment_stats(dev_dataset, "dev")

        return train_dataset, test_dataset, dev_dataset

    def load_data(self):
        train_dataset, test_dataset, dev_dataset = self.__get_generators()

        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=collate_fn,
            pin_memory=True
        )

        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count(),
            collate_fn=collate_fn,
            pin_memory=True
        )

        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count(),
            collate_fn=collate_fn,
            pin_memory=True
        )

        return train_loader, test_loader, dev_loader