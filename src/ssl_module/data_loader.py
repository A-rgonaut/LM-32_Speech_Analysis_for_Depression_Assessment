import os
import torch
import pandas as pd
import librosa
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from transformers import AutoFeatureExtractor

from .config import SSLConfig
from .sampler import BalancedParticipantSampler
from ..utils import get_splits, filter_edaic_samples
from ..audio_utils import load_audio, segment_audio_by_transcript, segment_audio_sliding_window
from ..preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, config : SSLConfig, participant_info: dict, is_train: bool = False):
        self.config = config
        self.participant_info = participant_info
        self.is_train = is_train

        if not config.use_preextracted_features:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name, do_normalize=False)

        # Crea una lista di "item" che il dataset può restituire.
        # Ogni item è un chunk.
        self.indexable_items = []
        if self.is_train:
            # In training, il sampler lavora con gli ID, quindi la lista è solo di ID.
            self.indexable_items = list(self.participant_info.keys())
        else:
            # In eval, creiamo una lista di tutti i chunk possibili in modo deterministico.
            for pid, info in self.participant_info.items():
                total_segments = info['total_segments']
                chunk_size = self.config.chunk_segments
                
                # Usiamo uno step uguale a chunk_size per chunk non sovrapposti
                for start_idx in range(0, total_segments, chunk_size):
                    self.indexable_items.append({'participant_id': pid, 'start_index': start_idx})

    def __len__(self):
        return len(self.indexable_items)

    def __getitem__(self, idx):
        if self.is_train:
            # --- TRAINING ---
            # 'idx' qui è un dizionario dal sampler {'id':..., 'start':...}
            participant_id = idx['participant_id']
            start_index = idx['start_index']
        else:
            # --- EVALUATION ---
            # 'idx' è un intero. Prendiamo l'istruzione dalla nostra lista di chunk
            item_info = self.indexable_items[idx]
            participant_id = item_info['participant_id']
            start_index = item_info['start_index']

        participant_data = self.participant_info[participant_id]
        label = torch.tensor(participant_data['label'], dtype=torch.float32)
        audio_id = torch.tensor(int(participant_id), dtype=torch.long)
        chunk_size = self.config.chunk_segments
        end_index = start_index + chunk_size

        if self.config.use_preextracted_features:
            feature_filename = f"{participant_id}_AUDIO_layer_{self.config.layer_to_use}.pt"
            feature_path = os.path.join(self.config.feature_path, feature_filename)
            all_features = torch.load(feature_path, map_location='cpu', weights_only=True)
            feature_chunk = all_features[start_index:end_index]

            # Padding se l'ultimo chunk è più corto
            actual_len = feature_chunk.shape[0]
            if actual_len < chunk_size:
                padding_size = chunk_size - actual_len
                padding = torch.zeros((padding_size, feature_chunk.shape[1]), dtype=feature_chunk.dtype)
                feature_chunk = torch.cat([feature_chunk, padding], dim=0)

            return {
                'segment_features': feature_chunk, # (chunk_size, feature_dim)
                'label': label,
                'audio_id': audio_id
            }
        else:
            audio_path = participant_data['path']
            audio = load_audio(audio_path, self.sample_rate)

            segments = []
            if self.config.segmentation_strategy == 'transcript':
                transcript_path = audio_path.replace("_AUDIO.wav", "_Transcript.csv")
                transcript_df = pd.read_csv(transcript_path)
                segments = segment_audio_by_transcript(audio, transcript_df, self.config.sample_rate, self.config.max_utt_seconds, 
                self.config.min_utt_seconds, self.config.overlap_seconds)
            else: # 'fixed_length'
                segments = segment_audio_sliding_window(audio, self.config.sample_rate, self.config.max_utt_seconds,
                self.config.min_utt_seconds, self.config.overlap_seconds)

            all_input_values = []
            all_attention_masks = []
            segment_samples = int(self.config.max_utt_seconds * self.config.sample_rate)
            
            for segment in segments:
                features = self.feature_extractor(
                    segment, 
                    sampling_rate=self.config.sample_rate,
                    max_length=segment_samples,
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

            chunk_input_values = stacked_input_values[start_index:end_index]
            chunk_attention_masks = stacked_attention_masks[start_index:end_index]

            actual_len = chunk_input_values.shape[0]
            chunk_size = self.config.chunk_segments
            if actual_len < chunk_size:
                padding_size = chunk_size - actual_len
                
                # Padding per input_values
                padding_values = torch.zeros((padding_size, chunk_input_values.shape[1]), dtype=chunk_input_values.dtype)
                chunk_input_values = torch.cat([chunk_input_values, padding_values], dim=0)
                
                # Padding per le maschere
                padding_masks = torch.zeros((padding_size, chunk_attention_masks.shape[1]), dtype=chunk_attention_masks.dtype)
                chunk_attention_masks = torch.cat([chunk_attention_masks, padding_masks], dim=0)

            return {
                'input_values': chunk_input_values,  # (chunk_size, seq_len)
                'attention_mask_segment': chunk_attention_masks,  # (chunk_size,)
                'label': label,
                'audio_id': audio_id,
            }

def collate_fn(batch):
    """
    Collate function per la nuova logica a chunk.
    Il batch è una lista di dizionari, ognuno contenente un chunk.
    """
    labels = torch.stack([item['label'] for item in batch])
    audio_ids = torch.stack([item['audio_id'] for item in batch])
    if 'segment_features' in batch[0]:
        features = torch.stack([item['segment_features'] for item in batch])
        
        return {
            'segment_features': features,
            'label': labels,
            'audio_id': audio_ids
        }
    else:
        input_values = torch.stack([item['input_values'] for item in batch])
        attention_masks = torch.stack([item['attention_mask_segment'] for item in batch])
        
        return {
            'input_values': input_values,
            'attention_mask_segment': attention_masks,
            'label': labels,
            'audio_id': audio_ids
        }

class DataLoader():
    def __init__(self, config : SSLConfig = SSLConfig()):
        self.config = config
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits_df = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits_df = filter_edaic_samples(self.splits_df)
        self._setup_participant_info()

    def _get_num_segments(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=self.config.sample_rate)
        if self.config.segmentation_strategy == 'transcript':
            transcript_path = audio_path.replace('_AUDIO.wav', '_Transcript.csv')
            transcript_df = pd.read_csv(transcript_path)
            indices = segment_audio_by_transcript(audio, transcript_df, self.config.sample_rate, self.config.max_utt_seconds, 
                self.config.min_utt_seconds, self.config.overlap_seconds, return_indices=True)
        else: # 'fixed_length'
            indices = segment_audio_sliding_window(audio, self.config.sample_rate, self.config.max_utt_seconds,
            self.config.min_utt_seconds, self.config.overlap_seconds, return_indices=True)

        return len(indices)

    def _setup_participant_info(self):
        self.participant_info = {}
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = get_splits(self.splits_df)
        all_paths = train_paths + dev_paths + test_paths
        all_labels = train_labels + dev_labels + test_labels
        path_to_label = {path: label for path, label in zip(all_paths, all_labels)}

        print("Configuring participant metadata...")
        _metadata_df_map = {}
        if self.config.use_preextracted_features:
            for split in ['train', 'dev', 'test']:
                metadata_csv_path = os.path.join(self.config.feature_path, f'{split}_metadata.csv')
                if os.path.exists(metadata_csv_path):
                    df = pd.read_csv(metadata_csv_path)
                    for _, row in df.iterrows():
                        _metadata_df_map[row['filename']] = row['num_segments']
        
        for path in all_paths:
            participant_id = int(os.path.basename(path).replace('_AUDIO.wav', ''))
            num_segments = 0
            if self.config.use_preextracted_features:
                feature_filename = os.path.basename(path).replace('.wav', '.pt')
                num_segments = _metadata_df_map.get(feature_filename, 0)
            else:
                num_segments = self._get_num_segments(path)

            self.participant_info[participant_id] = {
                'path': path, 'label': path_to_label.get(path, -1), 'total_segments': num_segments
            }

    def get_dataset(self, split: str):
        train_df, test_df, dev_df = self.splits_df

        if split == 'train':
            target_df = train_df
        elif split == 'dev':
            target_df = dev_df
        else:
            target_df = test_df

        participant_ids_for_split = [row["Participant_ID"] for _, row in target_df.iterrows()]

        split_participant_info = {
            pid: self.participant_info[pid] 
            for pid in participant_ids_for_split
            if pid in self.participant_info
        }

        return Dataset(
            config=self.config, 
            participant_info=split_participant_info, 
            is_train=(split == 'train')
        )

    def get_data_loader(self, split: str, dataset: TorchDataset = None):
        if dataset is None:
            dataset = self.get_dataset(split)

        is_train = split == 'train'
        
        if is_train and self.config.use_random_chunking:
            sampler = BalancedParticipantSampler(dataset, self.config.batch_size, self.config.chunk_segments)
            return TorchDataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=os.cpu_count(),
                collate_fn=collate_fn,
                pin_memory=True
            )
        else:
            return TorchDataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=os.cpu_count(),
                collate_fn=collate_fn,
                pin_memory=True
            )