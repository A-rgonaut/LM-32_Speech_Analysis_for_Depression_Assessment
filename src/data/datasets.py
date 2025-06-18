import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
import os

class AudioDepressionDataset(Dataset):
    def __init__(self, audio_paths, labels, return_filename=False, sample_rate=16_000, segment_ms=250, hop_ms=50):
        self.audio_paths = audio_paths
        self.labels = labels
        self.return_filename = return_filename
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * (segment_ms / 1000.0))
        self.hop_samples = int(sample_rate * (hop_ms / 1000.0))
        self.segments = []

        for audio_path in self.audio_paths:
            info = torchaudio.info(audio_path)
            num_frames = info.num_frames

            start = 0
            while start + self.segment_samples <= num_frames:
                self.segments.append({
                    "path": audio_path,
                    "filename": os.path.basename(audio_path), 
                    "start_sample": start,
                    "label": self.labels[self.audio_paths.index(audio_path)]
                })
                start += self.hop_samples

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment_info = self.segments[idx]
        file_path = segment_info["path"]
        start_sample = segment_info["start_sample"]
        label = segment_info["label"]

        waveform_segment, _ = torchaudio.load(
            file_path,
            frame_offset=start_sample,
            num_frames=self.segment_samples
        )
        
        if self.return_filename:
            return {
                'input_values': waveform_segment, 
                'label': torch.tensor([label], dtype=torch.float32),
                'filename': segment_info["filename"]
            }
        else:
            return {
                'input_values': waveform_segment, 
                'label': torch.tensor([label], dtype=torch.float32)
            }

class AudioDepressionDatasetSSL(Dataset):
    def __init__(self, audio_paths, labels, model_name, sample_rate=16_000, segment_length_seconds=20, max_segments=None):
        self.audio_paths = audio_paths  
        self.labels = labels            
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, do_normalize=False)
        self.sample_rate = sample_rate
        self.segment_length_samples = segment_length_seconds * sample_rate
        self.max_segments = max_segments

    def __len__(self):
        return len(self.audio_paths)
    
    def _load_audio(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)
        audio = audio / np.max(np.abs(audio))
        return audio
    
    def _segment_audio(self, audio):
        """Segmenta l'audio in chunks di lunghezza fissa"""
        segments = []
        
        # Se l'audio è più corto del segmento desiderato, pad con zeri
        if len(audio) < self.segment_length_samples:
            padded_audio = np.zeros(self.segment_length_samples)
            padded_audio[:len(audio)] = audio
            segments.append(padded_audio)
        else:
            # Dividi in segmenti
            for i in range(0, len(audio), self.segment_length_samples):
                segment = audio[i:i + self.segment_length_samples]
                
                # Se l'ultimo segmento è troppo corto, pad con zeri
                if len(segment) < self.segment_length_samples:
                    padded_segment = np.zeros(self.segment_length_samples)
                    padded_segment[:len(segment)] = segment
                    segment = padded_segment
                
                segments.append(segment)
                
                # Limita il numero di segmenti se specificato
                if self.max_segments and len(segments) >= self.max_segments:
                    break
        
        return np.array(segments)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        audio = self._load_audio(audio_path)
        segments = self._segment_audio(audio)
        
        segment_features = []
        for segment in segments:
            features = self.feature_extractor(
                segment, 
                sampling_rate=self.sample_rate,
                max_length=self.segment_length_samples,
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