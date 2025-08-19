import os
import torch
from transformers import AutoModel, AutoFeatureExtractor
from tqdm.auto import tqdm
import pandas as pd

from ..audio_utils import load_audio, segment_audio_by_transcript, segment_audio_sliding_window
from ..pooling_layers import MeanPoolingLayer

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.segment_samples = int(config.max_utt_seconds * self.sample_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading model: {config.ssl_model_name}")
        self.model = AutoModel.from_pretrained(config.ssl_model_name, output_hidden_states=True).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.ssl_model_name, do_normalize=True)
        self.mean_pooling = MeanPoolingLayer().to(self.device)
        self.model.eval() 

    @torch.no_grad()
    def extract_and_save(self, audio_paths, split_name):
        output_dir = self.config.feature_path
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving features for '{split_name}' split in: {output_dir}")
        metadata = []

        for audio_path in tqdm(audio_paths, desc=f"Extracting features for {split_name}"):
            # Build the output path
            filename = os.path.basename(audio_path).replace('.wav', '.pt')
            save_path = os.path.join(output_dir, filename)

            # Load and segment the audio
            audio = load_audio(audio_path, self.sample_rate)
            segments = [] 
            if self.config.segmentation_strategy == 'transcript':
                transcript_path = audio_path.replace('_AUDIO.wav', '_Transcript.csv')
                transcript_df = pd.read_csv(transcript_path)
                segments = segment_audio_by_transcript(audio, transcript_df, self.sample_rate, self.config.max_utt_seconds, 
                    self.config.min_utt_seconds, self.config.overlap_seconds)
            else: # 'fixed_length'
                segments = segment_audio_sliding_window(audio, self.sample_rate, self.config.max_utt_seconds,
                    self.config.min_utt_seconds, self.config.overlap_seconds)

            num_segments = len(segments)
            metadata.append({'filename': filename, 'num_segments': num_segments})

            # Prepare segments for the model
            inputs = self.feature_extractor(
                segments, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt", 
                max_length=self.segment_samples,
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
        
            # Move data to the correct device
            input_values = inputs.input_values.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Extract hidden states
            hidden_states = self.model(input_values, attention_mask=attention_mask).hidden_states
            frame_attention_mask = self.model._get_feature_vector_attention_mask(hidden_states[0].shape[1], attention_mask)

            # Esegui il pooling per ogni layer
            for i, hidden_state_layer in enumerate(hidden_states): 
                # hidden_state_layer has shape (num_segments, num_frames, hidden_size)
                pooling_mask = (frame_attention_mask == 0)
                pooled_output = self.mean_pooling(hidden_state_layer, mask=pooling_mask) # (num_segments, hidden_size)
                torch.save(pooled_output.cpu(), save_path.replace('.pt', f'_layer_{i}.pt'))
        
        metadata_path = os.path.join(self.config.feature_path, f'{split_name}_metadata.csv')
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")