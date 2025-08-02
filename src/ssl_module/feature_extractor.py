
import os
import torch
import librosa
from transformers import AutoModel, AutoFeatureExtractor
from tqdm.auto import tqdm
import pandas as pd

class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.max_segments = config.max_segments
        self.segment_samples = int(config.max_utt_seconds * self.sample_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        print(f"Loading model: {config.model_name}")
        self.model = AutoModel.from_pretrained(config.model_name, output_hidden_states=True).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
        self.model.eval() # Modello solo in inferenza

    def _segment_audio_by_transcript(self, audio, transcript_df):
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

    @torch.no_grad()
    def extract_and_save(self, audio_paths, split_name):
        """
        Estrae le feature da una lista di file audio e le salva su disco.
        """
        output_dir = os.path.join(self.config.feature_path, split_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving features for '{split_name}' split in: {output_dir}")

        for audio_path in tqdm(audio_paths, desc=f"Extracting features for {split_name}"):
            try:
                # Costruisci il percorso di output
                filename = os.path.basename(audio_path).replace('.wav', '.pt')
                save_path = os.path.join(output_dir, filename)

                # Salta se il file esiste già
                if os.path.exists(save_path):
                    continue

                # Carica e segmenta l'audio
                audio, _ = librosa.load(audio_path, sr=self.sample_rate)
                transcript_path = audio_path.replace('_AUDIO.wav', '_TRANSCRIPT.csv')
                transcript_df = pd.read_csv(transcript_path)
                segments = self._segment_audio_by_transcript(audio, transcript_df)

                # Prepara i segmenti per il modello
                inputs = self.feature_extractor(
                    segments, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt", 
                    padding=True
                )
                
                # Sposta i dati sul dispositivo corretto
                input_values = inputs.input_values.to(self.device)
                
                # Estrai gli hidden states
                outputs = self.model(input_values)
                
                # `outputs.hidden_states` è una tupla di tensori.
                # Ogni tensore ha shape (num_segments, num_frames, hidden_size)
                # Li spostiamo su CPU prima di salvarli
                hidden_states_cpu = tuple(h.cpu() for h in outputs.hidden_states)
                
                # Salva la tupla di tensori
                torch.save(hidden_states_cpu, save_path)

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")