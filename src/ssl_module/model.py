import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from typing import List

from .config import SSLConfig

class SSLModel(nn.Module):
    """
    Un modello end-to-end che classifica un'intera intervista.
    Contiene un estrattore di feature (Wav2Vec2) e un classificatore di sequenza (Transformer).
    """
    def __init__(self, config: SSLConfig):
        super(SSLModel, self).__init__()
        self.config = config
        self.model_name = config.model_name
        self.transformer_d_model = config.transformer_d_model  
        self.transformer_nhead = config.transformer_nhead
        self.transformer_num_layers = config.transformer_num_layers

        self.wav2vec = Wav2Vec2Model.from_pretrained(self.model_name)

        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
        wav2vec_output_dim = self.wav2vec.config.hidden_size # Di solito è 768 per il modello base

        # Dobbiamo proiettare l'output di Wav2Vec2 (768) alla dimensione del Transformer (128).
        self.input_projection = nn.Linear(wav2vec_output_dim, self.transformer_d_model)

        # --- FASE 2: Classificatore di Sequenza (Intervista) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_d_model,
            nhead=self.transformer_nhead,
            dim_feedforward=self.transformer_d_model * 4, # Scelta comune
            dropout=0.1,
            activation='relu',
            batch_first=True # (Batch, Seq, Feature)
        )
        self.sequence_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_num_layers)

        # Aggiungiamo un token [CLS] "imparabile" all'inizio di ogni sequenza di intervista.
        # La sua rappresentazione finale verrà usata per la classificazione.
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.transformer_d_model))

        # Testa di classificazione finale
        self.classifier_head = nn.Linear(self.transformer_d_model, 1) # Output singolo per classificazione binaria

    def forward(self, interviews_batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Prende in input una lista di interviste e restituisce i logits.
        interviews_batch: una lista di tensori. Ogni tensore ha shape 
                          (num_utterances, audio_samples) e rappresenta un'intervista.
        """
        
        # --- Logica per gestire il batch di interviste (la parte più complessa) ---
        
        # 1. Raccogliamo tutte le utterance da tutte le interviste in un unico mega-batch.
        #    Questo ci permette di processarle tutte in una volta con Wav2Vec2 (molto efficiente).
        utterance_counts = [interview.shape[0] for interview in interviews_batch]
        all_utterances = torch.cat(interviews_batch, dim=0) # Shape: (total_num_utterances, audio_samples)

        # --- FASE 1: Estrazione Feature per ogni utterance ---
        
        # Passiamo tutte le utterance attraverso Wav2Vec2
        # `last_hidden_state` ha shape: (total_num_utterances, num_frames, 768)
        hidden_states = self.wav2vec(all_utterances).last_hidden_state
        
        # Facciamo il pooling sul tempo (dim=1) per ottenere un vettore per utterance
        # Shape: (total_num_utterances, 768)
        utterance_embeddings_768 = torch.mean(hidden_states, dim=1)
        
        # Proiettiamo da 768 a 128 dimensioni
        # Shape: (total_num_utterances, 128)
        utterance_embeddings_128 = self.input_projection(utterance_embeddings_768)

        # 2. Ricostruiamo le sequenze di embedding per ogni intervista
        # `torch.split` è l'inverso di `torch.cat`.
        # `interview_sequences` è ora una tupla di tensori, dove ogni tensore
        # ha shape (num_utterances_in_this_interview, 128)
        interview_sequences = torch.split(utterance_embeddings_128, utterance_counts)

        # --- FASE 2: Classificazione delle sequenze ---

        # 3. Aggiungiamo il token [CLS] all'inizio di ogni sequenza
        batch_size = len(interviews_batch)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Espandi per il batch
        
        # Uniamo il token [CLS] con le sequenze di utterance
        interview_sequences_with_cls = [
            torch.cat([cls_tokens[i], seq], dim=0) for i, seq in enumerate(interview_sequences)
        ]

        # 4. Padding delle sequenze alla stessa lunghezza per il Transformer
        # `pad_sequence` prende una lista di tensori e li "impila" in un unico tensore,
        # aggiungendo padding dove necessario.
        padded_sequences = nn.utils.rnn.pad_sequence(interview_sequences_with_cls, batch_first=True, padding_value=0.0)
        # `padded_sequences` ha shape: (batch_size, max_interview_len, 128)
        
        # Creiamo una maschera per dire al Transformer di ignorare il padding
        # `src_key_padding_mask` ha shape (batch_size, max_interview_len)
        padding_mask = (padded_sequences.sum(dim=-1) == 0) # Semplice modo per trovare il padding

        # 5. Passiamo le sequenze attraverso il Transformer
        transformer_output = self.sequence_transformer(
            src=padded_sequences,
            src_key_padding_mask=padding_mask
        )
        # `transformer_output` ha shape: (batch_size, max_interview_len, 128)

        # 6. Prendiamo l'output corrispondente al solo token [CLS] (è in posizione 0)
        cls_output = transformer_output[:, 0, :] # Shape: (batch_size, 128)
        
        # 7. Classificazione finale
        logits = self.classifier_head(cls_output) # Shape: (batch_size, 1)
        
        return logits