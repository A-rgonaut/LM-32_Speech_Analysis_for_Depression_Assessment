import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from ..pooling_layers import AttentionPoolingLayer, MeanPoolingLayer, GatedAttentionPoolingLayer

class SSLModel(nn.Module):
    def __init__(self, config):
        super(SSLModel, self).__init__()

        # SSL model loading & config
        self.use_preextracted_features = config.use_preextracted_features
        if not self.use_preextracted_features:
            self.ssl_model = AutoModel.from_pretrained(config.ssl_model_name, output_hidden_states=False)
            self.ssl_hidden_size = self.ssl_model.config.hidden_size

            self.num_encoder_layers_to_use = config.layer_to_use
            
            if self.num_encoder_layers_to_use is not None:
                self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:self.num_encoder_layers_to_use]
                self.ssl_model.config.num_hidden_layers = len(self.ssl_model.encoder.layers)
                print(self.ssl_model.config.num_hidden_layers, "encoder layers will be used from the SSL model.")

            # Freeze SSL weights 
            for param in self.ssl_model.parameters():
                param.requires_grad = False
                
            self.segment_embeddings_pooling = AttentionPoolingLayer(embed_dim=self.ssl_hidden_size)
            #self.segment_embeddings_pooling = MeanPoolingLayer()

        ssl_config = AutoConfig.from_pretrained(config.ssl_model_name)
        self.ssl_hidden_size = ssl_config.hidden_size

        # Segment-level pooling
        self.segment_embedding_dim = self.ssl_hidden_size 

        # Add a projection layer to reduce dimensionality
        self.projection = nn.Linear(self.segment_embedding_dim, config.seq_hidden_size)

        self.seq_model_type = config.seq_model_type
        self.seq_hidden_size = config.seq_hidden_size
        self.dropout = config.dropout_rate
        self.num_layers = config.seq_num_layers

        if self.seq_model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.seq_hidden_size,
                nhead=config.transformer_nhead,  # nhead must be a divisor of d_model (e.g. 768 % 4 == 0)
                dim_feedforward=self.seq_hidden_size * 2,
                dropout=self.dropout,
                activation='relu',
                batch_first=True 
            )
            self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.seq_output_dim = self.seq_hidden_size
        elif self.seq_model_type == 'lstm':
            self.sequence_model = nn.LSTM(
                input_size=self.seq_hidden_size,
                hidden_size=self.seq_hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True
            )
            self.seq_output_dim = self.seq_hidden_size * 2

        self.audio_embedding_pooling = GatedAttentionPoolingLayer(
            embed_dim=self.seq_output_dim,
            attn_dim=self.seq_output_dim // 2,
            return_weights=False
        )

        #self.audio_embedding_pooling = MeanPoolingLayer()

        self.audio_embedding_dim = self.seq_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.audio_embedding_dim, 1)
        )

        self.init_weights()
    
    def init_weights(self):
        for name, param in self.classifier.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, batch):
        frame_mask = None
        chunk_padding_mask = batch.get('chunk_padding_mask', None) # (bs, num_segments)

        if self.use_preextracted_features:
            segment_embeddings = batch['segment_features'] # (bs, num_segments, hidden_size)
        else:
            input_values = batch['input_values'] # (bs, num_segments, seq_len)
            attention_mask_segment = batch['attention_mask_segment'] # (bs, num_segments, seq_len)
            batch_size, num_segments = input_values.shape[:2]

            attention_mask_segment = attention_mask_segment.view(batch_size * num_segments, -1) # (bs * num_segments, seq_len)

            # Flatten for SSL model: (bs * num_segments, seq_len)
            input_values = input_values.view(batch_size * num_segments, -1)

            with torch.no_grad():
                ssl_hidden_state = self.ssl_model(
                    input_values=input_values,
                    attention_mask=attention_mask_segment,
                    return_dict=True,
                ).last_hidden_state  # (bs * num_segments, num_frames, hidden_size)

            frame_mask = self.ssl_model._get_feature_vector_attention_mask(ssl_hidden_state.shape[1], attention_mask_segment) # (bs * num_segments, num_frames)

            pooling_mask = (frame_mask == 0)
            # Pool the sequence of frames into a single representation for the whole segment.
            segment_embeddings = self.segment_embeddings_pooling(ssl_hidden_state, mask=pooling_mask)  # (bs * num_segments, segment_embedding_dim)

            # Un-flatten the batch to restore sequence structure 
            # Reshape from (bs * num_segments, segment_embedding_dim) back to (bs, num_segments, segment_embedding_dim)
            segment_embeddings = segment_embeddings.view(batch_size, num_segments, self.segment_embedding_dim)

        # Project embeddings to the desired dimension for the sequence model
        projected_embeddings = self.projection(segment_embeddings) # (bs, num_segments, seq_hidden_size)

        # Sequence modeling across segments
        # Process the sequence of segment embeddings for each audio file.
        if self.seq_model_type == 'transformer':
            sequence_output = self.sequence_model(projected_embeddings, src_key_padding_mask=chunk_padding_mask)  # (bs, T, H)
        elif self.seq_model_type == 'lstm':
            lengths = (~chunk_padding_mask).sum(dim=1).clamp(min=1)  # (bs,)
            packed = nn.utils.rnn.pack_padded_sequence(
                projected_embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.sequence_model(packed)
            sequence_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=projected_embeddings.size(1)
            )  # (bs, T, H or H*2)


        # Pool segment-level outputs to get a single audio-level representation
        audio_embeddings = self.audio_embedding_pooling(sequence_output, mask=chunk_padding_mask)  # (bs, audio_embedding_dim)

        logits = self.classifier(audio_embeddings)  # (bs, 1)

        return logits.squeeze(-1)  # (bs,)