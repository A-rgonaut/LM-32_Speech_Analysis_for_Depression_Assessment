import torch
from torch import nn
from transformers import AutoModel, AutoConfig

from .config import SSLConfig
from ..pooling_layers import AttentionPoolingLayer, MeanPoolingLayer

class SSLModel(nn.Module):
    def __init__(self, config: SSLConfig):
        super(SSLModel, self).__init__()

        # SSL model loading & config
        self.use_preextracted_features = config.use_preextracted_features
        if not self.use_preextracted_features:
            self.num_layers_to_unfreeze = config.num_layers_to_unfreeze
            self.aggregate_layers = config.aggregate_layers
            self.ssl_model = AutoModel.from_pretrained(config.model_name, output_hidden_states=self.aggregate_layers)
            self.ssl_hidden_size = self.ssl_model.config.hidden_size

            """
            self.num_encoder_layers_to_use = 8
            
            if self.num_encoder_layers_to_use is not None:
                self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:self.num_encoder_layers_to_use]
                self.ssl_model.config.num_hidden_layers = len(self.ssl_model.encoder.layers)
            """
            # Freeze SSL weights 
            for param in self.ssl_model.parameters():
                param.requires_grad = False

            # Unfreeze last N encoder layers
            if self.num_layers_to_unfreeze > 0:
                encoder_layers = self.ssl_model.encoder.layers
                num_encoder_layers = len(encoder_layers)

                layers_to_unfreeze = min(self.num_layers_to_unfreeze, num_encoder_layers)
                
                print(f"Unfreezing the last {layers_to_unfreeze} transformer layers.")
                
                for layer in encoder_layers[-layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            
            if self.aggregate_layers:
                # Weighted sum of SSL model's hidden layers
                num_ssl_layers = self.ssl_model.config.num_hidden_layers
                layers_to_aggregate = num_ssl_layers + 1 # +1 for the initial embeddings

                self.layer_weights = nn.Parameter(torch.ones(layers_to_aggregate))
                self.layer_weights.requires_grad = True
                self.layer_norms = nn.ModuleList(
                    [nn.LayerNorm(self.ssl_hidden_size) for _ in range(layers_to_aggregate)]
                )
                self.layer_norms.requires_grad = True
                self.softmax = nn.Softmax(dim=-1)
                
            self.segment_embeddings_pooling = AttentionPoolingLayer(embed_dim=self.ssl_hidden_size)

        ssl_config = AutoConfig.from_pretrained(config.model_name)
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
                dim_feedforward=self.seq_hidden_size * 2, # Common practice
                dropout=self.dropout,
                activation='relu',
                batch_first=True 
            )
            self.sequence_model = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.seq_output_dim = self.seq_hidden_size
            #self.audio_embedding_pooling = AttentionPoolingLayer(embed_dim=self.seq_output_dim)
            self.audio_embedding_pooling = MeanPoolingLayer()

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

        if self.use_preextracted_features:
            segment_embeddings = batch['segment_features'] # (bs, num_segments, hidden_size)
        else:
            input_values = batch['input_values'] # (bs, num_segments, seq_len)
            attention_mask_segment = batch['attention_mask_segment'] # (bs, num_segments, seq_len)
            batch_size, num_segments = input_values.shape[:2]

            attention_mask_segment_flat = attention_mask_segment.view(batch_size * num_segments, -1) # (bs * num_segments, seq_len)

            # Flatten for SSL model: (bs * num_segments, seq_len)
            input_values_flat = input_values.view(batch_size * num_segments, -1)

            outputs = self.ssl_model(
                input_values=input_values_flat,
                attention_mask=attention_mask_segment_flat,
                return_dict=True,
            )

            if self.aggregate_layers:
                ssl_hidden_states = outputs.hidden_states
                # Combine all hidden layers from the SSL model using learned weights.
                ssl_hidden_state = torch.zeros_like(ssl_hidden_states[-1])  # (bs * num_segments, num_frames, hidden_size)
                weights = self.softmax(self.layer_weights)
                for i in range(len(ssl_hidden_states)):
                    ssl_hidden_state += weights[i] * self.layer_norms[i](ssl_hidden_states[i])
            else:
                ssl_hidden_state = outputs.last_hidden_state  # (bs * num_segments, num_frames, hidden_size)

            frame_mask = self.ssl_model._get_feature_vector_attention_mask(
                attention_mask=attention_mask_segment_flat,
                output_length=ssl_hidden_state.shape[1]
            ) # (bs * num_segments, num_frames)

            pooling_mask = (frame_mask == 0)
            # Pool the sequence of frames into a single representation for the whole segment.
            segment_embeddings_flat = self.segment_embeddings_pooling(ssl_hidden_state, mask=pooling_mask)  # (bs * num_segments, segment_embedding_dim)

            # Un-flatten the batch to restore sequence structure 
            # Reshape from (bs * num_segments, segment_embedding_dim) back to (bs, num_segments, segment_embedding_dim)
            segment_embeddings = segment_embeddings_flat.view(batch_size, num_segments, self.segment_embedding_dim)

        # Project embeddings to the desired dimension for the sequence model
        projected_embeddings = self.projection(segment_embeddings) # (bs, num_segments, seq_hidden_size)

        # Sequence modeling across segments
        # Process the sequence of segment embeddings for each audio file.
        if self.seq_model_type == 'transformer':
            sequence_output = self.sequence_model(projected_embeddings)
            # Pool segment-level outputs to get a single audio-level representation
            audio_embeddings = self.audio_embedding_pooling(sequence_output)  # (bs, audio_embedding_dim)

        logits = self.classifier(audio_embeddings)  # (bs, 1)

        return logits.squeeze(-1)  # (bs,)