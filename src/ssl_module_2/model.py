import torch
from torch import nn
from transformers import AutoModel

from .config import SSLConfig
from ..pooling_layers import AttentionPoolingLayer, MeanPoolingLayer

class SSLModel(nn.Module):
    def __init__(self, config: SSLConfig):
        super(SSLModel, self).__init__()

        # SSL model loading & config
        self.num_layers_to_unfreeze = config.num_layers_to_unfreeze
        self.aggregate_layers = config.aggregate_layers
        self.ssl_model = AutoModel.from_pretrained(config.model_name, output_hidden_states=self.aggregate_layers)
        self.ssl_hidden_size = self.ssl_model.config.hidden_size

        self.num_encoder_layers_to_use = config.layer_to_use
        
        if self.num_encoder_layers_to_use is not None:
            self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:self.num_encoder_layers_to_use]
            self.ssl_model.config.num_hidden_layers = len(self.ssl_model.encoder.layers)

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
        #self.segment_embeddings_pooling = MeanPoolingLayer()

        # Segment-level pooling
        self.segment_embedding_dim = self.ssl_hidden_size 

        self.classifier = nn.Sequential(
            nn.Linear(self.segment_embedding_dim, 1)
        )

        self.init_weights()
    
    def init_weights(self):
        for name, param in self.classifier.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for name, param in self.audio_embedding_pooling.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
    
    def forward(self, batch):
        input_values = batch['input_values'] # (bs, num_segments, seq_len)
        attention_mask_segment = batch['attention_mask_segment'] # (bs, num_segments, seq_len)
        batch_size, num_segments = input_values.shape[:2]

        attention_mask_segment = attention_mask_segment.view(batch_size * num_segments, -1) # (bs * num_segments, seq_len)

        # Flatten for SSL model: (bs * num_segments, seq_len)
        input_values = input_values.view(batch_size * num_segments, -1)

        if self.num_layers_to_unfreeze == 0:
            with torch.no_grad():
                outputs = self.ssl_model(
                    input_values=input_values,
                    attention_mask=attention_mask_segment,
                    return_dict=True,
                )
        else:
            outputs = self.ssl_model(
                input_values=input_values,
                attention_mask=attention_mask_segment,
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
            ssl_hidden_state[0].shape[1] if self.aggregate_layers else ssl_hidden_state.shape[1], 
            attention_mask_segment
        ) # (bs * num_segments, num_frames)

        pooling_mask = (frame_mask == 0)
        # Pool the sequence of frames into a single representation for the whole segment.
        segment_embeddings = self.segment_embeddings_pooling(ssl_hidden_state, mask=pooling_mask)  # (bs * num_segments, segment_embedding_dim)

        # Un-flatten the batch to restore sequence structure 
        # Reshape from (bs * num_segments, segment_embedding_dim) back to (bs, num_segments, segment_embedding_dim)
        segment_embeddings = segment_embeddings.view(batch_size, num_segments, self.segment_embedding_dim)

        logits = self.classifier(segment_embeddings)  # (bs, 1)

        return logits.squeeze(-1)  # (bs,)