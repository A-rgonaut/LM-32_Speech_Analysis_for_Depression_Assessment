import torch
from torch import nn
from transformers import AutoModel

from ..pooling_layers import AttentionPoolingLayer, MeanPoolingLayer, GatedAttentionPoolingLayer

class SSLModel2(nn.Module):
    def __init__(self, config):
        super(SSLModel2, self).__init__()

        # SSL model loading & config
        self.aggregate_layers = config.aggregate_layers
        self.ssl_model = AutoModel.from_pretrained(config.ssl_model_name, output_hidden_states=self.aggregate_layers)
        self.ssl_hidden_size = self.ssl_model.config.hidden_size
        self.dropout = config.dropout_rate

        self.num_encoder_layers_to_use = config.layer_to_use
        
        if self.num_encoder_layers_to_use is not None:
            self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:self.num_encoder_layers_to_use]
            self.ssl_model.config.num_hidden_layers = len(self.ssl_model.encoder.layers)
            print(self.ssl_model.config.num_hidden_layers, "encoder layers will be used from the SSL model.")

        # Freeze SSL weights
        self.num_layers_to_unfreeze = config.num_layers_to_unfreeze
        if self.num_layers_to_unfreeze != -1:
            for param in self.ssl_model.parameters():
                param.requires_grad = False

            if self.num_layers_to_unfreeze > 0:
                encoder_layers = self.ssl_model.encoder.layers
                for layer in encoder_layers[-self.num_layers_to_unfreeze:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Unfrozen last {self.num_layers_to_unfreeze} encoder layers.")
        else:
            print("All encoder layers are trainable (no freezing).")

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

        # Segment-level pooling
        self.segment_embedding_dim = self.ssl_hidden_size 

        self.segment_embeddings_pooling = GatedAttentionPoolingLayer(
            embed_dim=self.ssl_hidden_size,
            attn_dim=self.ssl_hidden_size // 2,
            return_weights=False
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.segment_embedding_dim, config.classifier_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(config.classifier_size, 1)
        )

        self.init_weights()
    
    def init_weights(self):
        for name, param in self.classifier.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, batch):
        input_values = batch['input_values'] # (bs, seq_len)
        attention_mask_segment = batch['attention_mask_segment'] # (bs, seq_len)

        if self.num_layers_to_unfreeze > 0:
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
            ssl_hidden_state = torch.zeros_like(ssl_hidden_states[-1])  # (bs, num_frames, hidden_size)
            weights = self.softmax(self.layer_weights)
            for i in range(len(ssl_hidden_states)):
                ssl_hidden_state += weights[i] * self.layer_norms[i](ssl_hidden_states[i])
        else:
            ssl_hidden_state = outputs.last_hidden_state  # (bs, num_frames, hidden_size)

        frame_mask = self.ssl_model._get_feature_vector_attention_mask(
            ssl_hidden_state[0].shape[1] if self.aggregate_layers else ssl_hidden_state.shape[1], 
            attention_mask_segment
        ) # (bs, num_frames)

        pooling_mask = (frame_mask == 0)
        # Pool the sequence of frames into a single representation for the whole segment.
        segment_embeddings = self.segment_embeddings_pooling(ssl_hidden_state, mask=pooling_mask)  # (bs, segment_embedding_dim)

        logits = self.classifier(segment_embeddings)  # (bs, 1)

        return logits.squeeze(-1)  # (bs,)