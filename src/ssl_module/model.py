import torch
from torch import nn

from .config import SSLConfig

class AttentionPoolingLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)
        
    def forward(self, x, mask=None):
        """
        Forward pass.
        Args:
            x: The input tensor of shape (batch_size, seq_len, embed_dim).
            mask: The padding mask of shape (batch_size, seq_len).
        Returns:
            The output tensor of shape (batch_size, embed_dim).
        """
        weights = self.linear(x)  # (bs, seq_len, embed_dim) -> (bs, seq_len, 1)

        # Apply the mask before softmax to ignore padding
        if mask is not None:
            # .unsqueeze(-1): (bs, seq_len) -> (bs, seq_len, 1)
            fill = torch.finfo(weights.dtype).min
            # Assign a very negative value where the mask is True (padding)
            weights.masked_fill_(mask.unsqueeze(-1), fill)

        weights = torch.softmax(weights, dim=1)  # Now masked elements will have ~0 weight

        # Weighted sum (bs, seq_len, 1) * (bs, seq_len, embed_dim) -> (bs, embed_dim)
        x = torch.sum(weights * x, dim=1) 

        return x
    
class SSLModel(nn.Module):
    def __init__(self, config: SSLConfig):
        super(SSLModel, self).__init__()

        self.ssl_hidden_size = 768
        
        '''      
        # Weighted sum of SSL model's hidden layers
        num_ssl_layers = self.ssl_model.config.num_hidden_layers
        layers_to_aggregate = num_ssl_layers + 1 # +1 for the initial embeddings

        self.layer_weights = nn.Parameter(torch.ones(layers_to_aggregate))
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.ssl_hidden_size) for _ in range(layers_to_aggregate)]
        )
        self.softmax = nn.Softmax(dim=-1)
        '''

        # Segment-level pooling
        self.segment_embeddings_pooling = AttentionPoolingLayer(embed_dim=self.ssl_hidden_size)
        self.segment_embedding_dim = self.ssl_hidden_size 

        # Add a projection layer to reduce dimensionality
        self.projection = nn.Linear(self.segment_embedding_dim, config.seq_hidden_size)

        self.seq_model_type = config.seq_model_type
        self.seq_hidden_size = config.seq_hidden_size
        self.dropout = config.dropout_rate
        self.num_layers = config.seq_num_layers

        if self.seq_model_type == 'bilstm':
            self.sequence_model = nn.LSTM(
                input_size=self.seq_hidden_size,
                hidden_size=self.seq_hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=True
            )
            self.seq_output_dim = self.seq_hidden_size * 2  # bidirectional
        elif self.seq_model_type == 'transformer':
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

        self.audio_embedding_pooling = AttentionPoolingLayer(embed_dim=self.seq_output_dim)
        self.audio_embedding_dim = self.seq_output_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.audio_embedding_dim, self.ssl_hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.ssl_hidden_size, 1),
        )

        self.init_weights()
    
    def init_weights(self):
        # initialize weights of classifier
        for name, param in self.classifier.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


    def forward(self, batch):
        ssl_hidden_state = batch['features'] # (bs, num_segments, num_frames, seq_len)
        attention_mask = batch.get('attention_mask', None) # (bs, num_segments)
        batch_size, num_segments, _, _ = ssl_hidden_state.shape

        '''
        # Combine all hidden layers from the SSL model using learned weights.
        ssl_hidden_state = torch.zeros_like(ssl_hidden_states[-1])  # (bs * num_segments, num_frames, hidden_size)
        weights = self.softmax(self.layer_weights)
        for i in range(len(ssl_hidden_states)):
            ssl_hidden_state += weights[i] * self.layer_norms[i](ssl_hidden_states[i])
        '''
        ssl_hidden_state_flat = ssl_hidden_state.view(batch_size * num_segments, ssl_hidden_state.shape[2], self.ssl_hidden_size)

        # Pool the sequence of frames into a single representation for the whole segment.
        segment_embeddings_flat = self.segment_embeddings_pooling(ssl_hidden_state_flat)  # (bs * num_segments, segment_embedding_dim)

        # Un-flatten the batch to restore sequence structure 
        # Reshape from (bs * num_segments, segment_embedding_dim) back to (bs, num_segments, segment_embedding_dim)
        segment_embeddings = segment_embeddings_flat.view(batch_size, num_segments, self.segment_embedding_dim)

        # Project embeddings to the desired dimension for the sequence model
        projected_embeddings = self.projection(segment_embeddings) # (bs, num_segments, seq_hidden_size)

        # Sequence modeling across segments
        # Process the sequence of segment embeddings for each audio file.
        if self.seq_model_type in ['bilstm', 'transformer']:
            if self.seq_model_type == 'transformer':
                sequence_output = self.sequence_model(projected_embeddings, src_key_padding_mask=attention_mask)
            else: # bilstm
                sequence_output, _ = self.sequence_model(projected_embeddings)
        else:
            sequence_output = projected_embeddings
        # Result shape: (bs, num_segments, seq_output_dim)
        
        # Pool segment-level outputs to get a single audio-level representation
        audio_embeddings = self.audio_embedding_pooling(sequence_output, mask=attention_mask)  # (bs, audio_embedding_dim)

        output = self.classifier(audio_embeddings)  # (bs, 1)

        return output