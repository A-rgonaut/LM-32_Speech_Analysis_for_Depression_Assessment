import torch
from torch import nn

class AttentionPoolingLayer(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

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

class MeanPoolingLayer(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, x, mask=None):
        """
        Forward pass per il Mean Pooling mascherato.
        Args:
            x: Tensore di input (batch_size, seq_len, embed_dim).
            mask: Maschera di padding (batch_size, seq_len), dove True indica padding.
        Returns:
            Tensore di output (batch_size, embed_dim).
        """
        if mask is None:
            # Se non c'è maschera, usa una media standard
            return torch.mean(x, dim=1)

        # 1. Metti a zero i valori di padding in x per non contarli nella somma
        # mask.unsqueeze(-1) espande la maschera a (batch_size, seq_len, 1)
        x_masked = x.masked_fill(mask.unsqueeze(-1), 0.0)

        # 2. Calcola la somma lungo la dimensione della sequenza
        summed = torch.sum(x_masked, dim=1)  # (batch_size, embed_dim)

        # 3. Calcola il numero di elementi non-paddati per ogni sequenza
        # ~mask inverte la maschera (True dove ci sono dati validi)
        num_valid_elements = (~mask).sum(dim=1).unsqueeze(-1) # (batch_size, 1)

        # 4. Evita la divisione per zero se una sequenza è completamente vuota
        num_valid_elements = torch.clamp(num_valid_elements, min=1e-9)

        # 5. Calcola la media
        mean_pooled = summed / num_valid_elements

        return mean_pooled
    
class GatedAttentionPoolingLayer(nn.Module):
    """
    Gated attention pooling (Ilse et al., 2018).
    Input:  x -> (bs, seq_len, d)
            mask -> (bs, seq_len) con True=padding da escludere
    Output: pooled -> (bs, d)
            (opz.) attn -> (bs, seq_len)
    """
    def __init__(self, embed_dim, attn_dim=None, return_weights=False):
        super().__init__()
        h = attn_dim or embed_dim  # dimensione dello spazio di attenzione
        self.V = nn.Linear(embed_dim, h, bias=True)   # ramo "tanh"
        self.U = nn.Linear(embed_dim, h, bias=True)   # ramo "sigmoid" (gate)
        self.w = nn.Linear(h, 1, bias=False)          # proiezione finale
        self.return_weights = return_weights

        # inizializzazioni stabili
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.V.bias)
        nn.init.zeros_(self.U.bias)

    def forward(self, x, mask=None):
        H = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))   # (bs, L, h)
        scores = self.w(H)                                     # (bs, L, 1)

        if mask is not None:
            # True = padding → metti -inf prima del softmax
            scores = scores.masked_fill(mask.unsqueeze(-1), float('-inf'))

        attn = torch.softmax(scores, dim=1)                    # (bs, L, 1)
        pooled = torch.sum(attn * x, dim=1)                    # (bs, d)

        if self.return_weights:
            return pooled, attn.squeeze(-1)
        return pooled