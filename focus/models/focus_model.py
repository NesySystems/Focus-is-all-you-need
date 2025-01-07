import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
from .focus_layer import FocusLayer

class FocusLSTM(nn.Module):
    """
    LSTM model with Focus attention mechanism for sequence classification.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        n_heads: int = 4,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Focus layer
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.focus = FocusLayer(lstm_out_dim, n_heads)
        
        # Classification head
        self.classifier = nn.Linear(lstm_out_dim, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            return_attention: If True, return attention weights and Gaussian parameters
            
        Returns:
            If return_attention is False:
                logits: Classification logits of shape (batch_size, num_classes)
            If return_attention is True:
                tuple of:
                    logits: Classification logits of shape (batch_size, num_classes)
                    attention_weights: Attention weights of shape (batch_size, seq_len)
                    mu: Focus center positions of shape (batch_size, 1)
                    sigma: Focus width parameters of shape (batch_size, 1)
        """
        # Get embeddings
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, hidden_dim)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply Focus mechanism
        context, attention_weights, mu, sigma = self.focus(lstm_out, attention_mask)
        
        # Get logits
        logits = self.classifier(context)
        
        if return_attention:
            return logits, attention_weights, mu, sigma
        return logits
