import torch
import torch.nn as nn
from typing import Tuple, Optional
import math

class FocusLayer(nn.Module):
    """
    Focus Layer: A novel attention mechanism that combines traditional attention with a dynamic Gaussian focus window.
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Multi-head attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Gaussian parameters
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Focus layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            
        Returns:
            tuple of:
                context: Attended context vector of shape (batch_size, hidden_dim)
                attention_weights: Focus attention weights of shape (batch_size, seq_len)
                mu: Focus center positions of shape (batch_size, 1)
                sigma: Focus width parameters of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to query, key, value
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        # Get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Get Gaussian parameters
        mu = self.mu_head(context.mean(dim=1))  # (batch_size, 1)
        sigma = torch.abs(self.sigma_head(context.mean(dim=1))) + 1e-3  # (batch_size, 1)
        
        # Create Gaussian attention
        positions = torch.arange(seq_len, dtype=torch.float32, device=hidden_states.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (batch_size, seq_len)
        
        mu = mu.expand(-1, seq_len)  # (batch_size, seq_len)
        sigma = sigma.expand(-1, seq_len)  # (batch_size, seq_len)
        
        gaussian_weights = torch.exp(-(positions - mu) ** 2 / (2 * sigma ** 2))
        
        # Combine with regular attention
        final_weights = attention_weights.mean(dim=1) * gaussian_weights  # (batch_size, seq_len)
        final_weights = final_weights / (final_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Apply final attention
        final_context = torch.bmm(final_weights.unsqueeze(1), hidden_states).squeeze(1)
        final_context = self.layer_norm(final_context)
        
        return final_context, final_weights, mu[:, 0], sigma[:, 0]
