import torch
import pytest
from focus.models import FocusLSTM, FocusLayer

def test_focus_layer():
    batch_size = 4
    seq_len = 32
    hidden_dim = 256
    
    # Initialize layer
    layer = FocusLayer(hidden_dim)
    
    # Create dummy input
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    context, weights, mu, sigma = layer(hidden_states, attention_mask)
    
    # Check shapes
    assert context.shape == (batch_size, hidden_dim)
    assert weights.shape == (batch_size, seq_len)
    assert mu.shape == (batch_size,)
    assert sigma.shape == (batch_size,)
    
    # Check attention weights sum to 1
    assert torch.allclose(weights.sum(dim=1), torch.ones(batch_size))

def test_focus_lstm():
    batch_size = 4
    seq_len = 32
    vocab_size = 1000
    hidden_dim = 256
    
    # Initialize model
    model = FocusLSTM(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test forward pass without attention
    logits = model(input_ids, attention_mask)
    assert logits.shape == (batch_size, 2)
    
    # Test forward pass with attention
    logits, weights, mu, sigma = model(input_ids, attention_mask, return_attention=True)
    assert logits.shape == (batch_size, 2)
    assert weights.shape == (batch_size, seq_len)
    assert mu.shape == (batch_size,)
    assert sigma.shape == (batch_size,)
