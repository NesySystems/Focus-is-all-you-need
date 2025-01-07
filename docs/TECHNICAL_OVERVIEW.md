# Technical Overview

The **Focus Mechanism** merges standard attention with a lens-inspired "focusing" operation, all within a common neural architecture (LSTM or Transformer). Below is a breakdown of the core elements:

## 1. Architecture Outline

- **Embedding**: Converts token IDs or features into a 256-dimensional space (configurable).
- **BiLSTM**: Two layers of bidirectional LSTM with hidden_dim=256 (512 when combined forward/backward).
- **Focus Layer**: Replaces or augments standard attention by incorporating Gaussian-based focusing.
- **Final Classifier**: Outputs 2 logits for binary classification (can adapt for multi-class or regression).

## 2. Focus Layer Components

1. **Query/Key/Value**  
   - Standard Q-K-V projections derived from the hidden states.
   - Typically multi-head (e.g., 4 heads).

2. **Gaussian Parameters (μ, σ)**  
   - Predicted from the hidden states to locate the "focal center" and "aperture width."
   - Combined into a lens-based weighting distribution.

3. **Combined Distribution**  
   - Standard attention weights multiplied (element-wise) by the lens distribution.
   - Normalized via softmax to ensure a valid probability distribution across tokens.

## 3. Training Configuration

| Hyperparameter | Value    |
|----------------|---------:|
| Embedding Dim  | 256      |
| Hidden Dim     | 256      |
| Layers (LSTM)  | 2        |
| Dropout        | 0.1      |
| Bidirectional  | True     |
| n_heads        | 4        |
| Loss Function  | CrossEntropy |
| Optimizer      | AdamW    |
| LR             | 2e-5     |
| Weight Decay   | 0.01     |
| Epochs         | 10       |

## 4. Implementation Highlights

```python
class FocusLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.mu_head    = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.n_heads    = n_heads
        self.dropout    = nn.Dropout(dropout)
        
        # optional: layer normalization, multi-head splitting, etc.
