import torch
from focus.models import FocusLSTM
from transformers import AutoTokenizer

def main():
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = FocusLSTM(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=256,
        n_layers=2,
        n_heads=4
    )
    
    # Example input
    text = "This is an example sentence to demonstrate the Focus mechanism."
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Get predictions and attention weights
    with torch.no_grad():
        logits, attention_weights, mu, sigma = model(
            inputs['input_ids'],
            inputs['attention_mask'],
            return_attention=True
        )
    
    # Print results
    print(f"Input text: {text}")
    print(f"Prediction logits: {logits}")
    print(f"Focus center (mu): {mu}")
    print(f"Focus width (sigma): {sigma}")
    print(f"Attention shape: {attention_weights.shape}")

if __name__ == "__main__":
    main()
