import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        # Learnable embedding table for positions 
        # Shape: [max_len, d_model]
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # 1. Generate position indices [0, 1, ..., seq_len - 1]
        # and expand them to match the batch size: [batch_size, seq_len]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        # 2. Retrieve positional vectors and add them element-wise to the token embeddings
        pos_emb = self.pos_embedding(positions)  # Shape: [batch_size, seq_len, d_model]
        return x + pos_emb

# ==========================================
# Example Usage & Verification
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 2
    seq_len = 5       # Number of tokens in our sentence
    d_model = 16      # Hidden dimension of the transformer
    max_len = 100     # Maximum supported sequence length

    # Simulate input token embeddings (e.g., output of a standard token embedding layer)
    # Shape: [2, 5, 16]
    token_embeddings = torch.randn(batch_size, seq_len, d_model)
    print(f"Input token embeddings shape: {token_embeddings.shape}")

    # Initialize the Learned Positional Encoding module
    pos_encoder = LearnedPositionalEncoding(max_len=max_len, d_model=d_model)

    # Pass the token embeddings through the positional encoder
    output = pos_encoder(token_embeddings)
    print(f"Output shape (Tokens + Learned Positions): {output.shape}")

    # Inspect the learnable weights for position index 0 (first 4 dimensions)
    print("\nLearned weights for position 0 (first 4 dimensions):")
    print(pos_encoder.pos_embedding.weight[0][:4])
