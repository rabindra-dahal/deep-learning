import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Populate even and odd dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension: shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as a non-trainable buffer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape (batch_size, seq_len, d_model)
        """
        # Slice precomputed encodings to match input sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Example usage
d_model = 512
pos_encoder = SinusoidalPositionalEncoding(d_model=d_model)
dummy_embeddings = torch.randn(32, 20, d_model)  # (batch_size, seq_len, d_model)
output = pos_encoder(dummy_embeddings)
print(output.shape)  # Output: torch.Size([32, 20, 512])
