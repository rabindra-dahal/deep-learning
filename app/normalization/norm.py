import torch
import torch.nn as nn

torch.manual_seed(42)


import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        # Small constant added to variance to avoid division by zero
        self.eps = eps
        # Learnable scale parameter (gamma), initialized to 1s per feature
        self.gamma = nn.Parameter(torch.ones(d_model))
        # Learnable shift/bias parameter (beta), initialized to 0s per feature
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # Compute mean across the feature dimension (d_model) for each token independently
        mean = x.mean(dim=-1, keepdim=True)
        # Compute population variance (unbiased=False) across d_model; keepdim=True preserves (B, T, 1) for broadcasting
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Center each token's features around 0 and scale to unit variance
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Apply learnable per-feature scale (gamma) and shift (beta)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # Small constant added inside the square root for numerical stability
        self.eps = eps
        # Learnable scale parameter (no bias/shift in standard RMSNorm), initialized to 1s
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # Compute Root Mean Square across the feature dimension (d_model) without subtracting the mean
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize each token by its RMS magnitude and apply learnable per-feature scaling
        return self.weight * (x / rms)


# x shape: (batch_size=2, seq_len=3, d_model=4)
x = torch.tensor([
    [
        [10.0, 12.0, 14.0, 16.0],     # Batch 0, Token 0
        [0.1,  0.2,  0.3,  0.4],      # Batch 0, Token 1
        [-5.0, 0.0,  5.0,  10.0],     # Batch 0, Token 2
    ],
    [
        [2.0,  2.0,  2.0,  2.0],      # Batch 1, Token 0 (constant)
        [1.0,  3.0,  5.0,  7.0],      # Batch 1, Token 1
        [100.0, 102.0, 104.0, 106.0], # Batch 1, Token 2
    ],
])

ln = LayerNorm(d_model=4, eps=1e-5)
rmsn = RMSNorm(d_model=4, eps=1e-6)

ln_out = ln(x)
rms_out = rmsn(x)

print("--- LayerNorm Output ---")
print(ln_out.round(decimals=4))

print("\n--- RMSNorm Output ---")
print(rms_out.round(decimals=4))
