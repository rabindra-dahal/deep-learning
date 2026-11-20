import math
import torch
import torch.nn as nn

torch.manual_seed(42)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        return x + self.pos_emb(positions)


class RoPE(nn.Module):
    """Applies 2D rotation to consecutive feature pairs (x0, x1) for each position."""

    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for pairwise rotation"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)  # (seq_len, d_model // 2)
        cos = torch.cos(angles).unsqueeze(0)  # (1, seq_len, d_model // 2)
        sin = torch.sin(angles).unsqueeze(0)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 2D rotation on each (even, odd) pair
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out


class ALiBiBias(nn.Module):
    """Computes distance penalty matrix to add to attention scores (seq_len, seq_len)."""

    def __init__(self, slope: float = 0.5):
        super().__init__()
        self.slope = slope

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        # Distance matrix: i - j for causal attention (penalizes distant past keys)
        rel_pos = idx.unsqueeze(1) - idx.unsqueeze(0)  # (seq_len, seq_len)
        bias = -self.slope * rel_pos.abs().float()
        # Mask future positions with -inf for standard causal attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return bias.masked_fill(causal_mask, float("-inf"))


# Test on a simple (batch_size=1, seq_len=4, d_model=4) tensor
x = torch.ones(1, 4, 4)

sin_pe = SinusoidalPositionalEncoding(d_model=4, max_len=32)
lrn_pe = LearnedPositionalEncoding(d_model=4, max_len=32)
rope = RoPE(d_model=4)
alibi = ALiBiBias(slope=0.5)

print("Sinusoidal output (pos 0 vs pos 3):")
sin_out = sin_pe(x)
print("pos 0:", sin_out[0, 0].round(decimals=4))
print("pos 3:", sin_out[0, 3].round(decimals=4))

print("\nLearned PE output (pos 0 vs pos 3):")
lrn_out = lrn_pe(x)
print("pos 0:", lrn_out[0, 0].detach().round(decimals=4))
print("pos 3:", lrn_out[0, 3].detach().round(decimals=4))

print("\nRoPE rotated vector (pos 0 vs pos 2):")
rope_out = rope(x)
print("pos 0:", rope_out[0, 0].round(decimals=4))
print("pos 2:", rope_out[0, 2].round(decimals=4))

print("\nALiBi attention score bias (4x4):")
print(alibi(seq_len=4, device=x.device))
