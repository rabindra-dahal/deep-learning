import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
            
        self.head_dim = head_dim
        self.base = base
        
        # 1. Compute inverse frequencies for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        # 2. Compute angles for all positions up to max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)  # Shape: (max_seq_len, head_dim // 2)
        
        # 3. Duplicate frequencies to match full head_dim and precompute sin/cos
        emb = torch.cat((freqs, freqs), dim=-1)   # Shape: (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Splits the last dimension in half, swaps them, and negates the first half."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to Query and Key tensors.
        Expected input shape: (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = q.shape[2]
        
        # Slice cached cos/sin to match current sequence length and reshape for broadcasting
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        
        # Apply 2D rotation formula
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated

# --- Example Usage ---
if __name__ == "__main__":
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64  # Must be an even number

    # Initialize RoPE module
    rope = RotaryPositionalEmbedding(head_dim=head_dim, max_seq_len=512)

    # Create dummy Query and Key tensors from a Multi-Head Attention layer
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Apply rotary embeddings
    q_rot, k_rot = rope(q, k)

    print(f"Original Q shape: {q.shape}")
    print(f"Rotated Q shape:  {q_rot.shape}")

    # Verify that vector norms are preserved (pure rotation property)
    orig_norm = torch.norm(q[0, 0, 0])
    rot_norm = torch.norm(q_rot[0, 0, 0])
    print(f"Norm preserved:   {torch.allclose(orig_norm, rot_norm)} ({orig_norm:.4f} vs {rot_norm:.4f})")
