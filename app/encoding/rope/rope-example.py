import math
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 128, base: float = 10000.0):
        super().__init__()
        dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = torch.exp(-dim_indices * (math.log(base) / head_dim))
        
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]  # Extracts seq_len from (batch, heads, seq_len, head_dim)
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

# --- Token-Level Demonstration ---
if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. Define vocabulary and tokenize sentence
    vocab = {"the": 0, "dog": 1, "bites": 2, "man": 3}
    sentence = ["the", "dog", "bites", "the", "man"]
    token_ids = torch.tensor([[vocab[word] for word in sentence]])  # Shape: (1, 5)

    # 2. Create embeddings and Q/K projections (1 batch, 1 head, 5 tokens, 8 dims)
    head_dim = 8
    embed = nn.Embedding(len(vocab), head_dim)
    w_q = nn.Linear(head_dim, head_dim, bias=False)
    w_k = nn.Linear(head_dim, head_dim, bias=False)

    x = embed(token_ids)
    q = w_q(x).unsqueeze(1)  # Shape: (1, 1, 5, 8)
    k = w_k(x).unsqueeze(1)  # Shape: (1, 1, 5, 8)

    # 3. Apply RoPE
    rope = RotaryPositionalEmbedding(head_dim=head_dim)
    q_rot, k_rot = rope(q, k)

    # Check 1: Same word ("the") at position 0 vs position 3
    the_pos0 = q_rot[0, 0, 0]
    the_pos3 = q_rot[0, 0, 3]
    print("Are 'the' (pos 0) and 'the' (pos 3) identical after RoPE?:", torch.allclose(the_pos0, the_pos3))
    print(f"Cosine similarity between 'the' at pos 0 and pos 3: {torch.cosine_similarity(the_pos0, the_pos3, dim=0):.4f}\n")

    # Check 2: Relative distance invariance ("dog bites" at pos 1->2 vs pos 3->4)
    # Score in "the [dog] [bites] the man" (pos 1 attending to pos 2, distance = -1)
    score_dog_bites_1 = torch.dot(q_rot[0, 0, 1], k_rot[0, 0, 2])

    # Swap word order: "the man the [dog] [bites]" (pos 3 attending to pos 4, distance = -1)
    swapped_sentence = ["the", "man", "the", "dog", "bites"]
    swapped_ids = torch.tensor([[vocab[w] for w in swapped_sentence]])
    x_swap = embed(swapped_ids)
    q_swap, k_swap = rope(w_q(x_swap).unsqueeze(1), w_k(x_swap).unsqueeze(1))
    score_dog_bites_2 = torch.dot(q_swap[0, 0, 3], k_swap[0, 0, 4])

    print(f"Attention score ('dog'->'bites' at pos 1,2): {score_dog_bites_1.item():.6f}")
    print(f"Attention score ('dog'->'bites' at pos 3,4): {score_dog_bites_2.item():.6f}")
    print("Relative distance property verified:", torch.allclose(score_dog_bites_1, score_dog_bites_2))
