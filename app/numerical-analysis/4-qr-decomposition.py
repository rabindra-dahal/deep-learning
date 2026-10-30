import torch

# Create a matrix (does not have to be square)
A = torch.tensor([[12.0, -51.0, 4.0],
                  [6.0, 167.0, -68.0],
                  [-4.0, 24.0, -41.0]], dtype=torch.float32)

# Compute QR decomposition
Q, R = torch.linalg.qr(A)

print("Q (Orthogonal Matrix):\n", Q)
print("\nR (Upper Triangular):\n", R)

# Verify: Q @ R should equal A
assert torch.allclose(Q @ R, A, atol=1e-5)
