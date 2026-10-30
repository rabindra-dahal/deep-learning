import torch

# Create a square 3x3 matrix
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 10.0]], dtype=torch.float32)

# Unpack directly into P, L, and U matrices
P, L, U = torch.linalg.lu(A)

print("P (Permutation Matrix):\n", P)
print("\nL (Lower Triangular):\n", L)
print("\nU (Upper Triangular):\n", U)

assert torch.allclose(P @ L @ U, A, atol=1e-5)

