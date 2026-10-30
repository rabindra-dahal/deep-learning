import torch

# 1. Generate synthetic data for Linear Regression: y = 2.5 * x1 + (-1.2) * x2 + 4.0
torch.manual_seed(42)
N, D = 100, 2  # 100 samples, 2 features

true_w = torch.tensor([2.5, -1.2])
true_b = torch.tensor([4.0])

X_raw = torch.randn(N, D)
y = X_raw @ true_w + true_b + torch.randn(N) * 0.1  # Add a bit of noise

# Add bias column of ones to form the design matrix X (N x (D + 1))
ones = torch.ones(N, 1)
X = torch.cat([X_raw, ones], dim=1)

# 2. Compute the Normal Equation components: A = X^T @ X, b = X^T @ y
# LU decomposition requires a SQUARE matrix, so we must compute X^T X
A = X.T @ X
b = X.T @ y

# 3. Apply LU Decomposition on the square matrix A
# A = P @ L @ U
P, L, U = torch.linalg.lu(A)

# 4. Step 1: Solve L @ z = P^T @ b
# Because PyTorch outputs A = P @ L @ U, we multiply by P.T to move it to the b side
Pb = P.T @ b
z = torch.linalg.solve_triangular(L, Pb.unsqueeze(1), upper=False)

# 5. Step 2: Solve U @ beta = z
beta = torch.linalg.solve_triangular(U, z, upper=True).squeeze()

# 6. Extract learned weights and bias
learned_w = beta[:D]
learned_b = beta[D:]

# Print results
print("--- LU Decomposition Results ---")
print(f"True Weights:    {true_w.tolist()}")
print(f"Learned Weights: {learned_w.tolist()}\n")
print(f"True Bias:       {true_b.item():.4f}")
print(f"Learned Bias:    {learned_b.item():.4f}")

# Verify predictions match PyTorch's native least squares solver
beta_native = torch.linalg.lstsq(X, y).solution
assert torch.allclose(beta, beta_native, atol=1e-5)
print("\nSuccess: Verification passed! Matches native stable solvers.")
