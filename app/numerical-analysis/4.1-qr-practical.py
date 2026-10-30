import torch

# 1. Generate synthetic data for Linear Regression: y = 2.5 * x1 + (-1.2) * x2 + 4.0
torch.manual_seed(42)
N, D = 100, 2  # 100 samples, 2 features

# True weights and bias
true_w = torch.tensor([2.5, -1.2])
true_b = torch.tensor([4.0])

# Features (X_raw) and Targets (y)
X_raw = torch.randn(N, D)
y = X_raw @ true_w + true_b + torch.randn(N) * 0.1  # Add a bit of noise

# 2. Add a column of ones to X_raw to account for the bias/intercept term
# This forms the design matrix X (shape: N x (D + 1))
ones = torch.ones(N, 1)
X = torch.cat([X_raw, ones], dim=1)

# 3. Apply QR Decomposition
# X = Q @ R
Q, R = torch.linalg.qr(X)

# 4. Compute Q^T @ y
# Since Q is orthogonal, we only need its transpose, not its inverse
Qty = Q.T @ y

# 5. Solve for weights (beta) using Back-Substitution
# R is upper triangular, so torch.linalg.solve_triangular is highly efficient and stable
# It solves the system: R @ beta = Q^T @ y
beta = torch.linalg.solve_triangular(R, Qty.unsqueeze(1), upper=True).squeeze()

# 6. Extract learned weights and bias
learned_w = beta[:D]
learned_b = beta[D:]

# Print results
print("--- Results ---")
print(f"True Weights:    {true_w.tolist()}")
print(f"Learned Weights: {learned_w.tolist()}\n")
print(f"True Bias:       {true_b.item():.4f}")
print(f"Learned Bias:    {learned_b.item():.4f}")

# Verify predictions match PyTorch's native least squares solver
beta_native = torch.linalg.lstsq(X, y).solution
assert torch.allclose(beta, beta_native, atol=1e-5)
print("\nSuccess: Verification passed! Matches native stable solvers.")
