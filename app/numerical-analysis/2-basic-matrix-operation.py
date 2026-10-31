import torch

# ---------------------------------------------------------
# 0. Setup: Create Sample Matrices
# ---------------------------------------------------------
# Creating 2x2 square matrices (using floats for division/inverse operations)
A = torch.tensor([[4.0, 7.0], 
                  [2.0, 6.0]])

B = torch.tensor([[1.0, 3.0], 
                  [5.0, 2.0]])

print("Matrix A:\n", A)
print("\nMatrix B:\n", B)
print("-" * 50)

A_diag = torch.diagonal(A)
print("Diagonal of A:\n", A_diag)
print("-" * 50)

# ---------------------------------------------------------
# 1. Basic Arithmetic (Element-wise)
# ---------------------------------------------------------
# Addition
matrix_add = A + B  # or torch.add(A, B)
print("1. Addition (A + B):\n", matrix_add)

# Subtraction
matrix_sub = A - B  # or torch.sub(A, B)
print("\n2. Subtraction (A - B):\n", matrix_sub)

# Element-wise Multiplication (Hadamard Product)
matrix_mul_element = A * B  # or torch.mul(A, B)
print("\n3. Element-wise Multiplication (A * B):\n", matrix_mul_element)
print("-" * 50)

# ---------------------------------------------------------
# 2. Advanced Matrix Multiplication
# ---------------------------------------------------------
# Standard Matrix Multiplication (Dot Product of rows and columns)
# Row 1, Col 1 calculation: (4*1) + (7*5) = 4 + 35 = 39
matrix_matmul = A @ B  # or torch.matmul(A, B) or torch.mm(A, B)
print("4. Matrix Multiplication (A @ B):\n", matrix_matmul)
print("-" * 50)

# ---------------------------------------------------------
# 3. Core Matrix Properties & Structural Changes
# ---------------------------------------------------------
# Transpose (Flipping rows and columns)
matrix_transpose = A.T  # or torch.t(A)
print("5. Transpose of A (A.T):\n", matrix_transpose)

# Trace (Sum of diagonal elements: 4.0 + 6.0)
matrix_trace = torch.trace(A)
print("\n6. Trace of A:", matrix_trace.item())

# Determinant (ad - bc: (4*6) - (7*2) = 24 - 14 = 10)
matrix_det = torch.det(A)
print("\n7. Determinant of A:", matrix_det.item())
print("-" * 50)

# ---------------------------------------------------------
# 4. Inverse & Adjugate (Adjunct)
# ---------------------------------------------------------
# Inverse Matrix
matrix_inv = torch.linalg.inv(A)  # or torch.inverse(A)
print("8. Inverse of A (A^-1):\n", matrix_inv)

# Verify Inverse property: A multiplied by its inverse equals Identity Matrix (I)
print("\nVerification (A @ A^-1):\n", torch.round(A @ matrix_inv))

# Adjugate (Adjunct) Matrix
# Formula: adj(A) = det(A) * A^-1
matrix_adj = torch.det(A) * torch.linalg.inv(A)
print("\n9. Adjugate (Adjunct) of A:\n", matrix_adj)
print("-" * 50)

# ---------------------------------------------------------
# 5. Advanced Spectral Decompositions
# ---------------------------------------------------------
# Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = torch.linalg.eig(A)
print("10. Eigenvalues of A:\n", eigenvalues)
print("    Eigenvectors of A:\n", eigenvectors)

# Singular Value Decomposition (SVD)
U, S, V = torch.linalg.svd(A)
print("\n11. SVD of A:")
print("    U matrix:\n", U)
print("    Singular values (S):\n", S)
print("    V matrix (V^T is the right singular vectors):\n", V)
