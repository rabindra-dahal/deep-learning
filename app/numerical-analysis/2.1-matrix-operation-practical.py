import torch

# ---------------------------------------------------------
# 1. Setup: Inputs, Weights, Biases, and Targets
# ---------------------------------------------------------
# Imagine a batch of 3 samples, each having 4 input features.
# Shape: [Batch Size, Input Features]
X = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                  [0.5, 1.5, 2.5, 3.5],
                  [2.0, 0.0, 1.0, -1.0]])

# Weight matrix for a layer mapping 4 inputs to 2 output neurons.
# Shape: [Input Features, Output Features]
W = torch.tensor([[0.1, 0.5],
                  [-0.2, 0.4],
                  [0.8, -0.1],
                  [-0.5, 0.2]])

# Bias vector for the 2 output neurons.
# Shape: [Output Features] (Will be broadcasted to match the batch size)
b = torch.tensor([0.1, -0.2])

# Ground truth target values we want the network to predict.
# Shape: [Batch Size, Output Features]
targets = torch.tensor([[1.5, 2.5],
                        [1.0, 2.0],
                        [-0.5, 1.0]])

print("--- Data Shapes ---")
print(f"Inputs X: {X.shape} | Weights W: {W.shape} | Bias b: {b.shape}")

# ---------------------------------------------------------
# 2. The Forward Pass (Linear Layer Operation)
# ---------------------------------------------------------
# Formula: Y = X @ W + b
# Matrix multiplication aligns [3, 4] @ [4, 2] to produce a [3, 2] output.
# The bias [2] is automatically added to each row (broadcasting).
predictions = torch.matmul(X, W) + b

print("\n--- Forward Pass Output ---")
print("Model Predictions (Y):\n", predictions)

# ---------------------------------------------------------
# 3. Calculating the Loss Function (Mean Squared Error)
# ---------------------------------------------------------
# Step A: Element-wise Subtraction to find the error/residual vector
# Shape: [3, 2]
errors = predictions - targets

# Step B: Element-wise Multiplication to square the errors
squared_errors = errors * errors  # or errors ** 2

# Step C: Reduce to a single scalar value using torch.mean
# Sum up all squared values and divide by total number of elements (6)
mse_loss = torch.mean(squared_errors)

print("\n--- Loss Calculation ---")
print("Prediction Errors:\n", errors)
print("Squared Errors:\n", squared_errors)
print("\nFinal Mean Squared Error (MSE) Loss Scalar:", mse_loss.item())
