import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate synthetic data with 3 input features
# Target formula: y = (3 * x1) + (-2 * x2) + (1.5 * x3) + 4.2
X = torch.randn(100, 3)  # 100 samples, 3 features each
true_weights = torch.tensor([[3.0], [-2.0], [1.5]])
true_bias = 4.2
y = torch.matmul(X, true_weights) + true_bias

# 2. Define the Neural Network Model with a Hidden Layer
class MultipleLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultipleLinearRegressionModel, self).__init__()
        # Input layer (1 feature) -> Hidden layer (e.g., 8 neurons)
        self.hidden = nn.Linear(in_features=3, out_features=8)
        # Activation function
        self.activation = nn.ReLU()
        # Hidden layer (8 neurons) -> Output layer (1 feature)
        self.output = nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# 3. Instantiate model, loss function, and optimizer
model = MultipleLinearRegressionModel()

# 4. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. Training loop
epochs = 500
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing X to the model
    y_pred = model(X)
    
    # Compute loss
    loss = criterion(y_pred, y)
    
    # Zero gradients, perform a backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. Check final loss and predictions
print("\nTraining completed. Final loss:", loss.item())
# 7. Check the learned parameters for each layer
print("=== HIDDEN LAYER PARAMETERS ===")
print("Weights (Shape: 8x1):\n", model.hidden.weight.data)
print("\nBiases (Shape: 8):\n", model.hidden.bias.data)

print("\n=== OUTPUT LAYER PARAMETERS ===")
print("Weights (Shape: 1x8):\n", model.output.weight.data)
print("\nBiases (Shape: 1):\n", model.output.bias.data)
