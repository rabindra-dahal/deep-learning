import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate synthetic data with 3 input features
# Target formula: y = (3 * x1) + (-2 * x2) + (1.5 * x3) + 4.2
X = torch.randn(100, 3)  # 100 samples, 3 features each
true_weights = torch.tensor([[3.0], [-2.0], [1.5]])
true_bias = 4.2
y = torch.matmul(X, true_weights) + true_bias

# 2. Instantiate the linear layer directly
# 3 input features, 1 output feature
model = nn.Linear(in_features=3, out_features=1)

# 3. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
epochs = 500
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 5. Check the learned parameters
print("\nTarget weights: [3.0, -2.0, 1.5] | Target bias: 4.2")
print(f"Learned weights: {model.weight.detach().numpy().flatten()}")
print(f"Learned bias: {model.bias.item():.4f}")
