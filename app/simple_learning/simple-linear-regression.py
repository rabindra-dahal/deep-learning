import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate synthetic data (y = 2x + 1)
X = torch.randn(100, 1)
y = 2 * X + 1

# 2. Instantiate the linear layer directly 
# 1 input feature, 1 output feature
model = nn.Linear(in_features=1, out_features=1)

# 3. Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
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

# 5. Check the learned parameters
print("\nTarget weights: weight=2.0, bias=1.0")
print(f"Learned weight: {model.weight.item():.4f}")
print(f"Learned bias: {model.bias.item():.4f}")
