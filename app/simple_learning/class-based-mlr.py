import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate synthetic data with 3 input features
# Target formula: y = (3 * x1) + (-2 * x2) + (1.5 * x3) + 4.2
X = torch.randn(100, 3)  # 100 samples, 3 features each
true_weights = torch.tensor([[3.0], [-2.0], [1.5]])
true_bias = 4.2
y = torch.matmul(X, true_weights) + true_bias

# 2. Define the Multiple Linear Regression Model
class MultipleLinearRegressionModel(nn.Module):
    def __init__(self):
        super(MultipleLinearRegressionModel, self).__init__()
        # 3 input features, 1 output feature
        self.linear = nn.Linear(in_features=3, out_features=1)
        
    def forward(self, x):
        return self.linear(x)

# 3. Instantiate model, loss function, and optimizer
model = MultipleLinearRegressionModel() 
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
print("\nTarget weights: [3.0, -2.0, 1.5] | Target bias: 4.2")
print(f"Learned weights: {model.linear.weight.detach().numpy().flatten()}") 
print(f"Learned bias: {model.linear.bias.item():.4f}")


