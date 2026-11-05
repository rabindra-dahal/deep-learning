import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# 1. Load the official scikit-learn housing dataset as a pandas frame
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# 2. Extract target and multi-variable features
X_raw = housing_data.data.values
y_raw = housing_data.target.values

# 3. Split into 80% Training and 20% Testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# 4. Standardize spatial and volumetric features to stabilize gradients
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# 5. Convert to PyTorch Tensors
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)

# Wrap tensors into a dataset structure and create PyTorch DataLoaders
# Batch size configuration (commonly 32, 64, or 128 for tabular data)
BATCH_SIZE = 64

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


# 6. Define Model Architecture with Batch Normalization
class HousingRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(HousingRegressionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),          # Replaced BatchNorm1d with LayerNorm
            nn.ReLU(),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),          # Replaced BatchNorm1d with LayerNorm
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)


# Instantiate the model
model = HousingRegressionModel(input_dim=X_train.shape[1])

# 7. Configure Loss and Optimization
criterion = nn.MSELoss()
# Batch normalization allows for a higher learning rate safely
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Add weight_decay to penalize overly dominant weights (like AveOccup)
# optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# 8. Mini-Batch Training loop
epochs = 60  # Reduced epochs because BatchNorm significantly accelerates convergence
for epoch in range(epochs):
  model.train()  # CRITICAL: Activates running tracking for BatchNorm
  running_loss = 0.0

  for batch_X, batch_y in train_loader:
    # Handle edge case where a final batch might have 1 sample
    # BatchNorm1d requires batch_size > 1 to compute variance
    if batch_X.size(0) <= 1:
      continue

    predictions = model(batch_X)
    loss = criterion(predictions, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * batch_X.size(0)

  epoch_loss = running_loss / len(train_loader.dataset)

  if (epoch + 1) % 10 == 0:
    print(f"Epoch [{epoch+1}/{epochs}], Training MSE Loss: {epoch_loss:.4f}")

# 9. Evaluation on Test Set via DataLoader
model.eval()  # CRITICAL: Freezes running mean/variance for evaluation
test_loss = 0.0
mae_criterion = nn.L1Loss()
test_mae = 0.0

with torch.no_grad():
  for batch_X, batch_y in test_loader:
    test_preds = model(batch_X)

    test_loss += criterion(test_preds, batch_y).item() * batch_X.size(0)
    test_mae += mae_criterion(test_preds, batch_y).item() * batch_X.size(0)

final_test_mse = test_loss / len(test_loader.dataset)
final_test_mae = test_mae / len(test_loader.dataset)

print("\n--- Model Evaluation Report ---")
print(f"Test Squared Loss (MSE): {final_test_mse:.4f}")
print(
    f"Average Target Mistake (MAE):"
    f" ${final_test_mae * 100000:.2f} (in actual USD)"
)

