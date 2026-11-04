import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Load the official scikit-learn housing dataset as a pandas frame
housing_data = fetch_california_housing(as_frame=True)
df = housing_data.frame

# 2. Extract target and multi-variable features
# Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
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
y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1) # Shape (N, 1)

X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)

# 6. Define Model Architecture dynamically matching the 8 input features
class HousingRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(HousingRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
        
    def forward(self, x):
        return self.linear(x)

# Instantiate the model with 8 input dimensions
model = HousingRegressionModel(input_dim=X_train.shape[1])

# 7. Configure Loss and Optimization
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 8. Training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 400 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Training MSE Loss: {loss.item():.4f}")

# 9. Performance Evaluation on Unseen Test Split
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_mse = criterion(test_preds, y_test).item()
    
    # Calculate Mean Absolute Error (MAE)
    mae_criterion = nn.L1Loss()
    test_mae = mae_criterion(test_preds, y_test).item()

print("\n--- Model Evaluation Report ---")
print(f"Test Squared Loss (MSE): {test_mse:.4f}")
print(f"Average Target Mistake (MAE): ${test_mae * 100000:.2f} (in actual USD)")

# 10. Extract Feature Contribution Weights
weights = model.linear.weight.detach().numpy()[0]
intercept = model.linear.bias.item()

print(f"\nModel Intercept Baseline: {intercept:.4f}")
for feature_name, weight_val in zip(housing_data.feature_names, weights):
    print(f"Weight Coefficient for [{feature_name}]: {weight_val:.4f}")


#11. Test the model on real data
# 1. Switch the model to evaluation mode
model.eval()

# 2. Define raw feature values for a single neighborhood block group
# Order must match exactly: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
custom_block = np.array([[
    4.5,     # MedInc: Median income in tens of thousands ($45,000)
    28.0,    # HouseAge: Median house age in years
    5.5,     # AveRooms: Average number of rooms per household
    1.0,     # AveBedrms: Average number of bedrooms per household
    1200.0,  # Population: Total block population
    2.8,     # AveOccup: Average household members
    34.05,   # Latitude: Geographic coordinate
    -118.24  # Longitude: Geographic coordinate (e.g., Los Angeles area)
]], dtype=np.float32)

# 3. Transform the raw features using your existing scaler instance
custom_block_scaled = scaler.transform(custom_block)

# 4. Convert to a PyTorch tensor
input_tensor = torch.tensor(custom_block_scaled, dtype=torch.float32)

# 5. Generate prediction without calculating gradients
with torch.no_grad():
    scaled_prediction = model(input_tensor).item()

# 6. Unscale the target (target is in units of $100,000)
final_home_value_usd = scaled_prediction * 100000

print(f"--- Single Record Prediction ---")
print(f"Predicted Median Home Value: ${final_home_value_usd:,.2f}")

