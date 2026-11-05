import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
y_train = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)  # Shape (N, 1)

X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)


# 6. Define a Multi-Layer Deep Neural Network (DNN) Architecture
class HousingRegressionModel(nn.Module):

  def __init__(self, input_dim):
    super(HousingRegressionModel, self).__init__()
    self.network = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

  def forward(self, x):
    return self.network(x)


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

  # Backward pass and optimization
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 400 == 0:
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

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


# 10. Extract First-Layer Feature Contribution Weights (DNN Adaptation)
# In a DNN, we check the first linear layer inside our nn.Sequential network
first_layer = model.network[0]
first_layer_weights = first_layer.weight.detach().numpy()

# We calculate the mean absolute weight magnitude for each input feature
# to understand their initial relative importance into the network
feature_importance = np.abs(first_layer_weights).mean(axis=0)

print(f"\n--- Initial Layer Feature Impact ---")
for feature_name, importance_val in zip(
    housing_data.feature_names, feature_importance
):
  print(
      f"Mean Weight Magnitude for [{feature_name}]: {importance_val:.4f}"
  )


# 11. Test the model on real data
# 1. Switch the model to evaluation mode
model.eval()

# 2. Define raw feature values for a single neighborhood block group
custom_block = np.array(
    [[
        4.5,  # MedInc: Median income in tens of thousands ($45,000)
        28.0,  # HouseAge: Median house age in years
        5.5,  # AveRooms: Average number of rooms per household
        1.0,  # AveBedrms: Average number of bedrooms per household
        1200.0,  # Population: Total block population
        2.8,  # AveOccup: Average household members
        34.05,  # Latitude: Geographic coordinate
        -118.24,  # Longitude: Geographic coordinate (e.g., Los Angeles area)
    ]],
    dtype=np.float32,
)

# 3. Transform the raw features using your existing scaler instance
custom_block_scaled = scaler.transform(custom_block)

# 4. Convert to a PyTorch tensor
input_tensor = torch.tensor(custom_block_scaled, dtype=torch.float32)

# 5. Generate prediction without calculating gradients
with torch.no_grad():
  scaled_prediction = model(input_tensor).item()

# 6. Unscale the target (target is in units of $100,000)
final_home_value_usd = scaled_prediction * 100000

print(f"\n--- Single Record Prediction ---")
print(f"Predicted Median Home Value: ${final_home_value_usd:,.2f}")

"""
Evaluation on output:
=======================================================

Model Evaluation Report
=======================================================
The Deep Neural Network (DNN) successfully learned non-linear spatial and socioeconomic trends within the 
California housing market. The training run shows steady optimization with no immediate signs of catastrophic 
overfitting.

1. Training Convergence & Optimization

• Loss Progression: The training loss decreased consistently across the 2,000 epochs:
	• Epoch 400: 0.2727
	• Epoch 2000: 0.2210 (A 19% reduction in loss over the final 1,600 epochs).
• Convergence Behavior: The loss reduction slowed down over time (e.g., drops of ~0.026 between epochs 400–800, 
but only ~0.007 between epochs 1600–2000). This indicates the optimizer reached a stable local minimum, though 
a tiny learning rate or scheduler could squeeze out marginal gains.

2. Generalization Performance

• Test MSE (0.2697) vs. Final Train Loss (0.2210): The model performs slightly worse on the unseen test split 
than on the training data. This delta is normal for neural networks. It indicates mild, healthy overfitting 
but confirms the model retains strong generalizability to new data.
• Predictive Accuracy (MAE): The average absolute mistake is $35,397.36. Given that the median house value 
in this dataset typically ranges between $100,000 and $500,000, a typical error of ~$35k represents a 
highly competitive and commercially viable prediction margin.

3. Feature Importance & Layer Input Weight

The mean absolute weight magnitudes entering the first hidden layer reveal what features the network prioritizes during its initial data-routing stage:

Feature	Importance Rank	Mean Weight Magnitude	Interpretation
AveOccup (Average Occupancy)	#1	0.6873	Crucial bottleneck feature; changes heavily dictate density-based price fluctuations.
Latitude (Geographic)	#2	0.3908	Strong indicator of the North-South price divide in California (e.g., Coastal vs. Inland).
Longitude (Geographic)	#3	0.3143	Maps the East-West proximity to the coastline, which is heavily tied to premium valuations.
AveRooms (Rooms per Household)	#4	0.2797	Primary structural metric indicating house size.
AveBedrms (Bedrooms per Household)	#5	0.2479	Secondary structural metric mapping configuration.
MedInc (Median Income)	#6	0.2334	Socioeconomic anchor.
HouseAge (Age of Structure)	#7	0.2237	Structural depreciation/historical value baseline.
Population (Block Population)	#8	0.1524	Least impactful factor; general block size matters less than occupancy density.

Takeaway: The network leans heavily on geographic and occupancy data in its initial connections to map non-linear 
regional clusters before passing features deeper into the architecture.

4. Custom Inference Validation

• Scenario: Evaluating a single neighborhood block group (Moderate income at $45k, average density, 
located around Los Angeles coordinates).
• Output: $263,492.08
• Sanity Check: This prediction aligns closely with historical regional medians for middle-income Los Angeles 
suburban pockets in this dataset. The pipeline properly handles data pipeline transforms (Raw Array \(\rightarrow \) Standard Scaling \(\rightarrow \) Tensor \(\rightarrow \) DNN Pass \(\rightarrow \) Currency Unscaling) without truncating or producing erratic outliers.


"""

