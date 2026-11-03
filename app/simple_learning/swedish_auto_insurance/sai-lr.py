import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Load the dataset from the URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv"
df = pd.read_csv(url, header=None, names=['Claims', 'TotalPayment'])

# 2. Split into features (X) and target variable (y)
X_df = df['Claims'] 
y_df = df['TotalPayment']

# 3. Convert pandas dataframes to PyTorch tensors (float32 is standard for PyTorch)
X = torch.tensor(X_df.values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_df.values, dtype=torch.float32).unsqueeze(1) # Reshape to (N, 1)

# 4. Define the Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 5. Define Loss and Optimizer
criterion = nn.MSELoss()
# Adam works much better out of the box here without needing extreme feature scaling
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 6. Training Loop
epochs = 1500
for epoch in range(epochs):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 7. Output Final Learned Parameters
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"\nLearned Equation: TotalPayment = {w:.2f} * Claims + {b:.2f}")

#8. For testing model -> Switch model to evaluation mode
model.eval()

# Turn off gradients for evaluation
with torch.no_grad():
    predictions = model(X)
    # 1. Define raw input
    test_claims = 25.0
    
    # 2. Convert to a 2D PyTorch tensor: shape [[25.0]]
    test_tensor = torch.tensor([test_claims], dtype=torch.float32)
    
    # 3. Get prediction
    prediction = model(test_tensor)
    
    print(f"Predicted Total Payment for claims={test_claims} : {prediction.item():.2f}")

# 9. Calculate errors using PyTorch functional tools
mse_loss = criterion(predictions, y).item()
rmse_loss = np.sqrt(mse_loss)

# calculate loss and error :
mae_criterion = nn.L1Loss()
mae_loss = mae_criterion(predictions, y).item()

print(f"Squared Loss (MSE): {mse_loss:.2f}")
print(f"Average Error (RMSE): {rmse_loss:.2f} thousand Kronor")
print(f"Average Absolute Error (MAE): {mae_loss:.2f} thousand Kronor")

"""

Output:

Epoch [500/1500], Loss: 1262.9137
Epoch [1000/1500], Loss: 1250.8319
Epoch [1500/1500], Loss: 1250.7421

Learned Equation: TotalPayment = 3.41 * Claims + 19.98
Predicted Total Payment for claims=25.0 : 105.34
Squared Loss (MSE): 1250.74
Average Error (RMSE): 35.37 thousand Kronor
Average Absolute Error (MAE): 28.48 thousand Kronor
==============================================================

Model Performance Summary

The model has successfully converged to the optimal linear solution for the Swedish Auto Insurance dataset.
The metrics show that while the squared loss number looks large due to the mathematical squaring effect, 
the actual real-world predictive power of the model is quite solid for a single-variable baseline.

Detailed Metrics Breakdown
==============================================================

Metric	Value	Interpretation
Final Training Loss (MSE)	1,250.74	The standard optimization target. It drops from 1,262.91 at 
Epoch 500 down to 1,250.74 by Epoch 1500, indicating the model stabilized and found its absolute 
"best fit" line.
Average Error (RMSE)	35.37 thousand Kronor	The standard deviation of the residuals. 
It penalizes larger outlier errors more heavily. On average, predictions deviate by about 35.37 units.
Average Absolute Error (MAE)	28.48 thousand Kronor	The true, unweighted average mistake.
 On a typical prediction, your model is only off by 28.48 thousand Kronor.


Equation & Practical Application

• Learned Formula: ({TotalPayment} = 3.41 * {Claims} + 19.98)
• Base Cost (Intercept): Even with (0) claims, the model establishes a baseline fixed cost 
profile of 19.98 thousand Kronor.
• Variable Cost (Slope): For every single additional claim filed, the total 
insurance payout is expected to rise by 3.41 thousand Kronor.

Test Scenario Validation

• Input: 25.0 Claims
• Math: ((3.41 * 25.0) + 19.98 = 85.25 + 19.98 = 105.23) (slight variation due to 
                                                                 float precision rounding)
• Prediction: 105.34 thousand Kronor
Looking closely at your raw dataset, row 32 contains a real historical entry of 25 claims 
with a total payment of 69.2. Your model predicted 105.34. This ~36 unit variance aligns perfectly 
with our RMSE/MAE metrics and highlights how scattered individual insurance payouts can be.

Final Assessment

Verdict: Underfitting by Design (Excellent Linear Baseline)
The model cannot optimize any further than a loss of 1250.74. This is not a failure of the 
PyTorch code; it is the physical limitation of a straight line. Real-world insurance claims 
are heavily influenced by chaotic elements (such as catastrophic accidents vs. tiny fender-benders) 
that a single line cannot perfectly parse.



"""