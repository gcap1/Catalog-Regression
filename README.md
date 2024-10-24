# 📊 E-commerce Dataset Analysis

## Overview

The **E-commerce dataset** pertains to a catalog firm specializing in selling software. The firm recently updated its catalog, resulting in 2000 customer purchases. The goal is to develop a predictive model for estimating customer spending based on this data.

The dataset is provided in CSV format and includes information on 2000 purchases with various attributes such as customer demographics, purchase behavior, and transaction details.

### Key Objectives
- Build a model to predict customer spending amount.
- Analyze the relationship between different customer and transaction features to gain insights into spending behavior.

## Dataset Information

The dataset consists of **2000 rows and 25 columns**. Below is a sample output:

| sequence_number | US | source_a | source_c | source_b | source_d | source_e | source_m | source_o | source_h | ... | source_x | source_w | Freq |
|-----------------|----|----------|----------|----------|----------|----------|----------|----------|----------|-----|----------|----------|------|
| 1               | 1  | 0        | 0        | 1        | 0        | 0        | 0        | 0        | 0        | ... | 0        | 0        | 2    |
| 2               | 1  | 0        | 0        | 0        | 0        | 1        | 0        | 0        | 0        | ... | 0        | 0        | 0    |
| 3               | 1  | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | ... | 0        | 0        | 2    |
| 4               | 1  | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0        | ... | 0        | 0        | 1    |
| 5               | 1  | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0        | ... | 0        | 0        | 1    |

## Steps for Analysis

```python
# Step 1: Load the Dataset
import pandas as pd

# Load the dataset
tayko = pd.read_csv("Tayko.csv")
tayko.head()

# Step 2: Inspect Dataset Columns
# Print out headers (in the order)
print(tayko.columns.values)

# Step 3: Inspect Dataset Dimensions
# Output the shape of the dataset
print(tayko.shape)

# Step 4: Data Visualization
from matplotlib import pyplot as plt

# Create a scatter plot
plt.scatter(tayko['Freq'], tayko['Spending'])
plt.xlabel("Frequency")
plt.ylabel("Spending")
plt.title("Spending vs Frequency")
plt.show()

# Step 5: Fit a Multiple Linear Regression Model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Choose predictors to construct attribute matrix
select_index = [1, 17, 18, 20, 21, 22]
tayko_X = tayko.iloc[:, select_index]  # Select predictor columns
tayko_y = tayko.iloc[:, -1]  # Target variable (Spending)

# Partition data
X_train, X_valid, y_train, y_valid = train_test_split(tayko_X, tayko_y, test_size=0.25, random_state=1)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model using the training set

# Print Fitted Coefficients
np.set_printoptions(precision=4, suppress=True)  # Set precision
print("Coefficients:", model.coef_)
print('Intercept: %.4f' % model.intercept_)  # Print intercept

# Make Predictions and Evaluate RMSE
# Make predictions on validation set
y_pred = model.predict(X_valid)

# Compute RMSE for validation set
rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))
print('Validation RMSE: ', '%.4f' % rmse)

# Make predictions on training set
y_pred_train = model.predict(X_train)

# Compute RMSE for training set
rmse_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
print('Train RMSE: ', '%.4f' % rmse_train)

# Fitted Predictive Equation
# Construct the fitted predictive equation
print("Fitted Predictive Equation: ")
print(f"Spending = {model.intercept_:.4f} + " + 
      " + ".join([f"{coef:.4f} * {col}" for coef, col in zip(model.coef_, tayko.columns[select_index])]))
