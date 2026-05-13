# Linear Regression using Boston Housing Dataset
# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# STEP 2: Load Dataset
# Make sure BostonHousing.csv is in same folder
df = pd.read_csv("BostonHousing.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# STEP 3: Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# STEP 4: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# STEP 5: Handle Missing Values
# Fill missing values with mean
df = df.fillna(df.mean())

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# STEP 6: Select Features and Target Variable
# Independent Variables
X = df.drop('medv', axis=1)

# Dependent Variable
y = df['medv']

# STEP 7: Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# STEP 8: Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed")

# STEP 9: Predict House Prices
y_pred = model.predict(X_test)

print("\nPredicted House Prices:")
print(y_pred[:10])

# STEP 10: Model Evaluation
mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error (MSE):", mse)

print("R2 Score:", r2)

# STEP 11: Visualization - Scatter Plot
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted House Prices")

plt.show()

# STEP 12: Visualization - Line Graph
plt.figure(figsize=(10,6))

plt.plot(y_test.values, label='Actual Prices')

plt.plot(y_pred, label='Predicted Prices')

plt.xlabel("Data Points")
plt.ylabel("House Prices")

plt.title("Actual and Predicted House Prices")

plt.legend()

plt.show()