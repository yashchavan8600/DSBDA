# Logistic Regression using Social_Network_Ads Dataset
# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# STEP 2: Load Dataset
df = pd.read_csv("Social_Network_Ads.csv")

print("First 5 Rows:")
print(df.head())

# STEP 3: Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 4: Select Features and Target Variable
# Independent Variables
X = df[['Age', 'EstimatedSalary']]

# Dependent Variable
y = df['Purchased']

# STEP 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# STEP 6: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# STEP 7: Train Logistic Regression Model
model = LogisticRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed")

# STEP 8: Prediction
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

# STEP 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# STEP 10: Extract TP, FP, TN, FN
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTrue Positive (TP):", TP)
print("False Positive (FP):", FP)
print("True Negative (TN):", TN)
print("False Negative (FN):", FN)

# STEP 11: Performance Metrics
accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("Error Rate:", error_rate)

print("Precision:", precision)

print("Recall:", recall)

# STEP 12: Visualization
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")

plt.title("Logistic Regression Classification")

plt.show()