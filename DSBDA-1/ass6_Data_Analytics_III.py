# Naive Bayes Classification using Iris Dataset
# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# STEP 2: Load Dataset
df = pd.read_csv("iris.csv")

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
X = df.drop('species', axis=1)

# Dependent Variable
y = df['species']

# STEP 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# STEP 6: Create Naive Bayes Model
model = GaussianNB()

# Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed")

# STEP 7: Prediction
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

# STEP 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# STEP 9: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# STEP 10: Error Rate
error_rate = 1 - accuracy

print("Error Rate:", error_rate)

# STEP 11: Precision and Recall
precision = precision_score(
    y_test,
    y_pred,
    average='macro'
)

recall = recall_score(
    y_test,
    y_pred,
    average='macro'
)

print("Precision:", precision)

print("Recall:", recall)

# STEP 12: Visualization
plt.scatter(
    X_test.iloc[:,0],
    X_test.iloc[:,1],
    c=pd.factorize(y_pred)[0]
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.title("Naive Bayes Classification")

plt.show()