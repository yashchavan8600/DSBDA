# Data Preprocessing and Data Normalization using Python

import pandas as pd
import numpy as np

# sklearn preprocessing tools
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# STEP 2: Load Dataset

# Read CSV file using pandas
data = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows of dataset
print("First 5 Rows of Dataset:")
print(data.head())

# STEP 3: Check Dataset Information

# Display number of rows and columns
print("\nShape of Dataset:")
print(data.shape)

# Display complete information about dataset
print("\nDataset Information:")
print(data.info())

# Display column names
print("\nColumn Names:")
print(data.columns)

# STEP 4: Data Preprocessing
# 4.1 Check Missing Values

print("\nMissing Values in Dataset:")
print(data.isnull().sum())

# 4.2 Statistical Summary

print("\nStatistical Summary:")
print(data.describe())

# 4.3 Variable Description

print("\nData Types of Variables:")
print(data.dtypes)

# STEP 5: Handle Missing Values

# Fill missing Age values with mean age
data['Age'] = data['Age'].fillna(data['Age'].mean())

# Fill missing Embarked values with mode
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Fill missing Cabin values with "Unknown"
data['Cabin'] = data['Cabin'].fillna("Unknown")

print("\nMissing Values After Handling:")
print(data.isnull().sum())

# STEP 6: Data Formatting and Type Conversion

# Convert Survived column into category type
data['Survived'] = data['Survived'].astype('category')

# Convert Pclass into category type
data['Pclass'] = data['Pclass'].astype('category')

print("\nUpdated Data Types:")
print(data.dtypes)

# STEP 7: Convert Categorical Variables into Numerical

# Create LabelEncoder object
encoder = LabelEncoder()

# Convert Sex column into numeric values
# male = 1, female = 0
data['Sex'] = encoder.fit_transform(data['Sex'])

# Convert Embarked column into numeric values
data['Embarked'] = encoder.fit_transform(data['Embarked'])

print("\nDataset After Encoding:")
print(data.head())

# STEP 8: Data Normalization
# Create StandardScaler object
scaler = StandardScaler()

# Normalize Age and Fare columns
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

print("\nDataset After Normalization:")
print(data[['Age', 'Fare']].head())

# STEP 9: Final Dataset Information

print("\nFinal Dataset Information:")
print(data.info())

print("\nFinal Processed Dataset:")
print(data.head())