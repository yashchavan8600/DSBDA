# STEP 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# STEP 2: Load Dataset
df = pd.read_csv("diabetes.csv")

print("First 5 Rows:")
print(df.head())

# STEP 3: Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# STEP 4: Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# STEP 5: Replace Invalid Zero Values with NaN
# In this dataset, zero values are invalid in some columns
columns = ['Glucose',
           'BloodPressure',
           'SkinThickness',
           'Insulin',
           'BMI']

for col in columns:
    df[col] = df[col].replace(0, np.nan)

print("\nMissing Values After Replacing 0 with NaN:")
print(df.isnull().sum())

# STEP 6: Fill Missing Values using Mean
for col in columns:
    df[col] = df[col].fillna(df[col].mean())

print("\nDataset After Handling Missing Values:")
print(df.head())

# STEP 7: Detect Outliers using Z-Score
numeric_columns = df.select_dtypes(include=np.number)

z_scores = np.abs(stats.zscore(numeric_columns))

outliers = (z_scores > 3)

print("\nOutliers Detected:")
print(outliers.sum())

# STEP 8: Handle Outliers
for col in numeric_columns.columns:
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    median = df[col].median()
    
    df[col] = np.where(df[col] > upper,
                       median,
                       df[col])
    
    df[col] = np.where(df[col] < lower,
                       median,
                       df[col])

print("\nDataset After Handling Outliers:")
print(df.head())

# STEP 9: Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Plot correlation heatmap

import seaborn as sns

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)

plt.title("Correlation Matrix")
plt.show()

# STEP 10: Discretization (Binning)
# Create bins for Age
bins = [20, 30, 40, 50, 60, 80]

labels = ['20-30',
          '31-40',
          '41-50',
          '51-60',
          '61-80']

df['Age_Group'] = pd.cut(df['Age'],
                         bins=bins,
                         labels=labels)

print("\nAge Groups After Binning:")
print(df[['Age', 'Age_Group']].head())

# STEP 11: Normalization using MinMaxScaler
scaler = MinMaxScaler()

normalized_columns = ['Glucose',
                      'BloodPressure',
                      'BMI']

df[normalized_columns] = scaler.fit_transform(
                            df[normalized_columns]
                        )

print("\nDataset After Normalization:")
print(df.head())

# STEP 12: Final Dataset Information
print("\nFinal Dataset Info:")
print(df.info())

print("\nFinal Statistical Summary:")
print(df.describe())