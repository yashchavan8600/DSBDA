# Data Visualization using Iris Dataset
# STEP 1: Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2: Load Dataset
df = pd.read_csv("iris.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:")
print(df.head())

# STEP 3: Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nColumn Names:")
print(df.columns)

print("\nStatistical Summary:")
print(df.describe())

# STEP 4: Features and Their Types
print("\nFeature Names and Data Types:")

print(df.dtypes)

# STEP 5: Histograms for Each Feature
df.hist(figsize=(10,8))

plt.suptitle("Histogram of Iris Dataset Features")

plt.show()

# STEP 6: Boxplots for Each Feature
plt.figure(figsize=(12,8))

for i, column in enumerate(df.columns[:-1], 1):
    
    plt.subplot(2, 2, i)
    
    sns.boxplot(y=df[column])
    
    plt.title(f"Boxplot of {column}")

plt.tight_layout()

plt.show()

# STEP 7: Identify Outliers
print("\nChecking Outliers Using IQR Method:\n")

for column in df.columns[:-1]:
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]
    
    print(f"{column} Outliers:")
    
    print(outliers[[column]])
    
    print("-----------------------------------")