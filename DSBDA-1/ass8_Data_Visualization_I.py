# Data Visualization using Titanic Dataset
# STEP 1: Import Required Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# STEP 2: Load Titanic Dataset
titanic = sns.load_dataset('titanic')

# Display first 5 rows
print("First 5 Rows of Titanic Dataset:")
print(titanic.head())

# STEP 3: Dataset Information
print("\nDataset Shape:")
print(titanic.shape)

print("\nDataset Information:")
print(titanic.info())

print("\nStatistical Summary:")
print(titanic.describe())

# STEP 4: Check Missing Values
print("\nMissing Values:")
print(titanic.isnull().sum())

# STEP 5: Histogram of Fare Column
plt.figure(figsize=(10,6))

sns.histplot(
    titanic['fare'],
    bins=30,
    kde=True
)

plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.title("Distribution of Titanic Ticket Fare")

plt.show()