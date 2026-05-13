# Basic Statistics - Employee Salary Dataset
# STEP 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: Create Employee Dataset
data = {
    'Employee_Name': ['Amit', 'Riya', 'Raj', 'Sneha',
                      'Karan', 'Pooja', 'Arjun', 'Neha'],
    
    'Department': ['IT', 'HR', 'IT', 'Finance',
                   'HR', 'Finance', 'IT', 'HR'],
    
    'Age': [25, 30, 28, 35, 32, 29, 26, 31],
    
    'Salary': [50000, 45000, 52000, 60000,
               47000, 58000, 51000, 48000]
}

df = pd.DataFrame(data)

print("Employee Dataset:")
print(df)

# STEP 3: Summary Statistics
print("\nSummary Statistics:")
print(df.describe())
# STEP 4: Group Salary by Department
grouped = df.groupby('Department')['Salary']

print("\nSalary Statistics Grouped by Department:")
print(grouped.describe())

# STEP 5: Create List of Salary Values
salary_list = grouped.apply(list)

print("\nList of Salary Values by Department:")
print(salary_list)

# STEP 6: Mean Salary by Department
mean_salary = grouped.mean()

print("\nMean Salary by Department:")
print(mean_salary)

# STEP 7: Visualization
mean_salary.plot(kind='bar')

plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.title("Average Salary by Department")

plt.show()