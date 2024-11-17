import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from 'data.csv' without a header and assign custom column names
data = pd.read_csv('data.csv', header=None, names=[
    'Sample code number', 'Clump Thickness', 'Uniformity of Cell Size',
    'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
    'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'
])

# Convert columns (except 'Sample code number') to numeric, replacing non-convertible values with NaN
for column in data.columns[1:]:  # Skip the first column ('Sample code number')
    data[column] = pd.to_numeric(data[column], errors='coerce')
    if data[column].isnull().sum() > 0:
        print(f"Invalid values in '{column}' replaced with NaN: {data[column].isnull().sum()} entries")
    
    if column != 'Class':
        # Replace NaN in feature columns with their mean
        data[column].fillna(data[column].mean(), inplace=True)
    else:
        # Replace NaN in 'Class' with the mode (most frequent value)
        data[column].fillna(data[column].mode()[0], inplace=True)

# Drop 'Sample code number' column for correlation analysis
data.drop(columns=['Sample code number'], inplace=True)

# Check data types after cleaning
print("\nData types after cleaning:\n", data.dtypes)

# Distribution plot of the 'Class' column
sns.countplot(x='Class', data=data)
plt.title('Tumor Type Distribution (Class)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Heatmap of correlation coefficients
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Feature Correlations')
plt.show()