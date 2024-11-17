import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score

# Load the dataset
data = pd.read_csv("data.csv")

# Replace '?' with NaN in 'Bare Nuclei' and convert to float
data['Bare Nuclei'] = data['Bare Nuclei'].replace('?', np.nan).astype(float)

# Fill missing values in 'Bare Nuclei' with the mean
data['Bare Nuclei'] = data['Bare Nuclei'].fillna(data['Bare Nuclei'].mean())

#Fill empty records in column 'Bare Nuclei' with mean in the Class
# data['Bare Nuclei'] = data.groupby('Class')['Bare Nuclei'].transform(lambda x: x.fillna(x.mean()))


# Using KNN to fill missing values
# Selecting features without irrelevant columns like 'Sample code number' and 'Class'
# from sklearn.impute import KNNImputer
# features = data.drop(columns=['Sample code number', 'Class'])
# # Creating an imputer using KNN with k=5
# imputer = KNNImputer(n_neighbors=5)
# # Filling missing values and replacing the filled data in the main DataFrame
# data_filled = imputer.fit_transform(features)
# data.update(pd.DataFrame(data_filled, columns=features.columns))


# Normalize the features (excluding 'Sample code number' and 'Class')
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Sample code number', 'Class']))

# Separate features and target
X = data.drop(columns=['Sample code number', 'Class'])
y = data['Class']

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Display Decision Tree rules
rules = export_text(dt, feature_names=list(X.columns))
print("Decision Tree Rules:\n", rules)

# Evaluate Decision Tree model
y_pred_dt = dt.predict(X_test)
dt_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_dt),
    "Recall": recall_score(y_test, y_pred_dt, pos_label=2),
    "Precision": precision_score(y_test, y_pred_dt, pos_label=2),
    "F1 Score": f1_score(y_test, y_pred_dt, pos_label=2),
}
print("\nDecision Tree Metrics:")
print(dt_metrics)
print("Classification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate Random Forest model
y_pred_rf = rf.predict(X_test)
rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf, pos_label=2),
    "Precision": precision_score(y_test, y_pred_rf, pos_label=2),
    "F1 Score": f1_score(y_test, y_pred_rf, pos_label=2),
}
print("\nRandom Forest Metrics:")
print(rf_metrics)
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# Compare models
comparison_results = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest"],
    "Accuracy": [dt_metrics["Accuracy"], rf_metrics["Accuracy"]],
    "Recall": [dt_metrics["Recall"], rf_metrics["Recall"]],
    "Precision": [dt_metrics["Precision"], rf_metrics["Precision"]],
    "F1 Score": [dt_metrics["F1 Score"], rf_metrics["F1 Score"]],
})
print("\nComparison Table:\n", comparison_results)
