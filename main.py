import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE  # For oversampling the minority class
from openpyxl import Workbook

# Load data from 'data.csv'
data = pd.read_csv("data.csv")

# Data preprocessing

# Replace '?' in the 'Bare Nuclei' column with NaN
data['Bare Nuclei'] = data['Bare Nuclei'].replace('?', np.nan).astype(float)

#Fill empty records in column 'Bare Nuclei' with mean
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


# Normalize features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Sample code number', 'Class']))

# Normalize features
data.columns = data.columns.str.replace(" ", "_")

# Separate features and target variable
X = data.drop(columns=['Sample_code_number', 'Class'])
y = data['Class']

# Convert class labels from 2 and 4 to 0 and 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Check for data imbalance
class_counts = np.bincount(y_encoded)
print(f"Class distribution before balancing: {dict(enumerate(class_counts))}")

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# Verify the class distribution in training and testing sets
train_class_counts = np.bincount(y_train)
test_class_counts = np.bincount(y_test)
print(f"Class distribution in training set: {dict(enumerate(train_class_counts))}")
print(f"Class distribution in test set: {dict(enumerate(test_class_counts))}")

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check class distribution after balancing
balanced_class_counts = np.bincount(y_train)
print(f"Class distribution after balancing: {dict(enumerate(balanced_class_counts))}")

# List of models
models = {
    "CART": DecisionTreeClassifier(),
    "C4.5": DecisionTreeClassifier(criterion='gini'),
    "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM":lgb.LGBMClassifier(max_depth=6, min_gain_to_split=0.1),
    "ExtraTrees": ExtraTreesClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

# Train models and store results
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred, pos_label=1),
        "Precision": precision_score(y_test, y_pred, pos_label=1),
        "F1 Score": f1_score(y_test, y_pred, pos_label=1),
        "Classification Report": classification_report(y_test, y_pred, output_dict=True)
    }


# Save results to a JSON file
with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Create and save a comparison table to Excel
results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [res["Accuracy"] for res in results.values()],
    "Recall": [res["Recall"] for res in results.values()],
    "Precision": [res["Precision"] for res in results.values()],
    "F1 Score": [res["F1 Score"] for res in results.values()]
})
results_df.to_excel("comparison_results.xlsx", index=False)

# Visualize decision trees
# Display and save the chart for the CART model
plt.figure(figsize=(20, 10))
plot_tree(models["CART"], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
plt.savefig("cart_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the C4.5 model
# Assumes C4.5 is similar to a decision tree and visualizes it in the same way
plt.figure(figsize=(20, 10))
plot_tree(models["C4.5"], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
plt.savefig("c45_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the RandomForest model
# Visualize one random tree from the RandomForest
plt.figure(figsize=(20, 10))
plot_tree(models["RandomForest"].estimators_[0], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
plt.savefig("random_forest_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the ExtraTrees model
# Visualize one random tree from the ExtraTrees
plt.figure(figsize=(20, 10))
plot_tree(models["ExtraTrees"].estimators_[0], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
plt.savefig("extra_trees_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the AdaBoost model
# In AdaBoost, weak decision trees are used
# Visualize one tree from AdaBoost
plt.figure(figsize=(20, 10))
plot_tree(models["AdaBoost"].estimators_[0], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
plt.savefig("adaboost_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the XGBoost model
# XGBoost is not a decision tree; use its dedicated tool for visualization
plt.figure(figsize=(20, 10))
xgb.plot_tree(models["XGBoost"], num_trees=0)
plt.savefig("xgboost_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the LightGBM model
# LightGBM is similar to XGBoost; use the same tool for tree visualization
plt.figure(figsize=(20, 10))
lgb.plot_tree(models["LightGBM"], tree_index=0)
plt.savefig("lgbm_tree_visualization.svg", format="svg")
# plt.show()

# Display and save the chart for the Gradient Boosting model
# In GradientBoosting, weak trees similar to AdaBoost are used
plt.figure(figsize=(20, 10))
plot_tree(models["GradientBoosting"].estimators_[0, 0],  # Use the first tree in the estimators_ set
          feature_names=X.columns, 
          filled=True, 
          rounded=True)
plt.savefig("gradient_boosting_tree_visualization.svg", format="svg")
# plt.show()

# Save decision tree rules to text files for tree-based models
# Save CART model decision tree rules
cart_rules = export_text(models["CART"], feature_names=list(X.columns))
with open("cart_tree_rules.txt", "w") as f:
    f.write(cart_rules)

# Save C4.5 model decision tree rules
c45_rules = export_text(models["C4.5"], feature_names=list(X.columns))
with open("c45_tree_rules.txt", "w") as f:
    f.write(c45_rules)

# Save decision rules for the first tree in the RandomForest model
random_forest_rules = export_text(models["RandomForest"].estimators_[0], feature_names=list(X.columns))
with open("random_forest_tree_rules.txt", "w") as f:
    f.write(random_forest_rules)

# Save decision rules for the first tree in the ExtraTrees model
extra_trees_rules = export_text(models["ExtraTrees"].estimators_[0], feature_names=list(X.columns))
with open("extra_trees_tree_rules.txt", "w") as f:
    f.write(extra_trees_rules)

# Save decision rules for the first tree in the AdaBoost model
adaboost_rules = export_text(models["AdaBoost"].estimators_[0], feature_names=list(X.columns))
with open("adaboost_tree_rules.txt", "w") as f:
    f.write(adaboost_rules)

# Save decision rules for the first tree in the Gradient Boosting model
gradient_boosting_rules = export_text(models["GradientBoosting"].estimators_[0, 0], feature_names=list(X.columns))
with open("gradient_boosting_tree_rules.txt", "w") as f:
    f.write(gradient_boosting_rules)