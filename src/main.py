# Standard library imports
import json
import os

# Third-party library imports
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE  # For oversampling the minority class
from lightgbm import LGBMClassifier
from openpyxl import Workbook
from sklearn.ensemble import (
     RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load data from 'data.csv'

data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data.csv'))

# Data preprocessing

# Replace '?' in the 'Bare Nuclei' column with NaN
data['Bare Nuclei'] = data['Bare Nuclei'].replace('?', np.nan).astype(float)

#Fill empty records in column 'Bare Nuclei' with mean
# data['Bare Nuclei'] = data['Bare Nuclei'].fillna(data['Bare Nuclei'].mean())

#Fill empty records in column 'Bare Nuclei' with mean in the Class
# data['Bare Nuclei'] = data.groupby('Class')['Bare Nuclei'].transform(lambda x: x.fillna(x.mean()))


# # Using KNN to fill missing values
# Selecting features without irrelevant columns like 'Sample code number' and 'Class'
from sklearn.impute import KNNImputer
features = data.drop(columns=['Sample code number', 'Class'])
# Creating an imputer using KNN with k=5
imputer = KNNImputer(n_neighbors=5)
# Filling missing values and replacing the filled data in the main DataFrame
data_filled = imputer.fit_transform(features)
data.update(pd.DataFrame(data_filled, columns=features.columns))


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

# results = {}
# # Practise 1 Start
# # List of models
# models = {
#     "CART": DecisionTreeClassifier(),
#     "C4.5": DecisionTreeClassifier(criterion='gini'),
#     "AdaBoost": AdaBoostClassifier(algorithm="SAMME"),
#     "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
#     "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
#     "LightGBM":lgb.LGBMClassifier(max_depth=6, min_gain_to_split=0.1),
#     "ExtraTrees": ExtraTreesClassifier(),
#     "GradientBoosting": GradientBoostingClassifier()
# }

# # Train models and store results
# for model_name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     results[model_name] = {
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Recall": recall_score(y_test, y_pred, pos_label=1),
#         "Precision": precision_score(y_test, y_pred, pos_label=1),
#         "F1 Score": f1_score(y_test, y_pred, pos_label=1),
#         "Classification Report": classification_report(y_test, y_pred, output_dict=True)
#     }


# # Save results to a JSON file
# with open("model_results.json", "w") as f:
#     json.dump(results, f, indent=4)

# # Create and save a comparison table to Excel
# results_df = pd.DataFrame({
#     "Model": list(results.keys()),
#     "Accuracy": [res["Accuracy"] for res in results.values()],
#     "Recall": [res["Recall"] for res in results.values()],
#     "Precision": [res["Precision"] for res in results.values()],
#     "F1 Score": [res["F1 Score"] for res in results.values()]
# })
# results_df.to_excel("comparison_results.xlsx", index=False)

# # Visualize decision trees
# # Display and save the chart for the CART model
# plt.figure(figsize=(20, 10))
# plot_tree(models["CART"], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
# plt.savefig("cart_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the C4.5 model
# # Assumes C4.5 is similar to a decision tree and visualizes it in the same way
# plt.figure(figsize=(20, 10))
# plot_tree(models["C4.5"], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
# plt.savefig("c45_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the RandomForest model
# # Visualize one random tree from the RandomForest
# plt.figure(figsize=(20, 10))
# plot_tree(models["RandomForest"].estimators_[0], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
# plt.savefig("random_forest_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the ExtraTrees model
# # Visualize one random tree from the ExtraTrees
# plt.figure(figsize=(20, 10))
# plot_tree(models["ExtraTrees"].estimators_[0], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
# plt.savefig("extra_trees_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the AdaBoost model
# # In AdaBoost, weak decision trees are used
# # Visualize one tree from AdaBoost
# plt.figure(figsize=(20, 10))
# plot_tree(models["AdaBoost"].estimators_[0], feature_names=X.columns, class_names=["benign", "malignant"], filled=True, rounded=True)
# plt.savefig("adaboost_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the XGBoost model
# # XGBoost is not a decision tree; use its dedicated tool for visualization
# plt.figure(figsize=(20, 10))
# xgb.plot_tree(models["XGBoost"], num_trees=0)
# plt.savefig("xgboost_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the LightGBM model
# # LightGBM is similar to XGBoost; use the same tool for tree visualization
# plt.figure(figsize=(20, 10))
# lgb.plot_tree(models["LightGBM"], tree_index=0)
# plt.savefig("lgbm_tree_visualization.svg", format="svg")
# # plt.show()

# # Display and save the chart for the Gradient Boosting model
# # In GradientBoosting, weak trees similar to AdaBoost are used
# plt.figure(figsize=(20, 10))
# plot_tree(models["GradientBoosting"].estimators_[0, 0],  # Use the first tree in the estimators_ set
#           feature_names=X.columns, 
#           filled=True, 
#           rounded=True)
# plt.savefig("gradient_boosting_tree_visualization.svg", format="svg")
# # plt.show()

# # Save decision tree rules to text files for tree-based models
# # Save CART model decision tree rules
# cart_rules = export_text(models["CART"], feature_names=list(X.columns))
# with open("cart_tree_rules.txt", "w") as f:
#     f.write(cart_rules)

# # Save C4.5 model decision tree rules
# c45_rules = export_text(models["C4.5"], feature_names=list(X.columns))
# with open("c45_tree_rules.txt", "w") as f:
#     f.write(c45_rules)

# # Save decision rules for the first tree in the RandomForest model
# random_forest_rules = export_text(models["RandomForest"].estimators_[0], feature_names=list(X.columns))
# with open("random_forest_tree_rules.txt", "w") as f:
#     f.write(random_forest_rules)

# # Save decision rules for the first tree in the ExtraTrees model
# extra_trees_rules = export_text(models["ExtraTrees"].estimators_[0], feature_names=list(X.columns))
# with open("extra_trees_tree_rules.txt", "w") as f:
#     f.write(extra_trees_rules)

# # Save decision rules for the first tree in the AdaBoost model
# adaboost_rules = export_text(models["AdaBoost"].estimators_[0], feature_names=list(X.columns))
# with open("adaboost_tree_rules.txt", "w") as f:
#     f.write(adaboost_rules)

# # Save decision rules for the first tree in the Gradient Boosting model
# gradient_boosting_rules = export_text(models["GradientBoosting"].estimators_[0, 0], feature_names=list(X.columns))
# with open("gradient_boosting_tree_rules.txt", "w") as f:
#     f.write(gradient_boosting_rules)
# #Practise 1 End

# #Practise 2 Start
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# import pandas as pd
# import numpy as np
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
# from mlxtend.plotting import plot_decision_regions

# # Initialize dictionaries for results and models
# results = {}
# model_rules = {}
# models = {}

# # Add KNN models for k=2 to k=10
# for k in range(2, 11):
#     models[f"KNN_k={k}"] = KNeighborsClassifier(n_neighbors=k)

# # Add SVM models with different kernels
# for kernel in ["linear", "poly", "rbf", "sigmoid"]:
#     models[f"SVM_{kernel}"] = SVC(kernel=kernel, probability=True)

# # Add Naive Bayes
# models["NaiveBayes"] = GaussianNB()

# # Train and evaluate models
# for model_name, model in models.items():
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     # Save performance metrics
#     results[model_name] = {
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Recall": recall_score(y_test, y_pred, pos_label=1),
#         "Precision": precision_score(y_test, y_pred, pos_label=1),
#         "F1 Score": f1_score(y_test, y_pred, pos_label=1),
#         "Classification Report": classification_report(y_test, y_pred, output_dict=True)
#     }

#     # Save rules
#     if model_name.startswith("KNN"):
#         knn_neighbors = model.kneighbors(X_test, return_distance=True)
#         model_rules[model_name] = [
#             {
#                 "Test_Instance_Index": i,
#                 "Nearest_Neighbors_Indices": neighbors.tolist(),
#                 "Distances": distances.tolist(),
#                 "Predicted_Class": model.predict([X_test.iloc[i]])[0]
#             }
#             for i, (distances, neighbors) in enumerate(zip(*knn_neighbors))
#         ]
#     elif model_name.startswith("SVM"):
#         model_rules[model_name] = {
#             "Support_Vectors": model.support_vectors_.tolist(),
#             "Coefficients": model.dual_coef_.tolist(),
#             "Intercept": model.intercept_.tolist()
#         }
#     elif model_name == "NaiveBayes":
#         model_rules[model_name] = {
#             "Class_Priors": model.class_prior_.tolist(),
#             "Class_Means": {f"Class_{i}": means.tolist() for i, means in enumerate(model.theta_)},
#             "Class_Variances": {f"Class_{i}": vars.tolist() for i, vars in enumerate(model.var_)}
#         }

# # Save results to JSON
# with open("model_results.json", "w") as f:
#     json.dump(results, f, indent=4)

# # Save rules to JSON
# def convert_to_serializable(obj):
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, (np.int64, np.int32)):
#         return int(obj)
#     elif isinstance(obj, (np.float64, np.float32)):
#         return float(obj)
#     elif isinstance(obj, dict):
#         return {key: convert_to_serializable(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_serializable(item) for item in obj]
#     else:
#         return obj

# serializable_rules = convert_to_serializable(model_rules)
# with open("model_rules.json", "w") as f:
#     json.dump(serializable_rules, f, indent=4)

# # Save comparison table to Excel
# results_df = pd.DataFrame({
#     "Model": list(results.keys()),
#     "Accuracy": [res["Accuracy"] for res in results.values()],
#     "Recall": [res["Recall"] for res in results.values()],
#     "Precision": [res["Precision"] for res in results.values()],
#     "F1 Score": [res["F1 Score"] for res in results.values()]
# })
# results_df.to_excel("comparison_results.xlsx", index=False)

# # Visualize confusion matrix and decision boundaries
# labels = ["Benign", "Malignant"]

# def plot_confusion_matrix(y_true, y_pred, model_name):
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
#     plt.ylabel("True Label")
#     plt.xlabel("Predicted Label")
#     plt.title(f"Confusion Matrix for {model_name}")
#     plt.savefig(f"{model_name}_confusion_matrix.svg", format="svg")

# for model_name, model in models.items():
#     y_pred = model.predict(X_test)
#     plot_confusion_matrix(y_test, y_pred, model_name)

#     # Plot decision boundary for 2D data
#     if X_test.shape[1] == 2:
#         try:
#             plot_decision_regions(X_test.values, y_test.values, clf=model, legend=2)
#             plt.title(f"Decision Boundary for {model_name}")
#             plt.savefig(f"{model_name}_decision_boundary.svg", format="svg")
#         except Exception as e:
#             print(f"Could not plot decision boundary for {model_name}: {e}")

# Practice 3
# Define individual base models
cart = DecisionTreeClassifier(random_state=42)
c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
rf = RandomForestClassifier(random_state=42)
gbm = GradientBoostingClassifier(random_state=42)
ada = AdaBoostClassifier(random_state=42)
xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
lgbm = LGBMClassifier(random_state=42)

# Define additional models for stacking
svc = SVC(probability=True, random_state=42)
knn = KNeighborsClassifier()
logreg = LogisticRegression()

# Create ensemble methods
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf), ('gbm', gbm), ('ada', ada), ('xgb', xgb), ('lgbm', lgbm)
    ],
    voting='soft'
)

bagging_clf = BaggingClassifier(estimator=rf, n_estimators=10, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[
        ('cart', cart),
        ('svc', svc),
        ('knn', knn),
        ('rf', rf)
    ],
    final_estimator=logreg
)

# Dictionary of all models to evaluate
models = {
    'CART': cart,
    'C4.5': c45,
    'Random Forest': rf,
    'Gradient Boosting': gbm,
    'AdaBoost': ada,
    'XGBoost': xgb,
    'LightGBM': lgbm,
    'Voting Ensemble': voting_clf,
    'Bagging Ensemble': bagging_clf,
    'Stacking Ensemble': stacking_clf
}

# Evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'Classification Report': classification_rep
    })


# Save results to JSON and Excel files
results_df = pd.DataFrame(results)
results_json_path = os.path.join("..", "6- Ensemble Methods - Without Checking Data Imbalance - Using KNN", "ensemble_model_results.json")
results_excel_path = os.path.join("..", "6- Ensemble Methods - Without Checking Data Imbalance - Using KNN", "ensemble_comparison_results.xlsx")

os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
results_df.to_json(results_json_path, orient='records', indent=4)
results_df.to_excel(results_excel_path, index=False)

print("Evaluation completed. Results saved.")