import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

import json

# Step 1: Load and preprocess the data
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'data.csv'))

# Replace '?' in 'Bare Nuclei' with NaN and convert to float
data['Bare Nuclei'] = data['Bare Nuclei'].replace('?', np.nan).astype(float)

# Use KNN to fill missing values
features = data.drop(columns=['Sample code number', 'Class'])
imputer = KNNImputer(n_neighbors=5)
data_filled = imputer.fit_transform(features)
data.update(pd.DataFrame(data_filled, columns=features.columns))

# Normalize features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Sample code number', 'Class']))
data[data.columns.difference(['Sample code number', 'Class'])] = data_scaled

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Class'])
X = data.drop(columns=['Sample code number', 'Class'])

# Cross-validation setup
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracies, recalls, precisions, f1_scores = [], [], [], []

# Initialize lists to store metrics
results = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, y_train)
    rf = grid_search.best_estimator_

    bagging_clf = BaggingClassifier(estimator=RandomForestClassifier(random_state=42), n_estimators=10, random_state=42)
    
    rf.fit(X_train, y_train)
    bagging_clf.fit(X_train, y_train)
    
    # Generate predictions on training and test sets
    rf_preds_train = rf.predict(X_train)
    rf_preds_test = rf.predict(X_test)
    bagging_preds_train = bagging_clf.predict(X_train)
    bagging_preds_test = bagging_clf.predict(X_test)
    
    # Combine predictions as new features
    X_train_meta = np.column_stack((X_train, rf_preds_train, bagging_preds_train))
    X_test_meta = np.column_stack((X_test, rf_preds_test, bagging_preds_test))
    
    # Train the meta-model
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(X_train_meta, y_train)
    
    # Evaluate meta-model
    meta_preds = meta_model.predict(X_test_meta)
    
    # Calculate performance metrics
    fold_metrics = {
        'Fold': fold,
        'Accuracy': accuracy_score(y_test, meta_preds),
        'Recall': recall_score(y_test, meta_preds),
        'Precision': precision_score(y_test, meta_preds),
        'F1 Score': f1_score(y_test, meta_preds)
    }
    results.append(fold_metrics)
    # Print metrics for the current fold
    print(f"Fold {fold}:")
    print(f"  Accuracy: {fold_metrics['Accuracy']:.6f}")
    print(f"  Recall: {fold_metrics['Recall']:.6f}")
    print(f"  Precision: {fold_metrics['Precision']:.6f}")
    print(f"  F1 Score: {fold_metrics['F1 Score']:.6f}")

# Aggregate results
aggregated_results = {
    'Average Accuracy': np.mean([r['Accuracy'] for r in results]),
    'Average Recall': np.mean([r['Recall'] for r in results]),
    'Average Precision': np.mean([r['Precision'] for r in results]),
    'Average F1 Score': np.mean([r['F1 Score'] for r in results])
}

# Save results to JSON file
results_path = os.path.join("..", "cv_results.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, 'w') as f:
    json.dump({'Folds': results, 'Aggregated': aggregated_results}, f, indent=4)

print("Cross-validation results saved to:", results_path)
