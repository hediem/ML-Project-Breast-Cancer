import os
import numpy as np
import pandas as pd
import logging
import time
import json
from json import JSONEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Enable experimental IterativeImputer API before importing
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_data():
    """
    Loads the dataset from a relative path, converts non-numeric entries,
    and returns features (X) and labels (y).
    """
    start_time = time.time()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'dataset', 'data.csv')
    
    print("📥 Loading data...")
    data = pd.read_csv(data_path)
    # Convert 'Bare Nuclei' to numeric (coerce '?' into NaN)
    data['Bare Nuclei'] = pd.to_numeric(
        data['Bare Nuclei'].replace('?', np.nan), errors='coerce'
    )
    # Drop an identifier column and convert class labels: 4 -> 1, otherwise 0.
    X = data.drop(columns=['Sample code number', 'Class'])
    y = data['Class'].apply(lambda x: 1 if x == 4 else 0)
    
    logging.info(
        f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features."
    )
    logging.info(f"Class distribution:\n{y.value_counts(normalize=True)}")
    end_time = time.time()
    print(f"✅ Data loaded successfully in {end_time - start_time:.2f}s.")
    return X, y

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importances provided by the estimator.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    df = df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

def build_pipeline():
    """
    Constructs a pipeline with default parameters.
    (Note: The hyperparameters will later be tuned using GridSearchCV.)
    """
    pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=42),
            threshold='median'
        )),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('classifier', RandomForestClassifier())
    ])
    return pipeline

def run_grid_search(X_train, y_train):
    """
    Runs GridSearchCV on the training set to search for the best hyperparameters.
    Two alternatives for the classifier (RandomForestClassifier and BaggingClassifier)
    are considered.
    """
    print("🔍 Starting grid search on training data...")
    pipeline = build_pipeline()
    
    # Define the parameter grid; note that these options include those that produced your best fold.
    param_grid = [
        {
            'classifier': [RandomForestClassifier(random_state=42, class_weight='balanced')],
            'classifier__n_estimators': [200, 400],
            'classifier__max_depth': [10, 20, None],
            'classifier__min_samples_split': [2, 5],
            'feature_selection__threshold': ['median', 'mean'],
            'imputer__n_neighbors': [3, 5, 7]
        },
        {
            'classifier': [BaggingClassifier(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                random_state=42
            )],
            'classifier__n_estimators': [50, 100],
            'classifier__max_samples': [0.7, 1.0],
            'classifier__max_features': [0.7, 1.0],
            'feature_selection__threshold': ['median', 'mean'],
            'imputer__n_neighbors': [3, 5, 7]
        }
    ]
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1,
        error_score='raise'
    )
    
    grid_search.fit(X_train, y_train)
    print("✅ Grid search completed.")
    print("\n✨ Best parameters found:")
    for k, v in grid_search.best_params_.items():
        print(f"   • {k}: {v}")
        
    # Return both the best parameters and the best estimator (in case you wish to inspect it)
    return grid_search.best_params_, grid_search.best_estimator_

def train_final_model(X_train, y_train, best_params):
    """
    Constructs and fits a final pipeline on the entire training set using the best parameters.
    The pipeline is rebuilt with the parameters (for the imputer, feature selection threshold,
    and classifier) discovered in grid search.
    """
    print("🔧 Training final model with best parameters...")
    
    # Set up the imputer with the best number of neighbors
    imputer_neighbors = best_params.get('imputer__n_neighbors', 5)
    
    # Use the threshold found for feature selection
    fs_threshold = best_params.get('feature_selection__threshold', 'median')
    
    # Determine which classifier type was selected
    selected_classifier = best_params.get('classifier')
    
    if isinstance(selected_classifier, BaggingClassifier):
        # Extract the best hyperparameters for the BaggingClassifier variant.
        n_estimators = best_params.get('classifier__n_estimators', 50)
        max_samples = best_params.get('classifier__max_samples', 0.7)
        max_features = best_params.get('classifier__max_features', 0.7)
        final_classifier = BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=42
        )
    else:
        # Otherwise, assume the classifier is a RandomForestClassifier.
        n_estimators = best_params.get('classifier__n_estimators', 200)
        max_depth = best_params.get('classifier__max_depth', None)
        min_samples_split = best_params.get('classifier__min_samples_split', 2)
        final_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',  # as specified in the grid
            random_state=42
        )
    
    # Build the final pipeline with the selected parameters.
    final_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=imputer_neighbors)),
        ('scaler', MinMaxScaler()),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
            threshold=fs_threshold
        )),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('classifier', final_classifier)
    ])
    
    # Fit the final model on the entire training set.
    final_pipeline.fit(X_train, y_train)
    
    # Extract and display selected features from the feature selection step.
    fs = final_pipeline.named_steps['feature_selection']
    mask = fs.get_support()
    selected_features = list(X_train.columns[mask])
    print(f"\n✨ Selected features: {selected_features}")
    
    # Optionally, plot the feature importances from the underlying estimator
    print("\n📊 Plotting feature importances of the feature selection estimator...")
    plot_feature_importance(fs.estimator_, X_train.columns)
    
    return final_pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the hold-out test set and prints common classification metrics.
    """
    print("🔍 Evaluating final model on hold-out test set...")
    preds = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'F1 Score': f1_score(y_test, preds),
        'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }
    
    print("\n📊 Final evaluation metrics:")
    for name, value in metrics.items():
        print(f"   • {name}: {value:.4f}")
    return metrics

def aggregate_and_save_results(results, current_dir):
    """
    Aggregates the provided metric(s) and saves them to a JSON file.
    """
    start_time = time.time()
    print("\n📝 Aggregating results...")
    aggregated = {
        'Average Accuracy': np.mean([r['Accuracy'] for r in results]),
        'Std Accuracy': np.std([r['Accuracy'] for r in results]),
        'Average Recall': np.mean([r['Recall'] for r in results]),
        'Std Recall': np.std([r['Recall'] for r in results]),
        'Average Precision': np.mean([r['Precision'] for r in results]),
        'Std Precision': np.std([r['Precision'] for r in results]),
        'Average F1 Score': np.mean([r['F1 Score'] for r in results]),
        'Std F1 Score': np.std([r['F1 Score'] for r in results]),
        'Average ROC AUC': np.mean([r['ROC AUC'] for r in results]),
        'Std ROC AUC': np.std([r['ROC AUC'] for r in results])
    }
    
    results_dir = os.path.join(current_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'final_results.json')
    
    with open(results_path, 'w') as f:
        json.dump(
            {'Metrics': results, 'Aggregated': aggregated},
            f,
            indent=4,
            cls=NumpyEncoder
        )
    
    logging.info(f"Results saved to: {os.path.abspath(results_path)}")
    end_time = time.time()
    print(f"📂 Results saved to: {os.path.abspath(results_path)}")
    print(f"🕒 Aggregation and saving took {end_time - start_time:.2f}s")
    return aggregated

def main():
    total_start_time = time.time()
    print("🔥 Starting the process...")
    
    # Load the data
    X, y = load_data()
    
    # Split the data into training (80%) and hold-out test (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 1. Perform grid search on the training set to obtain the best hyperparameters.
    best_params, best_estimator = run_grid_search(X_train, y_train)
    
    # 2. Using the best parameters, train a final model on the entire training set.
    final_model = train_final_model(X_train, y_train, best_params)
    
    # 3. Evaluate the final model on the independent hold-out test set.
    final_metrics = evaluate_model(final_model, X_test, y_test)
    
    # Optionally, aggregate and save the final metrics.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    aggregate_and_save_results([final_metrics], current_dir)
    
    total_time = time.time() - total_start_time
    logging.info(f"Total process time: {total_time:.2f}s")
    print(f"\n🎉 Total process time: {total_time:.2f}s")

if __name__ == '__main__':
    main()
