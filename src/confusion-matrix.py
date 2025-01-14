import json
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_classification_report(data, algorithm_name):
    if algorithm_name in data:
        return data[algorithm_name]["Classification Report"]
    else:
        raise ValueError(f"Algorithm {algorithm_name} not found in the data.")

def generate_confusion_matrix(classification_report):
    # Extracting the true and predicted values for the confusion matrix
    # Assuming binary classification for simplicity.
    # You can adapt this if the problem involves more classes.
    print("classification_report", classification_report)
    labels = [0, 1]
    y_true = []
    y_pred = []

    # True positives, false positives, false negatives, true negatives
    for label in labels:
        precision = classification_report[str(label)]["precision"]
        recall = classification_report[str(label)]["recall"]
        support = int(classification_report[str(label)]["support"])

        # Simulating true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN)
        tp = int(support * recall)
        fn = int(support - tp)
        fp = int(tp * (1 - precision) / precision) if precision != 0 else 0
        tn = int(support - tp - fp - fn)

        # For y_true and y_pred, we need to ensure the length matches by distributing TP, FN, FP, TN correctly
        # Adding true positives (label = 1 or 0 based on label)
        y_true.extend([label] * (tp + fn))
        # Assigning the predicted value based on the label
        y_pred.extend([label] * tp + [(1-label)] * fn)  # FN should be predicted as the other class

    # Calculating the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return cm

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))  # You can adjust the figure size if needed
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14})  # Adjust annotation font size
    plt.ylabel('True Label', fontsize=16)  # Adjust ylabel font size
    plt.xlabel('Predicted Label', fontsize=16)  # Adjust xlabel font size
    plt.title('Confusion Matrix', fontsize=18)  # Adjust title font size
    plt.xticks(fontsize=14)  # Adjust x-ticks font size
    plt.yticks(fontsize=14)  # Adjust y-ticks font size
    plt.show()

# Example usage
file_path = 'D:\\University\\Semester 3\\Applied Machine Learning\\Practice2\\ML-Project-Breast-Cancer\\results\\Ensemble\\3- Ensemble Methods - With Checking Data Imbalance - Using Class-Wise Mean\\ensemble_model_results.json'  # Replace with the actual path to your JSON file
algorithm_name = 'Bagging Ensemble'  # Change to the desired algorithm name

# Load data
data = load_json_data(file_path)

# Extract classification report for the given algorithm
classification_report = extract_classification_report(data, algorithm_name)

# Generate confusion matrix
cm = generate_confusion_matrix(classification_report)

# Plot confusion matrix
plot_confusion_matrix(cm, labels=["0: benign", "1: malignant"])
