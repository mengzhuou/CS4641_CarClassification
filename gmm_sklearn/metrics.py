import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, fowlkes_mallows_score

def map_clusters_to_labels(true_labels, predicted_labels):
    """
    Map predicted clusters to true labels using the Hungarian Algorithm.
    """
    from scipy.optimize import linear_sum_assignment
    
    # Build confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Perform Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    # Map predicted labels
    mapped_labels = np.array([mapping[label] for label in predicted_labels])
    return mapped_labels

def evaluate_metrics(true_labels, predicted_labels, class_names, output_dir):
    """
    Evaluate model performance metrics and save results.
    """
    # Get unique classes and align class names
    unique_classes = np.unique(true_labels)
    unique_class_names = [class_names[i] for i in unique_classes]

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    fowlkes = fowlkes_mallows_score(true_labels, predicted_labels)
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Fowlkes-Mallows Score: {fowlkes:.2f}")
    
    # Save metrics to a text file
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=unique_class_names, 
        zero_division=0,
        labels=unique_classes
    )
    report_path = os.path.join(output_dir, 'classification_report.txt')
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write(f"Fowlkes-Mallows Score: {fowlkes:.2f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save confusion matrix as a heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_class_names, yticklabels=unique_class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    heatmap_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(heatmap_path)
    print(f"Confusion matrix heatmap saved to {heatmap_path}")
    plt.close()
