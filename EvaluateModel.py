import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime


def load_model(model_path, model_type):
    """ Load model based on its type (CNN, ViT, GMM) """
    if model_type in ['resnet', 'vit']:
        # use tensorflow to load model
        model = tf.keras.models.load_model(model_path)
    elif model_type == 'gmm':
        # use scikit-learn to load model
        model = joblib.load(model_path)
    else:
        raise ValueError("Wrong model type. Choose from 'resnet', 'vit', or 'gmm'")
    
    return model

def evaluate_model(model, model_type, x_text, y_test, label_names, output_dir):
    """ Evaluate model and output results """
    if model_type in ['resnet', 'vit']:  # neural network models
        # get softmax probabilities and argmax predictions
        y_pred = model.predict(x_text)
        predicted_labels = np.argmax(y_pred, axis=1)
    elif model_type == 'gmm':
        # predict on reshaped data
        predicted_labels = model.predict(x_text.reshape((x_text.shape[0], -1)))
    
    # flatten labels
    true_labels = y_test.flatten()

    # calculate eval metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')  # avg is weighted by the num of true instances for each label
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # get curr timestamp to append to results name (prevent overwriting)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # output results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # create and save classification report
    report = classification_report(true_labels, predicted_labels, target_names=label_names)
    report_path = os.path.join(output_dir, f'classification_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # plot and save confusion matrix as a heatmap
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_type.upper()} Model')
    plt_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(plt_path)
    print(f"Confusion matrix saved to {plt_path}")
    plt.show()

if __name__ == "__main__":
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate ResNet, ViT, or GMM models on test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--model_type', type=str, choices=['resnet', 'vit', 'gmm'], required=True, help="Type of model: 'resnet', 'vit', or 'gmm'.")
    parser.add_argument('--x_test_path', type=str, required=True, help='Path to test data features (numpy array).')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to test data labels (numpy array).')
    parser.add_argument('--label_names_path', type=str, required=True, help='Path to label names (numpy array or list).')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save evaluation results.')

    args = parser.parse_args()

    # load test data and label names
    x_test = np.load(args.x_test_path)
    y_test = np.load(args.y_test_path)
    label_names = np.load(args.label_names_path) if args.label_names_path.endswith('.npy') else joblib.load(args.label_names_path)

    # create output directory if it doesnt exist
    os.makedirs(args.output_dir, exist_ok=True)

    # load model
    model = load_model(args.model_path, args.model_type)

    # evaluate and output results
    evaluate_model(model, args.model_type, x_test, y_test, label_names, args.output_dir)

    # to evaluate a model, run the following command from the terminal (replace the arguments with actual ones):
    # python eval.py --model_path "PATH/TO/MODEL" --model_type "MODEL_TYPE" --x_test_path "PATH/TO/x_test.npy" --y_test_path "PATH/TO/y_test.npy" --label_names_path "PATH/TO/label_names.npy" --output_dir "PATH/TO/SAVE/RESULTS"
