import tensorflow as tf
from tensorflow.data import Dataset
from resnetmodel import ResNetModel
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import Optional


def load_data(test_dir: str = "Split_Dataset/new_test", 
              train_dir: str = "Split_Dataset/new_train", 
              validation_dir: str = "Split_Dataset/validation", 
              batch_size:int = 32,
              img_height:int = 224,
              img_width: int = 224) -> dict[str, Dataset]: # train_ds, test_ds, val_ds are keys

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",           
        label_mode="int",             
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123
    )

    train_ds = train_ds.map(lambda x, y: (x / 255.0, y))  # Normalize images
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123
    )

    val_ds = val_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(tf.data.AUTOTUNE)


    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=False,  
        seed=123
    )

    test_ds = test_ds.map(lambda x, y: (x / 255.0, y)).cache().prefetch(tf.data.AUTOTUNE)
    return {"train_ds": train_ds, "test_ds": test_ds, "val_ds":val_ds}

def train_and_save_resnet(path_to_save: str, 
                          train_ds: Dataset,
                          val_ds: Dataset,
                          epochs: int = 10,
                          steps_per_epoch: Optional[int] = None,
                          img_height:int = 224,
                          img_width:int = 224,
                          channels: int = 3, 
                          num_classes: int = 192, 
                          kernel_size: tuple = (7,7), 
                          filters:int=32) -> None:
    inputs = tf.keras.Input(shape=(img_height, img_width, channels))
    outputs = ResNetModel(num_classes=num_classes, kernel_size=kernel_size, filters=filters)(inputs) 
    
    resnet_model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Create a Model object

    resnet_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    fit_params = {
        'x': train_ds,
        'epochs': epochs,
        'validation_data': val_ds,
        'verbose': 1
    }

    if steps_per_epoch is not None:
        fit_params.update({"steps_per_epoch":steps_per_epoch})

    resnet_model.fit(**fit_params)
    resnet_model.save(path_to_save)

def evaluate_model(model_path: str, test_ds: Dataset, output_dir: str, class_names: list[str]):
    """ Evaluate model and output results """
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(test_ds)
    predicted_indices = np.argmax(y_pred, axis=1)
    predicted_labels = [class_names[i] for i in predicted_indices]


    # grab the true labels from the test Dataset object
    true_indices = []
    for _, labels in test_ds:
        true_indices.extend(labels.numpy())
    true_labels = [class_names[i] for i in true_indices]

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
    report = classification_report(true_labels, predicted_labels, target_names=class_names)
    report_path = os.path.join(output_dir, f'classification_report_{timestamp}.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
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
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - ResNet Model')
    plt_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plt.savefig(plt_path)
    print(f"Confusion matrix saved to {plt_path}")
    plt.show()

if __name__ == "__main__":
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser(description='Run ResNet model pipeline')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--results_path', type=str, required=True, help='Directory to save evaluation results.')
    parser.add_argument('--epochs', type=int, default=10, required=False, help='If you want to set epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=10, required=False, help='If you want to run model faster')
    
    args = parser.parse_args()

    # load Dataset objects
    data = load_data()

    test_ds = data["test_ds"]
    train_ds = data["train_ds"]
    val_ds = data["val_ds"]

    train_and_save_resnet(path_to_save=args.model_path, 
                          train_ds=train_ds, 
                          val_ds=val_ds, 
                          epochs=args.epochs, 
                          steps_per_epoch=args.steps_per_epoch)
    
    # grab all folder names from here

    class_names = os.listdir('Split_Dataset/new_test')
    # evaluate and output results
    evaluate_model(model_path=args.model_path, 
                   test_ds=test_ds, 
                   output_dir=args.results_path,
                   class_names=class_names)

    # to evaluate a model, run the following command from the terminal (replace the arguments with actual ones):
    # python eval.py --model_path "PATH/TO/MODEL" --model_type "MODEL_TYPE" --x_test_path "PATH/TO/x_test.npy" --y_test_path "PATH/TO/y_test.npy" --label_names_path "PATH/TO/label_names.npy" --output_dir "PATH/TO/SAVE/RESULTS"
