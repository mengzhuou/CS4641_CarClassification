import matplotlib.pyplot as plt
import argparse
import re
import os

def parse_training_logs(results_path):
    """Parse training history from a text log file."""
    # Regular expression to extract training metrics
    metrics_pattern = re.compile(
        r"loss:\s(?P<train_loss>[0-9.]+)\s-\saccuracy:\s(?P<train_accuracy>[0-9.]+)\s-\sval_loss:\s(?P<val_loss>[0-9.]+)\s-\sval_accuracy:\s(?P<val_accuracy>[0-9.]+)"
    )

    # Initialize history dictionary
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Read and parse the file
    with open(results_path, "r") as file:
        for line in file:
            match = metrics_pattern.search(line)
            if match:
                history["train_loss"].append(float(match.group("train_loss")))
                history["train_accuracy"].append(float(match.group("train_accuracy")))
                history["val_loss"].append(float(match.group("val_loss")))
                history["val_accuracy"].append(float(match.group("val_accuracy")))

    return history

def plot_accuracy(history, output_path):
    """Plot training and validation accuracy."""
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(output_path, "accuracy_plot.png"))
    plt.close()

def plot_loss(history, output_path):
    """Plot training and validation loss."""
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(output_path, "loss_plot.png"))
    plt.close()

def main(args):
    """Main function to generate the plots."""
    # Parse training logs
    history = parse_training_logs(args.results_path)

    # Ensure the output path exists
    output_path = os.path.dirname(args.results_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Generate plots
    plot_accuracy(history, output_path)
    plot_loss(history, output_path)

    print(f"Plots saved in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training results and generate plots.")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the text log file containing training results.")
    args = parser.parse_args()

    main(args)
