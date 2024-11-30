# Group 19's Car Classification Project
## Directory Structure
- `Condensed_Dataset/`: Image dataset
  - `new_test/`: test data
      - `Acura/`: few samples of Acura car test images
      - `BMW/`: few samples of BMW car test images
  - `new_train/`: train data (same sub-dir structure as above)
  - `validation/`: validation data (same sub-dir structure as above)
- `models/model_32accuracy_917372`: contains weights and architecture of our best model (outputted and consumable by tf)
- `results/`:
    - `model_32accuracy_917372/`: contains evaluation metrics (e.g. confusion matrix, classification report on f1-score and other metrics, slurm report) for our best run
    - `model_#/`: metrics recorded from our other model experiments. More info regarding each run in our report
- `DataCleaning.py`: provides a data cleaning pipeline for our ML car model train and test dataset, includes functions to delete folders with 30 or fewer images, resize images to a consistent size with padding, and check for non-JPEG images, ensuring all files meet format requirements.
- `DataSplitting.py`: Condenses cleaned data from car models to car brands (ex. BMW X5/, BMW X6/ -> BMW/), and splits these reduced classes into train, validation, and test splits (80-10-10)
- `requirements.txt`: Necessary pip modules to install in python environment
- `resnet_exp.py`: Main pipeline file that loads images into Tensorflow Datasets, and instantiates + trains + saves + evaluates Resnet model. Allows for configs like # epochs or file paths to be set via Argument Parsing flags when running the file.
- `resnetmodel.py`: Contains Residual block class (inherits from tf's Layer class) and ResNet Model class (inherits from tf's Model class) with our architecture implementation. Instantiated in **resnet_exp.py** .
- `train_model_tf_mod`: SBATCH script to submit GPU-accelerated tensorflow jobs to pace/ice cluster. Loads necessary modules, python environment, and runs **resnet_exp.py** with tunable flags. 
- `analyzing_training`: Parses training logs from text file ending with .out, extracts relevant metrics (loss and accuracy), and generates plots to visualize the training and validation performance of a machine learning model.
