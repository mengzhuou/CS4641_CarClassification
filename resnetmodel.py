import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, AveragePooling2D, Activation, GlobalAveragePooling2D, MaxPooling2D, Dropout

class ResidualBlock(Layer):
    def __init__(self, block_name, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.block_name = block_name
        self.filters = filters
        self.kernel_size = kernel_size
        self.relu = Activation("relu")
        self.projection = Conv2D(self.filters, (1, 1), padding='same')
    
    # build layers
    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding = "same")
        self.bnorm1 = BatchNormalization()

        self.conv2 = Conv2D(self.filters, self.kernel_size, padding = "same")
        self.bnorm2 = BatchNormalization()
        
    # implement skip connection
    def call(self, inputs):
        c_1 = self.conv1(inputs)
        c_1 = self.bnorm1(c_1)
        c_1_out = self.relu(c_1)
        c_2 = self.conv2(c_1_out)
        c_2_out = self.bnorm2(c_2)
        inputs = self.projection(inputs)
        skip = Add()([inputs, c_2_out])
        skip = self.relu(skip)
    
        return skip

class ResNetModel(Model):

    def __init__(self, num_classes, filters, kernel_size, **kwargs):
        super(ResNetModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        
        self.init_data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        self.init_conv = Conv2D(filters=filters, kernel_size = kernel_size, padding="same")
        self.init_bnorm = BatchNormalization()
        self.relu = Activation("relu")
        self.init_pool = MaxPooling2D(padding="same")
        # padding = same by default in ResidualBlock config
        kernel_size = (kernel_size[0] - 2, kernel_size[1] - 2)
        self.res1 = ResidualBlock(block_name="res1", filters=filters, kernel_size=kernel_size)
        filters*=2
        self.res2 = ResidualBlock(block_name="res2", filters=filters, kernel_size=kernel_size)
        filters*=2
        self.res3 = ResidualBlock(block_name="res3", filters=filters, kernel_size=kernel_size)

        self.final_pool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes, activation = "softmax")

    def call(self, inputs):

        x = self.init_data_augmentation(inputs)

        x = self.init_conv(inputs)
        x = self.init_bnorm(x)
        x = self.relu(x)
        x = self.init_pool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.final_pool(x)
        #x = self.flatten(x)
        x = self.dense(x)
        return x

# test_loss, test_acc = resnet_model.evaluate(x_test, y_test, verbose=2)
# print(f'Test accuracy: {test_acc}')
    

# ## ==================== EVALUATION AND RESULTS ====================

# for eval and results
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # get model predictions
# y_pred = resnet_model.predict(x_test)
# predicted_classes = np.argmax(y_pred, axis=1)
# true_classes = y_test.flatten()  # flatten if y_test has shape (num_samples, 1)

# # calculate eval metrics
# accuracy = accuracy_score(true_classes, predicted_classes)
# precision = precision_score(true_classes, predicted_classes, average='weighted')  # avg is weighted by the num of true instances for each label
# recall = recall_score(true_classes, predicted_classes, average='weighted')
# f1 = f1_score(true_classes, predicted_classes, average='weighted')

# # output results
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")

# # create and print classification report
# print("Classification Report")
# print(classification_report(true_classes, predicted_classes, target_names=class_names))

# # plot confusion matrix as a heatmap
# conf_matrix = confusion_matrix(true_classes, predicted_classes)
# plt.figure(figsize=(10,8))
# sns.heatmap(conf_matrix,
#             annot=True,
#             fmt='d',
#             cmap='Blues',
#             xticklabels=class_names,
#             yticklabels=class_names)
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()





# https://medium.com/swlh/how-to-create-a-residual-network-in-tensorflow-and-keras-cd97f6c62557
# https://github.com/Prakadeeswaran05/Customising-your-models-with-TensorFlow-2-Coursera/blob/main/Week%204%20Programming%20Assignment-Residual%20Network.ipynb
# https://www.youtube.com/watch?v=Q1JCrG1bJ-A

# we need regular blocks and then residual blocks that add the outputs between blocks

# block 1
# block 2 = builds on block 1
# res1 = b1 + b2
# block 3 = builds on res1
# block 4 = builds on block 3
# res2 = b3 + b4
# block 5 = builds on res2
# block 6
# 


# 2 conv per block
# residuals are in between the convolutions per block
# Each layer is followed by Batch Normalization and a ReLU activation function.