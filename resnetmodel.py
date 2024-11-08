import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, AveragePooling2D, Activation, GlobalAveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, Activation, GlobalAveragePooling2D, MaxPooling2D, RandomFlip, RandomRotation, RandomZoom

class ResidualBlock(Layer):
    def __init__(self, block_name, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.block_name = block_name
        self.filters = filters
        self.kernel_size = kernel_size
        self.relu = Activation("relu")
        # only used for input dimension mismatches
        self.projection = Conv2D(self.filters, (1, 1), padding='same')
    
    # build layers
    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, padding = "same", activation="relu")
        self.bnorm1 = BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv2 = Conv2D(self.filters, self.kernel_size, padding = "same", activation="relu")
        self.bnorm2 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        
    # implement skip connection
    def call(self, inputs):
        c_1 = self.conv1(inputs)
        c_1 = self.bnorm1(c_1)
        c_2 = self.conv2(c_1)
        c_2_out = self.bnorm2(c_2)
        if inputs.shape[-1] != c_2_out.shape[-1]:
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
        
        self.init_data_augmentation = Sequential([
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
        ])
        self.init_conv = Conv2D(filters=filters, kernel_size = kernel_size, padding="same")
        self.init_bnorm = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = Activation("relu")
        self.init_pool = MaxPooling2D(padding="same")
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

        x = self.init_data_augmentation(inputs)
        x = self.init_conv(inputs)
        x = self.init_bnorm(x)
        x = self.relu(x)
        x = self.init_pool(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.final_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

# Inspo:
# https://medium.com/swlh/how-to-create-a-residual-network-in-tensorflow-and-keras-cd97f6c62557
# https://github.com/Prakadeeswaran05/Customising-your-models-with-TensorFlow-2-Coursera/blob/main/Week%204%20Programming%20Assignment-Residual%20Network.ipynb
# https://www.youtube.com/watch?v=Q1JCrG1bJ-A

