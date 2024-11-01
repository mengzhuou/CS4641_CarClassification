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

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten, Add, AveragePooling2D, Activation, GlobalAveragePooling2D, MaxPooling2D


class ResidualBlock(Layer):
    def __init__(self, name, filters, kernel_size, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.name = name
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
        print(f"Input shape: {inputs.shape}")
        c_1 = self.conv1(inputs)
        print(f"After conv1: {c_1.shape}")
        c_1 = self.bnorm1(c_1)
        print(f"After bnorm1: {c_1.shape}")
        c_1_out = self.relu(c_1)
        print(f"After relu1: {c_1_out.shape}")

        c_2 = self.conv2(c_1_out)
        print(f"After conv2: {c_2.shape}")
        c_2_out = self.bnorm2(c_2)
        print(f"c2_out After bnorm2: {c_2_out.shape}")
        inputs = self.projection(inputs)
        print(f"input shape after 1x1 conv projection {inputs}")

        skip = Add()([inputs, c_2_out])
        print(f"After add: {skip.shape}")
        skip = self.relu(skip)
        print(f"After relu: {skip.shape}")

        return skip
    
class ResNetModel(Model):

    def __init__(self, num_classes, filters, kernel_size, **kwargs):
        super(ResNetModel, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.init_conv = Conv2D(filters=filters, kernel_size = kernel_size, padding="same")
        self.init_bnorm = BatchNormalization()
        self.relu = Activation("relu")
        self.init_pool = MaxPooling2D(padding="same")
        # padding = same by default in ResidualBlock config
        # filters*=2
        kernel_size = (kernel_size[0] - 2, kernel_size[1] - 2)
        self.res1 = ResidualBlock(name="res1", filters=filters, kernel_size=kernel_size)
        filters*=2
        self.res2 = ResidualBlock(name="res2", filters=filters, kernel_size=kernel_size)
        filters*=2
        self.res3 = ResidualBlock(name="res3", filters=filters, kernel_size=kernel_size)

        self.final_pool = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.dense = Dense(self.num_classes, activation = "softmax")

    def call(self, inputs):
        x = self.init_conv(inputs)
        x = self.init_bnorm(x)
        x = self.relu(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        x = self.final_pool(x)
        #x = self.flatten(x)
        x = self.dense(x)
        return x

# test model on sample dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0  

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
inputs = tf.keras.Input(shape=(32, 32, 3))

# the first () is instantiating model, the second () is calling the model on given inputs
outputs = ResNetModel(num_classes=len(class_names), kernel_size=(3,3), filters=32)(inputs) 
resnet_model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Create a Model object

resnet_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
resnet_model.fit(x_train, y_train, epochs=1, batch_size=64, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = resnet_model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
    

