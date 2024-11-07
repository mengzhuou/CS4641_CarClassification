import tensorflow as tf

test_dir = "Split_Dataset/new_test"
train_dir = "Split_Dataset/new_train"
validation_dir = "Split_Dataset/validation"

batch_size = 32
img_height = 224  
img_width = 224   


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


from resnetmodel import ResNetModel
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
outputs = ResNetModel(num_classes=192, kernel_size=(3,3), filters=32)(inputs) 
resnet_model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Create a Model object

resnet_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


resnet_model.fit(train_ds, validation_data = val_ds, epochs=10, batch_size=batch_size)

test_loss, test_acc = resnet_model.evaluate(test_ds, verbose=2)
print(f'Test accuracy: {test_acc}')