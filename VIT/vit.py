# ADAPTED FROM 'dzlab' ON GITHUB: https://dzlab.github.io/notebooks/tensorflow/vision/classification/2021/10/01/vision_transformer.html

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Add, Dense, Dropout, Embedding, 
                                     GlobalAveragePooling1D, Layer, 
                                     LayerNormalization, MultiHeadAttention)

class PatchExtractor(Layer):
    def __init__(self, patch_size: int = 16, **kwargs):
        super(PatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        batch_size = tf.shape(images)[0]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEmbedding(Layer):
    def __init__(self, num_patches:int, 
                 projection_dim:int):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, self.projection_dim), dtype="float32")
        self.class_token = tf.Variable(initial_value=class_token, trainable=True)
        
    def build(self, input_shape):
        self.projection = Dense(units=self.projection_dim)
        self.position_embedding = Embedding(input_dim=self.num_patches+1, output_dim=self.projection_dim)

    def call(self, patch):
        batch = tf.shape(patch)[0]
        # reshape the class token embeddings
        class_token = tf.tile(self.class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, self.projection_dim))
        # calculate patches embeddings
        patches_embed = self.projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)
        
        # calcualte positional embeddings
        positions = tf.range(start=0, limit=self.num_patches+1, delta=1)
        positions_embed = self.position_embedding(positions)
   
        positions_embed = tf.expand_dims(positions_embed, axis=0)  # Add batch dimension
        positions_embed = tf.tile(positions_embed, multiples=[batch, 1, 1])  # Tile to match batch size
      
        # add both embeddings
        encoded = patches_embed + positions_embed
        return encoded


class MLP(Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1, activation = tf.nn.gelu):
        super(MLP, self).__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape):
        self.dense1 = Dense(self.hidden_features, activation=self.activation)
        self.dense2 = Dense(self.out_features)
        self.dropout1 = Dropout(self.dropout_rate)
        self.dropout2 = Dropout(self.dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        y = self.dropout2(x)
        return y

class Block(Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim, 
                                       dropout=self.dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(self.projection_dim * 2, self.projection_dim, self.dropout_rate)

    def call(self, x):
        # layer norm 1
        x1 = self.norm1(x) 
        # attn 
        attention_output = self.attn(x1, x1)
        # skip 1.
        attn = attention_output + x  #encoded_patches
        # layer norm 2
        x3 = self.norm2(attn)
        # MLP.
        mlp = self.mlp(x3)
        # skip 2
        return attn + mlp

class TransformerEncoder(Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.blocks = [Block(projection_dim, num_heads, dropout_rate) for _ in range(num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class VisionTransformer(Model):
    def __init__(self, patch_size:int, 
                 projection_dim:int, 
                 num_classes:int, 
                 dropout_rate:float,
                 img_width:int,
                 img_height:int,
                 **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.init_data_augmentation = Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            # brightness/contrast
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(factor=0.2), 
            # crop smaller than original size
            tf.keras.layers.RandomCrop(200, 200),
            tf.keras.layers.Resizing(224, 224), 
            tf.keras.layers.GaussianNoise(0.1) 
        ])
        self.patch_extraction = PatchExtractor(patch_size = patch_size)

        if img_height % patch_size != 0 or img_width % patch_size != 0:
            raise ValueError(f"Image ({img_height}, {img_width}) needs to be divisible by patch size {patch_size}.")

        num_patches = (img_height // patch_size) * (img_width // patch_size)

        self.patch_embedding = PatchEmbedding(num_patches= num_patches,
                                            projection_dim = projection_dim)
        self.transformer_encoder = TransformerEncoder(projection_dim = projection_dim)
        self.final_pool = GlobalAveragePooling1D()
        self.mlp = MLP(hidden_features=projection_dim, out_features=num_classes, dropout_rate=dropout_rate, activation=tf.nn.tanh)

    def call(self, inputs):
        x = self.init_data_augmentation(inputs)
        x = self.patch_extraction(inputs)
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.final_pool(x)
        x = self.mlp(x)
        return x
    
    

