import json
from pathlib import Path

import einops
import keras
from keras import layers

BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / 'model_utils' / 'model_settings.json'
categories_path = BASE_DIR / 'dataset_utils' / 'category_list.txt'

with open(config_path, 'r') as file:
    data = json.load(file)

HEIGHT = data.get('HEIGHT')
WIDTH = data.get('WIDTH')
FRAMES_COUNT = data.get('FRAMES_COUNT')
FRAME_STEP = data.get('FRAME_STEP')
BATCH_SIZE = data.get('BATCH_SIZE')
EPOCHS = data.get('EPOCHS')

with open(categories_path, 'r') as file:
    category_list = file.read().splitlines()
    CLASS_COUNT = len(category_list)


class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        """
          Custom convolutional layer that decomposes 3D convolution into:
            - spatial convolution (height, width)
            - temporal convolution (time dimension)
          This reduces computation while preserving performance.
        """
        super().__init__()
        self.seq = keras.Sequential([
            # Spatial decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(1, kernel_size[1], kernel_size[2]),
                          padding=padding),
            # Temporal decomposition
            layers.Conv3D(filters=filters,
                          kernel_size=(kernel_size[0], 1, 1),
                          padding=padding)
        ])

    def call(self, x):
        return self.seq(x)


class ResidualMain(keras.layers.Layer):
    """
        Residual block consisting of:
            - Conv2Plus1D
            - Layer Normalization
            - ReLU activation
            - Conv2Plus1D
            - Layer Normalization
    """

    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters,
                        kernel_size=kernel_size,
                        padding='same'),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


class Project(keras.layers.Layer):
    """
        Projection layer used to match tensor dimensions in residual connections.
        Applies Dense layer followed by Layer Normalization.
    """

    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)


def add_residual_block(input, filters, kernel_size):
    """
        Add residual blocks to the model. If the last dimensions of the input data
        and filter size does not match, project it such that last dimension matches.
    """
    out = ResidualMain(filters,
                       kernel_size)(input)

    res = input
    # Using the Keras functional APIs, project the last dimension of the tensor to
    # match the new filter size
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)

    return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
    """
       Custom layer for resizing video frames using einops and Keras Resizing layer.
    """

    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        self.resizing_layer = layers.Resizing(self.height, self.width)

    def call(self, video):
        """
            Resize video tensor frame-by-frame.
        """
        # b stands for batch size, t stands for time, h stands for height,
        # w stands for width, and c stands for the number of channels.
        old_shape = einops.parse_shape(video, 'b t h w c')
        images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
        images = self.resizing_layer(images)
        videos = einops.rearrange(
            images, '(b t) h w c -> b t h w c',
            t=old_shape['t'])
        return videos


def create_model():
    """
        Build and return the video classification model.

        Architecture:
        - Initial Conv2Plus1D + normalization + activation
        - Multiple residual blocks with downsampling
        - Global pooling and dense classification layer

        Returns:
            keras.Model: Compiled Keras model.
    """
    input_shape = (None, FRAMES_COUNT, HEIGHT, WIDTH, 3)
    input = layers.Input(shape=(input_shape[1:]))
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(CLASS_COUNT)(x)

    return keras.Model(input, x)
