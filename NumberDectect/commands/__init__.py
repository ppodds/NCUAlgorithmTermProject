import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.utils import plot_model

from NumberDectect.model import ResNet

# config

# 學習率設定
learning_rate = 0.001


def build_model(model_type):
    """ Create a compiled ResNet 152 model.
    """
    # optimizer宣告
    optimizer = tf.optimizers.Adam(learning_rate)

    inputs = Input(shape=(28, 28, 1))
    if model_type == "Res152":
        outputs = ResNet.create_resnet152(inputs)
        name = "ResNet152"
    elif model_type == "Res9":
        outputs = ResNet.create_resnet9(inputs)
        name = "ResNet9"
    else:
        print("Available model type: Res152, Res9")
        exit()

    # Create model
    model = Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

