import tensorflow as tf
from tensorflow.keras import *

from . import ResNet

import os

# config

# 學習率設定
learning_rate = 0.001


def build_model(model_type, model_path=None):
    """ Create a compiled model.

    Args:
        model_type (str): Model type of function generated. Only effective when model_path is None
        model_path (str): Existed model file path. Only accept hdf5 file.
    """
    # 檢查是否有已訓練好的model，有就直接讀取並建立
    if model_path and os.path.exists(model_path):
        model = models.load_model(model_path)
    else:
        # optimizer宣告
        optimizer = tf.optimizers.Adam(learning_rate)

        inputs = Input(shape=(28, 28, 1))
        # 確認建立的model種類
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
