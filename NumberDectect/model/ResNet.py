import tensorflow as tf
import numpy as np

from .layers import *


def resnet_bottleneck_block(model, output_filters, inter_filters, activation=True):
    """ Create a bottleneck block.

    Args:
        model: Sequential model.
        output_filters (int): Output filters amount.
        inter_filters (int): Inter filters amount.
        activation (bool): Whether to use Relu at the end.
    """
    # 建立卷積層
    c = conv_bn_relu(model, inter_filters, 1)
    c = conv_bn_relu(c, inter_filters, 3)
    c = conv_bn(c, output_filters, 1, bn_gamma_initializer='zeros')
    # 和原本輸出相加
    model = layers.Add()([c, model])
    # 附上活化函數
    if activation:
        return layers.Activation('relu')(model)
    else:
        return model


def resnet_bottleneck_inc_block(model, output_filters, inter_filters, strides1x1=(1, 1), strides3x3=(2, 2)):
    """ Create a bottleneck block with shortcut connection.

    Args:
        model: Sequential model.
        output_filters (int): Output filters amount.
        inter_filters (int): Inter filters amount.
        strides1x1 (tuple[int, int]): Strides of 1x1 filters.
        strides3x3 (tuple[int, int]): Strides of 3x3 filters.
    """
    # 建立卷積層
    c = conv_bn_relu(model, inter_filters, 1, strides=strides1x1)
    c = conv_bn_relu(c, inter_filters, 3, strides=strides3x3)
    c = conv_bn(c, output_filters, 1, bn_gamma_initializer='zeros')

    # shortcut connection
    strides = np.multiply(strides1x1, strides3x3)
    x = conv_bn(model, output_filters, 1, strides=strides)
    # 和原本輸出相加
    model = layers.Add()([c, x])
    # 附上活化函數
    return tf.keras.layers.Activation('relu')(model)


def resnet_small_block(model, filter_nums, activation=True):
    """ Create a small block.

    Args:
        model: Sequential model.
        filter_nums: Filter amount.
        activation (bool): Whether to use Relu at the end.
    """
    # 建立卷積層
    c = conv_bn_relu(model, filter_nums, 3)
    c = conv_bn(c, filter_nums, 3, bn_gamma_initializer='zeros')
    # 和原本輸出相加
    model = layers.Add()([c, model])
    # 附上活化函數
    if activation:
        return layers.Activation('relu')(model)
    else:
        return model


def resnet_small_inc_block(model, filter_nums):
    """ Create a small block with shortcut connection.

    Args:
        model: Sequential model.
        filter_nums: Filter amount.
    """
    # 建立卷積層
    c = conv_bn_relu(model, filter_nums, 3)
    c = conv_bn(c, filter_nums, 3, strides=(2, 2), bn_gamma_initializer='zeros')

    # shortcut connection
    x = conv_bn(model, filter_nums, 3, strides=(2, 2))
    # 和原本輸出相加
    model = layers.Add()([c, x])
    # 附上活化函數
    return tf.keras.layers.Activation('relu')(model)


def resnet_bottleneck_model(model, filter_sizes, repeat_sizes, small_model=False):
    """ Create a ResNet model.

    Args:
        model: Sequential model.
        filter_sizes (tuple): Filters' size of layers components.
        repeat_sizes (tuple): Filters' repeat times of layers components.
        small_model (bool): Whether the model is less than 50 layers.
    """
    # 檢查輸入資料是否合規
    assert len(filter_sizes) == len(repeat_sizes)
    # ResNet 開頭
    if small_model:
        model = conv_bn_relu(model, 16, 7, strides=(1, 1))
        # 疊出section
        for i in range(len(repeat_sizes)):
            model = resnet_small_inc_block(model, filter_nums=filter_sizes[i])
            for _ in range(repeat_sizes[i]):
                model = resnet_small_block(model, filter_nums=filter_sizes[i])
    else:
        model = conv_bn_relu(model, 64, 3, strides=(1, 1))
        # 疊出section
        for i in range(len(repeat_sizes)):
            model = resnet_bottleneck_inc_block(model, output_filters=filter_sizes[i]
                                                , inter_filters=filter_sizes[i] // 4,
                                                strides3x3=(2, 2) if i > 0 else (1, 1))
            for _ in range(repeat_sizes[i]):
                model = resnet_bottleneck_block(model, output_filters=filter_sizes[i],
                                                inter_filters=filter_sizes[i] // 4)
    return model


def create_resnet152(model, classes=10):
    """ Create a ResNet-152 model.

    Args:
        model: Sequential model.
        classes: Output layer's class amount.
    """
    # ResNet filter大小定義
    filter_sizes = (256, 512, 1024, 2048)
    # ResNet Block重複次數定義
    repeat_sizes = (2, 7, 35, 2)
    # 建立部分model
    model = resnet_bottleneck_model(model, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes)
    # 尾部平均池
    model = layers.AveragePooling2D(pool_size=(4, 4))(model)
    # 展平最後的隱藏層
    model = layers.Flatten()(model)
    # 加入輸出層(全連接層)
    model = layers.Dense(classes)(model)
    # 輸出層活化函數
    model = layers.Activation('softmax', dtype='float32')(model)

    return model


def create_resnet9(model, classes=10):
    """ Create a ResNet-9 model.

    Args:
        model: Sequential model.
        classes: Output layer's class amount.
    """
    # ResNet filter數量定義
    filter_sizes = (16, 32)
    # ResNet Block重複次數定義
    repeat_sizes = (1, 1)
    # 建立部分model
    model = resnet_bottleneck_model(model, filter_sizes=filter_sizes, repeat_sizes=repeat_sizes, small_model=True)
    # 尾部平均層
    model = layers.AveragePooling2D(pool_size=(7, 7))(model)
    # 展平最後的隱藏層
    model = layers.Flatten()(model)
    # 加入輸出層(全連接層)
    model = layers.Dense(classes)(model)
    # 輸出層活化函數
    model = layers.Activation('softmax', dtype='float32')(model)

    return model
