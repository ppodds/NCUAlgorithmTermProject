from tensorflow.keras import layers


def conv(model, filter_nums, filter_size, strides=(1, 1)):
    """ Create a Convolutional layer.

    Args:
        model: Sequential model.
        filter_nums (int): Filter amount.
        filter_size (int): Filter size.
        strides (tuple[int, int]): Filter strides.
    """
    c = layers.Conv2D(filter_nums, filter_size, strides=strides, padding='same'
                      , kernel_initializer='he_normal', use_bias=False)(model)
    return c


def conv_bn(model, filter_nums, filter_size, strides=(1, 1), bn_gamma_initializer='ones'):
    """ Create a Convolutional layer with batch normalization.

    Args:
        model: Sequential model.
        filter_nums (int): Filter amount.
        filter_size (int): Filter size.
        strides (tuple[int, int]): Filter strides.
        bn_gamma_initializer: Gamma initializer of batch normalization layer.
    """
    c = conv(model, filter_nums, filter_size, strides)
    c_bn = layers.BatchNormalization(gamma_initializer=bn_gamma_initializer)(c)
    return c_bn


def conv_bn_relu(model, filter_nums, filter_size, strides=(1, 1), bn_gamma_initializer='ones'):
    """ Create a Convolutional layer with batch normalization and Relu activation.

        Args:
            model: Sequential model.
            filter_nums (int): Filter amount.
            filter_size (int): Filter size.
            strides (tuple[int, int]): Filter strides.
            bn_gamma_initializer: Gamma initializer of batch normalization layer.
    """
    c_bn = conv_bn(model, filter_nums, filter_size, strides, bn_gamma_initializer)
    return layers.Activation('relu')(c_bn)
