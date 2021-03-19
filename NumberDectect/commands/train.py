from . import *


def load_dataset():
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 将像素范围缩至0到1
    x_train /= 255
    x_test /= 255
    # convert class vectors to binary class matrices
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    return x_train, x_test, y_train, y_test


def train(model_path, model_type, dataset, batch_size, epochs):
    """ Train model.


    Args:
        model_path (str): Model file path.
        model_type (str): Model type. Available: Res152, Res9
        dataset (tuple): Train dataset. Like (x_train, y_train, x_test, y,test).
        epochs (int):  Train epochs.
        batch_size (int): Train batch size.
    """
    model = build_model(model_type)
    # 設定checkpoint callback
    checkpoint = callbacks.ModelCheckpoint(model_path,
                                           monitor='loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           )
    x_train = dataset[0]
    x_test = dataset[1]
    y_train = dataset[2]
    y_test = dataset[3]

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint])
