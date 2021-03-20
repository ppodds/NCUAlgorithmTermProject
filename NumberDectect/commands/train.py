from . import *


def train(model_path, model_type, train_data, test_data, batch_size, epochs):
    """ Train model.


    Args:
        model_path (str): Model file path.
        model_type (str): Model type. Available: Res152, Res9
        train_data (tuple): Train dataset. Like (x_train, y_train).
        test_data (tuple): Test dataset. Like (x_test, y_test).
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
    model.fit(train_data,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=test_data,
              callbacks=[checkpoint])
