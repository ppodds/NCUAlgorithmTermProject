from . import *


def train(model_path, model_type, train_data, validation_data, epochs):
    """ Train model.


    Args:
        model_path (str): Model file path. If the file doesn't exist, it will create a new model, or it will use the existed model.
        model_type (str): Model type. Available: Res152, Res9
        train_data (tuple): Train dataset. Like (x_train, y_train).
        validation_data (tuple): Validation dataset. Like (x_validation, y_validation).
        epochs (int):  Train epochs.
    """
    model = build_model(model_type, model_path)
    # 設定checkpoint callback
    checkpoint = callbacks.ModelCheckpoint(model_path,
                                           monitor='loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           )
    model.fit(train_data,
              epochs=epochs,
              verbose=1,
              validation_data=validation_data,
              callbacks=[checkpoint])
