from . import *


def evaluate_project(model_path, test_data):
    """ Train model.


    Args:
        model_path (str): Model file path. If the file doesn't exist, it will create a new model, or it will use the existed model.
        test_data (tuple): Test dataset. Like (x_test, y_test).
    """
    model = build_model(None, model_path)
    # 用keras的funtion做model的evaluate
    model.evaluate(test_data, verbose=1)
