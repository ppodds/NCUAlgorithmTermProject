from . import *


def info(generate_pic, model_type):
    """ Get model info.


    Args:
        generate_pic (bool): If true, generate a picture at working directory.
        model_type (str): Model type. Available: Res152, Res9
    """
    model = build_model(model_type)
    print(model.summary())
    if generate_pic:
        # 產生model架構的圖片
        plot_model(model, model_type + '.png')
        print('Generated ' + model_type + '.png in working directory.')
