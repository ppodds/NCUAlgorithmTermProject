import os

import tensorflow as tf

def get_dataset():
    """ Get wrapped dataset

    """
    for basedir in os.listdir(os.path.join('ChineseNumDataset', 'train_image')):
        for label in os.listdir(os.path.join('ChineseNumDataset', 'train_image', basedir)):


