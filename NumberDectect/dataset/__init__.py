import os

import tensorflow as tf
from tensorflow.keras import utils


def get_input_data(folder_name):
    """ Load image file path and label from directory.

    Args:
        folder_name (str): Folder name of the type you want to get.

    Return:
        A tuple including two list. List 1 including image file path. List 2 including label data.
        ([image file path], [image label])
    """
    image_paths = []
    image_label = []
    for basedir in os.listdir(os.path.join('ChineseNumDataset', folder_name)):
        for label in os.listdir(os.path.join('ChineseNumDataset', folder_name, basedir)):
            for file_name in os.listdir(os.path.join('ChineseNumDataset', folder_name, basedir, label)):
                image_paths.append((os.path.join('ChineseNumDataset', folder_name, basedir, label, file_name)))
                image_label.append(label)
    return image_paths, image_label


class DatasetWrapper:
    """ Wrapper class for image data and label.

    """

    def __init__(self, image_data):
        self.image_data = image_data

    def get_dataset(self, batch_size):
        """ Get input dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.image_data)
        dataset = dataset.map(self.map_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self.map_transform_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def map_load_image(self, image_path, label):
        """ Load image from image path and transform to tensor.
        """
        image_raw = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_raw, channels=0)
        return image, label

    def map_transform_image_and_label(self, image, label):
        """ Tensorflow wrapper function.
        """
        # 沒有用py_function包裝起來keras.utils.to_categorical就會報錯
        wrapper = tf.py_function(self.map_transform_image_and_label_py, (image, label), (tf.float32, tf.float32))
        image = tf.ensure_shape(wrapper[0], [28, 28, 1])
        label = tf.ensure_shape(wrapper[1], [10])
        return image, label

    def map_transform_image_and_label_py(self, image, label):
        """ Transform label and normalize image input.
        """
        # 正規化圖片資料
        image /= 255
        # 轉換label為tensor
        label = utils.to_categorical(label, 10)

        return image, label


