import os

import tensorflow as tf
from tensorflow.keras import utils

from PIL import Image


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
    # 找出所有資料並確認路徑和label
    for basedir in os.listdir(os.path.join('ChineseNumDataset', folder_name)):
        for label in os.listdir(os.path.join('ChineseNumDataset', folder_name, basedir)):
            for file_name in os.listdir(os.path.join('ChineseNumDataset', folder_name, basedir, label)):
                image_paths.append((os.path.join('ChineseNumDataset', folder_name, basedir, label, file_name)))
                image_label.append(label)
    return image_paths, image_label


def normalize_image_format(image_paths):
    """ Check images format of dataset. If the image format is not equals, changed it.

    Args:
        image_paths (str): Image paths of a dataset.
    """
    # 事前轉檔，統一檔案格式
    for image_path in image_paths:
        img = Image.open(image_path)
        if not img.mode == 'P':
            img = img.convert('P')
            img.save(image_path)
            print('Image format has changed:', image_path)


class DatasetWrapper:
    """ Wrapper class for image data and label.

    """

    def __init__(self, image_data):
        self.image_data = image_data

    def get_dataset(self, batch_size):
        """ Get input dataset.
        """
        # 讀取資料集並轉為dataset物件
        dataset = tf.data.Dataset.from_tensor_slices(self.image_data)
        # 讀取資料集的圖片
        dataset = dataset.map(self.map_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # 正規化圖片並處理label
        dataset = dataset.map(self.map_transform_image_and_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # 分出batch
        dataset = dataset.batch(batch_size)
        # 效能優化處理
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset

    def map_load_image(self, image_path, label):
        """ Load image from image path and transform to tensor.
        """
        # 讀取檔案
        image_raw = tf.io.read_file(image_path)
        # 解析bmp
        image = tf.io.decode_image(image_raw, channels=0)
        return image, label

    def map_transform_image_and_label(self, image, label):
        """ Tensorflow wrapper function.
        """
        # 沒有用py_function包裝起來keras.utils.to_categorical就會報錯
        wrapper = tf.py_function(self.map_transform_image_and_label_py, (image, label), (tf.float32, tf.float32))
        # pyfunction會丟失形狀，ensure_shape可以讓形狀重新被確定
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


