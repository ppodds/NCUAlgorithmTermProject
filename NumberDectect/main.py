import tensorflow as tf
from NumberDectect.commands.train import train
from NumberDectect.commands.info import info
from NumberDectect.dataset import *

import os

from NumberDectect.model import *

tf.compat.v1.enable_eager_execution()
print(f'Tensorflow version: {str(tf.__version__)}')
print(f"GPU : {str(tf.config.list_physical_devices('GPU'))}")


def main():
    data = get_input_data('train_image')
    train_dataset = DatasetWrapper(data)
    data = get_input_data('test_image')
    test_dataset = DatasetWrapper(data)
    train_dataset = train_dataset.get_dataset(128)
    test_dataset = test_dataset.get_dataset(128)

    train(os.path.join('Project', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
          , "Res9", train_dataset, test_dataset, None, 30)


main()
