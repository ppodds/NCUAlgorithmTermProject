import tensorflow as tf
from NumberDectect.commands import train

print(f'Tensorflow version: {str(tf.__version__)}')
print(f"GPU : {str(tf.config.list_physical_devices('GPU'))}")


def main():
    train.train('weights.{epoch:02d}-{val_loss:.2f}.hdf5', "Res9", train.load_dataset(), 128, 30)


main()
