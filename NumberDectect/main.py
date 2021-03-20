from NumberDectect.commands.train import train
from NumberDectect.commands.info import info
from NumberDectect.commands.evaluate import evaluate_project
from NumberDectect.dataset import *

from NumberDectect.model import *

tf.compat.v1.enable_eager_execution()
print(f'Tensorflow version: {str(tf.__version__)}')
print(f"GPU : {str(tf.config.list_physical_devices('GPU'))}")

print('Input your operation')
print('1: Train model')
print('2: Evaluate model')
print('3: Get model info')
user_in = input()

if user_in == '1':
    model_path = input('Model path (where you want to place or your existed model file)')
    model_type = input('Model type (Res152/Res9) \n'
                       ' Note: If you choose a existed model, you can type anything.')
    try:
        epochs = int(input('Epochs (number)'))
        batch_size = int(input('Batch size (number)'))
    except ValueError:
        print('Integer only...')
        exit()
    data = get_input_data('train_image')
    train_dataset = DatasetWrapper(data)
    data = get_input_data('test_image')
    test_dataset = DatasetWrapper(data)
    train_dataset = train_dataset.get_dataset(batch_size)
    test_dataset = test_dataset.get_dataset(batch_size)
    train(model_path, model_type, test_dataset, test_dataset, epochs)
elif user_in == '2':
    model_path = input('Model path (where you want to place or your existed model file)')
    try:
        batch_size = int(input('Batch size (number)'))
    except ValueError:
        print('Integer only...')
        exit()
    data = get_input_data('test_image')
    test_dataset = DatasetWrapper(data)
    test_dataset = test_dataset.get_dataset(batch_size)
    evaluate_project(model_path, test_dataset)
elif user_in == '3':
    generate_pic = bool(int(input('Need to generate pic? (0/1) \n'
                                  'Note: If you type 0 equals no.')))
    model_type = input('Model type (Res152/Res9)')
    info(generate_pic, model_type)
else:
    print('Unknown command.')
    print('Exit...')
