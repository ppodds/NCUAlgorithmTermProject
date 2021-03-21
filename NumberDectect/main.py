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
print('4: Check image format')
user_in = input()

if user_in == '1':
    model_path = input('Model path (where you want to place or your existed model file)\n')
    model_type = input('Model type (Res152/Res9) \n'
                       ' Note: If you choose a existed model, you can type anything.\n')
    try:
        epochs = int(input('Epochs (number)\n'))
        batch_size = int(input('Batch size (number)\n'))
    except ValueError:
        print('Integer only...')
        exit()
    check_dataset = bool(int(input('Need to check image format? (0/1) \n'
                                   'Note: If you type 0 equals no.\n')))
    print('Preparing dataset... It may take some time...')
    train_data = get_input_data('train_image')
    test_data = get_input_data('test_image')
    if check_dataset:
        normalize_image_format(train_data[0])
        normalize_image_format(test_data[0])
    train_dataset = DatasetWrapper(train_data)
    test_dataset = DatasetWrapper(test_data)
    train_dataset = train_dataset.get_dataset(batch_size)
    test_dataset = test_dataset.get_dataset(batch_size)
    train(model_path, model_type, test_dataset, test_dataset, epochs)
elif user_in == '2':
    model_path = input('Model path\n')
    try:
        batch_size = int(input('Batch size (number)\n'))
    except ValueError:
        print('Integer only...')
        exit()
    check_dataset = bool(int(input('Need to check image format? (0/1) \n'
                                   'Note: If you type 0 equals no.\n')))
    print('Preparing dataset... It may take some time...')
    data = get_input_data('test_image')
    if check_dataset:
        normalize_image_format(data[0])
    test_dataset = DatasetWrapper(data)
    test_dataset = test_dataset.get_dataset(batch_size)
    evaluate_project(model_path, test_dataset)
elif user_in == '3':
    generate_pic = bool(int(input('Need to generate pic? (0/1) \n'
                                  'Note: If you type 0 equals no.\n')))
    model_type = input('Model type (Res152/Res9)\n')
    info(generate_pic, model_type)
elif user_in == '4':
    print('Checking images format. It may take some time...')
    train_data = get_input_data('train_image')
    test_data = get_input_data('test_image')
    normalize_image_format(train_data[0])
    normalize_image_format(test_data[0])
    print('Finish!')
else:
    print('Unknown command.')
    print('Exit...')
