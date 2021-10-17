import PIL.Image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib


def get_training_data():
    # train_dir = pathlib.Path('./data/training')
    # val_dir = pathlib.Path('./data/validation')
    # test_dir = pathlib.Path('./data/test')
    #
    # train_ds = image_dataset_from_directory(train_dir, image_size=(100, 40), batch_size=32)
    # val_ds = image_dataset_from_directory(val_dir, image_size=(100, 40), batch_size=32)
    # test_ds = image_dataset_from_directory(test_dir, image_size=(100, 40), batch_size=32)
    #
    # return train_ds, val_ds, test_ds
    data_set = ImageDataGenerator(rescale=1/255.,
                                  )

    train_gen = data_set.flow_from_directory('data/training/', target_size=(1000, 400), class_mode='categorical', batch_size=32)
    val_gen = data_set.flow_from_directory('data/validation/', target_size=(1000, 400), class_mode='categorical', batch_size=32)
    test_gen = data_set.flow_from_directory('data/test/', target_size=(1000, 400), class_mode='categorical', batch_size=32)
    return (train_gen, val_gen, test_gen)
