from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def get_training_data():
    data_set = ImageDataGenerator()
    train_it = data_set.flow_from_directory('data/training/', class_mode='categorical', batch_size=3)
    val_it = data_set.flow_from_directory('data/validation/', class_mode='categorical', batch_size=2)
    test_it = data_set.flow_from_directory('data/test/', class_mode='categorical', batch_size=2)
    return (train_it, val_it, test_it)
