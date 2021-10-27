from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_training_data_10sec():
    data_set = ImageDataGenerator(rescale=1. / 255)

    train_gen = data_set.flow_from_directory('data/training/', target_size=(100, 40), class_mode='categorical',
                                             batch_size=64)
    val_gen = data_set.flow_from_directory('data/validation/', target_size=(100, 40), class_mode='categorical',
                                           batch_size=64)
    test_gen = data_set.flow_from_directory('data/test/', target_size=(100, 40), class_mode='categorical',
                                            batch_size=64)
    return (train_gen, val_gen, test_gen, train_gen.class_indices.keys())


def get_training_data_3sec():
    data_set = ImageDataGenerator(rescale=1. / 255)

    train_gen = data_set.flow_from_directory('data3sec/training/', target_size=(30, 40), class_mode='categorical',
                                             batch_size=64)
    val_gen = data_set.flow_from_directory('data3sec/validation/', target_size=(30, 40), class_mode='categorical',
                                           batch_size=64)
    test_gen = data_set.flow_from_directory('data3sec/test/', target_size=(30, 40), class_mode='categorical',
                                            batch_size=64)
    return (train_gen, val_gen, test_gen, train_gen.class_indices.keys())
