import tensorflow as tf
import numpy as np


def get_dataset(dataset_name, dataset_type):
    name_map = {
        'cifar10': tf.keras.datasets.cifar10.load_data,
        'cifar100': tf.keras.datasets.cifar100.load_data,
    }
    (x_train, y_train), (x_test, y_test) = name_map[dataset_name]()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if dataset_type == 'train':
        return x_train, y_train
    elif dataset_type == 'test':
        return x_test, y_test
