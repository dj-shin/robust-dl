import tensorflow as tf
import numpy as np


def cifar10(batch_size, config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    def zca_whitening(x, principal_components):
        x_shape = x.shape
        flat_x = np.reshape(x, (-1, 32 * 32 * 3))
        x = np.matmul(flat_x, principal_components)
        return np.reshape(x, x_shape)

    def global_contrast_normalization(x, pixel_mean, pixel_stddev):
        x = (x - pixel_mean) / pixel_stddev
        return x

    if config.get('preprocess', {}).get('gcn'):
        pixel_mean = np.mean(x_train, axis=0)
        pixel_stddev = np.std(x_train, axis=0)

        x_train = global_contrast_normalization(x_train, pixel_mean, pixel_stddev)
        x_test = global_contrast_normalization(x_test, pixel_mean, pixel_stddev)

    if config.get('preprocess', {}).get('zca'):
        flat_x = np.reshape(x_train, (x_train.shape[0], -1))
        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = np.linalg.svd(sigma)
        s_inv = 1. / np.sqrt(s + 1e-7)
        principal_components = np.dot(u * s_inv, u.T)

        x_train = zca_whitening(x_train, principal_components)
        x_test = zca_whitening(x_test, principal_components)

    def train_preprocess(x, y):
        x = tf.pad(x, paddings=[[4, 4], [4, 4], [0, 0]])
        x = tf.random_crop(x, size=[32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        return x, y

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0])
    train_dataset = train_dataset.map(lambda x, y: train_preprocess(x, y), num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(x_train.shape[0] // batch_size + 1)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(x_test.shape[0] // batch_size + 1)

    return train_dataset, test_dataset
