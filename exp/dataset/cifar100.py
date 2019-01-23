import tensorflow as tf
import numpy as np


def cifar100(batch_size, config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
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

    if config.get('data', {}).get('preprocess', {}).get('gcn'):
        pixel_mean = np.mean(x_train, axis=0)
        pixel_stddev = np.std(x_train, axis=0)

        x_train = global_contrast_normalization(x_train, pixel_mean, pixel_stddev)
        x_test = global_contrast_normalization(x_test, pixel_mean, pixel_stddev)

    if config.get('data', {}).get('preprocess', {}).get('zca'):
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


def generate_npairs_batch(x_train, y_train, batch_size, num_classes):
    total_count = x_train.shape[0]

    x_train_separate = [list() for _ in range(num_classes)]
    y_train_separate = [list() for _ in range(num_classes)]
    for i in range(x_train.shape[0]):
        label = int(y_train[i])
        x_train_separate[label].append(x_train[i])
        y_train_separate[label].append(y_train[i])

    x_train = [None for _ in range(num_classes)]
    y_train = [None for _ in range(num_classes)]
    for i in range(num_classes):
        x_train[i] = np.stack(x_train_separate[i])
        y_train[i] = np.stack(y_train_separate[i])

    x_train_anchor = [np.copy(x) for x in x_train]
    x_train_positive = [np.copy(x) for x in x_train]

    for i in range(num_classes):
        np.random.shuffle(x_train_anchor[i])
        np.random.shuffle(x_train_positive[i])
    x_train_anchor = np.stack(x_train_anchor)
    x_train_positive = np.stack(x_train_positive)

    x_train_anchor = np.swapaxes(x_train_anchor, 0, 1)
    x_train_positive = np.swapaxes(x_train_positive, 0, 1)

    batches = np.zeros((total_count // (batch_size // 2), num_classes), dtype=int)
    for i in range(total_count // (batch_size // 2)):
        p = (np.exp(total_count / num_classes - np.sum(batches, axis=0)) - 1.) / num_classes
        must = (total_count / num_classes - np.sum(batches, axis=0) >= total_count // (batch_size // 2) - i)
        if np.any(must):
            np.putmask(p, must, 1e8)
        p = p / np.sum(p)
        try:
            selection = np.random.choice(range(num_classes), size=batch_size // 2, replace=False, p=p)
        except Exception as exc:
            print(p)
            raise exc
        batches[i, selection] = 1

    idx = np.where(batches, np.cumsum(batches, axis=0), np.zeros(shape=batches.shape)).astype(int)

    epoch = np.argwhere(idx > 0)
    batch_x_anchor = list()
    batch_x_positive = list()
    labels = list()
    for i in range(epoch.shape[0]):
        yield (x_train_anchor[idx[epoch[i][0], epoch[i][1]] - 1, epoch[i][1], :, :, :],
                x_train_positive[idx[epoch[i][0], epoch[i][1]] - 1, epoch[i][1], :, :, :],
                epoch[i][1])


def generate_mixed_npairs_batch(x_train, x_test, y_train, y_test, batch_size, num_classes):
    total_count = min([x_train.shape[0], x_test.shape[0]])

    x_train_separate = [list() for _ in range(num_classes)]
    y_train_separate = [list() for _ in range(num_classes)]
    x_test_separate = [list() for _ in range(num_classes)]
    y_test_separate = [list() for _ in range(num_classes)]

    for i in range(x_train.shape[0]):
        label = int(y_train[i])
        x_train_separate[label].append(x_train[i])
        y_train_separate[label].append(y_train[i])

    for i in range(x_test.shape[0]):
        label = int(y_test[i])
        x_test_separate[label].append(x_test[i])
        y_test_separate[label].append(y_test[i])

    x_train = [None for _ in range(num_classes)]
    y_train = [None for _ in range(num_classes)]
    for i in range(num_classes):
        x_train[i] = np.stack(x_train_separate[i])
        y_train[i] = np.stack(y_train_separate[i])

    x_train_anchor = [np.copy(x) for x in x_train]
    x_train_positive = [np.copy(x) for x in x_train]

    for i in range(num_classes):
        np.random.shuffle(x_train_anchor[i])
        np.random.shuffle(x_train_positive[i])
    x_train_anchor = np.stack(x_train_anchor)
    x_train_positive = np.stack(x_train_positive)

    x_train_anchor = np.swapaxes(x_train_anchor, 0, 1)
    x_train_positive = np.swapaxes(x_train_positive, 0, 1)

    x_test = [None for _ in range(num_classes)]
    y_test = [None for _ in range(num_classes)]
    for i in range(num_classes):
        x_test[i] = np.stack(x_test_separate[i])
        y_test[i] = np.stack(y_test_separate[i])

    x_test_anchor = [np.copy(x) for x in x_test]
    x_test_positive = [np.copy(x) for x in x_test]

    for i in range(num_classes):
        np.random.shuffle(x_test_anchor[i])
        np.random.shuffle(x_test_positive[i])
    x_test_anchor = np.stack(x_test_anchor)
    x_test_positive = np.stack(x_test_positive)

    x_test_anchor = np.swapaxes(x_test_anchor, 0, 1)
    x_test_positive = np.swapaxes(x_test_positive, 0, 1)

    batches = np.zeros((total_count // (batch_size // 2), num_classes), dtype=int)
    for i in range(total_count // (batch_size // 2)):
        p = (np.exp(total_count / num_classes - np.sum(batches, axis=0)) - 1.) / num_classes
        must = (total_count / num_classes - np.sum(batches, axis=0) >= total_count // (batch_size // 2) - i)
        if np.any(must):
            np.putmask(p, must, 1e8)
        p = p / np.sum(p)
        try:
            selection = np.random.choice(range(num_classes), size=batch_size // 2, replace=False, p=p)
        except Exception as exc:
            print(p)
            raise exc
        batches[i, selection] = 1

    idx = np.where(batches, np.cumsum(batches, axis=0), np.zeros(shape=batches.shape)).astype(int)

    epoch = np.argwhere(idx > 0)
    batch_x_anchor = list()
    batch_x_positive = list()
    labels = list()
    for i in range(epoch.shape[0]):
        yield (x_test_anchor[idx[epoch[i][0], epoch[i][1]] - 1, epoch[i][1], :, :, :],
                x_train_positive[idx[epoch[i][0], epoch[i][1]] - 1, epoch[i][1], :, :, :],
                epoch[i][1])


def cifar100_npairs(batch_size, config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
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

    if config.get('data', {}).get('preprocess', {}).get('gcn'):
        pixel_mean = np.mean(x_train, axis=0)
        pixel_stddev = np.std(x_train, axis=0)

        x_train = global_contrast_normalization(x_train, pixel_mean, pixel_stddev)
        x_test = global_contrast_normalization(x_test, pixel_mean, pixel_stddev)

    if config.get('data', {}).get('preprocess', {}).get('zca'):
        flat_x = np.reshape(x_train, (x_train.shape[0], -1))
        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = np.linalg.svd(sigma)
        s_inv = 1. / np.sqrt(s + 1e-7)
        principal_components = np.dot(u * s_inv, u.T)

        x_train = zca_whitening(x_train, principal_components)
        x_test = zca_whitening(x_test, principal_components)

    def train_preprocess(x_anc, x_pos, y):
        x_anc = tf.pad(x_anc, paddings=[[4, 4], [4, 4], [0, 0]])
        x_anc = tf.random_crop(x_anc, size=[32, 32, 3])
        x_anc = tf.image.random_flip_left_right(x_anc)

        x_pos = tf.pad(x_pos, paddings=[[4, 4], [4, 4], [0, 0]])
        x_pos = tf.random_crop(x_pos, size=[32, 32, 3])
        x_pos = tf.image.random_flip_left_right(x_pos)

        return x_anc, x_pos, y

    num_classes = 100

    train_dataset = tf.data.Dataset.from_generator(lambda: generate_npairs_batch(x_train, y_train, batch_size, num_classes),
            (tf.float32, tf.float32, tf.int64),
            (tf.TensorShape([32, 32, 3]), tf.TensorShape([32, 32, 3]), tf.TensorShape([])))
    test_dataset = tf.data.Dataset.from_generator(lambda: generate_npairs_batch(x_test, y_test, batch_size, num_classes),
            (tf.float32, tf.float32, tf.int64),
            (tf.TensorShape([32, 32, 3]), tf.TensorShape([32, 32, 3]), tf.TensorShape([])))

    train_dataset = train_dataset.map(lambda x_anc, x_pos, y: train_preprocess(x_anc, x_pos, y), num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size // 2)
    train_dataset = train_dataset.prefetch(x_train.shape[0] // (batch_size // 2))

    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size // 2)
    test_dataset = test_dataset.prefetch(x_test.shape[0] // (batch_size // 2))

    return train_dataset, test_dataset

def cifar100_histogram(batch_size, config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
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

    if config.get('data', {}).get('preprocess', {}).get('gcn'):
        pixel_mean = np.mean(x_train, axis=0)
        pixel_stddev = np.std(x_train, axis=0)

        x_train = global_contrast_normalization(x_train, pixel_mean, pixel_stddev)
        x_test = global_contrast_normalization(x_test, pixel_mean, pixel_stddev)

    if config.get('data', {}).get('preprocess', {}).get('zca'):
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

    num_classes = 100

    def sampler(x_train, y_train, batch_size):
        labels_unique = np.unique(y_train)
        num_batch = x_train.shape[0] // batch_size
        for i in range(num_batch):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int64)

            while inds.shape[0] < batch_size:
                sample_label = np.random.choice(labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                subsample_size = np.random.choice(range(5, 11))
                sample_label_ids = np.argwhere(np.in1d(y_train, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:subsample_size]
                inds = np.append(inds, subsample)

            inds = inds[:batch_size]
            for p in zip(x_train[inds, :], y_train[inds, :]):
                yield p

    train_dataset = tf.data.Dataset.from_generator(lambda: sampler(x_train, y_train, batch_size),
            (tf.float32, tf.int64),
            (tf.TensorShape([32, 32, 3]), tf.TensorShape([1])))

    train_dataset = train_dataset.map(lambda x, y: train_preprocess(x, y), num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(x_train.shape[0] // batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(x_test.shape[0] // batch_size)

    return train_dataset, test_dataset


def cifar100_npairs_mixed(batch_size, config):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
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

    if config.get('data', {}).get('preprocess', {}).get('gcn'):
        pixel_mean = np.mean(x_train, axis=0)
        pixel_stddev = np.std(x_train, axis=0)

        x_train = global_contrast_normalization(x_train, pixel_mean, pixel_stddev)
        x_test = global_contrast_normalization(x_test, pixel_mean, pixel_stddev)

    if config.get('data', {}).get('preprocess', {}).get('zca'):
        flat_x = np.reshape(x_train, (x_train.shape[0], -1))
        sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
        u, s, _ = np.linalg.svd(sigma)
        s_inv = 1. / np.sqrt(s + 1e-7)
        principal_components = np.dot(u * s_inv, u.T)

        x_train = zca_whitening(x_train, principal_components)
        x_test = zca_whitening(x_test, principal_components)

    def train_preprocess(x_anc, x_pos, y):
        x_anc = tf.pad(x_anc, paddings=[[4, 4], [4, 4], [0, 0]])
        x_anc = tf.random_crop(x_anc, size=[32, 32, 3])
        x_anc = tf.image.random_flip_left_right(x_anc)

        x_pos = tf.pad(x_pos, paddings=[[4, 4], [4, 4], [0, 0]])
        x_pos = tf.random_crop(x_pos, size=[32, 32, 3])
        x_pos = tf.image.random_flip_left_right(x_pos)

        return x_anc, x_pos, y

    num_classes = 100

    mixed_dataset = tf.data.Dataset.from_generator(lambda: generate_mixed_npairs_batch(x_train, x_test, y_train, y_test, batch_size, num_classes),
            (tf.float32, tf.float32, tf.int64),
            (tf.TensorShape([32, 32, 3]), tf.TensorShape([32, 32, 3]), tf.TensorShape([])))
    mixed_dataset = mixed_dataset.batch(batch_size // 2)
    mixed_dataset = mixed_dataset.prefetch(x_test.shape[0] // (batch_size // 2))

    return mixed_dataset
