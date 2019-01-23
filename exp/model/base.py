import tensorflow as tf
import numpy as np
import logging


def histogram_loss(features, classes, num_steps):
    step = 2 / (num_steps - 1)
    t = tf.reshape(tf.range(-1, 1 + step / 2, step), [-1, 1])
    tsize = t.get_shape().as_list()[0]

    classes_size = classes.get_shape().as_list()[0]
    classes_eq = tf.equal(
            tf.tile(tf.reshape(classes, (1, -1)), tf.constant([classes_size, 1])),
            tf.tile(tf.reshape(classes, (-1, 1)), tf.constant([1, classes_size])))
    dists = tf.matmul(features, features, transpose_b=True)
    _classes_eq_ones = tf.ones(classes_eq.shape)
    s_inds = tf.cast(tf.matrix_band_part(_classes_eq_ones, 0, -1) - tf.matrix_band_part(_classes_eq_ones, 0, 0), tf.bool)
    _classes_eq_s_inds = tf.boolean_mask(classes_eq, s_inds)
    pos_inds = tf.tile(tf.reshape(_classes_eq_s_inds, [1, -1]), tf.constant([tsize, 1]))
    neg_inds = tf.tile(tf.reshape(tf.logical_not(_classes_eq_s_inds), [1, -1]), tf.constant([tsize, 1]))
    pos_size = tf.reduce_sum(tf.cast(_classes_eq_s_inds, tf.float32))
    neg_size = tf.reduce_sum(tf.cast(tf.logical_not(_classes_eq_s_inds), tf.float32))
    s = tf.reshape(tf.boolean_mask(dists, s_inds), [1, -1])
    s_repeat = tf.tile(s, tf.constant([tsize, 1]))
    delta_repeat = tf.cast(tf.floor((s_repeat + 1.) / step) * step - 1., tf.float32)

    def histogram(inds, size):
        s_repeat_ = tf.cast(tf.identity(s_repeat), tf.float32)
        indsa = tf.less(tf.abs(delta_repeat - (t - step)), 1e-3) & inds
        indsb = tf.less(tf.abs(delta_repeat - t), 1e-3) & inds
        s_repeat_ = tf.where(~(indsb|indsa), tf.zeros_like(s_repeat_, dtype=tf.float32), s_repeat_)
        _s_repeat_lbound = s_repeat_ - t + tf.constant(step)
        s_repeat_ = tf.where(indsa, tf.divide(_s_repeat_lbound, step), s_repeat_)
        s_repeat_ = tf.where(indsb, tf.divide(-s_repeat_ + t + tf.constant(step), step), s_repeat_)

        return tf.divide(tf.reduce_sum(s_repeat_, axis=1), tf.cast(size, tf.float32))

    histogram_pos = histogram(pos_inds, pos_size)
    histogram_neg = histogram(neg_inds, neg_size)
    histogram_pos_repeat = tf.tile(tf.reshape(histogram_pos, [-1, 1]), tf.constant([1, histogram_pos.get_shape().as_list()[0]]))
    _hist_pos_ones = tf.ones(histogram_pos_repeat.shape)
    histogram_pos_inds = tf.cast(tf.matrix_band_part(_hist_pos_ones, -1, 0) - tf.matrix_band_part(_hist_pos_ones, 0, 0), tf.bool)
    histogram_pos_repeat = tf.where(histogram_pos_inds, tf.zeros_like(histogram_pos_repeat, dtype=tf.float32), histogram_pos_repeat)
    histogram_pos_cdf = tf.reduce_sum(histogram_pos_repeat, axis=0)
    loss = tf.reduce_sum(histogram_neg * histogram_pos_cdf)

    return loss


class BaseModel:
    def __init__(self, inputs, config):
        if config['task'] == 'train':
            self.inputs = inputs
        else:
            self.inputs = self.preprocess(inputs, config)

    def preprocess(self, inputs, config):
        def zca_whitening(x, principal_components):
            flat_x = tf.reshape(x, [-1, 32 * 32 * 3])
            x = tf.matmul(flat_x, principal_components)
            return tf.reshape(x, [-1, 32, 32, 3])

        def global_contrast_normalization(x, pixel_mean, pixel_stddev):
            x = (x - pixel_mean) / pixel_stddev
            return x

        if config.get('data', {}).get('name', 'cifar100') == 'cifar100':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        elif config.get('data', {}).get('name', 'cifar100') == 'cifar10':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        if config.get('data', {}).get('preprocess', {}).get('gcn'):
            pixel_mean = np.mean(x_train, axis=0)
            pixel_stddev = np.std(x_train, axis=0)

            inputs = global_contrast_normalization(inputs, pixel_mean, pixel_stddev)

        if config.get('data', {}).get('preprocess', {}).get('zca'):
            x_train = (x_train - pixel_mean) / pixel_stddev
            flat_x = np.reshape(x_train, (x_train.shape[0], -1))
            sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
            u, s, _ = np.linalg.svd(sigma)
            s_inv = 1. / np.sqrt(s + 1e-7)
            principal_components = np.dot(u * s_inv, u.T)

            inputs = zca_whitening(inputs, principal_components)
        return inputs

    @property
    def outputs(self):
        raise NotImplementedError

    def loss(self, labels, name='cross_entropy'):
        if name == 'cross_entropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, self.num_classes), logits=self._outputs))
        elif name == 'histogram':
            return histogram_loss(tf.nn.l2_normalize(self._outputs, axis=1), labels, num_steps=150)
