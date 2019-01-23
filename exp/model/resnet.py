import tensorflow as tf

from .base import BaseModel


def identity_block(input_tensor, filters, stage, block, training):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.layers.conv2d(inputs=input_tensor, filters=filters, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss,
            kernel_initializer=tf.variance_scaling_initializer(),
            padding='same', name=conv_name_base + '2a')
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2a',
            training=training)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, filters=filters, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss,
            kernel_initializer=tf.variance_scaling_initializer(),
            padding='same', name=conv_name_base + '2b')
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2b',
            training=training)

    x = x + input_tensor
    x = tf.nn.relu(x)
    return x


def conv_block(input_tensor, filters, stage, block, training, strides=(2, 2)):
    filters_in, filters_out, = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.layers.conv2d(inputs=input_tensor, filters=filters_in,
            kernel_regularizer=tf.nn.l2_loss,
            kernel_initializer=tf.variance_scaling_initializer(),
            strides=strides, padding='same', kernel_size=[3, 3], name=conv_name_base + '2a')
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2a',
            training=training)
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(inputs=x, filters=filters_out,
            kernel_regularizer=tf.nn.l2_loss,
            kernel_initializer=tf.variance_scaling_initializer(),
            kernel_size=[3, 3], padding='same', name=conv_name_base + '2b')
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2b',
            training=training)

    shortcut = tf.layers.conv2d(inputs=input_tensor, strides=strides,
            kernel_regularizer=tf.nn.l2_loss,
            kernel_initializer=tf.variance_scaling_initializer(),
            padding='same', kernel_size=[3, 3], filters=filters_out, name=conv_name_base + '1')
    shortcut = tf.layers.batch_normalization(shortcut, name=bn_name_base + '1',
            training=training)

    x = x + shortcut
    x = tf.nn.relu(x)
    return x


def _resnet20(input_tensor, training, data):
    input_shape = (32, 32, 3)

    if data == 'cifar100':
        filter1 = 128
        filter2 = 256
        filter3 = 512
        num_classes = 100
    elif data == 'cifar10':
        filter1 = 16
        filter2 = 32
        filter3 = 64
        num_classes = 10

    x = tf.layers.conv2d(inputs=input_tensor,
            kernel_regularizer=tf.nn.l2_loss,
            filters=filter1, kernel_size=[3, 3], padding='same', name='conv1')
    x = tf.layers.batch_normalization(x, name='bn_conv1',
            beta_regularizer=tf.nn.l2_loss,
            gamma_regularizer=tf.nn.l2_loss,
            training=training)
    x = tf.nn.relu(x)

    x = conv_block(x, [filter1, filter1], stage=2, block='a', strides=(1, 1), training=training)
    x = identity_block(x, filter1, stage=2, block='b', training=training)
    x = identity_block(x, filter1, stage=2, block='c', training=training)
    x = identity_block(x, filter1, stage=2, block='d', training=training)

    x = conv_block(x, [filter1, filter2], stage=3, block='a', training=training)
    x = identity_block(x, filter2, stage=3, block='b', training=training)
    x = identity_block(x, filter2, stage=3, block='c', training=training)
    x = identity_block(x, filter2, stage=3, block='d', training=training)

    x = conv_block(x, [filter2, filter3], stage=4, block='a', training=training)
    x = identity_block(x, filter3, stage=4, block='b', training=training)
    x = identity_block(x, filter3, stage=4, block='c', training=training)
    x = identity_block(x, filter3, stage=4, block='d', training=training)

    x = tf.layers.average_pooling2d(x, x.shape[1:3], 1)
    x = tf.reshape(x, (-1, filter3))
    outputs = tf.layers.dense(x, num_classes, activation=None, name='embeddings')
    return outputs


def _resnet56(input_tensor, training, data):
    input_shape = (32, 32, 3)

    if data == 'cifar100':
        filter1 = 128
        filter2 = 256
        filter3 = 512
        num_classes = 100
    elif data == 'cifar10':
        filter1 = 16
        filter2 = 32
        filter3 = 64
        num_classes = 10

    x = tf.layers.conv2d(inputs=input_tensor,
            kernel_regularizer=tf.nn.l2_loss,
            filters=filter1, kernel_size=[3, 3], padding='same', name='conv1')
    x = tf.layers.batch_normalization(x, name='bn_conv1',
            beta_regularizer=tf.nn.l2_loss,
            gamma_regularizer=tf.nn.l2_loss,
            training=training)
    x = tf.nn.relu(x)

    x = conv_block(x, [filter1, filter1], stage=2, block='a', strides=(1, 1), training=training)
    x = identity_block(x, filter1, stage=2, block='b', training=training)
    x = identity_block(x, filter1, stage=2, block='c', training=training)
    x = identity_block(x, filter1, stage=2, block='d', training=training)
    x = identity_block(x, filter1, stage=2, block='e', training=training)
    x = identity_block(x, filter1, stage=2, block='f', training=training)
    x = identity_block(x, filter1, stage=2, block='g', training=training)
    x = identity_block(x, filter1, stage=2, block='h', training=training)
    x = identity_block(x, filter1, stage=2, block='i', training=training)

    x = conv_block(x, [filter1, filter2], stage=3, block='a', training=training)
    x = identity_block(x, filter2, stage=3, block='b', training=training)
    x = identity_block(x, filter2, stage=3, block='c', training=training)
    x = identity_block(x, filter2, stage=3, block='d', training=training)
    x = identity_block(x, filter2, stage=3, block='e', training=training)
    x = identity_block(x, filter2, stage=3, block='f', training=training)
    x = identity_block(x, filter2, stage=3, block='g', training=training)
    x = identity_block(x, filter2, stage=3, block='h', training=training)
    x = identity_block(x, filter2, stage=3, block='i', training=training)

    x = conv_block(x, [filter2, filter3], stage=4, block='a', training=training)
    x = identity_block(x, filter3, stage=4, block='b', training=training)
    x = identity_block(x, filter3, stage=4, block='c', training=training)
    x = identity_block(x, filter3, stage=4, block='d', training=training)
    x = identity_block(x, filter3, stage=4, block='e', training=training)
    x = identity_block(x, filter3, stage=4, block='f', training=training)
    x = identity_block(x, filter3, stage=4, block='g', training=training)
    x = identity_block(x, filter3, stage=4, block='h', training=training)
    x = identity_block(x, filter3, stage=4, block='i', training=training)

    x = tf.layers.average_pooling2d(x, x.shape[1:3], 1)
    x = tf.reshape(x, (-1, filter3))
    # x = tf.layers.dense(x, 64, activation=tf.nn.relu, name='embeddings')
    outputs = tf.layers.dense(x, num_classes, name='embeddings')
    return outputs


def _resnet56_before_final_fc(input_tensor, training, data):
    input_shape = (32, 32, 3)

    if data == 'cifar100':
        filter1 = 128
        filter2 = 256
        filter3 = 512
        num_classes = 100
    elif data == 'cifar10':
        filter1 = 16
        filter2 = 32
        filter3 = 64
        num_classes = 10

    x = tf.layers.conv2d(inputs=input_tensor,
            kernel_regularizer=tf.nn.l2_loss,
            filters=filter1, kernel_size=[3, 3], padding='same', name='conv1')
    x = tf.layers.batch_normalization(x, name='bn_conv1',
            beta_regularizer=tf.nn.l2_loss,
            gamma_regularizer=tf.nn.l2_loss,
            training=training)
    x = tf.nn.relu(x)

    x = conv_block(x, [filter1, filter1], stage=2, block='a', strides=(1, 1), training=training)
    x = identity_block(x, filter1, stage=2, block='b', training=training)
    x = identity_block(x, filter1, stage=2, block='c', training=training)
    x = identity_block(x, filter1, stage=2, block='d', training=training)
    x = identity_block(x, filter1, stage=2, block='e', training=training)
    x = identity_block(x, filter1, stage=2, block='f', training=training)
    x = identity_block(x, filter1, stage=2, block='g', training=training)
    x = identity_block(x, filter1, stage=2, block='h', training=training)
    x = identity_block(x, filter1, stage=2, block='i', training=training)

    x = conv_block(x, [filter1, filter2], stage=3, block='a', training=training)
    x = identity_block(x, filter2, stage=3, block='b', training=training)
    x = identity_block(x, filter2, stage=3, block='c', training=training)
    x = identity_block(x, filter2, stage=3, block='d', training=training)
    x = identity_block(x, filter2, stage=3, block='e', training=training)
    x = identity_block(x, filter2, stage=3, block='f', training=training)
    x = identity_block(x, filter2, stage=3, block='g', training=training)
    x = identity_block(x, filter2, stage=3, block='h', training=training)
    x = identity_block(x, filter2, stage=3, block='i', training=training)

    x = conv_block(x, [filter2, filter3], stage=4, block='a', training=training)
    x = identity_block(x, filter3, stage=4, block='b', training=training)
    x = identity_block(x, filter3, stage=4, block='c', training=training)
    x = identity_block(x, filter3, stage=4, block='d', training=training)
    x = identity_block(x, filter3, stage=4, block='e', training=training)
    x = identity_block(x, filter3, stage=4, block='f', training=training)
    x = identity_block(x, filter3, stage=4, block='g', training=training)
    x = identity_block(x, filter3, stage=4, block='h', training=training)
    x = identity_block(x, filter3, stage=4, block='i', training=training)

    x = tf.layers.average_pooling2d(x, x.shape[1:3], 1)
    x = tf.reshape(x, (-1, filter3))
    return x


class ResNet56(BaseModel):
    def __init__(self, inputs, training, config):
        super(ResNet56, self).__init__(inputs, config)
        data = config.get('data', {}).get('name', 'cifar100')
        if data == 'cifar100':
            self.num_classes = 100
        elif data == 'cifar10':
            self.num_classes = 10

        self._outputs = _resnet56(self.inputs, training, data)

    @property
    def outputs(self):
        return self._outputs


class ResNet56Base(BaseModel):
    def __init__(self, inputs, training, config):
        super(ResNet56Base, self).__init__(inputs, config)
        data = config.get('data', {}).get('name', 'cifar100')
        if data == 'cifar100':
            self.num_classes = 100
        elif data == 'cifar10':
            self.num_classes = 10

        self._outputs = _resnet56_before_final_fc(self.inputs, training, data)

    @property
    def outputs(self):
        return self._outputs


class ResNet20(BaseModel):
    def __init__(self, inputs, training, config):
        super(ResNet20, self).__init__(inputs, config)
        data = config.get('data', {}).get('name', 'cifar100')
        if data == 'cifar100':
            self.num_classes = 100
        elif data == 'cifar10':
            self.num_classes = 10

        self._outputs = _resnet20(self.inputs, training, data)

    @property
    def outputs(self):
        return self._outputs
