import tensorflow as tf
import numpy as np


from .base import BaseModel


def _ELU_small(inputs_ph, training_ph, num_classes):
    x = tf.layers.conv2d(inputs=inputs_ph, filters=192, kernel_size=[5, 5],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack1_conv1')
    x = tf.layers.batch_normalization(x, name='stack1_bn1', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack1_pool')

    x = tf.layers.conv2d(inputs=x, filters=192, kernel_size=[1, 1],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack2_conv1')
    x = tf.layers.batch_normalization(x, name='stack2_bn1', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.conv2d(inputs=x, filters=240, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack2_conv2')
    x = tf.layers.batch_normalization(x, name='stack2_bn2', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack2_pool')
    x = tf.layers.dropout(x, rate=0.1, training=training_ph)

    x = tf.layers.conv2d(inputs=x, filters=240, kernel_size=[1, 1],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack3_conv1')
    x = tf.layers.batch_normalization(x, name='stack3_bn1', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.conv2d(inputs=x, filters=260, kernel_size=[2, 2],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack3_conv2')
    x = tf.layers.batch_normalization(x, name='stack3_bn2', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack3_pool')
    x = tf.layers.dropout(x, rate=0.2, training=training_ph)

    x = tf.layers.conv2d(inputs=x, filters=260, kernel_size=[1, 1],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack4_conv1')
    x = tf.layers.batch_normalization(x, name='stack4_bn1', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.conv2d(inputs=x, filters=280, kernel_size=[2, 2],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack4_conv2')
    x = tf.layers.batch_normalization(x, name='stack4_bn2', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack4_pool')
    x = tf.layers.dropout(x, rate=0.3, training=training_ph)

    x = tf.layers.conv2d(inputs=x, filters=280, kernel_size=[1, 1],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack5_conv1')
    x = tf.layers.batch_normalization(x, name='stack5_bn1', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.conv2d(inputs=x, filters=300, kernel_size=[2, 2],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack5_conv2')
    x = tf.layers.batch_normalization(x, name='stack5_bn2', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack5_pool')
    x = tf.layers.dropout(x, rate=0.4, training=training_ph)

    x = tf.layers.conv2d(inputs=x, filters=300, kernel_size=[1, 1],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack6_conv1')
    x = tf.layers.batch_normalization(x, name='stack6_bn1', training=training_ph)
    x = tf.nn.elu(x)
    x = tf.layers.dropout(x, rate=0.5, training=training_ph)

    x = tf.layers.conv2d(inputs=x, filters=num_classes, kernel_size=[1, 1],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack7_conv1')

    return tf.layers.flatten(x)


class ELU(BaseModel):
    def __init__(self, inputs, training, config):
        super(ELU, self).__init__(inputs, config)
        data = config.get('data', {}).get('name', 'cifar100')
        if data == 'cifar100':
            self.num_classes = 100
        elif data == 'cifar10':
            self.num_classes = 10

        if config['model'].get('size', 'small') == 'small':
            self._outputs = _ELU_small(self.inputs, training, self.num_classes)
        else:
            raise NotImplementedError

    @property
    def outputs(self):
        return self._outputs

    def loss(self, labels, name='cross_entropy'):
        if name == 'cross_entropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, self.num_classes), logits=self._outputs))
