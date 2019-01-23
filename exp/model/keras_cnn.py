import tensorflow as tf
import numpy as np


from .base import BaseModel


def _keras_cnn(inputs_ph, training_ph):
    x = tf.layers.conv2d(inputs=inputs_ph, filters=32, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack1_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='valid', name='stack1_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack1_pool')
    x = tf.layers.dropout(x, rate=0.25, training=training_ph)

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack2_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='valid', name='stack2_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack2_pool')
    x = tf.layers.dropout(x, rate=0.25, training=training_ph)

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 512)
    x = tf.nn.relu(x)
    x = tf.layers.dropout(x, rate=0.5, training=training_ph)
    x = tf.layers.dense(x, 10)

    return x


class KerasCNN(BaseModel):
    def __init__(self, inputs, training, config):
        super(KerasCNN, self).__init__(inputs, config)
        self.num_classes = 10
        self._outputs = _keras_cnn(self.inputs, training)

    @property
    def outputs(self):
        return self._outputs

    def loss(self, labels, name='cross_entropy'):
        if name == 'cross_entropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, self.num_classes), logits=self._outputs))
