import tensorflow as tf

from .base import BaseModel


def _vgg16(input_tensor, training):
    input_shape = (32, 32, 3)
    mean = 121.936 / 255
    std = 68.389 / 255
    inputs = (input_tensor - mean) / (std + 1e-7)

    x = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack1_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack1_bn1', training=training)
    x = tf.layers.dropout(x, rate=0.3, training=training)

    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack1_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack1_bn2', training=training)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack1_pool')

    # Stack 2
    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack2_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack2_bn1', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=128, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack2_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack2_bn2', training=training)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack2_pool')

    # Stack 3
    x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack3_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack3_bn1', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack3_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack3_bn2', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=256, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack3_conv3')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack3_bn3', training=training)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack3_pool')

    # Stack 4
    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack4_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack4_bn1', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack4_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack4_bn2', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack4_conv3')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack4_bn3', training=training)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack4_pool')

    # Stack 5
    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack5_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack5_bn1', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack5_conv2')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack5_bn2', training=training)
    x = tf.layers.dropout(x, rate=0.4, training=training)

    x = tf.layers.conv2d(inputs=x, filters=512, kernel_size=[3, 3],
            kernel_regularizer=tf.nn.l2_loss, padding='same', name='stack5_conv3')
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack5_bn3', training=training)
    x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2), name='stack5_pool')
    x = tf.layers.dropout(x, rate=0.5, training=training)

    # End
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 512)
    x = tf.nn.relu(x)
    x = tf.layers.batch_normalization(x, name='stack6_bn1', training=training)
    x = tf.layers.dropout(x, rate=0.5, training=training)
    x = tf.layers.dense(x, 100)
    return x


class Vgg16(BaseModel):
    def __init__(self, inputs, training, config):
        super(Vgg16, self).__init__(inputs, config)
        self.num_classes = 100
        self._outputs = _vgg16(self.inputs, training)

    @property
    def outputs(self):
        return self._outputs
