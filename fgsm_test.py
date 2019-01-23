import tensorflow as tf
import numpy as np
import os

from random import shuffle

from exp.model.factory import get_model
from exp.dataset.cifar100 import cifar100_npairs_mixed

from tensorflow.contrib.losses.python import metric_learning as metric_loss_ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def get_batch(x_train, y_train, x_test, y_test, batch_size):
    num_classes = 100
    x_train_separate = [list() for _ in range(num_classes)]
    x_test_separate = [list() for _ in range(num_classes)]

    for i in range(x_train.shape[0]):
        label = int(y_train[i])
        x_train_separate[label].append(x_train[i])

    for i in range(x_test.shape[0]):
        label = int(y_test[i])
        x_test_separate[label].append(x_test[i])

    for i in range(num_classes):
        shuffle(x_train_separate[i])
        shuffle(x_test_separate[i])
    
    labels = np.random.choice(num_classes, size=batch_size//2, replace=False)
    anchor = np.asarray([x_test_separate[c][0] for c in labels])
    positive = np.asarray([x_train_separate[c][0] for c in labels])
    return anchor, positive, labels


if __name__ == '__main__':
    config = {
        'task': 'predict',
        'model': {
            'name': 'resnet',
            'type': 'npairs',
            'path': 'result/checkpoint/1107-145400/cifar100.resnet_base.ensemble.epoch=210.ckpt',
        },
        'data': {
            'name': 'cifar100',
            'type': 'test',
            'preprocess': {
                'zca': False,
                'gcn': False,
            },
            'source': 'test',
        }
    }

    # model setup
    batch_size = 100
    inputs_ph = tf.placeholder(tf.float32, shape=(batch_size // 2, 32, 32, 3), name='inputs')
    pivot_ph = tf.placeholder(tf.float32, shape=(batch_size // 2, 32, 32, 3), name='pivots')
    labels_ph = tf.placeholder(tf.int64, shape=(batch_size // 2))

    with tf.variable_scope('model') as scope:
        embedding = get_model('resnet')(inputs_ph, False, config).outputs
    with tf.variable_scope(scope, reuse=True):
        embedding_pivot = get_model('resnet')(pivot_ph, False, config).outputs

    loss = metric_loss_ops.npairs_loss(labels_ph, embedding, embedding_pivot, reg_lambda=0.)
    grad = tf.gradients(loss, inputs_ph)[0][0]
    
    saver = tf.train.Saver()

    # ds = cifar100_npairs_mixed(batch_size, config)

    # it = ds.make_initializable_iterator()
    # batch = it.get_next()

    # prepare embeddings
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, config['model']['path'])
    # sess.run(it.initializer)

    grad_size = tf.norm(tf.reshape(grad, [-1]))

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    while True:
        # batch_anchor, batch_positive, batch_labels = sess.run(batch)
        batch_anchor, batch_positive, batch_labels = get_batch(x_train, y_train, x_test, y_test, batch_size)

        # calculate gradient based on naive method
        grad_size_val, grad_naive, loss_naive = sess.run([grad_size, grad, loss], feed_dict={
            inputs_ph: batch_anchor,
            pivot_ph: batch_positive,
            labels_ph: batch_labels,
        })
        grad_naive = grad_naive * (batch_size // 2)
        print(grad_size_val)
        if grad_size_val > 1e-1:
            break

    # calculate gradient based on optimzied method
    prod = tf.matmul(tf.reshape(embedding[0], [1, -1]), embedding_pivot, transpose_a=False, transpose_b=True)
    loss_opt = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(indices=0, depth=batch_size // 2), logits=prod)
    grad_reduced = tf.gradients(loss_opt, inputs_ph)[0][0]

    grad_opt = sess.run(grad_reduced, feed_dict={
        inputs_ph: batch_anchor,
        pivot_ph: batch_positive,
        labels_ph: batch_labels,
    })

    # compare
    assert np.allclose(grad_naive, grad_opt, rtol=1e-2, atol=1e-3)

    # print(np.nonzero(grad_opt))
