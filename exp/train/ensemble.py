import tensorflow as tf
import numpy as np
import os
import logging
import json

from exp.model.factory import get_model
from exp.train.utils import get_optimizer
from exp.dataset.cifar100 import cifar100_npairs

from tensorflow.contrib.losses.python import metric_learning as metric_loss_ops

GRIDS = 20
def core_function1d(x, y, grids=GRIDS):
    return tf.maximum((1. / (grids - 1.)) - tf.abs(tf.subtract(x, y)), 0)


def core_function2d(x1, x2, y1, y2, grids1 = GRIDS, grids2 = GRIDS):
    return core_function1d(x1, y1, grids1) + core_function1d(x2, y2, grids1)


def entropy1d(x, grids = GRIDS):
    shape1 = [x.get_shape().as_list()[0], 1, x.get_shape().as_list()[1]]
    shape2 = [1, grids, 1]

    gx = tf.linspace(0.0, 1.0, grids)

    X = tf.reshape(x, shape1)
    GX = tf.reshape(gx, shape2)

    mapping = core_function1d(GX, X, grids)
    mapping = tf.reduce_sum(mapping, 0)
    mapping = tf.add(mapping, 1e-10)
    mapping_normalized = tf.divide(mapping, tf.reduce_sum(mapping, 0, keepdims = True))

    entropy = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(mapping_normalized, tf.log(mapping_normalized * grids)), 0)))

    return entropy

def entropy2d(x, y, gridsx = GRIDS, gridsy = GRIDS):
    batch_size = x.get_shape().as_list()[0]
    x_szie = x.get_shape().as_list()[1]
    y_size = y.get_shape().as_list()[1]

    gx = tf.linspace(0.0, 1.0, gridsx)
    gy = tf.linspace(0.0, 1.0, gridsy)

    X = tf.reshape(x, [batch_size, 1, 1, x_szie, 1])
    Y = tf.reshape(y, [batch_size, 1, 1, 1, y_size])

    GX = tf.reshape(gx, [1, gridsx, 1, 1, 1])
    GY = tf.reshape(gy, [1, 1, gridsy, 1, 1])

    mapping = core_function2d(GX, GY, X, Y, gridsx, gridsy)
    mapping = tf.reduce_sum(mapping, 0)
    mapping = tf.add(mapping, 1e-10)
    mapping_normalized = tf.divide(mapping, tf.reduce_sum(mapping, [0, 1], keepdims = True))

    entropy = tf.negative(tf.reduce_sum(tf.reduce_sum(tf.multiply(mapping_normalized, tf.log(mapping_normalized * (gridsx *gridsy))), [0, 1])))

    return entropy

def mutual_information(x, y):
  ex = entropy1d(x)
  ey = entropy1d(y)
  exy = entropy2d(x, y)
  return ex + ey - exy


def train_ensemble(root_path, config):
    logging.info('Train ensemble model')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    num_classes = 100
    batch_size = 64

    inputs_anchor_ph = tf.placeholder(tf.float32, shape=(batch_size // 2, 32, 32, 3))
    inputs_positive_ph = tf.placeholder(tf.float32, shape=(batch_size // 2, 32, 32, 3))
    labels_ph = tf.placeholder(tf.int64, shape=(batch_size // 2))
    training_ph = tf.placeholder(tf.bool, shape=())

    uid = config['uid']
    model_path = os.path.join(root_path, 'result/checkpoint', uid)
    os.makedirs(model_path, exist_ok=True)

    model_name = config['model']['name']

    with tf.variable_scope('model') as scope:
        x = get_model(model_name)(inputs_anchor_ph, training_ph, config).outputs
        embedding_anchor = tf.layers.dense(x, num_classes, name='embeddings')
    with tf.variable_scope(scope, reuse=True):
        x = get_model(model_name)(inputs_positive_ph, training_ph, config).outputs
        embedding_positive = tf.layers.dense(x, num_classes, name='embeddings')
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = get_model(model_name)(inputs_anchor_ph, training_ph, config).outputs
        logits = tf.layers.dense(x, num_classes, name='logits')

    mutual_info_op = mutual_information(embedding_anchor, logits)

    a = config.get('alpha', 0.5)
    weight_decay = config.get('weight_decay')
    with tf.name_scope('loss'):
        npairs_loss = metric_loss_ops.npairs_loss(labels_ph, embedding_anchor, embedding_positive)
        softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels_ph, num_classes), logits=logits))
        loss = a * npairs_loss + (1. - a) * softmax_loss
        if weight_decay:
            loss += weight_decay * tf.losses.get_regularization_loss()
        loss += 1e-4 * mutual_info_op
    tf.summary.scalar('npairs', npairs_loss)
    tf.summary.scalar('softmax', softmax_loss)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_path, '{}-ensemble-{}/train'.format(model_name, uid)), sess.graph)

    lr_ph = tf.placeholder(tf.float32, shape=())
    optimizer = get_optimizer(config.get('optimizer', {}).get('name', 'sgd'), lr_ph, **config.get('optimizer', {}).get('params', {}))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
        train_step = optimizer.apply_gradients(zip(gradients, variables))

    global_step = tf.train.get_or_create_global_step(sess.graph)

    saver = tf.train.Saver(max_to_keep=1000)
    train_dataset, _ = cifar100_npairs(batch_size, config)
    train_iterator = train_dataset.make_initializable_iterator()
    train_batch = train_iterator.get_next()

    sess.run(tf.global_variables_initializer())
    epochs = config.get('epochs', 1000)
    steps_per_epoch = 50000 // batch_size

    restore = config['model'].get('restore', None)
    if restore:
        start_epoch = restore['epoch']
        saver.restore(sess, os.path.join(root_path, 'result/checkpoint', restore['uid'], 'cifar100.{}.ensemble.epoch={}.ckpt'.format(model_name, start_epoch)))
    else:
        start_epoch = 0

    lr = 0.01

    for epoch in range(start_epoch, epochs):
        sess.run(train_iterator.initializer)
        npairs_losses = np.ndarray(shape=(0,), dtype=np.float32)
        softmax_losses = np.ndarray(shape=(0,), dtype=np.float32)
        accuracies = np.ndarray(shape=(0,), dtype=np.float32)
        mis = np.ndarray(shape=(0,), dtype=np.float32)
        step = 0

        lr_list = config.get('lr')
        if lr_list:
            if str(epoch) in lr_list:
                lr = lr_list[str(epoch)]
                logging.info('Learning rate set to {}'.format(lr))

        while True:
            try:
                step += 1
                batch_anchor, batch_positive, batch_labels = sess.run(train_batch)
                summary, npairs_l, softmax_l, mi, _ = sess.run([merged, npairs_loss, softmax_loss, mutual_info_op, train_step], feed_dict={
                    inputs_anchor_ph: batch_anchor,
                    inputs_positive_ph: batch_positive,
                    labels_ph: batch_labels,
                    lr_ph: lr,
                    training_ph: True,
                })
                train_writer.add_summary(summary, epoch * steps_per_epoch + step)
                npairs_losses = np.append(npairs_losses, npairs_l)
                softmax_losses = np.append(softmax_losses, softmax_l)
                mis = np.append(mis, mi)
                print('\rEpoch %d, Training Step %d : softmax loss = %lf\tnpairs loss = %lf\tMI = %lf' % (epoch + 1, step, np.mean(softmax_losses), np.mean(npairs_losses), np.mean(mis)), end='')
                logging.debug('Epoch %d, Training Step %d : softmax loss = %lf\tnpairs loss = %lf' % (epoch + 1, step, softmax_l, npairs_l))
            except tf.errors.OutOfRangeError:
                break
        logging.info('Epoch %d : softmax loss = %lf\tnpairs loss = %lf' % (epoch + 1, np.mean(softmax_losses), np.mean(npairs_losses)))

        if (epoch + 1) % 30 == 0:
            saver.save(sess, os.path.join(model_path, 'cifar100.{}.ensemble.epoch={}.ckpt'.format(model_name, epoch + 1)))

        print('')
    saver.save(sess, os.path.join(model_path, 'cifar100.{}.ensemble.ckpt'.format(model_name)))
    sess.close()
