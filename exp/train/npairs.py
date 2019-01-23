import tensorflow as tf
import numpy as np
import os
import logging
import json

from exp.model.factory import get_model
from exp.train.utils import get_optimizer
from exp.dataset.cifar100 import cifar100_npairs

from tensorflow.contrib.losses.python import metric_learning as metric_loss_ops


def train_npairs(root_path, config):
    logging.info('Train npairs model')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    num_classes = 100
    batch_size = 100

    inputs_anchor_ph = tf.placeholder(tf.float32, shape=(batch_size // 2, 32, 32, 3))
    inputs_positive_ph = tf.placeholder(tf.float32, shape=(batch_size // 2, 32, 32, 3))
    labels_ph = tf.placeholder(tf.int64, shape=(batch_size // 2))
    training_ph = tf.placeholder(tf.bool, shape=())

    uid = config['uid']
    model_path = os.path.join(root_path, 'result/checkpoint', uid)
    os.makedirs(model_path, exist_ok=True)

    model_name = config['model']['name']

    with tf.variable_scope('model') as scope:
        embedding_anchor = get_model(model_name)(inputs_anchor_ph, training_ph, config).outputs
    with tf.variable_scope(scope, reuse=True):
        embedding_positive = get_model(model_name)(inputs_positive_ph, training_ph, config).outputs

    with tf.name_scope('loss'):
        loss = metric_loss_ops.npairs_loss(labels_ph, embedding_anchor, embedding_positive)
    tf.summary.scalar('loss', loss)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_path, '{}-npairs-{}/train'.format(model_name, uid)), sess.graph)

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
        saver.restore(sess, os.path.join(root_path, 'result/checkpoint', restore['uid'], 'cifar100.{}.npairs.epoch={}.ckpt'.format(model_name, start_epoch)))
    else:
        start_epoch = 0

    lr = 0.01

    for epoch in range(start_epoch, epochs):
        sess.run(train_iterator.initializer)
        losses = np.ndarray(shape=(0,), dtype=np.float32)
        accuracies = np.ndarray(shape=(0,), dtype=np.float32)
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
                summary, l, _ = sess.run([merged, loss, train_step], feed_dict={
                    inputs_anchor_ph: batch_anchor,
                    inputs_positive_ph: batch_positive,
                    labels_ph: batch_labels,
                    lr_ph: lr,
                    training_ph: True,
                })
                train_writer.add_summary(summary, epoch * steps_per_epoch + step)
                losses = np.append(losses, l)
                print('\rEpoch %d, Training Step %d : loss = %lf' % (epoch + 1, step, np.mean(losses)), end='')
                logging.debug('Epoch %d, Training Step %d : loss = %lf' % (epoch + 1, step, l))
            except tf.errors.OutOfRangeError:
                break
        logging.info('Epoch %d : loss = %lf' % (epoch + 1, np.mean(losses)))

        if (epoch + 1) % 30 == 0:
            saver.save(sess, os.path.join(model_path, 'cifar100.{}.npairs.epoch={}.ckpt'.format(model_name, epoch + 1)))

        print('')
    saver.save(sess, os.path.join(model_path, 'cifar100.{}.npairs.ckpt'.format(model_name)))
    sess.close()
