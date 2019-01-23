import tensorflow as tf
import numpy as np
import os
import logging
import json

from exp.model.factory import get_model

from exp.dataset.cifar100 import cifar100
from exp.dataset.cifar10 import cifar10

from exp.train.utils import get_optimizer


def train_softmax(root_path, config):
    logging.info('Train softmax model')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    batch_size = 100
    data = config.get('data', {}).get('name', 'cifar100')
    if data == 'cifar10':
        num_classes = 10
        train_dataset, test_dataset = cifar10(batch_size, config)
    elif data == 'cifar100':
        num_classes = 100
        train_dataset, test_dataset = cifar100(batch_size, config)

    inputs_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='inputs')
    labels_ph = tf.placeholder(tf.int64, shape=(None, 1))
    labels = tf.reshape(labels_ph, [-1])
    training_ph = tf.placeholder(tf.bool, shape=())

    uid = config['uid']
    model_path = os.path.join(root_path, 'result/checkpoint', uid)
    os.makedirs(model_path, exist_ok=True)

    model_name = config['model']['name']

    with tf.variable_scope('model') as scope:
        model = get_model(model_name)(inputs_ph, training_ph, config)
        logits = model.outputs

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    weight_decay = config.get('weight_decay')
    with tf.name_scope('loss'):
        loss = model.loss(labels)
        if weight_decay:
            loss += weight_decay * tf.losses.get_regularization_loss()

    tf.summary.scalar('loss', loss)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_path, '{}-softmax-{}/train'.format(model_name, uid)), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(log_path, '{}-softmax-{}/test'.format(model_name, uid)))

    lr_ph = tf.placeholder(tf.float32, shape=())
    optimizer = get_optimizer(config.get('optimizer', {}).get('name', 'sgd'), lr_ph, **config.get('optimizer', {}).get('params', {}))
    global_step = tf.train.get_or_create_global_step(sess.graph)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1000)
    train_iterator = train_dataset.make_initializable_iterator()
    train_batch = train_iterator.get_next()
    test_iterator = test_dataset.make_initializable_iterator()
    test_batch = test_iterator.get_next()

    sess.run(tf.global_variables_initializer())
    epochs = config.get('epochs', 1000)
    steps_per_epoch = 50000 // batch_size

    restore = config['model'].get('restore', None)
    if restore:
        start_epoch = restore['epoch']
        saver.restore(sess, os.path.join(root_path, 'result/checkpoint', restore['uid'], '{}.{}.softmax.epoch={}.ckpt'.format(data, model_name, start_epoch)))
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
                batch_x, batch_y = sess.run(train_batch)
                sess.run([train_step], feed_dict={
                    inputs_ph: batch_x,
                    labels_ph: batch_y,
                    training_ph: True,
                    lr_ph: lr,
                })
                summary, acc, l  = sess.run([merged, accuracy, loss], feed_dict={
                    inputs_ph: batch_x,
                    labels_ph: batch_y,
                    training_ph: False,
                })
                train_writer.add_summary(summary, epoch * steps_per_epoch + step)
                losses = np.append(losses, l)
                accuracies = np.append(accuracies, acc)
                print('\rEpoch %d, Training Step %d : loss = %lf, acc = %lf' % (epoch + 1, step, np.mean(losses), np.mean(accuracies)), end='')
                logging.debug('Epoch %d, Training Step %d : loss = %lf, acc = %lf' % (epoch + 1, step, l, acc))
            except tf.errors.OutOfRangeError:
                break
        logging.info('Epoch %d : loss = %lf, acc = %lf' % (epoch + 1, np.mean(losses), np.mean(accuracies)))

        sess.run(test_iterator.initializer)
        losses = np.ndarray(shape=(0,), dtype=np.float32)
        accuracies = np.ndarray(shape=(0,), dtype=np.float32)
        step = 0
        while True:
            try:
                step += 1
                batch_x, batch_y = sess.run(test_batch)
                summary, acc, l = sess.run([merged, accuracy, loss], feed_dict={
                    inputs_ph: batch_x,
                    labels_ph: batch_y,
                    training_ph: False,
                })
                test_writer.add_summary(summary, epoch * (10000 // batch_size) + step)
                losses = np.append(losses, l)
                accuracies = np.append(accuracies, acc)
                logging.debug('Epoch %d, Test Step %d : loss = %lf, acc = %lf' % (epoch + 1, step, l, acc))
            except tf.errors.OutOfRangeError:
                break
        print('\tTest : loss = %lf, acc = %lf' % (np.mean(losses), np.mean(accuracies)), end='')
        logging.info('Test Epoch %d : loss = %lf, acc = %lf' % (epoch + 1, np.mean(losses), np.mean(accuracies)))

        if (epoch + 1) % 30 == 0:
            saver.save(sess, os.path.join(model_path, '{}.{}.softmax.epoch={}.ckpt'.format(data, model_name, epoch + 1)))
        print('')
    saver.save(sess, os.path.join(model_path, '{}.{}.softmax.ckpt'.format(data, model_name)))
    sess.close()
