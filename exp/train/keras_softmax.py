import tensorflow as tf
import numpy as np
import os
import logging
import json

from exp.model.keras_cnn import KerasCNN
from exp.dataset.cifar10 import cifar10


def train_keras_softmax(root_path, **config):
    logging.info('Train KerasCNN softmax model')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    num_classes = 10
    batch_size = 100

    inputs_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='inputs')
    labels_ph = tf.placeholder(tf.int64, shape=(None, 1))
    labels = tf.reshape(labels_ph, [-1])
    training_ph = tf.placeholder(tf.bool, shape=())

    model_path = os.path.join(root_path, 'result/checkpoint', config['uid'])
    os.makedirs(model_path)

    with tf.variable_scope('model') as scope:
        logits = KerasCNN(inputs_ph, training_ph).outputs

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), labels), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels, num_classes), logits=logits))
    tf.summary.scalar('loss', loss)

    sess = tf.Session()

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_path, 'keras-{}/train'.format(config['uid'])), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(log_path, 'keras-{}/test'.format(config['uid'])))

    lr_ph = tf.placeholder(tf.float32, shape=())
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr_ph, decay=0.999999)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_ph)
    global_step = tf.train.get_or_create_global_step(sess.graph)
    train_step = optimizer.minimize(loss=loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=1000)
    train_dataset, test_dataset = cifar10(batch_size)
    train_iterator = train_dataset.make_initializable_iterator()
    train_batch = train_iterator.get_next()
    test_iterator = test_dataset.make_initializable_iterator()
    test_batch = test_iterator.get_next()

    sess.run(tf.global_variables_initializer())
    epochs = 500
    steps_per_epoch = 50000 // batch_size

    start_epoch = config.get('resume_from', None)
    if start_epoch:
        saver.restore(sess, os.path.join(model_path, 'cifar10.keras.softmax.epoch={}.ckpt'.format(start_epoch)))
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        sess.run(train_iterator.initializer)
        losses = np.ndarray(shape=(0,), dtype=np.float32)
        accuracies = np.ndarray(shape=(0,), dtype=np.float32)
        step = 0

        if epoch < 80:
            lr = 0.01
        elif epoch < 120:
            lr = 0.001
        else:
            lr = 0.0001

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

        if (epoch + 1) % 10 == 0:
            saver.save(sess, os.path.join(model_path, 'cifar10.keras.softmax.epoch={}.ckpt'.format(epoch + 1)))
        print('')
    sess.close()
