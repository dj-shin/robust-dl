import tensorflow as tf
import numpy as np
import os
import logging
import json

from exp.model.factory import get_model
from exp.train.utils import get_optimizer
from exp.dataset.cifar100 import cifar100_histogram


def train_histogram(root_path, config):
    logging.info('Train histogram model')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    num_classes = 100
    batch_size = 100

    inputs_ph = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='inputs')
    labels_ph = tf.placeholder(tf.int64, shape=(batch_size, 1))
    training_ph = tf.placeholder(tf.bool, shape=())

    uid = config['uid']
    model_path = os.path.join(root_path, 'result/checkpoint', uid)
    os.makedirs(model_path, exist_ok=True)

    model_name = config['model']['name']

    with tf.variable_scope('model') as scope:
        model = get_model(model_name)(inputs_ph, training_ph, config)

    weight_decay = config.get('weight_decay')
    with tf.name_scope('loss'):
        loss = model.loss(labels_ph, 'histogram')
        if weight_decay:
            loss += weight_decay * tf.losses.get_regularization_loss()

    tf.summary.scalar('loss', loss)

    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(log_path, '{}-hist-{}/train'.format(model_name, uid)), sess.graph)

    lr_ph = tf.placeholder(tf.float32, shape=())
    optimizer = get_optimizer(config.get('optimizer', {}).get('name', 'sgd'), lr_ph, **config.get('optimizer', {}).get('params', {}))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1e6)
        train_step = optimizer.apply_gradients(zip(gradients, variables))

    global_step = tf.train.get_or_create_global_step(sess.graph)

    saver = tf.train.Saver(max_to_keep=1000)
    train_dataset, _ = cifar100_histogram(batch_size, config)
    train_iterator = train_dataset.make_initializable_iterator()
    train_batch = train_iterator.get_next()

    sess.run(tf.global_variables_initializer())
    epochs = config.get('epochs', 1000)
    steps_per_epoch = 50000 // batch_size

    restore = config['model'].get('restore', None)
    if restore:
        start_epoch = restore['epoch']
        saver.restore(sess, os.path.join(root_path, 'result/checkpoint', restore['uid'], 'cifar100.{}.hist.epoch={}.ckpt'.format(model_name, start_epoch)))
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
                summary, l  = sess.run([merged, loss], feed_dict={
                    inputs_ph: batch_x,
                    labels_ph: batch_y,
                    training_ph: False,
                })
                train_writer.add_summary(summary, epoch * steps_per_epoch + step)
                losses = np.append(losses, l)
                print('\rEpoch %d, Training Step %d : loss = %lf' % (epoch + 1, step, np.mean(losses)), end='')
                logging.debug('Epoch %d, Training Step %d : loss = %lf' % (epoch + 1, step, l))
            except tf.errors.OutOfRangeError:
                break
        logging.info('Epoch %d : loss = %lf' % (epoch + 1, np.mean(losses)))

        if (epoch + 1) % 30 == 0:
            saver.save(sess, os.path.join(model_path, 'cifar100.{}.hist.epoch={}.ckpt'.format(model_name, epoch + 1)))

        print('')
    saver.save(sess, os.path.join(model_path, 'cifar100.{}.hist.ckpt'.format(model_name)))
    sess.close()
