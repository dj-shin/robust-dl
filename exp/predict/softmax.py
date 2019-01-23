import tensorflow as tf
import numpy as np
import os
import logging
import json
import time
import h5py

from exp.model.factory import get_model
from exp.dataset.factory import get_dataset


def gen_batch(x, y, batch_size):
    assert len(x) == len(y)
    for i in range(len(x) // batch_size):
        si, ei = i * batch_size, (i + 1) * batch_size
        yield x[si:ei], y[si:ei]
    left = len(x) % batch_size
    if left > 0:
        yield x[-left:], y[-left:]


def predict_softmax(root_path, config):
    logging.info('Predicting softmax model')
    logging.info(json.dumps(config))

    inputs_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    labels_ph = tf.placeholder(tf.int64, shape=(None, 1), name='labels')
    labels = tf.reshape(labels_ph, [-1])

    model_name = config['model']['name']
    dataset_name = config['data']['name']

    with tf.variable_scope('model') as scope:
        logits = get_model(model_name)(inputs_ph, False, config).outputs

    restore_uid = config['model'].get('restore', {}).get('uid')
    epoch = config['model'].get('restore', {}).get('epoch')
    model_path = config['model'].get('path')
    if model_path is None:
        if epoch is not None:
            model_path = os.path.join(root_path, 'result/checkpoint', restore_uid, '{}.{}.softmax.epoch={}.ckpt'.format(dataset_name, model_name, epoch))
        else:
            model_path = os.path.join(root_path, 'result/checkpoint', restore_uid, '{}.{}.softmax.ckpt'.format(dataset_name, model_name))

    correct_idx_op = tf.equal(tf.argmax(logits, axis=1), labels)
    correct_op = tf.reduce_sum(tf.cast(correct_idx_op, tf.int64))

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, model_path)

    batch_size = 100

    source = config.get('data', {}).get('source')
    if source is None or source == 'test':
        x_dataset, y_dataset = get_dataset(dataset_name, 'test')
    elif source == 'train':
        x_dataset, y_dataset = get_dataset(dataset_name, 'train')
    else:
        _, ext = os.path.splitext(source)
        if ext == '.npy':
            x_dataset = np.load(source)
            _, y_dataset = get_dataset(dataset_name, 'test')
        elif ext in ['.h5', '.hdf5']:
            with h5py.File(source, 'r') as f:
                x_dataset = f['ae'][:]
                _, y_dataset = get_dataset(dataset_name, 'test')
                idx = f['idx'][:]
                y_dataset = y_dataset[idx]

    correct = 0
    count = 0

    start_t = time.time()
    idx = list()
    for x, y in gen_batch(x_dataset, y_dataset, batch_size):
        correct_batch, correct_idx = sess.run([correct_op, correct_idx_op], feed_dict={inputs_ph: x, labels_ph: y})
        correct += correct_batch
        count += batch_size
        idx.append(correct_idx)

        print('\rAccuracy : %lf' % (correct / count), end='')
        logging.debug('Accuracy : %lf' % (correct / count))
        logging.debug(correct_idx)
    logging.info('Accuracy : %lf' % (correct / count))
    pred_time = time.time() - start_t
    print('')
    print('Prediction time : %lf sec' % pred_time)
    logging.info('Prediction time : %lf sec' % pred_time)
