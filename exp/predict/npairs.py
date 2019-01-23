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


def predict_embedding(batch_x, batch_y, x_train, y_train, test_emb_ph, train_emb_ph, dist, nn_index, sess):
    batch_size = 100
    pred = None
    min_dist = None
    for x_train_batch, y_train_batch in gen_batch(x_train, y_train, batch_size):
        current_dist, nn_idx = sess.run([dist, nn_index], feed_dict={
            test_emb_ph: batch_x,
            train_emb_ph: x_train_batch,
        })
        if min_dist is None:
            pred = y_train_batch[nn_idx].flatten()
            min_dist = current_dist
        else:
            pred = np.where(current_dist < min_dist, y_train_batch[nn_idx].flatten(), pred)
            min_dist = np.where(current_dist < min_dist, current_dist, min_dist)
    return pred == batch_y.flatten()


def predict_npairs(root_path, config):
    logging.info('Predicting npairs model')
    logging.info(json.dumps(config))

    num_classes = 100
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    inputs_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    labels_ph = tf.placeholder(tf.int64, shape=(None, 1), name='labels')
    labels = tf.reshape(labels_ph, [-1])

    model_name = config['model']['name']
    dataset_name = config['data']['name']

    with tf.variable_scope('model') as scope:
        embedding = get_model(model_name)(inputs_ph, False, config).outputs

    restore_uid = config['model'].get('restore', {}).get('uid')
    epoch = config['model'].get('restore', {}).get('epoch')
    model_path = config['model'].get('path')
    if model_path is None:
        if epoch is not None:
            model_path = os.path.join(root_path, 'result/checkpoint', restore_uid, '{}.{}.npairs.epoch={}.ckpt'.format(dataset_name, model_name, epoch))
        else:
            model_path = os.path.join(root_path, 'result/checkpoint', restore_uid, '{}.{}.npairs.ckpt'.format(dataset_name, model_name))

    test_emb_ph = tf.placeholder(tf.float32, shape=(None, embedding.shape[1]))
    train_emb_ph = tf.placeholder(tf.float32, shape=(None, embedding.shape[1]))
    dist = tf.norm(tf.expand_dims(train_emb_ph, axis=0) - tf.expand_dims(test_emb_ph, axis=1), ord=2, axis=2)
    nn_index = tf.argmin(dist, axis=1)
    min_dist = tf.reduce_min(dist, axis=1)

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, model_path)

    batch_size = 100

    x_train_emb = list()
    for batch_x, _ in gen_batch(x_train, y_train, batch_size):
        x_train_emb.append(sess.run(embedding, feed_dict={inputs_ph: batch_x}))
    x_train_emb = np.concatenate(x_train_emb)

    dataset_name = config['data']['name']

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
    num_classes = 100

    correct = 0
    count = 0

    start_t = time.time()
    for x, y in gen_batch(x_dataset, y_dataset, batch_size):
        data_emb = sess.run(embedding, feed_dict={inputs_ph: x})

        data_prediction = predict_embedding(data_emb, y, x_train_emb, y_train, test_emb_ph, train_emb_ph, min_dist, nn_index, sess)
        correct += np.sum(data_prediction.astype(np.int64))
        count += batch_size

        print('\rAccuracy : %lf' % (correct / count), end='')
        logging.debug('Accuracy : %lf' % (correct / count))
    logging.info('Accuracy : %lf' % (correct / count))
    pred_time = time.time() - start_t
    print('')
    print('Prediction time : %lf sec' % pred_time)
    logging.info('Prediction time : %lf sec' % pred_time)
