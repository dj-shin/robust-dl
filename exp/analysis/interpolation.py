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


def predict_embedding(batch_x, x_train, test_emb_ph, train_emb_ph, dist, nn_index, sess):
    batch_size = 100
    pred = None
    min_dist = None
    offset = 0
    for x_train_batch, _ in gen_batch(x_train, x_train, batch_size):
        current_dist, nn_idx = sess.run([dist, nn_index], feed_dict={
            test_emb_ph: batch_x,
            train_emb_ph: x_train_batch,
        })
        if min_dist is None:
            pred = nn_idx.flatten() + offset
            min_dist = current_dist
        else:
            pred = np.where(current_dist < min_dist, nn_idx.flatten() + offset, pred)
            min_dist = np.where(current_dist < min_dist, current_dist, min_dist)
        offset += batch_size
    return pred


def interpolation_npairs(root_path, config):
    logging.info('Interpolation of AE on npairs model')
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
    x_dataset, y_dataset = get_dataset(dataset_name, 'test')

    source = config.get('data', {}).get('source')
    assert source is not None and source != 'test' and source != 'train', 'invalid AE source'
    _, ext = os.path.splitext(source)
    if ext == '.npy':
        ae_dataset = np.load(source)
        idx = np.arange(ae_dataset.shape[0])
    elif ext in ['.h5', '.hdf5']:
        with h5py.File(source, 'r') as f:
            ae_dataset = f['ae'][:]
            idx = f['idx'][:]
    num_classes = 100

    correct = 0
    count = 0

    interpolation_count = 100    # number of interpolation points

    file_path = os.path.join(root_path, 'result/analysis', config['uid'], 'interpolation.{model_name}.h5'.format(model_name=model_name))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        embeddings_file = f.create_dataset('embedding', (ae_dataset.shape[0], interpolation_count + 1, embedding.shape[1]), dtype=np.float32)
        nn_idx_file = f.create_dataset('nn_idx', (ae_dataset.shape[0], interpolation_count + 1), dtype=np.int64)

        start_t = time.time()
        for i in range(ae_dataset.shape[0]):
            clean = x_dataset[idx[i]]
            ae = ae_dataset[i]
            label = y_dataset[idx[i]]

            x_seq = list()
            for p in range(interpolation_count + 1):
                x_seq.append(ae * p / interpolation_count + clean * (1. - p / interpolation_count))
            x_seq = np.stack(x_seq)
            data_emb = sess.run(embedding, feed_dict={inputs_ph: x_seq})
            data_nn_idx = predict_embedding(data_emb, x_train_emb, test_emb_ph, train_emb_ph, min_dist, nn_index, sess)

            embeddings_file[i] = data_emb
            nn_idx_file[i] = data_nn_idx

            print('\r%d / %d Done' % (i + 1, ae_dataset.shape[0]), end='')


def interpolation_softmax(root_path, config):
    logging.info('Interpolation of AE on npairs model')
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
        logits = get_model(model_name)(inputs_ph, False, config).outputs

    restore_uid = config['model'].get('restore', {}).get('uid')
    epoch = config['model'].get('restore', {}).get('epoch')
    if epoch is not None:
        model_path = os.path.join(root_path, 'result/checkpoint', restore_uid, '{}.{}.softmax.epoch={}.ckpt'.format(dataset_name, model_name, epoch))
    else:
        model_path = os.path.join(root_path, 'result/checkpoint', restore_uid, '{}.{}.softmax.ckpt'.format(dataset_name, model_name))

    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, model_path)

    batch_size = 100

    dataset_name = config['data']['name']
    x_dataset, y_dataset = get_dataset(dataset_name, 'test')

    source = config.get('data', {}).get('source')
    assert source is not None and source != 'test' and source != 'train', 'invalid AE source'
    _, ext = os.path.splitext(source)
    if ext == '.npy':
        ae_dataset = np.load(source)
        idx = np.arange(ae_dataset.shape[0])
    elif ext in ['.h5', '.hdf5']:
        with h5py.File(source, 'r') as f:
            ae_dataset = f['ae'][:]
            idx = f['idx'][:]
    num_classes = 100

    correct = 0
    count = 0

    interpolation_count = 100    # number of interpolation points

    file_path = os.path.join(root_path, 'result/analysis', config['uid'], 'interpolation.{model_name}.h5'.format(model_name=model_name))
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        logits_file = f.create_dataset('logits', (ae_dataset.shape[0], interpolation_count + 1, logits.shape[1]), dtype=np.float32)

        start_t = time.time()
        for i in range(ae_dataset.shape[0]):
            clean = x_dataset[idx[i]]
            ae = ae_dataset[i]
            label = y_dataset[idx[i]]

            x_seq = list()
            for p in range(interpolation_count + 1):
                x_seq.append(ae * p / interpolation_count + clean * (1. - p / interpolation_count))
            x_seq = np.stack(x_seq)
            data_logits = sess.run(logits, feed_dict={inputs_ph: x_seq})

            logits_file[i] = data_logits

            print('\r%d / %d Done' % (i + 1, ae_dataset.shape[0]), end='')
