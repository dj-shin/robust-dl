import h5py
import json
import logging
import tensorflow as tf
import numpy as np
import os

from exp.model.factory import get_model
from exp.dataset.factory import get_dataset


def define_softmax_model(config):
    inputs_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='inputs')
    labels_ph = tf.placeholder(tf.int64, shape=(None, 1))
    labels = tf.reshape(labels_ph, [-1])
    model_name = config['model']['name']

    with tf.variable_scope('model') as scope:
        model = get_model(model_name)(inputs_ph, False, config)
        clean_outputs = model.outputs
        loss = model.loss(labels)

    grad = tf.gradients(loss, inputs_ph)[0]

    eps_ph = tf.placeholder(tf.float32, shape=(), name='eps')
    fgsm_inputs = inputs_ph + tf.sign(grad) * eps_ph

    with tf.variable_scope(scope, reuse=True):
        fgsm_outputs = get_model(model_name)(fgsm_inputs, False, config).outputs

    return inputs_ph, labels_ph, eps_ph, fgsm_inputs, clean_outputs, fgsm_outputs


def define_npairs_model(config, batch_size):
    inputs_ph = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='inputs')
    pivot_ph = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='pivots')

    model_name = config['model']['name']

    with tf.variable_scope('model') as scope:
        embedding = get_model(model_name)(inputs_ph, False, config).outputs
    with tf.variable_scope(scope, reuse=True):
        embedding_pivot = get_model(model_name)(pivot_ph, False, config).outputs

    distance = tf.norm(embedding - embedding_pivot, axis=-1)
    grad = tf.gradients(distance, inputs_ph)[0]

    eps_ph = tf.placeholder(tf.float32, shape=(), name='eps')
    fgsm_inputs = inputs_ph + tf.sign(grad) * eps_ph

    with tf.variable_scope(scope, reuse=True):
        embedding_fgsm = get_model(model_name)(fgsm_inputs, False, config).outputs

    return inputs_ph, pivot_ph, eps_ph, fgsm_inputs, embedding, embedding_fgsm


def gen_batch(x, y, batch_size):
    assert len(x) == len(y)
    for i in range(len(x) // batch_size):
        si, ei = i * batch_size, (i + 1) * batch_size
        yield x[si:ei], y[si:ei]
    left = len(x) % batch_size
    if left > 0:
        yield x[-left:], y[-left:]


def predict_embedding(batch_x, batch_y, x_train, y_train):
    dist = np.linalg.norm(np.expand_dims(x_train, axis=0) - np.expand_dims(batch_x, axis=1), ord=2, axis=2)
    nn_index = np.argmin(dist, axis=1)
    prediction = y_train[nn_index]
    return np.sum((prediction == batch_y).astype(np.int64))

    
def fgsm(root_path, config):
    sess = tf.Session()
    logging.info('FGSM generation')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    batch_size = 100
    dataset_name = config['data']['name']
    dataset_type = config['data']['type']
    x_dataset, y_dataset = get_dataset(dataset_name, dataset_type)
    eps_list = config['generate']['eps_list']
    model_name = config['model']['name']

    if config['model']['type'] == 'npairs':
        inputs_ph, pivot_ph, eps_ph, fgsm_inputs, embedding, embedding_fgsm = define_npairs_model(config, batch_size)
        restore = config['model']['restore']
        epoch = restore.get('epoch')
        if epoch is not None:
            model_path = os.path.join(root_path, 'result/checkpoint', restore['uid'], '{}.{}.npairs.epoch={}.ckpt'.format(dataset_name, model_name, epoch))
        else:
            model_path = os.path.join(root_path, 'result/checkpoint', restore['uid'], '{}.{}.npairs.ckpt'.format(dataset_name, model_name))

        test_emb_ph = tf.placeholder(tf.float32, shape=(None, embedding.shape[1]))
        train_emb_ph = tf.placeholder(tf.float32, shape=(None, embedding.shape[1]))
        dist = tf.norm(tf.expand_dims(train_emb_ph, axis=0) - tf.expand_dims(test_emb_ph, axis=1), axis=2)
        nn_index = tf.argmin(dist, axis=1)
        min_dist = tf.reduce_min(dist, axis=1)

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        num_classes = 100
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        x_train_emb = list()
        for batch_x, _ in gen_batch(x_train, y_train, batch_size):
            x_train_emb.append(sess.run(embedding, feed_dict={inputs_ph: batch_x}))
        x_train_emb = np.concatenate(x_train_emb)

        for eps in eps_list:
            generated_inputs = list()
            np.random.seed(0)
            count = 0
            for x, y in gen_batch(x_dataset, y_dataset, batch_size):
                fgsm_inputs_data = sess.run(fgsm_inputs, feed_dict={
                    inputs_ph: x + np.random.normal(scale=1e-8, size=x.shape),
                    pivot_ph: x,
                    eps_ph: eps / 255,
                })

                count += len(x)
                generated_inputs.append(fgsm_inputs_data)
                print('\rEps : %d\t%d%% done' % (eps, count * 100 / x_dataset.shape[0]), end='')
            print('')

            generated_path = os.path.join(root_path, 'result/dataset', config['uid'], 'fgsm.{train_or_test}.{restore_uid}.{eps}.h5'.format(
                train_or_test=dataset_type, restore_uid=restore['uid'], eps=eps))
            os.makedirs(os.path.dirname(generated_path), exist_ok=True)
            with h5py.File(generated_path, 'w') as f:
                idx = f.create_dataset('idx', (x_dataset.shape[0],), dtype=np.int64)
                ae = f.create_dataset('ae', (x_dataset.shape), dtype=np.float32)

                idx[...] = np.arange(x_dataset.shape[0])
                ae[...] = np.concatenate(generated_inputs)
            logging.info('Generated {}'.format(generated_path))
    elif config['model']['type'] == 'softmax':
        inputs_ph, labels_ph, eps_ph, fgsm_inputs, clean_outputs, fgsm_outputs = define_softmax_model(config)
        labels = tf.reshape(labels_ph, [-1])
        clean_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(clean_outputs, axis=1), labels), tf.float32))
        fgsm_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(fgsm_outputs, axis=1), labels), tf.float32))

        restore = config['model']['restore']
        epoch = restore.get('epoch')
        if epoch is not None:
            model_path = os.path.join(root_path, 'result/checkpoint', restore['uid'], '{}.{}.softmax.epoch={}.ckpt'.format(dataset_name, model_name, epoch))
        else:
            model_path = os.path.join(root_path, 'result/checkpoint', restore['uid'], '{}.{}.softmax.ckpt'.format(dataset_name, model_name))

        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        for eps in eps_list:
            correct_clean = 0
            correct_fgsm = 0
            count = 0
            generated_inputs = list()
            for x, y in gen_batch(x_dataset, y_dataset, batch_size):
                fgsm_inputs_data, clean_accuracy_batch, fgsm_accuracy_batch = sess.run([fgsm_inputs, clean_accuracy, fgsm_accuracy], feed_dict={inputs_ph: x, eps_ph: eps / 255, labels_ph: y})
                correct_clean += clean_accuracy_batch
                correct_fgsm += fgsm_accuracy_batch
                count += len(x)

                generated_inputs.append(fgsm_inputs_data)
                print('\rEps : %d\tClean : %lf / FGSM : %lf' % (eps, correct_clean / count, correct_fgsm / count), end='')
                logging.debug('Eps : %d\tClean : %lf / FGSM : %lf' % (eps, correct_clean / count, correct_fgsm / count))
            logging.info('Eps : %d\tClean : %lf / FGSM : %lf' % (eps, correct_clean / count, correct_fgsm / count))
            print('')
            generated_path = os.path.join(root_path, 'result/dataset', config['uid'], 'fgsm.{train_or_test}.{restore_uid}.{eps}.h5'.format(
                train_or_test=dataset_type, restore_uid=restore['uid'], eps=eps))
            os.makedirs(os.path.dirname(generated_path), exist_ok=True)
            with h5py.File(generated_path, 'w') as f:
                idx = f.create_dataset('idx', (x_dataset.shape[0],), dtype=np.int64)
                ae = f.create_dataset('ae', (x_dataset.shape), dtype=np.float32)

                idx[...] = np.arange(x_dataset.shape[0])
                ae[...] = np.concatenate(generated_inputs)
            logging.info('Generated {}'.format(generated_path))
    print('')
