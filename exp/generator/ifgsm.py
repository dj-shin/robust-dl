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


def gen_batch(x, y, batch_size):
    assert len(x) == len(y)
    for i in range(len(x) // batch_size):
        si, ei = i * batch_size, (i + 1) * batch_size
        yield x[si:ei], y[si:ei]
    left = len(x) % batch_size
    if left > 0:
        yield x[-left:], y[-left:]

    
def ifgsm(root_path, config):
    sess = tf.Session()
    logging.info('Iterative FGSM generation')
    logging.info(json.dumps(config))

    log_path = os.path.join(root_path, 'result/log')

    batch_size = 100
    dataset_name = config['data']['name']
    dataset_type = config['data']['type']
    x_dataset, y_dataset = get_dataset(dataset_name, dataset_type)
    eps_list = config['generate']['eps_list']
    model_name = config['model']['name']

    n_iter = 4

    if config['model']['type'] == 'npairs':
        pass
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
                next_x, clean_accuracy_batch = sess.run([fgsm_inputs, clean_accuracy], feed_dict={inputs_ph: x, eps_ph: eps / n_iter / 255, labels_ph: y})
                for _ in range(1, n_iter):
                    next_x, fgsm_accuracy_batch = sess.run([fgsm_inputs, fgsm_accuracy], feed_dict={inputs_ph: next_x, eps_ph: eps / n_iter / 255, labels_ph: y})
                correct_clean += clean_accuracy_batch
                correct_fgsm += fgsm_accuracy_batch
                count += len(x)

                generated_inputs.append(next_x)
                print('\rEps : %d\tClean : %lf / I-FGSM-%d : %lf' % (eps, correct_clean / count, n_iter, correct_fgsm / count), end='')
                logging.debug('Eps : %d\tClean : %lf / I-FGSM-%d : %lf' % (eps, correct_clean / count, n_iter, correct_fgsm / count))
            logging.info('Eps : %d\tClean : %lf / I-FGSM-%d : %lf' % (eps, correct_clean / count, n_iter, correct_fgsm / count))
            print('')
            generated_path = os.path.join(root_path, 'result/dataset', config['uid'], 'ifgsm.{train_or_test}.{restore_uid}.{eps}.h5'.format(
                train_or_test=dataset_type, restore_uid=restore['uid'], eps=eps))
            os.makedirs(os.path.dirname(generated_path), exist_ok=True)
            with h5py.File(generated_path, 'w') as f:
                idx = f.create_dataset('idx', (x_dataset.shape[0],), dtype=np.int64)
                ae = f.create_dataset('ae', (x_dataset.shape), dtype=np.float32)


                idx[...] = np.arange(x_dataset.shape[0])
                ae[...] = np.concatenate(generated_inputs)
            logging.info('Generated {}'.format(generated_path))
    print('')
