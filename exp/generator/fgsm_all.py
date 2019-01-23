import h5py
import numpy as np
import tensorflow as tf

import logging
import json
import os

from exp.model.factory import get_model


def fgsm_all(root_path, config):
    sess = tf.Session()
    logging.info('FGSM all generation')
    logging.info(json.dumps(config))
    log_path = os.path.join(root_path, 'result/log')

    batch_size = 100

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    eps_list = config['generate']['eps_list']
    model_name = config['model']['name']
    restore = config['model']['restore']

    batch_size = 100
    inputs_ph = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='inputs')

    with tf.variable_scope('model') as scope:
        embedding_op = get_model(model_name)(inputs_ph, False, config).outputs

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, config['model']['path'])

    print('Preparing training data embeddings...')
    train_embeddings_path = 'train_embeddings_{}.npy'.format(restore['uid'])
    if os.path.exists(train_embeddings_path):
        train_embeddings = np.load(train_embeddings_path)
    else:
        train_embeddings = list()
        for i in range(x_train.shape[0] // batch_size):
            x_batch = x_train[i * batch_size : (i + 1) * batch_size]
            embedding = sess.run(embedding_op, feed_dict={
                inputs_ph: x_batch
            })
            train_embeddings.append(embedding)
        train_embeddings = np.concatenate(train_embeddings, axis=0)
        np.save(train_embeddings_path, train_embeddings)
    print(train_embeddings.shape)

    embedding_dim = 100

    train_embeddings_op = tf.constant(train_embeddings, tf.float32)
    train_labels_op = tf.constant(y_train, tf.int64)
    test_embeddings = embedding_op
    test_labels_ph = tf.placeholder(tf.int64, shape=(batch_size, 1))

    labels_equal = tf.cast(tf.equal(tf.reshape(test_labels_ph, [-1, 1]), tf.reshape(train_labels_op, [1, -1])), tf.float32)
    labels_op = labels_equal

    prod = tf.matmul(test_embeddings, train_embeddings_op, transpose_a=False, transpose_b=True)
    loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_op, logits=prod)
    grad_op = tf.gradients(loss_op, inputs_ph)[0]
    
    eps_ph = tf.placeholder(tf.float32)
    fgsm_op = inputs_ph + eps_ph * tf.sign(grad_op)

    for eps in eps_list:
        ae_data = list()
        for i in range(x_test.shape[0] // batch_size):
            x_batch = x_test[i * batch_size : (i + 1) * batch_size]
            y_batch = y_test[i * batch_size : (i + 1) * batch_size]
            loss, fgsm = sess.run([loss_op, fgsm_op], feed_dict={
                inputs_ph: x_batch,
                test_labels_ph: y_batch,
                eps_ph: eps / 255.,
            })
            ae_data.append(fgsm)
            loss_after = sess.run(loss_op, feed_dict={
                inputs_ph: fgsm,
                test_labels_ph: y_batch,
                eps_ph: eps / 255.,
            })

        generated_inputs = np.concatenate(ae_data, axis=0)
        generated_path = os.path.join(root_path, 'result/dataset', config['uid'], 'fgsm_all.{train_or_test}.{restore_uid}.{eps}.h5'.format(
            train_or_test='test', restore_uid=restore['uid'], eps=eps))
        os.makedirs(os.path.dirname(generated_path), exist_ok=True)
        with h5py.File(generated_path, 'w') as f:
            idx = f.create_dataset('idx', (x_test.shape[0],), dtype=np.int64)
            ae = f.create_dataset('ae', (x_test.shape), dtype=np.float32)

            idx[...] = np.arange(x_test.shape[0])
            ae[...] = np.stack(generated_inputs)
        logging.info('Generated {}'.format(generated_path))
