import tensorflow as tf
import numpy as np
import os

from resnet import ResNet56, embedding_size


def define_npairs(batch_size):
    num_classes = 100

    (x_train, _), _ = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train = x_train.astype('float32')
    x_train /= 255

    inputs_ph = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='inputs')

    w = tf.get_variable('adversarial', shape=(batch_size, 32, 32, 3), dtype=tf.float32)
    mid_points = tf.placeholder(tf.float32, shape=(num_classes, embedding_size), name='mid_points')
    target_embedding = tf.placeholder(tf.float32, shape=(batch_size, embedding_size), name='target_embedding')
    others_embedding = tf.placeholder(tf.float32, shape=(batch_size, num_classes - 1, embedding_size), name='target_embedding')

    adv_inputs = (tf.tanh(w) + 1) / 2.

    def preprocess(x, pixel_mean, pixel_stddev):
        x = (x - pixel_mean) / pixel_stddev
        return x

    pixel_mean = np.mean(x_train, axis=0)
    pixel_stddev = np.std(x_train, axis=0)

    adv_preprocessed = tf.map_fn(lambda x: preprocess(x, pixel_mean, pixel_stddev), adv_inputs)
    inputs = tf.map_fn(lambda x: preprocess(x, pixel_mean, pixel_stddev), inputs_ph)

    with tf.variable_scope('model') as scope:
        adv_embedding = ResNet56(adv_preprocessed)
    with tf.variable_scope(scope, reuse=True):
        embedding_clean = ResNet56(inputs)

    loss1 = tf.reduce_sum(tf.square(adv_inputs - inputs_ph), axis=[1, 2, 3])
    c = tf.placeholder(tf.float32, shape=(), name='c')
    loss2 = tf.reduce_sum(tf.square(adv_embedding - target_embedding), axis=-1) - tf.reduce_min(
            tf.reduce_sum(tf.square(tf.expand_dims(adv_embedding, axis=1) - target_embedding), axis=-1), axis=-1)

    loss = loss1 + c * loss2
    
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train_step = optimizer.minimize(loss, var_list=[w])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    init = tf.variables_initializer(var_list=[w]+new_vars)

    assign_op = tf.assign(w, tf.atanh(2 * inputs_ph - 1) + tf.random_normal(shape=inputs_ph.shape, stddev=0.1))

    return adv_inputs, inputs_ph, mid_points, target_embedding, others_embedding, c, train_step, embedding_clean, init, assign_op, adv_embedding, tf.reduce_mean(loss), tf.reduce_mean(loss2)


def define_softmax(batch_size):
    num_classes = 100

    (x_train, _), _ = tf.keras.datasets.cifar100.load_data(label_mode='fine')
    x_train = x_train.astype('float32')
    x_train /= 255

    inputs_ph = tf.placeholder(tf.float32, shape=(batch_size, 32, 32, 3), name='inputs')
    labels_ph = tf.placeholder(tf.int64, shape=(batch_size,), name='labels')

    w = tf.get_variable('adversarial', shape=(batch_size, 32, 32, 3), dtype=tf.float32)
    target_cls = tf.placeholder(tf.int64, shape=(batch_size,), name='target_class')
    target_onehot = tf.one_hot(target_cls, num_classes)

    adv_inputs = (tf.tanh(w) + 1) / 2.

    def preprocess(x, pixel_mean, pixel_stddev):
        x = (x - pixel_mean) / pixel_stddev
        return x

    pixel_mean = np.mean(x_train, axis=0)
    pixel_stddev = np.std(x_train, axis=0)

    adv_preprocessed = tf.map_fn(lambda x: preprocess(x, pixel_mean, pixel_stddev), adv_inputs)
    inputs = tf.map_fn(lambda x: preprocess(x, pixel_mean, pixel_stddev), inputs_ph)

    with tf.variable_scope('model') as scope:
        adv_embedding = ResNet56(adv_preprocessed)
        adv_logits = tf.layers.dense(adv_embedding, num_classes, activation=None, name='logits')

    loss1 = tf.reduce_sum(tf.square(adv_inputs - inputs_ph), axis=[1, 2, 3])
    c = tf.placeholder(tf.float32, shape=(), name='c')
    target_logit = tf.multiply(target_onehot, adv_logits)
    other_logit = tf.multiply(tf.ones(shape=target_onehot.shape) - target_onehot, adv_logits)
    loss2 = tf.reduce_max(other_logit - target_logit)

    loss = loss1 + c * loss2
    
    start_vars = set(x.name for x in tf.global_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    train_step = optimizer.minimize(loss, var_list=[w])
    end_vars = tf.global_variables()
    new_vars = [x for x in end_vars if x.name not in start_vars]
    init = tf.variables_initializer(var_list=[w]+new_vars)

    assign_op = tf.assign(w, tf.atanh(2 * inputs_ph - 1) + tf.random_normal(shape=inputs_ph.shape, stddev=0.1))

    adv_correct = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(adv_logits, axis=1), labels_ph), tf.float32))

    return adv_inputs, inputs_ph, labels_ph, target_cls, c, train_step, init, assign_op, adv_correct


def gen_batch(x, y, batch_size):
    assert len(x) == len(y)
    for i in range(len(x) // batch_size):
        si, ei = i * batch_size, (i + 1) * batch_size
        yield x[si:ei], y[si:ei]
    left = len(x) % batch_size
    if left > 0:
        yield x[-left:], y[-left:]


def predict_embedding(batch_x, batch_y, x_train, y_train):
    dist = np.linalg.norm(np.expand_dims(x_train, axis=0) - np.expand_dims(batch_x, axis=1), axis=2)
    nn_index = np.argmin(dist, axis=1)
    prediction = y_train[nn_index].flatten()
    return np.sum((prediction == batch_y).astype(np.int64))

    
def cw(root_path, config):
    sess = tf.Session()
    num_classes = 100
    batch_size = 25

    start_step = 30
    end_step = 40

    if config['model_type'] == 'npairs':
        adv_inputs, inputs_ph, mid_points, target_embedding, others_embedding, c, train_step, embedding_clean, init, assign_op, adv_embedding, loss1, loss2 = define_npairs(batch_size)

        model_path = 'checkpoint/npairs/cifar100.npairs.ckpt'
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        saver.restore(sess, model_path)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        idx = np.sort(np.concatenate([np.argwhere(y_test.flatten() == cls).flatten()[:10] for cls in range(num_classes)]))
        x_test = x_test[idx, ::]
        y_test = y_test[idx]

        if os.path.isfile('x_train_emb.npy'):
            x_train_emb = np.load('x_train_emb.npy')
        else:
            result = list()
            step = 0
            for x, y in gen_batch(x_train, y_train, batch_size):
                pred = sess.run(embedding_clean, feed_dict={inputs_ph: x})
                result.append(pred)
                step += 1
                print('\r[+] Inferencing training data : %d%%' % (step * 100 // (x_train.shape[0] // batch_size)), end='\t')
            x_train_emb = np.concatenate(result)
            np.save('x_train_emb.npy', x_train_emb)

        mid = np.asarray([np.mean(x_train_emb[np.where(y_train.flatten() == i)], axis=0) for i in range(num_classes)])

        correct_clean = 0
        correct_fgsm = 0
        count = 0
        result = list()

        print('[+] Generating AEs')
        step = start_step
        for x, y in gen_batch(x_test[start_step * batch_size : end_step * batch_size], y_test.flatten()[start_step * batch_size : end_step * batch_size], batch_size):
            target_y = np.asarray([np.random.choice([cls for cls in range(num_classes) if label != cls]) for label in y])
            target = mid[target_y]
            others_idx = np.asarray([[cls for cls in range(num_classes) if label != cls] for label in y])
            others = mid[others_idx]
            steps = int(1e4)
            min_c, max_c = 1e-2, 1e2
            c_value = 1e0
            adv = None
            for _ in range(10):
                sess.run(init)
                sess.run(assign_op, feed_dict={inputs_ph: x})
                for substep in range(steps):
                    l1, l2, _ = sess.run([loss1, loss2, train_step], feed_dict={
                        inputs_ph: x,
                        mid_points: mid,
                        target_embedding: target,
                        others_embedding: others,
                        c: c_value,
                    })
                    print('\rBatch %d : %d%% complete / loss1 = %.4f, loss2 = %.4f' % (step + 1, substep * 100 // steps, l1, l2), end='\t')
                adv_emb = sess.run(adv_embedding)
                current_adv = sess.run(adv_inputs)
                adv_acc = predict_embedding(adv_emb, y, x_train_emb, y_train)
                print('Adv accuracy = %.4f' % (adv_acc / batch_size))
                if adv_acc == 0:
                    adv = current_adv
                    c_value = (c_value + min_c) / 2
                else:
                    c_value = (c_value + max_c) / 2
            if adv is None:
                adv = sess.run(adv_inputs)
            step += 1
            assert adv.shape == (batch_size, 32, 32, 3)
            np.save('npairs_clean_{}.npy'.format(step), x)
            np.save('npairs_adv_{}.npy'.format(step), adv)
            result.append(adv)
        np.save('cw_adv.npy', np.stack(result))

    elif config['model_type'] == 'softmax':
        adv_inputs, inputs_ph, labels_ph, target_cls, c, train_step, init, assign_op, adv_correct = define_softmax(batch_size)

        model_path = 'checkpoint/softmax_real/cifar100.softmax.ckpt'
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'))
        saver.restore(sess, model_path)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='fine')
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        idx = np.sort(np.concatenate([np.argwhere(y_test.flatten() == cls).flatten()[:10] for cls in range(num_classes)]))
        x_test = x_test[idx, ::]
        y_test = y_test[idx]

        print('[+] Generating AEs')
        step = start_step
        for x, y in gen_batch(x_test[start_step * batch_size : end_step * batch_size], y_test.flatten()[start_step * batch_size : end_step * batch_size], batch_size):
            steps = int(1e4)
            min_c, max_c = 1e-2, 1e2
            c_value = 1e0
            adv = None
            for _ in range(20):
                sess.run(init)
                sess.run(assign_op, feed_dict={inputs_ph: x})
                target = np.asarray([np.random.choice([cls for cls in range(num_classes) if label != cls]) for label in y])
                steps = int(1e4)
                for substep in range(steps):
                    correct, _ = sess.run([adv_correct, train_step], feed_dict={
                        inputs_ph: x,
                        labels_ph: y.flatten(),
                        target_cls: target,
                        c: c_value,
                    })
                    print('\rBatch %d : %d%% complete. Accuracy = %.4f' % (step + 1, substep * 100 // steps, correct / batch_size), end='\t')
                    if correct == 0:
                        break
                current_adv = sess.run(adv_inputs)
                if correct == 0:
                    adv = current_adv
                    c_value = (c_value + min_c) / 2
                else:
                    c_value = (c_value + max_c) / 2
            if adv is None:
                current_adv = sess.run(adv_inputs)
            step += 1
            np.save('softmax_clean_{}.npy'.format(step), x)
            np.save('softmax_adv_{}.npy'.format(step), adv)
    print('')


if __name__ == '__main__':
    config = {
        'model_type': 'softmax',
    }
    main(config)
