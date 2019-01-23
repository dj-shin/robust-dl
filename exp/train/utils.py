import tensorflow as tf
import inspect


def fill_args(target, **kwargs):
    if inspect.isclass(target):
        params = dict()
        for argname in inspect.getfullargspec(target.__init__)[0][1:]:
            if kwargs.get(argname) is not None:
                params[argname] = kwargs.get(argname)
        return target(**params)
    elif inspect.isfunction(target):
        params = dict()
        for argname in inspect.getfullargspec(target)[0]:
            if kwargs.get(argname) is not None:
                params[argname] = kwargs.get(argname)
        return target(**params)

def get_optimizer(name, lr_ph, **params):
    name_map = {
        'sgd': tf.train.GradientDescentOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer,
        'momentum': tf.train.MomentumOptimizer,
        'adam': tf.train.AdamOptimizer,
    }
    return fill_args(name_map[name], learning_rate=lr_ph, **params)
