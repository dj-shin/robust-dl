from exp.analysis.interpolation import interpolation_npairs, interpolation_softmax

from exp.generator.fgsm import fgsm
from exp.generator.fgsm_all import fgsm_all
from exp.generator.old_fgsm import fgsm as old_fgsm
from exp.generator.ifgsm import ifgsm
from exp.generator.step_ll import step_ll
from exp.generator.old_step_ll import step_ll as old_step_ll

from exp.train.softmax import train_softmax
from exp.train.npairs import train_npairs
from exp.train.histogram import train_histogram
from exp.train.ensemble import train_ensemble

from exp.predict.softmax import predict_softmax
from exp.predict.npairs import predict_npairs
from exp.predict.histogram import predict_histogram

import json
import logging
import os
import pprint
from datetime import datetime


root_path = os.path.dirname(os.path.realpath(__file__))


def assert_config(config, params, warn=None):
    field = config
    try:
        for param in params:
            field = field[param]
    except KeyError as exc:
        if warn is not None:
            logging.warning(warn)
            print(warn)
        else:
            raise exc

def train(config):
    assert_config(config, ['model', 'name'])
    assert_config(config, ['data'], 'No dataset specified. Default is cifar100')
    assert_config(config, ['lr'], 'No learning rate schedule specified. Default is 0.01')
    assert_config(config, ['data', 'preprocess', 'gcn'], 'Global contrast normalization = False')
    assert_config(config, ['data', 'preprocess', 'zca'], 'ZCA whitening = False')
    assert_config(config, ['optimizer', 'name'], 'No optimizer specified. Default is SGD')

    config_constraint = {
        'model': {
            'name': None,
            'type': None,
        },
        'data': {
            'name': 'No dataset specified. Default is cifar100',
            'preprocess': {
                'gcn': 'Global contrast normalization = False',
                'zca': 'ZCA whitening = False',
            }
        },
        'lr': 'No learning rate schedule specified. Default is 0.01',
        'optimizer': {
            'name': 'No optimizer specified. Default is SGD'
        }
    }

    if config['model']['type'] == 'softmax':
        train_softmax(root_path, config)
    if config['model']['type'] == 'npairs':
        train_npairs(root_path, config)
    if config['model']['type'] == 'histogram':
        train_histogram(root_path, config)
    if config['model']['type'] == 'ensemble':
        train_ensemble(root_path, config)


def predict(config):
    config_constraint = {
        'model': {
            'name': None,
            'type': None,
            'restore': {
                'uid': None,
                'epoch': 'No epoch specified'
            }
        },
        'data': {
            'name': 'No dataset specified. Default is cifar100',
            'source': None,
            'preprocess': {
                'gcn': 'Global contrast normalization = False',
                'zca': 'ZCA whitening = False',
            }
        },
    }

    if config['model']['type'] == 'softmax':
        predict_softmax(root_path, config)
    elif config['model']['type'] == 'npairs':
        predict_npairs(root_path, config)
    elif config['model']['type'] == 'histogram':
        predict_histogram(root_path, config)


def generate(config):
    config_constraint = {
        'model': {
            'name': None,
            'type': None,
            'restore': {
                'uid': None,
                'epoch': 'No epoch specified'
            }
        },
        'data': {
            'name': 'No dataset specified. Default is cifar100',
            'type': 'Dataset type not specified. Default is test dataset',
            'preprocess': {
                'gcn': 'Global contrast normalization = False',
                'zca': 'ZCA whitening = False',
            }
        },
        'generate': {
            'method': None,
        }
    }
    if config['generate']['method'] == 'fgsm':
        fgsm(root_path, config)
    elif config['generate']['method'] == 'fgsm_all':
        fgsm_all(root_path, config)
    elif config['generate']['method'] == 'old_fgsm':
        old_fgsm(root_path, config)
    elif config['generate']['method'] == 'ifgsm':
        ifgsm(root_path, config)
    elif config['generate']['method'] == 'step_ll':
        step_ll(root_path, config)
    elif config['generate']['method'] == 'old_step_ll':
        old_step_ll(root_path, config)


def analysis(config):
    config_constraint = {
        'model': {
            'name': None,
            'type': None,
            'restore': {
                'uid': None,
                'epoch': 'No epoch specified'
            }
        },
        'data': {
            'name': 'No dataset specified. Default is cifar100',
            'source': 'Valid npy or h5 file should be specified',
            'preprocess': {
                'gcn': 'Global contrast normalization = False',
                'zca': 'ZCA whitening = False',
            }
        },
    }
    if config['model']['type'] == 'softmax':
        interpolation_softmax(root_path, config)
    elif config['model']['type'] == 'npairs':
        interpolation_npairs(root_path, config)


def main(config):
    uid = datetime.now().strftime('%m%d-%H%M%S')
    config['uid'] = uid
    print('============== UID : {} =============='.format(uid))

    logfile = os.path.join(root_path, 'result/summary', '{}.log'.format(uid))
    logging.basicConfig(handlers=[logging.FileHandler(logfile)], level=logging.INFO)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(config)
    if config['task'] == 'train':
        train(config)
    elif config['task'] == 'predict':
        predict(config)
    elif config['task'] == 'generate':
        generate(config)
    elif config['task'] == 'analysis':
        analysis(config)
    else:
        pass


if __name__ == '__main__':
    with open('config.json', 'rt') as f:
        config = json.load(f)
    main(config)
