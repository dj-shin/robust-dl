from exp.model.elu import ELU
from exp.model.resnet import ResNet56, ResNet20, ResNet56Base
from exp.model.vgg import Vgg16
from exp.model.keras_cnn import KerasCNN


def get_model(name):
    name_map = {
        'elu': ELU,
        'resnet': ResNet56,
        'resnet_base': ResNet56Base,
        'resnet56': ResNet56,
        'resnet20': ResNet20,
        'keras': KerasCNN,
        'vgg16': Vgg16,
    }

    return name_map[name]
