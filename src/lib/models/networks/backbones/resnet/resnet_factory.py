# *coding:utf-8 *


from .build_resnet import get_resnet_18
from .build_resnet import get_resnet_34
from .build_resnet import get_resnet_50
from .build_resnet import get_resnet_101
from .build_resnet import get_resnet_152

_resnet_backbone = {
    'resnet18': get_resnet_18,
    'resnet34': get_resnet_34,
    'resnet50': get_resnet_50,
    'resnet101': get_resnet_101,
    'resnet152': get_resnet_152,

}


def get_resnet_backbone(model_name):
    support_resnet_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    assert model_name in support_resnet_models, "We just support the following models: {}".format(support_resnet_models)

    model = _resnet_backbone[model_name]

    return model

if __name__ == '__main__':
    str1 = 'resnet18'
    model = get_resnet_backbone(str1)

    print(model(pretrain=True))