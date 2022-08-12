from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .build_mobilenet import get_mobilenetv2_10
from .build_mobilenet import get_mobilenetv2_5

_mobilenet_backbone = {
    'mobilenetv2_10': get_mobilenetv2_10,
    'mobilenetv2_5': get_mobilenetv2_5,

}


def get_mobilenet_backbone(model_name):
    support_mobile_models = ['mobilenetv2_10', 'mobilenetv2_5']
    assert model_name in support_mobile_models, "We just support the following models: {}".format(support_mobile_models)

    model = _mobilenet_backbone[model_name]

    return model


if __name__ == '__main__':
    str1 = 'mobilenetv2_10'
    model = get_mobilenet_backbone(str1)

    print(model(pretrain=True))
