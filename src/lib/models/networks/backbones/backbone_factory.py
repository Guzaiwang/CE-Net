from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from .mobilenet.mobilenet_factory import get_mobilenet_backbone
from .resnet.resnet_factory import get_resnet_backbone


def get_backbone_architecture(backbone_name):
    if "resnet" in backbone_name:
        return get_resnet_backbone(backbone_name)

    elif "mobilenet" in backbone_name:
        return get_mobilenet_backbone(backbone_name)