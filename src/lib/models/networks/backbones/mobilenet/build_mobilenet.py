import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

from .basic_module import ConvBNReLU, InvertedResidual, _make_divisible, load_model

model_urls = {
    'mobilenetv2_10': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 0
            [6, 24, 2, 2],  # 1
            [6, 32, 3, 2],  # 2
            [6, 64, 4, 2],  # 3
            [6, 96, 3, 1],  # 4
            [6, 160, 3, 2],  # 5
            [6, 320, 1, 1],  # 6
        ]

        self.feat_id = [1, 2, 4, 6]
        self.feat_channel = []

        # only check the first element, assuming user know t,c,n,s
        # are required

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks

        for id, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id:
                self.__setattr__("layer%d" % id, nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = []
        for id in self.feat_id:
            x = self.__getattr__("layer%d" % id)(x)
            y.append(x)

        return y

    def init_weights(self, model_name):
        url = model_urls[model_name]
        pretrained_state_dict = model_zoo.load_url(url)
        print('=> loading pretrained model {}'.format(url))
        self.load_state_dict(pretrained_state_dict, strict=False)


def get_mobilenetv2_10(pretrained=True, **kwargs):
    model = MobileNetV2(width_mult=1.0)
    if pretrained:
        model.init_weights(model_name='mobilenetv2_10')

    return model


def get_mobilenetv2_5(pretrained=True, **kwargs):
    model = MobileNetV2(width_mult=0.5)
    if pretrained:
        print("MobilenetV2_5 does not have the pretrained weight")

    return model

if __name__ == '__main__':

    model = get_mobilenetv2_10(pretrained=True)

    input = torch.zeros([1, 3, 512, 512])
    feats = model(input)
    print(feats[0].size())
    print(feats[1].size())
    print(feats[2].size())
    print(feats[3].size())







