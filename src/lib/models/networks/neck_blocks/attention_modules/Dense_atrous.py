import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(nn.ReLU, inplace=True)


class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilated1_out = nn.ReLU(self.dilate1(x))
        dilated2_out = nn.ReLU(self.conv1x1(self.dilate2(x)))
        dilated3_out = nn.ReLU(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilated4_out = nn.ReLU(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilated1_out + dilated2_out + dilated3_out + dilated4_out
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)

        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # F.upsample ----> F.interpolate
        # UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0.
        # Please specify align_corners=True if the old behavior is desired.
        # See the documentation of nn.Upsample for details.
        # warnings.warn("Default upsampling behavior when mode={} is changed "
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


if __name__ == '__main__':
    inp = torch.zeros(size=(10, 12, 14, 14))
    module = PSPModule(12)
    oup = module(inp)
    print(module)
    print(oup.size())
