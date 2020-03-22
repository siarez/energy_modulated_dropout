'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2dNormal
from torch.nn import Linear as LinearNormal
from custom_layers import Conv2DCustom, WandBLogger, LinearCustom


cfg = {
    'VGG_tiny': [32, 'M', 64, 'M', 128, 128, 'M'],
    'VGG_mini': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

Conv2d = None
Linear = None

class NamedNormalConv2D(Conv2dNormal):
    """This is a wrapper for normal convolution so it can accept a `name` parameter"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, name=None):
        super(NamedNormalConv2D, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        self.name = name

class VGG(nn.Module):
    def __init__(self, vgg_name, normal=True, dropout=0., bn_affine=True):
        super(VGG, self).__init__()
        global Conv2d
        global Linear
        if normal:
            Conv2d = NamedNormalConv2D
            Linear = LinearNormal
        else:
            Conv2d = Conv2DCustom
            Linear = LinearCustom
        self.features = self._make_layers(cfg[vgg_name], dropout=dropout, bn_affine=bn_affine)
        if vgg_name == 'VGG_mini':
            self.classifier = Linear(256, 10)
        elif vgg_name == 'VGG_tiny':
            self.classifier = Linear(2048, 10)
        else:
            self.classifier = Linear(512, 10)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out

    def _make_layers(self, cfg, dropout=0., bn_affine=True):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1, name='L-{}|ch-{}'.format(i, x)),
                           nn.BatchNorm2d(x, affine=bn_affine),
                           nn.ReLU(inplace=True),
                           nn.Dropout2d(p=dropout),
                           WandBLogger('L-{}|ch-{}_activation'.format(i, x), frac_zero=True)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    print('Num of parameters: ', sum(p.numel() for p in net.parameters() if p.requires_grad))
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
