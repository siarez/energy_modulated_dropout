'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2dNormal
from custom_layers import Conv2DCustom


cfg = {
    'VGG_tiny': [32, 'M', 64, 'M', 128, 128, 'M'],
    'VGG_mini': [32, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

Conv2d = None

class VGG(nn.Module):
    def __init__(self, vgg_name, normal=True, dropout=0.):
        super(VGG, self).__init__()
        global Conv2d
        if normal:
            Conv2d = Conv2dNormal
        else:
            Conv2d = Conv2DCustom
        self.features = self._make_layers(cfg[vgg_name], dropout=dropout)
        if vgg_name == 'VGG_mini':
            self.classifier = nn.Linear(256, 10)
        elif vgg_name == 'VGG_tiny':
            self.classifier = nn.Linear(2048, 10)
        else:
            self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        out = self.classifier(features)
        return out, features

    def _make_layers(self, cfg, dropout=0.):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True),
                           nn.Dropout2d(p=dropout)]
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
