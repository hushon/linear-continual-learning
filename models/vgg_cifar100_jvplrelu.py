"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from .modules import CustomLeakyReLU, CustomLinear, CustomConv2d, CustomBatchNorm2d, CustomSequential, CustomMaxPool2d

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes, dropout=False):
        super().__init__()
        self.features = features

        print(f"{dropout=}")

        if num_classes > 0:
            # self.classifier = nn.Sequential(
            #     nn.Linear(512, 4096),
            #     nn.LeakyReLU(),
            #     nn.Dropout() if dropout else nn.Identity(),
            #     nn.Linear(4096, 4096),
            #     nn.LeakyReLU(),
            #     nn.Dropout() if dropout else nn.Identity(),
            #     nn.Linear(4096, num_classes)
            # )
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward(self, x):
        jvp = torch.zeros_like(x)
        output, jvp = self.features(x, jvp)
        output = output + jvp
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [CustomMaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [CustomConv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [CustomBatchNorm2d(l)]

        layers += [CustomLeakyReLU()]
        input_channel = l

    return CustomSequential(*layers)

def vgg11_bn(num_classes=100):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)

def vgg13_bn(num_classes=100):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes)

def vgg16_bn(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes)

def vgg19_bn(num_classes=100):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes)