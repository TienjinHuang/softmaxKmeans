
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG16_100': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 100, 'M'],
    'VGG16_10': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 10, 'M'],
    'VGG16_2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 2, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Gauss(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features):
        super(Gauss, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, D):
        DX = D.mm(self.weight.t())
        out = -torch.sum(D**2,1).unsqueeze(1).expand_as(DX)
        out = out + 2*DX
        out = out - torch.sum(self.weight.t()**2,0).unsqueeze(0).expand_as(DX)
        return out

class VGGGauss(nn.Module):
    def __init__(self, vgg_name, gc=0):
        super(VGGGauss, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        if gc:
            #self.classifier = LinearKM(512, 10)
            classes = 11
        else:
            classes = 10
        if vgg_name == 'VGG16_100':
            self.classifier = Gauss(100,classes)
        elif vgg_name == 'VGG16_2':
            self.classifier = Gauss(2,classes)
        elif vgg_name == 'VGG16_10':
            self.classifier = Gauss(10,classes)
        else :
            self.classifier = Gauss(512,classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_D(self,x):
        out = self.features(x)
        return out.view(out.size(0), -1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGGGauss('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
