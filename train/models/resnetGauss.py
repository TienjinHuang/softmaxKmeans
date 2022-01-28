'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Gauss(nn.Module):
    __constants__ = ['in_features', 'out_features','num_classes']

    def __init__(self,in_features,out_features,num_classes,gamma):
        super(Gauss, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.gamma=gamma
        self.gamma=nn.Parameter(gamma*torch.ones(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.uniform_(self.weight,a=0,b=0.1)

    def forward(self, D):
        DX = D.mm(self.weight.t())
        out = -torch.sum(D**2,1).unsqueeze(1).expand_as(DX)
        out = out + 2*DX
        out = out - torch.sum(self.weight.t()**2,0).unsqueeze(0).expand_as(DX)
        return self.gamma* out

class ResNetGauss(nn.Module):
    def __init__(self, block, num_blocks, num_classes, gamma,  num_prot=1):
        super(ResNetGauss, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = Gauss(512*block.expansion, num_classes*num_prot, num_classes, gamma)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def gaussConf(D):
        X = self.classifier.weight.data.t()
        DX = D.mm(X)
        out = -torch.sum(D**2,1).unsqueeze(1).expand_as(DX)
        out = out + 2*DX
        out = out - torch.sum(X**2,0).unsqueeze(0).expand_as(DX)
        return torch.exp(out)

    def get_margins(self):
        #X is dxc, out is cxc matrix, containing the distances ||X_i-X_j||
        # only the upper triangle of out is needed
        X = self.classifier.weight.data.t()
        XX = X.t().mm(X)
        out = -torch.sum(X.t()**2,1).unsqueeze(1).expand_as(XX)
        out = out + 2*XX
        out = out - torch.sum(X**2,0).unsqueeze(0).expand_as(XX)
        triu_idx = torch.triu_indices(out.shape[0], out.shape[0],1)
        return -out[triu_idx[0],triu_idx[1]]

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
#        out = gaussConf(self,out)
        return out

    def get_D(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out.view(out.size(0), -1)

def ResNet18Gauss(c=10, gamma=1):
    return ResNetGauss(BasicBlock, [2,2,2,2],c,gamma)

def ResNet34Gauss(c=10, gamma=1):
    return ResNetGauss(BasicBlock, [3,4,6,3],c,gamma)

def ResNet50Gauss(c=10, gamma=1):
    return ResNetGauss(Bottleneck, [3,4,6,3],c,gamma)

def ResNet101Gauss(c=10, gamma=1):
    return ResNetGauss(Bottleneck, [3,4,23,3],c,gamma)

def ResNet152Gauss(c=10, gamma=1):
    return ResNetGauss(Bottleneck, [3,8,36,3],c,gamma)


def test():
    net = ResNet18Gauss()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
