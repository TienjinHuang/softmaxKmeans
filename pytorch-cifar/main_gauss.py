'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import random


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--eps', default=0.1, type=float, help='epsilon perturbation')
parser.add_argument('--net', default='VGG16', help='network architecture')
parser.add_argument('--epochs', default=100, type=int, help='flag whether weights are set to centroids after each epoch')
parser.add_argument('--prot', default=1, type=int, help='number of prototypes')
parser.add_argument('--idx',default = '0', help='idx for stored files')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc =0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
c=10
# Model
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='VGG':
    net = VGGGauss('VGG16')
elif args.net=='ResNet':
    net = ResNet18Gauss()
else : print('Net ',args.net,' is not known, choose between VGG and ResNet')
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckptGA%s.t7'%(args.net+'_'+args.idx))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
I=torch.eye(10).to(device)

def perturb(inputs,Y,epsilon):
    rand_perturb = torch.FloatTensor(inputs.shape).uniform_(-epsilon, epsilon)
    rand_perturb = rand_perturb.to(device)
    perturb_inputs = inputs + rand_perturb
    #factors =torch.exp(-0.1*torch.sum(torch.sum(torch.sum(rand_perturb**2,1),1),1))
    #print(torch.mean(factors).item())
    #Y = Y*factors.unsqueeze(1).expand_as(Y)
    return (perturb_inputs,Y)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    conf=0
    batch_idx=0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        Y = I[targets]
        #if bool(random.getrandbits(1)):
        #    inputs,Y = perturb(inputs,Y,args.eps)
        optimizer.zero_grad()
        outputs = net(inputs)
        P1 = torch.exp(outputs)
        #loss = criterion(torch.exp(outputs/16), Y)+0.95*criterion(torch.exp(outputs/4), Y)+0.68*criterion(P1, Y)+0.25*criterion(torch.exp(9.7*outputs),Y)
        loss = criterion(torch.exp(outputs/16), Y)
        loss+= 0.95*criterion(torch.exp(outputs/4), Y)
        loss+= 0.68*criterion(P1, Y)
        loss+= 0.38*criterion(torch.exp(4*outputs),Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        confBatch, predicted = P1.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (100*train_loss/batch_idx, 100.*correct/total, correct, total, 100*conf/total))

def update_centroids(epoch):
    net.eval()
    W=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            D = net.module.get_D(inputs)
            Y = I[targets]
            if batch_idx==0:
                #W = D.t().mm(Y-0.5)
                W = D.t().mm(Y)
            else:
                #W += D.t().mm(Y-0.5)
                W += D.t().mm(Y)
    W = W/y_sum
    net.module.classifier.weight.data = W.t()


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    conf =0
    batch_idx=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            P1 =torch.exp(outputs)
            loss = criterion(P1, I[targets])

            test_loss += loss.item()
            confBatch, predicted = P1.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            conf+=confBatch.sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (100*test_loss/batch_idx, 100.*correct/total, correct, total, 100*conf/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckptGA%s.t7'%(args.net+'_'+args.idx))
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    if (epoch+1)%1==0:
        with torch.no_grad():
            net.eval()
            X = net.module.classifier.weight.data.t()
            print('||X||^2:')
            print(torch.sum(X**2,0))


print('results are at ./checkpoint/ckptGA%s.t7'%(args.net+'_'+args.idx))
