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


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--eps', default=0.1, type=float, help='epsilon perturbation')
parser.add_argument('--net', default='LeNet', help='network architecture')
parser.add_argument('--initCentroids', default=False, type=int, help='flag whether to initialize with class centroids')
parser.add_argument('--epochs', default=100, type=int, help='flag whether weights are set to centroids after each epoch')
parser.add_argument('--prot', default=1, type=int, help='number of prototypes')
parser.add_argument('--idx',default = '0', help='idx for stored files')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc =0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=trans)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
# Select only some classes for motivating picture
idx = (trainset.targets==0) | (trainset.targets==1) | (trainset.targets==2)
trainset.targets = trainset.targets[idx]
trainset.data = trainset.data[idx]

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=trans)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)
# Select only some classes for motivating picture
idx = (testset.targets==0) | (testset.targets==1) | (testset.targets==2)
testset.targets = testset.targets[idx]
testset.data = testset.data[idx]

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
c=10
classes = ('0', '1', '2')
c=3
# Model
print('==> Building model..')
if args.net=='LeNet':
    net = LeNetGauss(embedding_dim=2, num_classes=c, gamma=1)
else : print('Net ',args.net,' is not known, choose between LeNet')
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
    net.load_state_dict(checkpoint['net'],strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.MSELoss()
my_list = ['module.classifier.weight','module.classifier.gamma']
params = list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))
base_params = list(filter(lambda kv: kv[0] not in my_list, net.named_parameters()))
optimizer = optim.SGD([
                {'params': [x[1] for x in base_params]},
                {'params': [x[1] for x in params], 'weight_decay': 0}],
                lr=args.lr, momentum=0.9, weight_decay=5e-4)
I=torch.eye(c).to(device)
y_sum = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(trainloader):
       targets = targets.to(device)
       Y = I[targets]
       if batch_idx==0:
           y_sum = torch.sum(Y,0)
       else:
           y_sum += torch.sum(Y,0)

def update_centroids():
    net.eval()
    W=0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            D = net.module.get_D(inputs)
            Y = I[targets]
            if batch_idx==0:
                W = D.t().mm(Y)
            else:
                W += D.t().mm(Y)
    W = W/y_sum
    net.module.classifier.weight.data = W.t()

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
        loss = criterion(torch.exp(outputs/32), Y)
        loss+= criterion(P1, Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        confBatch, predicted = P1.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (100*train_loss/batch_idx, 100.*correct/total, correct, total, 100*conf/total))

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

if args.initCentroids:
    update_centroids()
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    if (epoch+1)%1==0: #print some statistics
        with torch.no_grad():
            X = net.module.classifier.weight.data.t()
            print('||X||^2: %.1f +- %.3f'% (torch.mean(torch.sum(X**2,0)), torch.std(torch.sum(X**2,0))))
            margins = net.module.get_margins()
            print('Min margin: %.2f, mean margin: %.2f +- %.3f'% (torch.min(margins), torch.mean(margins), torch.std(margins)))
            print('gamma: %.2f +- %.3f'% (torch.mean(net.module.classifier.gamma), torch.std(net.module.classifier.gamma)))


print('results are at ./checkpoint/ckptGA%s.t7'%(args.net+'_'+args.idx))
