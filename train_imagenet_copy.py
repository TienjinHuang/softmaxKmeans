from __future__ import print_function
from urllib.parse import ParseResultBytes
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
#import matplotlib.pyplot as plt
import os
from train.models import *
import train.models.resnet_cifar as resnet_cifar
from train import *
from train.layers_copy import *
from train.loss_dist import *
from train.optimization_dist_copy import Optimizer
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import argparse
import torchvision
from torch.utils.data.distributed import DistributedSampler
import random


def add_parser_arguments(parser):
    parser.add_argument('--datadir', default='/projects/2/managed_datasets/imagenet/',help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',choices=['resnet18','resnet50'])
    parser.add_argument('--dataset', default='cifar10', type=str,choices=['imagenet','cifar10','cifar100'])
    parser.add_argument('--epochs', default=90, type=int, metavar='N',help='number of total epochs to run')    
    parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')                    
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--warmup', default=10, type=int,metavar='E', help='number of warmup epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum')
    parser.add_argument('--gamma', default=0.5, type=float, metavar='M',help='momentum')
    parser.add_argument('--seed', default=17, type=int,help='seed')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
add_parser_arguments(parser)
args = parser.parse_args()





# Data Loading functions {{{
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        #tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.tensor(nump_array)

    return tensor, targets


def prefetched_loader(loader, fp16):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
    if fp16:
        mean = mean.half()
        std = std.half()

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            if fp16:
                next_input = next_input.half()
            else:
                next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def get_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(), Too slow
            #normalize,
        ]))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    return train_loader

def get_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    valdir = os.path.join(data_path, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
        collate_fn=fast_collate)

    return val_loader
# }}}



#path = "checkpoint/"
def load_net(file,architecture):
    checkpoint = torch.load(file,map_location='cpu')
    architecture.load_state_dict(checkpoint['net'])
    print('Loaded ACC:',checkpoint['acc'])
    return architecture

def main():
    dist.init_process_group('nccl', init_method='env://')
    rank = dist.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    world_size = dist.get_world_size()
    print(f"rank = {rank} is initialized in {master_addr}:{master_port}; local_rank = {local_rank};world_size={world_size}")
    torch.cuda.set_device(rank)
    device=torch.device('cuda',local_rank)
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + local_rank)
        torch.cuda.manual_seed(args.seed + local_rank)
        np.random.seed(seed=args.seed + local_rank)
        random.seed(args.seed + local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + local_rank + id)
            random.seed(args.seed + local_rank + id)
    else:
        def _worker_init_fn(id):
            pass


    if args.dataset=='cifar10':
        c=10   
        name="Cifar10"
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
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        trainsampler=DistributedSampler(trainset,shuffle=True)
        testsampler=DistributedSampler(testset,shuffle=False)
        trainloader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,prefetch_factor=2,pin_memory=True,sampler=trainsampler,num_workers=4)
        testloader=torch.utils.data.DataLoader(testset,batch_size=args.batch_size,prefetch_factor=2,pin_memory=True,sampler=testsampler,num_workers=4)
    if args.dataset=='cifar100':
        c=100   
        name="Cifar100"
        print('==> Preparing data..')
        #CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        #CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        trainsampler=DistributedSampler(trainset,shuffle=True)
        testsampler=DistributedSampler(testset,shuffle=False)
        trainloader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,prefetch_factor=2,pin_memory=True,sampler=trainsampler,num_workers=4)
        testloader=torch.utils.data.DataLoader(testset,batch_size=args.batch_size,prefetch_factor=2,pin_memory=True,sampler=testsampler,num_workers=4)
    elif args.dataset=='imagenet':
        c=1000    
        name='imagenet'        
        traindir = os.path.join(args.datadir, 'train')
        valdir = os.path.join(args.datadir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        trainsampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        testsampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,prefetch_factor=2, shuffle=(trainsampler is None),num_workers=36, pin_memory=True, sampler=trainsampler)
        testloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, prefetch_factor=2,shuffle=False,num_workers=36, pin_memory=True, sampler=testsampler)    
        # trainloader = get_train_loader(args.datadir, args.batch_size, workers=30, _worker_init_fn=_worker_init_fn)  
        # testloader = get_val_loader(args.datadir, args.batch_size, workers=30, _worker_init_fn=_worker_init_fn)  
    ###Building model############
    if args.arch=='resnet18':
        d=512
        #classifier=nn.Linear(d, c, bias=True)
        classifier=Gauss(in_features = d, out_features = c, gamma=0.5)
        net = resnet_cifar.ResNet18(classifier)   
    elif args.arch=='resnet50':
        d=2048
        #classifier=Linear(d, c, bias=True)
        classifier=Gauss(in_features = d, out_features = c, gamma=0.5)
        net = resnet_cifar.ResNet50(classifier)
    else:
        print("Not defined arch!")
        assert False
    #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    criterion = BCE_GALoss(c, device)
    print('./checkpoint/%s%s%s_base.t7'%(name,net.__class__.__name__,net.classifier.__class__.__name__),flush=True)
    if args.warmup>0 and os.path.exists('./checkpoint/%s%s%s_base.t7'%(name,net.__class__.__name__,net.classifier.__class__.__name__)):
        
        net=load_net('./checkpoint/%s%s%s_base.t7'%(name,net.__class__.__name__,net.classifier.__class__.__name__),net)
        net=net.to(device)
        #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

        # criterion = CE_Loss(c, device)
        # criterion=criterion.to(device)
        # sgd = optim.SGD([{'params': net.parameters()},],lr=0.1, momentum=0.9, weight_decay=5e-4)
        # optimizer = Optimizer(sgd, trainloader, device,local_rank=local_rank,world_size=world_size)



    elif args.warmup>0 and ~os.path.exists('./checkpoint/%s%s%s_base.t7'%(name,net.__class__.__name__,net.classifier.__class__.__name__)):  
        if args.arch=='resnet18':
            classifier=Linear(d, c, bias=True)
            net = resnet_cifar.ResNet18(classifier).to(device)   
        elif args.arch=='resnet50':
            classifier=Linear(d, c, bias=True)
            net = resnet_cifar.ResNet50(classifier).to(device)
        ####Warmup using CE LOSS################
        #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
        criterion = CE_Loss(c, device)
        criterion=criterion.to(device)

        sgd = optim.SGD([{'params': net.parameters()},],lr=0.1, momentum=0.9, weight_decay=5e-4)
        optimizer = Optimizer(sgd, trainloader, device,local_rank=local_rank,world_size=world_size)

        best_acc, epoch_offset =0, 0
        for (lr,max_epochs) in [(0.1,20),(0.01,20),(0.001,20)]: # train base pre-trained model
        #for (lr,max_epochs) in [(0.05,10)]:
        #for (lr,max_epochs) in [(0.1,20),(0.01,30),(0.001,50)]:
            optimizer.optimizer.param_groups[0]['lr'] = lr
            #print("GPU id",local_rank,"===== Optimize with step size ",lr)
            for epoch in range(epoch_offset, epoch_offset+ max_epochs):
                if dist.get_rank()==0:
                    print('\nEpoch: %d' % epoch)
                trainsampler.set_epoch(epoch)
                #trainloader.sampler.set_epoch(epoch)
                #testsampler.set_epoch(epoch)
                optimizer.train_epoch(net, criterion)
                (acc,conf) = optimizer.test_acc(net,criterion, testloader)
                if acc > .99*best_acc:
                    if dist.get_rank()==0:
                        print('Saving..')
                        state = {
                            'net': net.module.state_dict(),
                            'acc': acc
                        }
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        #torch.save(state, './checkpoint/%s%s%s_baselinear.t7'%(name,net.module.__class__.__name__,net.module.classifier.__class__.__name__))
                        best_acc = acc
            epoch_offset +=max_epochs
        #####################Calculate the centroids#################################
        #(acc,conf) = optimizer.test_acc(net,criterion, testloader)
        classifier = Gauss(in_features = d, out_features = c, gamma=0.5).to(device)
        #classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        net.module.classifier = classifier
        #net = net.to(device)
        criterion = BCE_GALoss( c, device)
        criterion=criterion.to(device)         
        (acc,conf) = optimizer.test_acc(net,criterion, testloader)
        optimizer.optimize_centroids(net)       
        (acc,conf) = optimizer.test_acc(net,criterion, testloader)
            
        state = {
            'net': net.module.state_dict(),
            'acc': acc
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/%s%s%s_base.t7'%(name,net.module.__class__.__name__,net.module.classifier.__class__.__name__))
        #############reintialize model
        if args.arch=='resnet18':
            d=512
            #classifier=nn.Linear(d, c, bias=True)
            classifier=Gauss(in_features = d, out_features = c, gamma=0.5)
            net = resnet_cifar.ResNet18(classifier)   
        elif args.arch=='resnet50':
            d=2048
            #classifier=nn.Linear(d, c, bias=True)
            classifier=Gauss(in_features = d, out_features = c, gamma=0.5)
            net = resnet_cifar.ResNet50(classifier)
        else:
            print("Not defined arch!")
            assert False
        net=load_net('./checkpoint/%s%s%s_base.t7'%(name,net.__class__.__name__,net.classifier.__class__.__name__),net)
        net=net.to(device)
        #net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net=torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    ######Training for Gaussian Net####################

    

    weight_gp_embed=0
    criterion = BCE_GALoss( c, device)
    criterion=criterion.to(device)

    sgd = optim.SGD([
                    {'params': net.module.embed.parameters()},
                    {'params': net.module.classifier.weight, 'weight_decay': 0},
                    {'params': net.module.classifier.gamma, 'weight_decay': -1e-10},
                    {'params': net.module.classifier.gamma2, 'weight_decay': 1e-10}],
                    lr=0.1, momentum=0.9, weight_decay=5e-4)
    optimizer = Optimizer(sgd, trainloader, device,local_rank=local_rank,world_size=world_size)
    ############test#########################
    (acc,conf) = optimizer.test_acc(net,criterion, testloader)

    best_acc, epoch_offset =0,30    
    for lr, max_epoch in [(0.001, 15),(0.0005,25),(0.0004,15),(0.0001,15)]:
    #for lr, max_epoch in [(0.05, 15),(0.01,25),(0.002,25),(0.0004,25)]:
        #h=0.1
        optimizer.optimizer.param_groups[0]['lr'] = lr
        optimizer.optimizer.param_groups[1]['lr'] = lr
        optimizer.optimizer.param_groups[2]['lr'] = lr
        optimizer.optimizer.param_groups[3]['lr'] = lr
        if dist.get_rank()==0:
            print("Optimize with step size ",lr)
        for epoch in range(epoch_offset,epoch_offset+ max_epoch):            
            #if torch.distributed.is_initialized():
            #trainloader.sampler.set_epoch(epoch)
            #testsampler.set_epoch(epoch)
            trainsampler.set_epoch(epoch)
            if dist.get_rank()==0:
                print('\nEpoch: %d' % epoch)
            optimizer.train_epoch(net, criterion, weight_gp_embed=weight_gp_embed, verbose=False)
            (acc,conf) = optimizer.test_acc(net,criterion, testloader)
            if dist.get_rank()==0:
                print("gamma", net.module.classifier.gamma)
                print("gamma2", net.module.classifier.gamma2)
            if acc > .99*best_acc:
                if dist.get_rank()==0:
                    print('Saving..')
                    state = {
                        'net': net.module.state_dict(),
                        'acc': acc
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/%s%s%s.t7'%(name,net.module.__class__.__name__,net.module.classifier.__class__.__name__))
                    best_acc = acc
                
            if epoch%5==0:
                with torch.no_grad():
                    X = net.module.classifier.weight.data.t()
                    margins = torch.sqrt(net.module.classifier.get_margins())
                    gamma = net.module.classifier.gamma
                    if dist.get_rank()==0:
                        print('||X||^2: %.1f +- %.3f'% (torch.mean(torch.sum(X**2,0)), torch.std(torch.sum(X**2,0))))
                        print('Min margin: %.2f, mean margin: %.2f +- %.3f'% (torch.min(margins), torch.mean(margins), torch.std(margins)))
                        print('gamma: %.2f +- %.3f'% (torch.mean(gamma), torch.std(gamma)))
                    
        epoch_offset +=max_epoch
    
if __name__=='__main__':
    main()




        










    
 



