'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Gauss(nn.Module):
    __constants__ = ['in_features', 'out_features', 'num_classes']

    def __init__(self,in_features,out_features,num_classes, gamma):
        super(Gauss, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gamma=nn.Parameter(gamma*torch.ones(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        #self.weight.requires_grad=False
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight,a=0,b=0.1)

    def forward(self, D):
        DX = D.mm(self.weight.t())
        out = torch.sum(D**2,1).unsqueeze(1).expand_as(DX)
        out = out - 2*DX
        out = out + torch.sum(self.weight.t()**2,0).unsqueeze(0).expand_as(DX)
        return -F.relu(self.gamma*out)
    
    def conf(self,D):
        return torch.exp(self.forward(D))
    
class Gauss_MV(nn.Module):
    __constants__ = ['in_features', 'out_features', 'num_classes']

    def __init__(self,in_features,out_features,num_classes, gamma):
        super(Gauss_MV, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features ,out_features)) #centroids (dxc)
        self.W = nn.Parameter(torch.Tensor(out_features,in_features,in_features)) # Whitening matrix (cxrxd)
        #self.weight.requires_grad=False
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight,a=0,b=gamma)
        nn.init.uniform_(self.W,a=0,b=gamma)

    def forward(self, D):
        WDt = torch.matmul(self.W, D.t()) #c x r x m
        WZ = torch.einsum('crd,dc->cr', self.W, self.weight) # (cxr)
        DMZ = torch.einsum('crj,cr->jc', WDt, WZ) # m x c
        out = torch.sum(WDt**2,1).t() # m x c
        out = out - 2*DMZ
        out = out + torch.sum(WZ**2,1).unsqueeze(0).expand_as(DMZ)
        return -F.relu(out)

    def conf(self,D):
        return torch.exp(self.forward(D))

class LeNetGauss(nn.Module):
    def __init__(self,embedding_dim=84, num_classes=10, gamma=1):
        super(LeNetGauss, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, embedding_dim)
        #self.classifier  = Gauss(embedding_dim, num_classes,num_classes,  gamma)
        self.classifier  = Gauss_MV(embedding_dim, num_classes,num_classes,  gamma)

    def get_D(self,x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.classifier(out)
        return out

    def conf(self,x):
        return torch.exp(self.forward(x))

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

