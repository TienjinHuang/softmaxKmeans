'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Gauss(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features, gamma, gamma_min=0.1,gamma_max=1000,gamma2_min = 0.1, gamma2_max = 0.9):
        super(Gauss, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gamma=nn.Parameter(gamma*torch.ones(out_features))
        self.gamma2=nn.Parameter(torch.ones(out_features)*0.9)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) # (cxd)
        
        #self.weight.requires_grad=False
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight,a=5/(gamma**2),b=50/(gamma**2))
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma2_min=gamma2_min
        self.gamm2_max=gamma2_max

    def forward(self, D):
        #DX = D.mm(self.weight.t())
        #out = torch.sum(D**2,1).unsqueeze(1).expand_as(DX)
        #out = out - 2*DX
        #out = out + torch.sum(self.weight.t()**2,0).unsqueeze(0).expand_as(DX)
        out = D.unsqueeze(2) - self.weight.t().unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        return -self.gamma*torch.sum((out**2),1) # (mxc)
        #return -F.relu(self.gamma*out)
    
    def conf(self,D):
        return torch.exp(self.forward(D))
    
    def prox(self):
        torch.clamp_(self.gamma, self.gamma_min, self.gamma_max)
        torch.clamp_(self.gamma2,self.gamma2_min,self.gamm2_max)
        
    
    def get_margins(self):
        #X is dxc, out is cxc matrix, containing the distances ||X_i-X_j||
        # only the upper triangle of out is needed
        X = self.weight.data.t()
        XX = X.t().mm(X)
        out = -torch.sum(X.t()**2,1).unsqueeze(1).expand_as(XX)
        out = out + 2*XX
        out = out - torch.sum(X**2,0).unsqueeze(0).expand_as(XX)
        triu_idx = torch.triu_indices(out.shape[0], out.shape[0],1)
        return -out[triu_idx[0],triu_idx[1]]
class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features,bias=False):
        super(Linear, self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.gamma2=0.9
        self.linear=nn.Linear(in_features,out_features,bias=bias)
    def forward(self, D):
        out=self.linear(D)
        return out
    
    def conf(self,out):
        return self.softmax(self.forward(out))
    def prox(self):
        return 

    
class Gauss_MV(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features, gamma):
        super(Gauss_MV, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features ,out_features)) #centroids (dxc)
        self.W = nn.Parameter(torch.einsum('k,il->kil',gamma*torch.ones(out_features),torch.eye(in_features) )) # Whitening matrix (cxrxd) = (cxdxd)
        #self.weight.requires_grad=False
        #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight,a=5/(gamma**2),b=50/(gamma**2))

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
    
    def prox(self):
        return
    
class Gauss_DUQ(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, gamma, N_init=None, m_init=None, alpha=0.999):
        super(Gauss_DUQ, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gamma=gamma
        self.alpha=alpha
        if N_init==None:
            N_init = torch.ones(out_features)*10
        if m_init==None:
            m_init = torch.normal(torch.zeros(in_features, out_features), 0.05)
        self.register_buffer("N", N_init) # 
        self.register_buffer(
            "m", m_init # (dxc)
        )
        self.m = self.m * self.N
        self.W = nn.Parameter(torch.zeros(in_features, out_features, in_features)) # (dxcxr) (r=c?)
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

    def forward(self, D):
        DW = torch.einsum("ij,mnj->imn", D, self.W) # (mxdxc)
        Z = self.m / self.N.unsqueeze(0) # centroids (dxc)
        out = DW - Z.unsqueeze(0)
        return -self.gamma*torch.mean((out**2),1) # (mxc)
    

    def conf(self,D):
        return torch.exp(self.forward(D))
    
    def update_centroids(self, D, Y):
        DW = torch.einsum("ij,mnj->imn", D, self.W) # (mxdxc)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.alpha * self.N + (1 - self.alpha) * Y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", DW, Y)

        self.m = self.alpha * self.m + (1 - self.alpha) * features_sum

 
