from turtle import distance
import torch.nn as nn
import torch

class CE_Loss(nn.Module):
    def __init__(self, c, device):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        #self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.Y_pred = 0
 
    def forward(self, inputs, targets,gamma2=None):  
        #self.Y_pred = self.classifier(inputs) # prediction before softmax
        self.Y_pred=inputs
        return self.ce_loss(self.Y_pred, targets)
    
    def conf(self,inputs):
        return self.softmax(inputs)
    
    def prox(self):
        return

class BCE_GALoss(nn.Module):
    def __init__(self, c, device):
        super(BCE_GALoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.bce_loss = nn.BCELoss()
        #self.mse_loss = nn.MSELoss(reduction='none')
        #self.classifier = classifier.to(device)
        #self.gamma2 = nn.Parameter(torch.ones(c)*0.9)
        #self.gamma2_min = gamma2_min
        #self.gamma2_max = gamma2_max
 
    def forward(self, inputs, targets,gamma2):        
        Y = self.I[targets]
        try:
            #distances = -self.classifier(inputs) 
            distances=-1*inputs
            loss = self.bce_loss(torch.exp(-1*distances),Y) 
            #mse_M = self.mse_loss(Y@self.classifier.weight,inputs)
            #mse_M = torch.diag(Y@self.classifier.gamma) @ mse_M
            loss+= torch.mean(Y*distances/gamma2)
        except RuntimeError as e:
            print("min,max D",torch.min(inputs).item(), torch.max(inputs).item())
            print("min,max output",torch.min(torch.exp(inputs)).item(), torch.max(torch.exp(inputs)).item())
            print("nans output",torch.sum(torch.isnan(torch.exp(inputs))).item())
            print(f"{e},{e.__class_}")
        return loss
    
    #def conf(self,net):
        #return self.classifier.conf(inputs)
        #return net.module.classifier.conf(inputs)
    
    #def prox(self):
        #torch.clamp_(self.gamma2, self.gamma2_min, self.gamma2_max)
        #self.classifier.prox()
      

class BCE_DUQLoss(nn.Module):
    
    def __init__(self, classifier, c, device):
        super(BCE_DUQLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.I = torch.eye(c).to(device)
        self.classifier = classifier.to(device)
        self.Y_pred = 0 #predicted class probabilities
        self.Y= 0
    
    def forward(self, inputs, targets):
        self.Y = self.I[targets]
        self.Y_pred = torch.exp(self.classifier(inputs))
        loss = self.bce_loss(self.Y_pred, self.Y)
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
    
    def prox(self):
        return
