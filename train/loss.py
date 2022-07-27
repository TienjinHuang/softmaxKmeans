import torch.nn as nn
import torch

class BCE_GALoss(nn.Module):
    def __init__(self, classifier, c, device):
        super(BCE_GALoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.classifier = classifier.to(device)
        self.gamma2 = nn.Parameter(torch.Tensor([0.9]))
 
    def forward(self, inputs, targets):        
        Y = self.I[targets]
        try:
            distances = self.classifier(inputs)
            #loss = self.bce_loss(torch.exp(self.classifier(inputs)*self.gamma2),Y) 
            loss = self.bce_loss(torch.exp(distances*self.gamma2),Y) 
            #mse_M = self.mse_loss(Y@self.classifier.weight,inputs)
            #mse_M = torch.diag(Y@self.classifier.gamma) @ mse_M
            loss+= torch.mean(Y*distances)
        except RuntimeError as e:
            print("min,max D",torch.min(inputs).item(), torch.max(inputs).item())
            print("min,max output",torch.min(torch.exp(inputs)).item(), torch.max(torch.exp(inputs)).item())
            print("nans output",torch.sum(torch.isnan(torch.exp(inputs))).item())
            print(f"{e},{e.__class_}")
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
      

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
