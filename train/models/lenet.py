'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
#from .. import layers

class LeNetEmbed(nn.Module):
    def __init__(self,embedding_dim=84):
        super(LeNetEmbed, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 120)
        self.fc2   = nn.Linear(120, embedding_dim)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out
    
    
class LeNet(nn.Module):
    def __init__(self,embedding_dim=84, classifier):
        super(LeNet, self).__init__()
        self.embed = LeNetEmbed(embedding_dim=embedding_dim)
        #self.classifier   = nn.Linear(embedding_dim, num_classes,bias=False)
        self.classifier = classifier

    def forward(self, x):
        out = self.embed(x)
        out = self.classifier(out)
        return out
    
    #def conf(self,x):
    #    out = self.forward(x)
    #    return F.softmax(out)
