import torch
import torch.nn.functional as F

class Optimizer:
  def __init__(self, optimizer, trainloader, device, update_centroids=False):
    self.optimizer = optimizer
    self.trainloader = trainloader
    self.n = len(trainloader.dataset)
    self.update_centroids = update_centroids
    self.device=device
    self.best_acc=0
    
  def gradient_penalty(self, inputs, outputs):
    gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
        )[0]
    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty

  def train_epoch(self, net, criterion, weight_gp_pred=0, weight_gp_embed=0, verbose=False):
    train_loss, correct, conf = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
      net.train()
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      inputs.requires_grad_(True)
      embedding = net.embed(inputs)
      loss = criterion(embedding,targets)
      if verbose:
        print("loss:",loss.item())
      #----- gradient penalty
      if weight_gp_pred > 0:
        loss += weight_gp_pred * self.gradient_penalty(inputs, criterion.Y_pred)
      if weight_gp_embed>0:
        gp = self.gradient_penalty(inputs, embedding)
        loss+= weight_gp_embed * gp
        if verbose:
          print("GP:",gp.item())
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      inputs.requires_grad_(False)

      with torch.no_grad():
        if self.update_centroids:
          net.eval()
          criterion.classifier.update_centroids(embedding, criterion.Y)
        train_loss += loss.item()
        confBatch, predicted = criterion.conf(embedding).max(1)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (train_loss/len(self.trainloader), 100.*correct/self.n, correct, self.n, 100*conf/self.n))
    return (100.*correct/self.n, 100*conf/self.n)
  
  def test_epoch(self, net, criterion, data_loader):
    net.eval()
    test_loss, correct, conf = 0,0,0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = net.embed(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            confBatch, predicted = criterion.conf(outputs).max(1)
            correct += predicted.eq(targets).sum().item()
            conf+=confBatch.sum().item()
    total = len(data_loader.dataset)
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (test_loss/max(len(data_loader),1), 100.*correct/total, correct, total, 100*conf/total))
    return (100.*correct/total, 100*conf/total)
  
  def optimize_centroids(self, net):
    net.eval()
    d,c = net.classifier.in_features,net.classifier.out_features
    Z=torch.zeros(d,c).to(self.device)
    y_sum = torch.zeros(c).to(self.device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            D = net.embed(inputs)
            Y = F.one_hot(targets, c).float().to(self.device)
            Z += D.t().mm(Y)
            y_sum += torch.sum(Y,0)
    Z = Z/y_sum
    net.classifier.weight.data = Z.t()
