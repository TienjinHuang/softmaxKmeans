import torch

class Optimizer:
  def __init__(self, optimizer, trainloader, update_centroids, device):
    self.optimizer = optimizer
    self.trainloader = trainloader
    self.update_centroids = update_centroids
    self.device=device
    
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

  def train_epoch(self, net, criterion, weight_gp_pred=0, weight_gp_embed=0):
    #print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total, conf, batch_idx = 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      self.optimizer.zero_grad()
      inputs.requires_grad_(True)
      embedding = net.embed(inputs)
      loss = criterion(embedding,targets)
      #----- gradient penalty
      if weight_gp_pred > 0:
        loss += weight_gp_pred * self.gradient_penalty(inputs, criterion.Y_pred)
      if weight_gp_embed>0:
        loss+= weight_gp_embed * self.gradient_penalty(inputs, embedding)
      loss.backward()
      self.optimizer.step()
      inputs.requires_grad_(False)

      with torch.no_grad():
        if self.update_centroids:
          net.eval()
          criterion.classifier.update_centroids(embedding, criterion.Y)
        train_loss += loss.item()
        confBatch, predicted = criterion.conf(embedding).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        conf+=confBatch.sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (100*train_loss/batch_idx, 100.*correct/total, correct, total, 100*conf/total))
    return (100.*correct/total, 100*conf/total)
