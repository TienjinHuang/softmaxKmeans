import torch

def gradient_penalty(inputs, outputs):
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

def train_epoch(epoch, net, criterion, weight_gp_pred=0, weight_gp_embed=0, update_centroids = False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss, correct, total, conf, batch_idx = 0, 0, 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs.requires_grad_(True)
        outputs = net.embed(inputs)
        loss = criterion(outputs,targets)
        #----- gradient penalty
        if weight_gp_pred > 0:
            loss += weight_gp_pred * train.optimization.gradient_penalty(inputs, criterion.Y_pred)
        if weight_gp_embed>0:
            loss+= weight_gp_embed * train.optimization.gradient_penalty(inputs, outputs)
        loss.backward()
        optimizer.step()
        inputs.requires_grad_(False)
        
        with torch.no_grad():
            if update_centroids:
                net.eval()
                criterion.classifier.update_centroids(outputs, criterion.Y)
            train_loss += loss.item()
            confBatch, predicted = criterion.conf(outputs).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            conf+=confBatch.sum().item()
    print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Conf %.2f'% (100*train_loss/batch_idx, 100.*correct/total, correct, total, 100*conf/total))
