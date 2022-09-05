import torch
import torch.nn.functional as F
import time
import torch.distributed as dist
def reduce_tensor(tensor,world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= float(world_size)
    return rt
def reduceSum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt
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
class Optimizer:
  def __init__(self, optimizer, trainloader, device,world_size,local_rank, update_centroids=False):
    self.optimizer = optimizer
    self.trainloader = trainloader
    self.n = len(trainloader.dataset)/world_size
    self.update_centroids = update_centroids
    self.device=device
    self.best_acc=0
    self.world_size=world_size
    self.local_rank=local_rank
    
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
    start_time=time.time()
    net.train()
    n=0
    for batch_idx, (inputs, targets) in enumerate(self.trainloader):
      n+=1
      inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
      
      inputs.requires_grad_(True)
      embedding,pred = net(inputs)
      loss = criterion(pred,targets,net.module.classifier.gamma2)
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
        net.module.classifier.prox()
        if self.update_centroids:
          net.eval()
          net.module.classifier.update_centroids(embedding, criterion.Y)
        train_loss += reduce_tensor(loss,self.world_size).item()
        confBatch, predicted = net.module.classifier.conf(embedding).max(1)
        correct += reduce_tensor(predicted.eq(targets).sum().float(),self.world_size).item()
        conf+=reduce_tensor(confBatch.sum(),self.world_size).item()
      torch.cuda.synchronize()

    execution_time = (time.time() - start_time)
    if dist.get_rank()==0:
      print('Train | GPU id:%.1f | Loss: %.3f (%d) | Acc: %.3f%% (%d/%d) | Conf %.2f | time (s): %.2f'% (self.local_rank,train_loss/n,n, 100.*correct/self.n, correct, self.n, 100*conf/self.n, execution_time))
    return (100.*correct/self.n, 100*conf/self.n)
  
  def test_acc(self, net, criterion, data_loader, min_conf=0):
    net.eval()
    test_loss, correct, conf, total = 0,0,0,0
    n=0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            n+=1
            inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
            outputs,pred = net(inputs)
            loss = criterion(pred, targets,net.module.classifier.gamma2)

            test_loss += reduce_tensor(loss,self.world_size).item()
            confBatch, predicted = net.module.classifier.conf(outputs).max(1)
            idx = (confBatch.detach()>min_conf)

            correct += reduce_tensor(predicted[idx].eq(targets[idx]).sum().float(),self.world_size).item()
            conf+=reduce_tensor(confBatch[idx].sum(),self.world_size).item()
            total+= idx.sum()
            torch.cuda.synchronize()
    if dist.get_rank()==0:
      print('Test | GPU id:%.1f | Loss: %.3f (%d) | Acc: %.3f%% (%d/%d) | Conf %.2f'% (self.local_rank,test_loss/n,n, 100.*correct/total, correct, total, 100*conf/total))
    return (100.*correct/total, 100*conf/total)
  
  def test_grad_penalty(self, net, criterion, data_loader, gp_embed):
    net.eval()
    gp = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
        inputs.requires_grad_(True)
        embedding,Y_pred = net(inputs)
        #criterion(embedding, targets)

        if not gp_embed:
          gp += self.gradient_penalty(inputs, Y_pred).item()
        if gp_embed:
          gp += self.gradient_penalty(inputs, embedding).item()
        inputs.requires_grad_(False)
    print('GPUD id:%.1f | Gradient Penalty: %.3f'% (self.local_rank,gp/max(len(data_loader),1)))
    return gp
  
  def optimize_centroids(self, net):
    net.eval()
    d,c = net.module.classifier.in_features,net.module.classifier.out_features
    Z=torch.zeros(d,c).to(self.device)
    y_sum = torch.zeros(c).to(self.device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
            D,_ = net(inputs)
            
            # batch_D=[torch.zeros_like(D) for _ in range(self.world_size)]
            # batch_targets=[torch.zeros_like(targets) for _ in range(self.world_size)]
            # #print(D)
            # dist.all_gather(batch_D,D)
            # batch_D=torch.cat(batch_D,dim=0)
            # #print(batch_D)

            # dist.all_gather(batch_targets,targets)
            # batch_targets=torch.cat(batch_targets,dim=0)
            # print("batch_D",batch_D.shape,flush=True)

            # Y = F.one_hot(batch_targets, c).float().to(self.device)
            # temp_z=batch_D.t().mm(Y)
            # Z += temp_z
            # y_sum += torch.sum(Y,0)
            #y_sum=y_sum
            Y = F.one_hot(targets, c).float().to(self.device)
            Z += D.t().mm(Y)
            y_sum += torch.sum(Y,0)
    Z = Z/y_sum
    Z=reduce_tensor(Z,self.world_size)
    net.module.classifier.weight.data = Z.t()
