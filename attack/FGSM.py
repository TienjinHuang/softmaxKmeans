import torch

# FGSM attack code
def fgsm_perturbation(image, epsilon, data_grad):
  r"""
  FGSM Attack

  The ``fgsm_attack`` function takes three
  inputs, *image* is the original clean image $x$, *epsilon* is
  the pixel-wise perturbation amount $\epsilon$, and *data_grad*
  is the gradient of the loss w.r.t the input image
  ($\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y)$). The function
  then creates perturbed image as

  \begin{align}perturbed\_image = image + epsilon*sign(data\_grad) = x + \epsilon * sign(\nabla_{x} J(\mathbf{\theta}, \mathbf{x}, y))\end{align}

  Finally, in order to maintain the original range of the data, the
  perturbed image is clipped to range $[0,1]$.
  """
  # Collect the element-wise sign of the data gradient
  sign_data_grad = data_grad.sign()
  # Create the perturbed image by adjusting each pixel of the input image
  perturbed_image = image + epsilon*sign_data_grad
  # Adding clipping to maintain [0,1] range
  perturbed_image = torch.clamp(perturbed_image, -0.5, 0.5)
  # Return the perturbed image
  return perturbed_image

def attack( net, device, testloader, epsilon, criterion ):
  r"""
  Performing attacks on a dataset

  The test function performs FGSM attacks on the MNIST test set and reports a final accuracy. The more attacks are successful, the lower the accuracy gets. The inputs are the *model* under attack, the *device*, the loader of the MNIST test set, parameter *epsilon*, determining the step size of the FGSM attack and the confidence function *f_conf*. An attack is determined successful if the confidence of assigning the predicted class is larger than 0.1, since we have ten classes.  

  The function also saves and returns some
  successful adversarial examples to be visualized later.
  """
  # Accuracy counter
  correct, conf, attack_successes = 0, 0, 0
  adv_x = []
  net.eval()
  # Loop over all examples in test set
  for inputs, targets in testloader:
      # Send the data and label to the device
      inputs, targets = inputs.to(device), targets.to(device)
      inputs.requires_grad = True

      # Calculate the loss
      embedding = net.embed(inputs)
      loss = criterion(embedding,targets)

      net.zero_grad()
      loss.backward()
      data_grad = inputs.grad.data

      # Call FGSM Attack
      perturbed_data = fgsm_perturbation(inputs, epsilon, data_grad)
      
      # Re-classify the perturbed image
      embedding_perturbed = net.embed(perturbed_data)
      

      # Check for success
      conf_pert, pred_pert = criterion.conf(embedding).max(1)
      #final_pred = criterion.conf(embedding_perturbed).max(1, keepdim=True)[1].flatten() # get the index of the max log-probability
      #conf_pert = np.max(net.module.conf(inputs).detach().cpu().numpy())
      pred = criterion.conf(embedding).max(1)[1] 
      pred_is_correct = torch.eq(pred,targets)
      pred_pert_is_correct = torch.eq(pred_pert,targets)
      correct+=torch.sum(pred_pert_is_correct).item()
      attack_success = (pred_pert_is_correct== False) &  pred_is_correct
      attack_successes += torch.sum(attack_success).item()
      conf+=torch.sum(conf_pert[attack_success]).item()
      correct+= torch.sum(torch.eq(pred_pert,targets)).item()
      adv_x.append(perturbed_data[attack_success,:,:,:])

  # Calculate final accuracy for this epsilon
  attack_acc = correct/len(testloader.dataset)
  attack_conf = conf/max(attack_successes,1)
  print("Epsilon: {%.3f}\tTest Accuracy = {} / {} = {%.3f}\t conf attacks={%.3f}".format(epsilon, correct, len(testset), attack_acc, attack_conf))
  if len(adv_x)>0:
    adv_x= torch.cat(adv_x, dim=0)
  else:
    adv_x=None

  # Return the accuracy and adversarial examples
  return attack_acc, attack_conf, adv_x
