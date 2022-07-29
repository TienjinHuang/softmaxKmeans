
# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
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

def attack( model, device, test_loader, epsilon, loss_train ):
  r"""
  Performing attacks on a dataset

  The test function performs FGSM attacks on the MNIST test set and reports a final accuracy. The more attacks are successful, the lower the accuracy gets. The inputs are the *model* under attack, the *device*, the loader of the MNIST test set, parameter *epsilon*, determining the step size of the FGSM attack and the confidence function *f_conf*. An attack is determined successful if the confidence of assigning the predicted class is larger than 0.1, since we have ten classes.  

  The function also saves and returns some
  successful adversarial examples to be visualized later.
  """
  # Accuracy counter
  correct = 0
  adv_x, adv_D, adv_y, adv_yhat = [],[],[],[]
  conf = 0
  model.eval()
  # Loop over all examples in test set
  for data, target in test_loader:
      # Send the data and label to the device
      data, target = data.to(device), target.to(device)
      data.requires_grad = True

      # Forward pass the data through the model
      output = model(data)
      init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-confidence

      # Calculate the loss
      loss = loss_train(output, target)

      model.zero_grad()
      loss.backward()
      data_grad = data.grad.data

      # Call FGSM Attack
      perturbed_data = fgsm_attack(data, epsilon, data_grad)

      # Re-classify the perturbed image
      output = model(perturbed_data)

      # Check for success
      final_pred = output.max(1, keepdim=True)[1].flatten() # get the index of the max log-probability
      conf_pert = np.max(model.module.conf(data).detach().cpu().numpy())
      correct+= torch.sum(torch.eq(final_pred,target))
      adv_x.append(perturbed_data)

  # Calculate final accuracy for this epsilon
  final_acc = correct.item()/float(len(testset))
  print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(testset), final_acc))
  if len(adv_x)>0:
    adv_x= torch.cat(adv_x, dim=0)
  else:
    adv_x=None

  # Return the accuracy and an adversarial example
  return final_acc, adv_x, conf/(float(len(testset))-correct)
