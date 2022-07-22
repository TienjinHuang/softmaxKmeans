import torch

def gradient_penalty(inputs, P1):
  gradients = torch.autograd.grad(
            outputs=P1,
            inputs=inputs,
            grad_outputs=torch.ones_like(P1),
            create_graph=True,
        )[0]
  gradients = gradients.flatten(start_dim=1)
  # L2 norm
  grad_norm = gradients.norm(2, dim=1)
  # Two sided penalty
  gradient_penalty = ((grad_norm - 1) ** 2).mean()
  return gradient_penalty
