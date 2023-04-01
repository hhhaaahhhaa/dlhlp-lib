from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)


class WeightedSumLayer(nn.Module):
    def __init__(self, n_in_layers: int, specific_layer: Optional[int]=None) -> None:
        super().__init__()
        self.n_in_layers = n_in_layers
            
        # specific layer, fix weight_raw during training.
        if specific_layer is not None:
            weights = torch.ones(n_in_layers) * float('-inf')
            weights[specific_layer] = 10.0
            self.weight_raw = nn.Parameter(weights)
            self.weight_raw.requires_grad = False
        else:
            self.weight_raw = nn.Parameter(torch.randn(n_in_layers))

    def forward(self, x, dim: int):
        weight_shape = [1] * x.dim()
        weight_shape[dim] = self.n_in_layers
        weighted_sum = torch.reshape(F.softmax(self.weight_raw, dim=0), tuple(weight_shape)) * x  # B, L, d_in
        weighted_sum = weighted_sum.sum(dim=dim)
        
        return weighted_sum
