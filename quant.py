import torch
import torch.nn as nn
import torch.nn.functional as F


class STERoundClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min, max):
        return x.round().clamp(min, max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BinaryLinear(nn.Module):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float16).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float16).data)
        else:
            self.bias = None

    def forward(self, x):
        w = STEBinary().apply(self.weight)
        return F.linear(x, w, self.bias)
