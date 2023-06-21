import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class IrNetBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k=torch.tensor([10]).float().cuda(), t=torch.tensor([0.1]).float().cuda()):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None

class FdaBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, n):
        ctx.save_for_backward(inputs, n)
        out = torch.sign(inputs)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs, n = ctx.saved_tensors
        omega = 0.1
        grad_input = 4 * omega / np.pi * sum(([torch.cos((2 * i + 1) * omega * inputs) for i in range(n + 1)])) * grad_output
        grad_input[inputs.gt(1)] = 0
        grad_input[inputs.lt(-1)] = 0
        return grad_input, None


class BinaryLinear(nn.Module):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def forward(self, x):
        w = STEBinary().apply(self.weight)
        return F.linear(x, w, self.bias)

class XnorBinaryLinear(nn.Module):
    def __init__(self, weight, bias) -> None:
        super(XnorBinaryLinear,self).__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight
        #centeralization
        w = w - w.mean(-1).view(-1, 1)
        scaling_factor = w.abs().mean(-1).view(-1, 1).detach()
        w = STEBinary().apply(w)
        w = w * scaling_factor
        return F.linear(x, w, self.bias)

class IrBinaryLinear(nn.Module):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    # def forward(self, x):
    #     w = IrNetBinary().apply(self.weight)
    #     return F.linear(x, w, self.bias)
    def forward(self, x):
        w = self.weight
        #centeralization
        w = w - w.mean(-1).view(-1, 1)
        scaling_factor = w.abs().mean(-1).view(-1, 1).detach()
        w = IrNetBinary().apply(w)
        w = w * scaling_factor
        return F.linear(x, w, self.bias)

class FdaBinaryLinear(nn.Module):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        self.n = torch.tensor(int(10)).cuda()
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def forward(self, x):
        w = FdaBinary().apply(self.weight, self.n)
        return F.linear(x, w, self.bias)
