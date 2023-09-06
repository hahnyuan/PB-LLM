import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint


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
    def forward(
        ctx,
        input,
        k=torch.tensor([10]).float().cuda(),
        t=torch.tensor([0.1]).float().cuda(),
    ):
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
        grad_input = (
            4
            * omega
            / np.pi
            * sum(([torch.cos((2 * i + 1) * omega * inputs) for i in range(n + 1)]))
            * grad_output
        )
        grad_input[inputs.gt(1)] = 0
        grad_input[inputs.lt(-1)] = 0
        return grad_input, None


class BinaryInterface:
    def get_save_weight_dict(self):
        return {"weight": self.weight.data.half().cpu(), "bias": self.bias}


class BinaryLinear(nn.Module, BinaryInterface):
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


class IrBinaryLinear(nn.Module, BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def quant_weight(self):
        w = self.weight
        # centeralization
        w = w - w.mean(-1).view(-1, 1)
        scaling_factor = w.abs().mean(-1).view(-1, 1).detach()
        w = IrNetBinary().apply(w)
        w = w * scaling_factor
        return w

    def forward(self, x):
        w = checkpoint(self.quant_weight, use_reentrant=False)
        return F.linear(x, w, self.bias)


class FdaBinaryLinear(nn.Module, BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        self.n = torch.tensor(int(10)).cuda()
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def quant_weight(self):
        w = FdaBinary().apply(self.weight, self.n)
        return w

    def forward(self, x):
        w = checkpoint(self.quant_weight, use_reentrant=False)
        return F.linear(x, w, self.bias)


class BiRealLinear(nn.Module, BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def quant_weight(self):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = (
            binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        )
        return binary_weights

    def forward(self, input):
        x = input
        out_forward = torch.sign(input)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (
            1 - mask1.type(torch.float32)
        )
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (
            1 - mask2.type(torch.float32)
        )
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        input = out_forward.detach() - out3.detach() + out3
        w = checkpoint(self.quant_weight, use_reentrant=False)
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        output = F.linear(input, w)
        return output


class XnorBinaryLinear(nn.Module, BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super(XnorBinaryLinear, self).__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None

    def quant_weight(self, outlier_mask=None):
        w = self.weight
        w = w - w.mean(-1).view(-1, 1)  # oc, ic
        if outlier_mask is not None:
            w = w * (~outlier_mask)
        scaling_factor = w.abs().mean(-1).view(-1, 1).detach()
        w = STEBinary().apply(w)
        w = w * scaling_factor
        return w

    def forward(self, x):
        w = checkpoint(self.quant_weight, use_reentrant=False)
        return F.linear(x, w, self.bias)
