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
class BinaryInterface:
    pass
class BinaryLinear(nn.Module,BinaryInterface):
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
class BinaryExceptOutliersLinear(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None
        self.printed = False
    
    def binarize_except_outliers(self):
        w = self.weight
        w_flat = w.view(-1)
        # lower_threshold, upper_threshold = torch.quantile(w_flat, torch.tensor([0.01, 0.99]).to(w.device))
        mean = torch.mean(w_flat).to(w.device)
        std = torch.std(w_flat).to(w.device)
        lower_threshold = mean - 1*std
        upper_threshold = mean + 1*std
        outliers = (w < lower_threshold) | (w > upper_threshold)
        # if self.printed is not True:
        #     print(outliers.sum()/outliers.numel())
        #     self.printed = True
        w_bin = w.clone()
        # scaling_factor = w[~outliers].abs().mean(-1).view(-1, 1).detach()
        w = STEBinary().apply(w)
        w_bin[~outliers] = STEBinary().apply(w[~outliers])
        # w_bin[~outliers] = w[~outliers] * scaling_factor
        
        return w_bin
    def forward(self, x):
        # w = STEBinary().apply(self.weight)
        w = self.binarize_except_outliers()
        return F.linear(x, w, self.bias)

class BinaryXnorExceptOutliersLinear(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
            super().__init__()
            self.weight = nn.Parameter(weight.to(torch.float32).data)
            if bias is not None:
                self.bias = nn.Parameter(bias.to(torch.float32).data)
            else:
                self.bias = None
            self.printed = False
            self.outliers = None

    def binarize_except_outliers(self):
        if self.outliers is None:
            print(self.outliers)
            w = self.weight
            w_flat = w.view(-1)
            # lower_threshold, upper_threshold = torch.quantile(w_flat, torch.tensor([0.01, 0.99]).to(w.device))

            mean = torch.mean(w_flat).to(w.device)
            std = torch.std(w_flat).to(w.device)
            lower_threshold = mean - 1.96*std
            upper_threshold = mean + 1.96*std #1.96 : 95%, 

            outliers = (w < lower_threshold) | (w > upper_threshold)
            self.outliers = outliers
        # if self.printed is not True:
        #     print(outliers.sum()/outliers.numel())
        #     self.printed = True

        w = self.weight
        w_bin = w.clone()
        scaling_factor = w[~self.outliers].abs().mean(-1).view(-1, 1).detach()

        # w = STEBinary().apply(w)
        w_bin[~self.outliers] = STEBinary().apply(w[~self.outliers])
        w_bin[~self.outliers] = w[~self.outliers] * scaling_factor

        # with torch.no_grad():
        #     w_bin.grad[outliers] = 0.

        return w_bin, self.outliers

    def forward(self, x):
        # w = STEBinary().apply(self.weight)
        w, outliers = self.binarize_except_outliers()
        output = F.linear(x, w, self.bias)
        return output
    


class BinaryXnorExceptOutliersLinearColumn(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
            super().__init__()
            self.weight = nn.Parameter(weight.to(torch.float32).data)
            if bias is not None:
                self.bias = nn.Parameter(bias.to(torch.float32).data)
            else:
                self.bias = None
            self.printed = False
            self.outlier_columns = None

    def binarize_except_outliers(self):
        if self.outlier_columns is None:
            print(self.outlier_columns)
            w = self.weight
            column_norms = torch.norm(w, p=1, dim=0).float()
            lower_threshold = torch.quantile(column_norms, 0.05)
            upper_threshold = torch.quantile(column_norms, 0.95)

            outlier_columns = (column_norms > lower_threshold) & (column_norms < upper_threshold)
            self.outlier_columns = outlier_columns 

        w = self.weight
        w_bin = w.clone()
        scaling_factor = w[:, ~self.outlier_columns].abs().mean(-1).view(-1, 1).detach()

        w_bin[:, ~self.outlier_columns] = STEBinary().apply(w[:, ~self.outlier_columns])
        w_bin[:, ~self.outlier_columns] = w[:, ~self.outlier_columns] * scaling_factor

        return w_bin
    
    def forward(self, x):
        w = self.binarize_except_outliers()
        output = F.linear(x, w, self.bias)
        return output
    
class BinaryXnorExceptOutliersLinearActivationColumn(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
            super().__init__()
            self.weight = nn.Parameter(weight.to(torch.float32).data)
            if bias is not None:
                self.bias = nn.Parameter(bias.to(torch.float32).data)
            else:
                self.bias = None
            self.printed = False
            self.outlier_columns = None

    def binarize_except_outliers(self, x):
        if self.outlier_columns is None:
            print(self.outlier_columns)
            w = self.weight
            print(w.shape,a.shape)
            a = F.linear(x, w, self.bias)
            column_norms = torch.norm(w, p=1, dim=0)
            lower_threshold, upper_threshold = torch.quantile(column_norms, [0.05, 0.95])

            outlier_columns = (column_norms > lower_threshold) & (column_norms < upper_threshold)
            self.outlier_columns = outlier_columns 

        w = self.weight
        w_bin = w.clone()
        scaling_factor = w[:, ~self.outlier_columns].abs().mean(-1).view(-1, 1).detach()

        w_bin[:, ~self.outlier_columns] = STEBinary().apply(w[:, ~self.outlier_columns])
        w_bin[:, ~self.outlier_columns] = w[:, ~self.outlier_columns] * scaling_factor

        return w_bin
    
    def forward(self, x):
        w = self.binarize_except_outliers(x)
        output = F.linear(x, w, self.bias)
        return output

class XnorBinaryLinear(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super(XnorBinaryLinear,self).__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None
    def quant_weight(self):
        w = self.weight
        # #centeralization
        # w = w - w.mean(-1).view(-1, 1)
        w = w - w.mean(-1).view(-1, 1)
        scaling_factor = w.abs().mean(-1).view(-1, 1).detach()
        w = STEBinary().apply(w)
        w = w * scaling_factor
        return w
    def forward(self, x):
        w=checkpoint(self.quant_weight, use_reentrant=False)
        return F.linear(x, w, self.bias)
class IrBinaryLinear(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None
    def quant_weight(self):
        w = self.weight
        #centeralization
        w = w - w.mean(-1).view(-1, 1)
        scaling_factor = w.abs().mean(-1).view(-1, 1).detach()
        w = IrNetBinary().apply(w)
        w = w * scaling_factor
        return w
    def forward(self, x):
        w=checkpoint(self.quant_weight, use_reentrant=False)
        return F.linear(x, w, self.bias)
class FdaBinaryLinear(nn.Module,BinaryInterface):
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
        w=checkpoint(self.quant_weight, use_reentrant=False)
        return F.linear(x, w, self.bias)
class BiRealLinear(nn.Module,BinaryInterface):
    def __init__(self, weight, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None
    def quant_weight(self):
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        return binary_weights
    def forward(self, input):
        x = input
        out_forward = torch.sign(input)
        out_e_total = 0
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        input = out_forward.detach() - out3.detach() + out3
        w=checkpoint(self.quant_weight, use_reentrant=False)
        # y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
        output = F.linear(input, w)
        return output
