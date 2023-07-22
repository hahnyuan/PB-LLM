import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from .quantizer import STEBinary, IrNetBinary, FdaBinary, BinaryInterface


class BinaryExceptOutliersLinear(nn.Module, BinaryInterface):
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
        lower_threshold = mean - 1 * std
        upper_threshold = mean + 1 * std
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


class BinaryXnorExceptOutliersLinear(nn.Module, BinaryInterface):
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
            lower_threshold = mean - 1.96 * std
            upper_threshold = mean + 1.96 * std  # 1.96 : 95%,

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