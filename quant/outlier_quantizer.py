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
        lower_threshold = mean - 1.96 * std
        upper_threshold = mean + 1.96 * std
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
    def __init__(self, weight, bias, outlier_scale=1) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None
        self.printed = False
        self.outlier_mask = None
        self.outlier_scale = outlier_scale
        self.binary_scale = None

    def gen_outlier_mask(self):
        with torch.no_grad():
            w = self.weight
            w_flat = w.view(-1)
            # lower_threshold, upper_threshold = torch.quantile(w_flat, torch.tensor([0.01, 0.99]).to(w.device))

            mean = torch.mean(w_flat).to(w.device)
            std = torch.std(w_flat).to(w.device)
            lower_threshold = mean - 0.68 * std  # 1.65 : 90%
            upper_threshold = mean + 0.68 * std  # 1.96 : 95%, 2.32 : 98%, 

            outliers = (w < lower_threshold) | (w > upper_threshold)
            print(
                f"Generat outlier_mask, outlier_fraction: {outliers.sum()}/{outliers.numel()}({outliers.sum()/outliers.numel()})"
            )
            self.outlier_mask = outliers.detach()
            self.binary_scale = (
                w[~self.outlier_mask].abs().mean(-1).view(-1, 1).detach()
            )

    def binarize_except_outliers(self):
        if self.outlier_mask is None:
            self.gen_outlier_mask()

        # if self.printed is not True:
        #     print(outliers.sum()/outliers.numel())
        #     self.printed = True
        if self.training:
            self.binary_scale = (
                self.weight[~self.outlier_mask].abs().mean(-1).view(-1, 1).detach()
            )
        scaled_weight = self.weight * self.outlier_scale
        binary_weight = STEBinary().apply(self.weight) * self.binary_scale
        w_sim = torch.where(self.outlier_mask, scaled_weight, binary_weight)
        return w_sim

    def forward(self, x):
        # w = STEBinary().apply(self.weight)
        w = checkpoint(self.binarize_except_outliers)
        output = F.linear(x, w, self.bias)
        return output
