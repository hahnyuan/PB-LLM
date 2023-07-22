import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from .quantizer import STEBinary, IrNetBinary, FdaBinary, BinaryInterface


class BinaryXnorExceptOutliersLinearColumn(nn.Module, BinaryInterface):
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

            outlier_columns = (column_norms > lower_threshold) & (
                column_norms < upper_threshold
            )
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


class BinaryXnorExceptOutliersLinearActivationColumn(nn.Module, BinaryInterface):
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
            print(w.shape, a.shape)
            a = F.linear(x, w, self.bias)
            column_norms = torch.norm(w, p=1, dim=0)
            lower_threshold, upper_threshold = torch.quantile(
                column_norms, [0.05, 0.95]
            )

            outlier_columns = (column_norms > lower_threshold) & (
                column_norms < upper_threshold
            )
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
