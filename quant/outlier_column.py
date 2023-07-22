import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from .quantizer import (
    STEBinary,
    IrNetBinary,
    FdaBinary,
    BinaryInterface,
    XnorBinaryLinear,
)


class OutliersQLinearColumn(nn.Module, BinaryInterface):
    def __init__(
        self,
        weight,
        bias,
        outlier_percent=0.05,
        metric="L1",
        dense_class=XnorBinaryLinear,
    ) -> None:
        """
        weight is [OC,IC], some input channels (columns) are outliers
        """
        super().__init__()
        self.dense_quantizer = dense_class(weight, bias)
        self.printed = False
        self.outlier_percent = outlier_percent
        self.n_outlier_columns = int(outlier_percent * weight.shape[1])
        self.outlier_calibrated = False
        self.register_buffer(
            "outlier_columns_index",
            torch.zeros(self.n_outlier_columns, dtype=torch.int16),
        )
        self.outlier_weight = nn.Parameter(
            torch.zeros(
                [
                    weight.size(1),
                    self.n_outlier_columns,
                ]
            )
        )

        self.metric = metric

    def binarize_except_outliers(self):
        if not self.outlier_calibrated:
            if not (self.outlier_columns_index == 0).all():
                print(f"outlier_columns_index is calibrated, skip")
            else:
                print(
                    f"calibrating outlier columns, outlier_percent={self.outlier_percent}, metric={self.metric}"
                )
                w = self.dense_quantizer.weight
                if self.metric == "L1":
                    column_norms = torch.norm(w, p=1, dim=0).float()
                else:
                    raise NotImplementedError
                index = torch.argsort(column_norms)
                outlier_index_low = index[: self.n_outlier_columns // 2]
                outlier_index_high = index[-self.n_outlier_columns // 2 :]
                self.outlier_columns_index = torch.cat(
                    [outlier_index_low, outlier_index_high]
                )
                self.outlier_weight.data = w[:, self.outlier_columns_index]
            self.outlier_calibrated = True

        w = self.dense_quantizer.quant_weight()
        w[:, self.outlier_columns_index] = self.outlier_weight
        return w

    def forward(self, x):
        w=checkpoint(self.binarize_except_outliers, )
        # w = self.binarize_except_outliers()
        output = F.linear(x, w, self.dense_quantizer.bias)
        return output

    def get_save_weight_dict(self):
        return {
            "outlier_weight": self.outlier_weight,
            "outlier_columns_index": self.outlier_columns_index,
        }


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
        w=checkpoint(self.binarize_except_outliers, x)
        # w = self.binarize_except_outliers(x)
        output = F.linear(x, w, self.bias)
        return output
