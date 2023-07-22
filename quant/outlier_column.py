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
        outlier_metric="L1",
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

        self.outlier_metric = outlier_metric

    def outlier_calibration(self, x=None):
        with torch.no_grad():
            if not (self.outlier_columns_index == 0).all():
                print(f"outlier_columns_index is calibrated, skip")
            else:
                print(
                    f"calibrating outlier columns, outlier_percent={self.outlier_percent}, metric={self.outlier_metric}"
                )
                w = self.dense_quantizer.weight
                if self.outlier_metric == "L1":
                    sensitivity = torch.norm(w, p=1, dim=0).float()
                elif self.outlier_metric == "act_L1":
                    sensitivity = torch.norm(x, p=1, dim=0).float()
                else:
                    raise NotImplementedError
                index = torch.argsort(sensitivity)
                outlier_index_low = index[: self.n_outlier_columns // 2]
                outlier_index_high = index[-self.n_outlier_columns // 2 :]
                self.outlier_columns_index = torch.cat(
                    [outlier_index_low, outlier_index_high]
                )
                self.outlier_weight.data = w[:, self.outlier_columns_index]
            self.outlier_calibrated = True

    def binarize_except_outliers(self, x=None):
        w = self.dense_quantizer.quant_weight()
        w[:, self.outlier_columns_index] = self.outlier_weight
        return w

    def forward(self, x):
        if not self.outlier_calibrated:
            self.outlier_calibration(x)
        w = checkpoint(self.binarize_except_outliers)
        # w = self.binarize_except_outliers()
        output = F.linear(x, w, self.dense_quantizer.bias)
        return output

    def get_save_weight_dict(self):
        return {
            "outlier_weight": self.outlier_weight,
            "outlier_columns_index": self.outlier_columns_index,
        }

    def __repr__(self):
        return f"OutliersQLinearColumn({self.n_outlier_columns}, outlier_percent={self.outlier_percent}, outlier_metric={self.outlier_metric})"
