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


class OutliersQLinearWindowFix(nn.Module, BinaryInterface):
    def __init__(
        self,
        weight,
        bias,
        outlier_fraction=0.125,
        outlier_metric="L1",
        dense_class=XnorBinaryLinear,
    ) -> None:
        """
        outlier_fraction: fraction of outlier columns, default 0.125(1:8)
        weight is [OC,IC]
        """
        super().__init__()
        self.dense_quantizer = dense_class(weight, bias)
        self.printed = False
        self.outlier_fraction = outlier_fraction
        if outlier_fraction == 0.5:
            self.window_size = 2
        elif outlier_fraction == 0.125:
            self.window_size = 8
        else:
            raise NotImplementedError
        self.outlier_calibrated = False
        assert weight.numel() % self.window_size == 0
        self.n_outliers = weight.numel() // self.window_size
        self.register_buffer(
            "outlier_index",
            torch.zeros([self.n_outliers, 1], dtype=torch.int64).to(weight.device),
        )
        # self.outlier_weight = nn.Parameter(
        #     torch.zeros(
        #         [
        #             weight.size(1),
        #             self.n_outlier_columns,
        #         ]
        #     )
        # )

        self.outlier_metric = outlier_metric

    def outlier_calibration(self, x=None):
        with torch.no_grad():
            if not (self.outlier_index == 0).all():
                print(f"outlier_columns_index is calibrated, skip")
            else:
                print(
                    f"calibrating outlier columns, outlier_fraction={self.outlier_fraction}, metric={self.outlier_metric}"
                )
                w = self.dense_quantizer.weight
                if self.outlier_metric == "L1":
                    w = w.view(-1, self.window_size)
                    sensitivity = torch.abs(w)
                    # sensitivity = torch.norm(w, p=1, dim=0).float()
                else:
                    raise NotImplementedError
                index = torch.argmax(sensitivity, 1)
                self.outlier_index = index.view(-1, 1)
            self.outlier_calibrated = True

    def binarize_except_outliers(self):
        w = self.dense_quantizer.quant_weight()
        # w.view(-1, self.window_size)
        high_bit = self.dense_quantizer.weight.view(-1, self.window_size).gather(
            1, self.outlier_index
        )
        w.view(-1, self.window_size).scatter_(1, self.outlier_index, high_bit)
        #  = self.dense_quantizer.weight[:, self.outlier_columns_index]
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
            # "outlier_weight": self.outlier_weight,
            "index": self.outlier_index,
        }

    def __repr__(self):
        return f"OutliersQLinearWindowFix({self.n_outliers}, outlier_fraction={self.outlier_fraction}, outlier_metric={self.outlier_metric})"
