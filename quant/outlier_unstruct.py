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


class OutliersQLinearUnstruct(nn.Module, BinaryInterface):
    def __init__(
        self,
        weight,
        bias,
        outlier_fraction=0.05,
        outlier_metric="L1",
        dense_class=XnorBinaryLinear,
    ) -> None:
        """
        weight is [OC,IC], some input channels (columns) are outliers
        """
        super().__init__()
        self.dense_quantizer = dense_class(weight, bias)
        self.printed = False
        self.outlier_fraction = outlier_fraction
        self.n_outliers = int(outlier_fraction * weight.numel())
        self.register_buffer("outlier_calibrated", torch.tensor(False))

        self.register_buffer(
            "outlier_mask",
            torch.zeros(weight.shape, dtype=torch.bool),
        )
        self.outlier_metric = outlier_metric
        self.H_diag = None

    def add_batch(self, x):
        if self.outlier_metric == "hessain":
            if self.H_diag is None:
                self.H_diag = torch.zeros(x.shape[1], device=x.device)
            self.H_diag += (x**2).mean([0, 1])

    def outlier_calibration(self, x=None):
        with torch.no_grad():
            print(
                f"calibrating outlier, outlier_fraction={self.outlier_fraction}, metric={self.outlier_metric}"
            )
            w = self.dense_quantizer.weight
            if self.outlier_metric == "L1":
                sensitivity = w.abs()
            elif self.outlier_metric == "hessian":
                # from OWQ: Lessons learned from activation outliers for weight quantization in large language models
                H_diag = (x * x).mean([0, 1])  # shape ic
                w_hat = self.dense_quantizer.quant_weight()
                delta_w = (w_hat - w) ** 2  # shape oc,ic
                sensitivity = H_diag.view(1, -1) * delta_w  # shape oc,ic
            else:
                raise NotImplementedError
            thresh = torch.kthvalue(
                sensitivity.view(-1), w.numel() - self.n_outliers
            ).values
            self.outlier_mask.data = sensitivity >= thresh
            # self.outlier_weight.data = w[:, self.outlier_columns_index]
            self.outlier_calibrated.data[...] = True

    def binarize_except_outliers(self):
        w = self.dense_quantizer.quant_weight(outlier_mask=self.outlier_mask)
        w = torch.where(self.outlier_mask, self.dense_quantizer.weight, w)
        return w

    def forward(self, x):
        if not self.outlier_calibrated:
            self.outlier_calibration(x)
        # w = checkpoint(self.binarize_except_outliers)
        w = self.binarize_except_outliers()
        output = F.linear(x, w, self.dense_quantizer.bias)
        return output

    def get_save_weight_dict(self):
        return {
            "outlier_mask": self.outlier_mask,
        }

    def __repr__(self):
        return f"OutliersQLinearUnstruct({self.n_outliers}, outlier_fraction={self.outlier_fraction}, outlier_metric={self.outlier_metric})"
