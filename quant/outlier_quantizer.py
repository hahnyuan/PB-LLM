import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from .quantizer import STEBinary, IrNetBinary, FdaBinary, BinaryInterface


def weight_quant_8bit(w,simulated=True):
    raw_type=w.dtype
    # per channel assymetric quantization
    w_range = torch.max(w, dim=-1, keepdim=True)[0] - torch.min(
        w, dim=-1, keepdim=True
    )[0]
    w_range= w_range.type(torch.float32)
    w_zero_point = torch.round(torch.min(w, dim=-1, keepdim=True)[0])
    w_q = torch.round(
        (w - w_zero_point) / w_range * 255
    ).type(torch.uint8)
    # clip
    w_q = torch.clamp(w_q, 0, 255)
    if simulated:
        # dequantize
        w_q = w_q * (w_range / 255) + w_zero_point
        w_q=w_q.to(raw_type)
    else:
        w_q= w_q.type(torch.uint8)
    return w_q



class BinaryXnorExceptOutliersLinear(nn.Module, BinaryInterface):
    def __init__(
        self, weight, bias, outlier_fraction, outlier_scale=1, train_outlier=False
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.data)
        if bias is not None:
            self.bias = nn.Parameter(bias.data)
        else:
            self.bias = None
        self.printed = False
        self.outlier_mask = None
        self.outlier_scale = outlier_scale
        self.outlier_fraction = outlier_fraction
        self.binary_scale = None
        self.train_outlier = train_outlier
        self.outlier_nbits=None

        self.global_name = None


    def gen_outlier_mask(self):
        with torch.no_grad():
            w = self.weight
            w_flat = w.view(-1)
            lower_threshold, upper_threshold = (
                torch.kthvalue(
                    w_flat,
                    int(w_flat.numel() * self.outlier_fraction / 2),
                )[0],
                torch.kthvalue(
                    w_flat,
                    int(w_flat.numel() * (1 - self.outlier_fraction / 2)),
                )[0],
            )

            outliers = (w < lower_threshold) | (w > upper_threshold)

            self.outlier_mask = outliers.detach()
            self.binary_scale = (
                w[~self.outlier_mask].abs().mean(-1).view(-1, 1).detach()
            )
            self.weight.data = weight_quant_8bit(w)
            weight_unique = torch.unique(self.weight)
            print(f"n weight_unique: {weight_unique.numel()}")
            print(
                f"Generat outlier_mask, outlier_fraction: {outliers.sum()}/{outliers.numel()}({outliers.sum()/outliers.numel()})"
            )
            self.calc_memory_consumption()

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
        if not self.train_outlier:
            scaled_weight = scaled_weight.detach()
        w_sim = torch.where(self.outlier_mask, scaled_weight, binary_weight)
        return w_sim

    def forward(self, x):
        # w = STEBinary().apply(self.weight)
        # w = checkpoint(self.binarize_except_outliers)
        w = self.binarize_except_outliers()
        output = F.linear(x, w, self.bias)
        return output

    def to_regular_linear(self):
        w = self.binarize_except_outliers()
        linear = nn.Linear(w.shape[1], w.shape[0], bias=self.bias is not None)
        linear.weight.data = w
        if self.bias is not None:
            linear.bias.data = self.bias
        return linear
    
    def calc_memory_consumption(self):
        w =weight_quant_8bit(self.weight.data,simulated=False)
        w_outlier=w*self.outlier_mask

        w2=w_outlier.to_sparse_csr()
        outlier_nbits=(w2.col_indices().numel()*8+w2.values().shape.numel()*8+ w2.crow_indices().numel()*8)/w.numel()
        self.outlier_nbits=outlier_nbits
        # (w2.ccol_indices().numel()*16+w2.values().shape.numel()*8+ w2.row_indices().numel()*16)/w.numel()


class BinaryXnorExceptOutliersLinearHessian(BinaryXnorExceptOutliersLinear):
    def gen_outlier_mask(self):
        with torch.no_grad():
            w = self.weight
            low_frac = 1 - self.outlier_fraction
            if not os.path.exists(f"gptq_pb/outputs/mask/mask_{low_frac}_{self.global_name.replace('/','_')}.pkl"):
                print(f"generating mask for {self.global_name}, please generate it first, use magnitude instead")
                return super().gen_outlier_mask()
            mask = torch.load(
                f"gptq_pb/outputs/mask/mask_{low_frac}_{self.global_name.replace('/','_')}.pkl"
            )

            self.outlier_mask = ~mask.to(w.device)
            print(
                f"load mask, outlier_fraction: {self.outlier_mask.sum()}/{self.outlier_mask.numel()}({self.outlier_mask.sum()/self.outlier_mask.numel()})"
            )
            self.weight.data = weight_quant_8bit(w)
            self.calc_memory_consumption()
