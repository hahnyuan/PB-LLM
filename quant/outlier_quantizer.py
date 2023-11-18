import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint
from .quantizer import STEBinary, IrNetBinary, FdaBinary, BinaryInterface


def weight_quant_8bit(w):
    # per channel assymetric quantization
    w_min = w.amin(dim=1, keepdim=True)
    w_max = w.amax(dim=1, keepdim=True)
    w_range = w_max - w_min
    # align zero
    w_zero_point = torch.round(w_min - 128 * w_range / 255)
    # quantize
    w_q = torch.round((w - w_zero_point) * 255 / w_range)
    # clip
    w_q = torch.clamp(w_q, 0, 255)
    # dequantize
    w_q = w_q * w_range / 255 + w_zero_point
    return w_q


class BinaryXnorExceptOutliersLinear(nn.Module, BinaryInterface):
    def __init__(self, weight, bias, outlier_fraction, outlier_scale=1) -> None:
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

        self.global_name = None

    def gen_outlier_mask(self):
        with torch.no_grad():
            w = self.weight
            w_flat = w.view(-1)
            # lower_threshold, upper_threshold = torch.quantile(
            #     w_flat,
            #     torch.tensor(
            #         [self.outlier_fraction / 2, 1 - self.outlier_fraction / 2]
            #     ).to(w.device),
            # )
            # quantile() input tensor is too large
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
            # lower_threshold, upper_threshold = torch.kthvalue(
            #     w_flat,
            #     int(w_flat.numel() * self.outlier_fraction / 2),
            # )[0], torch.kthvalue(
            #     w_flat,
            #     int(w_flat.numel() * (1 - self.outlier_fraction / 2)),
            # )[0]

            # mean = torch.mean(w_flat).to(w.device)
            # std = torch.std(w_flat).to(w.device)
            # lower_threshold = mean - 1.6 * std  # 1.6 : 90%, 0.67 : 50%, 1.0, 70%
            # upper_threshold = mean + 1.6 * std  # 1.95 : 95%, 2.3 : 98%,

            outliers = (w < lower_threshold) | (w > upper_threshold)
            print(
                f"Generat outlier_mask, outlier_fraction: {outliers.sum()}/{outliers.numel()}({outliers.sum()/outliers.numel()})"
            )
            self.outlier_mask = outliers.detach()

            # import matplotlib.pyplot as plt
            # import numpy as np

            # # Assuming you have run your provided code and obtained the outliers tensor
            # # Convert the tensor to numpy for visualization
            # outliers_np = outliers.cpu().numpy()
            # outliers_np = outliers_np[0:255,0:255]

            # plt.figure(figsize=(10, 10))
            # plt.imshow(outliers_np, cmap='hot', interpolation='nearest')
            # plt.colorbar()
            # plt.title('Outliers Matrix Visualization')
            # plt.savefig('./111.pdf')
            # # plt.show()

            self.binary_scale = (
                w[~self.outlier_mask].abs().mean(-1).view(-1, 1).detach()
            )
            self.weight.data = weight_quant_8bit(w)

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


class BinaryXnorExceptOutliersLinearHessian(BinaryXnorExceptOutliersLinear):
    def gen_outlier_mask(self):
        with torch.no_grad():
            w = self.weight
            low_frac = 1 - self.outlier_fraction
            if "lm_head" in self.global_name:
                return super().gen_outlier_mask()
            mask = torch.load(
                f"gptq_pb/outputs/mask/mask_{low_frac}_{self.global_name.replace('/','_')}.pkl"
            )
            self.outlier_mask = ~mask
            self.weight.data = weight_quant_8bit(w)
