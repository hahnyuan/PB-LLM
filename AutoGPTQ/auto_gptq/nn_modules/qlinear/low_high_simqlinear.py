import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F

class QuantLinear(nn.Module):
    QUANT_TYPE = "low_high_simulated"

    def __init__(self, bits, group_size, infeatures, outfeatures,  bias=None, use_cuda_fp16=True, trainable=False) -> None:
        super().__init__()
        assert trainable == False
        
        self.simW = nn.Parameter(torch.zeros((outfeatures,infeatures), dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None
    
    def pack(self, linear, scales, zeros, g_idx=None,quantizer=None):
        W = linear.weight.data
        assert quantizer is not None
        simW=quantizer.quantize(W.float())
        assert self.simW.shape==W.shape
        self.simW.data=simW
        if self.bias is not None:
            self.bias.data=linear.bias.data
    
    def forward(self, x: torch.Tensor):
        out = F.linear(x,self.simW,self.bias)
        return out