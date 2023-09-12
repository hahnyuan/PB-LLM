import torch.nn as nn
import torch
from .quantizer import Quantizer

class LowHighQuantizer(nn.Module):
    def __init__(self, high_percent, low_bit,high_bit,perchannel,sym,low_mse,high_mse) -> None:
        super().__init__()
        self.high_percent=high_percent
        self.low_quantizer=Quantizer()
        self.low_quantizer.configure(low_bit,perchannel,sym,low_mse)
        self.high_quantizer=Quantizer()
        self.high_quantizer.configure(high_bit,perchannel,sym,high_mse)

    def gen_low_mask(self,x):
        high_num=int(x.numel()*self.high_percent)
        x_reshape=x.reshape(-1)
        low_thresh=torch.kthvalue(x_reshape,high_num//2)[0]
        high_thresh=torch.kthvalue(x_reshape,x.numel()-high_num//2)[0]
        mask=(x>low_thresh)&(x<high_thresh)
        return mask

    def find_params(self,x,weight=False):
        mask=self.gen_low_mask(x)
        low_x=x*mask
        high_x=x*~mask
        self.low_quantizer.find_params(low_x,weight)
        self.high_quantizer.find_params(high_x,weight)

    def quantize(self,x):
        if not self.ready():
            return x
        mask=self.gen_low_mask(x)
        low_x=x*mask
        high_x=x*~mask
        x1=self.low_quantizer.quantize(low_x)
        x2=self.high_quantizer.quantize(high_x)
        return x1+x2

    def enbable(self):
        return self.high_quantizer.enable()
    
    def ready(self):
        return self.high_quantizer.ready()
    
    @property
    def scale(self):
        scale=torch.cat([self.high_quantizer.scale.unsqueeze(-1),self.low_quantizer.scale.unsqueeze(-1)],-1)
        return scale

    @property
    def zero(self):
        zero=torch.cat([self.high_quantizer.zero.unsqueeze(-1),self.low_quantizer.zero.unsqueeze(-1)],-1)
        return zero