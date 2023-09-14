import torch
import torch.nn as nn
import math

class LowQuantizer(nn.Module):

    def __init__(self, weight, method="xnor",groupsize=128):
        super().__init__()
        oc,ic=weight.shape
        self.n_groups=math.ceil(ic/groupsize)
        self.register_buffer('scale', torch.zeros(self.n_groups,oc,1))

        self.method=method


    def calibrate(self, w, groupi):
        w = w - w.mean(-1).view(-1, 1)  # oc, ic(blocksize)
        # non_zero_nums = (w != 0).float().sum(-1,keepdim=True)
        # scale = w.abs().sum(-1,keepdim=True)/(non_zero_nums+1e-5)
        scale=w.abs().mean(-1,keepdim=True)
        self.scale[groupi]=scale
        self.scale.to(w.device)

    def quantize(self, w,groupi):
        w = w - w.mean(-1).view(-1, 1)  # oc, ic
        w = w.sign()
        # TODO remove to
        self.scale=self.scale.to(w.device)
        w = w * self.scale[groupi]
        return w
