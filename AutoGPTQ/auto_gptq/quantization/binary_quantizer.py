from logging import getLogger

import torch
import torch.nn as nn


logger = getLogger(__name__)



class BinaryQuantizer(nn.Module):

    def __init__(self, method="xnor"):
        super().__init__()
        self.register_buffer('scale', torch.zeros(1))
        self.method=method


    def find_params(self, w, weight=False):
        w = w - w.mean(-1).view(-1, 1)  # oc, ic
        scale = w.abs().mean(-1).view(-1, 1).detach()
        w = w.sign()
        w = w * scale
        self.scale.data=scale
        return 

    def quantize(self, w):
        w = w - w.mean(-1).view(-1, 1)  # oc, ic
        w = w.sign()
        w = w * self.scale
        return w
