import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LowQuantizer(nn.Module):

    def __init__(self, weight, method="xnor",groupsize=-1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        if "bit" in method:
            self.register_buffer('maxq', torch.tensor(1))
            self.register_buffer('zero', torch.zeros(self.n_groups,oc,1))

        self.register_buffer('scale', torch.zeros(self.n_groups,oc,1))
        self.register_buffer('mean', torch.zeros(self.n_groups,oc,1))
        self.method=method


    def calibrate(self, w, mask=None, groupi=0):
        if self.method=="xnor":
            # TODO: seems to have problem
            w_mean = w.mean(-1).view(-1, 1)  # oc, ic(blocksize)
            self.mean[groupi]=w_mean
            w = w - w_mean  # oc, ic(blocksize)
            # non_zero_nums = (w != 0).float().sum(-1,keepdim=True)
            # scale = w.abs().sum(-1,keepdim=True)/(non_zero_nums+1e-5)
            scale=w.abs().mean(-1,keepdim=True)
            # TODO: search mean and scale
        elif self.method=="sign":
            # w_relu=F.relu(w)
            # scale=w_relu.sum()/((w>0).float().sum()+1e-5)
            scale=F.relu(w).mean(-1,keepdim=True)
            # scale=w.abs().mean(-1,keepdim=True)
            # scale=w.mean(-1,keepdim=True)
        elif self.method=="rtn":
            scale=w.abs().mean(-1,keepdim=True)+1e-5
        elif self.method in ["no","prune"]:
            return
        elif self.method in ['2bit','4bit']:
            w=w
            dev = w.device
            if self.method=="2bit":
                self.maxq.fill_(3)
            elif self.method=="4bit":
                self.maxq.fill_(7)
            self.maxq= self.maxq.to(dev)
            self.scale= self.scale.to(dev)
            self.zero= self.zero.to(dev)
            w = w.flatten(1)
            tmp = torch.zeros(w.shape[0], device=dev)
            xmin = torch.minimum(w.min(1)[0], tmp)
            xmax = torch.maximum(w.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1

            scale = (xmax - xmin) / self.maxq
            scale=scale.reshape(-1,1)
            self.zero[groupi] = torch.round(-xmin / scale[groupi]).reshape(-1,1)
        else:
            raise NotImplementedError(f"method {self.method} not implemented")
        self.scale[groupi]=scale
        self.scale.to(w.device)

    def quantize(self, w,groupi=0):
        if w.device!=self.scale.device:
            self.scale=self.scale.to(w.device)
            self.mean=self.mean.to(w.device)
        if self.method=="xnor":
            # return torch.zeros_like(w)
            w_mean = self.mean[groupi]
            w = w - w_mean  # oc, ic
            w = w.sign()
            # TODO remove to
            w = w * self.scale[groupi]
            w+=w_mean
            
        elif self.method=="sign":
            w=(w>0).float()
            w*=self.scale[groupi]
        elif self.method=="rtn":
            w=F.relu(w)
            w_int=(w/self.scale[groupi]).round().clamp(0,1)
            w=w_int*self.scale[groupi]
        elif self.method in ['2bit','4bit']:
            q = torch.clamp(torch.round(w / self.scale[groupi]) + self.zero[groupi], 0, self.maxq)
            w= self.scale[groupi] * (q - self.zero[groupi])
        elif self.method=="prune":
            return torch.zeros_like(w)
        return w
