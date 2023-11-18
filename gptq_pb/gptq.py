import math
import time
import os
import torch
import torch.nn as nn
import transformers

DEBUG = False
OUTPUTMASK = 1

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class LowHighGPT:
    def __init__(
        self, layer, low_quantizer, high_quantizer, salient_metric, disable_gptq=False
    ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.low_quantizer = low_quantizer
        self.high_quantizer = high_quantizer
        self.salient_metric = salient_metric  # "magnitude" or "hessian"
        self.disable_gptq = disable_gptq

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        # breakpoint()

    def fasterquant(self, low_frac, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if not self.high_quantizer.ready():
            self.high_quantizer.calibrate(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        mask = None
        mask = torch.zeros_like(W, dtype=torch.bool)
        for groupi in range(self.low_quantizer.n_groups):
            st = groupi * self.low_quantizer.groupsize
            ed = min(st + self.low_quantizer.groupsize, self.columns)
            if self.salient_metric == "magnitude":
                saliency = torch.abs(W[:, st:ed])
                thresh = torch.sort(saliency.flatten())[0][
                    int(saliency.numel() * low_frac)
                ]
                mask[:, st:ed] = saliency <= thresh
            elif self.salient_metric == "hessian":
                tmp = (
                    W[:, st:ed] ** 2
                    / (torch.diag(H[st:ed, st:ed]).reshape((1, -1))) ** 2
                )
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * low_frac)]
                mask[:, st:ed] = tmp <= thresh
            else:
                raise NotImplementedError
            assert self.low_quantizer.groupsize % blocksize == 0
            self.low_quantizer.calibrate(
                W[:, st:ed] * mask[:, st:ed], mask[:, st:ed], groupi=groupi
            )
            # self.low_quantizer.calibrate(W[:,st:ed],mask[:,st:ed],groupi=groupi)

        if OUTPUTMASK:
            if os.path.exists("./outputs/mask") == False:
                os.mkdir("./outputs/mask")
            torch.save(
                mask,
                f"./outputs/mask/mask_{low_frac}_{self.layer.global_name.replace('/','_')}.pkl",
            )

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            n_cols = col_ed - col_st
            if self.disable_gptq:
                # RTN
                # print("RTN")
                w = W[:, col_st:col_ed]
                q_high = self.high_quantizer.quantize(w)
                groupi = col_st // self.low_quantizer.groupsize
                q_low = self.low_quantizer.quantize(w, groupi)
                q = q_high * ~mask[:, col_st:col_ed] + q_low * mask[:, col_st:col_ed]
                W[:, col_st:col_ed] = q
            else:
                # shape of W1: [oc, n_cols]
                W1 = W[:, col_st:col_ed].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

                if mask is not None:
                    mask1 = mask[:, col_st:col_ed]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    # TODO: use torch.kthvalue
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * low_frac)]
                    mask1 = tmp <= thresh

                for i in range(n_cols):
                    # shape of w: [oc, 1]
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q_high = self.high_quantizer.quantize(w.unsqueeze(1)).flatten()
                    # TODO: support groupsize>1
                    groupi = col_st // self.low_quantizer.groupsize
                    q_low = self.low_quantizer.quantize(
                        w.unsqueeze(1), groupi
                    ).flatten()
                    q = q_high * ~mask1[:, i] + q_low * mask1[:, i]

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2
                    # breakpoint()

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, col_st:col_ed] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

                if DEBUG:
                    self.layer.weight.data[:, :col_ed] = W[:, :col_ed]
                    self.layer.weight.data[:, col_ed:] = W[:, col_ed:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))

        torch.cuda.synchronize()
        print("time %.2f" % (time.time() - tick))
        print("error", torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        return {"error": torch.sum(Losses).item()}

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
