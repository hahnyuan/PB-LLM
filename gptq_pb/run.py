import time

import torch
import torch.nn as nn

from gptq import LowHighGPT
from high_quant import HighQuantizer
from low_quant import LowQuantizer
from modelutils import find_layers


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "huggyllama" in model:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = 2048
    return model


@torch.no_grad()
def quant_sequential(model, dataloader, dev):
    print("Starting ...")

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "huggyllama" in args.model:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "huggyllama" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")
    plt_x = []
    plt_error = []
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            low_quantizer = LowQuantizer(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=args.groupsize,
            )
            high_quantizer = HighQuantizer(
                args.high_bit,
                True,
                False,
                False,
            )
            gpts[name] = LowHighGPT(
                subset[name],
                low_quantizer,
                high_quantizer,
                salient_metric=args.salient_metric,
                disable_gptq=args.disable_gptq,
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print("Quantizing ...")
            info = gpts[name].fasterquant(
                args.low_frac, percdamp=args.percdamp, blocksize=args.blocksize
            )
            gpts[name].free()
            plt_x.append(f"{i}_{name}")
            plt_error.append(info["error"])

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    if args.plot:
        title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{args.low_frac}_{args.high_bit}"
        torch.save([plt_x, plt_error], "../output/" + title.replace("/", "_") + ".pkl")
        import matplotlib.pyplot as plt

        plt.plot(plt_error)
        plt.xticks(range(1, len(plt_x) + 1), plt_x)
        plt.title(title)
        plt.savefig("../output/" + title.replace("/", "_") + ".jpg")

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["xnor", "sign", "no", "2bit", "4bit", "prune"],
        help="quantization method; `xnor` is the method used in paper; `prune` is the method used in sparseGPTQ",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--low_frac", type=float, default=0, help="Target low_frac")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--groupsize", type=int, default=-1, help="Groupsize for GPTQ quantizing"
    )
    parser.add_argument(
        "--salient_metric",
        type=str,
        default="magnitude",
        choices=["magnitude", "hessian"],
    )
    parser.add_argument(
        "--high_bit",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    args = parser.parse_args()

    device = "cuda:0"
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{args.low_frac}_{args.high_bit}_{args.groupsize}_{args.salient_metric}"
    save_file = "../output/" + save_title.replace("/", "_") + ".pt"
    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
    elif args.low_frac:
        model = get_model(args.model)
        model.eval()
        tick = time.time()
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )
        quant_sequential(model, dataloader, device)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if "fc2" in n:
                break
        print(time.time() - tick)

    for dataset in ["wikitext2", "ptb", "c4"]:
        # for dataset in ['c4']:
        # for dataset in ['wikitext2']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        if "opt" in args.model:
            from eval_ppl_utils import opt_eval

            opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "huggyllama" in args.model:
            from eval_ppl_utils import llama_eval

            llama_eval(model, testloader, device, dataset, args.log_wandb)

    if args.save:
        save_path = os.path.dirname(save_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_file)
