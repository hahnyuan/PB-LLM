import sys

sys.path.append(".")

import argparse
import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from transformers.models.opt.modeling_opt import OPTForCausalLM
from datasets import load_dataset
from datautils import get_ptq_calib_data
from quant import (
    XnorBinaryLinear,
    BinaryLinear,
    IrBinaryLinear,
    BiRealLinear,
    OutliersQLinearColumn,
    BinaryXnorExceptOutliersLinear,
    OutliersQLinearUnstruct,
)

# , FdaBinaryLinear
from utils import (
    print_trainable_parameters,
    print_memory_usage,
    prepare_model_for_training,
    save_bnn,
)
from evaluate import evaluate_model

"""
Usage
python experiments/ptq_binary.py --binarization_method xnor --debug
"""


def replace_qlinear(args, root_module, name_prefix=""):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            print(name)
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            # choose binariztaion method
            if args.binarization_method == "xnor":
                quant_cls = XnorBinaryLinear
            else:
                print(f"not supported binarization method {args.binarization_method}")
                raise NotImplementedError
            if args.outlier_fraction > 0:
                qlinear = OutliersQLinearUnstruct(
                    module.weight,
                    module.bias,
                    dense_class=quant_cls,
                    outlier_metric=args.outlier_metric,
                    outlier_fraction=args.outlier_fraction,
                )
            else:
                qlinear = quant_cls(module.weight, module.bias)

            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace layer {name_prefix}{name} with {qlinear}")


@torch.inference_mode()
def quantize(
    model,
    examples,
    use_cuda_fp16: bool = True,
    autotune_warmup_after_quantized: bool = False,
    cache_examples_on_gpu: bool = True,
):
    device_map = model.hf_device_map
    layer_inputs = []
    attention_masks = []
    position_ids = []
    layer_input_kwargs = []
    layer_outputs = []

    class LayerHijacker(nn.Module):
        """hijack layer's forward pass to cache data"""

        def __init__(self, m, device):
            super().__init__()
            self.module = m
            self.data_device = device

        def forward(self, inp=None, **kwargs):
            if (
                inp is None
            ):  # some models use all key-value arguments in forward pass call
                for kwarg_name in ["hidden_states"]:
                    if kwarg_name in kwargs:
                        inp = kwargs[kwarg_name]
                        break
            layer_inputs.append(move_to_device(inp, self.data_device))
            attention_masks.append(kwargs["attention_mask"].to(self.data_device))
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to_device(pos_ids, self.data_device))
            one_kwargs = dict()
            for k, v in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    if isinstance(v, torch.Tensor):
                        one_kwargs[k] = move_to_device(v, self.data_device)
                    else:
                        one_kwargs[k] = v
            layer_input_kwargs.append(one_kwargs)
            raise ValueError

    forward_pass_use_cache = model.model.config.use_cache
    model.model.config.use_cache = False

    num_batches = len(examples)
    layers = get_module_by_name_prefix(model.model, model.layers_block_name)

    force_layer_back_to_cpu = False
    if get_device(layers[0]) == CPU:
        layers[0] = layers[0].to(CUDA_0)
        force_layer_back_to_cpu = True

    cur_layer_device = get_device(layers[0])
    ori_outside_layer_module_devices = {}
    for module_name in model.outside_layer_modules:
        module = get_module_by_name_prefix(model.model, module_name)

        if module is None:
            continue

        ori_outside_layer_module_devices[module_name] = get_device(module)
        if module is not None:
            move_to_device(module, cur_layer_device)

    # get inputs for first layer
    layers[0] = LayerHijacker(layers[0], cur_layer_device)
    for example in examples:
        for k, v in example.items():
            if len(v.shape) == 1:
                v = v.unsqueeze(0)
            example[k] = move_to_device(v, cur_layer_device)
        try:
            model.model(**example)
        except ValueError:
            pass
    layers[0] = layers[0].module

    move_to_device(layers[0], CPU if force_layer_back_to_cpu else cur_layer_device)
    for module_name in model.outside_layer_modules:
        module = get_module_by_name_prefix(model.model, module_name)
        if module is not None:
            move_to_device(module, ori_outside_layer_module_devices[module_name])

    torch.cuda.empty_cache()

    # resize attention mask and position ids for some special models
    attention_masks = model._resize_attention_mask(attention_masks)
    position_ids = model._resize_position_ids(position_ids)

    inside_layer_modules = model.inside_layer_modules
    if not model.quantize_config.true_sequential:
        inside_layer_modules = [sum(inside_layer_modules, [])]
    quantizers = {}
    for i in range(len(layers)):
        logger.info(f"Start quantizing layer {i + 1}/{len(layers)}")
        layer = layers[i]
        force_layer_back_to_cpu = False
        if get_device(layer) == CPU:
            move_to_device(layer, CUDA_0)
            force_layer_back_to_cpu = True
        cur_layer_device = get_device(layer)

        full = find_layers(layer)
        for names in inside_layer_modules:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer.configure(
                    model.quantize_config.bits,
                    perchannel=True,
                    sym=model.quantize_config.sym,
                    mse=False,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                layer_attention_mask = move_to_device(
                    attention_masks[j], cur_layer_device
                )
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_position_ids = (
                    None
                    if not position_ids
                    else move_to_device(position_ids[j], cur_layer_device)
                )
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[j].items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                    else:
                        additional_layer_inputs[k] = v
                layer(layer_input, **additional_layer_inputs)
            for h in handles:
                h.remove()

            for name in subset:
                logger.info(f"Quantizing {name} in layer {i + 1}/{len(layers)}...")
                scale, zero, g_idx = gptq[name].fasterquant(
                    percdamp=model.quantize_config.damp_percent,
                    group_size=model.quantize_config.group_size,
                    actorder=model.quantize_config.desc_act,
                    static_groups=model.quantize_config.static_groups,
                )
                quantizers[f"{model.layers_block_name}.{i}.{name}"] = (
                    gptq[name].quantizer.to(
                        CPU if force_layer_back_to_cpu else cur_layer_device
                    ),
                    move_to_device(
                        scale, CPU if force_layer_back_to_cpu else cur_layer_device
                    ),
                    move_to_device(
                        zero, CPU if force_layer_back_to_cpu else cur_layer_device
                    ),
                    move_to_device(
                        g_idx, CPU if force_layer_back_to_cpu else cur_layer_device
                    ),
                )
                gptq[name].free()

        for j in range(num_batches):
            layer_input = move_to_device(layer_inputs[j], cur_layer_device)
            layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
            additional_layer_inputs = {"attention_mask": layer_attention_mask}
            layer_position_ids = (
                None
                if not position_ids
                else move_to_device(position_ids[j], cur_layer_device)
            )
            if layer_position_ids is not None:
                additional_layer_inputs["position_ids"] = layer_position_ids
            for k, v in layer_input_kwargs[j].items():
                if isinstance(v, torch.Tensor):
                    additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                else:
                    additional_layer_inputs[k] = v
            layer_output = move_to_device(
                layer(layer_input, **additional_layer_inputs)[0],
                cur_layer_device if cache_examples_on_gpu else CPU,
            )
            layer_outputs.append(layer_output)

        layers[i] = move_to_device(
            layer, CPU if force_layer_back_to_cpu else cur_layer_device
        )
        del layer
        del gptq
        del layer_inputs
        layer_inputs, layer_outputs = layer_outputs, []
        torch.cuda.empty_cache()

    pack_model(
        model=model.model,
        quantizers=quantizers,
        bits=model.quantize_config.bits,
        group_size=model.quantize_config.group_size,
        use_triton=use_triton,
        use_cuda_fp16=use_cuda_fp16,
        desc_act=model.quantize_config.desc_act,
        warmup_triton=autotune_warmup_after_quantized,
        force_layer_back_to_cpu=force_layer_back_to_cpu,
    )
    if device_map:
        model.model = remove_hook_from_module(model.model, recurse=True)
        model.model = simple_dispatch_model(model.model, device_map)
    model.model.config.use_cache = forward_pass_use_cache

    model._quantized = True

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-125m",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--calib_dataset", type=str, default="wikitext2", help="Dataset name"
    )
    parser.add_argument("--n_calib_samples", type=int, default=10)
    parser.add_argument("--binarization_method", type=str, default="xnor")
    parser.add_argument("--outlier_metric", type=str, default="hessian")
    parser.add_argument(
        "--outlier_fraction", type=float, default=0.5, help="Percentage of outliers"
    )
    parser.add_argument("--eval", action="store_true", help="evaluate the model")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")

    replace_qlinear(args, model)
    model = model.eval()
    default_checkpoint_path = f"outputs/{args.model_id.replace('/','_')}/ptq_{args.binarization_method}_outlier_{args.outlier_metric}_{args.outlier_fraction}.pth"

    if args.eval:
        if args.checkpoint == "":
            state_dict = torch.load(default_checkpoint_path)
        else:
            state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)
        model = model.eval()
        evaluate_model(model, tokenizer, args.model_id, "piqa,boolq", limit=200)
    else:
        calib_dataset = get_ptq_calib_data(
            args.calib_dataset,
            tokenizer,
            args.model_id,
            args.n_calib_samples,
            seqlen=2048,
        )
        data = torch.cat([d["input_ids"] for d in calib_dataset], dim=0)
        with torch.no_grad():
            model.forward(data)
        state_dict = model.state_dict()
        if not os.path.exists(os.path.dirname(default_checkpoint_path)):
            os.makedirs(os.path.dirname(default_checkpoint_path))
        torch.save(
            state_dict,
            default_checkpoint_path,
        )
