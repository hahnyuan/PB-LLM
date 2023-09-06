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
    parser.add_argument("--n_calib_samples", type=int, default=128)
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
            args.calib_dataset, tokenizer, args.n_calib_samples, seqlen=2048
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
