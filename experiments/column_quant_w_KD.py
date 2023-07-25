import sys
import copy
sys.path.append(".")
import argparse
import os
import torch
import torch.nn as nn

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     DataCollatorForLanguageModeling,
#     TrainingArguments,
#     Trainer,
# )
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

# from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from datautils import get_qat_dataset
from quant import (
    XnorBinaryLinear,
    BinaryLinear,
    IrBinaryLinear,
    BiRealLinear,
    OutliersQLinearColumn,
    BinaryXnorExceptOutliersLinear,
)

# , FdaBinaryLinear
from utils import (
    print_trainable_parameters,
    print_memory_usage,
    prepare_model_for_training,
    save_bnn,
)
from evaluate import evaluate_model
import torch.nn.functional as F

"""
Usage
python bnn_train_layerwise.py --binarization_method xnor --debug
"""


def replace_qlinear(root_module, name_prefix=""):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            # choose binariztaion method
            if "xnor" in args.binarization_method:
                quant_cls = XnorBinaryLinear
            elif "ste" in args.binarization_method:
                quant_cls = BinaryLinear
            elif "ir" in args.binarization_method:
                quant_cls = IrBinaryLinear
            elif "fda" in args.binarization_method:
                quant_cls = FdaBinaryLinear
            elif "bireal" in args.binarization_method:
                quant_cls = BiRealLinear
            else:
                print(f"not supported binarization method {args.binarization_method}")
                raise NotImplementedError
            if "act_outlier_column" in args.binarization_method:
                qlinear = OutliersQLinearColumn(
                    module.weight,
                    module.bias,
                    dense_class=quant_cls,
                    outlier_metric="act_L1",
                    outlier_fraction=args.outlier_fraction,
                )
            elif "outlier_column" in args.binarization_method:
                qlinear = OutliersQLinearColumn(
                    module.weight,
                    module.bias,
                    dense_class=quant_cls,
                    outlier_fraction=args.outlier_fraction,
                )
            # elif "outlier" in args.binarization_method:
            #     qlinear = OutliersLinear(
            #         module.weight, module.bias, dense_class=quant_cls
            #     )
            elif "outlier" in args.binarization_method:
                qlinear = BinaryXnorExceptOutliersLinear(module.weight, module.bias)
            else:
                qlinear = quant_cls(module.weight, module.bias)

            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace layer {name_prefix}{name} with {qlinear}")


def iterative_train(model, teacher_model,ordered_name_modules, data, tokenizer):
    """
    ordered_name_modules: [(name, module), ...]
    """

    for module_name, module in ordered_name_modules:
        print_trainable_parameters(model)
        replace_qlinear(module, f"{module_name}.")

        # Define training arguments
        if args.train_steps:
            training_args = TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                max_steps=args.train_steps,
                learning_rate=1e-4,
                fp16=False,
                logging_steps=1,
                output_dir="outputs",
                optim="adamw_torch",
                report_to="tensorboard",
            )

            # Create trainer
            class Trainer_w_Distiller(Trainer):
                def __init__(self, *args, teacher_model=None, **kwargs):
                    super().__init__(*args, **kwargs)

                    self.teacher = teacher_model
                    # place teacher on same device as student
                    self._move_model_to_device(self.teacher, self.model.device)
                    self.teacher.eval()

                def kl_loss(self, tensor1, tensor2, temperature=0.5):
                    # student: tensor1, teacher: tensor2
                    tensor1 = F.log_softmax(tensor1 / temperature, dim=-1)
                    tensor2 = F.softmax(tensor2 / temperature, dim=-1)

                    kl_loss = F.kl_div(tensor1, tensor2, reduction='batchmean') * (temperature**2) / tensor1.shape[0]

                    return kl_loss

                def compute_loss(self, model, inputs, return_outputs=False):
                    """
                    How the loss is computed by Trainer. By default, all models return the loss in the first element.

                    Subclass and override for custom behavior.
                    """
                    # print(1111111111111111111111111111111111111111111111111111111111111111111)
                    if self.label_smoother is not None and "labels" in inputs:
                        labels = inputs.pop("labels")
                    else:
                        labels = None
                    outputs = model(**inputs)
                    outputs_teacher = teacher_model(**inputs)
                    kd_loss = self.kl_loss(outputs[1], outputs_teacher[1])
                    # kd_loss = F.mse_loss(outputs[1],outputs_teacher[1])
                    # print(kd_loss)

                    # Save past state if it exists
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index]

                    if labels is not None:
                        if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                            loss = self.label_smoother(outputs, labels, shift_labels=True)
                        else:
                            loss = self.label_smoother(outputs, labels)
                    else:
                        if isinstance(outputs, dict) and "loss" not in outputs:
                            raise ValueError(
                                "The model did not return a loss from the inputs, only the following keys: "
                                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                            )
                        # We don't use .loss here since the model may return tuples instead of ModelOutput.
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                    alpha = 0.01
                    loss += alpha*kd_loss
                    return (loss, outputs) if return_outputs else loss

            trainer = Trainer_w_Distiller(
                model=model,
                args=training_args,
                teacher_model=teacher_model,
                train_dataset=data["train"],
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )

            model.config.use_cache = (
                False  # silence the warnings. Please re-enable for inference!
            )

            # Train the model
            trainer.train()
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False

        print_memory_usage()
        model.eval()
        result = evaluate_model(
            model, tokenizer, args.model_id, "piqa,boolq", limit=100
        )
        boolq = result["results"]["boolq"]["acc"]
        piqa = result["results"]["piqa"]["acc"]
        print(boolq, piqa)
        with open("outputs/intrain_eval.log", "a+") as f:
            f.write(
                f"{args.model_id}: {args.binarization_method} {args.outlier_fraction} {args.train_steps} {args.dataset} {module_name} {boolq} {piqa}\n"
            )

        save_bnn(
            model,
            args.model_save_dir
            + f"/{args.granularity}/{args.model_id.replace('/','_')}_o{args.outlier_fraction}_{module_name}",
        )

        model.eval()
        evaluate_model(model, tokenizer, args.model_id, 'piqa,boolq', limit=-1)



def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="auto")

    # Enable gradient checkpointing and prepare model for k-bit training
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()

    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("prepare training data")
    data = get_qat_dataset(args.dataset, tokenizer, args.data_percent)

    if args.granularity == "per_block":
        if isinstance(model, OPTForCausalLM):
            ordered_name_modules = [
                (f"block{i}", _) for i, _ in enumerate(model.model.decoder.layers)
            ]
        else:
            # LLaMA
            ordered_name_modules = [
                (f"block{i}", _) for i, _ in enumerate(model.base_model.layers)
            ]
        if args.order == "reverse":
            ordered_name_modules = ordered_name_modules[::-1]
    elif args.granularity == "per_linear":
        ordered_name_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                ordered_name_modules.append((name, module))
    elif args.granularity == "whole_model":
        # ordered_name_modules = {name: module for name, module in model.named_modules()}
        ordered_name_modules = [("whole_model", model)]
    else:
        raise NotImplementedError
    iterative_train(model, teacher_model, ordered_name_modules, data, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-350m",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default="whole_model",
        choices=["per_block", "per_linear", "whole_model"],
    )
    parser.add_argument(
        "--dataset", type=str, default="Abirate/english_quotes", help="Dataset name"
    )
    parser.add_argument(
        "--data_percent", type=float, default=100, help="Percentage of data to use"
    )
    parser.add_argument(
        "--order", type=str, default="forward", choices=["forward", "reverse"]
    )
    parser.add_argument(
        "-s", "--train_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--binarization_method",
        type=str,
        default="xnor_act_outlier_column",
        choices=[
            "ste",
            "ir",
            "fda",
            "xnor",
            "bireal",
            "ste_outlier",
            "xnor_outlier",
            "xnor_outlier_column",
            "ir_outlier_column",
            "xnor_act_outlier_column",
            "ir_act_outlier_column",
            "fda_act_outlier_column",
            "bireal_act_outlier_column",
        ],
    )
    parser.add_argument(
        "--outlier_fraction", type=float, default=0.05, help="Percentage of outliers"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (only 10 steps)"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="./checkpoints",
        help="saving model to this directory",
    )
    args = parser.parse_args()

    main(args)
