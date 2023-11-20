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
)

# from transformers import LlamaTokenizer, LlamaForCausalLM
from datautils import get_qat_dataset
from quant import (
    BinaryInterface,
    BinaryXnorExceptOutliersLinear,
    BinaryXnorExceptOutliersLinearHessian,
)

from utils import (
    print_trainable_parameters,
    prepare_model_for_training,
)

from transformers import get_cosine_with_hard_restarts_schedule_with_warmup


def get_scheduler(num_training_steps: int):
    def lr_scheduler(optimizer):
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=num_training_steps,
            num_cycles=5,
        )

    return lr_scheduler


def replace_with_qlinear(root_module):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            if args.binarization_method == "xnor_outlier":
                qlinear = BinaryXnorExceptOutliersLinear(
                    module.weight, module.bias, args.outlier_fraction
                )
            elif args.binarization_method == "xnor_outlier_hessian":
                qlinear = BinaryXnorExceptOutliersLinearHessian(
                    module.weight, module.bias, args.outlier_fraction
                )
            else:
                raise NotImplementedError
            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace layer {name} with {qlinear}")
            qlinear.global_name = args.model_id + name


def to_regular_linear(root_module):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, BinaryInterface):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            linear = module.to_regular_linear()
            setattr(father, name[ind + 1 :], linear)
            print(f"replace layer {name} with {linear}")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16
    )
    # model=model.to_bettertransformer()

    model = prepare_model_for_training(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("prepare training data")
    data = get_qat_dataset(args.dataset, tokenizer, args.data_percent)

    # Training
    print_trainable_parameters(model)
    replace_with_qlinear(model)

    # Print mean bit width
    tot_bit=0
    tot_params=0
    for name, module in model.named_modules():
        if isinstance(module, BinaryInterface):
            module.gen_outlier_mask()
            # print(module.outlier_nbits)
            tot_bit+=(module.outlier_nbits+1)*module.weight.numel()
            tot_params+=module.weight.numel()
    print(f"mean_bit: {tot_bit/tot_params} frac: {tot_bit/tot_params/16}")

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=args.train_steps * 0.05,
        max_steps=args.train_steps,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_torch",
        report_to="tensorboard",
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False

    # Train the model
    trainer.train()

    # Save model
    model.eval()
    save_dir = f"outputs/{args.model_id}/{args.binarization_method}_{args.outlier_fraction}_{args.train_steps}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    to_regular_linear(model)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-350m",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--dataset", type=str, default="Abirate/english_quotes", help="Dataset name"
    )
    parser.add_argument(
        "--data_percent", type=float, default=100, help="Percentage of data to use"
    )
    parser.add_argument(
        "-s", "--train_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--binarization_method",
        type=str,
        default="xnor_outlier",
        choices=[
            "xnor_outlier",
            "xnor_outlier_hessian",
        ],
    )
    parser.add_argument(
        "--outlier_fraction", type=float, default=0.1, help="Percentage of outliers"
    )
    args = parser.parse_args()

    main(args)
