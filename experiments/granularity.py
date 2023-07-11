import sys

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

# from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from quant import BinaryLinear, IrBinaryLinear, FdaBinaryLinear, XnorBinaryLinear
from utils import print_trainable_parameters,print_memory_usage,prepare_model_for_training,save_bnn

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
            if args.binarization_method == "ste":
                qlinear = BinaryLinear(module.weight, module.bias)
            elif args.binarization_method == "ir":
                qlinear = IrBinaryLinear(module.weight, module.bias)
            elif args.binarization_method == "xnor":
                qlinear = XnorBinaryLinear(module.weight, module.bias)
            elif args.binarization_method == "fda":
                qlinear = FdaBinaryLinear(module.weight, module.bias)
            else:
                print("not support this binarization method ")

            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace layer {name_prefix}{name} with {qlinear}")


def iterative_train(model, ordered_name_modules, data, tokenizer):
    """
    ordered_name_modules: [(name, module), ...]
    """
    

    for module_name, module in ordered_name_modules:
        print_trainable_parameters(model)
        replace_qlinear(module, f"{module_name}.")

        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=2 if args.debug else 200,
            learning_rate=1e-4,
            fp16=False,
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

        model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )

        # Train the model
        trainer.train()
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(name, "requires grad, and set to not required")
                param.requires_grad = False
            # else:
                # print(name, "does not require grad")
        
        print_memory_usage()
        save_bnn(model, args.model_save_dir+f'/{module_name}')
        # trainer.save_model(output_dir=args.model_save_dir+f'/layer{i}')
        # bnn_meta=get_bnn_meta(model)
        # torch.save(bnn_meta,args.model_save_dir+f'/layer{i}/bnn_meta.pt')



def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
    model = LlamaForCausalLM.from_pretrained(args.model_id,device_map="auto")
    # model = LlamaForCausalLM.from_pretrained(
    #     args.model_id,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    # Enable gradient checkpointing and prepare model for k-bit training
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("prepare training data")
    if args.dataset == "red_pajama":
        from datautils import get_redpajama_train

        data = get_redpajama_train(tokenizer, args.data_percent)
    else:
        raise NotImplementedError
        data = load_dataset(args.dataset)

        data = data.map(
            lambda samples: tokenizer(samples["text"], truncation=True, max_length=512),
            batched=True,
            batch_size=100,
            writer_batch_size=100,
            num_proc=os.cpu_count(),
        )
        print(data.shape)
    # for name, _ in model.named_modules():
    #     print(name)
    if args.granularity == "per_block":
        ordered_name_modules = [(f"block{i}", _) for i, _ in enumerate(model.base_model.layers)]
        if args.order == "reverse":
            ordered_name_modules = ordered_name_modules[::-1]
    else:
        raise NotImplementedError
    iterative_train(model,ordered_name_modules, data, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="huggyllama/llama-7b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--granularity", type=str, default="per_block", choices=["per_block","per_linear"]
    )
    parser.add_argument(
        "--dataset", type=str, default="red_pajama", help="Dataset name"
    )
    parser.add_argument(
        "--data_percent", type=float, default=100, help="Percentage of data to use"
    )
    parser.add_argument(
        "--order", type=str, default="forward", choices=["forward", "reverse"]
    )
    parser.add_argument(
        "--binarization_method",
        type=str,
        default="xnor",
        choices=["ste", "ir", "fda", "xnor"],
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
