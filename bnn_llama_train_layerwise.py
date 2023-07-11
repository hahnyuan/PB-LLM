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
from utils import *

"""
Usage
CUDA_VISIBLE_DEVICES='3' XDG_CACHE_HOME='/data/shangyuzhang/' python bnn_llama_train_layerwise_1.py --binarization_method='ir' --debug --model_id 'openlm-research/open_llama_7b'
CUDA_VISIBLE_DEVICES='4,5' XDG_CACHE_HOME='/data/shangyuzhang/' python bnn_llama_train_layerwise_1.py --binarization_method='xnor' --dataset='togethercomputer/RedPajama-Data-1T-Sample' --model_save_dir "./checkpoints/llama-7b-xnor"
CUDA_VISIBLE_DEVICES='4,5' XDG_CACHE_HOME='/data/shangyuzhang/' python bnn_llama_train_layerwise.py --binarization_method='ste' --dataset='togethercomputer/RedPajama-Data-1T-Sample' --model_save_dir "./checkpoints/huggyllama-7b-ste" --model_id=huggyllama/llama-7b
"""

def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
    model = LlamaForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map='auto',
    )
    # model = LlamaForCausalLM.from_pretrained(
    #     '/data/shangyuzhang/BinaryLLM/checkpoints/llama-7b-xnor', torch_dtype=torch.float16, device_map='auto',
    # )

    # generate a sample
    # prompt = "Hey, are you conscious? Can you talk to me?"
    prompt = "Hey, is llama the best language model?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=60)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(outputs)

    # Enable gradient checkpointing and prepare model for k-bit training
    # model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)

    # Load dataset
    data = load_dataset(args.dataset)
    print('prepare training data')
    data = data.map(lambda samples: tokenizer(samples["text"], truncation=True, max_length=512), batched=True,
                    batch_size=100, writer_batch_size=100, num_proc=os.cpu_count())
    print(data.shape)
    # for name, _ in model.named_modules():
    #     print(name)
    layers = [(f"layer{i}", _) for i, _ in enumerate(model.base_model.layers)]
    if args.order == "reverse":
        layers = layers[::-1]
    for i, layer in layers:
        module_name_dict = {name: module for name, module in layer.named_modules()}
        for name, module in module_name_dict.items():
            if isinstance(module, nn.Linear):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]
                #choose binariztaion method
                if args.binarization_method == 'ste':
                    qlinear = BinaryLinear(module.weight, module.bias)
                elif args.binarization_method == 'ir':
                    qlinear = IrBinaryLinear(module.weight, module.bias)
                elif args.binarization_method == 'xnor':
                    qlinear = XnorBinaryLinear(module.weight, module.bias)
                elif args.binarization_method == 'fda':
                    qlinear = FdaBinaryLinear(module.weight, module.bias)
                else:
                    print('not support this binarization method ')

                setattr(father, name[ind + 1 :], qlinear)
                print(f"replace layer{i} {name} with {qlinear}")

        print_trainable_parameters(model)

        # Set tokenizer properties
        tokenizer.pad_token = tokenizer.eos_token

        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            max_steps=10 if args.debug else 200,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_torch",
            report_to="tensorboard",
        )

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, "requires grad")
        #     else:
        #         print(name, "does not require grad")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
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
                print(name, "requires grad, and set to not required")
                param.requires_grad = False
            else:
                print(name, "does not require grad")
        print_memory_usage()
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    binarized_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(binarized_output)
    trainer.save_model(output_dir=args.model_save_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id", type=str, default="openlm-research/open_llama_7b", help="Pretrained model ID"
    )
    parser.add_argument(
        "--dataset", type=str, default="Abirate/english_quotes", help="Dataset name"
    )
    parser.add_argument(
        "--order", type=str, default="forward", choices=["forward", "reverse"]
    )
    parser.add_argument(
        "--binarization_method", type=str, default="ste", choices=["ste", "ir", "fda", 'xnor']
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (only 10 steps)"
    )
    parser.add_argument(
        "--model_save_dir", type=str, default="./checkpoints/llama-7b-xnor", help="saving model to this directory"
    )
    args = parser.parse_args()

    main(args)
