import argparse
import torch
import torch.nn as nn
import os
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
)
from datasets import load_dataset
from quant import BinaryLinear, IrBinaryLinear, FdaBinaryLinear, XnorBinaryLinear
from utils import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map={"": 0})

    # Enable gradient checkpointing and prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=50)
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(outputs)

    # Load dataset
    data = load_dataset(args.dataset)
    # sample = data['train'][0]
    # print(sample.keys())
    # Slice first 1000 samples
    # data = data['train'].select(range(1500))
    print('prepare training data')
    data = data.map(lambda samples: tokenizer(samples["text"], truncation=True, max_length=128), batched=True, batch_size=1000, num_proc=os.cpu_count())
    for name, _ in model.named_modules():
        print(name)
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
            warmup_steps=100,
            max_steps=10 if args.debug else 1000,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_torch",
            report_to="tensorboard",
            logging_dir='./outputs/runs/logging/',
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataset=data['train'],
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # trainer_w_distiller = Trainer(
        #     model=model,
        #     train_dataset=data["train"],
        #     args=training_args,
        #     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # )

        model.config.use_cache = (
            False  # silence the warnings. Please re-enable for inference!
        )

        # Train the model
        trainer.train()
    trainer.save_model(output_dir='./checkpoints/lamma7b-ste')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id", type=str, default="huggyllama/llama-7b", help="Pretrained model ID"
    )
    parser.add_argument(
        "--dataset", type=str, default='togethercomputer/RedPajama-Data-1T-Sample', help="Dataset name"
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
    args = parser.parse_args()

    main(args)
