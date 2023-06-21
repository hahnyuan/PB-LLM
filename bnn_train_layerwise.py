import argparse
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from quant import BinaryLinear
from utils import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map={"": 0})

    # Enable gradient checkpointing and prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)

    # Load dataset
    data = load_dataset(args.dataset)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    for layer in model.base_model.decoder.layers[::-1]:
        module_name_dict = {name: module for name, module in layer.named_modules()}
        for name, module in module_name_dict.items():
            if isinstance(module, nn.Linear):
                ind = name.rfind(".")
                if ind == -1:
                    # continue
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]
                qlinear = BinaryLinear(module.weight, module.bias)
                setattr(father, name[ind + 1 :], qlinear)
                print(f"replace {name} with {qlinear}")

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
            optim="paged_adamw_8bit",
            report_to="tensorboard",
        )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id", type=str, default="facebook/opt-350m", help="Pretrained model ID"
    )
    parser.add_argument(
        "--dataset", type=str, default="Abirate/english_quotes", help="Dataset name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode (only 10 steps)"
    )
    args = parser.parse_args()

    main(args)
