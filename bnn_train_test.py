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
from evaluate import evaluate_model


def main(model_id, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})

    # Enable gradient checkpointing and prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)

    module_name_dict = {name: module for name, module in model.named_modules()}

    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                continue
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            qlinear = BinaryLinear(module.weight, module.bias)
            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace {name} with {qlinear}")

    print_trainable_parameters(model)

    # Load dataset
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    # Set tokenizer properties
    tokenizer.pad_token = tokenizer.eos_token

    # Define training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        max_steps=2,
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
    model.eval()
    evaluate_model(
        model,
        tokenizer,
        args.model_id,
        args.tasks,
        limit=args.eval_limit,
        batch_size=args.eval_batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id", type=str, default="facebook/opt-350m", help="Pretrained model ID"
    )
    parser.add_argument(
        "--dataset", type=str, default="Abirate/english_quotes", help="Dataset name"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="evaluate tasks name, can be tasks separated by , lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq",
    )
    parser.add_argument(
        "--eval_limit",
        default=-1,
        type=int,
        help="number of test samples for debug, set to -1 is no limit",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=2,
        type=int,
        help="eval batch size, default is 2",
    )
    args = parser.parse_args()

    main(args.model_id, args.dataset)
