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
from peft import prepare_model_for_kbit_training


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main(model_id, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})

    # Enable gradient checkpointing and prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

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
        warmup_steps=100,
        max_steps=10000,
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
    args = parser.parse_args()

    main(args.model_id, args.dataset)
