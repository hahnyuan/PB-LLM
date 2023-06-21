# %%
from transformers import AutoModelForCausalLM

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "facebook/opt-350m"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, quantization_config=bnb_config, device_map={"": 0}
# )
model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0})

# %%
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# %%
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


# %%
from peft import LoraConfig, get_peft_model

# config = LoraConfig(
#     r=8,
#     lora_alpha=32,
#     # target_modules=["k_proj","v_proj","q_proj","project_out","out_proj","fc1","fc2"],
#     target_modules=["k_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
# model = get_peft_model(model, config)

from quant import BinaryLinear
import torch.nn as nn

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

# %%
from datasets import load_dataset
import os

# os.environ["http_proxy"] = "127.0.0.1:7890"
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# %%
import transformers

# needed for gpt-neo-x tokenizer
tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# %%
import wandb

wandb.__version__

# %%
