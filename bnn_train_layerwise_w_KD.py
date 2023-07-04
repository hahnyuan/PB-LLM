import argparse
import copy

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
)
from datasets import load_dataset
from quant import BinaryLinear, IrBinaryLinear, FdaBinaryLinear, XnorBinaryLinear
from utils import *
import torch.nn.functional as F


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map={"": 0})

    # Enable gradient checkpointing and prepare model for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_training(model)
    teacher_model = copy.deepcopy(model)

    # Load dataset
    data = load_dataset(args.dataset)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    layers = [(f"layer{i}", _) for i, _ in enumerate(model.base_model.decoder.layers)]
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
        )

        class Trainer_w_Distiller(Trainer):
            def __init__(self, *args, teacher_model=None, **kwargs):
                super().__init__(*args, **kwargs)

                self.teacher = teacher_model

                # place teacher on same device as student
                self._move_model_to_device(self.teacher, self.model.device)

                self.teacher.eval()

            # def Distiller(self, ):
            #     teacher_output = teacher_model()
            def kl_loss(self, tensor1, tensor2, temperature=0.5):
                tensor1 = F.softmax(tensor1 / temperature, dim=-1)
                tensor2 = F.softmax(tensor2 / temperature, dim=-1)

                kl_loss = F.kl_div(torch.log(tensor1), tensor2, reduction='batchmean')

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
                # TODO: feature-based DK
                # kd_loss = self.kl_loss(outputs[1],outputs_teacher[1])
                kd_loss = F.mse_loss(outputs[1],outputs_teacher[1])
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


        # Create trainer
        # trainer = Trainer(
        #     model=model,
        #     train_dataset=data["train"],
        #     args=training_args,
        #     data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        # )
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
    trainer.save_model(output_dir=args.model_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id", type=str, default="facebook/opt-350m", help="Pretrained model ID"
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
        "--model_save_dir", type=str, default="./checkpoints/vanilla_kd", help="saving model to this directory"
    )
    args = parser.parse_args()

    main(args)
