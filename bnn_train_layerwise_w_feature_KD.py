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
    # teacher_model = copy.deepcopy(model)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map={"": 0})

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
                replaced_layer_index = int(i[5:])

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
            def __init__(self, *args, teacher_model=None, replaced_layer_index, **kwargs):
                super().__init__(*args, **kwargs)

                self.teacher_model = teacher_model
                # place teacher on same device as student
                self._move_model_to_device(self.teacher_model, self.model.device)
                self.teacher_model.eval()
                self.replaced_layer_index = replaced_layer_index

            def kl_loss(self, tensor1, tensor2, temperature=0.5):
                # student: tensor1, teacher: tensor2
                tensor1 = F.log_softmax(tensor1 / temperature, dim=-1)
                tensor2 = F.softmax(tensor2 / temperature, dim=-1)

                kl_loss = F.kl_div(tensor1, tensor2, reduction='batchmean') * (temperature**2) / tensor1.shape[0]

                return kl_loss

            def featurekd_loss(self, layer_index, inputs, model, teacher_model):
                def hook_fn(module, input, output):
                    module.output = output.detach()

                hook_student = model.model.decoder.layers[layer_index].fc2.register_forward_hook(hook_fn)
                model(**inputs)
                student_feature = model.model.decoder.layers[layer_index].fc2.output
                # print(student_feature)
                # print('student hook checked')

                hook_teacher = teacher_model.model.decoder.layers[layer_index].fc2.register_forward_hook(hook_fn)
                teacher_model(**inputs)
                teacher_feature = teacher_model.model.decoder.layers[layer_index].fc2.output
                # print(teacher_feature)

                featurekd_loss = F.mse_loss(student_feature, teacher_feature)
                # print(featurekd_loss)
                return featurekd_loss

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
                teacher_model = self.teacher_model
                outputs_teacher = teacher_model(**inputs)

                kd_loss = self.kl_loss(outputs[1], outputs_teacher[1])

                def hook_fn(module, input, output):
                    module.output = output.detach()

                replaced_layer_index = self.replaced_layer_index
                # print(replaced_layer_index)
                hook_student = model.model.decoder.layers[replaced_layer_index].fc2.register_forward_hook(hook_fn)
                model(**inputs)
                student_feature = model.model.decoder.layers[replaced_layer_index].fc2.output
                # print(student_feature)
                # print('student hook checked')

                hook_teacher = teacher_model.model.decoder.layers[replaced_layer_index].fc2.register_forward_hook(hook_fn)
                teacher_model(**inputs)
                teacher_feature = teacher_model.model.decoder.layers[replaced_layer_index].fc2.output
                # print(teacher_feature)

                featurekd_loss = F.mse_loss(student_feature, teacher_feature)
                # print(featurekd_loss)

                # featurekd_loss = self.featurekd_loss(layer_index=self.replaced_layer_index, inputs=inputs, model=model, teacher_model=teacher_model)

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
                beta = 1
                loss += alpha*kd_loss + beta*featurekd_loss
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
            replaced_layer_index=replaced_layer_index,
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
        "--model_save_dir", type=str, default="./checkpoints/feature_kd", help="saving model to this directory"
    )
    args = parser.parse_args()

    main(args)
