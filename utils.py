import torch
import torch.nn as nn
import torch.cuda
import quant
import json
import os


def print_memory_usage():
    mem = torch.cuda.memory_allocated()
    print(f"memory_allocated: {mem / 1024 / 1024} MB")


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


def prepare_model_for_training(model):
    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    return model


def prepare_model_for_eval(model):
    model.eval()

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    for param in model.parameters():
        param.data = param.data.to(torch.float16)
    return model


def get_bnn_meta(model):
    meta = {}
    for name, module in model.named_modules():
        if isinstance(module, quant.BinaryInterface):
            meta[name] = module.__class__.__name__
    return meta


def get_bnn_weights(model):
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, quant.BinaryInterface):
            layer_weight_dict = module.get_save_weight_dict()
            layer_weight_dict = {
                name + "_" + k: v for k, v in layer_weight_dict.items()
            }
            weights.update(layer_weight_dict)
            # weights[name] = module.weight.data.half().cpu()
            # weights[name + "_bias"] = module.bias
    return weights


def save_bnn(model, save_path):
    print(f"saving bnn model to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    meta = get_bnn_meta(model)
    weights = get_bnn_weights(model)
    json.dump(meta, open(save_path + "/meta.json", "w"))
    torch.save(weights, save_path + "/weights.pth")


def load_bnn(model, load_path):
    print(f"loading bnn model from {load_path}")
    bnn_meta = json.load(open(load_path + "/meta.json", "r"))
    bnn_weights = torch.load(load_path + "/weights.pth")
    print(bnn_weights.keys())

    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            # choose binariztaion method
            if name in bnn_meta:
                binarization_method = bnn_meta[name]
                weight = bnn_weights[name + "_weight"]
                # weight = bnn_weights[name]
                bias = bnn_weights[name + "_bias"]
                # weight=weight.to(module.weight.device)
                # if bias is not None:
                #     bias=bias.to(module.weight.device)
                qlinear = getattr(quant, binarization_method)(weight, bias)

                setattr(father, name[ind + 1 :], qlinear)
                print(f"replace layer {name} with {qlinear}")
    return model


def generate_sample_test(model, tokenizer):
    # generate a sample
    # prompt = "Hey, are you conscious? Can you talk to me?"
    prompt = "Hey, is llama the best language model?"
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=60)
    outputs = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(outputs)
