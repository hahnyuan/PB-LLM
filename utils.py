import torch
import torch.nn as nn
import torch.cuda
import quant

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

def prepare_model_for_eval(model,bnn_meta=None):
    model.eval()

    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            #choose binariztaion method
            if name in bnn_meta:
                binarization_method=bnn_meta[name]
                qlinear=getattr(quant,binarization_method)(module.weight, module.bias)

                setattr(father, name[ind + 1 :], qlinear)
                print(f"replace layer {name} with {qlinear}")

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    for param in model.parameters():
        param.data = param.data.to(torch.float16)
    return model

def get_bnn_meta(model):
    meta={}
    for name,module in model.named_modules():
        if isinstance(module,quant.BinaryInterface):
            meta[name] = module.__class__.__name__
    return meta