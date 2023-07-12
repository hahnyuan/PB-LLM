import argparse
import copy
import os

import torch
import torch.nn as nn
import time

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
    LlamaTokenizer,
    LlamaForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline,
    AutoConfig,
)
import quant
from utils import prepare_model_for_eval, load_bnn
import torch.nn.functional as F
from evaluate import evaluate_model


def main(args):
    tokenizer = LlamaTokenizer.from_pretrained(args.model_id)

    bnn_meta_path = os.path.join(args.checkpoint, "meta.json")
    if os.path.exists(bnn_meta_path):
        model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        load_bnn(model, args.checkpoint)
    else:
        model = LlamaForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model = prepare_model_for_eval(model)
    results = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        args.tasks,
        limit=args.eval_limit,
        batch_size=args.eval_batch_size,
        num_fewshot=args.eval_num_fewshot,
    )
    acc_values = [item["acc"] for item in results["results"].values() if "acc" in item]
    if len(acc_values) > 0:
        mean_acc = sum(acc_values) / len(acc_values)
    else:
        mean_acc = None
    with open("outputs/evaluate_result.log", "a+") as f:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{date_time} {args.model_id} {args.checkpoint}\n {results['results']}\n mean_acc={mean_acc}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "--model_id",
        type=str,
        default="huggyllama/llama-7b",
        help="Pretrained model ID, huggyllama/llama-7b, openlm-research/open_llama_7b",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="to-be-evaluated bnn checkpoint dir, default is empty(raw net)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="boolq",
        help="evaluate tasks name, can be tasks separated by , lambada_openai,piqa,arc_easy,arc_challenge,openbookqa, boolq",
    )
    parser.add_argument(
        "--eval_limit",
        default=-1,
        type=int,
        help="number of test samples for debug, set to -1 is no limit",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=32,
        type=int,
        help="eval batch size, default is 32",
    )
    parser.add_argument(
        "--eval_num_fewshot",
        default=5,
        type=int,
        help="mmlu eval number of few-shot, default is 5",
    )

    args = parser.parse_args()

    main(args)
