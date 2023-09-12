import argparse
import copy
import os

import torch
import torch.nn as nn
import time
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


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
from utils import prepare_model_for_eval, load_bnn, generate_sample_test
import torch.nn.functional as F
from evaluate import evaluate_model


def main(args):
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    model = AutoGPTQForCausalLM.from_quantized(args.checkpoint, quantize_config,use_low_high=True)

    model = prepare_model_for_eval(model)

    # generate_sample_test(model,tokenizer)

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
        default="boolq,piqa",
        help="evaluate tasks name, can be tasks separated by , llmqat is the evaluation in the llmqat paper",
    )
    parser.add_argument(
        "--eval_limit",
        default=-1,
        type=int,
        help="number of test samples for debug, set to -1 is no limit",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
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
