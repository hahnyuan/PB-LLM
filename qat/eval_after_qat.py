import sys

sys.path.append(".")
import argparse
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate import evaluate_model


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.path, device_map="auto", torch_dtype=torch.float16
    )
    breakpoint()

    # Quick evaluate
    result = evaluate_model(model, tokenizer, args.model_id, "piqa,boolq", limit=100)
    boolq = result["boolq"]["acc"]
    piqa = result["piqa"]["acc"]
    print(f"boolq: {boolq:.4f}, piqa: {piqa:.4f}, avg: {(boolq+piqa)/2:.4f}")

    evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "llmqat",
        limit=200,
        eval_ppl="wikitext2,ptb,c4",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Script")
    parser.add_argument(
        "path",
        type=str,
        help="Saved model path",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-350m",
        help="Pretrained model ID",
    )
    args = parser.parse_args()

    main(args)
