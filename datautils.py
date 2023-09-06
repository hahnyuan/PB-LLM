import os
import numpy as np
import torch
from datasets import load_dataset
import random

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(
        tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count()
    )
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


def get_wikitext2(nsamples, seed, seqlen, model, cache_dir):
    print("get_wikitext2")
    from datasets import load_dataset

    traindata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        cache_dir=f"{cache_dir}/wikitext/",
        split="train",
    )
    testdata = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        cache_dir=f"{cache_dir}/wikitext/",
        split="test",
    )

    from transformers import AutoTokenizer

    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model, cache_dir=cache_dir, use_fast=False
        )

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, model, cache_dir):
    print("get_ptb")
    from datasets import load_dataset

    traindata = load_dataset(
        "ptb_text_only",
        "penn_treebank",
        cache_dir=f"{cache_dir}/ptb_text_only/",
        split="train",
    )
    valdata = load_dataset(
        "ptb_text_only",
        "penn_treebank",
        cache_dir=f"{cache_dir}/ptb_text_only/",
        split="validation",
    )

    from transformers import AutoTokenizer

    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model, cache_dir=cache_dir, use_fast=False
        )

    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptq_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3):
    print(f" get_ptq_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = (
        f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}.pt"
    )
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    torch.save(traindataset, cache_file)
    return traindataset


def get_c4(nsamples, seed, seqlen, model, cache_dir):
    print("get_c4")

    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        cache_dir=f"{cache_dir}/allenai--c4/",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        cache_dir=f"{cache_dir}/allenai--c4/",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    from transformers import AutoTokenizer

    if "llama" in model:
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model, cache_dir=cache_dir, use_fast=False
        )

    import random

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    # class TokenizerWrapper:
    #     def __init__(self, input_ids):
    #         self.input_ids = input_ids
    # valenc = TokenizerWrapper(valenc)

    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="", cache_dir=""):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, cache_dir)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model, cache_dir)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, cache_dir)
    if "mix" in name:
        wiki_train, wiki_val = get_wikitext2(
            nsamples // 3 + (nsamples - nsamples // 3 * 3),
            seed,
            seqlen,
            model,
            cache_dir,
        )
        ptb_train, ptb_val = get_ptb(nsamples // 3, seed, seqlen, model, cache_dir)
        c4_train, c4_val = get_c4(nsamples // 3, seed, seqlen, model, cache_dir)
        train = wiki_train + ptb_train + c4_train
        val = None
        return train, val
