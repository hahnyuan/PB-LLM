# PB-LLM: Partially Binarized Large Language Models    
**[Yuzhang Shang*](https://42shawn.github.io/), [Zhihang Yuan*](http://hahnyuan.com/), Qiang Wu, [Zhen Dong](https://dong-zhen.com/)** (* Equal Contribution)     

This work explores network binarization, a radical form of quantization, compressing model weights to a single bit, specifically for Large Language Models (LLMs) compression. 
Due to previous binarization methods collapsing LLMs, we propose a novel approach, Partially-Binarized LLM (PB-LLM), which can achieve extreme low-bit quantization while maintaining the linguistic reasoning capacity of quantized LLMs. 
Specifically, our exploration first uncovers the ineffectiveness of naïve applications of existing binarization algorithms and highlights the imperative role of salient weights in achieving low-bit quantization. 
Thus, PB-LLM filters a small ratio of salient weights during binarization, allocating them to higher-bit storage, i.e. partially-binarization. 
PB-LLM is extended to recover the capacities of quantized LMMs, by analyzing from the perspective of post-training quantization (PTQ) and quantization-aware training (QAT). 
Under PTQ, combining the concepts from GPTQ, we reconstruct the binarized weight matrix guided by the Hessian matrix and successfully recover the reasoning capacity of PB-LLM in low-bit. 
Under QAT, we freeze the salient weights during training, explore the derivation of optimal scaling factors crucial for minimizing the quantization error, and propose a scaling mechanism based on this derived scaling strategy for residual binarized weights. 
Those explorations and the developed methodologies significantly contribute to rejuvenating the performance of low-bit quantized LLMs and present substantial advancements in the field of network binarization for LLMs. 
The paper is available at [arxiv](https://arxiv.org/abs/2310.00034).


## Tested Models

Huggingface models
- facebook/opt-125m
- facebook/opt-1.3b
- facebook/opt-6.7b
- huggyllama/llama-7b
- huggyllama/llama-13b

## Usage

### Environment Setting

If you use conda, you can create a new environment and install the dependencies with the following commands:
```shell
conda create -n binary_llm python=3.10 pip
```

Install the python dependencies:
```shell
pip install torch transformers lm_eval accelerate tensorboardX bitsandbytes sentencepiece
```
Note python version must>=3.10

### PTQ (GPTQ-PB)

The GPTQ-PB is implemented in the [gptq_pb](gptq_pb) folder.
Please go to the folder and run the script with the desired arguments:
```
usage: run.py [-h] [--plot] [--load_quantized] [--seed SEED] [--nsamples NSAMPLES] [--percdamp PERCDAMP] [--low_frac LOW_FRAC] [--blocksize BLOCKSIZE] [--groupsize GROUPSIZE] [--salient_metric {magnitude,hessian}] [--high_bit HIGH_BIT]
              [--minlayer MINLAYER] [--maxlayer MAXLAYER] [--quant_only QUANT_ONLY] [--invert] [--save] [--disable_gptq] [--log_wandb]
              model {wikitext2,ptb,c4} {xnor,sign,no,2bit,4bit,prune}

positional arguments:
  model                 model to load; for example `huggyllama/llama-7b`.
  {wikitext2,ptb,c4}    Where to extract calibration data from.
  {xnor,sign,no,2bit,4bit,prune}
                        quantization method; `xnor` is the method used in paper; `prune` is the method used in sparseGPTQ

--low_frac LOW_FRAC   fraction of binarized weight
--salient_metric {magnitude,hessian}    metric to measure salient weights
```

For example

```shell
cd gptq_pb
# for llama-7b
CUDA_VISIBLE_DEVICES=1 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.5 --high_bit 8 --salient_metric hessian
CUDA_VISIBLE_DEVICES=2 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.8 --high_bit 8 --salient_metric hessian
CUDA_VISIBLE_DEVICES=3 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian
CUDA_VISIBLE_DEVICES=0 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.95 --high_bit 8 --salient_metric hessian
```

### QAT

The QAT for PB-LLM is implemented in the [qat](qat) folder.

For example

```shell
# Testing for debug
CUDA_VISIBLE_DEVICES='0' python qat/run_qat.py --binarization_method=xnor_outlier --model_id=facebook/opt-125m --train_step=20 --dataset=red_pajama --outlier_fraction 0.1
# Evaluate
CUDA_VISIBLE_DEVICES='0' python qat/eval_after_qat.py outputs/facebook/opt-125m/xnor_outlier_0.1_20 --model_id=facebook/opt-125m


# for opt-1.3b
CUDA_VISIBLE_DEVICES='1' python qat/run_qat.py --binarization_method=xnor_outlier --model_id=facebook/opt-1.3b --train_step=10000 --dataset=red_pajama --outlier_fraction 0.1
# Evaluate
CUDA_VISIBLE_DEVICES='1' python qat/eval_after_qat.py outputs/facebook/opt-1.3b/xnor_outlier_0.1_10000 --model_id=facebook/opt-1.3b

# hessian based outlier
CUDA_VISIBLE_DEVICES='2' python qat/run_qat.py --binarization_method=xnor_outlier_hessian --model_id=facebook/opt-1.3b --train_step=10000 --dataset=red_pajama --outlier_fraction 0.1
CUDA_VISIBLE_DEVICES='2' python qat/eval_after_qat.py outputs/facebook/opt-1.3b/xnor_outlier_hessian_0.1_10000 --model_id=facebook/opt-1.3b


```
