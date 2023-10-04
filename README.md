# Binary Quantization for Large Language Models

This work explores network binarization, a radical form of quantization, compressing model weights to a single bit, specifically for Large Language Models (LLMs) compression. 
Due to previous binarization methods collapsing LLMs, we propose a novel approach, Partially-Binarized LLM (PB-LLM), which can achieve extreme low-bit quantization while maintaining the linguistic reasoning capacity of quantized LLMs. 
Specifically, our exploration first uncovers the ineffectiveness of naïve applications of existing binarization algorithms and highlights the imperative role of salient weights in achieving low-bit quantization. 
Thus, PB-LLM filters a small ratio of salient weights during binarization, allocating them to higher-bit storage, i.e. partially-binarization. 
PB-LLM is extended to recover the capacities of quantized LMMs, by analyzing from the perspective of post-training quantization (PTQ) and quantization-aware training (QAT). 
Under PTQ, combining the concepts from GPTQ, we reconstruct the binarized weight matrix guided by the Hessian matrix and successfully recover the reasoning capacity of PB-LLM in low-bit. 
Under QAT, we freeze the salient weights during training, explore the derivation of optimal scaling factors crucial for minimizing the quantization error, and propose a scaling mechanism based on this derived scaling strategy for residual binarized weights. 
Those explorations and the developed methodologies significantly contribute to rejuvenating the performance of low-bit quantized LLMs and present substantial advancements in the field of network binarization for LLMs. 
The paper is available at [paper](https://arxiv.org/abs/2310.00034).


## Model support

Huggingface models
- facebook/opt-125m
- facebook/opt-1.3b
- huggyllama/llama-7b
- huggyllama/llama-13b
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- openchat/openchat_v3.2

## Usage

### Enviroment Setting

```shell
conda create -n binary_llm python=3.10 pip
pip install torch transformers lm_eval accelerate tensorboardX bitsandbytes sentencepiece
```
Note python version must>=3.10

### Training

Run the script with the desired arguments:

```shell
python train_model.py --model_id <pretrained_model_id> --dataset <dataset_name> [--debug]
```

Arguments:
- model_id: Pretrained model ID (default: "facebook/opt-350m")
- dataset: Dataset name (default: "Abirate/english_quotes")
- debug: Enable debug mode (optional)

Example: binarizing via Xnor algorithm with the guidance of KD.

```shell
CUDA_VISIBLE_DEVICES=5 python bnn_train_layerwise_w_KD.py --binarization_method='xnor' 
```
