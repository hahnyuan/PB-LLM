# Binary Quantization for Large Language Models

This project focuses on binary quantization techniques applied to large language models. Binary quantization is a process of reducing the memory footprint and computational complexity of neural networks by representing weights and activations as binary values (-1 or +1) instead of traditional floating-point numbers. This technique has gained significant attention in the field of deep learning due to its potential to improve efficiency and enable deployment on resource-constrained devices.

## Introduction
The exponential growth of language models has posed challenges in terms of model size, computational requirements, and energy consumption. Binary quantization offers a promising solution to address these challenges by reducing the memory footprint and improving the inference speed of large language models. This project aims to provide an implementation of binary quantization techniques that can be applied to various popular language models.

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
