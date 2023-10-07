# for llama-7b
CUDA_VISIBLE_DEVICES=1 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.5 --high_bit 8 --salient_metric hessian
CUDA_VISIBLE_DEVICES=2 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.8 --high_bit 8 --salient_metric hessian
CUDA_VISIBLE_DEVICES=3 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian
CUDA_VISIBLE_DEVICES=0 python run.py huggyllama/llama-7b c4 xnor --low_frac 0.95 --high_bit 8 --salient_metric hessian

# for sparseGPT 
# CUDA_VISIBLE_DEVICES=1 python run.py huggyllama/llama-7b c4 prune --low_frac 0.5 --high_bit 8
# for GPTQ 
# CUDA_VISIBLE_DEVICES=2 python run.py huggyllama/llama-7b c4 prune --low_frac 0.001 --high_bit 4


# experiments on opt-1.3b
# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.5 --high_bit 8
# CUDA_VISIBLE_DEVICES=1 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.8 --high_bit 8
# CUDA_VISIBLE_DEVICES=2 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.9 --high_bit 8
# CUDA_VISIBLE_DEVICES=3 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.95 --high_bit 8

# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.5 --high_bit 8  --salient_metric hessian
# CUDA_VISIBLE_DEVICES=1 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.8 --high_bit 8 --salient_metric hessian
# CUDA_VISIBLE_DEVICES=2 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian
# CUDA_VISIBLE_DEVICES=3 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.95 --high_bit 8 --salient_metric hessian

# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.5 --high_bit 8  --salient_metric hessian --groupsize 128
# CUDA_VISIBLE_DEVICES=1 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.8 --high_bit 8 --salient_metric hessian --groupsize 128
# CUDA_VISIBLE_DEVICES=2 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian --groupsize 128
# CUDA_VISIBLE_DEVICES=3 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.95 --high_bit 8 --salient_metric hessian --groupsize 128

# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.5 --high_bit 8  --groupsize 128
# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.8 --high_bit 8 --groupsize 128
# CUDA_VISIBLE_DEVICES=1 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.9 --high_bit 8 --groupsize 128
# CUDA_VISIBLE_DEVICES=3 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.95 --high_bit 8 --groupsize 128

# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.5 --high_bit 8 --disable_gptq 
# CUDA_VISIBLE_DEVICES=1 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.8 --high_bit 8 --disable_gptq
# CUDA_VISIBLE_DEVICES=2 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.9 --high_bit 8 --disable_gptq
# CUDA_VISIBLE_DEVICES=3 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.95 --high_bit 8 --disable_gptq

# CUDA_VISIBLE_DEVICES=0 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.5 --high_bit 8 --disable_gptq  --salient_metric hessian
# CUDA_VISIBLE_DEVICES=1 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.8 --high_bit 8 --disable_gptq --salient_metric hessian
# CUDA_VISIBLE_DEVICES=2 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.9 --high_bit 8 --disable_gptq --salient_metric hessian
# CUDA_VISIBLE_DEVICES=3 python run.py facebook/opt-1.3b c4 xnor --low_frac 0.95 --high_bit 8 --disable_gptq --salient_metric hessian