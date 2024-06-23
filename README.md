# LLM training


## Introduction

Generative AI (GAI) offers unprecedented opportunities for research and innovation, but its commercialization has raised concerns about transparency, reproducibility, and safety. Many open GAI models lack the necessary components for full understanding and reproducibility, and some use restrictive licenses whilst claiming to be “open-source”. To address these concerns, we follow the [Model Openness Framework (MOF)](https://arxiv.org/pdf/2403.13784), a ranked classification system that rates machine learning models based on their completeness and openness, following principles of open science, open source, open data, and open access. 

By promoting transparency and reproducibility, the MOF combats “openwashing” practices and establishes completeness and openness as primary criteria alongside the core tenets of responsible AI. Wide adoption of the MOF will foster a more open AI ecosystem, benefiting research, innovation, and adoption of state-of-the-art models. 

We follow MOF to release the datasets during training, the training scripts, and the trained models. 



## Environment

### 1. Dataset config
To prepare the dataset, it needs to install the following package,
```
pip install datasets
```

### 2. Cuda install

We use cuda 11.7. Other cuda versions may also work.
```
get https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run    
```

### 3. Install pytorch

We use pytorch 2.0.0. 
```
conda create --name llm_train python==3.10
conda activate llm_train
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

### 4. Install other packages

To install other packages, follow the requirements.txt
```
pip install -r requirements.txt
```

### 5. Install flash attention

We use flash-attention 2.2.1.
```
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
git checkout a1576ad                ##  flash-attention 2.2.1
python setup.py  install
cd ./csrc
cd fused_dense_lib  && pip install -v .
cd ../xentropy && pip install -v .
cd ../rotary && pip install -v .
cd ../layer_norm && pip install -v .
```


## Datasets

### 1. Sample dataset

You can do some test on a small dataset, which is a small sample from RedPajama 1T.
```
import datasets 
ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
```

### 2. Large dataset

We use the SlimPajama dataset for pretraining. You can download the dataset using Hugging Face datasets:
```
import datasets 
ds = datasets.load_dataset("cerebras/SlimPajama-627B")
```
SlimPajama is the largest extensively deduplicated, multi-corpora, open-source dataset for training large language models. SlimPajama was created by cleaning and deduplicating the 1.2T token RedPajama dataset from Together. By filtering out low quality data and duplicates, it  removes 49.6% of bytes, slimming down the RedPajama dataset from 1210B to 627B tokens.   SlimPajama offers the highest quality and most compute efficient data to train on for runs up to 627B tokens. When upsampled, SlimPajama is expected   to perform equal to or better than RedPajama-1T when training at trillion token scale. 

In the case that SlimPajama is too large, you can use a smaller split (such as the first 10% data) of the original dataset as follows,
```
train_10pct_ds = datasets.load_dataset("cerebras/SlimPajama-627B", split='train[:10%]')
train_10pct_ds.save_to_disk("Path/to/save")
```
Then, you can load the dataset from the local disk,
```
train_10pct_ds = datasets.load_from_disk("Path/to/save")
```
It is a better method to load from disk as it does not require online data preprecessing with simultaneous API calls due to multiple nodes. 


## Model
You can download our efficient stable diffusion model from this [link](https://huggingface.co/piuzha/llm_ckpts). It is located on Huggingface with 'piuzha/llm_ckpts'.


## Training

We follow the [ColossalAI](https://github.com/hpcaitech/ColossalAI) framework to train the LLM model. Colossal-AI provides a collection of parallel components for the training. It aims to support   to write the distributed deep learning models just like how you write your model on your laptop. It provides user-friendly tools to kickstart distributed training and inference in a few lines. 

We provide a few examples to show how to run benchmark or pretraining based on Colossal-AI. 

### 1. Training LLM

You can find the shell scripts in 'scripts/train_7B' directory. The main command should be in the format of:
```
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
benchmark.py --OTHER_CONFIGURATIONS
```

#### a. Running on a sinlge node
we provide an example to run the training on a single node as below,
```
colossalai run --nproc_per_node 1 pretrain.py \
        --config 7b \
        --dataset togethercomputer/RedPajama-Data-1T-Sample \
        --batch_size 1 \
        --num_epochs 5 \
        --save_interval 5000 \
        --max_length 2048 \
        --save_dir output-checkpoints \
        --plugin zero2_cpu \
        --lr 2e-5 \
        --expanded_model hpcai-tech/Colossal-LLaMA-2-7b-base
```
In the example, it uses the sample dataset 'togethercomputer/RedPajama-Data-1T-Sample' for training. It trains the 7B model 'hpcai-tech/Colossal-LLaMA-2-7b-base'. You can refer the main file 'run.sh' and 'pretrain.py' for more details. 

#### b. Running on a sinlge node

we provide an example to run the training on multiple nodes as below,
```
srun colossalai run --num_nodes 8 --nproc_per_node 8 pretrain.py \
        --config 7b \
        --dataset cerebras/SlimPajama-627B \
        --batch_size 1 \
        --num_epochs 10 \
        --save_interval 50000 \
        --max_length 2048 \
        --save_dir output-checkpoints \
        --flash_attention \
        --plugin zero2_cpu \
        --lr 1e-5 \
        --expanded_model hpcai-tech/Colossal-LLaMA-2-7b-base
```
It uses 8 nodes. Put your host file (`hosts.txt`) in this directory with your real host ip or host name.
Here is a sample `hosts.txt`:
```text
hostname1
hostname2
hostname3few
hostname8
```
You can refer to   the main file 'run-multi-server.sh' and 'pretrain.py' for more details.


### 2. Benchmark


You can find the shell scripts in 'scripts/benchmark_7B' directory. The benchmark mainly test the throughput of the LLM, without actual model training.  The main command should be in the format of:
```
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
benchmark.py --OTHER_CONFIGURATIONS
```

Here we will show an example of how to run training llama pretraining with 'gemini, batch_size=16, sequence_length=4096, gradient_checkpoint=True, flash_attn=True'.

#### a. Running environment

This experiment was performed on 4 computing nodes with 32 L40S GPUs in total for LLaMA-2 7B. The nodes are connected with RDMA and GPUs within one node are fully connected with NVLink. 

#### b. Running command

```bash
cd scripts/benchmark_7B
```

First, put your host file (`hosts.txt`) in this directory with your real host ip or host name.

Here is a sample `hosts.txt`:
```text
hostname1
hostname2
hostname3few
hostname4
```

Then add environment variables to script if needed.

Finally, run the following command to start training:

```bash
bash gemini.sh
```


## Inference



## Timeline


Timeline for OminiX for Stable Diffusion development and open-source

| Time          	| Task                                                                           	| Open Source version                                                   	|
|---------------	|--------------------------------------------------------------------------------	|-----------------------------------------------------------------------	|
| 05/21         	|                                                                                	| The first open source: Current version of the LLM training checkpoint 	|
| 05/21 – 07/15 	| Distributed training of the LLM                                                	| 07/16 Training checkpoint, data, and evaluation results               	|
| 07/16         	| Reinformcement learning based finetuning, distillation for smaller model, etc. 	| TBM                                                                   	|

