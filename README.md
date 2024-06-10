# LLM training


## Introduction

Generative AI (GAI) offers unprecedented opportunities for research and innovation, but its commercialization has raised concerns about transparency, reproducibility, and safety. Many open GAI models lack the necessary components for full understanding and reproducibility, and some use restrictive licenses whilst claiming to be “open-source”. To address these concerns, we follow the [Model Openness Framework (MOF)](https://arxiv.org/pdf/2403.13784), a ranked classification system that rates machine learning models based on their completeness and openness, following principles of open science, open source, open data, and open access. 

By promoting transparency and reproducibility, the MOF combats “openwashing” practices and establishes completeness and openness as primary criteria alongside the core tenets of responsible AI. Wide adoption of the MOF will foster a more open AI ecosystem, benefiting research, innovation, and adoption of state-of-the-art models. 

We follow MOF to release the datasets during training, the training scripts, and the trained models. 



## Environment


## Datasets

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

## Usage


## Timeline


Timeline for OminiX for Stable Diffusion development and open-source

| Time          	| Task                                                                           	| Open Source version                                                   	|
|---------------	|--------------------------------------------------------------------------------	|-----------------------------------------------------------------------	|
| 05/21         	|                                                                                	| The first open source: Current version of the LLM training checkpoint 	|
| 05/21 – 07/15 	| Distributed training of the LLM                                                	| 07/16 Training checkpoint, data, and evaluation results               	|
| 06/15 – 07/15 	| Schedule with Linux Foundation about PR                                        	| Schedule with Linux Foundation                                        	|
| 07/16         	| Reinformcement learning based finetuning, distillation for smaller model, etc. 	| TBM                                                                   	|

