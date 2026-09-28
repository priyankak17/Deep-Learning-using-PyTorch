# Deep Learning using PyTorch

A hands on deep learning portfolio focused on understanding model internals, training behavior, memory and precision tradeoffs, and practical deployment workflows with PyTorch and the Hugging Face ecosystem.

This repository progresses from PyTorch fundamentals into diffusion model training, Stable Diffusion fine tuning experiments, and parameter efficient fine tuning of a 7B parameter language model with LoRA.

The notebooks are intentionally experiment driven. Alongside model training, they document architecture choices, optimization settings, inference behavior, checkpointing strategies, mixed precision, and model publishing to the Hugging Face Hub.

## Portfolio highlights

| Area | What is demonstrated |
| --- | --- |
| PyTorch foundations | Tensor construction, tensor dimensions, shapes, and core deep learning concepts |
| Diffusion models | Training custom UNet based diffusion models and working with DDPM noise schedules |
| Few shot generative modeling | Training a custom 256 × 256 UNet2D diffusion model on a small 272 example dataset |
| Stable Diffusion | Fine tuning Stable Diffusion v1.5 on Naruto image caption data |
| Systems experimentation | Controlled FP32, FP16, and FP16 plus gradient checkpointing comparison |
| Parameter efficient fine tuning | LoRA adaptation of BLOOM 7B1 while training only about 0.11% of parameters |
| Hugging Face | Publishing Diffusers pipelines, PEFT adapters, model cards, checkpoints, and Safetensors artifacts |

## Repository walkthrough

### 1. PyTorch deep learning overview

[`PyTorch_DL_Overview_00.ipynb`](./PyTorch_DL_Overview_00.ipynb)

Introduces the conceptual foundation for the rest of the repository.

The notebook covers the distinction between traditional machine learning and deep learning, common neural network families, supervised and unsupervised learning, transfer learning, tensors, and some practical limitations of deep learning such as explainability, data requirements, and unpredictable failure modes.

This notebook establishes the vocabulary and motivation used in the later implementation notebooks.

### 2. PyTorch fundamentals

[`PyTorch_Basics_01.ipynb`](./PyTorch_Basics_01.ipynb)

A practical introduction to PyTorch tensors.

The notebook works through scalar, vector, matrix, and higher dimensional tensor construction, then inspects tensor rank, shape, indexing, and random tensors. It is intended as the low level PyTorch foundation for the training code used later in the repository.

### 3. Baseline UNet diffusion model training

[`basic_unet_diffusion_model_training.ipynb`](./basic_unet_diffusion_model_training.ipynb)

Builds and trains a compact UNet based diffusion model and follows the complete diffusion workflow from adding noise through iterative denoising and image sampling.

The notebook is focused on understanding the mechanics of diffusion training rather than treating a generative pipeline as a black box. It provides the baseline used for later architecture and training experiments.

Hugging Face model:

[pynk17/my-small_diff-model](https://huggingface.co/pynk17/my-small_diff-model)

### 4. Modified UNet diffusion experiment

[`modified_unet_diffusion_model_training.ipynb`](./modified_unet_diffusion_model_training.ipynb)

Extends the baseline diffusion experiment by modifying the UNet configuration and training setup.

This notebook is part of the progression from a small baseline architecture toward larger and more deliberate diffusion experiments, with attention to model capacity, denoising behavior, optimization, and generated samples.

### 5. Few shot UNet2D DDPM experiment

[`pynk_fewshot_unet2d_diff_model.ipynb`](./pynk_fewshot_unet2d_diff_model.ipynb)

A larger custom diffusion experiment using Hugging Face Diffusers `UNet2DModel` and `DDPMScheduler`.

The experiment trains at 256 × 256 resolution on 272 examples and uses a six stage UNet with spatial attention, 1,000 diffusion timesteps, the `squaredcos_cap_v2` beta schedule, BF16 mixed precision, gradient accumulation, periodic sampling, and checkpointing.

Key configuration:

| Setting | Value |
| --- | ---: |
| Resolution | 256 × 256 |
| Training examples | 272 |
| Train batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Epochs | 100 |
| Peak learning rate | 3e-4 |
| Warmup steps | 500 |
| Mixed precision | BF16 |
| DDPM timesteps | 1,000 |

Hugging Face model:

[pynk17/pk_fewshot-aur-model](https://huggingface.co/pynk17/pk_fewshot-aur-model)

### 6. Stable Diffusion v1.5 fine tuning and systems comparison

[`Stable_diff_fine_tuning.ipynb`](./Stable_diff_fine_tuning.ipynb)

Fine tunes Stable Diffusion v1.5 on the `lambdalabs/naruto-blip-captions` dataset and uses the experiment to compare training configurations rather than only produce a single model.

Three related runs are examined:

| Experiment | Training configuration | Purpose |
| --- | --- | --- |
| A | FP32 | Baseline |
| B | FP16 mixed precision | Precision comparison |
| C | FP16 plus gradient checkpointing | Memory oriented training configuration |

The notebook uses a shared inference setup and prompt suite to compare the resulting pipelines under controlled conditions.

Core training configuration:

| Setting | Value |
| --- | ---: |
| Base model | Stable Diffusion v1.5 |
| Dataset | lambdalabs/naruto-blip-captions |
| Resolution | 512 × 512 |
| Train batch size | 1 |
| Gradient accumulation | 4 |
| Maximum training steps | 100 |
| Learning rate | 1e-5 |
| LR scheduler | constant |
| Seed | 42 |

Published Hugging Face models:

[FP32 baseline](https://huggingface.co/pynk17/sd-naruto-fp32-baseline)

[FP16](https://huggingface.co/pynk17/sd-naruto-fp16)

[FP16 plus gradient checkpointing](https://huggingface.co/pynk17/sd-naruto-fp16-grad-checkpointing)

### 7. BLOOM 7B1 LoRA fine tuning with PEFT

[`PEFT_Finetune_Bloom7B_LoRa.ipynb`](./PEFT_Finetune_Bloom7B_LoRa.ipynb)

Demonstrates parameter efficient fine tuning of `bigscience/bloom-7b1` using Hugging Face PEFT and LoRA.

The base model is frozen and the notebook trains only LoRA adapter parameters. Gradient checkpointing is enabled to reduce activation memory requirements, FP16 is used during training, and the resulting adapter is stored as a small Safetensors artifact rather than another full 7B parameter checkpoint.

The training data comes from `Abirate/english_quotes`. Each example is converted into a causal language modeling sequence of the form:

```text
<quote> ->: <tags>
```

LoRA and training configuration:

| Setting | Value |
| --- | ---: |
| Base model | bigscience/bloom-7b1 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Trainable parameters | 7,864,320 |
| Total parameters | 7,076,880,384 |
| Trainable percentage | ~0.111% |
| Per device batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Maximum steps | 200 |
| Warmup steps | 100 |
| Learning rate | 2e-4 |
| Precision | FP16 |
| Gradient checkpointing | Enabled |
| Training runtime | 86.569 s |
| Final reported aggregate train loss | 2.3122 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |

The resulting adapter artifact is approximately 31.5 MB.

Hugging Face adapter:

[pynk17/pkbloom-7b1-lora](https://huggingface.co/pynk17/pkbloom-7b1-lora)

## Engineering themes

This repository is designed to show more than successful notebook execution.

It explores how model architecture, numerical precision, gradient accumulation, gradient checkpointing, scheduler choices, parameter efficient fine tuning, and checkpoint format affect the practical process of training modern deep learning systems.

The diffusion experiments move from a small UNet baseline to a larger custom DDPM configuration and then to Stable Diffusion fine tuning.

The language model experiment extends the same systems perspective to large model adaptation, where LoRA reduces the trainable parameter count from more than 7 billion parameters to roughly 7.9 million.

## Hugging Face portfolio

The corresponding model artifacts are published under:

[pynk17 on Hugging Face](https://huggingface.co/pynk17)

Current experiments represented in this repository include:

| Model | Experiment |
| --- | --- |
| [my-small_diff-model](https://huggingface.co/pynk17/my-small_diff-model) | Small diffusion baseline |
| [pk_fewshot-aur-model](https://huggingface.co/pynk17/pk_fewshot-aur-model) | Custom few shot UNet2D DDPM |
| [sd-naruto-fp32-baseline](https://huggingface.co/pynk17/sd-naruto-fp32-baseline) | Stable Diffusion FP32 baseline |
| [sd-naruto-fp16](https://huggingface.co/pynk17/sd-naruto-fp16) | Stable Diffusion FP16 |
| [sd-naruto-fp16-grad-checkpointing](https://huggingface.co/pynk17/sd-naruto-fp16-grad-checkpointing) | Stable Diffusion FP16 with gradient checkpointing |
| [pkbloom-7b1-lora](https://huggingface.co/pynk17/pkbloom-7b1-lora) | BLOOM 7B1 LoRA adapter |

## Technology stack

| Category | Tools |
| --- | --- |
| Core ML | PyTorch |
| Generative vision | Hugging Face Diffusers, Stable Diffusion, DDPM, UNet |
| LLM fine tuning | Transformers, PEFT, LoRA |
| Training infrastructure | Accelerate, gradient accumulation, gradient checkpointing, mixed precision |
| Data | Hugging Face Datasets |
| Model distribution | Hugging Face Hub, Safetensors |
| Experimentation | Jupyter, Google Colab, TensorBoard |
| GPU compute | CUDA |

## Reproducibility and documentation

Where available, the notebooks retain model configuration, training hyperparameters, loss traces, generated outputs, inference examples, checkpointing behavior, and publishing steps.

The corresponding Hugging Face repositories contain model cards describing intended use, training configuration, limitations, and loading instructions.

This is an evolving portfolio. Some early notebooks are intentionally foundational, while later notebooks place more emphasis on experimental design, reproducibility, systems tradeoffs, and deployable model artifacts.

## Repository structure

```text
Deep-Learning-using-PyTorch/
│
├── PyTorch_DL_Overview_00.ipynb
├── PyTorch_Basics_01.ipynb
│
├── basic_unet_diffusion_model_training.ipynb
├── modified_unet_diffusion_model_training.ipynb
├── pynk_fewshot_unet2d_diff_model.ipynb
│
├── Stable_diff_fine_tuning.ipynb
├── PEFT_Finetune_Bloom7B_LoRa.ipynb
│
├── Full_fine-tuning/
└── README.md
```

## About

Built as a practical machine learning engineering portfolio spanning deep learning foundations, generative vision, large language model adaptation, optimization, and model publishing.

Author: Priyanka

GitHub: [priyankak17](https://github.com/priyankak17)

Hugging Face: [pynk17](https://huggingface.co/pynk17)
