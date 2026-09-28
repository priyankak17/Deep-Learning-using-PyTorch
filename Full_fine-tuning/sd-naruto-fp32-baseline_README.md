---
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
datasets:
  - lambdalabs/naruto-blip-captions
pipeline_tag: text-to-image
library_name: diffusers
tags:
  - stable-diffusion
  - text-to-image
  - diffusers
  - pytorch
  - safetensors
  - tensorboard
  - fp32
  - experimental
---

# sd-naruto-fp32-baseline

`sd-naruto-fp32-baseline` is an experimental Stable Diffusion v1.5 fine-tune on the `lambdalabs/naruto-blip-captions` dataset.

This repository is the **FP32 baseline** in a three-run engineering comparison covering:

1. FP32 training
2. FP16 mixed-precision training
3. FP16 training with gradient checkpointing

The goal of this baseline is to provide a reference point for comparing model behavior, inference output, precision choices, and memory-oriented training optimizations.

## Model Details

### Model Description

The model was fine-tuned from `stable-diffusion-v1-5/stable-diffusion-v1-5` using the Hugging Face Diffusers `train_text_to_image.py` training workflow.

The training run used the standard Stable Diffusion text-to-image components, including the VAE, text encoder, tokenizer, conditional UNet, scheduler, safety checker, and feature extractor.

- **Developed by:** [pynk17](https://huggingface.co/pynk17)
- **Model type:** Stable Diffusion text-to-image latent diffusion model
- **Training precision:** FP32 baseline (`mixed precision type: no`)
- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **Dataset:** `lambdalabs/naruto-blip-captions`
- **Framework:** PyTorch / Hugging Face Diffusers
- **Pipeline:** `StableDiffusionPipeline`
- **Model repository:** `pynk17/sd-naruto-fp32-baseline`

### Repository Contents

The training output includes the standard Stable Diffusion pipeline components:

- `unet`
- `vae`
- `text_encoder`
- `tokenizer`
- `scheduler`
- `safety_checker`
- `feature_extractor`
- `model_index.json`
- TensorBoard `logs`
- `checkpoint-50`

The `checkpoint-50/unet` directory contains the UNet configuration and Safetensors weights.

## Uses

### Direct Use

The model can be used for experimental text-to-image generation with Naruto-style training data.

It is particularly useful as the FP32 reference model for comparing:

- FP32 versus FP16 training
- memory usage across precision configurations
- gradient checkpointing effects
- inference behavior across otherwise similar fine-tuning runs

### Out-of-Scope Use

This model is an experimental fine-tune and has not been validated for production deployment.

It should not be assumed to provide:

- reliable photorealism
- robust prompt following outside the training distribution
- unbiased outputs
- production-level safety guarantees
- consistent anatomical or scene-level accuracy

## How to Get Started with the Model

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "pynk17/sd-naruto-fp32-baseline"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
)

pipe = pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt="a ninja character with blue armor and spiky hair",
    negative_prompt="blurry, low quality, distorted",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=256,
    width=256,
    generator=generator,
).images[0]

image.save("sd_naruto_fp32_sample.png")
```

## Training Details

### Training Data

The model was fine-tuned on:

`lambdalabs/naruto-blip-captions`

The Diffusers training script used:

```text
image_column = "image"
caption_column = "text"
```

### Training Procedure

Training was performed with the Hugging Face Diffusers `train_text_to_image.py` example using Stable Diffusion v1.5 as the pretrained starting point.

The FP32 baseline was launched without mixed precision.

### Training Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Dataset | `lambdalabs/naruto-blip-captions` |
| Resolution | 512 × 512 |
| Train batch size | 1 |
| Gradient accumulation steps | 4 |
| Maximum training steps | 100 |
| Learning rate | `1e-5` |
| LR scheduler | `constant` |
| LR warmup steps | 0 |
| Random seed | 42 |
| Mixed precision | None / FP32 |
| Accelerator processes | 1 |
| Device | CUDA |

A checkpoint was saved at training step 50.

## Evaluation

### Evaluation Procedure

Evaluation in the experiment notebook was primarily qualitative and comparative.

The same prompt set was run across the FP32, FP16, and FP16 + gradient-checkpointing models using a controlled inference configuration.

For the comparison run, the notebook used:

```text
seed = 42
num_inference_steps = 30
guidance_scale = 7.5
height = 256
width = 256
negative_prompt = "blurry, low quality, distorted"
```

The five comparison prompts were:

1. `a ninja character with blue armor and spiky hair`
2. `a female ninja character standing in a forest`
3. `a ninja character using a fire attack`
4. `a ninja character wearing black clothes at night`
5. `a detailed portrait of a ninja character`

### Results

The FP32 baseline successfully loaded through `StableDiffusionPipeline` and generated images during the comparison run.

Observed inference times for the five prompts were approximately:

| Prompt | Inference time |
|---:|---:|
| 1 | 1.37 s |
| 2 | 1.35 s |
| 3 | 1.36 s |
| 4 | 1.37 s |
| 5 | 1.41 s |

These timings came from the specific notebook environment and should not be treated as hardware-independent benchmark results.

No FID, KID, CLIP score, or other formal generative quality metric was reported in this experiment.

## Bias, Risks, and Limitations

The model inherits limitations from both Stable Diffusion v1.5 and the Naruto-caption fine-tuning dataset.

The experiment did not include a dedicated bias, memorization, or safety evaluation.

The Stable Diffusion safety checker remains part of the saved pipeline. During comparative inference, the notebook showed that the safety checker can replace a generated result with a black image when potential NSFW content is detected.

Generated outputs should therefore be treated as experimental samples rather than validated production outputs.

## Technical Specifications

### Model Architecture and Objective

This model uses the Stable Diffusion v1.5 latent diffusion architecture.

The fine-tuning workflow initializes the model components from the pretrained Stable Diffusion v1.5 checkpoint, including:

- `AutoencoderKL`
- `UNet2DConditionModel`
- text encoder
- tokenizer
- diffusion scheduler

The conditional UNet is optimized for the text-to-image denoising objective using image-caption pairs from the Naruto BLIP captions dataset.

### Compute Infrastructure

The run used one CUDA process.

The notebook experiment was executed in a GPU environment, and the broader experiment notebook also documents testing on an NVIDIA A100 environment.

Because the exact GPU used for every logged FP32 training segment is not unambiguously established by the retained output, the model card does not claim a specific GPU model for this training run.

## Reproducibility Notes

The key baseline configuration is:

```bash
accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42
```

This repository serves as the FP32 reference point for the related precision and memory-optimization experiments.

## Model Card Author

Priyanka / [pynk17](https://huggingface.co/pynk17)
