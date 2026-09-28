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
  - fp16
  - gradient-checkpointing
  - mixed-precision
  - experimental
---

# sd-naruto-fp16-grad-checkpointing

`sd-naruto-fp16-grad-checkpointing` is an experimental Stable Diffusion v1.5 fine-tune on the `lambdalabs/naruto-blip-captions` dataset using **FP16 mixed precision with gradient checkpointing enabled**.

This repository is **Experiment C** in a three-run engineering comparison:

1. FP32 baseline
2. FP16 mixed precision
3. FP16 mixed precision with gradient checkpointing

The purpose of this run is to evaluate a memory-oriented training configuration while keeping the main fine-tuning setup aligned with the FP32 and FP16 experiments.

## Model Details

### Model Description

The model was fine-tuned from `stable-diffusion-v1-5/stable-diffusion-v1-5` using the Hugging Face Diffusers `train_text_to_image.py` workflow.

The run uses the standard Stable Diffusion text-to-image components, including the VAE, text encoder, tokenizer, conditional UNet, scheduler, safety checker, and feature extractor.

- **Developed by:** [pynk17](https://huggingface.co/pynk17)
- **Model type:** Stable Diffusion text-to-image latent diffusion model
- **Training precision:** FP16 mixed precision
- **Memory optimization:** Gradient checkpointing
- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **Dataset:** `lambdalabs/naruto-blip-captions`
- **Framework:** PyTorch / Hugging Face Diffusers
- **Pipeline:** `StableDiffusionPipeline`
- **Model repository:** `pynk17/sd-naruto-fp16-grad-checkpointing`

### Repository Contents

The experiment output contains the standard Stable Diffusion pipeline components:

- `unet`
- `vae`
- `text_encoder`
- `tokenizer`
- `scheduler`
- `safety_checker`
- `feature_extractor`
- `model_index.json`
- TensorBoard `logs`

## Uses

### Direct Use

The model can be used for experimental Naruto-style text-to-image generation.

It is especially useful as the memory-optimized arm of a controlled systems comparison examining:

- FP32 versus FP16 mixed precision
- FP16 with and without gradient checkpointing
- memory-oriented training optimizations
- inference behavior across the resulting pipelines
- trade-offs between training configuration and runtime characteristics

### Out-of-Scope Use

This model is an experimental fine-tune and has not been validated for production deployment.

It should not be assumed to provide:

- robust prompt following outside the fine-tuning distribution
- consistent photorealism
- unbiased outputs
- production-level reliability
- complete safety guarantees
- stable anatomical or scene-level correctness

## How to Get Started with the Model

```python
import torch
from diffusers import StableDiffusionPipeline

model_id = "pynk17/sd-naruto-fp16-grad-checkpointing"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
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

image.save("sd_naruto_fp16_grad_checkpointing_sample.png")
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

Training was performed with the Hugging Face Diffusers `train_text_to_image.py` example.

This experiment used FP16 mixed precision and enabled gradient checkpointing.

The principal difference from Experiment B is the addition of:

```text
--gradient_checkpointing
```

while retaining the same main fine-tuning configuration.

### Training Hyperparameters

| Hyperparameter | Value |
|---|---:|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Dataset | `lambdalabs/naruto-blip-captions` |
| Image column | `image` |
| Caption column | `text` |
| Resolution | 512 × 512 |
| Train batch size | 1 |
| Gradient accumulation steps | 4 |
| Maximum training steps | 100 |
| Learning rate | `1e-5` |
| LR scheduler | `constant` |
| LR warmup steps | 0 |
| Random seed | 42 |
| Mixed precision | FP16 |
| Gradient checkpointing | Enabled |
| Accelerator processes | 1 |
| Device | CUDA |

The effective batch size implied by batch size 1 and four gradient-accumulation steps is approximately **4 samples per optimizer update** for a single process.

## Evaluation

### Evaluation Procedure

The experiment notebook compares the FP32, FP16, and FP16 plus gradient-checkpointing models under the same controlled inference configuration.

The comparison uses:

```text
seed = 42
num_inference_steps = 30
guidance_scale = 7.5
height = 256
width = 256
negative_prompt = "blurry, low quality, distorted"
```

The shared prompt set is:

1. `a ninja character with blue armor and spiky hair`
2. `a female ninja character standing in a forest`
3. `a ninja character using a fire attack`
4. `a ninja character wearing black clothes at night`
5. `a detailed portrait of a ninja character`

### Results

The FP16 plus gradient-checkpointing model successfully loaded through `StableDiffusionPipeline` and participated in the same five-prompt inference comparison as the FP32 and FP16 models.

The notebook reports the following aggregate inference-time result:

```text
C_FP16_GC: 1.40 ± 0.02 sec/image
```

For reference, the same notebook reports:

```text
A_FP32: 1.37 ± 0.02 sec/image
B_FP16: 1.39 ± 0.02 sec/image
```

These measurements reflect the specific notebook runtime and inference configuration. They should not be interpreted as hardware-independent benchmarks.

No FID, KID, CLIP score, or other formal generative-quality metric was reported.

## Bias, Risks, and Limitations

The model inherits limitations from both Stable Diffusion v1.5 and the Naruto BLIP caption fine-tuning dataset.

The experiment did not include a dedicated bias, memorization, or safety evaluation.

The saved Stable Diffusion pipeline includes a safety checker. In the broader comparison notebook, the safety checker is shown replacing a generated result with a black image when potential NSFW content is detected.

Generated outputs should therefore be treated as experimental samples rather than validated production outputs.

## Technical Specifications

### Model Architecture and Objective

This model uses the Stable Diffusion v1.5 latent diffusion architecture.

The training workflow initializes the standard pretrained components, including:

- `AutoencoderKL`
- `UNet2DConditionModel`
- text encoder
- tokenizer
- diffusion scheduler

The conditional UNet is optimized on image-caption pairs from the Naruto BLIP captions dataset.

The distinguishing systems configuration for this repository is **FP16 mixed-precision fine-tuning with gradient checkpointing enabled**.

Gradient checkpointing reduces activation-memory requirements during training by recomputing selected intermediate activations during the backward pass rather than retaining all of them in memory.

## Compute Infrastructure

The training run was launched through Hugging Face Accelerate with FP16 mixed precision and gradient checkpointing enabled.

The notebook records a single CUDA process for the experiment.

The retained notebook output does not unambiguously establish a specific GPU model for every segment of this training run, so no exact GPU model is claimed here.

## Reproducibility Notes

The FP16 plus gradient-checkpointing configuration is:

```bash
accelerate launch \
  --mixed_precision="fp16" \
  train_text_to_image.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=100 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42
```

This repository should be interpreted together with:

- `pynk17/sd-naruto-fp32-baseline`
- `pynk17/sd-naruto-fp16`

Together, the three runs form a controlled experiment around training precision and memory-oriented optimization.

## Model Card Author

Priyanka / [pynk17](https://huggingface.co/pynk17)
