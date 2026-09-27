# Stable Diffusion Fine-Tuning Efficiency Experiments

## Overview

This experiment series evaluates the effects of mixed precision, gradient checkpointing, and workload sizing on Stable Diffusion v1.5 fine-tuning behavior.

The comparison includes:

1. **Smoke Test**: FP16 + gradient checkpointing, 10 optimization steps
2. **Experiment A**: FP32 baseline, 100 optimization steps
3. **Experiment B**: FP16, 100 optimization steps
4. **Experiment C**: FP16 + gradient checkpointing, 100 optimization steps

All runs used a single CUDA device, batch size 1, an effective batch size of 4 through gradient accumulation, and the same 1,221-example training dataset.

## Experimental Configuration

| Run | Precision | Gradient Checkpointing | Optimization Steps | Batch Size | Effective Batch Size | Gradient Accumulation |
|---|---|---:|---:|---:|---:|---:|
| Smoke Test | FP16 | Yes | 10 | 1 | 4 | 4 |
| Experiment A | FP32 | No | 100 | 1 | 4 | 4 |
| Experiment B | FP16 | No | 100 | 1 | 4 | 4 |
| Experiment C | FP16 | Yes | 100 | 1 | 4 | 4 |

## Training Results

The primary training timing is taken from the progress line at completion of the optimization loop.  
The separate end-to-end value includes post-training model assembly and serialization.

| Run | Train Loop Time | Seconds / Step | Throughput | End to End Time | End to End Sec / Step | Final Displayed Step Loss | Status | Peak GPU VRAM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Smoke Test | ~16 s | 1.40 | ~0.714 steps/s | ~34 s | 3.43 | 0.0131 | Success | Not measured |
| Experiment A | 1 min 40 s | ~0.990 | 1.01 steps/s | 1 min 50 s | 1.11 | 0.106 | Success | Not measured |
| Experiment B | 1 min 33 s | ~0.917 | 1.09 steps/s | 1 min 43 s | 1.04 | 0.106 | Success | Not measured |
| Experiment C | 2 min 21 s | 1.40 | ~0.714 steps/s | 2 min 32 s | 1.52 | 0.106 | Success | Not measured |

### Training behavior

All four runs completed their configured optimization steps on A100 with 80 GB of VRAM without an observed CUDA out-of-memory failure.

The smoke test completed 10 of 10 optimization steps and ended with a displayed `step_loss` of `0.0131`.

Experiments A, B, and C each completed 100 of 100 optimization steps and ended with a displayed `step_loss` of `0.106`.

Only the final displayed step loss is available from these logs, so these results should not be interpreted as a full loss convergence comparison.

## Inference Benchmark

Average inference latency:

| Model | Average Inference Time |
|---|---:|
| A_FP32 | **1.37 ± 0.02 sec/image** |
| B_FP16 | **1.39 ± 0.02 sec/image** |
| C_FP16_GC | **1.40 ± 0.02 sec/image** |

Inference latency was effectively very similar across the three trained models in this benchmark. The observed differences are small relative to the overall latency and should not be treated as evidence of a meaningful inference speed advantage without additional repeated measurements.

## Comparison of Optimization Effects

### Mixed precision: Experiment A vs Experiment B

Experiment A is the FP32 baseline, and Experiment B changes training precision to FP16 while keeping the 100-step workload and batch configuration constant.

| Metric | A: FP32 | B: FP16 |
|---|---:|---:|
| Train loop time | 100 s | 93 s |
| Throughput | 1.01 steps/s | 1.09 steps/s |
| Seconds / step | ~0.990 | ~0.917 |
| Final displayed step loss | 0.106 | 0.106 |
| Inference time | 1.37 ± 0.02 s/image | 1.39 ± 0.02 s/image |

For this run, FP16 increased measured training throughput from 1.01 to 1.09 steps/s, approximately a **7.9% increase**, while reducing the optimization loop time from about 100 seconds to 93 seconds.

Peak VRAM was not instrumented, so the expected memory effect of mixed precision cannot be quantified from these runs.

### Gradient checkpointing: Experiment B vs Experiment C

Experiment C adds gradient checkpointing to the FP16 configuration used in Experiment B.

| Metric | B: FP16 | C: FP16 + GC |
|---|---:|---:|
| Train loop time | 93 s | 141 s |
| Throughput | 1.09 steps/s | ~0.714 steps/s |
| Seconds / step | ~0.917 | 1.40 |
| Final displayed step loss | 0.106 | 0.106 |
| Inference time | 1.39 ± 0.02 s/image | 1.40 ± 0.02 s/image |

In this measurement, enabling gradient checkpointing increased training time and reduced throughput. This is consistent with recomputation overhead during backpropagation, but the corresponding VRAM reduction cannot be quantified because peak memory was not recorded.

### Workload sizing: Smoke Test vs Experiment C

The smoke test and Experiment C both use FP16 with gradient checkpointing, but differ in workload size.

| Metric | Smoke Test | Experiment C |
|---|---:|---:|
| Optimization steps | 10 | 100 |
| Seconds / step | 1.40 | 1.40 |
| Throughput | ~0.714 steps/s | ~0.714 steps/s |
| Training loop time | ~16 s | 141 s |
| Status | Success | Success |

The matching reported 1.40 seconds per step suggests similar steady training cost per optimization step under the same FP16 plus gradient checkpointing configuration. The 100 step run naturally incurs substantially more total training time.

The smoke test is intended as a pipeline and memory feasibility check rather than a direct training quality benchmark.

## What Was Not Measured

Peak GPU VRAM was not recorded for these runs.

Therefore, this experiment currently supports quantitative comparison of:

1. training latency
2. training throughput
3. total optimization loop runtime
4. completion versus OOM behavior
5. final displayed step loss
6. inference latency

It does **not** yet support a numerical comparison of GPU memory savings from FP16 or gradient checkpointing.

For future runs, peak allocated and peak reserved GPU memory should be instrumented during the training subprocess.

## Summary

The experiment isolates three practical training system choices:

**Mixed precision** improved measured training throughput in the FP32 vs FP16 comparison.

**Gradient checkpointing** reduced training throughput in the FP16 vs FP16 plus checkpointing comparison, reflecting its compute-for-memory tradeoff.

**Workload sizing** behaved approximately linearly at the observed per-step timing for the FP16 plus checkpointing configuration.

Inference latency remained tightly clustered across the three trained variants at approximately **1.37 to 1.40 seconds per image**.

The largest missing systems metric is peak GPU VRAM. Recording that value in the next benchmark will make it possible to quantify the full memory versus throughput tradeoff.
