# SmoothQuant ONNX rewrite

**Status**: implemented, partial win
**Last updated**: 2026-07-20
**Related**: [[../issues/llm-outlier-saturation]], [[../decisions/smoothquant-alpha]], [[../models/qwen25-05b]]

## Idea

Transformer activations have outlier channels 5–50× larger than median. When quantized to uint8/int16, the outliers dominate scale and everything else compresses into a few bins → catastrophic loss.

SmoothQuant (Xiao et al. 2022) shifts the outlier magnitude from activations into weights:

```
For each Linear Y = X @ W:
    s_j = max(|X_:,j|)^α / max(|W_j,:|)^(1-α)   per input channel j
    X' = X / s
    W' = W * s[:, None]
    Y = X' @ W' == X @ W    (mathematically equivalent, but easier to quantize)
```

- Activation range narrows (outliers gone) → per-tensor uint8 fits cleanly
- Weight range widens (absorbed the outlier) → still fine because weights are known statically at quant time

## Implementation (`M2_qwen/smoothquant_onnx.py`)

1. Load ONNX, find all MatMul nodes with an initializer weight (Q/K/V/O + gate/up/down projs — 167 for Qwen-0.5B)
2. Add each Linear's input activation as a graph output
3. Run ONNX Runtime with calibration prompts; collect per-channel abs-max for each activation
4. Compute `s = act_max^α / weight_max^(1-α)`, clip `s ∈ [1e-4, 1e4]`
5. Insert `Mul(1/s)` node before each MatMul on activation input
6. Multiply the weight initializer by `s` along input channel axis
7. Save modified ONNX

Skip lm_head if its weight is a Transpose result (dynamic) — for Qwen with tied embeddings, lm_head weight IS produced by Transpose so it's already skipped.

## Choices

- **α**: 0.5 balanced; 0.8 shifts more magnitude to weights (turned out worse in our tests)
- **Skip lm_head**: yes for tied-embedding models (weight isn't initializer)
- **Calibration set**: 35 diverse multilingual prompts for Qwen

## Results (Qwen2.5-0.5B, host CPU emulation)

| Variant | argmax match | top5 overlap | cos_last |
|---|---|---|---|
| plain uint8 | 0/32 | 0.12/5 | 0.11 |
| SmoothQuant α=0.5 uint8 | 0/32 | 0.12/5 | 0.51 |
| SmoothQuant α=0.8 uint8 | 0/32 | 0.78/5 | 0.43 |
| **SmoothQuant α=0.5 int16** | **25/32** | **3.53/5** | 0.16 |
| plain int16 | 1/32 (saturation) | 0/5 | 0.04 |

α=0.5 + int16 dfp is the current best. See [[../results/qwen-sq-int16-host]].

## Device gap

Same NB on T527 shows saturation (18% at ±32767 for int16, all-tied +15.36 for uint8). Not solved. See [[../issues/host-vs-device-drift]].

## Not attempted yet

- Per-layer α (different α for first vs middle vs last layers)
- Skip SmoothQuant on well-behaved layers (only apply where activation abs-max > threshold)
- Apply SmoothQuant to attention softmax output (currently only Linear inputs)
- Fold Mul(1/s) into preceding RMSNorm's gamma to eliminate the extra node
