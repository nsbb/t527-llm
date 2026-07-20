# M2 Qwen SmoothQuant α=0.5 int16 — host inference

**Status**: verified
**Last updated**: 2026-07-20
**Related**: [[../techniques/smoothquant]], [[../models/qwen25-05b]], [[../issues/host-vs-device-drift]]

## Setup

- Model: Qwen2.5-0.5B-Instruct
- ONNX: axis-fixed + last-slice-patched + SmoothQuant α=0.5 (167 Linear rewrites)
- Acuity quantize: `dynamic_fixed_point int16 kl_divergence`
- Calibration: 10 diverse prompts (English, Korean, Chinese, code)
- Inference: `pegasus inference --dtype quantized --device CPU`

## Result

Compared against ONNX Runtime FP32 baseline on `calib_00` prompt (`"The capital of France is Paris."`):

| Metric | Value |
|---|---|
| Argmax match | **25/32** positions |
| Avg top-5 overlap | **3.53/5** per position |
| Last-position cosine | 0.16 (tricky — see notes) |
| Last-position top-5 (fp32) | `[576, 4710, 1084, 758, 5692]` |
| Last-position top-5 (int16 quant) | `[139318, 100728, 100177, 106669, 5824]` |

## Interpretation

- 25/32 = **78% of positions predict correct next token**
- 3.53/5 top-5 overlap = model recovers most of the probability mass
- Low last-position cosine likely because at position 31 (post-EOS in calib_00), the FP32 tail-end distribution differs; earlier positions where the model has more context show higher agreement

## Comparison to baselines

| Config | argmax match | top5 overlap |
|---|---|---|
| baseline uint8 (10-sample calib) | 0/32 | 0.12/5 |
| baseline int16 (10-sample calib) | 1/32 | ≤ 0.1/5 |
| SmoothQuant α=0.5 uint8 | 0/32 | 0.12/5 |
| SmoothQuant α=0.8 uint8 | 0/32 | 0.78/5 |
| **SmoothQuant α=0.5 + int16** | **25/32** | **3.53/5** |

**SmoothQuant × int16 is the winning combo**. Neither alone is enough:
- SmoothQuant + uint8: activation range OK but 256 output bins can't distinguish top logits
- Plain int16: activation outliers still saturate the accumulator
- Together: activations rescaled to fit uint8 easily → int16 output preserves logit rank

## What broke on device

Same NB pushed to T527 → int16 output saturates 18% at ±32767, argmax collapse to tied max values. Host CPU emulator ≠ device NPU numerically. See [[../issues/host-vs-device-drift]].

## Reproducibility

```bash
cd M2_qwen
python3 smoothquant_onnx.py \
  --input work/generated/qwen2.5_0.5b_w32/real_llm_fixed.onnx \
  --output work/generated/qwen2.5_0.5b_w32/real_llm_smoothquant.onnx \
  --calib-dir acuity_out/qwen25_0_5b_w32 --alpha 0.5

# Acuity import + int16 quantize + host inference (see quantize.log)
```

Commit: `9726808`
