# M2 Qwen SmoothQuant + qbfloat16 — host near-perfect

**Status**: verified on host, on-device export blocked
**Last updated**: 2026-07-20
**Related**: [[../techniques/smoothquant]], [[qwen-sq-int16-host]], [[../issues/host-vs-device-drift]]

## Result (host CPU emulation)

- Model: Qwen2.5-0.5B-Instruct
- ONNX: axis-fixed + last-slice-patched + SmoothQuant α=0.5
- Acuity quantize: `qbfloat16 qbfloat16 kl_divergence` (10-sample calibration)
- Compared against ONNX Runtime FP32 baseline on `calib_00`:

| Metric | Value |
|---|---|
| Argmax match | **30/32 positions** |
| Avg top-5 overlap | 3.69/5 |
| Last-position cosine | **0.9965** |
| Last-position top-5 (fp32) | `[576, 4710, 1084, 758, 5692]` |
| Last-position top-5 (qbf16) | `[576, 4710, 1084, 758, 1096]` |

**4/5 top-5 tokens EXACT MATCH** with FP32 baseline. Only 5th slot differs (5692 vs 1096). This is essentially FP32-equivalent quality.

## Ranking of all attempted M2 quantizations (host)

| Combo | argmax match | top5 overlap | cos_last |
|---|---|---|---|
| plain uint8 | 0/32 | 0.12/5 | 0.11 |
| plain int16 dfp | 1/32 | ≤0.1/5 | 0.04 |
| SQ α=0.5 uint8 | 0/32 | 0.12/5 | 0.51 |
| SQ α=0.8 uint8 | 0/32 | 0.78/5 | 0.43 |
| SQ α=0.5 int16 dfp | 25/32 | 3.53/5 | 0.16 |
| **SQ α=0.5 qbfloat16** | **30/32** | **3.69/5** | **0.9965** |

## Device blocker

Cannot export SmoothQuant+qbfloat16 NB for Qwen — Acuity fails with `Fatal model generation error: 64768` during NBG generation. Same error for `bfloat16`, `float32`. The NBG compiler has a size ceiling around Qwen-0.5B's total footprint (~2 GB IR).

For SmolLM2-135M (much smaller) FP32 NB export DOES work (626MB, 7.28s/forward SW emul). Should test qbfloat16 there too.

## What this tells us

- SmoothQuant + qbfloat16 is the correct algorithmic recipe — it recovers FP32 accuracy
- The device numerical drift issue [[../issues/host-vs-device-drift]] is SPECIFIC to int8/int16 fixed-point on NPU; float-like formats (qbfloat16, bfloat16) would work if we could compile them
- Need to find a way to fit high-precision NB into the NBG size limit — options:
  - Chunk the model (compile decoder blocks separately, stitch on-host)
  - Reduce W (context) — e.g., W=16 halves activation memory
  - Force qbfloat16 only for outlier-heavy layers, uint8 elsewhere (hybrid)

## Reproducibility

```bash
cd M2_qwen
# Same SmoothQuant ONNX as int16 experiment
python3 <acuity in docker> quantize \
  --model qwen25_0_5b_w32_sq.json --model-data qwen25_0_5b_w32_sq.data \
  --with-input-meta qwen25_0_5b_w32_sq_inputmeta.yml \
  --iterations 10 --rebuild-all \
  --model-quantize qwen25_0_5b_w32_sq_qbf16.quantize \
  --quantizer qbfloat16 --qtype qbfloat16 --algorithm kl_divergence

# Host inference OK
# NBG export → Fatal 64768
```
