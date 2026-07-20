# Host CPU emulation vs T527 NPU device numerical drift

**Status**: unresolved
**Last updated**: 2026-07-20
**Severity**: blocks device coherence even when host inference is correct
**Related**: [[llm-outlier-saturation]], [[../techniques/smoothquant]]

## Symptom

An Acuity NBG that gives 25/32 argmax match on host CPU emulator (`pegasus inference --dtype quantized --device CPU`) shows 0/32 match when the same NB runs on the T527 NPU. Device output values exceed the calibrated max_value and saturate at int16 extremes.

Numerical:
- Host: int16 output values within ±20 (calibrated max_value ≈ 27.4)
- Device: int16 output has values at ±32767 (saturation) for ~18% of tensor

## Hypotheses

1. **Different accumulator width**: NPU MAC may use different accumulator precision (int32 vs int48) with different post-add saturation behavior than TF CPU sim.
2. **Different rounding mode**: NPU may use RTNE truncation or biased rounding vs TF's true round-half-to-even.
3. **Fixed-point rescaling drift**: int16 dfp requires rescaling multiplications between layers — the drift may compound over 24 layers (Qwen) more on NPU than CPU sim.
4. **Missing per-op quantize params on NPU**: some intermediate tensor's `fl` might be interpreted differently by NPU firmware.

## Attempted

- Reduce output tensor `fl` (10→8, range ×4) via `patch_quantize_headroom.py`: sat drops 18% → 17.6%, still 0/32 match. Confirms saturation is NOT just at the output tensor — earlier layers may saturate and propagate.

## Not tried

- Enable Acuity's `--device GPU` for host inference (compare against another reference)
- Instrument device to dump per-layer intermediate tensor values (would need custom vpm_run patch)
- Try `symmetric_affine int16` (not supported by Acuity 6.12 for this quantizer)
- Rebuild with Acuity 6.21 or newer (no access)

## Practical workaround

For M2, if the goal is to demonstrate coherent generation on device, the current best path is **FP32 NB** — proved to work for SmolLM2-135M (7.28s/forward, coherent tokens) but does NOT compile for Qwen (fatal 64768). Need to find a middle ground:
- Try `qbfloat16` quantizer (SW-emulated bf16, might work where full FP32 fails)
- Try hybrid: keep only lm_head at higher precision (FP16/int16), rest uint8

## References

- Commit `9726808` — SmoothQuant + int16 host works, device drifts
- [[../results/qwen-sq-int16-host]] — the 25/32 host result
