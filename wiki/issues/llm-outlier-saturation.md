# LLM activation outlier → quantization catastrophic loss

**Status**: partially mitigated (SmoothQuant), unresolved on-device
**Last updated**: 2026-07-20
**Severity**: blocks coherent text generation on any quantized NB
**Related**: [[../techniques/smoothquant]], [[../decisions/uint8-vs-int16]], [[host-vs-device-drift]]

## Summary

Transformer activations (especially at attention softmax input and FFN gate output) contain outlier channels 5–50× larger than median values. Per-tensor uint8/int16 quantization sees these outliers as the max_value; the resulting scale forces all normal values into just a few bins, destroying rank order at the output logits.

## Symptom

- **uint8** NB on device: all top-5 next-token candidates share the SAME dequant value (`+15.36` for scale=0.12/zp=128 → max output = (255-128)*0.12 = 15.24)
- Argmax picks arbitrarily among tied tokens
- Result: random rare vocabulary tokens (`Rothschild`, `Inflammation`, `avorable`) rather than the plausible `\n`, `,`, ` Paris` etc.

- **int16 dfp**: 18–28% of tensor values saturate at ±32767, top-10 tied at max ±31.999
- Reducing `fl` (increase range) doesn't help — outliers scale together

## Baseline (FP32 without quantization)

FP32 NB (SW-emulated at 7.28s/forward on T527 for SmolLM2) produces coherent tokens for the same prompts. This proves the compiled graph is mathematically correct — the failure is purely quantization.

## Attempted mitigations

| Approach | Result |
|---|---|
| More calibration samples (1 → 10 diverse prompts) | Host cos 0.11 → 0.93 for SmolLM2 |
| `--algorithm moving_average` vs `kl_divergence` | Marginal |
| `--algorithm normal` (minmax) | cos -0.27, worse |
| `--MLE` (per-layer error min) | Acuity crash `'NoneType' object has no attribute 'ext_attr'` |
| `--hybrid` | Not tried yet |
| bfloat16 export | Acuity error 64768 (SmolLM2), same for Qwen |
| **SmoothQuant α=0.5 + int16 dfp (host)** | **match 25/32, top5 3.53/5** |
| SmoothQuant α=0.5 + int16 dfp (device) | Still 0/32, values still saturate |
| SmoothQuant α=0.8 | cos 0.43, worse than α=0.5 |
| Reduce output tensor `fl` (10 → 8, range ×4) | Sat drops 18% → 17%, still 0/32 |

## Root of on-device gap

The host CPU emulator (`pegasus inference`) uses TensorFlow's floating-point simulation of quantized ops, which is a REFERENCE implementation. The device NPU implements the SAME ops in silicon with different accumulator widths, rounding modes, and possibly different saturation behavior. See [[host-vs-device-drift]].

## Next attempts (not yet tried)

1. Per-layer sensitivity analysis: keep outlier-heavy layers (first + last attention) at higher precision, aggressive on middle layers
2. Larger calibration set (100+ prompts) with intentional outlier prompts (numbers, special chars)
3. `--hybrid` quantize mode with manual per-layer qtype selection
4. Rebuild pipeline with newer Acuity version (6.21) if we can obtain
5. Apply SmoothQuant to lm_head output too (currently skipped due to dynamic weight from tied embedding)

## References

- Xiao et al. 2022, "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
- Commit `9726808` (SmoothQuant impl)
- [[../results/qwen-sq-int16-host]] — 25/32 match evidence
