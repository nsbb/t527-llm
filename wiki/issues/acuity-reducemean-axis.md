# Acuity 6.12 ReduceMean axis off-by-one

**Status**: fixed (workaround)
**Last updated**: 2026-07-16
**Severity**: catastrophic — cos=-0.38 vs FP32 without fix
**Related**: [[../techniques/axis-fix-patch]], [[../pipeline/acuity-import]], [[../models/smollm2-135m]]

## Summary

Acuity 6.12 promotes 3-D ONNX tensors like `[1, 32, 576]` to internal 4-D `[1, 1, 32, 576]` but does NOT renumber attribute `axes`. So `ReduceMean(axes=[2])` — intended to reduce hidden dim (576) — ends up reducing seq_len (32) instead. Every RMSNorm in the graph produces wrong output.

## Symptom

Acuity FP32 host inference disagrees with ONNX Runtime FP32:
- Cosine similarity `-0.38` at last-position logits
- 0/32 argmax match
- Systematically predicts rare vocab tokens ('avorable', 'Rothschild', 'omson')

Every layer's RMSNorm output diverges progressively:
| Layer | Cos vs ORT |
|---|---|
| Gather (embedding) | 1.0000 |
| Layer 0 RMSNorm | 0.8750 ← first divergence |
| Layer 0 Q projection | 0.9726 |
| Layer 29 lm_head | -0.3818 |

## Root cause proof

Numerical exact match between Acuity's ReduceMean output and `sq.mean(axis=1)` (seq axis) instead of `sq.mean(axis=-1)` (hidden axis).

```
max_abs_diff(acuity_mean, sq.mean(axis=1)) = 0.000000e+00
```

Acuity IR JSON confirmed: input layer has shape `[1, 1, 32]` (channels=1 added), so ONNX axis 2 → Acuity axis 3 shift.

## Fix

Change all ONNX `ReduceMean` nodes from `axes=[2]` to `axes=[-1]`. `-1` is layout-agnostic and Acuity handles it correctly regardless of internal rank promotion.

Applied via [[../techniques/axis-fix-patch|patch_reducemean_axes.py]] — 61 nodes for SmolLM2-135M, 49 nodes for Qwen2.5-0.5B (equal to num of RMSNorm instances in each model).

## Verification

After patch:
- Acuity FP32 vs ORT FP32: **32/32 argmax match, cos = 1.0000**
- Same fix works for both SmolLM2 (Llama arch) and Qwen (Qwen2 arch)

## Not addressed

- Whether Acuity 6.21 (newer version) has this fixed — no access
- Whether other ONNX ops (Softmax `axis=3`, Concat `axis=3`) have similar issue — those ops happen on already-4D tensors so no shift, verified correct
- Whether other 3-D ONNX ops elsewhere in the graph have axis attributes we haven't audited

## Sources

- Commit `f86a5c4`
- petayyyy uses newer Acuity (via `ubuntu-npu:v2.0.10.1`) so did not hit this
