# Axis-fix ONNX patch

**Status**: verified fix
**Last updated**: 2026-07-16
**Related**: [[../issues/acuity-reducemean-axis]]

## Script

`M1_smollm2/patch_reducemean_axes.py`

## What it does

Rewrites every `ReduceMean` node's attribute `axes=[2]` → `axes=[-1]`.

## Why

Acuity 6.12 silently promotes 3-D ONNX tensors (like `[1, 32, hidden]`) to internal 4-D layout `[1, 1, 32, hidden]` without renumbering the `axes` attribute of downstream ops. So `ReduceMean(axes=[2])` — meant to reduce hidden dim — ends up reducing seq_len. `axes=[-1]` is rank-invariant → always reduces the last dim regardless of Acuity's rank promotion.

## Effect

- 61 nodes patched for SmolLM2-135M (30 attn RMSNorm + 30 FFN RMSNorm + 1 final)
- 49 nodes for Qwen2.5-0.5B (24 + 24 + 1)
- Acuity FP32 host vs ORT FP32: `0/32 → 32/32 argmax match`, `cos: -0.38 → 1.0000`

## Usage

```bash
python3 patch_reducemean_axes.py input.onnx output.onnx
```

## Verified on

- SmolLM2-135M (commit `f86a5c4`)
- Qwen2.5-0.5B (commit `faf1be2`)

## Applies to

Any model with RMSNorm decomposed to primitive ops via petayyyy's `make_real_llm_onnx.py`. Would need adaptation for models with native `LayerNormalization` (opset 17) — but Acuity 6.12 doesn't support that op anyway, so all models get decomposed RMSNorm.
