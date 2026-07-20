# Last-slice ONNX patch

**Status**: verified workaround
**Last updated**: 2026-07-16
**Related**: [[../issues/slice-last-hidden]]

## Script

`M1_smollm2/patch_onnx_last_slice.py`

## What it does

Removes the `slice_last_hidden` ONNX node from petayyyy's static LM ONNX graph, rewires `final_last_token` consumers to read from `final_rms_out` directly. Updates graph output `logits` shape from `[1, 1, V]` → `[1, seq, V]`.

## Why

Acuity 6.12's ONNX→TF converter miscalculates the `size` parameter for the last-token seq-axis slice, computing `[1, -30, 32, 576]` (size=-30 is invalid). Causes `ValueError: Invalid value in tensor used for shape: -30` during quantize.

Removing the slice sidesteps the bug. The lm_head then computes logits for ALL 32 positions, and the host takes `logits[:, -1, :]` at inference time (~32× extra unused work at lm_head, but avoids the compiler bug).

## Usage

```bash
python3 patch_onnx_last_slice.py input.onnx output.onnx \
    --vocab 49152 --seq-len 32   # or 151936 for Qwen
```

## Verified on

- SmolLM2-135M: `--vocab 49152`
- Qwen2.5-0.5B: `--vocab 151936`

Both accept the patched ONNX through Acuity import successfully.
