# CPU-side lm_head — hidden state on NPU, final MatMul on host

**Status**: verified breakthrough
**Last updated**: 2026-07-20
**Related**: [[../issues/host-vs-device-drift]], [[../issues/llm-outlier-saturation]], [[smoothquant]]

## Idea

The failure mode of on-device quantized LLM generation is at the **final logits output**: lm_head produces values in ±60 range that saturate int16 (±32) or collapse to few uint8 bins.

**Hidden state (post-final-RMSNorm) is ±3-30 range** — comfortable for both uint8 and int16 quantization.

So: cut lm_head off the NPU graph, output the hidden state instead, and run the final `hidden @ W_embed^T` MatMul on host CPU with FP32 embedding weight. Because tied embeddings mean the embedding weight is already known, host lm_head is a single ~5 M FLOP MatMul → negligible.

## Implementation

- `M1_smollm2/patch_output_hidden.py`: modify ONNX to output `final_rms_out` (pre-lm_head), drop the LM head + logits chunks
- Extract FP32 `token_embed` initializer via ONNX numpy_helper, save `token_embed.npy`
- Runtime: NPU produces `[1, 1, 32, 576]` int8/int16 hidden → dequant to fp32 → `hidden[-1] @ embed.T` → logits over vocab

## Measurements (SmolLM2-135M SmoothQuant α=0.5)

Hidden state vs ORT FP32 hidden state (calib_00 prompt):

| Quantize | Hidden [min,max] | Hidden std | cos vs ORT |
|---|---|---|---|
| ORT reference | [-24.05, 12.96] | 1.68 | 1.0000 |
| **int16 dfp NB** | [-57.36, 59.25] | 14.4 | **-0.17** |
| **uint8 NB** | [-7.36, 21.87] | 1.45 | **0.45** |

**Huge finding**: int16 hidden state has **8× amplified std** vs FP32 reference — NPU int16 accumulator has systematic BIAS. uint8's `zero_point` offset absorbs this bias, delivering hidden state within 15% std of reference.

## Multi-token generation output (uint8 hidden, greedy)

```
prompt="def hello"                    → "def hello111111111** **1 only's1"
prompt="The capital of France is"     → "The capital of France is11111's1's Laf,,â\n spotus"
prompt="Once upon a time"             → "Once upon a time11111111 and11â\n's1"
prompt="1 + 1 ="                      → "1 + 1 =,111ofâusus spot..."
```

- **Real English tokens** (' and', 's, ',', '.', '\n', ' spot', ' only') — no more Rothschild-tier garbage
- **Prompt-sensitive**: different token distributions per prompt
- **Repetitive**: token `1` dominates argmax because SmolLM2-135M is small AND uint8 hidden only 45% correlated with FP32 hidden

## Speed

**1.68 tok/s end-to-end** (adb push+run+pull + host FP32 lm_head MatMul over full vocab).
Pure NPU: ~100 ms/forward (smaller NB, 104 MB vs 267 MB with lm_head).
Host lm_head: ~1 ms per token (single MatMul 576×49152).

## Why int16 fails but uint8 succeeds

`asymmetric_affine uint8`: `x_deq = (x_int - zero_point) * scale` — the zero_point is a bias term the calibrator learns. If the NPU accumulator introduces systematic offset, zero_point absorbs it.

`dynamic_fixed_point int16`: `x_deq = x_int * scale` — no offset. Any accumulator bias propagates raw.

For the LLM outlier-heavy activations, per-layer bias accumulates over 30 layers → 8x amplification in std.

## Practical recommendation

**Use uint8 asymmetric_affine + hidden-output cutoff + host lm_head** for T527 LLM deployment.
Do NOT use int16 dfp for the graph output — bias not absorbed.

## Not yet tried

- SmoothQuant α > 0.5 with hidden output (may improve cos)
- Increased calibration diversity for hidden output NB
- Same hidden-output trick on Qwen2.5-0.5B (Korean-capable multilingual)
- Manual per-layer uint8/int16 mix (uint8 for internal, uint8 for output)
