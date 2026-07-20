# M1 SmolLM2 SmoothQuant + int16 dfp device — first-token semantic

**Status**: partial win, saturation-limited
**Last updated**: 2026-07-20
**Related**: [[../techniques/smoothquant]], [[../issues/host-vs-device-drift]], [[../issues/llm-outlier-saturation]]

## Setup

- Model: SmolLM2-135M-Instruct
- ONNX: axis-fixed + last-slice-patched + SmoothQuant α=0.5 (209 Linear rewrites)
- Acuity: `dynamic_fixed_point int16 kl_divergence`, 10 English calibration prompts
- NBG: 267 MB on T527, output `data_format=5 dfp=9` (int16 scale 1/512)

## Host inference (Acuity CPU emul)

- Argmax match: 25/32
- Avg top-5 overlap: 3.91/5
- Cosine at last position: -0.67 (position 31 artifact — earlier positions much higher)

## Device (T527 NPU) with `-b 0` (write output)

Output has 19% saturation at ±32767 (int16 max @ fl=9 corresponds to ±64.0).

Single-step argmax varies per prompt (previously identical for baseline uint8):

| Prompt | argmax | text | Correct? |
|---|---|---|---|
| `"def hello"` | 24 | `(` | ✅ plausible next token in Python (`def hello(...)`) |
| `"class Point"` | ? | `*` | roughly plausible |
| `"print('hello"` | ? | `2` | wrong (should be `'` or `,`) |
| `"1 + 1 ="` | 22 | `&` | wrong (should be ` 2`) |
| `"Once upon a time"` | 15 | `?` | wrong |
| `"The capital of France is"` | 1 | `` | tie-break BOS |

## Multi-token generation (greedy, 15 tokens)

```
prompt="def hello"           → "def hello(5)/�?2/9:"
prompt="def fibonacci"        → "def fibonacci1#0"
prompt="class Point"          → "class Point*K\")f\""
prompt="print('hello"         → "print('hello20!+&&"
```

Pattern: first ~1 token is often plausible (correctly picks up code/punctuation context), then rapidly degrades. The feedback loop amplifies saturation errors — each incorrect token pushed back into window shifts activation stats further from calibration.

## Speed

**~0.75 tok/s** end-to-end (dominated by adb 500 ms/token, pure NPU 147 ms).

## Interpretation

- **SmoothQuant works**: argmax now varies with input (baseline uint8 gave identical top-5 across all inputs)
- **int16 dfp on NPU has systematic saturation**: values exceed calibrated range → activation propagation clips → top logits tie
- **First token often correct** because attention still captures immediate context before saturation compounds
- Not a compilation issue — Acuity FP32 host emul of the SAME graph gives 25/32 match. Device NPU int16 hardware has extra numerical drift not reproduced in the CPU sim.

## Next attempts

- `--hybrid` quantize: keep logit-output layer at higher-precision (may still be within HW envelope)
- **Reduce activation range further** via more aggressive SmoothQuant (α approaching 1.0 shifts nearly all outliers to weights)
- Manual per-tensor fl in `.quantize` for outlier attention/softmax outputs
- Investigate device saturation location (which layer first saturates on-device?)

## Files

- IR: `M1_smollm2/acuity_out/smollm2_sq/smollm2_sq.json`
- Quantize table: `smollm2_sq_i16.quantize`
- NBG: `wksp_nbg_i16_nbg_unify/network_binary.nb`
- Meta: `nbg_meta_sq_i16.json`
- Runner: `M1_smollm2/device/generate_i16.py`
