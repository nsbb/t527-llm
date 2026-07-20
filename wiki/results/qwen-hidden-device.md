# M2 Qwen SmoothQuant + uint8 hidden device — mixed results

**Status**: pipeline verified, cos too low for coherent generation
**Last updated**: 2026-07-20
**Related**: [[../techniques/cpu-lm-head]], [[smollm2-sq-int16-device]], [[../issues/host-vs-device-drift]]

## Setup

- Model: Qwen2.5-0.5B-Instruct
- ONNX: axis-fix + last-slice + SmoothQuant α=0.5 + hidden-output cutoff (via `patch_output_hidden.py`)
- Acuity: `asymmetric_affine uint8 kl_divergence`, 10 multilingual calib
- NBG: 349 MB on T527
- Output: `[1,1,32,896]` uint8, scale=1.5567, zp=117

## Result

For 6 test prompts (English + Korean + code + math):

| Prompt | cos(hidden vs ORT) | Top-5 tokens after host lm_head |
|---|---|---|
| `한국의 수도는` (KR) | 0.060 | ` `, ` -`, ` (`, `,`, `.` |
| `The capital of France is` | **0.308** | `-`, `-G`, ` minimum`, ` or`, ` menu` |
| `안녕하세요` (KR) | -0.016 | ` `, ` R`, ` C`, ` A`, ` K` |
| `def hello` | 0.007 | ` `, ` now`, ` R`, `1`, `3` |
| `1 + 1 =` | **0.463** | ` `, ` that`, ` are`, `  `, `   ` |
| `봄이 오면` (KR: "when spring comes") | 0.144 | ` `, ` -`, ` (`, `2`, `,` |

## Why worse than SmolLM2

SmolLM2-135M hidden cos=0.45; Qwen-0.5B hidden cos=0.006~0.46 (much more variance).

**Qwen scale=1.56 vs SmolLM2 scale=0.20** → **8× coarser resolution** on the hidden state. Same absolute quantization error → 8× more damage to the FP32 approximation.

Root cause: SmoothQuant absorbed the LM head's implicit weight scale into the final RMSNorm's gamma → gamma has max_value=214 → hidden range becomes ±214 → uint8 with 256 bins gives step 1.56.

Combined with the deeper Qwen graph (24 layers × wider FFN) accumulating more numerical drift on NPU:
- Long Korean prompts especially suffer because most tokens are pad (151643 = eos) and only 3-5 real tokens carry context

## What we learned

1. Pipeline works end-to-end for Qwen (Korean tokens through NPU, host lm_head produces logits)
2. **uint8 vs int16**: uint8 still better (zero_point absorbs bias)
3. **Model size scales the drift problem**: 135M model workable, 500M model marginal, 2B (Midm) likely worse
4. English prompts get better cos than Korean because SmolLM2/Qwen calib set had more English samples with less padding

## Next attempts

- Force smaller hidden state range: hand-edit `.quantize` to reduce hidden max_value (need to also scale post-RMSNorm accordingly, tricky)
- Increase calibration to 100+ diverse prompts biased toward Korean
- Try SmolLM2 with Korean-specific fine-tuning first (as pipeline verification)
- Chunked decoder — compile 12 layers separately, chain on host with FP32 in between

## Speed

Same ~1.68 tok/s end-to-end (adb dominates).
