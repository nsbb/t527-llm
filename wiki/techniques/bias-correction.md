# Per-channel bias correction

**Status**: verified single-token, multi-token limited
**Last updated**: 2026-07-24
**Related**: [[cpu-lm-head]], [[../issues/host-vs-device-drift]]

## Idea

T527 VIP9000 int8 accumulator has systematic per-channel bias vs FP32 CPU emulation. For SmolLM2:
- 51 channels |bias|>1
- 3 channels |bias|>3
- Extreme: channel 507 = -15.1

Subtract this bias vector from device hidden output before host lm_head.

## Recipe

1. Prepare N calibration prompts (20-100)
2. For each prompt, run ORT FP32 hidden and device NB → collect diff
3. Average diff across prompts and positions to get bias vector [hidden_dim]

Two variants:
- **all-position**: mean over all 32 positions × N prompts
- **content-position**: mean over positions 24-31 (skip pad) × N prompts

Content variant works best in practice.

## SmolLM2 results (100 calib prompts)

| Prompt | cos raw | cos +content-bias | Top-3 after correction |
|---|---|---|---|
| `def hello` | 0.33 | 0.58 | `\n` ` ` `(` |
| `The capital of France is Paris.` | 0.45 | 0.59 | `,` (semantic!) |
| `Once upon a time` | 0.65 | 0.69 | (varied) |
| `print('hello` | 0.31 | 0.54 | `\n` `(` `is` |
| `1 + 1 = 2 + 2 =` | 0.59 | 0.69 | (varied) |

## Qwen results (20 calib prompts = 15 KR + 5 EN)

| Prompt | cos raw | cos +content-bias | Top-3 after correction |
|---|---|---|---|
| `한국의 수도는` | **0.06** | **0.88** | `\n` `\xa0` `,` |
| `안녕하세요, 저는` | 0.03 | **0.82** | `\n` ` ` `,` |
| `봄이 오면 벚꽃이` | 0.04 | **0.85** | `\n` `\xa0` `\n\n` |
| `The capital of France is` | 0.31 | 0.19 | (English hurt by Korean-heavy calib) |
| `def hello` | 0.01 | 0.07 | (English hurt) |

**Korean-specific bias is a very strong recovery signal for Qwen.** This is essentially the M2 breakthrough — Korean tokens now traverse the pipeline coherently.

## Multi-token limitation

Static bias fails after 1-2 generation steps because sliding window moves activation distribution into unfamiliar territory not seen at calibration. Correction learned on `[..., pad, pad, prompt]` doesn't apply to `[..., pad, prompt, newtoken]`.

**Options to fix**:
- Online per-step bias update (impractical — needs ORT reference at inference)
- Position-conditioned bias (attempted — helped some prompts, hurt others)
- Bake bias into NBG (would remove host-side subtraction but doesn't fix the drift-distribution issue)
- Fine-tune model with device-drift-aware augmentation (M3 territory)

## Runtime

- SmolLM2 (135M): 1.75 tok/s end-to-end (adb + npu + host mm)
- Qwen (0.5B): 0.68 tok/s
- Bias subtraction cost: negligible (576-dim subtraction)

## Files

- `M1_smollm2/compute_bias.py` — measure bias
- `M1_smollm2/device_bias_c100_content.npy` — SmolLM2 100-prompt bias
- `M1_smollm2/device/generate_hidden_biascorr.py` — inference w/ bias
- `M2_qwen/device_bias_content.npy` — Qwen KR-heavy bias
- `M2_qwen/device/generate_hidden_biascorr.py` — Qwen inference
