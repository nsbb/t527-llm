# Log

Append-only chronological record. Every substantive change/finding gets a line.

Format: `YYYY-MM-DD HH:MM — <what happened> [commit-hash]`

---

## 2026-07-15

- 10:03 — Project started, empty t527-llm folder
- 10:48–11:00 — 4 parallel research reports (Allwinner, Verisilicon, community, reference NPUs) [d35378f]
- 14:56 — Korean LLM quantization research [db24f9c]
- 17:08 — PLAN.md 4-stage validation ladder [f3bed7a]
- 18:57 — M0 dummy ONNX + pegasus import batch runner [091282e]
- 19:10 — M0 workaround compositions verified (Cos via Sin, LayerNorm decomposed, GELU via Erf) — 33/33 ops covered [4322698]
- 19:16 — Cloned petayyyy/a733_npu_driver, downloaded SmolLM2-135M-Instruct
- 19:18 — Static ONNX generated (514MB) — RoPE precomputed to constants
- 19:20 — pegasus import success (v1) [0a34cbc]

## 2026-07-16

- 00:24–00:26 — quantize errors triaged: perchannel int16 unsupported, slice_last_hidden bug, type NPY issue, category image default
- 09:28 — patch_onnx_last_slice.py written [70cf811]
- 09:31 — re-import v2 with patched ONNX (2454-layer Acuity IR)
- 09:38 — uint8 asymmetric_affine quantize success [1c3e69a]
- 09:40 — NBG export for T527 PID0X10000016 (124MB) [7d05aee]
- 10:14 — Device connection verified, vpm_run_aarch64 present, /vendor/lib64 drivers
- 10:15 — First NPU forward pass — 93ms = 10.9 tok/s pure NPU
- 10:20–10:38 — FP32 golden collected, quantized output all-mismatch, layer permutation attempts fail
- 11:04 — Layer-by-layer divergence: Gather cos=1.0, first RMSNorm cos=0.875, lm_head cos=-0.38
- 11:11 — **ROOT CAUSE FOUND**: Acuity 6.12 promotes 3D→4D but doesn't renumber `axes` attribute → RMSNorm's `axes=[2]` reduces seq_len(32) instead of hidden(576). Numeric proof: acuity_mean == sq.mean(axis=1) exactly.
- 11:14 — patch_reducemean_axes.py: 61 nodes `axes=[2]` → `axes=[-1]` [f86a5c4]
- 11:16 — Acuity FP32 vs ORT FP32: **32/32 match, cos=1.0000** ★
- 11:19 — v3 uint8 NB → device: 92ms/forward but quantize collapses to 0/32
- 11:24 — int16 dfp saturates 28% at ±32767
- 12:15 — FP32 NB export success (626MB), 7.28s/forward on T527, real language tokens: "The capital of France is" → " as", " of", "."  ★ [6de9350]
- 12:30 — PROGRESS.md updated [b2a5684]
- 15:39 — 10-sample diverse calibration → cos 0.11 → **0.93** [b5a16f1]
- 16:35 — Multi-token decode: greedy collapses, top-k rare tokens [9b013b9]

## 2026-07-20

- 09:00–09:30 — Fetched Qwen2.5-0.5B-Instruct via HF direct curl (HuggingFace 504 mirror workaround, then 943MB direct)
- 09:35 — Qwen static ONNX 1.9GB, 24-layer Llama-family, GQA 14/2, vocab 151936
- 09:40 — Patches applied (49 ReduceMean + last-slice)
- 09:47 — Acuity import Qwen (2.4GB IR, 1523 nodes)
- 10:00 — 35 multilingual calibration prompts (EN + KR + CN + code)
- 10:04 — uint8 quantize 10-sample calib
- 10:05 — Qwen uint8 NBG (393MB), device 198ms/forward = 5 tok/s
- 10:10 — Korean tokens push through NPU pipeline successfully
- 10:12 — Qwen uint8 still collapses at +15.359 (all top-5 tied at max int8 dequant)
- 10:15 — Qwen FP32 export FAILS (fatal 64768 — model too big for FP32 NB)
- 10:17 — Qwen int16 dfp NB (891MB), 290ms/forward, 18% saturation
- 10:25 — SmoothQuant Python impl started
- 10:40 — smoothquant_onnx.py: 167 MatMul rewrites via ORT abs-max collection + Mul(1/s) insertion
- 10:44 — SmoothQuant IR imported (α=0.5)
- 10:49 — SmoothQuant+uint8 host: cos 0.51 (baseline 0.11)
- 10:55 — SmoothQuant+uint8 NB device: still tied at +15.359
- 11:02 — SmoothQuant α=0.8: cos 0.43, top5 overlap 0.78/5 (worse than α=0.5)
- 11:09 — **SmoothQuant α=0.5 + int16 host: match 25/32, top5 overlap 3.53/5** ★★
- 11:13 — SmoothQuant+int16 NBG export (887MB)
- 11:16 — Device runs it → 18% still saturating, still 0/32 argmax match
- 11:20 — Host vs device numerical drift confirmed (calibrated ±27.4 → device values exceed ±32)
- 11:25 — Attempt: reduce output fl 10→8 (range ×4) via patch_quantize_headroom.py
- 11:28 — hr NB tested on device: sat still 7.8%, 0/32 match [9726808]
- 11:32 — CHANGELOG.md added, retroactive from v0.0.1 to v0.5.0 [2a4e27c]
- 11:45 — Karpathy LLM Wiki pattern adopted for this repo
- 11:55 — wiki/ directory + schema.md + index.md + log.md scaffolding

---
- 12:28 — SmoothQuant α=0.5 + qbfloat16 quantize (Qwen)
- 12:29 — **★★ HOST BREAKTHROUGH**: match 30/32, top5 3.69/5, cos_last 0.9965 (FP32-equivalent)
- 12:35 — NBG export FAILS `Fatal model generation error: 64768` (also bfloat16 and float32 all fail — Qwen too big)
- 12:38 — SmolLM2 FP32 NB was 626MB — Qwen would be ~4GB, likely exceeds NBG compiler ceiling
- 12:42 — Result page wiki/results/qwen-sq-qbf16-host.md
- 12:50 — Attempt SmolLM2 SmoothQuant + qbfloat16: quantize OK, NBG export FAILS 64768 (same as Qwen)
- 12:54 — SmolLM2 SmoothQuant + uint8: host cos 0.93, argmax 0/32 (uint8 tie-break)
- 12:58 — Device test: argmax always token 1 (BOS tie-break), but top-5 now prompt-sensitive with real English tokens (Fred, cig, Ober) — SmoothQuant improved over baseline garbage
- 13:00 — Result page wiki/results/smollm2-sq-uint8-device.md [504aeb5+]
- 13:07 — Hardware precision table added to wiki/hardware/t527-vip9000.md
- 13:10 — Strategy pivot: any coherent-generation solution MUST fit within uint8/int16 fixed-point. qbf16 host wins are informational only.
- 13:15 — SmolLM2 SmoothQuant + int16 quantize + export success (267 MB NBG)
- 13:16 — Device test with -b 0: 19% saturation, argmax varies per prompt
- 13:17 — **★ First-token semantic success**: 'def hello' → argmax token 24 = `(`  (correct Python next-token)
- 13:18 — Multi-token greedy: first token often correct, then degrades (`def hello(5)/�?2/9:`) — saturation feedback loop
- 13:22 — Result page wiki/results/smollm2-sq-int16-device.md
- 14:28 — W=16 NBG export 264MB success
- 14:35 — Device test: saturation reduced 19% → 11% (W=16 halves activation memory) but top-5 still tied at int16 max (fl=9 → ±64.0)
- 14:36 — argmax varies per prompt with W=16, but semantically weak (small model + still-saturating output)
- 14:45 — W=32 wide-fl patch: output fl 9→6, outlier layers fl-=1 for max_value>20
- 14:46 — Wide-fl NBG export 267MB
- 14:47 — Device test: **saturation 19% → 4.7%** (fl=8 gives range ±128) but top-5 still all tied at ±128.0
- 14:50 — Confirmed: device NPU int16 output values 2x+ FP32 range even with wide fl → true numerical drift limit
## Rules

- Only append. Never edit past entries.
- Every entry ties to a specific commit if code was pushed
- Findings marked with ★ = notable, ★★ = breakthrough- 13:05 — Realization: T527 VIP9000-NanoSI-Plus has NO FP HW. bf16/qbf16 export failures aren't bugs — Acuity NBG compiler correctly refuses to emit code for absent HW. SmolLM2 FP32 NB "works" only via CPU fallback SW-emul (80x slower). Only uint8/int16 are viable for production.- 14:15 — W=16 SmolLM2 SmoothQuant + int16 quantize (10 English calib)- 14:41 — W=8 SmolLM2 NBG export (264 MB), device sat=38% (higher than W=32 because Acuity auto-picked fl=10 range ±32)