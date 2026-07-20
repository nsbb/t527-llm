# M1 SmolLM2 SmoothQuant + uint8 device — prompt-sensitive but incoherent

**Status**: verified device execution
**Last updated**: 2026-07-20
**Related**: [[../techniques/smoothquant]], [[../issues/llm-outlier-saturation]], [[../results/qwen-sq-qbf16-host]]

## Setup

- Model: SmolLM2-135M-Instruct
- ONNX: axis-fixed + last-slice-patched + SmoothQuant α=0.5 (209 Linear rewrites)
- Acuity: `asymmetric_affine uint8 kl_divergence`, 10 English calib prompts
- NBG: 122 MB on T527

## Host inference (Acuity CPU-emul)

Compared against ORT FP32 on `calib_00` (English):
- argmax match: 0/32
- top-5 overlap avg: 0.62/5
- last-position cosine: **0.9305**

Cos is high but argmax still 0 — 256 output bins can't distinguish top logits.

## Device (T527 NPU)

Argmax always **1** (BOS token) — tie-break artifact.
Top-5 tokens vary per prompt (this IS progress vs pre-SmoothQuant which had identical top-5 for all inputs):

| Prompt | Top-5 |
|---|---|
| `"The capital of France is"` | ` cig`, ` Fred`, `bott`, `tp`, ` Subsequent` |
| `"Once upon a time"` | ` lawsuit`, ` Daniel`, ` highlighting`, ` �`, `\n        \n   ` |
| `"1 + 1 ="` | ` ordinance`, `aturday`, ` afflict`, ` Traditional`, ` Adapt` |
| `"def hello"` | `calorie`, ` cranial`, ` Mythology`, ` NS`, `quantity` |
| `"Hello, my name"` | ` Ober`, `stat`, ` placement`, ` reserve`, ` pointing` |

All actual English tokens (unlike pre-SmoothQuant which surfaced rare non-English like `Rothschild`, `Inflammation`). Not semantically coherent because 135M is small and uint8 caps logit resolution.

## Verdict

- SmoothQuant successfully redistributes activation outliers
- But uint8 output-tensor quantization still destroys logit rank order for the SINGLE top choice
- Prompt-sensitivity restored — the graph IS running correct math, just clipped
- Would benefit from qbfloat16 output (but SmolLM2 SmoothQuant+qbfloat16 also fails 64768 fatal — likely combination of SmoothQuant Mul nodes + qbfloat16 path breaks NBG compiler)

## Next attempts

- Hybrid quantize: keep output MatMul at int16/qbfloat16, rest at uint8
- Compile decoder blocks separately (each block small enough to fit qbfloat16 NBG)
- Enable native `--hybrid` flag (may auto-pick per-tensor precision)
