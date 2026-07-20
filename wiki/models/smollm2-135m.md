# SmolLM2-135M-Instruct on T527

**Status**: compile+run verified, quantized coherence pending
**Last updated**: 2026-07-16
**Related**: [[../pipeline/onnx-generation]], [[../techniques/axis-fix-patch]], [[../results/smollm2-fp32-nb]]

## Model card

- HuggingFace: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Architecture: `LlamaForCausalLM`, 30 layers
- hidden 576, num_attention_heads 9, num_key_value_heads 3 (GQA), head_dim 64
- intermediate_size 1536, SwiGLU FFN
- vocab_size 49152, tied embedding
- RMSNorm eps 1e-5, RoPE θ 100000
- Original dtype bf16 (257 MB safetensors)

## Static ONNX (petayyyy `make_real_llm_onnx.py --seq-len 32`)

- Output ONNX: 514 MB, 1901 nodes
- Input `token_ids [1, 32]` int32
- Output `logits [1, 32, 49152]` fp32 (after [[../techniques/last-slice-patch|last-slice patch]])
- RoPE sin/cos precomputed to constant tensors → sidesteps Acuity Cos gap
- RMSNorm decomposed (Pow → ReduceMean → Sqrt → Reciprocal → Mul → Mul) — requires [[../techniques/axis-fix-patch|axis-fix]]

## Compiled artifacts

| Variant | Size | Device time | Accuracy vs FP32 (host) |
|---|---|---|---|
| uint8 asymmetric_affine (v3 axis-fixed, 10 calib) | 124 MB | 92 ms | cos 0.93 (host), 0/32 argmax (device) |
| int16 dfp (v3 axis-fixed) | 268 MB | 147 ms | Saturation 28% |
| **FP32 non-quant (v3 axis-fixed)** | **626 MB** | **7.28 s** | **cos 0.805, coherent tokens** |

## Coherent generation proof (FP32 NB)

Prompt: `"The capital of France is"`
- top-5: `" as"`, `" of"`, `"."`, `" in"`, `","`
- All grammatically plausible; SmolLM2-135M is too small to say "Paris" but the model is running correctly

Prompt: `"1 + 1 ="`
- top-5: `"\n   "`, `"."`, `"ed"`, `"\n"`, `"ing"` (no arithmetic understanding — expected for 135M)

See [[../results/smollm2-fp32-nb]].

## Quantized coherence — unresolved

uint8/int16 both collapse. Root cause: [[../issues/llm-outlier-saturation]]. SmoothQuant not yet applied to SmolLM2 (currently only Qwen).

## Pipeline commands

```bash
# 1. HF download
huggingface-cli download HuggingFaceTB/SmolLM2-135M-Instruct \
  --local-dir work/models/smollm2-135m-instruct

# 2. Static ONNX
python3 scripts/host/make_real_llm_onnx.py \
  --model-dir work/models/smollm2-135m-instruct \
  --output-dir work/generated/smollm2_135m_w32 --seq-len 32

# 3. Patches (last-slice + axis-fix)
python3 patch_onnx_last_slice.py <input> <output> --vocab 49152 --seq-len 32
python3 patch_reducemean_axes.py <input> <output>

# 4. Acuity pipeline (see /home/nsbb/travail/claude/T527/t527-llm/M1_smollm2/*.sh)

# 5. Device
bash device/push_and_run.sh
```

## Sources

- HF: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- Reference impl: [petayyyy/a733_npu_driver](https://github.com/petayyyy/a733_npu_driver)
