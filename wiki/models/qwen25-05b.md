# Qwen2.5-0.5B-Instruct on T527

**Status**: compile+run verified, quantized coherence pending, Korean tokens traversed
**Last updated**: 2026-07-20
**Related**: [[../techniques/smoothquant]], [[../results/qwen-sq-int16-host]], [[../issues/host-vs-device-drift]]

## Model card

- HuggingFace: `Qwen/Qwen2.5-0.5B-Instruct`
- Architecture: `Qwen2ForCausalLM`, 24 layers
- hidden 896, num_attention_heads 14, num_key_value_heads 2 (aggressive GQA), head_dim 64
- intermediate_size 4864 (3× SmolLM2), SwiGLU FFN
- vocab_size 151936 (~3× SmolLM2), tied embedding
- **Q/K/V projections have biases** (Llama doesn't) — handled by petayyyy's `optional_tensor` mechanism
- RMSNorm eps 1e-6, RoPE θ **1,000,000** (10× SmolLM2 = supports longer context)
- Multilingual training incl. Korean, Chinese, code
- Original dtype bf16 (943 MB safetensors)

## Static ONNX

- Output: 1.9 GB (`work/generated/qwen2.5_0.5b_w32/real_llm.onnx`)
- 1523 nodes: 217 MatMul, 315 Mul, 241 Add, 49 RMSNorm (= 24 attn + 24 ffn + 1 final)
- Input `token_ids [1, 32]` int32
- Output `logits [1, 32, 151936]` fp32 (after last-slice patch)

## Compiled artifacts

| Variant | Size | Device time | Notes |
|---|---|---|---|
| uint8 asymmetric_affine (calib 10) | 393 MB | 198 ms (5.0 tok/s) | Tied at +15.36 collapse |
| int16 dfp (calib 10) | 891 MB | 290 ms (3.4 tok/s) | 18% saturation |
| **SmoothQuant α=0.5 + uint8** | 389 MB | 191 ms | Host cos 0.51, device still collapse |
| **SmoothQuant α=0.5 + int16** | 887 MB | ~300 ms | **Host 25/32 match**, device drift |
| FP32 non-quant | — | **fails export** | Fatal 64768 (model too big) |

## Korean pipeline verification

Input prompt: `"한국의 수도는"` (Korean: "The capital of Korea is")
- Tokenized: `[..., 23573, 124785, 20401, 134013, 16560]` — Korean tokens present in vocab
- Push int32 [1, 1, 32] to `/data/local/tmp/qwen_llm/input_0.dat`
- vpm_run executes with `cid=0x10000016` (T527 PID matches)
- Output emerges (uint8 or int16) — pipeline is language-agnostic

But top-5 still incoherent due to [[../issues/llm-outlier-saturation]].

## Best result so far

**SmoothQuant α=0.5 + int16 dfp on host CPU emulation**:
- 25/32 argmax match vs ORT FP32
- Avg top-5 overlap 3.53/5
- **Real proof that SmoothQuant can recover most semantics**

On device the same NB drifts — output values exceed calibrated range, saturate. See [[../issues/host-vs-device-drift]].

## Pipeline commands

```bash
cd M2_qwen

# 1. Download (HF direct via curl -L on resolve URL)
curl -L "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors" \
     -o work/models/qwen2.5-0.5b-instruct/model.safetensors

# 2. Static ONNX (reuse SmolLM2 script)
python3 scripts/host/make_real_llm_onnx.py \
  --model-dir work/models/qwen2.5-0.5b-instruct \
  --output-dir work/generated/qwen2.5_0.5b_w32 --seq-len 32

# 3. Patches
python3 ../M1_smollm2/patch_onnx_last_slice.py input.onnx output.onnx \
        --vocab 151936 --seq-len 32
python3 ../M1_smollm2/patch_reducemean_axes.py input.onnx output.onnx

# 4. SmoothQuant
python3 smoothquant_onnx.py --input <onnx> --output <onnx> \
  --calib-dir <dir> --alpha 0.5

# 5. Acuity pipeline (see M2_qwen/run_export.sh)
```

## Sources

- HF: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Qwen2 tech report: https://arxiv.org/abs/2412.15115
- Commit `faf1be2` (pipeline), `9726808` (SmoothQuant), `2a4e27c` (CHANGELOG)
