# VeriSilicon / Vivante NPU — LLM Porting Research Report

Target hardware: Allwinner T527, Vivante VIP9000-NanoSI-Plus NPU (~2-3 TOPS INT8).
Current baseline (user): Acuity Toolkit 6.12 (`pegasus`) + VivanteIDE 5.7.2 + VIPLite v1.13/v2.0 driver, with Conformer/CitriNet ASR already working end-to-end (ONNX → import → quantize → NBG export → device run).

Research date: 2026-07-15. All star / release numbers as observed on the day of the fetch.

---

## 0. Executive Summary (TL;DR)

1. **VeriSilicon has a real, actively-maintained LLM software stack** — it centers on the newer `acuitylite` deployment tool and the freshly-published Acuity model zoo LLM catalog (Qwen1.5/2/2.5/3, Llama-2/3, Phi-2/3, Gemma-2/3, DeepSeek-R1 distills, Whisper, InternVL, Qwen-VL, SD-XL). The catalog is not just marketing: entries include GPTQ-Int4 and Q4_0-GGUF variants, and VeriSilicon's own README explicitly says the toolkit is meant to consume these formats.
2. **However, the LLM-catalog entries are targeted at newer VIP9000 revisions with 4-bit hardware support**, not the small NanoSI-Plus in T527. VeriSilicon's LLM claims (Llama 2, SD) are made at the IP-family level ("VIP9000 series … 4-bit quantization and compression") without per-variant benchmarks. NanoSI-Plus @ ~3 TOPS with only INT8/INT16/FP16/BF16 in Acuity 6.12 will not fit a 7B model in memory or throughput terms.
3. **Op coverage in TIM-VX is transformer-adequate but LLM-partial.** LayerNorm, MatMul, Softmax, Gelu, Swish (SiLU), Embedding_lookup, StridedSlice, Concat, Reshape, Gather, Cast, Sin, Sqrt/Rsqrt, Erf are all mapped. `Cos` is NOT explicit in the TIM-VX ops README (only `Sin` is), and there is **no** first-class RMSNorm, RoPE, ScaledDotProductAttention, MultiHeadAttention, GroupedQueryAttention, causal-mask op, or KV-cache primitive. These must be composed from primitives (RMSNorm = Square + ReduceMean + Rsqrt + Mul; RoPE = Sin + [emulated Cos] + Mul + Add), and the compilation-time success of that composition depends on the pegasus/Acuity version.
4. **Ecosystem realistically usable for T527 today:**
   - `VeriSilicon/acuity-models` — reference model zoo, JSON-format graphs viewable in Netron
   - `VeriSilicon/acuitylite` — successor to the pegasus-based flow, importer + INT8/UINT8 quantizer, exports TFLite / TIM-VX
   - `VeriSilicon/TIM-VX` — the C++ interface library on the device side
   - `VeriSilicon/tflite-vx-delegate` — TFLite external delegate that routes to TIM-VX
   - `VeriSilicon/LiteRT-LM` — fork of Google's LiteRT-LM (LLM runtime); Gemma3-1B / Qwen2.5-1.5B / Phi-4-mini already packaged as `.litertlm` — but the fork currently advertises Qualcomm/MediaTek NPU targets, not VeriSilicon. Watch this repo.
   - `VeriSilicon/FlagGems` — VeriSilicon fork of FlagOpen's Triton-based LLM operator library (RMSNorm, RoPE, layer_norm, apply_rotary_position_embedding, softmax, gelu, silu). VSI-main branch, but current backend documented is NVIDIA GPU; a Triton-VSI plugin (`triton-vsi-backend`, `vsi-pjrt-plugin`) exists in the same org.
5. **Community LLM-on-VIP9000 activity is thin.** Most "edge LLM on ARM SoC" work is on Rockchip RK3588 (rkllm), not on VeriSilicon. There is an active ONNX Runtime issue (microsoft/onnxruntime #28244) proposing a VIPLite Execution Provider for Allwinner A733/T527 — closed as stale, no PR yet. One external MLIR project (MaverickLong/MLIR-TIM-VX) targets A733 but has explicitly **no transformer support** yet.
6. **NXP's eIQ Gen-AI Flow** is worth studying as a precedent: NXP claims Llama-2-7B running on the i.MX 8M Plus (which uses VeriSilicon VIP8000 with 2.3 TOPS — same generation family) and Llama/Qwen/Danube pre-optimized ONNX models. This is the closest publicly-documented LLM-on-Vivante-NPU shipping stack.

**Recommendation for the T527 target:** aim first at **sub-1B LLMs** (Gemma-3-270m, Qwen3-0.6B, TinyLlama-1.1B) with **INT8 activations + INT8/INT16 weights** (native Acuity 6.12 support). Don't bank on native 4-bit weights until you can move to a newer Acuity release (6.21+ or acuitylite). Plan for KV-cache external to the NBG graph (host-managed, model exported per-step with fixed shapes — the same pattern that worked for your CitriNet 300/500-frame fixed-length export).

---

## 1. Verisilicon Official GitHub — Full Repo Inventory

Source: https://github.com/orgs/VeriSilicon/repositories (27 public repos as of 2026-07-15)

### 1.1 Directly LLM-relevant (star / last-update / relevance)

| Repo | Stars | Last update | Officially blessed? | Relevance |
|---|---|---|---|---|
| [VeriSilicon/acuity-models](https://github.com/VeriSilicon/acuity-models) | 158 | 2026-07-13 | Yes | Model zoo. LLM catalog: Qwen (1.5-0.5B → Qwen3-8B, incl. GPTQ-Int4 / GPTQ-Int8 / Q4_0-GGUF variants), LLaMa (TinyLlama-1.1B, Llama-2-7b, Llama-3.2-1B/3B), DeepSeek-R1 distills, Phi-2/3, Gemma-2-2B / Gemma-3 (270m/1B/4B, incl. `gemma-3-1b-it-qat-q4_0-gguf`), MiniCPM-1B, ChatGLM3-6b, T5-small, Whisper (tiny/base/small/large-v3-turbo), Llava-1.5-7b, InternVL3 1B/8B, FastVLM-0.5B, Qwen2.5-VL, Qwen3-VL, OpenVLA-7b, SD-XL, EmbeddingGemma, OuteTTS. |
| [VeriSilicon/acuitylite](https://github.com/VeriSilicon/acuitylite) | 22 | 2026-07-14 | Yes | "End-to-end neural-network deployment tool." Successor / lightweight variant of the closed-source Acuity Toolkit. Importers: Caffe / DarkNet / ONNX / TF / TFLite. Exporters: TFLite, TIM-VX. Quant: asymmetric uint8 + symmetric int8 (no 4-bit here). Docs: [verisilicon.github.io/acuitylite](https://verisilicon.github.io/acuitylite/README.html). |
| [VeriSilicon/TIM-VX](https://github.com/VeriSilicon/TIM-VX) | 258 | 2026-03-30 (v1.2.22, 2025-01-08 release) | Yes | Device-side C++ tensor interface. "150+ operators." Backend for Android NN, TFLite (via delegate), Tengine, TVM, Paddle-Lite, OpenCV, ONNXRuntime. |
| [VeriSilicon/tflite-vx-delegate](https://github.com/VeriSilicon/tflite-vx-delegate) | 48 | 2026-03-12 | Yes | TFLite external delegate → TIM-VX. Requires "Vivante SDK ≥ 6.4.22 && ovxlib ≥ 1.2.26 && viplite ≥ 2.0.0" — matches your v2.0 driver, but 6.4.22 predates your Acuity 6.12 quite a bit. Can consume NBG (Acuity-compiled) models via a cache-load path. |
| [VeriSilicon/LiteRT-LM](https://github.com/VeriSilicon/LiteRT-LM) | 0 | 2025-12-23 (v0.8.0, Nov 2025) | Yes (fork) | Fork of `google-ai-edge/LiteRT-LM`. C++ LLM runtime, `.litertlm` package format. Google upstream packages Gemma3-1B (4-bit, 557MB), Gemma-3n-E2B/E4B (4-bit), Phi-4-mini (8-bit), Qwen2.5-1.5B (8-bit), FunctionGemma-270M (8-bit). Fork target: presumably to add a VeriSilicon backend. As of the fetch the README still only lists Qualcomm/MediaTek NPU acceleration. |
| [VeriSilicon/LiteRT](https://github.com/VeriSilicon/LiteRT) | (Apache 2.0) | 2026-04-01 | Yes (fork) | Fork of Google LiteRT (the on-device ML/GenAI framework). Same story — VSI likely adding a delegate. |
| [VeriSilicon/FlagGems](https://github.com/VeriSilicon/FlagGems) | 0 | 2025-05-09 (fork), 506 commits on vsi-main | Yes (fork) | Fork of FlagOpen FlagGems (Triton-based LLM operator library). Ops present: LayerNorm, RMSNorm, skip_rms_norm, skip_layer_norm, apply_rotary_position_embedding, GELU, SiLU, ReLU, sigmoid, tanh, softmax, log_softmax, dropout, group normalization. Backend advertised in README: NVIDIA GPU (fp16/fp32/bf16). VSI hookup would be via `triton-vsi-backend`. |
| [VeriSilicon/triton-vsi-backend](https://github.com/VeriSilicon/triton-vsi-backend) | 7 | 2025-05-09 | Yes | "Backend for triton language to compile and execute triton kernel." The bridge that would let FlagGems ops run on Vivante hardware. |
| [VeriSilicon/vsi-pjrt-plugin](https://github.com/VeriSilicon/vsi-pjrt-plugin) | 7 | 2024-04-28 | Yes | PJRT (XLA/JAX) plugin for VeriSilicon NPU. |
| [VeriSilicon/nn-sl](https://github.com/VeriSilicon/nn-sl) | 5 | 2024-01-06 (v23.12) | Yes | Android NNAPI support library over TIM-VX. Not LLM-focused. |
| [VeriSilicon/OpenNMT-Tokenizer](https://github.com/VeriSilicon/OpenNMT-Tokenizer) | — | 2025-03-18 | Yes | Fast BPE tokenizer library. Useful as a companion to any LLM runtime. |
| [VeriSilicon/triton-shared](https://github.com/VeriSilicon/triton-shared) | 1 | 2025-01-07 | Yes (fork) | Triton middle-layer for compilation. |
| [VeriSilicon/ZenCompiler](https://github.com/VeriSilicon/ZenCompiler) | — | 2025-03-21 (**archived**) | Was official | "Ultimate AI compiler based on MLIR." Archived. |
| [VeriSilicon/VPEX](https://github.com/VeriSilicon/VPEX) | — | 2025-03-18 (**archived**) | Was official | VeriSilicon PyTorch EXtension. Archived. |
| [VeriSilicon/onnxruntime](https://github.com/VeriSilicon/onnxruntime) | — | 2024-11-21 (**archived**) | Was official | Their ONNX Runtime fork. Archived — implies a strategy shift toward LiteRT + TFLite delegate + TIM-VX rather than an ORT execution provider. |
| [VeriSilicon/tvm](https://github.com/VeriSilicon/tvm) | 9 | 2021-12-31 | Stale | TVM fork. Not maintained recently. |
| [VeriSilicon/pytorch](https://github.com/VeriSilicon/pytorch) | — | 2025-01-16 | Yes (fork) | PyTorch fork. |
| [VeriSilicon/tensorflow](https://github.com/VeriSilicon/tensorflow) | 4 | 2024-11-20 | Yes (fork) | TF fork. |
| [VeriSilicon/mlcommons-inference](https://github.com/VeriSilicon/mlcommons-inference) | — | 2022-05-25 | Yes (fork) | MLPerf inference reference. |
| [VeriSilicon/acuity-dataset](https://github.com/VeriSilicon/acuity-dataset) | 6 | 2019-01-29 | Legacy | Acuity dataset scripts. |
| [VeriSilicon/caffe](https://github.com/VeriSilicon/caffe) | 3 | 2018-08-14 | Legacy | Caffe fork. |
| [VeriSilicon/Vinaro](https://github.com/VeriSilicon/Vinaro) | 1 | 2023-01-06 | Yes | "Vinaro open source SDK" — unclear scope. |
| [VeriSilicon/VGLite_4.x](https://github.com/VeriSilicon/VGLite_4.x), VGLite_Tests, vpe, ffmpeg, sworker | — | various | Yes | Graphics / video codec — not LLM. |

**Notable absences.** There is no `acuity-toolkit` (that binary is distributed to licensees only, not on GitHub), no `viplite` (proprietary driver), no `NBGlinker` source (v2.0 driver internal). Also no dedicated `llm-examples` repo.

### 1.2 Community forks / independent efforts (unofficial)

| Repo | Stars | Notes |
|---|---|---|
| [MaverickLong/MLIR-TIM-VX](https://github.com/MaverickLong/MLIR-TIM-VX) | 3 | TOSA v1 → custom `timvx` dialect → C++ lowering. Tested on Radxa Cubie A7Z (Allwinner A733). ResNet-50 = 8.0 ms vs 7.3 ms with official Acuity — near-parity on CNN. **Explicitly states: "There is currently no support for Transformers etc., so you cannot run LLM on the thing yet."** INT8 only. |
| [antkillerfarm/TIM-VX](https://github.com/antkillerfarm/TIM-VX) | — | Personal fork of TIM-VX. |
| [nxp-imx/tim-vx-imx](https://github.com/nxp-imx/tim-vx-imx) | — | NXP's downstream TIM-VX for i.MX. |
| [nxp-imx/tflite-vx-delegate-imx](https://github.com/nxp-imx/tflite-vx-delegate-imx) | — | NXP's downstream delegate. |
| [torizon/tflite-vx-delegate-imx-deb](https://github.com/torizon/tflite-vx-delegate-imx-deb) | — | Toradex Debian packaging. |
| [jackhe183/radxa-dev](https://github.com/jackhe183/radxa-dev) | — | Complete workflow for enabling NPU + AI inference on Radxa Cubie A7Z (A733). No pegasus/LLM examples. |

---

## 2. TIM-VX / tflite-vx-delegate — LLM Op Coverage

Source: `VeriSilicon/TIM-VX/src/tim/vx/ops/README.md` at main branch (fetched raw).

### 2.1 Transformer-relevant ops that ARE mapped ✅

Add, Sub, Multiply, Divide, Pow, Sqrt, **Rsqrt**, Square, Neg, Abs, Exp, Log, Sin, **Erf** (accurate GELU), Sigmoid, Tanh, **Softmax**, **LogSoftmax**, **Swish** (a.k.a. SiLU), HardSwish, **Gelu**, HardSigmoid, Mish, Selu, Celu, SoftSign, ReLU/ReLU1/ReLU6, LeakyRelu, PRelu, ELU, **Matmul** (MATRIXMUL), **FullyConnected**, **LayerNormalization** (LAYER_NORM), InstanceNormalization, BatchNorm, L2Normalization, **ReduceMean/Sum/Max/Min/Prod/Any/All**, **Moments** (mean+variance), **Reshape**, **Transpose** (PERMUTE), **Concat**, **Slice**, **StridedSlice**, Split, Unstack, Stack, Squeeze, Tile, Cast, **Gather**, **Gather_elements**, **GatherNd**, **ScatterND**, **EmbeddingLookup**, **HashtableLookup**, **Select** (tf.where — usable for masking), Equal/NotEqual/Greater/Less/GEq/LEq, LogicalAnd/Or/Not, Clip, Ceil/Floor/Round, Sign, Mod, CumSum, Rcp (1/x), OneHot, **Topk** ("mapped, limited support"), Broadcast (EXPAND_BROADCAST).

### 2.2 LLM-critical ops that are NOT explicitly present ❌ / require composition

| Op | Status | Recommended workaround |
|---|---|---|
| **RMSNormalization** | Not in TIM-VX op table (ONNX opset-23 op). | Compose: `x * Rsqrt(ReduceMean(Square(x)) + eps) * gamma`. All primitives are mapped. |
| **RotaryEmbedding (RoPE)** | Not present. **`Cos` is not in the TIM-VX ops README** — only `Sin` is listed. | (a) Precompute cos/sin tables offline and bake as constants into the graph; freq_ids become an input. (b) If Cos truly missing, use identity `cos(x) = sin(x + π/2)`. Then apply pairwise rotation via Mul + Add + Slice/Concat. |
| **Cos** | Not explicit in ops README. Possibly hidden as an internal / composed op in pegasus. Verify with a probe ONNX. | See above. |
| **ScaledDotProductAttention / MultiHeadAttention** | Not present as a fused op. | Compose from MatMul, Softmax, Mul, causal mask via Select+Constant. Increases graph size but should be legal for pegasus import. |
| **GroupedQueryAttention (GQA)** | Not present. | Emulate as MHA with Q reshape/broadcast against KV heads (Gather + broadcast on K/V). |
| **Causal Mask** | No dedicated op. `SEQUENCE_MASK` marked TBD. | Precompute additive mask (0 / -inf) as a constant input, apply via Add before Softmax. |
| **KV-cache** | No hardware/graph-level primitive. Same as with your CitriNet fixed-length trick. | Host-side KV-cache buffer. Either export a single-step decode graph with fixed context length (paged into cache manually), or roll and re-run. VeriSilicon's own solution in `LiteRT-LM` appears to follow the same pattern. |
| **PagedAttention / FlashAttention** | Not present. | Not achievable on VIP9000 hardware; use static-shape re-execution. |
| **Dynamic shape / dynamic seq_len** | Pegasus historically pins shapes at import. Your CitriNet 300-frame / 500-frame separate NBs are the canonical mitigation. | Same pattern: build one NB per bucket (e.g. 64 / 128 / 256 / 512 context lengths). |
| **BF16 weights** | Advertised at IP level. Support in Acuity 6.12 for arbitrary graphs unclear. | Stick to INT8/INT16 activations & weights on NanoSI-Plus. |
| **INT4 (W4A16 / GPTQ / AWQ) weights** | Advertised at IP-family level ("4-bit quantization and compression"); VeriSilicon's `acuity-models` catalog lists GPTQ-Int4 and Q4_0-GGUF entries. But Acuity 6.12 native quantizer options are asymmetric_affine uint8 / symmetric int8 (per your working CitriNet flow). No public docs of a 4-bit path for 6.12. | Assume 4-bit requires either (a) a newer Acuity version, or (b) mapping to an INT8-packed representation with the pegasus custom-op interface. Verify with VeriSilicon ML_Support if possible. |

### 2.3 Published LLM benchmarks on TIM-VX / tflite-vx-delegate

**None.** The tflite-vx-delegate README contains no benchmarks and no LLM-example programs. The `label_image.py` minimal example is CNN-only. No LLM benchmarks were found on VIP9000 in any public source.

---

## 3. Acuity Toolkit — Transformer / LLM Examples

### 3.1 Model zoo (`acuity-models` repo)

The zoo README (fetched from raw.githubusercontent.com) lists ~90 models. Under `### Transformer`:

- `BERTBase` → origin `uncased_L-12_H-768_A-12` (TF)
- `ViT` (Google Research)
- `Swin-Transformer` (Microsoft)

Under `### Large Language Model`:

- **LLM-Chat**
  - Qwen: Qwen1.5-0.5B, Qwen1.5-1.8B-Chat, Qwen2-1.5B-Instruct, Qwen2.5-{0.5B,0.5B-GPTQ-Int4,0.5B-Q4_0-GGUF,1.5B,3B,3B-Instruct,3B-Instruct-GPTQ-Int8,7B-Instruct}, Qwen3-{0.6B,0.6B-Base,0.6B-GPTQ-Int8,1.7B-Base,1.7B-GPTQ-Int8,4B,8B}, Qwen3.5-0.8B
  - MiniCPM-1B-sft-bf16
  - LLaMa: TinyLlama-1.1B, Llama-2-7b-hf, Llama-2-7b-chat-hf, Llama-3.2-3B, Llama-3.2-1B-Instruct
  - DeepSeek: DeepSeek-R1-Distill-Qwen-{1.5B,7B}, DeepSeek-R1-Distill-Llama-8B
  - GLM: chatglm3-6b
  - Phi: phi-2, Phi-3-mini-4k-instruct
  - Gemma: gemma-2-2b-it, gemma-3-1b-it, gemma-3-1b-it-qat-q4_0-gguf, gemma-3-270m-it
  - Google-T5: T5-small
- **Audio to Text**: Whisper base/small/tiny/large-v3-turbo
- **VLM**: Llava-1.5-7b-hf, InternVL3-1B/8B, InternVL3.5-1B, FastVLM-0.5B, gemma-3-4b-it, Qwen2.5-VL-3B/7B, Qwen3-VL-2B, OpenVLA-7b
- **TTS**: Llama-OuteTTS-1.0-1B
- **Text-to-Image**: stable-diffusion-xl-base-1.0
- **Translation**: GemmaX2-28-2B-v0.1
- **RAG**: embeddinggemma-300m

**Important caveat.** The zoo README documents *model presence in the catalog*, NOT step-by-step conversion recipes, NOT per-VIP-variant feasibility. Each entry is a JSON graph (viewable in Netron). There is **no** command-line usage example, **no** stated driver / NBG version, **no** target-hardware guidance in the README. Contact ML_Support@verisilicon.com for actual conversion recipes (referenced in acuitylite docs).

### 3.2 Acuity Toolkit User Guide v0.94 (2022-03-25)

Public copy on Allwinner's forum: `https://bbs.aw-ol.com/assets/uploads/files/1665722572028-...vivante.programming.acuity.toolkit.user.guide-v0.94-b-20220325.pdf` (~10+ MB, too large for WebFetch in one shot). Predates the LLM push and describes the pre-LLM op set and pegasus subcommands. Your Acuity 6.12 corresponds roughly to this generation. For newer coverage, `docs/acuity_toolkit/` v0.96 is the most recent English guide, and NPU_模型部署_开发指南.pdf (Chinese) shows a YOLOv5s recipe — those match what's already in your project's `docs/acuity_toolkit/`.

### 3.3 acuitylite documentation

- Page: [verisilicon.github.io/acuitylite](https://verisilicon.github.io/acuitylite/README.html)
- Importers: Caffe, DarkNet, ONNX, TensorFlow, TFLite
- Exporters: TFLite, TIM-VX (TFLite can also be run on TIM-VX via `tflite-vx-delegate`)
- Quantization: asymmetric UINT8 and symmetric INT8 only
- No LLM-specific instructions, no 4-bit in this tool

---

## 4. VIP9000 LLM Claims — Whitepapers & Press

### 4.1 Official VeriSilicon claims

- [Press release: NPU 100M shipped](https://verisilicon.com/en/PressRelease/NPU100M) — **The single most concrete LLM claim.** Quote: *"VIP9000 series NPU IP offers scalable and high-performance processing capabilities for both Transformer and Convolutional Neural Networks, and features 4-bit quantization and compression technologies to facilitate the deployment of AIGC and LLM algorithms such as Stable Diffusion and Llama 2 on embedded devices."* 72 licensees across 10 sectors. **No tokens/sec, no memory footprint, no per-variant benchmark.**
- [Vivante VIP9000 product page](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000) — TOPS range 0.5 → 20. Data formats INT8/INT16/FP16/BF16 + hybrid quantization. No LLM claim on this page.
- [Vivante NPU IP overview](https://www.verisilicon.com/en/IPPortfolio/VivanteNPUIP) — Lists Pico, VIP9000, VIP9400 (80 TOPS, data-center/automotive). No LLM detail.
- [VIP9000 Pico product page](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico) — wearables/IoT.
- [VIP9000NanoOi-FS ASIL-B](https://www.verisilicon.com/en/PressRelease/VIP9000NanoOi-FS) — safety-cert variant; mentions LLM+CNN inferencing capability.
- [Original VIP9000 launch (2019-08-07, prnewswire)](https://www.prnewswire.com/news-releases/verisilicon-launches-vip9000-new-generation-of-neural-processor-unit-ip-300897558.html) — pre-LLM era, CV focus.
- Chinese-language press mirror: [verisilicon.com/cn/PressRelease/VIP9000](https://verisilicon.com/cn/PressRelease/VIP9000)

### 4.2 Third-party echoes

- CNX Software: A527/T527/A733 have VIP9000 3 TOPS INT8 — [Datasheets released Jul 2025](https://www.cnx-software.com/2025/07/07/allwinner-a527-t527-and-a733-datasheets-user-manuals-and-linux-sdk-released/)
- Notebookcheck: [Allwinner A733 Processor specs](https://www.notebookcheck.net/Allwinner-A733-Processor-Benchmarks-and-Specs.951751.0.html) — VIP9000 3 TOPS INT8, no LLM benchmark listed.
- Radxa Cubie A7A NPU dev page (404'd from doc.radxa.com/en/cubie/a7a/... during this research) — was cited by CNX; the Radxa doc host presumably relocated it.

### 4.3 Closest analogue — NXP eIQ Gen-AI Flow (VeriSilicon-family NPU)

The most useful public precedent for LLM-on-Vivante:

- NXP i.MX 8M Plus uses VeriSilicon VIP8000 (2.3 TOPS) — same NPU family as VIP9000.
- [NXP eIQ Gen-AI Flow page](https://www.nxp.com/design/design-center/software/embedded-software/simplified-and-optimized-generative-ai-at-the-edge-with-eiq-genai-flow:GEN-AI-FLOW) (returned 404 to WebFetch, but summarized in NXP marketing search results): pre-optimized ONNX bundles for Llama, Qwen, Danube; runs on CPU or NPU on i.MX 95 / 93 / 8M Plus.
- [NXP LLM Studio Showcase — Llama 2-7B on i.MX 8M Plus + Ara-2 NPU](https://www.nxp.com/company/about-nxp/smarter-world-videos/LLM-STUDIO-SHOWCASING-LLAMA-2-7B-VID2)
- [NXP Transformer networks on i.MX 8M Plus (community)](https://community.nxp.com/t5/i-MX-Processors/Transformer-networks-on-i-MX-8M-Plus/m-p/1697658)
- [NXP i.MX Machine Learning User's Guide UG10166 (2026-03)](https://www.nxp.com/docs/en/user-guide/UG10166.pdf) — likely contains transformer-op configuration for the VIP-delegate path (worth downloading for your project).
- **Caveat**: NXP i.MX 95 has moved off VeriSilicon to their own Neutron NPU. But the 8M Plus / 93 flow uses the same Vivante lineage as T527's VIP9000-NanoSI-Plus.

---

## 5. Community Discussion — Forums / Issues / Blogs

### 5.1 GitHub Issues / PRs

- **[microsoft/onnxruntime#28244 — Add VIPLite Execution Provider for Allwinner VIP9000 NPU (A733/T527)](https://github.com/microsoft/onnxruntime/issues/28244)** — the single most on-point community post for T527. Proposes an ONNX Runtime EP over `libVIPhal.so`. Closed as stale. No PR. No benchmark. Sole hardware claim: *"VeriSilicon VIP9000, 3 TOPS, supports INT8/INT16/FP16/BF16."* Notes that Allwinner ACUITY Toolkit is publicly available on GitLab without NDA. Motivation: enabling Immich (photo management) NPU acceleration.
- No open TIM-VX issue found for RMSNorm/RoPE/GQA support.
- No Acuity-models issue thread on LLM conversion recipes.

### 5.2 Forums

- **[Armbian: MLIR Support Framework for Allwinner A733 / T527](https://forum.armbian.com/topic/59659-mlir-support-framework-for-allwinner-a733-t527/)** — the MaverickLong/MLIR-TIM-VX project (see §1.2). Motivation: *"Radxa and VeriSilicon advertised NPU MLIR support that didn't yet exist."* ResNet-50 8.0ms vs Acuity 7.3ms on Cubie A7Z. **Explicitly no transformer support yet.**
- **[Armbian: Orange Pi 4 Pro (A733) SBC](https://forum.armbian.com/topic/55919-35-orange-pi-4-pro-%E2%80%93-an-allwinner-a733-edge-ai-sbc-with-up-to-16gb-lpddr5-wifi-6-npu/)** — hardware discussion, no LLM.
- **[NXP Community: 8M Plus Capabilities — LLM & CV Models](https://community.nxp.com/t5/i-MX-Processors/8M-Plus-Capabilities-LLM-amp-CV-Models/td-p/2104486)** — customer questions about running LLMs on the closest analogue chip.
- **Allwinner-forum (bbs.aw-ol.com)** — hosts the Acuity Toolkit User Guide PDF, few active LLM threads.

### 5.3 Chinese-language search (Zhihu / CSDN)

- Search on Zhihu for `Vivante NPU 大模型` / `T527 LLM 部署` returned **no dedicated posts**. All Chinese-language "端侧大模型" (edge LLM) traffic points to Ascend / Rockchip / Qualcomm / MNN, not Vivante.
- Allwinner-focused CSDN posts (`全志T527`) discuss the SoC's 2-TOPS NPU generically but do not describe an LLM deployment.
- MNN 3.3 added NPU support (Qualcomm), which is the more common Chinese-community edge-LLM path today.

### 5.4 English blogs

- [Radxa docs — Vivante NPU SDK Dev Guide](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie_acuity_sdk) (returned 404 during fetch — path may have moved; original CNX reference confirms it exists)
- No public LLM-on-Vivante tutorial blog found. This is genuinely a green-field area on the T527 side.

---

## 6. Practical Recommendations for T527 / VIP9000-NanoSI-Plus

Distilled from the above and from your existing CitriNet workflow.

### 6.1 Model size sanity check

VIP9000-NanoSI-Plus @ ~3 TOPS INT8 with typical T527 memory (2–8 GB LPDDR4/5 shared). Rough feasibility:
- **Feasible (baseline):** Whisper-tiny / -base (~40–75M), Gemma-3-270M, Qwen3-0.6B, TinyLlama-1.1B (INT8 weights).
- **Marginal:** Qwen2.5-1.5B, Phi-3-mini (~3.8B) — requires memory streaming or aggressive INT8/INT4 quantization; single-token latency likely > 500 ms.
- **Impractical without newer HW/4-bit support:** Llama-2-7B, Llama-3-8B, ChatGLM3-6B, InternVL3-8B.

### 6.2 Conversion recipe candidate (best-guess starting point)

Given your existing pegasus-in-Docker + LD_LIBRARY_PATH pattern:

1. **Export the target LLM to ONNX with fixed shapes.** Use HuggingFace `optimum-cli export onnx --task text-generation-with-past` and fix `seq_len` to a small bucket (e.g. 128). Provide `past_key_values` as static-shape external inputs.
2. **Simplify with onnxsim** (as with CitriNet) — will help fold RoPE composed ops.
3. **Probe pegasus import.** Expect these failure modes:
   - `Cos` op unmapped → precompute rotary sin/cos as constant inputs.
   - `Where` / dynamic mask → precompute causal mask as static constant.
   - `LayerNorm` with dynamic axis → set explicit axis.
   - `MatMul` with rank > 4 → reshape to 4D.
4. **Quantize INT8 asymmetric_affine (as CitriNet).** Consider `--algorithm moving_average` if `kl_divergence` degrades PPL. Provide calibration prompts (100–1000 short texts) in the same format as `inputmeta.yml`.
5. **Export NBG** with the same `VIP9000NANOSI_PLUS_PID0X10000016` optimize flag your CitriNet pipeline uses.
6. **Runtime**: host-managed KV-cache in `awnn`-style code (mirroring `examples/libawnn_viplite/awnn_lib.c`), one `awnn_run` per generated token, updating the past_kv static input each step.

### 6.3 If native LLM support is blocked

Fallback ideas ordered by realism:

1. **Encoder-only tasks first** (BERT/ViT/Swin/Whisper) — the acuity-models zoo already includes converted graphs. Try to reproduce BERTBase INT8 on your T527 as a smoke test of the transformer-op path before touching Llama.
2. **Hybrid CPU+NPU decode**: run attention on CPU (via `llama.cpp` or a small custom kernel), offload only the MLP + FFN MatMul-heavy blocks to NPU as a submodule. Precedent: RKLLM effectively partitions similarly.
3. **Ask VeriSilicon for an Acuity 6.21+ or a targeted LLM support drop**. NXP's public Gen-AI Flow suggests VeriSilicon has internal LLM enablement — the question is whether they'll expose it for T527's VIP9000-NanoSI-Plus revision.
4. **Watch `VeriSilicon/LiteRT-LM`** — if a VSI backend lands, it becomes the fastest path.

### 6.4 Immediate probes worth running this week

- Take one of the Acuity-zoo transformer JSONs (`bert_base_vsi_frozen.json` or `vit.json`) into pegasus 6.12 on your setup and check that import succeeds. This calibrates what your version of the toolkit can already digest.
- Export a tiny 1-layer decoder-only transformer to ONNX and run it through the full pegasus flow. Failure modes here map directly to what you'll hit on a real LLM.
- Contact `ML_Support@verisilicon.com` referencing acuitylite docs — ask specifically: (a) is INT4/GPTQ supported in Acuity 6.12 or only ≥6.21? (b) is there an "LLM enablement pack" for VIP9000-NanoSI-Plus? (c) is there a canonical KV-cache-friendly export pattern?
- Read `docs/acuity_toolkit/NPU模块开发指南/NPU_量化调试_参考指南.pdf` for hybrid-quantization instructions — the LLM-friendly path likely goes through hybrid uint8/int16 (not INT4) on your current toolkit.

---

## 7. Structured Index of Sources

### 7.1 VeriSilicon official
- [GitHub org](https://github.com/orgs/VeriSilicon/repositories)
- [Acuity Models zoo (repo)](https://github.com/VeriSilicon/acuity-models) — [live catalog](https://verisilicon.github.io/acuity-models/)
- [Acuity Models zoo — raw README (with all LLM entries)](https://raw.githubusercontent.com/VeriSilicon/acuity-models/master/README.md)
- [acuitylite (repo)](https://github.com/VeriSilicon/acuitylite) — [docs](https://verisilicon.github.io/acuitylite/README.html)
- [TIM-VX (repo)](https://github.com/VeriSilicon/TIM-VX) — [op list](https://github.com/VeriSilicon/TIM-VX/blob/main/src/tim/vx/ops/README.md) — [customized_op doc](https://github.com/VeriSilicon/TIM-VX/blob/main/docs/customized_op.md) — [releases](https://github.com/VeriSilicon/TIM-VX/releases)
- [tflite-vx-delegate (repo)](https://github.com/VeriSilicon/tflite-vx-delegate)
- [LiteRT-LM (VSI fork)](https://github.com/VeriSilicon/LiteRT-LM) — [LiteRT (VSI fork)](https://github.com/VeriSilicon/LiteRT)
- [FlagGems (VSI fork)](https://github.com/VeriSilicon/FlagGems) — [triton-vsi-backend](https://github.com/VeriSilicon/triton-vsi-backend) — [vsi-pjrt-plugin](https://github.com/VeriSilicon/vsi-pjrt-plugin)
- [nn-sl (Android NNAPI)](https://github.com/VeriSilicon/nn-sl)
- [ZenCompiler (archived)](https://github.com/VeriSilicon/ZenCompiler) — [VPEX (archived)](https://github.com/VeriSilicon/VPEX) — [onnxruntime (archived)](https://github.com/VeriSilicon/onnxruntime)
- [Acuity Toolkit User Guide v0.94 PDF (mirror on aw-ol.com)](https://bbs.aw-ol.com/assets/uploads/files/1665722572028-29acddec-fca6-481a-ba20-660cbf0705eb-vivante.programming.acuity.toolkit.user.guide-v0.94-b-20220325.pdf)

### 7.2 VeriSilicon press / whitepapers
- [Press release: NPU 100M shipped (LLM/Llama 2 claim)](https://verisilicon.com/en/PressRelease/NPU100M)
- [VIP9000 launch press release (2019)](https://www.verisilicon.com/en/PressRelease/VIP9000) — [prnewswire mirror](https://www.prnewswire.com/news-releases/verisilicon-launches-vip9000-new-generation-of-neural-processor-unit-ip-300897558.html)
- [Vivante VIP9000 product page](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000)
- [Vivante VIP9000 Pico](https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000Pico)
- [Vivante NPU IP overview](https://www.verisilicon.com/en/IPPortfolio/VivanteNPUIP)
- [VIP9000NanoOi-FS ASIL-B](https://www.verisilicon.com/en/PressRelease/VIP9000NanoOi-FS)
- [VIP9000 + ZSP in iCatch automotive SoC](https://www.verisilicon.com/en/PressRelease/VIP9000andZSPAdoptedbyiCatch)
- [Chinese press mirror (VIP9000)](https://verisilicon.com/cn/PressRelease/VIP9000)

### 7.3 NXP (VeriSilicon-family NPU precedent)
- [NXP eIQ Gen-AI Flow](https://www.nxp.com/design/design-center/software/embedded-software/simplified-and-optimized-generative-ai-at-the-edge-with-eiq-genai-flow:GEN-AI-FLOW)
- [NXP LLM Studio — Llama 2-7B on i.MX 8M Plus](https://www.nxp.com/company/about-nxp/smarter-world-videos/LLM-STUDIO-SHOWCASING-LLAMA-2-7B-VID2)
- [NXP i.MX Machine Learning User's Guide UG10166](https://www.nxp.com/docs/en/user-guide/UG10166.pdf)
- [NXP Community: Transformer networks on i.MX 8M Plus](https://community.nxp.com/t5/i-MX-Processors/Transformer-networks-on-i-MX-8M-Plus/m-p/1697658)
- [NXP Community: 8M Plus LLM & CV Models](https://community.nxp.com/t5/i-MX-Processors/8M-Plus-Capabilities-LLM-amp-CV-Models/td-p/2104486)
- [NXP i.MX 95 — moving off VeriSilicon to eIQ Neutron NPU (EE Times)](https://www.eetimes.com/nxp-goes-in-house-for-application-processor-npu/)
- [nxp-imx/tim-vx-imx](https://github.com/nxp-imx/tim-vx-imx) — [nxp-imx/tflite-vx-delegate-imx](https://github.com/nxp-imx/tflite-vx-delegate-imx)

### 7.4 Allwinner / A733 / T527 ecosystem
- [CNX Software — Allwinner A527/T527/A733 datasheets released (Jul 2025)](https://www.cnx-software.com/2025/07/07/allwinner-a527-t527-and-a733-datasheets-user-manuals-and-linux-sdk-released/)
- [Notebookcheck — Allwinner A733 specs](https://www.notebookcheck.net/Allwinner-A733-Processor-Benchmarks-and-Specs.951751.0.html)
- [Armbian forum — Orange Pi 4 Pro A733](https://forum.armbian.com/topic/55919-35-orange-pi-4-pro-%E2%80%93-an-allwinner-a733-edge-ai-sbc-with-up-to-16gb-lpddr5-wifi-6-npu/)
- [Armbian forum — MLIR support for A733/T527](https://forum.armbian.com/topic/59659-mlir-support-framework-for-allwinner-a733-t527/)
- [MaverickLong/MLIR-TIM-VX](https://github.com/MaverickLong/MLIR-TIM-VX)
- [ONNX Runtime issue #28244 — VIPLite EP for A733/T527](https://github.com/microsoft/onnxruntime/issues/28244)
- [Radxa Cubie A7A NPU dev doc (was live, currently 404 during fetch)](https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie_acuity_sdk)
- [jackhe183/radxa-dev](https://github.com/jackhe183/radxa-dev)
- [Allwinner-forum (bbs.aw-ol.com)](https://bbs.aw-ol.com/)

### 7.5 Related community
- [antkillerfarm/TIM-VX](https://github.com/antkillerfarm/TIM-VX)
- [torizon/tflite-vx-delegate-imx-deb](https://github.com/torizon/tflite-vx-delegate-imx-deb)
- [Rockchip precedent — airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — [Pelochus/ezrknn-llm](https://github.com/Pelochus/ezrknn-llm) — [Pelochus/ezrkllm-collection on HF](https://huggingface.co/Pelochus/ezrkllm-collection)

### 7.6 Background / theory
- [LayerNorm vs RMSNorm — MachineLearningMastery](https://machinelearningmastery.com/layernorm-and-rms-norm-in-transformer-models/)
- [ONNX RMSNormalization op (opset-23)](https://onnx.ai/onnx/operators/onnx__RMSNormalization.html)
- [Microsoft Research — low-bit quantization on edge](https://www.microsoft.com/en-us/research/blog/advances-to-low-bit-quantization-enable-llms-on-edge-devices/)
- [Anton Maltsev — Edge AI Boards tier list 2026](https://medium.com/@zlodeibaal/the-ultimate-tier-list-for-edge-ai-boards-running-llms-and-vlms-in-2026-da06573efcd5)

---

## 8. What This Report Deliberately Does NOT Claim

- **No confirmed running LLM on VIP9000-NanoSI-Plus / T527 is documented publicly.** Everything in this report about T527 LLM feasibility is inference from op-tables, press releases, and precedent chips.
- **Tokens/sec numbers do not exist in public sources for any VIP9000 variant running an LLM.** The `T527_NPU_常见网络性能测试报告.pdf` in your project's `docs/acuity_toolkit/` covers CNN benchmarks only.
- **The `acuity-models` zoo listing an LLM is not a promise it runs on your chip.** It's a promise that Acuity's importer can parse the ONNX / HF graph. Whether pegasus 6.12 can then successfully quantize + NBG-export it for VIP9000-NanoSI-Plus is unverified.
- **The VeriSilicon LiteRT-LM fork has no publicly-visible VSI backend yet.** Its LLM claims are inherited from Google's upstream and target Qualcomm/MediaTek NPUs.

If any of these turn out to be different upon direct contact with VeriSilicon ML Support, update this document.
