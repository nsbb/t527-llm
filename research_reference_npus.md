# Reference NPU Platforms for T527 LLM Port

**Target hardware:** Allwinner T527 — Cortex‑A55 quad-core @ ~1.8GHz + Vivante VIP9000‑NanoSI‑Plus (~1 TOPS int8), single-channel DDR ~1.2GHz, 2–4GB RAM budget.
**Goal:** distill actionable, copyable architectural patterns from adjacent small‑NPU platforms that already ship (or have publicly attempted) on-device LLM inference.
**Research date:** 2026‑07‑15.

---

## TL;DR — What to copy, what to skip

1. **RK3588 / RKLLM is the only fully working small‑NPU LLM story.** Copy its API shape, fixed-size KV buffer, host-side tokenizer/chat-template, and w8a8 group-quant scheme. Do NOT assume w4a16 works on Vivante silicon — Rockchip's own first‑gen NPU couldn't do it despite exposing the flag.
2. **Same‑family Vivante NPUs (Amlogic A311D, NXP i.MX 8M Plus) have zero public LLM story.** NXP's own answer for GenAI was to skip Vivante and wait for i.MX 95's new "Neutron" accelerator. Treat this as a red‑flag signal: don't plan on running a whole LLM graph on VIP9000; plan on **CPU/NPU hybrid partitioning** from day one.
3. **The reusable architectural pattern across mllm-NPU / mobile-NPU research: keep only bulk INT8 GEMMs on the NPU; run LayerNorm/RMSNorm/softmax/attention‑reshape and ~0.1% activation outlier channels on CPU.** This aligns with the operator gaps documented for Vivante's `vsi_npu` ONNX Runtime EP.
4. **Baseline runtime should be llama.cpp CPU** with Q4_0 or Q4_K_M; NPU is a prefill accelerator, not a full replacement. Decode is bandwidth‑bound and will scale ~8–15× worse than RK3588 (not 6× as raw TOPS ratio suggests) because of T527's single‑channel DRAM.
5. **First target models (ranked):** Qwen2.5‑1.5B‑Instruct → Qwen3‑0.6B/1.7B → Gemma‑3‑1B (only model with first‑party QAT int4). Phi‑3‑mini and Qwen2.5‑3B are ruled out (memory / license).
6. **Rough expected performance on T527:** 0.5B model ≈ 4–6 tok/s decode; 1.5B model ≈ 1–3 tok/s decode (CPU‑only) with NPU‑assisted prefill possibly delivering meaningfully lower TTFT.

---

## Rockchip RK3588 — RKNN-Toolkit2 + RKLLM

*Primary source: [github.com/airockchip/rknn-llm](https://github.com/airockchip/rknn-llm), cross-checked with [releases](https://github.com/airockchip/rknn-llm/releases), [benchmark.md](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md), the [runtime C header](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h), open issues, and community HuggingFace conversions.*

### 1. Repo state — actively maintained

Latest tag is **v1.3.0 (17 Jun 2026)**. Release cadence has been roughly monthly-to-quarterly and is accelerating:

- **v1.3.0** (17 Jun 2026): Qwen3.5, Gemma4, SmolLM3 support; multi-EOS + `ignore_eos_token`; better cache-reuse strategy; tokenizer/embedding callbacks; long-context fixes on RK3576; memory/overflow fixes.
- **v1.2.3** (24 Nov 2025): InternVL3.5, DeepSeekOCR, Qwen3-VL; automatic cache reuse for embedding input; Gemma3n embedding support; external chat-template file loading.
- **v1.2.2** (30 Sep 2025): Gemma3n, InternVL3; multi-instance inference; LongRoPE; chat-template parsing fixes.
- **v1.2.1** (25 Jun 2025): RWKV7, Qwen3, MiniCPM4; RV1126B platform; function calling, cross-attention, multi-batch inference, perf-stats reporting.
- **v1.2.0** (08 Apr 2025): custom model conversion, chat-template config, context expanded to 16K, GRQ int4 algorithm, multimodal (InternVL2, Janus, Qwen2.5-VL), Python 3.9/3.11/3.12.

Rockchip clearly runs two parallel workstreams: "add model architecture" releases vs "fix runtime memory/cache" releases. New architectures (Qwen3.5 Gated DeltaNet, Gemma4) sometimes ship with transitional bugs — see §11 below.

**Copy this for T527:** split "add model architecture" from "fix runtime memory bugs" into separate release trains to reduce regression risk.

### 2. Supported models & confirmed working sizes

README-listed families: LLAMA, TinyLLAMA, Qwen2/2.5/3/3.5, Phi2/Phi3, ChatGLM3-6B, Gemma2/3/3n/4, InternLM2, MiniCPM3/4, TeleChat2, DeepSeek-R1-Distill, RWKV7, plus multimodal Qwen2-VL/Qwen3-VL, MiniCPM-V-2.6, Janus-Pro-1B, InternVL2/3-1B, SmolVLM/SmolLM3, DeepSeekOCR.

The [official benchmark table](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md) is the strongest evidence of **actually-tested** sizes on RK3588 (w8a8): Qwen2 0.5B, MiniCPM4 0.5B, Qwen3 0.6B, Qwen3.5 0.8B, TinyLLAMA 1.1B, Qwen2.5 1.5B, RWKV7 1.5B, InternLM2 1.8B, Gemma2 2B, Gemma3n 2B, Qwen3-VL 2B, Qwen3.5 2B, TeleChat2 3B, DeepSeekOCR 3B, Phi3 3.8B, MiniCPM3 4B, Qwen3.5 4B, Gemma4-E2B, and **ChatGLM3-6B**. No 7B/8B row appears in the vendor's own table, but community HF conversions confirm 7B+ works end-to-end, just slowly: [Qwen2.5-VL-7B-Instruct-rk3588](https://huggingface.co/dulimov/Qwen2.5-VL-7B-Instruct-rk3588-1.2.1), [Qwen2-VL-7B-Instruct-rk3588](https://huggingface.co/dulimov/Qwen2-VL-7B-Instruct-rk3588-1.2.1), a [NextCoder-7B w8a8_g128 conversion](https://huggingface.co/jamescallander/NextCoder-7B_w8a8_g128_rk3588.rkllm), and even [Qwen2.5-14B-Instruct-1M](https://huggingface.co/limcheekin/Qwen2.5-14B-Instruct-1M-rk3588-1.1.4). Llama-2/3/3.2 are covered only generically as "LLAMA" in the README — not itemized by version — but are conversion targets in the community ([ez-er-rkllm-toolkit](https://github.com/c0zaut/ez-er-rkllm-toolkit)). DeepSeek-R1-Distill-Qwen-1.5B ships as a first-party example directory in the repo.

### 3. Quantization schemes

- **w8a8**: 8-bit weight, 8-bit activation, "Normal" quant algorithm — the workhorse; every row in the RK3588 benchmark table uses it.
- **w4a16**: 4-bit weight, 16-bit activation. Plain RTN (round-to-nearest) is not accurate enough at 4-bit, so Rockchip added a **GDQ** ("grouped/GRQ" — naming varies release to release) higher-order algorithm specifically to hold accuracy at 4 bits.
- **Grouped variants**: `w4a16_g32/g64/g128`, `w8a8_g128/g256/g512` — "group" means the per-channel scale/zero-point is computed over a block of N weight elements instead of one scale for the whole channel; smaller group size (32) → finer granularity → better accuracy, worse compute/storage overhead. In the RK3588 table this cost is visible directly: ChatGLM3-6B is 4.98 tok/s at plain w8a8 vs 2.48–2.70 tok/s at `w8a8_g128` — group quant roughly halves throughput for the accuracy gain.
- **Hybrid quant**: mixes grouped and non-grouped quantization across layers by a specified ratio (accuracy/speed dial), plus a **GPTQ-int8** weight-only path for some models.
- **Important asymmetry**: [benchmark.md](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md) shows w4a16 numbers **only under RK3576/RV1126B**, not RK3588 — consistent with a [community report](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html) (2025-11-07): "the RK3588 only supports W8A8 quantization for LLM inference, not W4A16." This suggests RK3588's first-generation NPU cores lack a native int4 datapath that the newer RK3576/RV1126B NPU has.

**Copy this for T527:** don't assume int4-weight support is "free" just because the toolkit exposes the flag — verify the actual VIP9000-NanoSI-Plus op/datapath support for sub-8-bit weights before promising w4a16-equivalent speedups.

### 4. KV cache handling

The C API is deliberately thin: `rkllm_clear_kv_cache(handle, clear_all)` wipes the cache, and `rkllm_load_prompt_cache(path)` reloads a previously-saved prefix cache to skip re-prefilling shared system prompts. `max_context_len` must be `≤ 16384` and a multiple of 32 (default in examples: 4096). No documented paged-attention or ring-buffer scheme — behavior looks like a single pre-allocated KV buffer sized to `max_context_len`, i.e. simplest-possible fixed-size cache. NPU-memory OOM reports (e.g. [issue #335](https://github.com/airockchip/rknn-llm/issues/335), "failed to malloc npu memory, size: 3085697024") scale with both model size and context length, implying the KV cache — not just weights — lives in NPU-addressable memory. Recent releases (v1.2.3, v1.3.0) added "automatic cache reuse for embedding input" / "cache reuse strategy" — closer to prefix-caching than true paging.

**Copy this for T527:** a fixed-size, statically-allocated KV buffer (size = `max_context_len × hidden × layers × 2`) sized once at model load is dramatically simpler than paged attention, and Rockchip's own runtime doesn't bother with paging even at 16K context — a strong signal we should not over-engineer KV management for a 1 TOPS chip that will realistically run ≤2K context.

### 5. Tensor layout / .rkllm internals, RoPE/RMSNorm/SwiGLU

Rockchip does not publicly document whether `.rkllm` bundles one graph or separate prefill/decode graphs — this is **undocumented** in both the README and DeepWiki mirror. Indirect evidence: the general NPU-LLM pattern is two fixed-shape graphs — one chunked-prefill graph, one single-token decode graph reusing KV state — and RKLLM's API (separate handling of prompt-input vs. incremental decode) is consistent with this but **not confirmed** from source. On the operator side, the parallel [rknpu2](https://github.com/rockchip-linux/rknpu2) driver repo notes that "Rockchip NPU2 improves support for CPU operators including Cast, Sin, Cos, RMSNorm, ScalerND, and GRU" — implying RMSNorm has some native/accelerated path while Sin/Cos (used for RoPE) may still fall back to CPU-side computation.

### 6. Tokenizer & chat template

Tokenization stays firmly on the host CPU: every community `.rkllm` HuggingFace repo ships `tokenizer.json`/`tokenizer_config.json` alongside the `.rkllm` binary (e.g. [dulimov/Qwen3-Embedding-0.6B-rk3588](https://huggingface.co/dulimov/Qwen3-Embedding-0.6B-rk3588-1.2.1)). Chat templating is a lightweight, non-Jinja mechanism: `rkllm_set_chat_template(handle, system_prompt, prefix, postfix)` just wraps turns with string prefix/postfix.

**Copy this for T527:** keep tokenization and chat-template formatting entirely on the ARM host (sentencepiece/HF-tokenizers style), and keep the "chat template" concept as dumb string prefix/postfix concatenation rather than a templating engine.

### 7. Runtime C API shape

From [rkllm.h](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h):

```
rkllm_init(handle, param, callback)
  → rkllm_run / rkllm_run_async
       (RKLLMInput union: prompt / token / embedding / multimodal)
  → rkllm_destroy
```

Streaming is callback-driven: `LLMResultCallback(RKLLMResult*, userdata, LLMCallState)` fires per-token with states `RKLLM_RUN_NORMAL` / `_FINISH` / `_ERROR`; the callback return value can pause/abort generation (`rkllm_abort` also exists). Extras: `rkllm_load_lora`, `rkllm_set_cross_attn_params`, `rkllm_clear_kv_cache`, `rkllm_load_prompt_cache`. `RKLLMExtendParam` exposes `enabled_cpus_mask` (CPU affinity) and `n_batch` (>1 enables batching).

### 8. Real-world tokens/s on RK3588 (6 TOPS)

From [benchmark.md](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md), w8a8, RK3588:

| Model     | Size  | tok/s | TTFT (ms) | Mem (MB) |
|-----------|------:|------:|----------:|---------:|
| Qwen2     | 0.5B  | 41.58 |     145.9 |      670 |
| Qwen3     | 0.6B  | 32.91 |     199.1 |      791 |
| TinyLLAMA | 1.1B  | 24.43 |     243.9 |     1094 |
| Qwen2.5   | 1.5B  | 16.69 |     378.3 |     1689 |
| InternLM2 | 1.8B  | 15.45 |     380.1 |     1776 |
| Gemma2    | 2B    | 10.37 |     598.4 |     2779 |
| Phi3      | 3.8B  |  7.45 |    1017.3 |     3758 |
| MiniCPM3  | 4B    |  5.94 |    1408.5 |     4384 |
| ChatGLM3  | 6B    |  4.98 |    1352.9 |     5986 |

7B-class community numbers put decode around 3–5 tok/s ([cnx-software synthesis](https://www.cnx-software.com/2024/07/15/rockchip-rkllm-toolkit-npu-accelerated-large-language-models-rk3588-rk3588s-rk3576/)). Prefill/TTFT scales roughly with model size and is NPU-compute-bound; decode is clearly bandwidth-bound — tok/s roughly halves every ~2× in parameter count, tracking memory-read cost of weights per token rather than TOPS.

### 9. Scaling estimate to ~1 TOPS (T527)

RK3588's NPU is ~6 TOPS int8 (3×2 TOPS cores) paired with dual-channel LPDDR4X/LPDDR5. T527's VIP9000-NanoSI-Plus is ~1 TOPS int8 with a narrower single-channel DRAM (1.2GHz — a fraction of RK3588's effective bandwidth). Because **decode is bandwidth-bound, not TOPS-bound**, the naive "6× slower" estimate is optimistic — decode tok/s likely degrades **8–15×**, not 6×, once DRAM bandwidth becomes the bottleneck. Prefill, being compute-bound with high arithmetic intensity, should scale closer to the raw 6× TOPS ratio.

Conservative extrapolation:

| Model class          | RK3588 decode | T527 est. decode | T527 est. prefill scaling |
|----------------------|--------------:|-----------------:|--------------------------:|
| 0.5B (Qwen2 0.5B)    |    41.6 tok/s |     ~4–6 tok/s   |    ~6× slower TTFT        |
| 1.5B (Qwen2.5 1.5B)  |    16.7 tok/s |     ~1–3 tok/s   |    ~6× slower TTFT        |
| 3.8B (Phi3)          |     7.5 tok/s |    <1 tok/s      |    ~6× slower TTFT        |

**Confirm needed** — no direct VIP9000-NanoSI-Plus LLM benchmark exists publicly; this is an extrapolation, not a measurement.

### 10. RKLLM-Toolkit conversion flow

Python/PyTorch-based:

```
rkllm.load_huggingface(model)
  → configure target_platform, quant dtype/algorithm, num_npu_core
  → generate_data_quant.py  (produces data_quant.json)
  → rkllm.build(...)
  → export_rkllm()
```

Community automation exists ([ez-er-rkllm-toolkit](https://github.com/c0zaut/ez-er-rkllm-toolkit)) wrapping HF/GGUF → rkllm end to end. Quirk: brand-new architectures lag — [issue #472](https://github.com/airockchip/rknn-llm/issues/472) shows Qwen3.5's Gated-DeltaNet/hybrid-attention hitting a tensor shape mismatch at initial release. Custom architecture conversion was only enabled starting v1.2.0.

### 11. Pain points from GitHub Issues

- **NPU memory OOM**: [#335](https://github.com/airockchip/rknn-llm/issues/335) — 3GB malloc failure loading Qwen2.5-VL-3B language model; driver-version mismatch (0.9.6 vs 0.9.7) implicated but unresolved.
- **New-architecture breakage**: [#472](https://github.com/airockchip/rknn-llm/issues/472) — Qwen3.5 Gated DeltaNet tensor shape mismatch at launch.
- **Long-session leaks**: community reports of NPU runtime memory growth over long sessions, mitigated by periodic `rkllm_clear_kv_cache`.
- Quality-loss data specifically for RKLLM's w4a16 vs w8a8 is **not publicly quantified** by Rockchip.

### Copy-this-for-T527 summary

1. **Fixed-size KV buffer, no paging** — match Rockchip's simplicity.
2. **Tokenizer + chat template stay on host CPU**, string-based template, not embedded in the NPU binary.
3. **Separate "new model support" from "runtime stability" release cadence.**
4. **Budget for bandwidth-bound decode degrading worse than the raw TOPS ratio** — target 0.5B–1B models first; treat w4a16 as unproven on this NPU generation until measured.
5. **API shape**: `init → run/run_async (streaming callback) → destroy`, plus `clear_kv_cache` and `load_prompt_cache`. Mirror this directly.

---

## Other Small-NPU Platforms — Amlogic / NXP / MediaTek

### A) Amlogic A311D / S905X4 (5 TOPS, Vivante-derived)

The A311D/S905X4 NPU sits on the same lineage as T527's VIP9000: Amlogic ships `aml_npu_sdk` ([khadas/aml_npu_sdk](https://github.com/khadas/aml_npu_sdk)), which wraps the **Acuity Toolkit** — the identical `import → quantize → export` flow used in this repo's pipeline, just with a different `optimize` PID string (`VIPNANOQI_PID0X88` for VIM3 vs. T527's `VIP9000NANOSI_PLUS_PID0X10000016`) ([CNX Software](https://www.cnx-software.com/2020/01/13/getting-started-with-amlogic-npu-on-khadas-vim3-vim3l/), [Khadas Docs](https://docs.khadas.com/products/sbc/vim3/npu/npu-sdk)). This confirms the Acuity/`pegasus` conversion pattern is the vendor-standard flow across the whole Vivante ecosystem.

**LLM story: none, publicly.** A 2026 Khadas Community thread explicitly states "ollama doesn't have any support for NPUs in any of the khadas boards, they are more primarily cuda focused," and that the VIM3/VIM4 NPUs are "architecturally designed for convolutional neural networks rather than large language models," citing "memory bandwidth bottlenecks" and their "primary adaptation to better running convolution operation" ([Khadas Forum: Ollama and Khadas NPUs](https://forum.khadas.com/t/ollama-and-khadas-npus/22066)). The same thread notes that Khadas's Edge2 (RK3588) can run TinyLlama/Qwen small models via RKLLM — underscoring that it's specifically the Vivante-class NPU that lacks an LLM path.

**Assessment: no public LLM story on Vivante-derived Amlogic silicon.** Useful only as confirmation of the conversion-flow pattern, not as an inference-architecture reference.

### B) NXP i.MX 8M Plus / 93 / 95 — closest architectural sibling

This is the most important comparison. The i.MX 8M Plus NPU (2.3 TOPS) is a Vivante GPU/NPU using the same TIM-VX backend ([VeriSilicon/TIM-VX](https://github.com/VeriSilicon/TIM-VX); NXP fork [nxp-imx/tim-vx-imx](https://github.com/nxp-imx/tim-vx-imx)) that OpenVX/Acuity flows target elsewhere.

Critical finding: **NXP's own eIQ GenAI Flow demonstrator explicitly does NOT enable NPU acceleration on the i.MX 8M Plus.** Per [nxp-appcodehub/dm-eiq-genai-flow-demonstrator](https://github.com/nxp-appcodehub/dm-eiq-genai-flow-demonstrator/blob/main/README.md):
- **i.MX 95**: full LLM pipeline with NPU acceleration (new "Neutron" IP, not Vivante)
- **i.MX 8M Plus**: full pipeline but **CPU-only, no NPU acceleration**
- **i.MX 93**: partial (lighter models only)
- **i.MX 91/8MM/8MN**: RAG-only, minimal

The i.MX 95's NPU acceleration comes from **"Neutron", a different newer accelerator**, not the Vivante VIP used on 8M Plus/93 — NXP's own path to LLM-on-NPU was to leave Vivante behind entirely. Even so, the improvement is marginal: cited GenAI Flow numbers show **~9 tok/s on i.MX 95 w/ Neutron NPU vs. ~8.7 tok/s on i.MX 8M Plus CPU-only** for the Danube-500M demo ([NXP GenAI Flow product page](https://www.nxp.com/design/design-center/software/embedded-software/eiq-genai-flow-conversational-ai-software-pipeline-on-edge-devices:GEN-AI-FLOW), [NXP Community: Getting Started with GenAI Flow on i.MX95](https://community.nxp.com/t5/Generative-AI-LLMs/Getting-Started-with-GenAI-Flow-on-i-MX95-need-help-and-tips/td-p/2147372)).

Direct evidence of *why* Vivante struggles with transformers: NXP Community threads on the `vsi_npu` ONNX Runtime execution provider report broad operator fallback to CPU — DynamicQuantizeLinear, Mul, Div, Floor, Cast, Reshape, Add, PReLU all flagged unsupported at various times ([Unsupported Node in onnxruntime with vsi npu](https://community.nxp.com/t5/i-MX-Processors/Unsupported-Node-in-onnxruntime-with-vsi-npu/m-p/1536164), [ONNXRuntime i.MX 8M-Plus vsi_npu create vx tensor fail error](https://community.nxp.com/t5/i-MX-Processors/ONNXRuntime-i-MX-8M-Plus-vsi-npu-execution-provider-create-vx/td-p/1489565), [ONNX model slower on NPU than CPU](https://community.nxp.com/t5/eIQ-Machine-Learning-Software/ONNX-model-slower-on-NPU-than-CPU/m-p/1196957)). Mul/Div/Reshape/Cast are exactly the ops that saturate a transformer graph (LayerNorm, softmax, attention reshape/transpose). Reference NXP docs worth pulling: [AN14411 – Enabling eIQ Core NPU Delegates for i.MX Android Applications](https://www.nxp.com/docs/en/application-note/AN14411.pdf), [UG10166 i.MX Machine Learning User's Guide](https://www.nxp.com/docs/en/user-guide/UG10166.pdf).

**Assessment: useful reference, as a cautionary signal.** Same Vivante NPU family as T527, and the vendor's conclusion was "don't run LLMs on this NPU." Architectural takeaway: any T527 LLM port should assume LayerNorm/softmax/attention-reshape ops will not map cleanly to the Vivante graph compiler, and plan for CPU fallback of those subgraphs.

### C) MediaTek Genio 700/1200 / Dimensity APU

Genio embedded parts (Genio 700: 4 TOPS 5th-gen NPU; Genio 1200: 4.8 TOPS) use MediaTek's **Neuron SDK** (part of NeuroPilot), which compiles TFLite models via `ncc-tflite` into a proprietary DLA binary format, with a C/C++ Neuron Runtime API ([Neuron Compiler and Runtime docs](https://genio.mediatek.com/doc/iot-aihub/ai_hub/ai-workflow/neuron-sdk.html), [Genio 510/700-EVK ML guide](https://mediatek.gitlab.io/aiot/doc/aiot-dev-guide/master/sw/yocto/ml-guide/ml-g700-evk.html)). **No public LLM demos or benchmarks for Genio parts specifically were found.**

Adjacent evidence from MediaTek phone silicon (Dimensity 9300, APU 790): a "hardware generative AI engine" claimed 8× faster at transformer ops, mixed-precision INT4 quantization, NeuroPilot memory hardware compression; 7B LLMs at ~20 tok/s ([Dimensity 9300 press release](https://www.mediatek.com/press-room/mediatek-boosts-flagship-smartphone-performance-with-dimensity-9300-soc), [Embedded.com](https://www.embedded.com/mediatek-new-flagship-dimensity-9300-targets-on-device-llms/)). As of December 2025, Google's LiteRT NeuroPilot Accelerator brings this stack to Qwen3-0.6B, Gemma-3-270M/1B, Gemma-3n-E2B and EmbeddingGemma-300M as first-class targets ([Google Developers Blog](https://developers.googleblog.com/mediatek-npu-and-litert-powering-the-next-generation-of-on-device-ai/), [MarkTechPost coverage](https://www.marktechpost.com/2025/12/09/google-litert-neuropilot-stack-turns-mediatek-dimensity-npus-into-first-class-targets-for-on-device-llms/)) — but for Dimensity APU 790, not Genio's Neuron SDK/older APU generations.

**mllm-NPU (ASPLOS'25)** — most transferable architectural pattern regardless of vendor ([arXiv:2407.05858](https://arxiv.org/html/2407.05858v1)):
1. **Chunk-sharing execution graphs** — fixed 256-token chunks to avoid graph recompilation for variable-length prompts.
2. **Shadow outlier execution** — the ~0.1–0.3% of outlier activation channels run in float on CPU/GPU while the bulk INT8 matmuls run on NPU.
3. **Out-of-order subgraph scheduling** to minimize NPU pipeline stalls.

This CPU/NPU hybrid-outlier pattern matches the fallback strategy implied by the i.MX 8M Plus operator gaps and is the single most directly transferable architectural idea for T527.

**Assessment: no public LLM story for Genio specifically; Dimensity flagship story exists but not confirmed portable to Genio's smaller NPU.**

### Cross-cutting takeaways

1. All three vendors' **model-conversion tooling for Vivante-class NPUs converges on the Acuity/`pegasus` import→quantize→export flow** — T527's existing pipeline is the standard approach.
2. **No vendor has a working NPU-accelerated LLM story on Vivante-generation small NPUs.** NXP's own answer was to skip Vivante for GenAI. Full-NPU LLM inference on VIP9000-NanoSI-Plus (weaker than the 8M Plus's 2.3 TOPS) is not realistic without heavy hybrid CPU/NPU partitioning.
3. **Reusable architectural pattern** (mllm-NPU + i.MX vsi_npu operator-fallback evidence): split the graph so LayerNorm/softmax/attention-reshape/outlier channels run on CPU (fp32/int16) while bulk quantized matmuls run on NPU.
4. TIM-VX ([VeriSilicon/TIM-VX](https://github.com/VeriSilicon/TIM-VX)) is the common low-level backend across Amlogic/NXP/T527 Vivante chips and has no transformer/LLM operator set or examples as of this search — monitor its issue tracker rather than expecting near-term support.

---

## CPU Runtime Baselines

### llama.cpp on Cortex‑A55 class hardware

No benchmark repo publishes numbers for T527/Allwinner or RK3566/VIM3 specifically — [sbc-bench](https://github.com/ThomasKaiser/sbc-bench/blob/master/Results.md) only covers classic CPU/memory microbenchmarks. Closest proxy data comes from Cortex‑A72 (RPi4) and Cortex‑A76 (RPi5), which are ~1.3–1.8× faster per-core than A55 at the same clock.

- **RPi5 (Cortex‑A76 quad, no NPU)**: TinyLlama‑1.1B Q4_0 → **14.4 tok/s** decode; Q4_K_M → **12–18 tok/s** ([tinyweights.dev](https://tinyweights.dev/posts/run-llms-raspberry-pi-5/), [localaimaster.com](https://localaimaster.com/blog/llm-raspberry-pi-5)). 1.5B (Qwen2.5‑1.5B) → **5–15 tok/s**; 3B → **2–5 tok/s** ([arXiv:2511.07425](https://arxiv.org/html/2511.07425v1)). Sub‑1B models exceed 20 tok/s.
- **RPi4 (Cortex‑A72 quad)**: TinyLlama ~**5–7 tok/s**, Llama‑3.2‑1B ~**4–5 tok/s**, Llama‑2‑7B Q4_K_M ~**2–3 tok/s** ([medium.com/@thomasnahon](https://medium.com/@thomasnahon/llms-on-a-budget-testing-serving-frameworks-on-the-raspberry-pi-4-5fc56623840e)).
- **T527 (Cortex‑A55 quad, ~1.8GHz)** — no direct citation; extrapolating from RPi4 with A55 IPC/clock penalty: expect roughly **3–6 tok/s decode for a 1B model at Q4_0**, dropping to **1–3 tok/s for 3B**. **Confirm on-device.**

### Architecture notes relevant to porting

- **Prefill vs decode split**: prefill = compute-bound, dominated by large GEMM (batched matmul over the whole prompt); decode = memory-bound, dominated by GEMV (matvec, one token at a time) + KV-cache traffic ([Arm Learning Paths](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/2_llama.cpp_intro/)). **This is the split that matters for hybrid CPU/NPU designs**: NPU-friendly batched GEMM for prefill, CPU-friendly GEMV for decode.
- **ARM kernels**: llama.cpp's `ggml-cpu` uses NEON/SVE intrinsics plus ARM **i8mm** (8-bit matrix-multiply) and **dotprod** where available, calling Arm's **KleidiAI** micro-kernels — e.g. `kai_run_matmul_..._i8mm` for prefill GEMM, `..._dotprod` for decode GEMV ([Arm Learning Paths](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama_cpp_streamline/4_analyze_token_prefill_decode/)). **Cortex‑A55 does not implement i8mm/dotprod as standard** (those arrived with A76/A78/X1+ or optional extensions) — confirm the exact A55 ISA extensions on T527 before assuming these fast paths are available; if absent, llama.cpp falls back to plain NEON int8.
- **KV cache**: contiguous per-sequence buffer per layer. Paged KV (vLLM-style) is only experimental (`--kv-paged` flag) — [Discussion #21961](https://github.com/ggml-org/llama.cpp/discussions/21961), [deep-dive blog](https://rockyrunnr.github.io/posts/paged-attention-llama-cpp-deep-dive/). For T527, assume **contiguous per-sequence KV cache**.
- **Quantization block layouts** ([DeepWiki quant docs](https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques), [Discussion #2094](https://github.com/ggml-org/llama.cpp/discussions/2094)):
  - `Q4_0` / `Q8_0`: legacy, flat 32-element blocks, one fp16 scale per block, symmetric. `Q8_0` = 34 bytes/32 weights (8.5 bpw).
  - `Q4_K_M` / `Q4_K_S`: "k-quant" 256-element superblocks with sub-block (16 or 32) scales/mins, ~4.5 bpw for `_M`.
  - `IQ2_XXS/IQ3_XXS/IQ4_NL`: importance-matrix (imatrix) sub-4-bit; need calibration data. **For T527, IQ-quants add complexity for marginal gains at 1–3B scale** — start with Q4_0 / Q4_K_M.

### MLC-LLM

TVM-compiled, produces per-device `.so` binaries; Android via OpenCL 3.0 (GPU) or LLVM (CPU) ([mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm), [Android SDK docs](https://llm.mlc.ai/docs/deploy/android.html)). **No native VIP9000/Vivante backend** — accelerated path targets Mali/Adreno GPU via OpenCL/Vulkan; reports on non-Apple mobile GPUs show only 5–20% ALU utilization ([arXiv:2501.14794](https://arxiv.org/html/2501.14794v2)). **Not recommended as primary runtime for T527** (no OpenCL-capable GPU available); heavier build/toolchain than llama.cpp with no upside.

### MNN-LLM (Alibaba)

Actively maintained ([alibaba/MNN](https://github.com/alibaba/MNN), [MNN-LLM paper arXiv:2506.10443](https://arxiv.org/pdf/2506.10443)), claims prefill **8.6× faster than llama.cpp** and decode **2.3× faster** on Android CPU. NPU backends exist for **Huawei HiAI** and **Qualcomm QNN/Hexagon** ([MNN wiki/npu](https://github.com/alibaba/MNN/wiki/npu)) — **no Vivante/VIP9000 backend**. Any T527 NPU offload via MNN would require a custom backend. Its CPU path is worth benchmarking vs llama.cpp given the claimed prefill speedup.

### ExecuTorch

PyTorch edge runtime; added **Arm Ethos‑U NPU support with A8W4** in 2026 ([PyTorch blog](https://pytorch.org/blog/efficient-edge-ai-on-arm-cpus-and-npus/)). Ethos‑U is a different NPU family (Arm's own microNPU) — no Vivante backend. **Not directly portable to T527**, but worth tracking for its Arm-CPU delegate quality.

### Why NPU offload for LLMs is hard on VIP9000-class hardware

Research on mobile-NPU LLM inference (`llm.npu` / `mllm-NPU`, ASPLOS'25) is directly relevant: mobile NPUs support only **static input shapes**, while LLM prompts are variable length; and mobile NPUs typically **can't do per-group MatMul**, conflicting with standard group-wise weight quantization ([arXiv:2407.05858](https://arxiv.org/html/2407.05858v2), [ASPLOS paper](https://xumengwei.github.io/files/ASPLOS25-NPU.pdf)). Their solution (chunking prompts to fixed sizes, routing outliers to CPU/GPU, mixed block scheduling) is the template to follow if attempting VIP9000 prefill-offload — expect similar static-shape and per-tensor-quant constraints from the Acuity/Pegasus toolchain already in this repo.

---

## Small-LLM Candidate Shortlist (July 2026)

| Model | Params | License | Attention | KV heads | RoPE | Vocab / Tok | Tied embed | GGUF | ONNX |
|---|---:|---|---|---:|---|---|---|---|---|
| TinyLlama‑1.1B | 1.1B | Apache 2.0 | MHA (32H) | 32 (=Q) | trad, θ=10k | 32,000 BPE (Llama2) | No | Yes | [onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX](https://huggingface.co/onnx-community/TinyLlama-1.1B-Chat-v1.0-ONNX) |
| Qwen2.5‑0.5B | 0.5B | Apache 2.0 | GQA 14Q/2KV | 2 | RoPE θ | 151,646 BPE | Tied | Yes | community |
| Qwen2.5‑1.5B | 1.5B | Apache 2.0 | GQA 12Q/2KV | 2 | RoPE θ | 151,646 | Tied | Yes | community |
| Qwen2.5‑3B | 3B | **Qwen Research** (non-commercial) | GQA 16Q/2KV | 2 | RoPE θ | 151,646 | Tied | Yes | community |
| Qwen3‑0.6B | 0.6B | Apache 2.0 | GQA 16Q/8KV | 8 | RoPE θ | ~151k | Tied | Yes | community |
| Qwen3‑1.7B / 4B | 1.7B/4B | Apache 2.0 | GQA (KV≤8) | ~8 | RoPE θ | ~151k | Tied (≤4B) | Yes | community |
| Gemma‑3‑1B | 1B | Gemma custom (permissive but not OSI) | GQA + QK-norm | few | RoPE | **256k SentencePiece** | Tied | Yes, incl. **official QAT int4** | limited |
| Gemma‑3‑4B | 4B | Gemma | GQA | few | RoPE | 256k | Tied | Yes (QAT variants) | limited |
| Phi‑3‑mini‑3.8B / 3.5‑mini | 3.8B | **MIT** | GQA 32Q/8KV | 8 | RoPE | 32,064 BPE | ? | Yes | **Official MS int4 ONNX (AWQ)** — [nvidia/Phi-3.5-mini-Instruct-ONNX-INT4](https://huggingface.co/nvidia/Phi-3.5-mini-Instruct-ONNX-INT4), [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| MiniCPM3‑4B | 4B | Apache 2.0 + MiniCPM model license | dense | n/a | RoPE | n/a | n/a | Yes — [openbmb/MiniCPM3-4B-GGUF](https://huggingface.co/openbmb/MiniCPM3-4B-GGUF), Q4_K_M=2.47GB | limited |
| MiniCPM5/4‑1B | ~1B | Apache 2.0 | dense | n/a | RoPE | n/a | n/a | Yes — [Mungert/MiniCPM5-1B-GGUF](https://huggingface.co/Mungert/MiniCPM5-1B-GGUF) | limited |
| Llama‑3.2‑1B | 1B | Llama 3.2 Community | GQA 32Q/8KV | 8 | RoPE θ=500k | 128,256 | Tied | Yes | community |
| Llama‑3.2‑3B | 3B | Llama 3.2 Community | GQA 24Q/8KV | 8 | RoPE θ=500k | 128,256 | Tied | Yes | community |
| SmolLM2‑135M/360M/1.7B | 0.135–1.7B | Apache 2.0 | Llama2-style, SwiGLU | small | RoPE θ=10k | Llama2/BPE | Tied | Yes | **Official** — HF supports GGUF/Safetensors/ONNX natively |
| H2O‑Danube3‑500M/4B | 0.5B/4B | Apache 2.0 | adjusted Llama2 | n/a | RoPE | 32,000 (Mistral tok) | n/a | community | limited |

Sources: [Qwen2.5 report](https://arxiv.org/pdf/2412.15115), [Qwen2.5-0.5B card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen3 report](https://arxiv.org/html/2505.09388v1), [Gemma 3 report](https://arxiv.org/html/2503.19786v1), [Gemma 3 QAT blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/), [Phi-3 report](https://arxiv.org/pdf/2404.14219), [Llama-3.2-1B card](https://huggingface.co/meta-llama/Llama-3.2-1B), [SmolLM2 paper](https://www.researchgate.net/publication/388753897_SmolLM2_When_Smol_Goes_Big_--_Data-Centric_Training_of_a_Small_Language_Model), [H2O-Danube3 report](https://arxiv.org/abs/2407.09276), [MiniCPM GitHub](https://github.com/OpenBMB/MiniCPM).

### Approximate memory budget (int8 vs int4 weights, + fp16 KV @ 2k ctx)

Using `KV_bytes ≈ 2 × n_layers × kv_heads × head_dim × ctx_len × bytes/elem`:

- **TinyLlama‑1.1B** (MHA, 22L × 32kv × 64d): weights int8 ≈ 1.1GB / int4 ≈ 0.55GB; KV@2k fp16 ≈ **~370MB** — MHA makes KV cache unusually expensive.
- **Qwen2.5‑1.5B** (GQA, 2 kv heads, 28L): weights int8 ≈ 1.5GB / int4 ≈ 0.75GB; KV@2k fp16 ≈ **~60MB** — GQA's 2 KV heads make cache nearly free.
- **Qwen3‑1.7B** (GQA, ~8 kv heads): weights int8 ≈ 1.7GB / int4 ≈ 0.85GB; KV@2k fp16 ≈ **~230MB**.
- **Gemma‑3‑1B**: weights int8 ≈ 1.0GB / int4 (official QAT) ≈ **0.5GB** ([Google blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)); but 256k-vocab embedding table adds hundreds of MB (tied embeds help).
- **Phi‑3‑mini‑3.8B**: weights int8 ≈ 3.8GB / int4 ≈ 1.9GB (matches MS int4 ONNX ~2GB) — **too large for 2–4GB DDR budget alongside OS + app**.
- **MiniCPM3‑4B**: Q4_K_M GGUF measured at **2.47GB** — tight but plausible on a 4GB board.
- **SmolLM2‑360M**: weights int8 ≈ 0.36GB / int4 ≈ 0.18GB — trivially fits, capability ceiling low.

(Order-of-magnitude estimates from cited architecture parameters — **confirm actual footprint after quantization/export**.)

### Ranked recommendation for T527 (1 TOPS int8 NPU, 2–4GB DDR, Cortex‑A55 quad CPU fallback)

1. **Qwen2.5‑1.5B‑Instruct (Apache 2.0)** — best balance: GQA with only 2 KV heads keeps int8/int4 weight footprint (~0.75–1.5GB) and KV cache (~60MB) small; abundant GGUF/AWQ/GPTQ community quantizations to validate against; Apache 2.0 is unambiguous for commercial embedded use ([HF card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)). Target ~3–8 tok/s on A55 quad CPU with room for NPU-assisted prefill.
2. **Qwen3‑0.6B / 1.7B (Apache 2.0)** — newer training recipe than 2.5 at similar or smaller size, same GQA architecture, fully Apache-licensed across all sizes (avoids the 2.5‑3B licensing trap). Benchmark head-to-head with Qwen2.5‑1.5B for quality/speed.
3. **Gemma‑3‑1B (official QAT int4)** — the only candidate with **first-party QAT-trained int4 checkpoints** verified to preserve near-bf16 quality at 0.5GB ([Google blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/)) — removes a major quantization-risk unknown. Caveat: 256k vocab embedding table is a real memory tax; Gemma license is custom (not Apache/MIT) — legal check before shipping.

**Not recommended as primary targets**:
- **Phi‑3‑mini** — great tooling / official int4 ONNX, but 3.8B is oversized for 1 TOPS + 2–4GB.
- **Qwen2.5‑3B** — non-commercial research license blocks product use.
- **TinyLlama‑1.1B** — MHA KV overhead + older training data → strictly dominated by Qwen2.5/3 at similar footprint.

---

## Consolidated Architectural Recommendations for T527

Distilling the above into a T527-specific plan:

### API surface (mirror RKLLM)

```c
awllm_init(&handle, param, callback);
awllm_run(handle, RKLLMInput{prompt|tokens|embedding}); // streaming callback
awllm_clear_kv_cache(handle, all);
awllm_load_prompt_cache(handle, path);
awllm_destroy(handle);
```

Streaming callback: `cb(result*, userdata, state)` where state ∈ `{NORMAL, FINISH, ERROR}`. Return value can abort generation.

### Model file layout

- Split into two fixed-shape NPU graphs: **prefill** (chunked, e.g. 128- or 256-token chunk) + **decode** (single-token, reuses KV state).
- Fixed-size KV buffer allocated at load, sized by `max_context_len` (target 2048; multiple of 32).
- Tokenizer + chat template live in host-side files alongside the NPU binary; NPU binary does not embed them.

### Quantization

- **Start with w8a8 (int8 weights + int8 activations)** — the only scheme confirmed working on first-gen Vivante NPU class.
- Do not promise w4a16 until measured; the RK3588 experience shows silicon-level int4 datapath is not guaranteed.
- Calibration set: 100–200 representative prompts (following RKLLM's `data_quant.json` pattern).

### Op partitioning (the mllm-NPU pattern — most important for Vivante)

Given documented `vsi_npu` operator gaps (Mul/Div/Reshape/Cast/DynamicQuantizeLinear fall back to CPU), design for hybrid execution from day one:

| Op class | Where | Notes |
|---|---|---|
| Bulk INT8 GEMM (attention proj, FFN) | NPU | The 80% of FLOPs. Must run here. |
| RMSNorm / LayerNorm | CPU (or NPU if op available) | Check Vivante VIP9000-NanoSI-Plus op set — likely NPU-supported per rknpu2 notes. |
| RoPE (sin/cos application) | CPU | Sin/Cos are documented CPU-fallback on adjacent Vivante NPUs. |
| Softmax + attention reshape/transpose | CPU | These consistently trip up vsi_npu. |
| Embedding lookup | CPU | Cheap, not worth NPU roundtrip. |
| Sampling (argmax/top-k/top-p) | CPU | Trivially CPU. |
| Outlier channels (~0.1% of activations) | CPU float | mllm-NPU pattern — preserves quality at int8. |

### Performance envelope to plan for

| Model | Prefill (TTFT for 100 tokens) | Decode | Memory (int8) |
|---|---|---|---|
| Qwen2.5-0.5B / Qwen3-0.6B | ~1.5 s | ~4–6 tok/s | ~700MB |
| TinyLlama-1.1B | ~2.5 s | ~2–3 tok/s | ~1.1GB (+ 370MB KV @ 2k, MHA!) |
| Qwen2.5-1.5B / Qwen3-1.7B | ~4 s | ~1–3 tok/s | ~1.5–1.7GB |
| Gemma-3-1B (int4 QAT) | ~3 s | ~2–4 tok/s | ~0.5GB weights + big embed |

(All estimates extrapolated from RK3588 numbers with 6–15× bandwidth-adjusted degradation. **Not measured.**)

### First-quarter roadmap (implied)

1. **CPU baseline first** — llama.cpp with Q4_0 for TinyLlama, Qwen2.5-0.5B, Qwen2.5-1.5B. Establish reference tok/s on T527.
2. **Op inventory pass** — export each candidate model to ONNX, run through Acuity import, catalog which ops go NPU vs CPU-fallback. This determines partitioning.
3. **Prefill-on-NPU prototype** — fixed 128-token chunk, w8a8, decoder still on CPU. Measure TTFT delta.
4. **Full hybrid pipeline** — decode on NPU where bulk GEMM dominates; CPU handles the "glue" ops.
5. **Model choice locked** after (2) — pick whichever candidate has the cleanest NPU op-fallback profile.

### Known unknowns / confirm on-device

- Exact ARM ISA extensions available on T527's A55 (i8mm? dotprod?) — determines llama.cpp KleidiAI fast-path availability.
- Whether VIP9000-NanoSI-Plus has an int4 weight datapath (RK3588 does not; RK3576 does).
- Whether Acuity/Pegasus can compile a fixed-shape LLM decoder graph at all — no public example exists.
- KV cache placement — is Vivante NPU memory large enough to hold a 2k-context KV cache for a 1.5B model?
- Actual sustained DRAM bandwidth on T527 under NPU + CPU concurrent load.
