# Community & Third-Party LLM on T527-Class Hardware — Research Report

Compiled 2026-07-15. Target: Allwinner T527 (8x Cortex-A55 @ up to 1.8 GHz, Vivante VIP9000-NanoSI-Plus NPU ~2 TOPS INT8) with Acuity 6.12 → VIPLite runtime.

## TL;DR

- **No public project has run an LLM on Allwinner T527 or its sister SoCs (A527/A733/T536/H728) on the Vivante NPU.** The closest attempt (Anton Maltsev on Cubie A7A / A733) *failed* on transformers.
- **Vivante VIP9000 marketing** claims 4-bit quantization and Llama 2 / Stable Diffusion support, but no working community port has been demonstrated, and the low-end "NanoSI-Plus" variant in T527 is the least-provisioned of the family.
- **Reference blueprint exists on Rockchip RK3588** (RKLLM): TinyLlama-1.1B ~15 tok/s on 6 TOPS NPU. This is the plausible ceiling if a similar VIPLite-based port could be built.
- **CPU-only fallback (llama.cpp on 8x Cortex-A55)** is memory-bandwidth-bound. Extrapolating from Pi 4 (4x A72 @ 1.5 GHz → 8–12 tok/s TinyLlama Q4_K_M) and RK3588 A55-thread data, a realistic T527 CPU-only ceiling is **~3–6 tok/s for a 1.1B Q4_K_M model**, less for 1.5B, essentially unusable at 3B+.
- **RAM is the harder constraint than TOPS.** 1.9 GB compiled DeepSeek-R1-Distill-Qwen-1.5B (RKLLM) barely fits a 2 GB T527 board once OS+runtime are loaded. Target 0.5–1.1 B params at Q4.

---

## 1. Direct T527 LLM Projects

**Result: zero found.** No GitHub repo, blog post, video, or forum thread demonstrates an LLM running on any T527 board (BPI-M4-Berry variants, Avaota-A1, MangoPi Mcore-T527, WalnutPi 2B, Radxa Cubie A5E, MYIR MYD-LT527).

Reviewers who tried these boards either did not attempt AI at all, or exercised only vision workloads (YOLOv5, ResNet, MobileNet):

- **WalnutPi 2B (T527)** — Hackaday review, Jan 2026. Reviewer characterizes it as basic due to "missing (proprietary) Allwinner IP block drivers." No AI/NPU testing. https://hackaday.com/2026/01/16/trying-out-the-allwinner-based-walnut-pi-sbc/
- **Avaota-A1 (T527)** — Open-source SBC by YuzukiHD. Marketed for edge AI; only YOLOv5-class workloads documented. https://github.com/AvaotaSBC/Avaota-A1
- **Radxa Cubie A5E (A527/T527)** — Radxa's official NPU docs only cover object detection. https://docs.radxa.com/en/cubie/a5e ; https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie_yolov5

---

## 2. Sister-SoC Ports (A527 / A733 / T536 / H728)

### 2.1 Radxa Cubie A7A (Allwinner A733 — 3 TOPS VIP9000, same SDK family as T527)

**Anton Maltsev's hands-on review (Medium)** — the single most relevant data point in this entire report.

- URL: https://medium.com/@zlodeibaal/radxa-cubie-a7a-f7401a185694
- SoC: Allwinner A733 (2x A76 + 6x A55, VIP9000 ~3 TOPS)
- Framework: Allwinner ACUITY / VIPLite (same toolchain the user targets)
- Attempted: transformer / depth-estimation models on NPU
- **Result: "I wasn't able to export transformers"** on VIP9000. Simple transformers ran on a different (Debex) board — the failure is with the Allwinner-flavored tooling, not fundamentally the NPU IP.
- Ancillary friction: **no Python API** (C++ only for VIPLite runtime), Docker-only export flow, poor error messages.
- Verdict: "unsuitable for LLM inference" per this review.

**Direct read-across for T527:** the Acuity 6.12 → VIPLite v2.0 pipeline the user is on is the same stack that blocked this reviewer, on a slightly bigger NPU. Model *conversion* is the risky step, not runtime.

### 2.2 ONNX Runtime VIPLite Execution Provider Request

- URL: https://github.com/microsoft/onnxruntime/issues/28244
- Opened April 27 2026 by @evgen-pervenenko; labeled *stale*; no assignee, no PR, no branch.
- Explicit target workloads listed: face recognition, image classification, smart search (Immich). **No LLM mention.**
- Acuity Toolkit publicly on GitLab (no NDA) — confirms the toolchain is at least openly available for a future porter.

### 2.3 Allwinner T536 / H728

- T536 (4x A55 + RISC-V, up to 3 TOPS NPU) and H728 (8x A55) both use VIP9000-class NPUs but are newer (2025) and have no LLM demos found.
- Sources: https://www.cnx-software.com/2025/04/14/allwinner-t536-quad-core-arm-cortex-a55-risc-v-industrial-soc-supports-ecc-ram-up-to-3-top-ai-accelerator/ ; https://linuxgizmos.com/walnutpi-2b-is-a-raspberry-pi-style-sbc-with-allwinner-t527-and-2-tops-npu/

---

## 3. Vivante NPU LLM Ports on Other SoCs

### 3.1 VeriSilicon official claim

- VIP9000 family officially supports "4-bit quantization... facilitating the deployment of AIGC and LLM algorithms such as Stable Diffusion and Llama 2 on embedded devices."
- URL: https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000
- Caveat: this is IP-vendor marketing. The T527 ships the *NanoSI-Plus* (bottom-of-line) VIP9000 at ~2 TOPS INT8, not the full VIP9000. No public demo of Llama on any VIP9000 variant found.

### 3.2 NXP i.MX 8M Plus / 9x (Vivante GC7000UL / VIP8000, older Vivante families — not VIP9000)

- URL: https://www.nxp.com/docs/en/user-guide/UG10166.pdf
- NXP has *demonstrated* Llama 2 and BlenderBot on i.MX 95 (eIQ Neutron NPU — **NXP's own IP, not Vivante**). i.MX 8M Plus uses Vivante VIP8000 but NXP's LLM work uses their newer Neutron NPU instead.
- Practical takeaway: even NXP, with vendor-grade Vivante support, chose not to build the LLM path on Vivante silicon. Bearish signal.

### 3.3 Amlogic A311D2 / other Vivante consumers

- No LLM ports found on any Amlogic Vivante-NPU SoC.

---

## 4. Reference: LLM-on-NPU Ports (Non-Vivante, for Approach Transfer)

### 4.1 Rockchip RKLLM (RK3588 / RK3576 / RK3588S — 6 TOPS NPU)

The most mature open-source SBC-NPU LLM stack, and the best template for what a T527 port would look like.

- Toolkit: https://github.com/airockchip/rknn-llm
- Overview: https://www.cnx-software.com/2024/07/15/rockchip-rkllm-toolkit-npu-accelerated-large-language-models-rk3588-rk3588s-rk3576/

Radxa-reported RK3588 numbers (RKLLM runtime, NPU-accelerated):

| Model | tok/s |
|---|---|
| TinyLlama 1.1B | **15.03** |
| Qwen 1.8B | **14.18** |
| Phi-3 3.8B | 6.46 |
| ChatGLM3 6B | 3.67 |
| DeepSeek-R1-Distill-Qwen-1.5B | **14.93** (compiled model 1.9 GB) — https://www.cnx-software.com/2025/02/09/deepseek-rockchip-rk3588-npu-ai-acceleration-15-tokens-per-second/ |

Supported model families in RKLLM: TinyLlama, Qwen/Qwen2, Phi-2, Phi-3, Gemma 2B, InternLM2 1.8B, MiniCPM 2B, ChatGLM3.

**Note:** RKLLM's NPU-only advantage over CPU is largely on *prefill*; decode speed on RK3588 is memory-bandwidth-bound and comparable to well-tuned llama.cpp on CPU.

### 4.2 rk-llama.cpp — RKNPU as GGML backend

- URL: https://github.com/invisiofficial/rk-llama.cpp
- Approach: NPU as a GGML compute backend (matmul offload). No documented tok/s in README, but referenced repeatedly in RK3588 community.

### 4.3 Original RKNPU2/GGML experiment (clehaxze, 2023)

- URL: https://clehaxze.tw/gemlog/2023/10-22-experiemtal-rknpu2-backend-for-ggml-llamacpp.gmi
- LLaMA2-7B, RK3588 NPU, matmul-only offload
- **10% speedup** vs. CPU — modest
- Running *all* layers on NPU produced gibberish because RKNPU2 only supports INT8 for weights and activations; accumulated quant error broke the model.
- **Takeaway for T527:** even with Q8 on-NPU matmul working, whole-model NPU offload with INT8 weights *and* INT8 activations is unlikely to produce coherent output. Realistic plan is partial offload (matmul-only), which caps speedup at ~10–30% over CPU.

### 4.4 haozixu/llama.cpp-npu — Qualcomm Hexagon

- URL: https://github.com/haozixu/llama.cpp-npu
- Snapdragon 8 Gen 2+; supports <4B (32-bit address space limit on Hexagon cDSP).
- Research prototype only. Referenced paper: arXiv:2509.23324.
- Approach uses FastRPC CPU↔DSP communication — architecturally similar to what VIPLite offers via ovxlib.

### 4.5 Fast On-device LLM Inference with NPUs (ASPLOS'25)

- URL: https://arxiv.org/html/2407.05858v2 ; https://xumengwei.github.io/files/ASPLOS25-NPU.pdf
- Systems-paper reference for mobile NPU LLM (Qualcomm), useful for understanding CPU/NPU op-partitioning strategy.

---

## 5. CPU-only LLM on Cortex-A55 (Fallback Path)

### 5.1 Ceiling estimates

The T527 has 8x Cortex-A55 @ up to 1.8 GHz sharing LPDDR4/4X memory (typically 1–4 GB). LLM decode is memory-bandwidth-bound; extra cores past 4 give diminishing returns on Cortex-A55.

**Direct-analog data points** (no dedicated T527 llama.cpp benchmarks exist yet — this is by proxy):

| Board | CPU | Model | Quant | tok/s | Source |
|---|---|---|---|---|---|
| Pi 4 (4 GB) | 4x A72 @ 1.5 GHz | TinyLlama 1.1B | Q4_K_M | 8–12 | https://ohyaan.github.io/tips/local_llm_optimization_with_llama.cpp_-_on-device_ai/ |
| Pi 4 (8 GB) | 4x A72 @ 1.5 GHz | TinyLlama 1.1B | Q4_K_M | 5–7 | https://localaimaster.com/blog/llm-raspberry-pi-5 |
| Pi 4 (8 GB) | 4x A72 @ 1.5 GHz | Llama 3.2 1B | Q4_K_M | 4–5 | ditto |
| Pi 5 (8 GB) | 4x A76 @ 2.4 GHz | TinyLlama 1.1B | Q4_K_M | 14.4–18.4 | https://tinyweights.dev/posts/run-llms-raspberry-pi-5/ |
| Pi 5 (8 GB) | 4x A76 @ 2.4 GHz | Qwen2.5 1.5B | Q4_K_M | ~10–15 | https://localaimaster.com/blog/llm-raspberry-pi-5 |
| Orange Pi 5 Pro (RK3588S) | 4x A76 + 4x A55 | TinyLlama 1.1B | Q4_K_M | 27.5 (llamafile, 4 threads only) | arXiv 2511.07425 |
| Orange Pi 5 Pro | same, 8 threads | TinyLlama 1.1B | Q4_K_M | *slower* than 4-thread | ditto |

Recurring pattern **confirmed in multiple sources**: on RK3588 (and by extension anything with A55s + faster cores), 4 threads on the fast cores beat 8 threads that include A55s — the A55s bottleneck the pipeline.

### 5.2 Rough extrapolation for T527 (8x A55 only, no A76/A72 to lean on)

Per-core, A55 ≈ 0.4–0.5x A72 on integer-dot-product-heavy workloads. Best-case T527 numbers you should plan around:

| Model | Quant | Realistic tok/s (CPU-only) | Notes |
|---|---|---|---|
| SmolLM2 135M / 360M | Q4_K_M | 20–40 | Only regime that feels interactive |
| Qwen2.5 0.5B | Q4_K_M | 8–15 | Feasible for chat |
| TinyLlama 1.1B | Q4_K_M | 3–6 | **Marginal; likely OK for offline batch** |
| Qwen2.5 1.5B | Q4_K_M | 2–4 | Slow but usable |
| Gemma 2B | Q4_K_M | 1–3 | Slow, RAM tight on 2 GB board |
| Llama 3.2 3B | Q4_K_M | <1.5 | Not viable |

**These are point estimates from Pi-4 scaling — not measured on T527.** Real numbers will differ with LPDDR4 speed (T527 spec varies by board vendor).

### 5.3 Frameworks to try (in order of expected effort)

1. **llama.cpp** — best ARM NEON path (Q4_0, Q4_K_M, Q8_0 all NEON-vectorized). Compile with `-DGGML_ARM_ARCH=armv8.2-a+dotprod+fp16` — A55 has ARMv8.2 dot-product (SDOT/UDOT) which is critical for Q8 speed.
2. **llamafile** — Justine Tunney's fork, ARM64 tinyBLAS. arXiv 2511.07425 measured 3–4x throughput vs Ollama and 30–40% less power on Orange Pi 5 Pro.
3. **MNN-LLM** — Alibaba, historically strong ARM. Less documentation, worth a benchmark run.
4. **MLC-LLM** — TVM-based; higher setup cost, mainly a win on GPU-endowed devices.

---

## 6. Hybrid CPU+NPU LLM Frameworks

**No embedded-Linux, VIP9000-compatible hybrid framework exists off the shelf.** The candidates one might build on top of:

- **GGML backend interface** — cleanest ABI to plug a VIPLite matmul kernel into llama.cpp (see clehaxze RKNPU2 experiment for the template). Practical challenge: activations need to stay INT8/INT16 on-NPU, and requantization overhead can eat the speedup.
- **ONNX Runtime + VIPLite EP** — no such EP exists (issue 28244 still stale). If it landed, one could run whole-model ONNX-exported LLMs via ORT on NPU — but the transformer export failure noted by Anton Maltsev suggests Acuity currently can't handle the full attention/KV-cache pattern.
- **Custom pipeline** — export each transformer block as a separate NB, run per-token with CPU-side attention + KV cache and NPU-side FFN. This is what a serious port would look like.

### 6.1 Quantization options on VIP9000

Acuity supports **INT8, INT16, FP16, BF16** natively. **Native INT4 support depends on VIP9000 subvariant** — VeriSilicon markets INT4 broadly, but the NanoSI-Plus in T527 is the low-end tile array. Verify with `pegasus quantize --qtype` options in your local Acuity 6.12 install; if int4 isn't accepted, you're bounded to INT8 (which per §4.3 breaks whole-model LLM offload).

---

## 7. Practical Constraints for T527

| Constraint | Value | Impact |
|---|---|---|
| CPU | 8x Cortex-A55 @ ≤1.8 GHz | Memory-bandwidth-bound; use 4 threads pinned to a cluster |
| NPU | VIP9000-NanoSI-Plus, ~2 TOPS INT8 | Enough compute for 1–2B model matmuls; INT8 weight+activation may destroy accuracy on full offload |
| RAM | 1–4 GB LPDDR4 (board-dependent) | **Hard cap on model size**. A Q4_K_M 1.1B model = ~640 MB; add KV cache, runtime, OS → 2 GB minimum, 4 GB comfortable |
| Storage | eMMC / SD | Fine for model files |
| Toolchain | Acuity 6.12 + VIPLite v2.0 (`libNBGlinker.so`, `libVIPhal.so`) | Same stack that failed on transformer export in Cubie A7A review |
| Python API | Absent | Runtime is C/C++ only |

### 7.1 Recommended target set (in priority order)

1. **Qwen2.5-0.5B-Instruct Q4_K_M** — smallest capable modern chat model, ~350 MB. Best chance of interactive speed CPU-only.
2. **SmolLM2-360M Q4_K_M** — even smaller, ~220 MB; useful for a "does the toolchain even work" milestone.
3. **TinyLlama-1.1B-Chat Q4_K_M** — the industry reference; if it works ≥3 tok/s the port is publishable.
4. **DeepSeek-R1-Distill-Qwen-1.5B Q4** — only if you have 4 GB RAM; strong reasoning per parameter.

---

## 8. Recommended First Milestones

1. **Baseline CPU-only.** Get llama.cpp compiled for T527 with `+dotprod` NEON. Bench Qwen2.5-0.5B and TinyLlama Q4_K_M with 4 threads. This defines the floor.
2. **Verify NPU is even a win.** Reproduce the clehaxze approach: hand-write a VIPLite Q8 matmul kernel, wire it into GGML as a backend. Bench with `--n-gpu-layers` equivalent set to 1–4 blocks only. If you can't beat CPU by ≥20%, the NPU is not worth it.
3. **Transformer export smoke test.** Before investing in kernels, try `pegasus import onnx` on a *single* Qwen2.5-0.5B decoder block ONNX. If it fails like Anton Maltsev saw on A733, the port is blocked at the compiler and no runtime work matters until Allwinner ships a fixed Acuity.

---

## 9. Repository / Link Index

- Acuity SDK (public, no NDA): search "Allwinner ACUITY Toolkit" on gitlab.com
- Radxa Cubie A7A review (**must read** — closest failure mode): https://medium.com/@zlodeibaal/radxa-cubie-a7a-f7401a185694
- ONNX Runtime VIPLite EP request: https://github.com/microsoft/onnxruntime/issues/28244
- Avaota-A1 board (T527): https://github.com/AvaotaSBC/Avaota-A1
- Radxa Cubie A5E (T527/A527): https://docs.radxa.com/en/cubie/a5e
- Radxa Cubie A7A (A733) NPU docs: https://docs.radxa.com/en/cubie/a7a/app-dev/npu-dev/cubie-acuity-sdk
- RKLLM (reference approach): https://github.com/airockchip/rknn-llm
- CNX Software RKLLM writeup: https://www.cnx-software.com/2024/07/15/rockchip-rkllm-toolkit-npu-accelerated-large-language-models-rk3588-rk3588s-rk3576/
- rk-llama.cpp (GGML backend approach): https://github.com/invisiofficial/rk-llama.cpp
- clehaxze RKNPU2/GGML experiment (matmul-offload template): https://clehaxze.tw/gemlog/2023/10-22-experiemtal-rknpu2-backend-for-ggml-llamacpp.gmi
- haozixu llama.cpp-npu (Snapdragon Hexagon): https://github.com/haozixu/llama.cpp-npu
- Fast On-device LLM Inference with NPUs (ASPLOS'25): https://arxiv.org/html/2407.05858v2
- SBC LLM eval survey (Pi 4/5, Orange Pi 5 Pro): https://arxiv.org/html/2511.07425v1
- Vivante VIP9000 (vendor page, Llama 2 claim): https://www.verisilicon.com/en/IPPortfolio/VivanteVIP9000
- NXP i.MX ML User Guide (Neutron NPU LLM): https://www.nxp.com/docs/en/user-guide/UG10166.pdf
- Rockchip RK3588 llama.cpp perf issue: https://github.com/ggml-org/llama.cpp/issues/722
- WalnutPi Hackaday review: https://hackaday.com/2026/01/16/trying-out-the-allwinner-based-walnut-pi-sbc/

---

## 10. Bottom Line

Nobody has run an LLM on T527 or any Vivante-NPU SBC in public. The closest data point — a knowledgeable reviewer with the same Acuity toolchain — got stuck at *transformer export*, not at runtime. If you proceed:

- **Do the transformer-export smoke test first.** Everything else is contingent on it.
- **Plan for CPU-only.** Set expectations at 3–6 tok/s for TinyLlama Q4_K_M, ~10 tok/s for Qwen2.5-0.5B. If the NPU eventually gives 1.5–2x on top, that's a win.
- **Budget RAM before TOPS.** A 2 GB T527 board will not comfortably run anything above ~1.5B parameters at Q4.
- **Bias toward llama.cpp/llamafile CPU + optional VIPLite matmul GGML backend.** Whole-model NPU offload is a research project, not a port.
