# Allwinner T527 / VIP9000 LLM — GitHub Asset Survey

**Task:** Find every publicly-available asset (GitHub + official mirrors) relevant to porting an LLM onto the Allwinner T527 SoC's Vivante VIP9000-NanoSI-Plus NPU via the Acuity Toolkit / VIPLite / VivanteIDE pipeline the user already uses for ASR.

**Bottom line up-front:**
- No Allwinner-official (`allwinnertech`, `allwinner-zh`, `Allwinner-Homlet`, `tinalinux`) repo ships an LLM sample or a T527/A733 LLM `.nb`. Their orgs are legacy H3/H6/A20 kernels + Tina3.x — nothing NPU/LLM.
- The current authoritative Allwinner Tina5.0 AIoT source tree is on **GitLab (`Tina5.0_AIOT/*`)**, published July 2025. It contains `ai-sdk` as a project but the repo is **empty** (no branches). NPU docs live under `product/docs`, no LLM sample.
- The living downstream ai-sdk is **`ZIFENG278/ai-sdk`** (Radxa employee) — same tree as the user's local one, YOLO/ResNet/LeNet only, no LLM.
- The only concrete community LLM-on-VIP9000 work is on **A733 (Radxa Cubie A7A/A7S, OrangePi Zero 3W)**, not T527. Two independently-run projects: **`petayyyy/a733_npu_driver`** (SmolLM2-135M/360M on NPU, Qwen2.5-0.5B on CPU, MobileCLIP-S0 on NPU, honest failure notes for larger LLMs) and **`waz664/vip9000-embeddinggemma`** (EmbeddingGemma-300M encoder on NPU + Qwen3-0.6B on PowerVR GPU via patched llama.cpp).
- VeriSilicon's `acuity-models` model zoo **does list** `qwen2.5_7b_decode/prefill`, `llama2_7b_decode/prefill`, `bert_base`, `vit`, `whisper` — but only as `.json` topology dumps, not runnable NBGs, and never claimed to fit VIP9000-NanoSI-Plus (3 TOPS, no DDR headroom for 7B).
- **PID `0X10000016`** (the user's exact NanoSI-Plus PID) appears in **zero** public repos. A733 boards report `VIP9000NANODI_PLUS_PID0X1000003B` — same family, different SKU.

---

## 1. Allwinner-Official GitHub Orgs (all legacy — no NPU/LLM)

| Org | URL | Content | Verdict |
|---|---|---|---|
| `allwinnertech` | https://github.com/AllwinnerTech | ~37 repos, all `platform_external_*` Android forks + `linux-2.6.36` + `device_*`. Last active ~2013. | **Dead-end for NPU/LLM.** No ai-sdk, no NPU. |
| `allwinner-zh` | https://github.com/allwinner-zh | 6 repos: `media-codec`, `linux-3.4-sunxi`, `bootloader`, `documents`, `media-codec-lib`, `sunxi-tools`. | **Dead-end.** Nothing beyond legacy media codecs. |
| `Allwinner-Homlet` | https://github.com/Allwinner-Homlet | 11 repos, all H3/H6 BSP (kernel 4.4/4.9), CedarC video codec. | **Dead-end for NPU/LLM.** |
| `allwinner-dev-team` | https://github.com/allwinner-dev-team | 14 repos, `linux-allwinner` marked WILL BE REBASED. Community-managed. | **Dead-end for NPU/LLM.** |
| `allwinner-android` | https://github.com/allwinner-android | 103 repos of Android platform forks. | **Dead-end for NPU/LLM** (no ai-sdk visible). |
| `tinalinux` | https://github.com/tinalinux | 11 repos (`linux-3.10`, `package`, `dl`, `brandy`, `target`, `manifest`, `tools`, `toolchain`, `prebuilt`, `docs`). Tina3.x-era. | **Dead-end.** `tinalinux/package` top-level directories: `allwinner, base-files, devel, dragontools, firmware, kernel, lang, libs, luci, minigui, multimedia, network, qt, routing, system, utils` — **no `npu`, `awnn`, `eyesee-npu`, `ai-sdk`.** |
| `linux-sunxi` | https://github.com/linux-sunxi | Kernel/bootloader upstream community. No NPU driver work in tree yet for VIP9000. | Not relevant for LLM. |

### 1a. Allwinner's actual current source drop → GitLab, not GitHub

**Group:** https://gitlab.com/tina5.0_aiot — 94 subprojects, publicly readable (no NDA), published 2025-06-25. Contains the A527/T527/A733/T736/A523 SDK the user's environment came from.

Notable projects:
- `tina5.0_aiot/ai-sdk` — https://gitlab.com/tina5.0_aiot/ai-sdk — **empty repo** (default branch `main` but no tree; probably a placeholder that never got its content pushed).
- `tina5.0_aiot/product/docs` — https://gitlab.com/tina5.0_aiot/product/docs (branch `product-aiot-stable`) contains `Software 软件类文档/SDK模块开发指南/NPU模块开发指南/` with the same 5 Chinese NPU PDFs the user already has in `docs/acuity_toolkit/NPU模块开发指南/` (NPU_RuntimeAPI, NPU_开发环境部署, NPU_快速入门, NPU_模型部署, NPU_算子支持列表, T527_NPU_常见网络性能_测试报告). **No LLM chapter.**
- `tina5.0_aiot/product/tina/tina-ng/target/{t527,a733,a527,t736,a523}` — per-SoC target configs. No NPU/LLM.
- `tina5.0_aiot/product/linux/external/lib*` — user-space libs (libcedarc, libgpu, libawion, libAWIspApi). **No `libawnn`, no `libVIPlite` source** — those ship as binaries only.

**Verdict:** the "official Allwinner NPU code" is exactly the shape the user already has — a binary VIPLite/NBGlinker driver + Acuity PDFs + a few CV examples. There is **no upstream Allwinner LLM sample.**

---

## 2. ai-sdk / awnn ecosystem — downstream forks

The user's local `ai-sdk` (`examples/libawnn_viplite`, `machinfo/T527/config.mk` with `NPU_SW_VERSION = v1.13`, resnet50/yolov5/deepspeech2/multi_thread) matches these two repos byte-for-byte in the parts they share:

### `ZIFENG278/ai-sdk` — the de-facto authoritative fork

- URL: https://github.com/ZIFENG278/ai-sdk
- Description: "radxa cubie series NPU ai-sdk"
- Stars: 20 | Last push: 2025-10-20 | Language: C 96.3 %
- Directories: `examples/{lenet, libawnn_viplite, libawutils, multi_thread, resnet50, vpm_run, yolact, yolov5}`, `models/{MobileNetV2_Imagenet, inception_v1, lenet, lstm_mnist, mobilenet_v1_1.0_224_quant, resnet50-sim, squeezenet1_0, yolov3_tiny, yolov5s-sim}`, `machinfo/`, `unified-tina/`, `viplite-tina/`, `scripts/`, `tools/`.
- **LLM content: none.** No transformer, attention, KV-cache, tokenizer, or GPT/LLaMA/Qwen sample. Pure CV + one LSTM-MNIST toy.
- Cited by Radxa Docs as the official ai-sdk (`docs.radxa.com/en/cubie/a7s/app-dev/npu-dev/`).
- **Maintained.** Recent commits, tags per NPU driver version.

### `radxa-edge/ai-sdk` — advertised in Radxa docs but 404

- URL was `https://github.com/radxa-edge/ai-sdk.git` — **returns HTTP 404** as of query. The `radxa-edge` org exists (`radxa-edge/TPU-Edge-AI` is for Sophgo BM1684X, not Vivante).
- Verdict: **stale doc reference. Use `ZIFENG278/ai-sdk` instead.**

### Nothing newer than v2.0 driver

No public ai-sdk ships attention/KV-cache/token-generation demos, transformer helper ops, or a decode/prefill sample. There is **no upstream ai-sdk with LLM examples.**

---

## 3. Model zoo / pretrained NB for VIP9000-NanoSI-Plus

### VeriSilicon official

| Repo | URL | Content | Notes |
|---|---|---|---|
| `VeriSilicon/acuity-models` | https://github.com/VeriSilicon/acuity-models | 66 commits. Includes `models/{llama2_7b_decode, llama2_7b_prefill, qwen2.5_7b_decode, qwen2.5_7b_prefill, bert_base, vit, swin_transformer, whisper, ...}` plus 80+ CV/audio models. | **LLM entries are Acuity `.json` topology only, no `.data` weights, no NBG.** Almost certainly targets VIP9500/9700-class NPUs with wider MAC arrays than NanoSI-Plus. Great reference for op-mapping but not a drop-in NB. |
| `VeriSilicon/acuitylite` | https://github.com/VeriSilicon/acuitylite | 22 stars, 23 commits. End-to-end deploy tool: Caffe/ONNX/TF/TFLite → TIM-VX / TFLite w/ optional NBG export via `OvxlibExporter`. MIT license. | LLM not explicitly mentioned. Useful as an alternative to Pegasus/Acuity6.12. |
| `VeriSilicon/TIM-VX` | https://github.com/VeriSilicon/TIM-VX | Tensor Interface Module, "150+ operators", backend for TFLite external delegate. Latest release v1.2.22 (2025-01-08). | Ref boards: i.MX 8M Plus, A311D, S905D3 — **A733/T527 not listed.** VIP9000 not explicitly named. |
| `VeriSilicon/tflite-vx-delegate` | https://github.com/VeriSilicon/tflite-vx-delegate | TFLite external delegate over TIM-VX. | Potentially usable on T527 if TIM-VX links against the Allwinner driver — but no public evidence anyone has done it. |
| `VeriSilicon/LiteRT-LM` | https://github.com/VeriSilicon/LiteRT-LM | Fork of Google's on-device LLM runtime. v0.8.0 Nov 2025. Supports Gemma3-1B, Gemma-3n, Phi-4-mini, Qwen2.5-1.5B, FunctionGemma-270M in `.litertlm` format. | NPU backends: **Qualcomm + MediaTek only**. **No Vivante backend, no Allwinner.** 0 stars. |
| `VeriSilicon/vsi-pjrt-plugin` | https://github.com/VeriSilicon/vsi-pjrt-plugin | PJRT plugin for VeriSilicon NPU. MIT. | JAX/XLA path. Not proven on VIP9000-Nano. |
| `VeriSilicon/FlagGems` | https://github.com/VeriSilicon/FlagGems | Triton operator library for LLMs. | Server-class NPUs. Not relevant for edge VIP9000-Nano. |
| `VeriSilicon/triton-vsi-backend` | https://github.com/VeriSilicon/triton-vsi-backend | Triton compile-and-execute backend. | Also server-class. |

### T527-specific pretrained NB

- **None found.** No `network_binary.nb` for TinyLlama, Qwen, Gemma, Phi, or LLaMA targeting VIP9000-NanoSI-Plus / `PID0X10000016` exists in any public repo.
- All working LLM ports are on **A733** (`VIP9000NANODI_PLUS_PID0X1000003B`) — same NPU generation, different SKU. Their NBs are compiled per-board and **not directly loadable** on T527 (Acuity/Pegasus PID + optimize flag must match).

---

## 4. Tina Linux / OpenWRT-Tina AI packages

- `tinalinux/package` (github.com/tinalinux/package) — surveyed, **no `npu` / `awnn` / `eyesee-npu` / `ai` package.** The old `package/allwinner/eyesee-npu/*` path referenced in Allwinner docs must live in the vendor-only Tina3.5-Homlet drop; it is not on this public mirror.
- `chainsx/openwrt-sunxi-aiot` — https://github.com/chainsx/openwrt-sunxi-aiot | 8 stars | 2025-12-15 | OpenWrt 24.10 fork targeting A733 (Cubie A7A/A7Z). **No AI packages, no NPU package.** Pure OS bring-up.
- `lindenis-org/lindenis-v536-softwinner` — Allwinner V536 (older SoC, V-series) with `eyesee-mpp` multimedia SDK. Not relevant (no VIP9000).
- Tina5.0-AIOT GitLab has no `eyesee-npu` package either.

**Verdict:** no public OpenWrt-Tina LLM package exists.

---

## 5. Board-vendor forks / SBC ecosystems

### 5a. Community LLM-on-VIP9000 work (all A733, none T527)

| Repo | URL | Stars | Last push | What it contains | Value to your T527 port |
|---|---|---|---|---|---|
| **`petayyyy/a733_npu_driver`** | https://github.com/petayyyy/a733_npu_driver | 0 | 2026-06-25 | **Highest-signal LLM-on-Vivante work anywhere.** ONNX → ACUITY 6.30.22 (Docker `ubuntu-npu:v2.0.10.1`) → NBG → VIPLite 2.0.3.2 → board. Working NPU: SmolLM2-135M (21 tok/s), SmolLM2-360M (8.4 tok/s), MobileCLIP-S0 (22.6 ms/frame). CPU-only fallback: Qwen2.5-0.5B Q8_0 (18 tok/s on 2×A76), SmolVLM-256M/500M. **Explicit failure list:** Qwen2.5-0.5B and SmolLM2-1.7B could not be made to run on NPU (activation outliers + toolchain limits). **No KV-cache — static-shape NBG, fixed window W≤64 only.** Target board: Radxa Cubie A7Z / OrangePi Zero 3W. `app/` = chat CLI, `docs/`, `scripts/` = host+board tools, `reports/`. | **Extremely relevant.** Same NPU family as T527 (VIP9000-Nano), same ACUITY/VIPLite pipeline as user, honest failure notes, benchmarks. Read this first before spending effort. |
| **`waz664/vip9000-embeddinggemma`** | https://github.com/waz664/vip9000-embeddinggemma | 0 | 2026-05-11, v0.2.4 | EmbeddingGemma-300M FP32 seq128 transformer → NBG on Radxa Cubie A7S (VIP9000NANODI_PLUS_PID0X1000003B). Achieves 0.944 cos-sim vs CPU TFLite reference. Handles ONNX-op gaps by replacing `Expand`→multiply-by-ones, `Gelu`→tanh. `LLAMA_VK_NO_OUTPUT_OFFLOAD=1 --no-kv-offload` for Qwen3-0.6B on PowerVR GPU via patched llama.cpp (not on NPU). MIT. Python 55 % / C++ 15 % / Shell 13 %. | **Very relevant** for encoder-style transformer on VIP9000-Nano. Shows practical ONNX-op workarounds and dual-input attention-bias trick. Different PID (`3B` vs your `16`) — expect optimize-flag differences. |
| `unnamedwild-ux/frigate_npu_vivante` | https://github.com/unnamedwild-ux/frigate_npu_vivante | ? | 2025 late | Frigate custom detector plugin using YOLOv8 NBG on Cubie A7A / A733 / VIP9000NANODI_PLUS. VIPLite 2.0.3.2 + ACUITY 6.21.16 (Docker `ubuntu-npu:v1.8.11`). Recommends int16 over uint8. | **CV-only.** No LLM. Useful only as a second reference for int16 NB workflow. |

### 5b. Board-vendor SBC repos (bring-up, not LLM)

**AvaotaSBC** — https://github.com/AvaotaSBC (T527 open-source SBC)
- `Avaota-A1` (123 stars, 2024-08) — schematics + docs, no software.
- `AvaotaOS` (28 stars) — Ubuntu-based OS, no NPU/LLM samples.
- `linux`, `u-boot`, `toolchains`, `openwrt` — vanilla forks. **No NPU/LLM code.**

**YuzukiHD** — same designer as Avaota. `YuzukiHD/SyterKit` = bare-metal bootloader framework for T527/A527. **No LLM.**

**WalnutPi (T527)** — https://github.com/walnutpi
- `walnutpi/npu-model-transform` (1 star, 2025-06-20) — https://github.com/walnutpi/npu-model-transform — 100 % Shell (22 commits). "Tool for converting models to T527 NPU format." Includes `npu-transfer-yolo` shortcut with uint8 quantization + built-in FP preprocessing. **YOLO-only wrapper around Acuity/Pegasus. No LLM support.**
- Other walnutpi repos focus on the **K230** RISC-V AI SoC (different KPU NPU) — not applicable.

**MangoPi / BananaPi** — BPI-F5 spec is T527 but no public NPU/LLM repos yet.

**DongshanPI** — https://github.com/DongshanPI/T527-AvaotaA1_Tina5SDK (0 stars, 2026-07-02, C++ 81 %) is a Tina5 SDK snapshot for Avaota-A1. **No ai-sdk / LLM visible** in the README summary.

**Radxa** (main downstream) — `radxa-build/radxa-cubie-a7z`, `radxa-cubie-a7a`, `radxa-cubie-a5e` are OS build metadata. `ZIFENG278/ai-sdk` is the Radxa employee-maintained ai-sdk. `radxa-edge/TPU-Edge-AI` is Sophgo BM1684X (unrelated NPU).

**Other T527 repos (all non-AI)** — surveyed via `q=T527` (30 hits): mostly kernel/u-boot forks (`REG-Linux/linux-t527`, `matheusants/cedrus-t527`, `matheusants/sunxi-venc-t527`, `ut-slayer/orangepi-4a-mainline`, `ocean9914/u-boot-t527-vendor`, `tinysoul/u-boot-2018-t527`, `arm-sbc/T527-Multimedia-boards`, `embired/BSP`, `embired/linux_for_t527-avaota-a1-`, `Fengmi666/T527`, `zjr00/T527`, `nickfox-taterli/a527-gstreamer1`). None ship LLM code. `sc-bin/t527-yolo11`, `sc-bin/t527-facedetection`, `sc-bin/t527-facenet` are single-example CV repos.

---

## 6. Tangential / marketing-only (flag for skip)

- `t5274746/*`, `T527962308/*`, `YingShi001/t527`, `zjr00/T527`, `keithroett/yrj_t527`, `Nopiskl/T527_MicroPC`, `1120yhm/CarCentralControl_RK3566` — either personal placeholders, unrelated (RK3568), or user-profile pages with the string "T527" incidentally. **Skip.**
- `HBConline/orangepi4pro-skill` — a 2 900-line Claude Code skill / SKILL.md, not runnable inference code. **Documentation-only.**
- `intel/ipex-llm`, `airockchip/rknn-llm`, `NotPunchnox/rkllama` — LLM stacks for Intel / Rockchip NPUs. **Wrong silicon**, occasionally useful as architectural precedent for RKLLM-style KV-cache handling on constrained NPUs.
- `VeriSilicon/ZenCompiler`, `VeriSilicon/tvm`, `VeriSilicon/pytorch`, `VeriSilicon/tensorflow` — server/desktop-scale forks, archived or unmaintained. Not useful for VIP9000-Nano LLM.
- `sihyeong/Awesome-LLM-Inference-Engine`, `xlite-dev/Awesome-LLM-Inference` — lists, no code.
- `microsoft/onnxruntime` issue #28244 — request to add VIPLite EP for A733/T527 VIP9000 — **stale, no PR, closed without implementation**. Not actionable.
- `DeciHD/allwinner_docs` — mirror of public datasheets/TRMs. No NPU code.
- `100askTeam/Tina5-LinuxSDK` — Tina5 SDK metadata (9 stars, 3 commits). No NPU/LLM package inside.

---

## Concrete next-step reading order for the user

1. **`petayyyy/a733_npu_driver` — `docs/` and `reports/`** — closest-analog effort, gives you a realistic ceiling for what fits on VIP9000-Nano-class NPU (~SmolLM2-360M with fixed window ≤64, no KV cache). Read the failure list.
2. **`waz664/vip9000-embeddinggemma`** — read the ONNX-op replacement tricks (`Expand`, `Gelu`) and the FP32 seq128 dual-input pattern; those are the practical gotchas you'll hit converting attention blocks with Acuity 6.12.
3. **`VeriSilicon/acuity-models/models/qwen2.5_7b_decode/qwen2.5_7b_decode.json`** and `llama2_7b_prefill.json` — Acuity JSON topologies. Read them as ground truth for the decode/prefill split VeriSilicon officially recommends.
4. **`VeriSilicon/acuitylite`** — evaluate as an alternative to Pegasus for the ONNX-to-NBG hop if Acuity 6.12 lacks needed ops.
5. **`ZIFENG278/ai-sdk/examples/multi_thread`** and **`libawnn_viplite`** — you already have this; keep it as your host-side runtime template. Any LLM inference wrapper needs to plug in here.
6. **Tina5.0_AIOT GitLab NPU docs PDFs** — you already have the same ones under `docs/acuity_toolkit/NPU模块开发指南/`; the GitLab copy is not newer.

## Gaps / things that do not exist publicly

- No official Allwinner LLM `.nb`.
- No public `libawnn_viplite` LLM decode/prefill helper. If you want token-by-token generation you'll write the KV-cache management yourself around static-shape NBG runs.
- No T527-specific optimize flag for LLM workloads; you're stuck with `VIP9000NANOSI_PLUS_PID0X10000016`, which nobody else has published NB assets for.
- No ONNX Runtime VIPLite Execution Provider (feature request open, no work).
- No llama.cpp Vivante backend (Vulkan works on the PowerVR GPU on A733; the T527 has an ARM Mali-G57 instead, so the A7S llama.cpp Vulkan trick from `waz664` does not transfer to T527).
