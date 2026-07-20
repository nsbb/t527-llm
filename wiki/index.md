# Wiki Index

Master navigation for T527 NPU LLM porting project.

**Last updated**: 2026-07-20

---

## Overview

- [Project schema](schema.md) — how this wiki is organized
- [Chronological log](log.md) — every experiment/finding in order
- Root [CHANGELOG.md](../CHANGELOG.md) — version-tagged milestones
- Root [PLAN.md](../PLAN.md) — original 4-stage plan (M0→M1→M2→M3)

## Models

- [SmolLM2-135M-Instruct on T527](models/smollm2-135m.md) — 30-layer Llama, 10.9 tok/s uint8, FP32 baseline coherent
- [Qwen2.5-0.5B-Instruct on T527](models/qwen25-05b.md) — 24-layer Qwen2 multilingual (한/영/중), 5 tok/s uint8

## Hardware

- [T527 Vivante VIP9000-NanoSI-Plus](hardware/t527-vip9000.md) — PID 0X10000016, ~2 TOPS, driver v1.13

## Pipeline stages

- [Static ONNX generation](pipeline/onnx-generation.md) — petayyyy `make_real_llm_onnx.py`, sin/cos preomputation
- [Acuity import](pipeline/acuity-import.md) — pegasus import, opset 11, 3D→4D promotion caveat
- [Quantization](pipeline/quantization.md) — supported quantizer/qtype matrix, calibration recipe
- [NBG export](pipeline/nbg-export.md) — VivanteIDE env, `--optimize` PID flag
- [Device execution](pipeline/device-execution.md) — vpm_run_aarch64, sample.txt format

## Techniques

- [Axis-fix ONNX patch](techniques/axis-fix-patch.md) — 61 ReduceMean nodes `axes=[2]` → `axes=[-1]`
- [Last-slice ONNX patch](techniques/last-slice-patch.md) — remove `slice_last_hidden` for Acuity converter bug
- [SmoothQuant ONNX rewrite](techniques/smoothquant.md) — 167 Linear layers activation-scale shift
- [Sliding-window multi-token decode](techniques/sliding-window-decode.md) — W=32 no-KV recompute strategy

## Issues (bugs + workarounds)

- [Acuity 6.12 ReduceMean axis off-by-one](issues/acuity-reducemean-axis.md) — **★ Biggest find**
- [slice_last_hidden `size=-30` converter bug](issues/slice-last-hidden.md)
- [LLM activation outlier catastrophic quantization loss](issues/llm-outlier-saturation.md) — the M2 blocker
- [Host CPU emulation vs device NPU int16 numerical drift](issues/host-vs-device-drift.md) — unsolved

## Decisions

- [uint8 vs int16 quantization recipe](decisions/uint8-vs-int16.md)
- [SmoothQuant alpha choice](decisions/smoothquant-alpha.md)
- [Model selection for M2/M3](decisions/model-selection.md)

## Results

- [M1 SmolLM2 FP32 NB coherent tokens](results/smollm2-fp32-nb.md) — first proof T527 runs LLM correctly
- [M1 SmolLM2 calib10 uint8](results/smollm2-calib10.md) — cos 0.11 → 0.93
- [M2 Qwen uint8 baseline](results/qwen-uint8-baseline.md) — 0/32 collapse
- [M2 Qwen SmoothQuant α=0.5 int16 host](results/qwen-sq-int16-host.md) — **25/32 match**
- [M2 Qwen SmoothQuant device drift](results/qwen-sq-device.md) — unsolved

## Raw sources

- [`raw/catalog.tsv`](../raw/catalog.tsv) — ingested source index
- [`raw/source-notes/`](../raw/source-notes/) — per-source notes

## External refs

- Repo: [github.com/nsbb/t527-llm](https://github.com/nsbb/t527-llm)
- Reference: [petayyyy/a733_npu_driver](https://github.com/petayyyy/a733_npu_driver)
- Karpathy LLM Wiki pattern: [gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
