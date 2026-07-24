#!/usr/bin/env python3
"""Measure device-vs-ORT hidden state bias over calibration prompts.

Outputs three bias variants:
  device_bias_all.npy       — mean over all seq positions + prompts, shape [576]
  device_bias_content.npy   — mean over positions 24-31 only,        shape [576]
  device_bias_pos.npy       — per-position mean,                     shape [32, 576]

Usage:
    python3 compute_bias.py --meta nbg_meta_hidden_c100.json \
        --onnx work/generated/.../real_llm_sq_hidden.onnx \
        --calib-dir acuity_out/smollm2_hidden_c100 \
        --n 100 --out-dir .
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/smollm2_llm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=576)
    ap.add_argument("--dev-dir", default=DEV_DIR)
    ap.add_argument("--out-prefix", default="device_bias")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    q = list(meta["Outputs"].values())[0]["quantize"]
    scale, zp = q["scale"], q["zero_point"]
    print(f"NB output scale={scale}, zp={zp}")

    calibs = sorted(args.calib_dir.glob("calib_*.npy"))[:args.n]
    print(f"using {len(calibs)} calib files")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    ort_all = np.zeros((len(calibs), args.seq_len, args.hidden), dtype=np.float32)
    dev_all = np.zeros_like(ort_all)

    for i, cf in enumerate(calibs):
        tokens = np.load(cf).astype(np.int32).reshape(1, args.seq_len)
        h_ort = sess.run(["final_rms_out"], {"token_ids": tokens})[0][0]
        ort_all[i] = h_ort

        tokens.tofile("/tmp/input_0.dat")
        subprocess.run([ADB, "push", "/tmp/input_0.dat", f"{args.dev_dir}/input_0.dat"],
                       check=True, capture_output=True)
        subprocess.run([ADB, "shell",
            f"cd {args.dev_dir} && LD_LIBRARY_PATH=/vendor/lib64 "
            f"/data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"],
            check=True, capture_output=True)
        subprocess.run([ADB, "pull", f"{args.dev_dir}/output_0.dat", "/tmp/output_0.dat"],
                       check=True, capture_output=True)
        raw = np.fromfile("/tmp/output_0.dat", dtype=np.uint8)
        h_dev = ((raw.astype(np.int32) - zp).astype(np.float32) * scale).reshape(
            args.seq_len, args.hidden)
        dev_all[i] = h_dev
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(calibs)}")

    diff = dev_all - ort_all  # [N, seq, hidden]

    bias_all = diff.mean(axis=(0, 1))
    bias_content = diff[:, 24:32, :].mean(axis=(0, 1))
    bias_pos = diff.mean(axis=0)  # [seq, hidden]

    print("\nBias stats:")
    print(f"  all-position:     |max|={np.abs(bias_all).max():.3f}, |mean|={np.abs(bias_all).mean():.3f}")
    print(f"  content-position: |max|={np.abs(bias_content).max():.3f}, |mean|={np.abs(bias_content).mean():.3f}")
    print(f"  per-position:     |max|={np.abs(bias_pos).max():.3f}, |mean|={np.abs(bias_pos).mean():.3f}")

    np.save(f"{args.out_prefix}_all.npy", bias_all)
    np.save(f"{args.out_prefix}_content.npy", bias_content)
    np.save(f"{args.out_prefix}_pos.npy", bias_pos)
    print(f"saved {args.out_prefix}_{{all,content,pos}}.npy")


if __name__ == "__main__":
    main()
