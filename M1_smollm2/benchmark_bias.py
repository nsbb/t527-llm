#!/usr/bin/env python3
"""First-token accuracy benchmark: does bias correction hit ORT-FP32 argmax?

Uses SmolLM2 pipeline for reproducibility.
Prints per-prompt result + aggregate top-1 / top-5 accuracy.
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--dev-dir", default="/data/local/tmp/smollm2_llm")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=576)
    ap.add_argument("--pad", type=int, default=2)
    args = ap.parse_args()

    TOK = Tokenizer.from_file(args.tokenizer)
    meta = json.loads(Path(args.meta).read_text())
    q = list(meta["Outputs"].values())[0]["quantize"]
    scale, zp = q["scale"], q["zero_point"]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    embed = np.load(args.embed)
    bias = np.load(args.bias)

    # 12 SmolLM2-relevant prompts
    TESTS = [
        "The capital of France is",
        "def fibonacci(n):",
        "print('hello",
        "class MyClass:",
        "import numpy as",
        "The three primary colors are red, blue,",
        "Water boils at 100 degrees",
        "The sun rises in the",
        "One plus one equals",
        "The best programming language is",
        "In machine learning, a neural network",
        "def hello():",
    ]

    hits_top1_none = hits_top1 = hits_top5 = 0
    for prompt in TESTS:
        ids = TOK.encode(prompt).ids
        ids = ([args.pad] * (args.seq_len - len(ids)) + ids) if len(ids) < args.seq_len else ids[-args.seq_len:]
        tokens = np.asarray([ids], dtype=np.int32)
        h_ort = sess.run(["final_rms_out"], {"token_ids": tokens})[0][0, -1]

        tokens.tofile("/tmp/input_0.dat")
        subprocess.run([ADB, "push", "/tmp/input_0.dat", f"{args.dev_dir}/input_0.dat"], check=True, capture_output=True)
        subprocess.run([ADB, "shell", f"cd {args.dev_dir} && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"], check=True, capture_output=True)
        subprocess.run([ADB, "pull", f"{args.dev_dir}/output_0.dat", "/tmp/output_0.dat"], check=True, capture_output=True)
        raw = np.fromfile("/tmp/output_0.dat", dtype=np.uint8)
        h_dev = ((raw.astype(np.int32) - zp).astype(np.float32) * scale).reshape(args.seq_len, args.hidden)[-1]

        fp32_top1 = int((h_ort @ embed.T).argmax())
        dev_none = int((h_dev @ embed.T).argmax())
        dev_corr = h_dev - bias
        dev_top1 = int((dev_corr @ embed.T).argmax())
        dev_top5 = set((dev_corr @ embed.T).argsort()[-5:].tolist())

        hits_top1_none += (fp32_top1 == dev_none)
        hits_top1 += (fp32_top1 == dev_top1)
        hits_top5 += (fp32_top1 in dev_top5)
        print(f"  {prompt[-40:]!r}: FP32={TOK.decode([fp32_top1])!r}  raw={TOK.decode([dev_none])!r}  bias={TOK.decode([dev_top1])!r}")

    N = len(TESTS)
    print(f"\n== TOP-1 no bias: {hits_top1_none}/{N} ({hits_top1_none/N*100:.1f}%)")
    print(f"== TOP-1 bias:    {hits_top1}/{N} ({hits_top1/N*100:.1f}%)")
    print(f"== TOP-5 bias:    {hits_top5}/{N} ({hits_top5/N*100:.1f}%)")


if __name__ == "__main__":
    main()
