#!/usr/bin/env python3
"""Qwen multi-token gen with per-channel bias correction + host K multiplier."""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
WORK = HERE.parent
ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/qwen_llm"
SEQ_LEN = 32
HIDDEN = 896
PAD = 151643
TOK = Tokenizer.from_file(str(WORK / "work/models/qwen2.5-0.5b-instruct/tokenizer.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="한국의 수도는")
    ap.add_argument("--tokens", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--embed", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--k", type=float, default=8.0)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    q = list(meta["Outputs"].values())[0]["quantize"]
    scale, zp = q["scale"], q["zero_point"]
    embed = np.load(args.embed)
    bias = np.load(args.bias)
    K = args.k

    ids = TOK.encode(args.prompt).ids
    window = ([PAD] * (SEQ_LEN - len(ids)) + ids) if len(ids) < SEQ_LEN else ids[-SEQ_LEN:]
    print(f"prompt: {args.prompt!r}")
    print(args.prompt, end="", flush=True)
    total = 0.0
    for _ in range(args.tokens):
        (HERE / "input_0.dat").write_bytes(np.asarray([window], dtype=np.int32).tobytes())
        subprocess.run([ADB, "push", str(HERE / "input_0.dat"), f"{DEV_DIR}/input_0.dat"], check=True, capture_output=True)
        t = time.perf_counter()
        subprocess.run([ADB, "shell", f"cd {DEV_DIR} && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"], check=True, capture_output=True)
        total += time.perf_counter() - t
        subprocess.run([ADB, "pull", f"{DEV_DIR}/output_0.dat", str(HERE / "output_0.dat")], check=True, capture_output=True)
        raw = np.fromfile(HERE / "output_0.dat", dtype=np.uint8)
        h_dev = ((raw.astype(np.int32) - zp).astype(np.float32) * scale).reshape(1, 1, SEQ_LEN, HIDDEN)[0, 0, -1]
        h_corr = h_dev - bias
        logits = K * h_corr @ embed.T
        if args.top_k > 0:
            idx = logits.argsort()[-args.top_k:]
            probs = np.exp((logits[idx] - logits[idx].max()) / args.temperature)
            probs /= probs.sum()
            next_tok = int(idx[np.random.choice(len(idx), p=probs)])
        else:
            next_tok = int(logits.argmax())
        print(TOK.decode([next_tok]), end="", flush=True)
        window = window[1:] + [next_tok]
    print(f"\n\n{args.tokens} tokens in {total:.2f}s = {args.tokens/total:.2f} tok/s")


if __name__ == "__main__":
    main()
