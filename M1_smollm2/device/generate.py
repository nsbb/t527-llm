#!/usr/bin/env python3
"""Multi-token sliding-window generation on T527 NPU (uint8 NB).

Static W=32 window, no KV cache — every new token requires full recompute.
Strategy: encode prompt, left-pad to 32, run NPU, take argmax of last position,
drop oldest token, append new token, repeat.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
WORK = HERE.parent
ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/smollm2_llm"
SEQ_LEN = 32
VOCAB = 49152
PAD = 2

TOK = Tokenizer.from_file(str(WORK / "work/models/smollm2-135m-instruct/tokenizer.json"))


def push_input(token_ids: np.ndarray) -> None:
    (HERE / "input_0.dat").write_bytes(token_ids.astype(np.int32).tobytes())
    subprocess.run([ADB, "push", str(HERE / "input_0.dat"), f"{DEV_DIR}/input_0.dat"],
                   check=True, capture_output=True)


def run_npu() -> float:
    t = time.perf_counter()
    r = subprocess.run([ADB, "shell",
        f"cd {DEV_DIR} && LD_LIBRARY_PATH=/vendor/lib64 "
        f"/data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 1"],
        check=True, capture_output=True, text=True)
    return time.perf_counter() - t


def pull_and_dequant(scale: float, zp: int) -> np.ndarray:
    subprocess.run([ADB, "pull", f"{DEV_DIR}/output_0.dat", str(HERE / "output_0.dat")],
                   check=True, capture_output=True)
    raw = np.fromfile(HERE / "output_0.dat", dtype=np.uint8)
    return ((raw.astype(np.int32) - zp).astype(np.float32) * scale).reshape(1, 1, SEQ_LEN, VOCAB)


def sample(logits: np.ndarray, top_k: int = 0, temperature: float = 1.0) -> int:
    """Greedy if top_k==0, else top-k sampling."""
    if top_k == 0:
        return int(logits.argmax())
    top_idx = logits.argsort()[-top_k:]
    probs = np.exp((logits[top_idx] - logits[top_idx].max()) / temperature)
    probs /= probs.sum()
    choice = np.random.choice(len(top_idx), p=probs)
    return int(top_idx[choice])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=0, help="0 = greedy")
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--zp", type=int, required=True)
    args = ap.parse_args()

    ids = TOK.encode(args.prompt).ids
    if len(ids) < SEQ_LEN:
        window = [PAD] * (SEQ_LEN - len(ids)) + ids
    else:
        window = ids[-SEQ_LEN:]

    print(f"prompt: {args.prompt!r}")
    print(f"generating {args.tokens} tokens...\n")
    print(args.prompt, end="", flush=True)

    generated = []
    total_ms = 0.0
    for step in range(args.tokens):
        tokens = np.asarray([window], dtype=np.int32)
        push_input(tokens)
        run_ms = run_npu() * 1000
        total_ms += run_ms
        logits = pull_and_dequant(args.scale, args.zp)
        next_tok = sample(logits[0, 0, -1], top_k=args.top_k)
        generated.append(next_tok)

        text = TOK.decode([next_tok])
        print(text, end="", flush=True)
        window = window[1:] + [next_tok]

    print(f"\n\n{args.tokens} tokens in {total_ms/1000:.2f}s "
          f"= {args.tokens / (total_ms/1000):.2f} tok/s (incl. adb overhead)")
    print(f"generated ids: {generated}")


if __name__ == "__main__":
    main()
