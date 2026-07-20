#!/usr/bin/env python3
"""Multi-token generation: NPU produces hidden state, host CPU runs lm_head.

The key trick: LM head output logits saturate on NPU int16/uint8 (±60+ range
compresses to 256 or ±32k bins). Hidden state values are ±3-30 (RMSNorm output)
which fits uint8/int16 quantization cleanly.

By stopping the NPU graph at final_rms_out and running the LM head MatMul
on CPU with FP32 embedding weight, we sidestep the on-device saturation
entirely.

Usage:
    python3 generate_hidden.py --meta nbg_meta_hidden_u8.json \\
        --embed token_embed.npy --prompt "def hello" --tokens 20
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
WORK = HERE.parent
ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/smollm2_llm"
SEQ_LEN = 32
HIDDEN = 576
PAD = 2
TOK = Tokenizer.from_file(str(WORK / "work/models/smollm2-135m-instruct/tokenizer.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def hello")
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--embed", required=True)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    q = list(meta["Outputs"].values())[0]["quantize"]
    embed = np.load(args.embed)  # [vocab, hidden]
    is_u8 = q["qtype"] == "u8"
    if is_u8:
        scale = q["scale"]; zp = q["zero_point"]
        dtype = np.uint8
        def dequant(raw):
            return (raw.astype(np.int32) - zp).astype(np.float32) * scale
    else:
        fl = q["fl"]; s = 1.0/(2**fl)
        dtype = np.int16
        def dequant(raw):
            return raw.astype(np.float32) * s

    ids = TOK.encode(args.prompt).ids
    if len(ids) < SEQ_LEN:
        window = [PAD] * (SEQ_LEN - len(ids)) + ids
    else:
        window = ids[-SEQ_LEN:]

    print(f"prompt: {args.prompt!r}")
    print(f"dtype: {dtype.__name__}")
    print(f"generating {args.tokens} tokens (top-k={args.top_k})...\n")
    print(args.prompt, end="", flush=True)

    generated = []
    total = 0.0
    for _ in range(args.tokens):
        (HERE / "input_0.dat").write_bytes(np.asarray([window], dtype=np.int32).tobytes())
        subprocess.run([ADB, "push", str(HERE / "input_0.dat"), f"{DEV_DIR}/input_0.dat"],
                       check=True, capture_output=True)
        t = time.perf_counter()
        subprocess.run([ADB, "shell",
            f"cd {DEV_DIR} && LD_LIBRARY_PATH=/vendor/lib64 "
            f"/data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"],
            check=True, capture_output=True)
        total += time.perf_counter() - t
        subprocess.run([ADB, "pull", f"{DEV_DIR}/output_0.dat", str(HERE / "output_0.dat")],
                       check=True, capture_output=True)
        raw = np.fromfile(HERE / "output_0.dat", dtype=dtype)
        hidden = dequant(raw).reshape(1, 1, SEQ_LEN, HIDDEN)
        last_h = hidden[0, 0, -1]
        logits = last_h @ embed.T

        if args.top_k > 0:
            idx = logits.argsort()[-args.top_k:]
            probs = np.exp((logits[idx] - logits[idx].max()) / args.temperature)
            probs /= probs.sum()
            next_tok = int(idx[np.random.choice(len(idx), p=probs)])
        else:
            next_tok = int(logits.argmax())

        generated.append(next_tok)
        print(TOK.decode([next_tok]), end="", flush=True)
        window = window[1:] + [next_tok]

    print(f"\n\n{args.tokens} tokens in {total:.2f}s = {args.tokens/total:.2f} tok/s")


if __name__ == "__main__":
    main()
