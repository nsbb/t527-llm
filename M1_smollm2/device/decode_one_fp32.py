#!/usr/bin/env python3
"""1-step decode with FP32 NBG (accurate but slow: ~7s/forward).

Uses the axis-fixed v3 FP32 NBG on T527 (626 MB). No quantization loss.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
WORK = HERE.parent
ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/smollm2_llm_fp32"
SEQ_LEN = 32
VOCAB = 49152
PAD = 2


def encode(text: str) -> np.ndarray:
    tok = Tokenizer.from_file(str(WORK / "work/models/smollm2-135m-instruct/tokenizer.json"))
    ids = tok.encode(text).ids
    if len(ids) < SEQ_LEN:
        ids = [PAD] * (SEQ_LEN - len(ids)) + ids
    else:
        ids = ids[-SEQ_LEN:]
    return np.asarray([ids], dtype=np.int32)


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    tok = Tokenizer.from_file(str(WORK / "work/models/smollm2-135m-instruct/tokenizer.json"))

    tokens = encode(prompt)
    print(f"prompt: {prompt!r}")
    print(f"tokens (last-10): {tokens[0, -10:].tolist()}")

    tokens.astype(np.int32).tofile(HERE / "input_0.dat")
    subprocess.run([ADB, "push", str(HERE / "input_0.dat"), f"{DEV_DIR}/input_0.dat"],
                   check=True, capture_output=True)
    print("running FP32 NBG on NPU (~7s)...")
    r = subprocess.run([ADB, "shell",
        f"cd {DEV_DIR} && LD_LIBRARY_PATH=/vendor/lib64 "
        f"/data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"],
        check=True, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "run time" in line:
            print(f"  {line.strip()}")
    subprocess.run([ADB, "pull", f"{DEV_DIR}/output_0.dat", str(HERE / "output_fp32_decode.dat")],
                   check=True, capture_output=True)

    raw = np.fromfile(HERE / "output_fp32_decode.dat", dtype=np.float32).reshape(1, 1, SEQ_LEN, VOCAB)
    last = raw[0, 0, -1]
    top5 = last.argsort()[-5:][::-1].tolist()
    print(f"\nnext token top-5:")
    for tid in top5:
        print(f"  {tid:>6} ({last[tid]:+7.3f}) → {tok.decode([tid])!r}")


if __name__ == "__main__":
    main()
