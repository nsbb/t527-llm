#!/usr/bin/env python3
"""1-step decode with Qwen int16 NBG on T527."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
WORK = HERE.parent
ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/qwen_llm"
SEQ_LEN = 32
PAD = 151643
TOK = Tokenizer.from_file(str(WORK / "work/models/qwen2.5-0.5b-instruct/tokenizer.json"))
VOCAB = 151936


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="한국의 수도는")
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    q = list(meta["Outputs"].values())[0]["quantize"]
    fl = q["fl"]           # fractional length
    scale = 1.0 / (2 ** fl)

    ids = TOK.encode(args.prompt).ids
    if len(ids) < SEQ_LEN:
        ids = [PAD] * (SEQ_LEN - len(ids)) + ids
    else:
        ids = ids[-SEQ_LEN:]
    tokens = np.asarray([ids], dtype=np.int32)
    print(f"prompt: {args.prompt!r}")
    print(f"tokens (last-10): {tokens[0, -10:].tolist()}")

    tokens.tofile(HERE / "input_0.dat")
    subprocess.run([ADB, "push", str(HERE / "input_0.dat"), f"{DEV_DIR}/input_0.dat"],
                   check=True, capture_output=True)
    r = subprocess.run([ADB, "shell",
        f"cd {DEV_DIR} && LD_LIBRARY_PATH=/vendor/lib64 "
        f"/data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"],
        check=True, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "run time" in line: print("  ", line.strip())
    subprocess.run([ADB, "pull", f"{DEV_DIR}/output_0.dat", str(HERE / "output_0.dat")],
                   check=True, capture_output=True)

    raw = np.fromfile(HERE / "output_0.dat", dtype=np.int16)
    logits = (raw.astype(np.float32) * scale).reshape(1, 1, SEQ_LEN, VOCAB)
    last = logits[0, 0, -1]
    top10 = last.argsort()[-10:][::-1].tolist()
    print(f"\ntop-10 next tokens:")
    for tid in top10:
        print(f"  {tid:>7} ({last[tid]:+7.3f}) → {TOK.decode([tid])!r}")


if __name__ == "__main__":
    main()
