#!/usr/bin/env python3
"""1-step decode: prompt → NPU → next token.

Uses the axis-fixed v3 uint8 NBG on T527. Since uint8 quantization degrades
accuracy significantly (see eval_report.md), the produced token is unlikely
to be semantically correct — this is a pipeline-end-to-end demo only.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).parent
WORK = HERE.parent
ADB = "/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe"
DEV_DIR = "/data/local/tmp/smollm2_llm"

SEQ_LEN = 32
VOCAB = 49152
PAD_TOKEN = 2   # SmolLM2 pad token id


def encode_prompt(text: str) -> np.ndarray:
    tok = Tokenizer.from_file(str(WORK / "work/models/smollm2-135m-instruct/tokenizer.json"))
    ids = tok.encode(text).ids
    # left pad to SEQ_LEN
    if len(ids) < SEQ_LEN:
        ids = [PAD_TOKEN] * (SEQ_LEN - len(ids)) + ids
    else:
        ids = ids[-SEQ_LEN:]
    return np.asarray([ids], dtype=np.int32)


def push_input(token_ids: np.ndarray) -> None:
    input_path = HERE / "input_0.dat"
    token_ids.astype(np.int32).tofile(input_path)
    subprocess.run([ADB, "push", str(input_path), f"{DEV_DIR}/input_0.dat"],
                   check=True, capture_output=True)


def run_npu() -> None:
    subprocess.run([ADB, "shell",
        f"cd {DEV_DIR} && LD_LIBRARY_PATH=/vendor/lib64 "
        f"/data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0"],
        check=True, capture_output=True)


def pull_output() -> np.ndarray:
    out_path = HERE / "output_decode.dat"
    subprocess.run([ADB, "pull", f"{DEV_DIR}/output_0.dat", str(out_path)],
                   check=True, capture_output=True)
    return np.fromfile(out_path, dtype=np.uint8)


def dequantize(raw: np.ndarray, scale: float, zp: int) -> np.ndarray:
    return (raw.astype(np.int32) - zp).astype(np.float32) * scale


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
    print(f"prompt: {repr(prompt)}")

    tokens = encode_prompt(prompt)
    print(f"tokens: {tokens[0].tolist()}")

    push_input(tokens)
    print("running NPU...")
    run_npu()
    raw = pull_output()

    meta = json.loads((WORK / "nbg_meta.json").read_text())
    out_info = list(meta["Outputs"].values())[0]
    scale = out_info["quantize"]["scale"]
    zp = out_info["quantize"]["zero_point"]

    logits_flat = dequantize(raw, scale, zp)
    logits = logits_flat.reshape(1, 1, SEQ_LEN, VOCAB)

    last = logits[0, 0, -1]
    argmax = int(last.argmax())
    top5 = last.argsort()[-5:][::-1].tolist()

    tok = Tokenizer.from_file(str(WORK / "work/models/smollm2-135m-instruct/tokenizer.json"))
    print(f"\nnext token argmax: {argmax} = {tok.decode([argmax])!r}")
    print(f"top-5: {top5}")
    for tid in top5:
        print(f"  {tid:>6} → {tok.decode([tid])!r}")


if __name__ == "__main__":
    main()
