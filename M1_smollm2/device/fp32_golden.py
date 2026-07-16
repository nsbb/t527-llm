#!/usr/bin/env python3
"""FP32 golden logits via ONNX Runtime on the patched ONNX.

Runs the same token_ids through the CPU/FP32 ONNX (before quantization) to get
the reference logits, saved to fp32_logits.npy for the compare step.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent
ONNX = Path("/home/nsbb/travail/claude/T527/t527-llm/M1_smollm2/work/generated/smollm2_135m_w32/real_llm_nolastslice.onnx")

token_ids = np.fromfile(HERE / "input_0.dat", dtype=np.int32).reshape(1, 32)
print("token_ids:", token_ids.tolist())

sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
print(f"input: {inp_name} {sess.get_inputs()[0].shape}")
print(f"output: {out_name} {sess.get_outputs()[0].shape}")

logits = sess.run([out_name], {inp_name: token_ids})[0]
print("fp32 logits shape:", logits.shape, "dtype:", logits.dtype)
print("last-position argmax:", int(logits[0, -1].argmax()), "top5:",
      logits[0, -1].argsort()[-5:][::-1].tolist())

np.save(HERE / "fp32_logits.npy", logits)
print("saved:", HERE / "fp32_logits.npy")
