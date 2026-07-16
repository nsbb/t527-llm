#!/usr/bin/env python3
"""Compare FP32 golden vs T527 int8 NB output.

FP32 logits: [1, 32, 49152] fp32
NB output:   [1, 1, 32, 49152] uint8, dequant = (u8 - zp) * scale

Metrics for the LAST position (index 31, where the actual next-token logits live):
  - Top-1 argmax match
  - Top-5 overlap
  - KL divergence softmax(fp32) || softmax(int8)
  - Cosine similarity
  - RMSE
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
META = json.loads((HERE.parent / "nbg_meta.json").read_text())
print("nbg_meta:", json.dumps(META, indent=2))

fp32 = np.load(HERE / "fp32_logits.npy")  # [1, 32, 49152] fp32
print("fp32 shape:", fp32.shape, "dtype:", fp32.dtype)

# NB output uint8, dequant scale+zp
raw = np.fromfile(HERE / "output_0.dat", dtype=np.uint8)
print("raw uint8 output bytes:", raw.nbytes, "unique:", np.unique(raw)[:5].tolist(), "...")

# Layout per vpm_run stdout: dim 49152 32 1 1 → C=49152, W=32, H=1, N=1
# Acuity typically returns row-major with the outermost dim first. Given output name
# uid_2459_out_0 and shape [1,1,32,49152] in nbg_meta, we try N=1 H=1 W=32 C=49152
# stride: (H*W*C, W*C, C, 1) = (1572864, 1572864, 49152, 1)
scale = 0.492302
zp = 133
try:
    q_i16 = raw.astype(np.int32) - zp
    logits_int8 = q_i16.astype(np.float32) * scale
    # try [1, 1, 32, 49152]
    logits_int8 = logits_int8.reshape(1, 1, 32, 49152)
    print("int8 dequant shape (try [1,1,32,49152]):", logits_int8.shape)
except Exception as e:
    print("reshape error:", e)

fp32_last = fp32[0, -1]                # [49152]
int8_last = logits_int8[0, 0, -1]      # [49152]

# ---------- metrics ----------
def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

fp32_argmax = int(fp32_last.argmax())
int8_argmax = int(int8_last.argmax())
top5_fp32 = fp32_last.argsort()[-5:][::-1].tolist()
top5_int8 = int8_last.argsort()[-5:][::-1].tolist()
overlap = len(set(top5_fp32) & set(top5_int8))

p = softmax(fp32_last)
q = softmax(int8_last)
kl = float(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30))))
cos = float(np.dot(fp32_last, int8_last) / (np.linalg.norm(fp32_last) * np.linalg.norm(int8_last) + 1e-30))
rmse = float(np.sqrt(np.mean((fp32_last - int8_last) ** 2)))

print("\n===== last-position (index 31) =====")
print(f"  fp32 argmax: {fp32_argmax}  |  top5: {top5_fp32}")
print(f"  int8 argmax: {int8_argmax}  |  top5: {top5_int8}")
print(f"  top1 match:  {fp32_argmax == int8_argmax}")
print(f"  top5 overlap: {overlap}/5")
print(f"  KL(fp32 || int8): {kl:.4f}")
print(f"  cosine similarity: {cos:.4f}")
print(f"  RMSE: {rmse:.4f}")

# also whole-sequence match rate
match_per_pos = []
for pos in range(32):
    match_per_pos.append(int(fp32[0, pos].argmax() == logits_int8[0, 0, pos].argmax()))
print(f"\nargmax match per position (32 total): {sum(match_per_pos)}/{32}")
print(f"  match array: {match_per_pos}")

# Save condensed report
report = {
    "fp32_argmax": fp32_argmax,
    "int8_argmax": int8_argmax,
    "top5_fp32": top5_fp32,
    "top5_int8": top5_int8,
    "top1_match_last": fp32_argmax == int8_argmax,
    "top5_overlap_last": overlap,
    "kl_last": kl,
    "cosine_last": cos,
    "rmse_last": rmse,
    "argmax_match_all_positions": match_per_pos,
    "argmax_match_count": sum(match_per_pos),
}
(HERE / "compare_report.json").write_text(json.dumps(report, indent=2))
print(f"\nsaved: compare_report.json")
