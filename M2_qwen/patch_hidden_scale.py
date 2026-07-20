#!/usr/bin/env python3
"""Force a smaller max_value/min_value on the hidden output tensor to sharpen
uint8 resolution. Trades saturation for precision.

For SmoothQuant-processed models, the final RMSNorm gamma absorbs outlier
magnitude → hidden state range widens far beyond what the actual signal
occupies. Actual FP32 hidden usually sits at ±30-50, but calibration sees the
outlier tail and picks max=214.

Empirically clipping to the 99th-percentile of activation gives much better
resolution with tolerable clipping loss (a few % of values saturate).
"""
import argparse, re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--target", default="attach_Mul_final_rms_gamma", help="substring in tensor name")
    ap.add_argument("--new-max", type=float, default=50.0, help="new max_value")
    ap.add_argument("--new-min", type=float, default=-50.0, help="new min_value")
    args = ap.parse_args()

    text = args.input.read_text()
    # Find the block and rewrite max/min/scale/zero_point
    # asymmetric_affine uint8: scale = (max - min) / 255, zp = round(-min / scale)
    scale = (args.new_max - args.new_min) / 255.0
    zp = int(round(-args.new_min / scale))
    print(f"target={args.target!r} → max={args.new_max} min={args.new_min} scale={scale:.6f} zp={zp}")

    lines = text.splitlines(keepends=True)
    out = []
    in_target = False
    patched = 0
    for line in lines:
        m = re.match(r"^  '([^']+)':\s*$", line)
        if m:
            in_target = args.target in m.group(1)
            if in_target:
                patched += 1
        if in_target and line.strip().startswith("max_value:"):
            line = re.sub(r"max_value:\s+[-\d.]+", f"max_value: {args.new_max}", line)
        elif in_target and line.strip().startswith("min_value:"):
            line = re.sub(r"min_value:\s+[-\d.]+", f"min_value: {args.new_min}", line)
        elif in_target and line.strip().startswith("scale:"):
            line = re.sub(r"scale:\s+[-\d.e+]+", f"scale: {scale}", line)
        elif in_target and line.strip().startswith("zero_point:"):
            line = re.sub(r"zero_point:\s+[-\d]+", f"zero_point: {zp}", line)
        out.append(line)

    args.output.write_text("".join(out))
    print(f"patched {patched} blocks")


if __name__ == "__main__":
    main()
