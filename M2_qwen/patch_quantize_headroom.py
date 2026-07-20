#!/usr/bin/env python3
"""Reduce fractional length (fl) of specific tensors in an Acuity .quantize file.

Acuity's dynamic_fixed_point int16 uses fl to control scale (scale = 1/2^fl).
Smaller fl → wider representable range → less saturation but coarser resolution.

Empirically, T527 NPU produces values that exceed calibration range (host CPU
emulation vs on-device numerical drift). Reducing fl by 1 for the output
logits tensor doubles the range and eliminates saturation while sacrificing
1-bit resolution.

Usage:
    python3 patch_quantize_headroom.py <input.quantize> <output.quantize> \\
        [--reduce-by 1] [--target 'logits']
"""
import argparse
import re
from pathlib import Path


def patch(input_path: Path, output_path: Path, reduce_by: int,
          target_substr: str) -> None:
    src = input_path.read_text()
    # Match blocks: 'tensor_name': ... fl: X
    out_lines = []
    lines = src.splitlines(keepends=True)
    i = 0
    in_target = False
    patched = 0
    while i < len(lines):
        line = lines[i]
        # detect the block header (starts with 2-space indent, quoted name, colon)
        m = re.match(r"^  '([^']+)':\s*$", line)
        if m:
            name = m.group(1)
            in_target = target_substr in name
        if in_target:
            fm = re.match(r"^(\s+fl:\s+)(-?\d+)(\s*)$", line)
            if fm:
                indent, val, tail = fm.groups()
                new_val = int(val) - reduce_by
                new_line = f"{indent}{new_val}{tail}"
                if not new_line.endswith("\n"):
                    new_line += "\n"
                out_lines.append(new_line)
                print(f"  {name}: fl {val} → {new_val}")
                patched += 1
                i += 1
                continue
        out_lines.append(line)
        i += 1
    output_path.write_text("".join(out_lines))
    print(f"patched {patched} tensors matching '{target_substr}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--reduce-by", type=int, default=1)
    ap.add_argument("--target", default="logits", help="substring in tensor name")
    args = ap.parse_args()
    patch(args.input, args.output, args.reduce_by, args.target)


if __name__ == "__main__":
    main()
