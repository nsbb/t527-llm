#!/usr/bin/env python3
"""ONNX 패치: 모든 ReduceMean의 axes=[2] → axes=[-1].

Acuity 6.12 버그 우회: Acuity는 3D ONNX 텐서를 내부 4D 레이아웃 [N,C,H,W]로 확장하면서
축 번호를 shift한다. 원래 axis=2 (RMSNorm의 hidden dim)가 seq_len 축으로 잘못 매핑돼
RMSNorm 계산 자체가 broken (cos=0.87 divergence 시작 지점).

해결: axes=[-1]로 지정하면 Acuity가 항상 last dim으로 해석 → 확장된 4D에서도 정상 동작.

Usage:
    python3 patch_reducemean_axes.py input.onnx output.onnx
"""
import argparse
from pathlib import Path

import onnx


def patch(input_path: Path, output_path: Path) -> None:
    m = onnx.load(str(input_path))
    patched = 0
    for n in m.graph.node:
        if n.op_type != 'ReduceMean':
            continue
        for a in n.attribute:
            if a.name == 'axes' and list(a.ints) == [2]:
                del a.ints[:]
                a.ints.append(-1)
                patched += 1
    print(f"patched {patched} ReduceMean nodes: axes=[2] → axes=[-1]")
    onnx.save(m, str(output_path))
    print(f"saved: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    patch(args.input, args.output)


if __name__ == "__main__":
    main()
