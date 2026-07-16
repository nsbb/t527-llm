#!/usr/bin/env python3
"""ONNX 패치: slice_last_hidden 노드 제거.

Acuity 6.12는 ONNX→TF 변환 시 seq_len 축의 마지막 토큰 slice에서
size = [1, -30, 32, 576] 로 잘못 계산해 `ValueError: Invalid value in tensor used
for shape: -30` 로 quantize 단계에서 실패한다.

우회: ONNX에서 slice_last_hidden 노드 자체를 제거하고 `final_last_token` 참조를
`final_rms_out` 로 리와이어. logits shape은 [1, 1, V] → [1, seq, V] 로 확장되며
마지막 토큰 로그잇은 CPU에서 `logits[:, -1, :]` 로 추출한다.

Usage:
    python3 patch_onnx_last_slice.py input.onnx output.onnx [--seq-len 32] [--vocab 49152]
"""
import argparse
from pathlib import Path

import onnx


def patch(input_path: Path, output_path: Path, seq_len: int, vocab: int) -> None:
    m = onnx.load(str(input_path))

    target_idx = None
    target_node = None
    for i, n in enumerate(m.graph.node):
        if n.name == "slice_last_hidden":
            target_idx, target_node = i, n
            break

    if target_node is None:
        raise SystemExit("no slice_last_hidden node found — is this the expected ONNX?")

    src_input = target_node.input[0]
    dst_output = target_node.output[0]
    print(f"Removing slice_last_hidden: {src_input} → {dst_output}")

    changed = 0
    for other in m.graph.node:
        for k, inp in enumerate(other.input):
            if inp == dst_output:
                other.input[k] = src_input
                changed += 1
    print(f"  redirected {changed} consumer inputs")

    del m.graph.node[target_idx]

    for out in m.graph.output:
        if out.name != "logits":
            continue
        s = out.type.tensor_type.shape
        while len(s.dim) > 0:
            s.dim.pop()
        for v in [1, seq_len, vocab]:
            d = s.dim.add()
            d.dim_value = v
        print(f"logits output shape updated to [1, {seq_len}, {vocab}]")

    onnx.save(m, str(output_path))
    print(f"saved: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=49152)
    args = ap.parse_args()
    patch(args.input, args.output, args.seq_len, args.vocab)


if __name__ == "__main__":
    main()
