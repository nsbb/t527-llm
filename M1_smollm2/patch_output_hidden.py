#!/usr/bin/env python3
"""Cut the LM head off the ONNX graph — output the final hidden state
(post-RMSNorm) instead of logits. Host then runs MatMul(final_hidden, W_embed^T)
in FP32 for the final projection to vocab.

Rationale:
    Hidden state values are ~normalized (RMSNorm output typically in ±3);
    int16 quantization fits cleanly with no saturation.
    LM_head is a giant matmul (hidden × vocab = 576 × 49152) whose output
    logits can exceed ±60 → int16 saturates on-device.
    Moving that final matmul to CPU (few million FLOPs, negligible latency)
    sidesteps the on-device saturation completely.

Usage:
    python3 patch_output_hidden.py input.onnx output.onnx [--seq-len 32]
"""
import argparse
import onnx
from pathlib import Path


def patch(input_path: Path, output_path: Path, seq_len: int) -> None:
    m = onnx.load(str(input_path))

    # Find "final_rms_out" tensor (RMSNorm output of last transformer block,
    # right before lm_head MatMul in petayyyy's graph)
    target = "final_rms_out"

    # Verify it exists as an intermediate tensor
    all_outputs = set()
    for n in m.graph.node:
        for o in n.output:
            all_outputs.add(o)
    if target not in all_outputs:
        # Try alternative
        for candidate in ("final_last_token", "final_hidden", "final_rms_out"):
            if candidate in all_outputs:
                target = candidate
                break
        else:
            raise SystemExit(f"could not find final hidden tensor; nodes ending in _rms_out or _hidden: {sorted(o for o in all_outputs if 'rms_out' in o or 'hidden' in o)[:10]}")

    print(f"exposing tensor as graph output: {target}")

    # Remove downstream: any node consuming final_rms_out or its descendants beyond lm_head
    # Simplest: rewrite graph to have final_rms_out as the sole output, drop all lm_head MatMul + Concat + logits nodes
    keep_output_name = target

    # Collect nodes to keep — do BFS/topological from inputs up to (and including) the node producing keep_output_name
    keep_nodes = []
    keep_tensor = {inp.name for inp in m.graph.input}
    keep_tensor.update(i.name for i in m.graph.initializer)
    # Do simple forward pass: iterate nodes in order, keep if all inputs available OR this node produces keep_output
    producers_of_kept = set()
    for n in m.graph.node:
        if keep_output_name in n.output:
            # produce this node's output — keep
            keep_nodes.append(n)
            for o in n.output:
                keep_tensor.add(o)
            producers_of_kept.add(id(n))
            # after finding producer of keep_output, prune
            break
        # else: keep if it feeds forward and hasn't been established as unused
        keep_nodes.append(n)
        for o in n.output:
            keep_tensor.add(o)
    print(f"keeping {len(keep_nodes)} of {len(m.graph.node)} nodes")

    # Rebuild graph
    del m.graph.node[:]
    m.graph.node.extend(keep_nodes)

    # Remove old outputs, set new
    del m.graph.output[:]
    # Infer hidden dim
    from onnx import TensorProto, helper
    hidden_dim = 576  # SmolLM2; may need param for Qwen (896)
    # Try to auto-detect from initializers via ln gamma shape
    for init in m.graph.initializer:
        if 'final_rms' in init.name and 'gamma' in init.name:
            hidden_dim = int(init.dims[-1]) if init.dims else hidden_dim
            break
    new_out = helper.make_tensor_value_info(target, TensorProto.FLOAT, [1, seq_len, hidden_dim])
    m.graph.output.append(new_out)

    onnx.save(m, str(output_path))
    print(f"saved: {output_path}  (output = {target} shape=[1,{seq_len},{hidden_dim}])")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--seq-len", type=int, default=32)
    args = ap.parse_args()
    patch(args.input, args.output, args.seq_len)


if __name__ == "__main__":
    main()
