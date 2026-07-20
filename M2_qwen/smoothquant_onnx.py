#!/usr/bin/env python3
"""SmoothQuant applied directly to a static-shape ONNX LLM graph.

Algorithm (Xiao et al. 2022):
    For each Linear Y = X @ W (X: [..., C_in], W: [C_in, C_out]):
        s_j = max(|X_:,j|)^alpha / max(|W_j,:|)^(1-alpha)
        X' = X / s               (elementwise per input channel)
        W' = W * s[:, None]      (per input row)
        Y = X' @ W' == X @ W     (mathematically equivalent)

Goal: shift activation outliers into weights so that per-tensor quantization
of the *activation* uses a smaller range (int8/uint8 range fits without
clipping), while the weight range grows but tolerates it (weights are known
at quantize time and don't get outlier-clipped the way activations do).

Practical: insert a Mul(1/s) node before each MatMul and scale its weight
initializer in place. Acuity/pegasus imports the modified ONNX exactly the
same way — only the numerical distributions shift.

Usage:
    python3 smoothquant_onnx.py --input <onnx> --output <onnx> \\
        --calib-dir <dir_of_calib_npy> --tokenizer <tokenizer.json> \\
        --alpha 0.5
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto


def find_matmul_weights(model: onnx.ModelProto):
    """Return list of (matmul_node, weight_name, weight_np, is_first_input_weight).

    In LLM graphs the layout is Y = X @ W where W is the second input (initializer).
    Sometimes lm_head does W @ X.T style; we handle both.
    """
    initializers = {i.name: i for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type != "MatMul":
            continue
        # find weight (input that is an initializer)
        for slot in (1, 0):
            if slot < len(n.input) and n.input[slot] in initializers:
                w_init = initializers[n.input[slot]]
                w = numpy_helper.to_array(w_init)
                # only treat 2-D weights as Linear
                if w.ndim == 2:
                    out.append((n, n.input[slot], w, slot))
                break
    return out


def collect_activation_stats(onnx_path: Path, calib_files: "list[Path]",
                              target_input_names: "list") -> "dict":
    """For each activation name in target_input_names, collect per-channel abs-max
    across all calibration samples using ONNX Runtime with exposed intermediate
    outputs.
    """
    m = onnx.load(str(onnx_path))
    # add each target as graph output so ORT surfaces it
    existing = {o.name for o in m.graph.output}
    for name in target_input_names:
        if name in existing:
            continue
        # infer shape unknown; leave None
        vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        m.graph.output.append(vi)
    tmp_path = onnx_path.with_suffix(".sq_tmp.onnx")
    onnx.save(m, str(tmp_path))

    sess = ort.InferenceSession(str(tmp_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    stats = {n: None for n in target_input_names}
    for cf in calib_files:
        tokens = np.load(str(cf)).astype(np.int32)
        outs = sess.run(target_input_names, {input_name: tokens})
        for name, arr in zip(target_input_names, outs):
            a = np.abs(arr.reshape(-1, arr.shape[-1])).max(axis=0)
            if stats[name] is None:
                stats[name] = a
            else:
                stats[name] = np.maximum(stats[name], a)
    tmp_path.unlink(missing_ok=True)
    return stats


def apply_smoothquant(input_path: Path, output_path: Path,
                      calib_dir: Path, alpha: float = 0.5,
                      skip_lm_head: bool = True) -> None:
    m = onnx.load(str(input_path))

    matmuls = find_matmul_weights(m)
    print(f"found {len(matmuls)} MatMul-with-weight nodes")

    # Filter: only keep ones where weight is (in_dim, out_dim) with in_dim on axis 0
    # (this is standard for HF-style transposed weights). Skip lm_head (last MatMul).
    candidates = []
    for node, wname, w, slot in matmuls:
        # We need the activation tensor for this Linear = the OTHER input
        act_slot = 0 if slot == 1 else 1
        act_name = node.input[act_slot]
        # in_dim is on axis 0 of W if slot=1 (Y = X @ W [in, out]),
        # or axis 1 if slot=0 (Y = W @ X where W is [out, in]) — rare.
        if slot == 1:
            in_dim = w.shape[0]
        else:
            in_dim = w.shape[1]
        candidates.append((node, act_name, wname, w, slot, in_dim))
    # Skip last MatMul if requested — that's lm_head with huge vocab.
    if skip_lm_head and candidates:
        last_node = candidates[-1][0]
        candidates = [c for c in candidates if c[0].name != last_node.name]
        print(f"skip lm_head: {last_node.name}")

    print(f"{len(candidates)} Linear ops to SmoothQuant with alpha={alpha}")

    # Collect activation stats via ORT
    calib_files = sorted(calib_dir.glob("calib_*.npy"))
    print(f"using {len(calib_files)} calib files")
    target_acts = list({c[1] for c in candidates})
    print(f"collecting per-channel abs-max for {len(target_acts)} activation tensors...")
    act_max = collect_activation_stats(input_path, calib_files, target_acts)

    # Now transform: insert Mul(1/s) before Linear, multiply weight by s
    init_by_name = {i.name: i for i in m.graph.initializer}
    added_scales = 0
    new_inits = []
    node_replacements = {}  # old_input_name -> new_input_name (for a given node only)

    for node, act_name, wname, w, slot, in_dim in candidates:
        a_max = act_max[act_name]
        if a_max is None:
            print(f"  no stats for {act_name}, skip")
            continue
        if a_max.shape[0] != in_dim:
            print(f"  shape mismatch: act_max {a_max.shape} vs in_dim {in_dim} for {node.name}, skip")
            continue

        # Weight abs-max per input channel:
        if slot == 1:
            w_max = np.abs(w).max(axis=1)   # per row = per in-channel
        else:
            w_max = np.abs(w).max(axis=0)

        s = np.power(np.maximum(a_max, 1e-8), alpha) / np.power(np.maximum(w_max, 1e-8), 1 - alpha)
        # clip s to avoid extreme values
        s = np.clip(s, 1e-4, 1e4).astype(np.float32)

        # New weight = W * s (along in-channel axis)
        if slot == 1:
            new_w = w * s[:, None]
        else:
            new_w = w * s[None, :]

        # Replace weight initializer
        new_w_init = numpy_helper.from_array(new_w.astype(np.float32), name=wname)
        # remove old initializer and add new (in-place)
        init_by_name[wname].CopyFrom(new_w_init)

        # Insert Mul(1/s) before Linear on the activation input
        inv_s = (1.0 / s).astype(np.float32)
        scale_const_name = f"sq_scale_{node.name}"
        # broadcast shape [in_dim]; will broadcast over ...,in_dim tensor
        scale_init = numpy_helper.from_array(inv_s, name=scale_const_name)
        new_inits.append(scale_init)

        new_act_name = f"{act_name}__sq_{node.name}"
        mul_node = helper.make_node(
            "Mul",
            inputs=[act_name, scale_const_name],
            outputs=[new_act_name],
            name=f"sq_mul_{node.name}"
        )
        # insert before the current node in graph node list
        # (append later; we track new inputs via replacement map)
        node_replacements.setdefault(id(node), []).append((act_name, new_act_name))
        # rewire the MatMul input
        act_slot_val = 0 if slot == 1 else 1
        node.input[act_slot_val] = new_act_name
        # attach the Mul node right before (we'll re-order below)
        m.graph.node.insert(list(m.graph.node).index(node), mul_node)
        added_scales += 1

    for init in new_inits:
        m.graph.initializer.append(init)
    print(f"inserted {added_scales} SmoothQuant scale Mul nodes + weight updates")

    onnx.save(m, str(output_path))
    print(f"saved: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-skip-lm-head", action="store_true")
    args = ap.parse_args()
    apply_smoothquant(args.input, args.output, args.calib_dir,
                      alpha=args.alpha, skip_lm_head=not args.no_skip_lm_head)


if __name__ == "__main__":
    main()
