#!/usr/bin/env python3
"""M0 phase 2: 실패 op에 대한 조립 우회 검증.

Cos, LayerNorm(opset17), Gelu(opset20)이 native로 실패했으므로 조립 그래프로 대체.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from pathlib import Path

OUT = Path(__file__).parent / "onnx_dummies"

def save(model, name):
    onnx.checker.check_model(model)
    onnx.save(model, OUT / f"{name}.onnx")
    print(f"  wrote {name}.onnx")

def build(name, nodes, inputs, outputs, initializers=(), opset=17, ir_version=8):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    op_imports = [helper.make_opsetid("", opset)]
    model = helper.make_model(graph, opset_imports=op_imports, producer_name="m0")
    model.ir_version = ir_version
    save(model, name)

# ==============================================================
# Cos → Sin(π/2 - x)
# ==============================================================
half_pi = numpy_helper.from_array(np.array(np.pi / 2, np.float32), name="half_pi")
build(
    "cos_via_sin",
    [
        helper.make_node("Sub", ["half_pi", "X"], ["shifted"]),
        helper.make_node("Sin", ["shifted"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
    initializers=[half_pi],
)

# ==============================================================
# LayerNorm 분해 (Sub-mean → Var(pow+mean) → Add(eps) → Sqrt → Reciprocal → Mul → Mul(scale) → Add(bias))
# ==============================================================
scale = numpy_helper.from_array(np.ones((64,), np.float32), name="ln_scale")
bias = numpy_helper.from_array(np.zeros((64,), np.float32), name="ln_bias")
exp2 = numpy_helper.from_array(np.array(2.0, np.float32), name="ln_exp2")
eps = numpy_helper.from_array(np.array(1e-5, np.float32), name="ln_eps")
build(
    "layernorm_composed",
    [
        helper.make_node("ReduceMean", ["X"], ["mean"], axes=[-1], keepdims=1),
        helper.make_node("Sub", ["X", "mean"], ["x_centered"]),
        helper.make_node("Pow", ["x_centered", "ln_exp2"], ["x2"]),
        helper.make_node("ReduceMean", ["x2"], ["var"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["var", "ln_eps"], ["var_eps"]),
        helper.make_node("Sqrt", ["var_eps"], ["std"]),
        helper.make_node("Reciprocal", ["std"], ["inv_std"]),
        helper.make_node("Mul", ["x_centered", "inv_std"], ["normed"]),
        helper.make_node("Mul", ["normed", "ln_scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "ln_bias"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
    initializers=[scale, bias, exp2, eps],
)

# ==============================================================
# Gelu via Erf: 0.5 * x * (1 + erf(x / sqrt(2)))
# ==============================================================
half = numpy_helper.from_array(np.array(0.5, np.float32), name="g_half")
one = numpy_helper.from_array(np.array(1.0, np.float32), name="g_one")
inv_sqrt2 = numpy_helper.from_array(np.array(1.0 / np.sqrt(2.0), np.float32), name="g_inv_sqrt2")
build(
    "gelu_erf_composed",
    [
        helper.make_node("Mul", ["X", "g_inv_sqrt2"], ["x_scaled"]),
        helper.make_node("Erf", ["x_scaled"], ["erf_x"]),
        helper.make_node("Add", ["erf_x", "g_one"], ["erf_plus"]),
        helper.make_node("Mul", ["X", "erf_plus"], ["prod"]),
        helper.make_node("Mul", ["prod", "g_half"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
    initializers=[half, one, inv_sqrt2],
)

# ==============================================================
# Gelu tanh 근사 (waz664 트릭): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# ==============================================================
gt_half = numpy_helper.from_array(np.array(0.5, np.float32), name="gt_half")
gt_one = numpy_helper.from_array(np.array(1.0, np.float32), name="gt_one")
gt_k = numpy_helper.from_array(np.array(np.sqrt(2.0 / np.pi), np.float32), name="gt_k")
gt_c = numpy_helper.from_array(np.array(0.044715, np.float32), name="gt_c")
gt_exp3 = numpy_helper.from_array(np.array(3.0, np.float32), name="gt_exp3")
build(
    "gelu_tanh_composed",
    [
        helper.make_node("Pow", ["X", "gt_exp3"], ["x3"]),
        helper.make_node("Mul", ["x3", "gt_c"], ["x3_c"]),
        helper.make_node("Add", ["X", "x3_c"], ["inner"]),
        helper.make_node("Mul", ["inner", "gt_k"], ["inner_k"]),
        helper.make_node("Tanh", ["inner_k"], ["t"]),
        helper.make_node("Add", ["t", "gt_one"], ["t1"]),
        helper.make_node("Mul", ["X", "t1"], ["prod"]),
        helper.make_node("Mul", ["prod", "gt_half"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
    initializers=[gt_half, gt_one, gt_k, gt_c, gt_exp3],
)

# ==============================================================
# Tanh 단독 (gelu_tanh 조립에 필요)
# ==============================================================
build(
    "tanh",
    [helper.make_node("Tanh", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
)

# ==============================================================
# Sub (LayerNorm/RoPE에 필요)
# ==============================================================
build(
    "sub",
    [helper.make_node("Sub", ["A", "B"], ["Y"])],
    [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 16, 64]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 16, 64]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
)

# ==============================================================
# SwiGLU / SiLU (Swish) — Qwen FFN
# SiLU(x) = x * sigmoid(x)
# ==============================================================
build(
    "sigmoid",
    [helper.make_node("Sigmoid", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
)

build(
    "silu_composed",
    [
        helper.make_node("Sigmoid", ["X"], ["s"]),
        helper.make_node("Mul", ["X", "s"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
)

# ==============================================================
# RoPE 최소 조립 (Sin/Cos 활용, cos는 sin(pi/2-x)로 대체)
# 입력: x (shape [1, S, D]) + 사전 계산된 sin/cos 테이블 (shape [S, D])
# 이 그래프는 실제 rotate 없이 sin/cos 곱만 시연 (M1에서 완전한 RoPE로 확장)
# ==============================================================
freq_sin = numpy_helper.from_array(np.random.randn(16, 64).astype(np.float32), name="freq_sin")
freq_cos_shift = numpy_helper.from_array(np.random.randn(16, 64).astype(np.float32), name="freq_cos_shift")  # π/2 - freq
build(
    "rope_minimal",
    [
        # sin 부분
        helper.make_node("Sin", ["freq_sin"], ["sin_vals"]),
        # cos 부분: sin(π/2 - freq)
        helper.make_node("Sin", ["freq_cos_shift"], ["cos_vals"]),
        # x_rotated = x * cos_vals + (rotate_half(x)) * sin_vals
        # rotate_half은 별도 op 필요하므로 여기선 단순 곱만 검증
        helper.make_node("Mul", ["X", "cos_vals"], ["x_cos"]),
        helper.make_node("Mul", ["X", "sin_vals"], ["x_sin"]),
        helper.make_node("Add", ["x_cos", "x_sin"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
    initializers=[freq_sin, freq_cos_shift],
)

print(f"\n✔ Wrote workaround ONNX to {OUT}/")
