#!/usr/bin/env python3
"""M0: LLM-critical op별 최소 ONNX 그래프 생성.

각 op를 단독 노드로 담은 min ONNX를 만들고, pegasus import 성공 여부로
Acuity 6.12 지원 여부를 실측한다.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from pathlib import Path

OUT = Path(__file__).parent / "onnx_dummies"
OUT.mkdir(parents=True, exist_ok=True)

def save(model, name):
    onnx.checker.check_model(model)
    onnx.save(model, OUT / f"{name}.onnx")
    print(f"  wrote {name}.onnx")

def make(name, opset=17, ir_version=8):
    return dict(name=name, opset=opset, ir_version=ir_version)

def build(name, nodes, inputs, outputs, initializers=(), opset=17, ir_version=8):
    graph = helper.make_graph(nodes, name, inputs, outputs, list(initializers))
    op_imports = [helper.make_opsetid("", opset)]
    model = helper.make_model(graph, opset_imports=op_imports, producer_name="m0")
    model.ir_version = ir_version
    save(model, name)


# ==============================================================
# 1. Sin (RoPE 필수)
# ==============================================================
build(
    "sin",
    [helper.make_node("Sin", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
)

# ==============================================================
# 2. Cos (RoPE 필수)
# ==============================================================
build(
    "cos",
    [helper.make_node("Cos", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
)

# ==============================================================
# 3. Gather (Embedding lookup)
# ==============================================================
build(
    "gather",
    [helper.make_node("Gather", ["data", "indices"], ["Y"], axis=0)],
    [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, [100, 64]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [16]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [16, 64])],
)

# ==============================================================
# 4. GatherND (KV cache 슬라이싱 후보)
# ==============================================================
build(
    "gathernd",
    [helper.make_node("GatherND", ["data", "indices"], ["Y"])],
    [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 8, 16]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [4, 2]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 16])],
)

# ==============================================================
# 5. ScatterND (KV cache in-place 업데이트)
# ==============================================================
build(
    "scatternd",
    [helper.make_node("ScatterND", ["data", "indices", "updates"], ["Y"])],
    [
        helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 8, 16]),
        helper.make_tensor_value_info("indices", TensorProto.INT64, [4, 2]),
        helper.make_tensor_value_info("updates", TensorProto.FLOAT, [4, 16]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16])],
)

# ==============================================================
# 6. LayerNormalization (opset 17 native)
# ==============================================================
scale = numpy_helper.from_array(np.ones((64,), np.float32), name="scale")
bias = numpy_helper.from_array(np.zeros((64,), np.float32), name="bias")
build(
    "layernorm",
    [helper.make_node("LayerNormalization", ["X", "scale", "bias"], ["Y"], axis=-1, epsilon=1e-5)],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
    initializers=[scale, bias],
)

# ==============================================================
# 7. Softmax
# ==============================================================
build(
    "softmax",
    [helper.make_node("Softmax", ["X"], ["Y"], axis=-1)],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
)

# ==============================================================
# 8. Pow (RMSNorm 구성)
# ==============================================================
exp = numpy_helper.from_array(np.array(2.0, np.float32), name="exp")
build(
    "pow",
    [helper.make_node("Pow", ["X", "exp"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
    initializers=[exp],
)

# ==============================================================
# 9. ReduceMean (RMSNorm 구성)
# ==============================================================
build(
    "reducemean",
    [helper.make_node("ReduceMean", ["X"], ["Y"], axes=[-1], keepdims=1)],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 1])],
)

# ==============================================================
# 10. Sqrt + Reciprocal (RMSNorm의 Rsqrt 조립)
# ==============================================================
build(
    "sqrt",
    [helper.make_node("Sqrt", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 1])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 1])],
)

build(
    "reciprocal",
    [helper.make_node("Reciprocal", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 1])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 1])],
)

# ==============================================================
# 11. CumSum (attention mask 생성 후보)
# ==============================================================
axis = numpy_helper.from_array(np.array(-1, np.int64), name="axis")
build(
    "cumsum",
    [helper.make_node("CumSum", ["X", "axis"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16])],
    initializers=[axis],
)

# ==============================================================
# 12. MatMul (baseline sanity)
# ==============================================================
build(
    "matmul",
    [helper.make_node("MatMul", ["A", "B"], ["Y"])],
    [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 16, 64]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 64, 32]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 32])],
)

# ==============================================================
# 13. Erf (GELU 구성)
# ==============================================================
build(
    "erf",
    [helper.make_node("Erf", ["X"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
)

# ==============================================================
# 14. Gelu (opset 20 native — 6.12에서 아마 안 됨, Erf 조립으로 대체 확인용)
# ==============================================================
try:
    build(
        "gelu",
        [helper.make_node("Gelu", ["X"], ["Y"], approximate="none")],
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64])],
        opset=20,
        ir_version=10,
    )
except Exception as e:
    print(f"  gelu (opset 20) skipped: {e}")

# ==============================================================
# 15. Where (causal mask 후보)
# ==============================================================
build(
    "where",
    [helper.make_node("Where", ["cond", "X", "Y"], ["Z"])],
    [
        helper.make_tensor_value_info("cond", TensorProto.BOOL, [1, 16, 16]),
        helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 16]),
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 16]),
    ],
    [helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 16, 16])],
)

# ==============================================================
# 16. Mul (RoPE, RMSNorm 등 반복 사용)
# ==============================================================
build(
    "mul",
    [helper.make_node("Mul", ["A", "B"], ["Y"])],
    [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 16, 64]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 16, 64]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
)

# ==============================================================
# 17. Add
# ==============================================================
build(
    "add",
    [helper.make_node("Add", ["A", "B"], ["Y"])],
    [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 16, 64]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 16, 64]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
)

# ==============================================================
# 18. Slice (KV cache 슬라이싱 후보)
# ==============================================================
starts = numpy_helper.from_array(np.array([0], np.int64), name="starts")
ends = numpy_helper.from_array(np.array([8], np.int64), name="ends")
axes = numpy_helper.from_array(np.array([1], np.int64), name="axes")
build(
    "slice",
    [helper.make_node("Slice", ["X", "starts", "ends", "axes"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 64])],
    initializers=[starts, ends, axes],
)

# ==============================================================
# 19. Concat (KV cache append 후보)
# ==============================================================
build(
    "concat",
    [helper.make_node("Concat", ["A", "B"], ["Y"], axis=1)],
    [
        helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 8, 64]),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, 8, 64]),
    ],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
)

# ==============================================================
# 20. Transpose
# ==============================================================
build(
    "transpose",
    [helper.make_node("Transpose", ["X"], ["Y"], perm=[0, 2, 1])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 64, 16])],
)

# ==============================================================
# 21. Reshape
# ==============================================================
shape = numpy_helper.from_array(np.array([1, 8, 128], np.int64), name="shape")
build(
    "reshape",
    [helper.make_node("Reshape", ["X", "shape"], ["Y"])],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 128])],
    initializers=[shape],
)

# ==============================================================
# 22. Split (GQA head 분리 후보)
# ==============================================================
build(
    "split",
    [helper.make_node("Split", ["X"], ["Y1", "Y2"], axis=-1)],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [
        helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 16, 32]),
        helper.make_tensor_value_info("Y2", TensorProto.FLOAT, [1, 16, 32]),
    ],
)

# ==============================================================
# 23. RMSNorm 조립 그래프 (Pow → ReduceMean → Add → Sqrt → Reciprocal → Mul → Mul)
# ==============================================================
weight = numpy_helper.from_array(np.ones((64,), np.float32), name="weight")
exp2 = numpy_helper.from_array(np.array(2.0, np.float32), name="exp2")
eps = numpy_helper.from_array(np.array(1e-5, np.float32), name="eps")
build(
    "rmsnorm_composed",
    [
        helper.make_node("Pow", ["X", "exp2"], ["x2"]),
        helper.make_node("ReduceMean", ["x2"], ["m"], axes=[-1], keepdims=1),
        helper.make_node("Add", ["m", "eps"], ["m_eps"]),
        helper.make_node("Sqrt", ["m_eps"], ["s"]),
        helper.make_node("Reciprocal", ["s"], ["inv"]),
        helper.make_node("Mul", ["X", "inv"], ["x_norm"]),
        helper.make_node("Mul", ["x_norm", "weight"], ["Y"]),
    ],
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 16, 64])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 16, 64])],
    initializers=[weight, exp2, eps],
)

print(f"\n✔ Wrote {len(list(OUT.glob('*.onnx')))} dummy ONNX files to {OUT}/")
