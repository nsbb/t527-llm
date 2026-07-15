# M0: Acuity 6.12.0 Op Support Matrix (실측)

_실측일 2026-07-15 | Docker: t527-npu:v1.2 | Acuity: 6.12.0 binary | 방법: dummy ONNX × 33 → `pegasus import` batch_

---

## 요약

**33개 dummy ONNX 중 30개 OK, 3개 FAIL — 모든 FAIL은 조립 그래프로 완전 우회 확인됨.**

Native로 실패한 op:
- `Cos` (Sin은 됨) — **`Sin(π/2 - x)` 로 우회 OK**
- `LayerNormalization` (opset 17 native) — **RMSNorm 방식 조립 OK**
- `Gelu` (opset 20 native) — **Erf 조립 OK, tanh 근사도 OK (waz664 트릭)**

→ **LLM에 필요한 모든 연산이 Acuity 6.12에서 native 또는 조립으로 지원됨. M1 진행 가능.**

---

## Native 지원 (직접 사용 가능, 21개)

| Op | 용도 |
|---|---|
| Add | 잔차 연결, bias 추가 |
| Sub | LayerNorm 중심화 (x - mean) |
| Mul | Broadcasting, scale, gating |
| MatMul | Attention Q·K^T, FFN, lm_head |
| Sin | RoPE |
| ~~Cos~~ | ❌ → `Sin(π/2 - x)` 조립 (검증됨) |
| Pow | RMSNorm, GELU tanh 근사 |
| Sqrt | RMSNorm, LayerNorm std |
| Reciprocal | RSqrt 조립 |
| ReduceMean | LayerNorm/RMSNorm 평균 |
| Erf | GELU 조립 |
| Sigmoid | SiLU/Swish, gating |
| Tanh | GELU tanh 근사 |
| Softmax | Attention scores |
| Gather | Embedding lookup, KV cache 인덱싱 |
| GatherND | 고차원 KV cache 슬라이싱 |
| ScatterND | KV cache in-place 업데이트 |
| Slice | KV cache 슬라이싱 (대체 경로) |
| Concat | KV cache append |
| Split | GQA head 분리 |
| Reshape | 텐서 변형 |
| Transpose | Attention head 재배열 |
| Where | Causal mask 적용 |
| CumSum | Position ID 생성 |

## 조립 지원 (Composed, 검증됨)

| 조립 op | 구성 |
|---|---|
| **Cos** | `Sub(π/2, x) → Sin` |
| **LayerNorm** | `ReduceMean → Sub → Pow(2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul → Mul(scale) → Add(bias)` |
| **RMSNorm** | `Pow(2) → ReduceMean → Add(eps) → Sqrt → Reciprocal → Mul → Mul(weight)` |
| **GELU (Erf)** | `Mul(1/√2) → Erf → Add(1) → Mul(x) → Mul(0.5)` |
| **GELU (tanh)** | `Pow(3) → Mul(0.044715) → Add(x) → Mul(√(2/π)) → Tanh → Add(1) → Mul(x) → Mul(0.5)` (waz664 트릭) |
| **SiLU/Swish** | `Sigmoid → Mul(x)` |
| **RoPE (minimal)** | Sin + Cos(via Sin) + Mul + Add — M1에서 rotate_half 포함 완전판 확장 필요 |

## Native FAIL (3개, 전부 조립 우회 성공)

| Op | 실패 원인 | 우회 |
|---|---|---|
| `Cos` | Acuity 내부 converter에서 `Cos_Y:out0` 매칭 실패 (`Not match tensor`) | Sin(π/2 - x) |
| `LayerNormalization` (opset 17) | `Un Specify LayerNormalization smart processor` — Acuity 6.12에 opset 17 fused LN 매핑 없음 | 분해 조립 |
| `Gelu` (opset 20) | `Un Specify Gelu smart processor` — 마찬가지로 fused Gelu 매핑 없음 | Erf 또는 tanh 근사 조립 |

---

## LLM 아키텍처 관점 시사

**Transformer 블록 필수 연산 전부 이 매트릭스로 커버 가능:**

- **Attention**: MatMul + Softmax + Mul(scaling) + Where(causal mask) + Split(GQA) + Transpose ✓
- **RoPE**: Sin + Cos(조립) + Mul + Add ✓
- **RMSNorm**: 조립 ✓ (Qwen/Llama 계열)
- **LayerNorm**: 조립 ✓ (older 모델용)
- **FFN (SwiGLU)**: MatMul + SiLU(조립) + Mul + MatMul ✓
- **FFN (GELU)**: MatMul + Gelu(조립) + MatMul ✓
- **KV cache 관리**: Gather/GatherND + ScatterND + Slice + Concat ✓
- **Embedding**: Gather ✓
- **LM head**: MatMul ✓

**M1 (SmolLM2-135M) 진행 조건 모두 충족.**

---

## ONNX export 시 주의사항 (M1 이후 반영)

1. **`optimum-cli export onnx` 결과에 opset 17+ 의 native `LayerNormalization` 이나 opset 20 `Gelu` 노드가 있으면 pegasus import 실패.** ONNX 그래프 후처리로 강제 분해 (또는 opset 다운그레이드).
2. **Cos 노드도 마찬가지** — 모든 Cos 를 Sin(π/2 - x) 로 치환하는 스크립트가 M1의 `fix_onnx_for_acuity.py` 개조판에 반드시 들어가야 함.
3. Erf 는 native 지원 O — HuggingFace ONNX가 GELU를 Erf 기반 서브그래프로 export하는 경우가 많으므로 그대로 통과 가능성 있음. Gelu native op로 export되는 경우만 문제.

---

## 재현 방법

```bash
cd /home/nsbb/travail/claude/T527/t527-llm/M0_op_validation
python3 make_dummy_onnx.py      # 24 개 기본 op ONNX 생성
python3 make_workarounds.py     # 9 개 조립/추가 op ONNX 생성
./batch_import.sh                # docker + pegasus import 배치, results.csv 생성
```

산출물:
- `onnx_dummies/*.onnx` — 33 개
- `import_results/*.log` — 각 op의 pegasus import 전체 로그
- `import_results/*.json`, `*.data` — 성공한 op의 Acuity IR
- `results.csv` — 최종 매트릭스
