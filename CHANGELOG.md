# CHANGELOG

T527 NPU LLM 포팅 프로젝트. 날짜/버전별 진행 기록.

---

## v0.5.0 — 2026-07-20 (SmoothQuant 도입)

**M2 Qwen SmoothQuant 첫 실측.**

- `smoothquant_onnx.py`: ONNX 레벨 SmoothQuant 구현 (α 조절 가능)
- 167 MatMul-with-weight 노드 rewrite: `Mul(1/s)` 사전 삽입 + weight × s
- Host 정확도 실측 (calib_00 = 영어 프롬프트 기준):

  | 조합 | argmax match | top5 overlap | cos_last |
  |---|---|---|---|
  | baseline uint8 | 0/32 | 0.12/5 | 0.11 |
  | SQ α=0.5 + uint8 | 0/32 | 0.12/5 | 0.51 |
  | SQ α=0.8 + uint8 | 0/32 | 0.78/5 | 0.43 |
  | **SQ α=0.5 + int16** | **25/32** | **3.53/5** | 0.16 |

- SQ+int16 NB (887MB) 디바이스 push+실행 → 18% saturation 여전
- Host CPU 에뮬 vs T527 NPU int16 산술 numerical drift 확인
- 다음: calibration range 여유 추가, per-layer sensitivity, fl 수동 조정

Commit: `9726808`

---

## v0.4.0 — 2026-07-20 (Qwen2.5-0.5B 다국어 파일럿)

**M2 시작 — 한국어 지원 모델 파이프라인 검증.**

- Qwen2.5-0.5B-Instruct 다운로드 (943MB safetensors, HF direct via curl -L)
- Static ONNX 생성 (1.9GB, `make_real_llm_onnx.py`): 24-layer Llama-계열, GQA 14/2, vocab 151936
- Q/K/V bias 자동 처리 확인 (petayyyy `optional_tensor`)
- axis-fix + last-slice patch 재사용 (49 ReduceMean 노드)
- Acuity import → uint8 quantize (35 → 10 sample 캘리브레이션)
- **uint8 NB 393MB, 198ms/forward = 5 tok/s**
- **int16 dfp NB 891MB, 290ms/forward = 3.4 tok/s**
- 한국어 프롬프트 (`한국의 수도는`) tokens int32 [1,1,32] 디바이스 통과 확인
- Coherent 텍스트: 여전히 quant collapse

Commit: `faf1be2`

---

## v0.3.0 — 2026-07-16 (SmolLM2 실측 마무리)

**M1 완결 + 파이프라인 심화.**

- 10-sample 다양 프롬프트 캘리브레이션 → cos 0.11 → **0.93** 대폭 개선
- Multi-token sliding-window 생성 스크립트 (`generate.py`)
- Greedy → token 12 무한 반복, top-k=10 → varied but incoherent rare tokens
- 병목: adb 왕복 500ms + NPU 93ms → 1.5 tok/s

Commit: `9b013b9`

### FP32 NB 정확도 실증 (동일 날짜)

- FP32 NB export 성공 (626MB, non-quantized)
- T527에서 7.28s/forward SW emulation
- **실제 언어 tokens 출력**: "The capital of France is" → " as", " of", ".", " in", ","
- Cos=0.805 vs ORT FP32
- 결정적 실증: 정확도 문제는 quantization만, pipeline 자체는 완전

Commit: `6de9350`

---

## v0.2.0 — 2026-07-16 오후 (Acuity ReduceMean 축 버그 발견/수정)

**프로젝트 최대 발견.**

Layer-by-layer 발산 추적:
- Gather (embedding): cos=1.0000 ✓
- Layer 0 RMSNorm: **cos=0.8750 ← 발산 시작**
- lm_head: cos=-0.3818 (30 layer 누적)

**원인**: Acuity 6.12가 3D ONNX 텐서 `[1, 32, 576]`을 내부 4D `[1, 1, 32, 576]`으로 확장하면서 attribute `axes`는 renumber 안 함. RMSNorm의 `axes=[2]`가 hidden(576) 대신 seq_len(32) 축을 reduce.

증명: `acuity_mean == sq.mean(axis=1)` (max_abs_diff = 0.0).

**수정**: `patch_reducemean_axes.py` — 61개 ReduceMean 노드 `axes=[2]` → `axes=[-1]`.

결과: Acuity FP32 vs ORT FP32 → **32/32 argmax match, cos=1.0000** 완전 복구.

Commit: `f86a5c4`

---

## v0.1.5 — 2026-07-16 오전 (M1 initial compile)

**첫 T527 NPU에서 LLM 컴파일 성공.**

- HuggingFace SmolLM2-135M-Instruct 다운로드 (257MB)
- petayyyy `make_real_llm_onnx.py --seq-len 32` → 514MB static ONNX
- `patch_onnx_last_slice.py`: Acuity converter의 `slice_last_hidden size=-30` 버그 우회
- pegasus import → 2454-layer Acuity IR (Error 0)
- uint8 asymmetric_affine 양자화 (Conformer 레시피)
- NBG export (`--optimize VIP9000NANOSI_PLUS_PID0X10000016`)
- **`network_binary.nb` 124MB 생성**
- 디바이스에서 forward pass 성공, 93ms = 10.9 tok/s pure NPU
- 정확도: 0/32 (Acuity 축 버그 미인지)

Commits: `70cf811`, `0a34cbc`, `1c3e69a`, `7d05aee`

---

## v0.1.0 — 2026-07-15 (M0 파이프라인 배관 검증)

**33개 dummy op × pegasus import 실측.**

- Acuity 6.12 op 지원 매트릭스 확립
- Native OK 30개: MatMul, Softmax, Sin, Gather, Erf, Sigmoid, Tanh, Pow, Sqrt, Reciprocal, Slice, Concat, Where, Split, Transpose, Reshape, ReduceMean, Add/Mul/Sub, CumSum, ScatterND/GatherND ...
- Native FAIL 3개, 전부 조립 우회 검증됨:
  - `Cos` → `Sin(π/2 − x)`
  - `LayerNormalization` (opset 17) → 분해 조립
  - `Gelu` (opset 20) → Erf/tanh 조합
- 모든 Transformer 필수 연산 커버 확인

Commits: `091282e`, `4322698`

---

## v0.0.1 — 2026-07-15 오전 (리서치 + 계획)

**T527 LLM 포팅 landscape 조사.**

- 5개 병렬 리서치 리포트 (Allwinner/Verisilicon/커뮤니티/레퍼런스 NPU/한국어 LLM 양자화)
- 핵심 발견: 공개된 T527 LLM 포팅 성공 사례 0건
- 참고 청사진: `petayyyy/a733_npu_driver` (A733에서 SmolLM2-135M 21 tok/s)
- `PLAN.md`: 4단계 검증 사다리 (M0 → M1 → M2 → M3)
- 첫 타겟 한국어 LLM: Qwen2.5-0.5B (Apache) → Midm-2.0-Mini (MIT, KT)

Commits: `d35378f`, `db24f9c`, `f3bed7a`

---

## 남은 작업 (M2 → M3)

- [ ] SmoothQuant + int16 device 결과 개선 (calibration range 여유, per-layer skip)
- [ ] Coherent 한국어 텍스트 생성 확인
- [ ] Multi-token sliding window Korean decode
- [ ] M3: Midm-2.0-Mini-Instruct (2.3B, MIT, KT 한국어 특화)

---

## 참조 규약

- 커밋 hash는 `git log` 순번대로
- 각 버전마다 관련 commit hash 명시
- 새 실측/발견마다 반드시 이 파일에 추가 후 커밋
- 대박 발견은 별도 `##` 소섹션 (예: axis bug)
