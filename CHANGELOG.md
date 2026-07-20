# CHANGELOG

T527 NPU LLM 포팅 프로젝트. 날짜/버전별 진행 기록.

---

## v0.9.0 — 2026-07-20 (Wide-fl 실측: 진짜 device drift 벽)

**W 축소 실험(W=8)로 sat 오히려 증가 (Acuity auto-fl 문제).**
**wide-fl 수동 patch (output fl=6, outlier layers fl-=1)로 W=32 saturation 4.7%까지 감소.**

- W=32 baseline int16 (fl=9): 19% sat
- W=16 int16 (fl=9): 11% sat (activation 절반)
- W=8 int16 (fl=10): 38% sat (Acuity가 auto로 좁은 fl 선택)
- **W=32 wide-fl (output=fl6 range ±128): 4.7% sat**

**하지만 top-5는 여전히 int16 max tied.** device NPU int16 산술이 FP32 대비 값 2배+ 부풀림. wider fl로도 못 커버. 진짜 device drift 물리적 한계.

`patch_quantize_headroom.py` (fl 조정 유틸) + inline python 스크립트로 outlier layer 자동 검출.

Commit: (this one)

---

## v0.8.5 — 2026-07-20 (W=16 실측: saturation 절반)

**Window 축소 실험: W=32 → W=16.**

- SmolLM2 SmoothQuant α=0.5 + int16 dfp, W=16 static ONNX (514MB) → NBG 264MB
- Device saturation: **19% → 11%** (activation size 절반 효과 확인)
- Argmax는 여전히 saturation tie-break, top-5 전부 +63.998 (fl=9 max)
- W=8 이하로 더 줄이면 saturation 계속 감소할지 실험 대상

Commit: (this one)

---

## v0.8.0 — 2026-07-20 (SmolLM2 SmoothQuant + int16 device: 첫 토큰 semantic 성공)

**M1 SmolLM2 SmoothQuant + int16 dfp NBG (267 MB) 디바이스 실측.**

- 파이프라인이 HW native path 사용: quantize dynamic_fixed_point int16 → NBG export 성공 (Qwen과 달리 fatal 64768 안 남 — SmolLM2가 작아서)
- Device output `data_format=5, dfp=9` (int16, scale = 1/512)
- **★ 첫 토큰 실제 정확 예측**:
  - `"def hello"` → argmax = **`(`** (token 24) — Python 문법 정확
  - `"class Point"` → argmax = `*`
- 하지만 19% saturation → multi-token feedback loop에서 급속 붕괴
- Host: match 25/32, top5 3.91/5 (Qwen 결과와 유사)

`wiki/results/smollm2-sq-int16-device.md` 추가.

다음: 
- `--hybrid` 모드 (per-layer precision 자동 선택)
- α 더 aggressive (0.8~1.0)로 activation outlier 완전 제거
- lm_head output만 별도 precision 유지

Commit: (this one)

---

## v0.7.5 — 2026-07-20 (Hardware reality check)

**결정적 재해석: T527 VIP9000-NanoSI-Plus는 INT8 HW only.**

- FP16 / bf16 / qbf16 / FP32 — NPU에 하드웨어 없음
- Acuity NBG export `Fatal 64768` = 컴파일러가 없는 HW를 위한 코드 emit 거부 (버그 아님, 올바른 동작)
- SmolLM2 FP32 NB (626MB, 7.28s/forward)는 CPU fallback SW emulation — 실사용 불가
- **실전 배포는 uint8 or int16 이외 선택지 없음**
- qbf16 host 30/32 결과는 정보용, 디바이스 이식 불가

`wiki/hardware/t527-vip9000.md` 정밀도 지원 테이블 추가.

다음 방향: `--hybrid` 모드 or per-layer manual quantize seed로 uint8+int16 mix.

Commit: (this one)

---

## v0.7.0 — 2026-07-20 (SmoothQuant + qbfloat16 near-FP32 host)

**★★★ SmoothQuant α=0.5 + qbfloat16 host inference로 30/32 argmax match, cos=0.9965 달성.**

- top-5 last-position 4/5 정확히 일치 (`[576, 4710, 1084, 758, ...]`)
- FP32 baseline과 사실상 동등한 품질을 quantize에서 회복
- 그러나 **NBG export 실패**: `Fatal model generation error: 64768`
- 원인: Qwen NBG 사이즈가 컴파일러 상한 초과 (bfloat16, qbfloat16, float32 전부 같은 에러)
- SmolLM2-135M FP32 NB는 성공했으므로 (626MB) 크기 문제 확실
- 다음 시도: chunked NB or W=16 (반절 context)로 사이즈 줄이기, or hybrid layer quant

새 wiki 페이지: `wiki/results/qwen-sq-qbf16-host.md`

Commit: (this one)

---

## v0.6.0 — 2026-07-20 (Karpathy LLM Wiki 패턴 도입)

**저장소 내 wiki 구조 세팅.**

- `wiki/schema.md`: 3-layer 아키텍처 (raw / wiki / schema) 규약
- `wiki/index.md`: 마스터 네비게이션
- `wiki/log.md`: append-only chronological
- `wiki/models/`: smollm2-135m.md, qwen25-05b.md
- `wiki/hardware/`: t527-vip9000.md
- `wiki/techniques/`: axis-fix-patch.md, last-slice-patch.md, smoothquant.md
- `wiki/issues/`: acuity-reducemean-axis.md, llm-outlier-saturation.md, host-vs-device-drift.md
- `wiki/results/`: qwen-sq-int16-host.md
- `raw/catalog.tsv`: 12개 raw source 등록

패턴: [Karpathy gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). 앞으로 매 발견마다 raw catalog + source note + 관련 wiki 페이지 + log.md + CHANGELOG.md 갱신 후 커밋.

Commit: (this one)

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
