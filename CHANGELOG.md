# CHANGELOG

T527 NPU LLM 포팅 프로젝트. 날짜/버전별 진행 기록.

---

## v0.12.5 — 2026-07-22 (SmolLM2-360M 실측: 예상대로 열등)

**바로 큰 모델(360M) → 더 나쁜 결과 확인.**

- 360M SmoothQuant + uint8 hidden NB (252MB) export 성공 (fatal 아님, Qwen-0.5B와 크기 비슷하지만 vocab 3배 작아서)
- Device 실측 cos: **0.05~0.38** (135M은 0.33~0.65)
- 여러 프롬프트에서 top-5 tokens 동일 (`staking`, `graduates`, `queous`, `tees`, `uator`) — quantization drift가 입력 신호 압도

**135M이 T527 int8 NPU sweet spot 확정**:
- 파라미터 적음 → 32→30 layers × narrower hidden → drift accumulation 적음
- 135M: 실제 semantic tokens 나옴 (cos 0.5+)
- 360M: 상수 tokens (bias-dominated)
- 500M+ Qwen: 유사 문제

배포용 권장: **SmolLM2-135M-Instruct + SmoothQuant + uint8 hidden + host lm_head**.

Commit: (this one)

---

## v0.13.5 — 2026-07-24 (Repetition penalty + first-token benchmark)

**Multi-token variety + reproducible benchmark harness.**

- `generate_hidden_reppen.py` (both SmolLM2 and Qwen): top-k + rep-penalty
- SmolLM2 rep-penalty output: more varied tokens (` (`, ` \n`, ` de`, ` spot`) but still noise
- Qwen Korean rep-penalty: 첫 토큰 여전히 `\n` 지배
- `benchmark_bias.py`: 12 held-out prompts, measures top-1/top-5 vs FP32
- **Result**: 135M raw 0/12, +bias 1/12, +bias top-5 2/12
- 135M is genuinely too small to make bench meaningful — most FP32 top-1s are `\n`

Conclusion: bias correction is a real signal (Qwen cos 0.06→0.88, SmolLM2 cos 0.33→0.58) 
but 135M SmolLM2 is too small for benchmark validation. Move to larger models (360M, Qwen 500M).

Commit: (this one)

---

## v0.13.0 — 2026-07-24 (★★ Qwen Korean bias correction: cos 0.06 → 0.88)

**Big Korean LLM breakthrough on T527.**

- SmolLM2 100-prompt bias measurement (`compute_bias.py`): confirmed |max|=6.5 content bias
- **Qwen with 15 KR + 5 EN calibration bias**:
  - `한국의 수도는` cos 0.06 → **0.88**
  - `안녕하세요, 저는` cos 0.03 → **0.82**
  - `봄이 오면 벚꽃이` cos 0.04 → **0.85**
  - Top-3 after correction: `\n`, `\xa0`, `,` (whitespace/punctuation semantically OK)
- Bias vector is Korean-specific — English prompts hurt (0.31 → 0.19 for France)
- Multi-token still limited (window shift moves distribution)

Files:
- `M1_smollm2/compute_bias.py` — bias measurement tool
- `M1_smollm2/device_bias_c100_{all,content,pos}.npy`
- `M2_qwen/device_bias_{all,content}.npy` — Korean bias
- `M2_qwen/device/generate_hidden_biascorr.py`
- `wiki/techniques/bias-correction.md`

**Effective M2 completion**: Korean-capable LLM produces plausible first-token predictions on T527 NPU.

Commit: (this one)

---

## v0.12.0 — 2026-07-22 (100-sample calibration: cos 0.45 → 0.65)

**SmolLM2 hidden uint8 재양자화 with 100 diverse English prompts** (facts × code × narrative × QA × math × chat 각 20~25).

- Cos vs ORT hidden: prompts별 0.33~0.65 (평균 ~0.50, 이전 10-sample은 0.45 평균)
- 실제 semantic 정확도 개선:
  - `"1 + 1 ="` top-5: `1`, **`2`**, `3`, `'s`, `0` — 답 `2`가 top-2에 위치
  - `"Once upon a time"` top-5: 포함 **`Bob`** (name plausible)
  - `"def hello"` top-5: 포함 `\n` (Python 다음 줄 plausible)
- 그러나 multi-token generation은 여전히 feedback loop로 붕괴 → 반복 `1`
- **결론**: 단일 토큰 예측 품질은 계속 개선 가능, coherent generation은 cos > 0.9 필요 (FP32-level)

동시에 SmolLM2-360M safetensors 다운로드 중 (720MB, HF 느림 ~5MB/min).

Commit: (this one)

---

## v0.12.0 — 2026-07-24 (★ Per-channel bias correction: first-token semantic 회복)

**진단**: NPU device output hidden state에 **per-channel systematic bias** 존재.
- Channel 507: -15.1 offset (극단)
- 51개 채널 |bias| > 1
- 3개 채널 |bias| > 3

**해결**: 20개 캘리브 프롬프트로 device_hidden - ort_hidden 평균을 채널별로 계산 → subtract at inference.

**결과 (single-position argmax)**:
- 'def hello' cos 0.33 → **0.57**, top-5: `\n` `(` ` ` `and` `the` (Python 문법 정확)
- 'The capital of France is Paris.' cos 0.45 → 0.55, argmax `,` (period → comma OK)
- 'Once upon a time' cos 0.65 → 0.70

**한계**: Multi-token greedy는 여전히 반복적 — window sliding으로 activation distribution이 calib과 벗어남. 다음 방향: recursive calibration (매 스텝 window에서 bias 재계산).

`device/generate_hidden_biascorr.py` 추가.

Commit: (this one)

---

## v0.11.0 — 2026-07-20 (Gamma scaling: 표면 개선 있으나 근본 해결 아님)

**최종 실험: `final_rms_gamma /= 8` 로 hidden range 축소 → host에서 K=8 복원.**

- ONNX 재빌드: gamma 8배 축소 → hidden max 214→27 (uint8 scale 1.56→0.19)
- Device std 1.20 (ORT 1.34에 근접), 하지만 **cos 동일** (0.31 for France)

교훈: NPU int8 산술의 시스템적 drift는 quantization scale과 무관. Numerical한 문제이지 표현력의 문제 아님.

**최종 상태 (M2 partial win)**:
- SmolLM2 hidden uint8: cos 0.45, English tokens
- Qwen hidden uint8: cos 0.006-0.46 (varies), English tokens for math/simple, Korean weak

M2/M3 completion 위한 남은 방향:
1. Chunked decoder (per-layer NB, host FP32 in between)
2. QAT (quantization-aware training) — 인프라 큰 투자
3. Newer Acuity 버전 (6.21+) 접근 가능성 조사
4. Alternative NPU (RK3588 rknn-llm 결과와 비교)

Commit: (this one)

---

## v0.10.7 — 2026-07-20 (Hidden scale sharpening 실험: 실패)

**Qwen hidden output uint8 scale=1.56 → 더 fine하게 만들려는 시도.**

1. patch_hidden_scale.py로 .quantize의 max_value 수동 축소 → Acuity export가 .data에서 원본 scale 재사용, 무시됨
2. --hybrid re-quantize → calibration이 range 재계산, 원상복구
3. **ONNX에 Clip[-50, 50] 노드 삽입** → 성공적으로 scale=0.39 얻음 (4배 fine)
4. 그러나 device 실측 cos **오히려 감소** (0.31→0.16 for France) — clip이 tail 정보 손실 유발

교훈: SmoothQuant가 outlier magnitude를 hidden state로 이전한 결과, 이 값들이 실제 정보를 담고 있음. Clip으로 잘라내면 오히려 hurt.

M2 남은 방향:
- SmoothQuant α=0.3 (덜 aggressive → hidden에 덜 이전)
- Chunked decoder (레이어별 별도 NB로 drift 격리)
- 캘리브레이션 100+ Korean-heavy prompts

Commit: (this one)

---

## v0.10.5 — 2026-07-20 (Qwen hidden 실측: 크기 스케일링 문제)

**Qwen SmoothQuant + uint8 hidden NB 실측 결과.**

- SmolLM2 hidden trick 그대로 이식 → 349MB NB export 성공
- Device 실측 cos: English 0.31~0.46, Korean 0.006~0.14 (variance 큼)
- 원인: **SmoothQuant가 final RMSNorm gamma로 outlier magnitude 흡수 → hidden range ±214 → uint8 scale 1.56** (SmolLM2 scale 0.20의 8배)
- Qwen 24 layers × wider FFN → NPU 누적 drift도 더 큼

Wiki: `wiki/results/qwen-hidden-device.md`

교훈: 파이프라인은 작동하지만 500M+ 모델은 quantization + drift 이중고. Midm-2.0 (2B) 그대로 시도하면 더 나빠질 것.

Commit: (this one)

---

## v0.10.0 — 2026-07-20 (★★ CPU-side lm_head + uint8 hidden: real English tokens)

**결정적 진전**: LM head를 NPU 그래프에서 잘라내고 hidden state만 device에서 계산, 최종 MatMul은 host CPU에서 FP32로.

- `patch_output_hidden.py`: ONNX에서 final_rms_out 노출, lm_head 제거
- SmolLM2 hidden NB uint8: **104MB, saturation 0.05%**
- 결정적 발견: **int16 hidden은 std 8배 부풀림** (accumulator bias), **uint8은 zero_point가 bias 흡수** → cos 0.45 (vs int16의 -0.17)
- Multi-token 생성 결과: 실제 영어 tokens `,` `.` ` and` `\n` ` only` `'s` 등 (더 이상 Rothschild-tier 랜덤 아님)
- Speed: 1.68 tok/s end-to-end (adb 포함, pure NPU ~100ms 훨씬 작은 NB)

**진짜 배포 recipe 확정**: SmoothQuant + uint8 asymmetric_affine + hidden-output NB + host lm_head.

Wiki: `wiki/techniques/cpu-lm-head.md`

Commit: (this one)

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
