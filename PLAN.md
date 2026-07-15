# T527 LLM Porting Plan

_작성일 2026-07-15 | 근거: `research_allwinner.md`, `research_verisilicon.md`, `research_community.md`, `research_reference_npus.md`, `research_korean_llm_quantization.md`_

---

## 0. 목표

Allwinner T527 (Vivante VIP9000-NanoSI-Plus, ~2 TOPS, PID `0X10000016`) NPU에 **한국어 LLM**을 올려 Android 앱에서 실시간(≥ 3 tok/s decode) 추론.

기존 Conformer/CitriNet 파이프라인 (`.nemo → ONNX → Acuity import → quantize uint8 → NB export → VIPLite runtime`) 을 최대한 재활용, 신규 LLM 특유 문제(KV-cache, auto-regressive decode, activation outlier)만 새로 해결.

---

## 1. 제약과 상수 (안 바뀌는 것)

| 항목 | 값 | 출처 |
|---|---|---|
| NPU TOPS | ~2 (NanoSI-Plus) | 벤더 문서 |
| 지원 양자화 | **W8A8 asymmetric_affine (uint8) only** | Acuity 6.12/6.21, W4/FP16 하드웨어 미지원 |
| 그래프 형태 | **Static shape NBG** (dynamic shape 불가) | Conformer 실증 |
| Optimize 플래그 | `VIP9000NANOSI_PLUS_PID0X10000016` | 필수, 없으면 NB 로드 실패 |
| 타겟 RAM | 1~2 GB (보드별) | T527 표준 |
| 타겟 CPU | Cortex-A55 quad @ 1.8 GHz | fallback 및 non-NPU op 처리 |
| 실행 환경 | Android + JNI (`awnn_lib.h`) | Conformer 검증 완료 |

---

## 2. 전략 요약

**"모델 검증 사다리"** — 작은 것부터 올려 파이프라인 배관을 안정화한 뒤 큰 것으로 스케일업.

```
M0. 환경 검증        (Acuity 6.12 op 지원 실측)
     ↓
M1. SmolLM2-135M     (파이프라인 배관, 아키텍처 검증 — a733_npu_driver 재현)
     ↓
M2. Qwen2.5-0.5B     (한국어 최소 응답, SmoothQuant 도입, KV-cache 전략 확정)
     ↓
M3. Midm-2.0-Mini    (한국어 실전 타겟, MIT 라이센스, 2.3B 상용 가능)
     또는
     Qwen2.5-1.5B    (Midm 실패 시 폴백, Apache)
```

각 단계 실패 시 이전 단계 자산은 그대로 유지 — 최악의 경우 M1 완료만으로도 "T527에 LLM 올림" 실증 확보.

---

## 3. 첫 타겟 한국어 LLM 후보 (Top-3)

| 순위 | 모델 | 크기 | 라이센스 | 근거 |
|---|---|---|---|---|
| 1 | **Qwen2.5-0.5B-Instruct** | 0.5B | Apache 2.0 | ONNX export 검증, GQA+tied embedding, RAM 여유, 다국어(한국어 포함) |
| 2 | **Midm-2.0-Mini-Instruct** (KT) | 2.3B | **MIT** | 한국어 특화 + 상업 배포 자유. RAM 상한 걸침 (uint8 후 ~1.15GB) |
| 3 | **Qwen2.5-1.5B / Qwen3-1.7B** | 1.5~1.7B | Apache 2.0 | Midm 실패 시 대체, 아키텍처 표준 |

**참고 (라이센스 리스크로 상용 배포 배제)**: kanana-nano-2.1b (CC-BY-NC, KMMLU 44.80), EXAONE-3.5-2.4B (EXAONE NC), SmolLM2-360M (한국어 학습 없음, 배관 검증용).

**배제**: KORMo-10B (RAM 초과), Trillion-1.4B (실존 미확인), Phi-3-mini (3.8B 초과), TinyLlama (MHA로 KV 오버헤드 큼).

---

## 4. 예상 성능 (실측 전 잠정치)

| 모델 | Decode tok/s (T527) | 근거 |
|---|---|---|
| SmolLM2-135M | 15~21 (KV-cache 없이) | petayyyy A733 실측 21 tok/s |
| Qwen2.5-0.5B | **5~10** | RK3588 TinyLlama 10-15 tok/s × (2/6) TOPS × (0.5/1.1) 파라미터 |
| Qwen2.5-1.5B / Qwen3-1.7B | 2~5 | 위 스케일링 |
| Midm-2.0-Mini (2.3B) | **1~3** | 위 스케일링, RAM 병목 가능 |

**주의**: TOPS 비례 외삽은 조잡. 실제로는 메모리 대역폭과 KV-cache 오버헤드가 지배 변수. M1 실측 후 재계산.

---

## 5. 마일스톤 상세

### M0. 환경 & Op 지원 실측 (0.5~1일)

- [ ] Acuity 6.12.0 bin에서 native op 지원 실측: 더미 ONNX에 `Sin`, `Cos`, `Gather_ND`, `ScatterND`, `LayerNorm`, `EmbeddingLookup`, `SoftMax`, `Pow`, `ReduceMean`, `Rsqrt`, `CumSum` 각각 단독 노드로 넣고 `pegasus import` 성공 여부 확인
- [ ] 위 op 중 unsupported 있으면 어느 것을 CPU 폴백/분해 조립해야 하는지 목록화
- [ ] **결정 게이트**: `Sin/Cos/RMSNorm 조립 불가`면 M1 방향 재검토 (SmolLM2 대신 아예 다른 아키텍처 후보 탐색)

**산출물**: `M0_op_support_matrix.md`

---

### M1. SmolLM2-135M 재현 (2~3일)

**목적**: petayyyy/a733_npu_driver의 A733 성공 사례를 T527용으로 재빌드. 파이프라인 배관 검증.

- [ ] `petayyyy/a733_npu_driver` 클론, 스크립트 구조 파악
- [ ] SmolLM2-135M ONNX export (HuggingFace `optimum-cli export onnx`)
- [ ] `fix_onnx_for_acuity.py` 를 LLM용으로 개조: dynamic `seq_len`/`past_key_values` shape → 상수 치환, window ≤64 고정 (petayyyy 방식)
- [ ] `inputmeta.yml` 다중 입력 대응 (`input_ids`, `attention_mask` 최소 2개, KV cache 도입 시 추가)
- [ ] `pegasus import → quantize (uint8, kl_divergence) → export ovxlib` 파이프라인 실행
  - Optimize 플래그: `VIP9000NANOSI_PLUS_PID0X10000016` (T527 전용 — A733의 `PID0X1000003B`로는 NB 로드 안 됨)
- [ ] `vpm_run`으로 디바이스 상 output 검증 (golden logits FP32 vs uint8 KL divergence)
- [ ] JNI/awnn 래퍼로 Android에서 1토큰 forward 성공

**산출물**: `smollm2_135m/network_binary.nb`, `vpm_run` 결과, Android 앱 최소 데모

**성공 기준**: 디바이스에서 15+ tok/s (KV-cache 없이 재계산 방식)

---

### M2. Qwen2.5-0.5B 한국어 파일럿 (5~7일)

**목적**: 한국어 최소 응답 확보. LLM 특유 문제 (SmoothQuant, KV-cache 관리) 실전 도입.

- [ ] Qwen2.5-0.5B-Instruct ONNX export (Optimum), GQA/tied embedding 구조 확인
- [ ] **SmoothQuant** 적용: ONNX 그래프에 activation outlier를 weight로 이전하는 rescale 노드 사전 삽입 (구현은 `smoothquant` 공식 repo 또는 `optimum-quanto` 참고)
- [ ] RMSNorm 분해 (`Pow` → `ReduceMean` → `Rsqrt` → `Multiply`) — M0에서 native op 지원 확인된 경우
- [ ] RoPE 분해 (`Sin/Cos` + `Multiply/Add` + `Gather` 조합)
- [ ] **Prefill NB**: 고정 시퀀스 길이 (예: 128 tokens) 1회 실행용
- [ ] **Decode NB**: 1 token 입력 + 고정 크기 KV-cache 슬롯 (예: max 256 tokens) 재실행용
- [ ] KV-cache 관리: host side 순환 버퍼 (petayyyy 방식) — 슬롯 다 차면 sliding window로 밀거나 재컴파일된 다른 길이 NB로 스위칭
- [ ] Calibration corpus: AIHub STT transcript 텍스트 + 모두의 말뭉치 혼합, 100~500 샘플
- [ ] KMMLU/HAERAE subset 20샘플 quick eval → FP32 vs uint8 정확도 손실 측정

**산출물**: `qwen2_5_0_5b/prefill.nb`, `qwen2_5_0_5b/decode.nb`, calibration set, quick-eval 결과

**성공 기준**: 
- 5+ tok/s decode
- KL divergence (FP32 vs uint8 logits) < 0.1 (첫 기준선)
- 한국어 짧은 문장 생성 시 반복/붕괴 없음

**결정 게이트**: SmoothQuant 없이 이미 KL < 0.1이면 M3에서도 생략 가능. 아니면 M3에서도 필수.

---

### M3. Midm-2.0-Mini (2.3B) 실전 (7~10일)

**목적**: 상용 배포 가능한 한국어 LLM 실전 배포.

- [ ] Midm-2.0-Mini-Instruct ONNX export
- [ ] RAM 실측: uint8 weight 크기 계산 (2.3B × 1byte ≈ 2.3GB) → **T527 1GB 보드는 불가**, 2GB 이상 보드 전용. 필요 시 layer sharding (host RAM ↔ NPU 스왑) 검토
- [ ] GQA head 수, RoPE base 등 아키텍처 세부 확인 (arXiv 기술보고서 원문 필요 — 리서치에서 미확인 항목)
- [ ] M2에서 확정한 SmoothQuant + prefill/decode 분리 파이프라인 그대로 적용
- [ ] 벤치마크: KMMLU subset, HAERAE, 실제 대화 샘플 사람 평가

**Fallback**: RAM 초과 또는 정확도 붕괴 시 → **Qwen2.5-1.5B / Qwen3-1.7B**로 대체 (Apache, 아키텍처 표준, 디버깅 자료 풍부)

**산출물**: Android 앱 배포용 번들 (nb + tokenizer.model + config.json)

**성공 기준**:
- 2+ tok/s decode (2.3B 기준)
- 한국어 대화 3턴 붕괴 없음
- KMMLU quick subset FP32 대비 정확도 손실 < 5%p

---

## 6. 파이프라인 자산: 재활용 vs 신규

| 요소 | Conformer 자산 | LLM 상태 |
|---|---|---|
| Acuity import/quantize/export 순서 | 확립 | 재활용 |
| `--quantizer asymmetric_affine --qtype uint8 --algorithm kl_divergence` | 확립 | 재활용 (baseline) |
| Optimize 플래그 | 확립 | 재활용 |
| Docker 환경변수 (LD_LIBRARY_PATH, VivanteIDE 세팅) | 확립 | 재활용 |
| SentencePiece tokenizer 통합 (JNI) | 확립 | 재활용 (Qwen/Midm 대부분 SentencePiece) |
| `inputmeta.yml` 다중 입력 매핑 | 단일 입력만 경험 | **신규** — `input_ids`, `attention_mask`, KV cache tensors |
| Static shape 고정 | 301 프레임 확립 | **개조 필요** — LLM은 `seq_len`, `past_len` 등 여러 축 |
| Auto-regressive decode 루프 | 없음 | **신규** — prefill/decode 분리 NB 2종 |
| KV-cache 관리 | 없음 | **신규** — host-side 순환 버퍼 |
| SmoothQuant | 없음 | **신규** — ONNX 사전 rescale |
| RMSNorm/RoPE 분해 | 없음 | **신규** — native op 조합 |
| lm_head 배치 (NPU vs CPU) | 없음 | **신규** — 대형 matmul, vocab×hidden ≈ 150k×2048 → CPU 오프로드 vs NPU 유지 실험 필요 |
| 검증 지표 | CER | **신규** — KL divergence, perplexity, Top-1 토큰 일치율 |
| Calibration | 오디오 10~500 샘플 | **신규** — 텍스트 코퍼스, 다양한 토큰 분포 우선 |
| QAT | 검증 완료 (Conformer CER 개선) | **후순위** — PTQ 실패 시에만 |

---

## 7. 리스크 및 대응

| 리스크 | 확률 | 영향 | 대응 |
|---|---|---|---|
| Acuity 6.12에 Sin/Cos/RMSNorm 필수 op 미지원 | 중 | 치명 | M0에서 실측 우선. 실패 시 6.21 업그레이드 or CPU 폴백 재설계 |
| ONNX static shape 변환 실패 (dynamic 잔존) | 중 | 치명 | petayyyy 스크립트 이식, dummy input shape 강제 |
| Activation outlier로 uint8 양자화 정확도 붕괴 | 높음 | 큼 | SmoothQuant M2에서 도입, 실패 시 layer-wise mixed (특정 레이어만 int16 시도, Acuity가 지원할 경우) |
| KV-cache RAM 초과 | 중 | 큼 | max context 256~512로 제한, sliding window, KV int8 저장 |
| Midm-2.0-Mini 2.3B가 1GB 보드에서 안 뜸 | 높음 | 중 | 2GB+ 보드 요구 or Qwen2.5-1.5B로 대체 |
| Decode tok/s < 1 (사용 불가 수준) | 중 | 큼 | lm_head CPU 오프로드, prefill/decode NB 최적화, M1 실측 후 재검토 |
| lm_head가 병목 (vocab_size × hidden = 3억 params) | 높음 | 큼 | Vocab pruning (자주 안 나오는 토큰 제거), tied embedding 확인 |
| Vivante 커뮤니티/공식 지원 없음 → 디버깅 고립 | 확정 | 지속적 | petayyyy 리포 이슈 트래킹, onnxruntime #28244 모니터링, 자체 디버깅 문서화 |

---

## 8. 검증 체크리스트 (각 마일스톤 공통)

1. `pegasus import` 성공 (unsupported op 없음)
2. `pegasus quantize` 성공 (calibration corpus로 KL divergence 수렴)
3. `pegasus inference` FP32 vs uint8 output 비교 (KL < 0.1 목표)
4. `pegasus export` NB 파일 생성 (`network_binary.nb`)
5. `vpm_run` 디바이스 상 실행 (output 일치 확인)
6. JNI/awnn 래퍼로 Android에서 forward 성공
7. Golden logits 대비 Top-1 토큰 일치율 측정
8. 한국어 sample prompt 3개 실제 생성 확인 (반복/붕괴 여부 육안 판단)
9. tok/s 실측 (100 토큰 생성 시간 평균)
10. RAM/전력 실측 (`adb shell dumpsys meminfo`)

---

## 9. 미확인 사항 (선행 조사 필요)

- Acuity 6.12.0에서 Sin/Cos/RMSNorm 구성 op 실제 지원 → **M0 실측으로 즉시 해소**
- Midm-2.0-Mini / kanana-nano의 GQA head 수, RoPE base, 레이어 수 → arXiv 기술보고서 원문 확인
- Qwen2.5-0.5B KMMLU 정확한 수치 → 공식 벤치 리더보드 재확인
- SmoothQuant를 Acuity 파이프라인에 삽입한 사례 → 오픈소스에 없음, **자체 구현 필요 (M2 신규 작업)**
- lm_head 크기 실측: Qwen2.5-0.5B / Midm-2.0-Mini의 vocab_size × hidden_dim
- T527 보드 RAM 실측 (개발 보드 스펙 확인)
- AIHub STT transcript 텍스트를 LLM calibration corpus로 재활용 가능한지 (도메인 편향)

---

## 10. 참고 자료 (핵심만)

**직접 참고할 리포/문서:**
- `petayyyy/a733_npu_driver` — SmolLM2 실증 코드, KV-cache 없는 재계산 방식
- `waz664/vip9000-embeddinggemma` — ONNX op 치환 트릭
- `VeriSilicon/acuity-models` — Qwen/Llama JSON 토폴로지 (weights 없음)
- Allwinner PDF: `docs/acuity_toolkit/NPU模块开发指南/NPU_算子支持列表.pdf` — Acuity 6.21 op 지원 실체

**우리 팀 기존 자산:**
- `/home/nsbb/travail/claude/T527/t527-stt/conformer/docs/NEMO_TO_NB_CONVERSION_GUIDE.md`
- `/home/nsbb/travail/claude/T527/t527-stt/docs/260325_t527_npu_stt_quantization_comprehensive_analysis.md`
- `/home/nsbb/travail/claude/T527/t527-stt/docs/260326_quantization_deep_dive_discussion.md`
- `/home/nsbb/travail/claude/T527/t527-stt/docs/260326_quantization_expert_roadmap.md`

**리서치 리포트 (본 계획의 근거):**
- `research_allwinner.md` — Allwinner/보드벤더 상황
- `research_verisilicon.md` — Acuity/TIM-VX op 지원
- `research_community.md` — 실제 포팅 사례 및 실패
- `research_reference_npus.md` — RK3588 RKLLM 청사진, 타겟 모델
- `research_korean_llm_quantization.md` — 양자화 심화, 한국어 LLM 후보

---

## 11. 즉시 다음 액션 (M0 시작)

```bash
# 1. 작업 디렉토리 준비
cd /home/nsbb/travail/claude/T527/t527-llm
mkdir -p M0_op_validation

# 2. 더미 ONNX 스크립트 작성 (각 op 개별 노드 포함)
# → 목표: Sin/Cos/Gather_ND/ScatterND/LayerNorm/EmbeddingLookup 등 개별 import 시도

# 3. Docker + Acuity 6.12로 pegasus import 배치 실행
# → 성공/실패 매트릭스 → M0_op_support_matrix.md 작성

# 4. petayyyy/a733_npu_driver 클론 & 코드 리뷰
git clone https://github.com/petayyyy/a733_npu_driver

# 5. SmolLM2-135M ONNX export 준비
```
