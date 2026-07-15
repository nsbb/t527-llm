# 웹 리서치: T527(Vivante VIP9000 NPU)에 한국어 LLM 올리기 — 양자화 전략 중심

_조사일: 2026-07-15 | Facets: 6 | 참고 소스: 45+_

---

## Executive Summary

T527 NPU(Acuity 6.12/6.21, UINT8/INT16 강제, FP16 SW 에뮬레이션, mixed precision 사실상 불가)에 한국어 LLM을 얹는 프로젝트는 **"안 되는 이유"보다 "어떻게 쪼개서 되게 만드는가"의 엔지니어링 문제**다. 핵심 발견 3가지:

1. **Allwinner 자체 Acuity 6.21 op 지원 리스트(로컬 PDF, 2024-02-04)를 직접 확인한 결과, LLM에 필요한 핵심 연산(Softmax, LayerNorm, Sin/Cos, Pow, MatMul, Gather/GatherND, ScatterND, EmbeddingLookup, Erf, GELU/Swish, ReduceMean, CumSum)이 이미 native op로 지원된다.** 기존에 알려진 "TIM-VX(오픈소스)에는 RMSNorm/RoPE 1급 지원 없음"이라는 결론은 **오픈소스 TIM-VX 한정 이야기**이고, Allwinner가 실제로 배포하는 Acuity 6.21 op 테이블은 그보다 넓다. RMSNorm은 (Square→ReduceMean→Rsqrt→Multiply) 조합으로, RoPE는 (Sin/Cos + Multiply/Add + Gather) 조합으로 **분해 조립이 충분히 가능**하다는 근거가 확인됐다. (단, 우리가 실제 쓰는 Acuity 6.12.0 binary에서 동일하게 지원되는지는 **확인 필요** — 이 PDF는 6.21.x 기준.)
2. **RK3588(RKLLM)도 T527과 마찬가지로 W8A8-only(FP16 mixed 없음)에서 TinyLlama-1.1B 10-15 tok/s를 냄** — T527(2 TOPS)은 RK3588(6 TOPS)의 1/3 성능이므로, 같은 클래스 모델에서 **decode 3-5 tok/s 선이 현실적 상한**으로 추정된다.
3. **Static-shape NPU에서 autoregressive decode를 돌리는 표준 패턴은 이미 학계에 존재**(NITRO/Intel NPU, Quant.npu, KV-RM 등 2025-2026 논문) — "step-specific 컴파일 그래프를 여러 개 만들어 KV-cache 길이별로 스위칭"하는 방식이 사실상 컨센서스. Conformer의 static-shape NBG 경험이 그대로 재활용 가능한 지점이다.

**첫 타겟 추천**: 라이선스+아키텍처+양자화 친화성을 종합할 때 **Midm-2.0-Mini-Instruct(2.3B, MIT 라이선스, KT)** 또는 **Qwen2.5-0.5B/1.5B-Instruct(Apache-2.0, 다국어지만 한국어 포함)** 를 1차 후보로, **kanana-nano-2.1b(카카오, CC-BY-NC — 상업화 시 라이선스 문제)** 를 참고 벤치마크용으로 두는 것을 권고한다. 검증 마일스톤은 **SmolLM2-135M(아키텍처 검증) → Qwen2.5-0.5B(한국어+영어 균형, Apache) → Midm-2.0-Mini 또는 Qwen2.5-1.5B(실전 타겟)** 순.

---

## Detailed Findings

### Facet 1: 한국어 LLM 후보 & 양자화 친화성

#### 1.1 후보 비교표

| 모델 | 파라미터 | 라이선스 | 구조 특징 | 한국어 벤치 | 비고 |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B/1.5B-Instruct** | 0.5B/1.5B | **Apache-2.0** | GQA(14Q/2KV heads @0.5B), tied embedding, RoPE θ=10^6, SwiGLU, RMSNorm | KMMLU 구체 수치 미확인(확인 필요), 다국어 18T 토큰에 한국어 포함 | ONNX 커뮤니티 export 존재(`onnx-community/*`), Optimum 지원 |
| **Qwen3-0.6B/1.7B** | 0.6B/1.7B | **Apache-2.0** | tied embedding, q/k에 RMSNorm 적용(QK-Norm), GQA, RoPE θ=10^6 | 미확인 | Optimum 마스터 브랜치에서만 export 가능(2026-07 기준 stable 미지원) — [Qwen3-Embedding ONNX 이슈](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/discussions/18) |
| **Midm-2.0-Mini-Instruct (KT)** | **2.3B** | **MIT** (매우 permissive) | Base 모델 pruning+distillation | 미확인 (Base 모델은 KMMLU 계열 벤치 다수 사용) | [K-intelligence/Midm-2.0-Mini-Instruct](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct), GGUF 변형 다수 존재(mykor, Mungert) → 커뮤니티 검증 활발 |
| **kanana-nano-2.1b (Kakao)** | 2.1B | **CC-BY-NC-4.0** (비상업) | 미공개(HF 카드에 레이어수 등 명시 안 됨, 확인 필요) | **KMMLU 44.80, HAE-RAE 77.09**, KoMT-Bench 5.857 / MMLU 54.83, HumanEval 31.10 | [kakaocorp/kanana-nano-2.1b-instruct](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct) — 한국어 성능 강함, 상업 배포 시 라이선스 협의 필요 |
| **EXAONE-3.5-2.4B (LG)** | 2.4B | **EXAONE AI Model License 1.1-NC (비상업)** | 소형기기 배포 최적화 명시 | 문서상 벤치 다수 보유(확인 필요) | [LG-AI-EXAONE/EXAONE-3.5](https://github.com/LG-AI-EXAONE/EXAONE-3.5) — 라이선스가 상업 배포의 걸림돌 |
| **Trillion-1.4B** | — | — | **검색으로 실물 확인 안 됨.** Trillion Labs는 Tri-0.5B/1.9B/7B, Trillion-7B-preview 라인업만 확인됨(1.4B 모델명 없음) | — | **확인 필요/존재 불명확** — Tri-1.9B가 근접 사이즈 |
| **Gemma-3-1B** | 1B | Gemma 라이선스(준-오픈) | 140+ 언어 지원, 한중일 인코딩 개선됨(Gemma3 특징) | GSM8K/HumanEval서 Llama-3.2-1B보다 우위(영어 기준) | 한국어 전용 벤치 비교 데이터 부족 |
| **Llama-3.2-1B** | 1B | Llama 커뮤니티 라이선스 | 범용 지식 벤치서 강세, 수학/코드 약함 | 한국어 지원은 공식 목록 외(비공식 지원) | |
| **SmolLM2-135M/360M** | 135M/360M | Apache-2.0 | GSM8K 48.2%, IFEval 56.7%(135M) | 한국어 학습 데이터 사실상 없음 → **1차 아키텍처 검증용으로만 적합** | [HuggingFaceTB/SmolLM2-135M](https://skywork.ai/blog/models/huggingfacetb-smollm2-135m-free-chat-online-skywork-ai/), 이미 A733/T527 계열 NPU에서 petayyyy가 21/8.4 tok/s 확인(기존 리서치) |
| **KORMo-10B** | 10.8B | 완전 오픈(모델+데이터+코드) | 68.74% 한국어 synthetic data | 다국어 baseline과 동등 성능 주장 | **크기 초과** — T527 RAM 제약(1-2GB) 밖. 참고용 |

출처: [Navigating Korean LLM Research #1: Models](https://huggingface.co/blog/amphora/navigating-ko-llm-research-1), [Mi:dm 2.0 Korea-centric Bilingual LM](https://arxiv.org/pdf/2601.09066), [KORMo arXiv](https://arxiv.org/pdf/2510.09426), [Kanana GitHub](https://github.com/kakao/kanana)

#### 1.2 라이선스가 실질적 1차 필터다

- **상업/자유 배포 가능**: Qwen2.5/Qwen3(Apache-2.0), Midm-2.0-Mini(MIT), SmolLM2(Apache-2.0)
- **비상업 전용**: kanana-nano(CC-BY-NC-4.0), EXAONE-3.5(EXAONE license 1.1-NC)
- T527 프로젝트가 상업 제품(SDK)에 들어간다면 kanana/EXAONE은 **법무 검토 없이는 배제**. 사내 R&D/PoC 단계라면 성능 벤치마크 참고용으로는 계속 유용.

#### 1.3 아키텍처 관점 랭킹 근거

- **GQA(Grouped Query Attention)**: Qwen2.5/Qwen3 전부 채택 — KV-cache 크기 축소되어 NPU RAM 제약에 유리. Kanana/Midm/EXAONE의 GQA 여부는 **확인 필요**(HF 카드에 명시 안 됨).
- **Tied embedding**: Qwen2.5-0.5B, Qwen3-0.6B 모두 tie — lm_head와 embedding 공유 시 weight 크기 절감(양자화 시 embedding outlier 문제를 lm_head까지 전파시킬 위험도 있음, 주의).
- **QK-Norm(Qwen3)**: attention q/k에 RMSNorm 추가 — activation range 통제에 유리할 수 있으나 op 하나 더 분해해야 함.

---

### Facet 2: LLM 양자화 기법 심화

#### 2.1 PTQ 기법 비교

| 기법 | 원리 | Calibration 필요 | NPU(T527류 W8A8-only) 적용성 |
|---|---|---|---|
| **SmoothQuant** | activation outlier를 weight로 이전 (`s_j = max|X_j|^α / max|W_j|^(1-α)`) | 필요(activation 통계) | **W8A8에 최적화된 원조 기법** — T527처럼 activation도 강제 int8인 환경에 정확히 맞음. ONNX 그래프에 diag(s) 곱셈을 미리 삽입하는 방식으로 Acuity import 이전 단계에서 적용 가능 |
| **AWQ** | activation 크기 기준으로 중요 weight 채널만 정밀 보존 | 필요(activation 통계, 소량) | 기본적으로 **weight-only(W4A16)** 기법 — activation까지 int8인 T527에는 그대로 안 맞음. 다만 "salient channel 보호" 아이디어는 SmoothQuant와 결합 가능 |
| **GPTQ** | 레이어별 양자화 오차를 Hessian 기반으로 다른 weight에 보상 | 필요(calibration set 128~) | Weight-only(W4/W3) 위주. T527엔 weight만 좋아져도 activation 병목은 안 풀림 |
| **HQQ** | calibration-free, half-quadratic splitting으로 optimize | **불필요** | 빠르지만 weight-only. Calibration 없이 빠른 1차 스크리닝용으로 유용(어떤 모델이 양자화에 강건한지 빠르게 훑을 때) |
| **LLM.int8()** | outlier 채널만 FP16 분리, 나머지 int8 | 불필요(런타임 탐지) | **NPU에서 재현 불가** — mixed precision runtime dispatch는 T527에 없음(우리 팀 로드맵 문서의 결론과 일치: "Split Model 시도 결과 encoder가 이미 uint8에서 망가진 feature 출력") |

Ascend NPU PTQ 비교 논문([2602.17693](https://arxiv.org/pdf/2602.17693))의 핵심 결론: **W8A8이 W4A16보다 NPU에서 수치적으로 더 안정적**이다 — "aggressive 4-bit weight-activation 조합은 NPU 커널에서 layer-wise calibration instability를 일으켜 long-context reasoning에서 붕괴"라고 명시. 이는 T527이 UINT8 W8A8만 지원하는 게 오히려 "억지로 낮은 정밀도를 강요당하는 것"이 아니라 **이미 업계에서 검증된 안정적인 조합**이라는 재해석을 가능케 한다. ([A Case Study of Selected PTQ Baselines for Reasoning LLMs on Ascend NPU](https://arxiv.org/pdf/2602.17693))

#### 2.2 QAT for LLM — 실전 사용 현황

- LLM-QAT, EfficientQAT, PEQA는 논문 단계에서는 활발하나(예: [LLM-QAT arXiv:2305.17888](https://arxiv.org/pdf/2305.17888)), 우리 프로젝트처럼 **작은 팀이 자체 GPU로 QAT를 직접 돌리는 사례**는 Conformer 규모(122M)에 비해 LLM(0.5B~2B+)에서는 학습 비용이 훨씬 큼. Conformer QAT 성공 경험(CER 10.02%→9.59%~ 개선)을 그대로 재현하려면 **fine-tuning 데이터셋 + GPU 시간**이 상당히 더 필요.
- 현실적 절충안: **PTQ + SmoothQuant 우선 시도 → 정확도 부족 시에만 QAT 투입**(Conformer 때와 동일한 순서).

#### 2.3 W8A8 vs W4A16 vs W8A16 — NPU별 실제 speedup

| 조합 | 의미 | T527에서 가능한가 |
|---|---|---|
| W8A8 | weight int8 + activation int8 | **유일한 네이티브 옵션** (2 TOPS MAC array가 uint8×uint8 전제) |
| W4A16 | weight 4bit + activation fp16 | **불가** — Acuity 6.12/6.21이 4bit weight 자체를 지원 안 함(이전 리서치 결론 재확인), FP16은 SW 에뮬 라서 25-42배 느림 |
| W8A16 | weight int8 + activation fp16 | **불가능에 가까움** — activation을 fp16으로 유지하려면 FP16 HW 가속이 필요한데 T527엔 없음. RK3588은 FP16 HW가 있어 이 조합이 실제로 유의미하지만 T527은 아님 |

→ **결론: T527에서 LLM은 W8A8 asymmetric_affine(uint8) 이외의 선택지가 사실상 없다.** Conformer 파이프라인의 `--quantizer asymmetric_affine --qtype uint8`을 그대로 계승.

#### 2.4 Activation Outlier 문제와 우회

- Transformer activation outlier는 우리 팀이 이미 wav2vec2 한국어 실험에서 "activation range가 영어보다 5~50배 넓다"는 것으로 간접 경험함. LLM에서는 이 현상이 **embedding layer와 특정 attention head(outlier feature dimension)** 에서 훨씬 심하다는 것이 SmoothQuant 논문의 핵심 동기.
- NPU가 dynamic quant(런타임마다 min/max 재계산)를 지원하지 않는 이상(T527도 static calibration만 지원), **SmoothQuant를 ONNX export 직후 적용**하는 것이 유일한 현실적 대응책. 우리 팀의 "Acuity `.quantize` YAML 직접 생성" 인사이트(`260326_quantization_deep_dive_discussion.md`)와 결합하면: **ONNX 레벨 SmoothQuant(weight/activation rescale) → Acuity import → 커스텀 .quantize 생성(레이어별 sensitivity 반영) → Acuity export**라는 파이프라인이 그려진다.

#### 2.5 RoPE / RMSNorm / Softmax 양자화 — NPU 실행 방안 (신규 핵심 발견)

로컬 문서 확인 결과 (`/home/nsbb/travail/claude/T527/docs/acuity_toolkit/NPU모듈개발지침/NPU_算子支持列表.pdf`, v1.1, 2024-02-04, 대상: V85X/R853/MR527/**T527**/AI985, 기반: **Acuity 6.21.x / IDE 5.8.2**):

| 필요 연산 | Acuity native op 지원 여부 | 지원 dtype |
|---|---|---|
| Softmax | **지원** (`softmax`) | i8/u8/i16/fp32/fp16/bf16 |
| Log-Softmax | **지원** (`log_softmax`) | 동일 |
| LayerNorm | **지원** (`layer_norm`) | 동일 |
| RMSNorm | **명시적 op 없음** — 단, `pow`(제곱), `reduce`/`moments`(mean), `rsqrt`, `multiply`가 모두 지원되므로 **분해 조립 가능** | 각 구성 op 대부분 int8 포함 |
| Sin/Cos (RoPE) | **지원** (`sin`, `cos`) | i8/u8/i16/fp32/fp16/bf16 |
| Gather/GatherND/GatherElements | **지원** (RoPE 위치 인덱싱, KV-cache 슬라이싱용) | i8/u8/i16/fp32/fp16/bf16 |
| ScatterND/ScatterElements | **지원** (KV-cache in-place 업데이트용) | 동일 |
| EmbeddingLookup | **지원** | i8/u8/i16/fp32/fp16/bf16 |
| MatMul/BatchMatMul | **지원** (`matrixmul`, TF 매핑 `tf.batch_matmul`) | i8/u8/i16/fp32/fp16/bf16 |
| Erf/GELU/Swish/Mish | **지원** | i8/u8/i16/fp16(TP엔진) |
| CumSum | **지원** | i8/u8/i16/fp32/fp16/bf16 |
| SequenceMask (causal mask용) | **지원** | i8/u8/i16/fp32/fp16/bf16 |

**중요 재해석**: 기존에 확보된 리서치(research_verisilicon.md 등)에서 "TIM-VX에 RMSNorm/RoPE/MHA/GQA/KV-cache primitive 1급 지원 없음, Cos op 미등재"라고 정리했던 것은 **오픈소스 TIM-VX 리포지토리** 기준이었다(본 조사에서 TIM-VX의 `docs/Operators.md`를 재확인해도 LayerNorm/RMSNorm은 명시 안 됨, 그러나 Softmax/Sin/Cos/Gather/MatMul은 지원 확인됨). 반면 **Allwinner가 실제로 배포하는 Acuity 6.21 벤더 문서에는 Sin/Cos/LayerNorm/MatMul/Gather/Scatter가 전부 native op로 등재**되어 있다. 즉 **TIM-VX(공개 SDK 레이어)보다 Acuity(비공개 컴파일러, 우리가 실제 쓰는 것)의 op 커버리지가 더 넓다** — 이는 우리 프로젝트에 유리한 오차 방향이다.

**확인 필요**: 이 PDF는 Acuity 6.21.x 기준이고 우리 파이프라인은 6.12.0 binary를 쓴다. 6.12에서 위 op들이 다 되는지, 특히 sin/cos/gather_nd/scatter_nd가 6.12 시점에 이미 있었는지는 **실제 import 시도로 검증 필요**(`pegasus import onnx`가 unsupported op를 만나면 즉시 에러를 내므로 확인이 빠름).

#### 2.6 KV-Cache 양자화

- **KIVI**(2bit asymmetric, calibration-free), **KVQuant**(sub-4bit, learned params) 등은 GPU/서버 서빙 최적화가 목적이라 **NPU edge 배포와 목표가 다름**. T527에서는 K/V를 int8(또는 int16)로 저장하는 정도가 현실적 — 정밀도보다 **고정 크기 텐서로 선언 가능한가**가 핵심.
- [GPU-Accelerated INT8 Quantization for KV Cache Compression](https://arxiv.org/html/2601.04719v1): per-channel(head dimension 축) quantization으로 4x 메모리 절감, 정확도 손실 미미 — T527처럼 int8-only 하드웨어에 참고할 만한 granularity 전략.
- **정적 그래프 KV-cache 실전 패턴**([KV-RM: Regularizing KV-Cache Movement for Static-Graph LLM Serving](https://arxiv.org/pdf/2605.09735), [Qualcomm NPU 방식 요약](https://arxiv.org/pdf/2607.01108)): "KV-cache를 최대 길이 M으로 고정 할당하고, 매 스텝마다 M 전체에 대해 attention을 계산"하거나, **"현재 캐시 길이에 맞는 step-specific 그래프를 여러 개 미리 컴파일해두고 스위칭"**하는 두 가지가 업계 표준 우회법. 후자가 T527의 "NB 파일 = 고정 그래프" 특성과 정확히 맞아떨어짐 — **Conformer가 301프레임 고정 shape NB를 만든 것과 동일한 발상**으로 "K개 토큰 길이별 NB 여러 개"를 만드는 접근이 유력.

#### 2.7 한국어 특화 Calibration

- 일반 PTQ 연구([Calibration Data for LLM Quantization](https://apxml.com/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/calibration-data-selection)): calibration corpus를 WikiText→도메인 특화 corpus로 바꾸면 정확도가 28%→65%까지 뛰는 사례가 보고됨 — **calibration corpus의 도메인 일치가 일반 PTQ보다 LLM PTQ에서 훨씬 민감**하다는 신호.
- 한국어 LLM 학습에는 AI-Hub 코퍼스, 국립국어원 "모두의 말뭉치"가 실제로 쓰이고 있음(Midm-2.0, EXAONE 계열 공통 언급) — **우리 팀이 Conformer QAT 때 이미 확보한 AIHub 데이터(`/nas04/nlp_sk/STT/data/train/`)를 텍스트 코퍼스로도 활용 가능한지 확인 가치 있음**(음성-텍스트 페어의 transcript 텍스트만 뽑아 LLM calibration corpus로 재활용).
- Conformer 프로젝트에서 확립한 원칙 "calibration은 양보다 다양성"(`260326_quantization_deep_dive_discussion.md`)이 LLM 텍스트 도메인에도 그대로 적용될 것으로 추정되나, **LLM은 activation outlier가 특정 rare token(이모지, 특수문자, 숫자 등)에서 튀는 경우가 많아 "다양한 화자"보다 "다양한 토큰 분포"가 기준이 되어야** — 확인 필요.

---

### Facet 3: Conformer → LLM 이식 관점

#### 3.1 그대로 재활용 가능한 것

| 요소 | Conformer에서 확립됨 | LLM에서 그대로 적용 |
|---|---|---|
| Acuity 파이프라인 순서 | import→quantize→export | 동일 |
| `--quantizer asymmetric_affine --qtype uint8` | 확립 | LLM도 W8A8-only이므로 동일 |
| `--algorithm kl_divergence` (or moving_average) | 확립, calibration 10~500개 | LLM calibration도 동일 알고리즘 실험 가능(단 텍스트 도메인 재검증 필요) |
| `inputmeta.yml` 구조 + `lid` 매칭 | 확립 | LLM은 입력이 audio_signal 하나가 아니라 `input_ids`, `attention_mask`, (KV-cache 여러 개) — **다중 입력 lid 매핑 신규 학습 필요** |
| `--optimize VIP9000NANOSI_PLUS_PID0X10000016` | 확립(실패 반복 후 확정) | 동일 플래그 그대로 사용 예상(하드웨어 고정값이므로) |
| 커스텀 `.quantize` YAML 직접 생성 가능성 | 발견됨(Insight 3) | LLM PTQ 커스터마이징(SmoothQuant 반영, 레이어별 sensitivity)에 **핵심적으로 활용**해야 함 |
| static shape 고정 (Conformer: 301프레임) | 확립 | LLM auto-regressive decode도 "고정 시�퀀스 길이 그래프 여러 개" 전략으로 동일 원리 적용(3.2 참고) |

#### 3.2 새로 알아야 할 것 — Conformer와 근본적으로 다른 점

| 차이 | Conformer | LLM |
|---|---|---|
| 실행 패턴 | Encoder 1회 forward (single-shot) | **Prefill(1회, 긴 시퀀스) + Decode(토큰마다 반복, N회)** — decode 루프를 어떻게 NPU에 태울지가 핵심 미해결 과제 |
| KV-cache | 없음 | **필수** — static shape NB에서 어떻게 관리할지가 프로젝트 성패를 가름 (Facet 2.6 참고, "길이별 NB 여러 개 컴파일" 전략 유력) |
| Tokenizer/embedding/lm_head 배치 | 해당 없음(CTC decoder는 vocab_size 출력만) | **embedding_lookup, lm_head(대형 matmul, vocab_size 5~15만)를 NPU에 둘지 CPU에 둘지 결정 필요** — Conformer 팀의 "lm_head FP32 분리" 시도(Split Model)가 참고가 되나, LLM lm_head는 훨씬 큼(vocab×hidden, 예: 150k×2048 = 3억 파라미터) → **CPU 분리 시 이 자체가 병목**이 될 가능성 |
| 검증 방법 | CER (문자 오류율) | **Perplexity, KL/JSD(golden logits vs quantized logits), 실제 생성 텍스트 비교** — Facet 5 참고 |
| 그래프 반복 구조 | 없음 | Transformer block N개 반복 — **레이어 수가 많을수록 activation range 누적 오차 위험 큼(Conformer의 "레이어 지날수록 range 커지면 실패" 원칙과 동일선상, 그러나 LLM은 레이어당 residual + attention + FFN이 훨씬 복잡)** |

---

### Facet 4: LLM 배포 프레임워크와 NPU 통합

| 프레임워크 | Vivante/T527 연결성 | 비고 |
|---|---|---|
| **llama.cpp custom backend** | **없음 확인** — CANN(Ascend), OpenVINO(Intel), Qualcomm HTP(Hexagon) 백엔드는 존재하나 Vivante 전용 백엔드는 2026-07 기준 미확인 | [llama.cpp backend 목록](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md) |
| **mlc-llm** | Vulkan/OpenCL 백엔드 존재하나 "non-Apple 모바일 GPU에서 ALU utilization 5-20%"로 저조 — VIP9000의 OpenVX 방식과 결이 다름(TVM 컴파일 vs Acuity 컴파일) | [MLC-LLM 2026 가이드](https://localaimaster.com/blog/mlc-llm-setup-guide) |
| **MNN-LLM** (Alibaba) | mobile CPU/GPU 최적화 중심, "llm.npu"는 float 연산을 CPU/GPU로 보내고 int8 MatMul만 NPU로 라우팅하는 **하이브리드 스케줄링** 방식 채택 — T527처럼 FP16 HW가 없는 칩엔 CPU 라우팅 비중이 커질 것으로 예상 | [MNN-LLM arXiv](https://arxiv.org/html/2506.10443v1) |
| **onnxruntime + VIPLite EP** | **커뮤니티에서 이미 요청 중.** [microsoft/onnxruntime#28244](https://github.com/microsoft/onnxruntime/issues/28244) — evgen-pervenenko가 2026-04-27에 A733/T527용 VIPLite Execution Provider 추가를 요청(stale 처리, 메인테이너 응답 없음). 우리가 필요하면 **직접 만들어야 하는 상황** |
| **ExecuTorch (Meta)** | 2025-10 1.0 GA, 2026-01 v1.1.0. Qualcomm/MediaTek/Samsung Exynos/Vulkan/NXP 등 12+ 백엔드 지원하나 **Vivante 없음**. 아키텍처(50KB 런타임, KV-cache 양자화, sliding-window attention, 4bit group-wise weight quant 지원)는 T527에 이식할 때 설계 참고자료로 유용 | [ExecuTorch 논문](https://arxiv.org/pdf/2605.08195) |
| **Cactus (YC startup, 2025-12)** | Mac/iPhone/Galaxy 벤치마크(173/136/91 tok/s)만 확인, Vivante/T527 미언급. INT8 NPU 가속 + FP32→2bit 유연 양자화 지원 컨셉은 참고 가치 | [Cactus v1 InfoQ](https://www.infoq.com/news/2025/12/cactus-on-device-inference/) |
| **T-MAN (2025-11)** | Qualcomm NPU 전용 저비트 table-lookup 커널 — T527엔 직접 적용 불가하나 "table lookup으로 int8 MAC array에서 저비트 연산 흉내"라는 아이디어 자체는 이식 검토 가치 | [T-MAN arXiv](https://arxiv.org/html/2511.11248) |

**결론**: 상용 프레임워크 중 Vivante VIP9000을 1급 지원하는 것은 **없다**. 우리 팀이 이미 하고 있는 것처럼 **Acuity Toolkit(pegasus) 직접 사용 + 자체 JNI/awnn 래퍼(Conformer에서 검증된 `awnn_lib.h` API)**로 가는 것이 유일한 실전 경로. onnxruntime EP 요청 이슈는 앞으로 커뮤니티 움직임을 모니터링할 가치 있음.

---

### Facet 5: 실전 팁 / 함정

#### 5.1 Tokenizer
- SentencePiece(Qwen, EXAONE, Midm 대부분 사용)와 BPE(GPT계열)는 Android 배포 시 둘 다 라이브러리 크기·속도 면에서 큰 차이 없음 — **Conformer 프로젝트가 이미 SentencePiece 배포 경험 보유**(`tokenizer.model`, `vocab_ko.txt`)이므로 SentencePiece 계열 모델(Qwen, Midm) 선택 시 **토크나이저 통합 코드 재사용 가능**.

#### 5.2 Static-shape NBG로 auto-regressive 돌리는 패턴
- 업계 공통 해법 두 가지(Facet 2.6에서 상술):
  1. **최대 길이(M) 고정 + 매 스텝 전체 M에 대해 연산**(단순하나 낭비 큼, Qualcomm 초기 방식)
  2. **길이 구간별로 여러 개의 static NB를 미리 컴파일**(chunk-by-chunk) — T527/Conformer 경험과 가장 잘 맞음. 예: prefill용 NB(긴 시퀀스 1개), decode용 NB(1토큰 입력 + 고정 크기 KV-cache 슬롯) 2종 조합
- **prefill과 decode를 분리**하는 것은 거의 모든 최신 NPU LLM 논문의 공통 패턴(Quant.npu, NITRO, MNN-LLM llm.npu 전부 이 구조).

#### 5.3 LLM golden tensor 검증
- Conformer는 `output_0.dat` 직접 비교(argmax 일치율)로 검증했으나, LLM은 **다음 토큰 분포(logits)이므로 argmax만 보면 정보 손실이 큼**.
- 권장 검증 지표:
  - **Top-1/Top-5 토큰 일치율**(Conformer의 argmax 방식과 유사)
  - **KL divergence / Jensen-Shannon divergence(logits softmax 후)** — [TAQ-KL](https://arxiv.org/pdf/2604.13440) 등 최신 연구에서 "SQNR보다 KL이 autoregressive LM에서 더 잘 일반화"한다고 보고
  - **Perplexity** (held-out 한국어 코퍼스에서 FP32 대비 증가율)
  - 최종적으로는 **실제 생성 텍스트의 사람 평가/자동평가(KMMLU 등 벤치 subset)**까지 가야 진짜 검증이지만, 1차 스크리닝은 KL/perplexity로 충분

---

## 상충 정보 / 주의사항

- **op 지원 범위**: TIM-VX(오픈소스, GitHub) 문서와 Allwinner 벤더 Acuity 6.21 PDF 간 op 커버리지가 다르다. 실제 배포에 쓰는 Acuity 6.12.0 binary가 6.21 PDF와 동일한 op 셋을 가졌는지는 **미검증** — import 단계에서 직접 확인 필요.
- **Trillion-1.4B**: 사용자가 언급한 모델명이 웹 검색으로 실물 확인 안 됨. Trillion Labs 라인업은 Tri-0.5B/1.9B/7B, Trillion-7B-preview로 확인됨 — **모델명 재확인 필요**.
- **kanana-nano-2.1b / EXAONE-3.5 아키텍처 세부사항(레이어수, GQA head 수 등)**: 공식 HF 카드에 명시 안 됨 — 기술보고서(arXiv) 원문 확인 필요.
- **RK3588 vs T527 성능 외삽**: RK3588 TinyLlama-1.1B 10-15 tok/s는 6 TOPS 기준. T527은 2 TOPS(1/3)이나 TOPS와 tok/s가 선형 비례하지 않음(메모리 대역폭, 오버헤드 등 변수) — **실측 전까지는 추정치**.
- **NITRO 논문 PDF**: 텍스트 추출이 실패하여 상세 내용(정량적 벤치마크)을 확인하지 못함 — **재조사 필요 시 arXiv HTML 버전(`arxiv.org/abs/2412.11053`)으로 재시도 권장**.

---

## Sources (신뢰도순)

**A. 공식 / 권위**
- [NPU_算子支持列表.pdf (Allwinner, 로컬)](file:///home/nsbb/travail/claude/T527/docs/acuity_toolkit/NPU模块开发指南/NPU_算子支持列表.pdf) — v1.1, 2024-02-04, Acuity 6.21.x/IDE 5.8.2, T527 대상
- [VeriSilicon/TIM-VX Operators.md](https://github.com/VeriSilicon/TIM-VX/blob/main/docs/Operators.md)
- [Qwen2.5 Technical Report](https://arxiv.org/pdf/2412.15115)
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- [Mi:dm 2.0 Korea-centric Bilingual Language Models](https://arxiv.org/pdf/2601.09066)
- [KORMo: Korean Open Reasoning Model for Everyone](https://arxiv.org/pdf/2510.09426)
- [K-intelligence/Midm-2.0-Mini-Instruct (HF)](https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct)
- [kakaocorp/kanana-nano-2.1b-instruct (HF)](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)
- [LG-AI-EXAONE/EXAONE-3.5 (GitHub)](https://github.com/LG-AI-EXAONE/EXAONE-3.5)
- [A Case Study of Selected PTQ Baselines for Reasoning LLMs on Ascend NPU](https://arxiv.org/pdf/2602.17693) — 2026
- [Quant.npu: Enabling Efficient Mobile NPU Inference for on-device LLMs via Fully Static Quantization](https://arxiv.org/pdf/2605.20295) — 2026
- [EXECUTORCH - A Unified PyTorch Solution to Run AI Models On-Device](https://arxiv.org/pdf/2605.08195)
- [MNN-LLM: A Generic Inference Engine for Fast LLM Deployment on Mobile Devices](https://arxiv.org/pdf/2506.10443)

**B. GitHub / Maintainer**
- [Add VIPLite Execution Provider for Allwinner VIP9000 NPU (A733/T527 SoCs) · Issue #28244 · microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/issues/28244) — 2026-04-27, stale, 미해결
- [Running Frigate on the Allwinner A733 / Vivante VIP9000 NPU · Discussion #23418](https://github.com/blakeblackshear/frigate/discussions/23418) — 2026-06-06, int16 권장, CHW/HWC 버그 발견 사례
- [GatekeeperZA/InternLM2-1.8B-w8a8-RKLLM-v1.2.3 (HF)](https://huggingface.co/GatekeeperZA/InternLM2-1.8B-w8a8-RKLLM-v1.2.3)
- [Pelochus/ezrkllm-collection (HF)](https://huggingface.co/Pelochus/ezrkllm-collection)

**C. Stack Overflow / HN / arXiv 서베이**
- [KIVI 2-bit KV cache quantization](https://dl.acm.org/doi/10.5555/3692070.3693381)
- [GPU-Accelerated INT8 Quantization for KV Cache Compression](https://arxiv.org/html/2601.04719v1)
- [KV-RM: Regularizing KV-Cache Movement for Static-Graph LLM Serving](https://arxiv.org/pdf/2605.09735)
- [NPUsper: Eliminating Redundant Computation for Real-Time Whisper on Mobile NPUs](https://arxiv.org/pdf/2607.01108) — static-shape NPU decode 우회 패턴
- [A KL Lens on Quantization](https://arxiv.org/pdf/2604.13440)
- [Half-Quadratic Quantization (HQQ) blog](https://dropbox.github.io/hqq_blog/)

**D. 블로그 / 커뮤니티 / 벤치마크 사이트**
- [TinyComputers.io — RK3588 NPU Deep Dive](https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html)
- [Navigating Korean LLM Research #1: Models (HF blog, amphora)](https://huggingface.co/blog/amphora/navigating-ko-llm-research-1)
- [On-Device LLM Inference: The Definitive 2025-2026 Guide (Octomil)](https://docs.octomil.com/blog/on-device-llm-inference-2025-2026/)
- [Cactus v1 (InfoQ)](https://www.infoq.com/news/2025/12/cactus-on-device-inference/)
- [MLC-LLM Setup 2026 (Local AI Master)](https://localaimaster.com/blog/mlc-llm-setup-guide)

---

## 실행 계획 제안

### 1. 첫 타겟 한국어 LLM (Top-3, 라이선스/성능/구조 종합)

1. **Qwen2.5-0.5B-Instruct (Apache-2.0)** — 1차 파일럿. 라이선스 리스크 없음, GQA+tied embedding으로 크기 작고, 다국어 코퍼스에 한국어 포함되어 최소한의 한국어 응답 가능. ONNX export 경로가 이미 커뮤니티에서 검증됨.
2. **Midm-2.0-Mini-Instruct (2.3B, MIT, KT)** — 진짜 한국어 실전 타겟. MIT라 상업 배포 자유롭고, KT가 온디바이스 배포를 염두에 두고 pruning+distillation한 모델이라 우리 목적(T527 SDK 탑재)과 결이 맞음. 단 2.3B는 T527 RAM 제약(1-2GB) 상단에 걸치므로 **가장 먼저 weight 용량(양자화 후 예상 ~2.3GB→uint8 시 절반 이하)을 계산해 RAM 여유 확인** 필요.
3. **Qwen2.5-1.5B-Instruct 또는 Qwen3-1.7B (Apache-2.0)** — Midm이 안 되면 대체. 다국어라 한국어 품질은 Midm/kanana보다 낮을 가능성 있으나 아키텍처가 가장 표준적(Optimum ONNX export, HF 생태계 자료 풍부)이라 디버깅 난이도가 낮음.

*(참고 벤치마크 전용, 상업 배포 전 라이선스 재검토 필요: kanana-nano-2.1b(KMMLU 44.80, 한국어 강함), EXAONE-3.5-2.4B)*

### 2. 양자화 파이프라인 순서

**Phase 0 (재활용):**
- Conformer 파이프라인 그대로 사용: NeMo Docker 대신 **HuggingFace `optimum-cli export onnx`**로 대체, 이후 `fix_onnx_for_acuity.py`류 static-shape 고정 스크립트를 LLM용으로 개조(dynamic `seq_len`, `past_key_values` shape을 전부 상수로 치환).
- `--quantizer asymmetric_affine --qtype uint8 --algorithm kl_divergence`부터 시작(baseline).

**Phase 1 (신규, 우선순위 높음):**
- ONNX 레벨에서 **SmoothQuant 적용**(activation outlier를 weight로 이전) — 이게 Conformer에는 없던, LLM 성패를 가를 가능성이 가장 큰 신규 기법. 오픈소스 구현(`smoothquant` 공식 repo 또는 HF `optimum-quanto`/AMD Quark 참고) 활용.
- RMSNorm/RoPE를 Acuity native op 조합으로 분해하는 ONNX 그래프 전처리 스크립트 작성(6.12 op 지원 여부 먼저 `pegasus import`로 검증).

**Phase 2 (신규):**
- Prefill/Decode 분리 NB 2종 컴파일. Decode용은 고정 길이 K개 토큰(예: 1) + 고정 크기 KV-cache 슬롯을 입력으로 받는 구조.
- Calibration corpus: AIHub STT 데이터의 transcript 텍스트 재활용 + 일반 한국어 코퍼스(모두의 말뭉치) 혼합, 100~500 샘플로 시작(Conformer 경험치 그대로 적용, 사후 재검증).

**Phase 3 (검증):**
- Conformer의 CER 대신 **Top-1 토큰 일치율 + KL divergence(FP32 vs uint8 logits) + perplexity**로 검증 지표 전환.
- 필요 시 커스텀 `.quantize` YAML 생성(레이어별 sensitivity 분석) 단계 투입 — 이미 팀이 확립한 "Acuity quantize 단계 우회 가능" 인사이트 그대로 사용.

### 3. 예상 성능 (tok/s decode) — 추정치, 실측 전 잠정

| 모델 | 근거 | T527(2 TOPS) 추정 decode 속도 |
|---|---|---|
| SmolLM2-135M | petayyyy A733 실측 21 tok/s(KV-cache 없이 재계산) | 검증 완료 기준점 |
| Qwen2.5-0.5B | RK3588(6 TOPS) TinyLlama-1.1B 10-15 tok/s 외삽(파라미터 절반, TOPS 1/3) | **5~10 tok/s** (낙관적 추정, KV-cache 있을 시) |
| Midm-2.0-Mini(2.3B) / Qwen2.5-1.5B | 위와 동일 스케일링, 파라미터 1.5~2배 | **2~5 tok/s** |

**주의**: 이 표는 RK3588 대비 TOPS 비례 외삽에 불과하며, 실제로는 메모리 대역폭·NB tiling 효율·KV-cache 오버헤드가 지배적 변수가 될 수 있어 **큰 오차 가능성이 있음(확인 필요)**. Conformer 233ms/inference(single-shot, 122M) 실측치와 비교해 LLM decode 루프의 오버헤드가 어느 정도인지 첫 실측이 나오기 전까지는 참고치 이상으로 쓰지 말 것.

### 4. 첫 검증 마일스톤

1. **SmolLM2-135M** — 아키텍처 검증(RoPE/RMSNorm/Softmax op 분해가 Acuity 6.12에서 실제로 import되는지 1차 확인). 한국어 품질은 기대하지 않음, 순수 파이프라인 배관 검증용.
2. **Qwen2.5-0.5B-Instruct** — 한국어+영어 균형, Apache 라이선스, GQA/tied embedding 검증. SmoothQuant 적용 유무 A/B 비교.
3. **Midm-2.0-Mini-Instruct(2.3B) 또는 Qwen2.5-1.5B** — 실전 타겟 규모. RAM/속도 한계 실측, prefill/decode NB 분리 전략 완성.
4. (선택) **kanana-nano-2.1b** — 라이선스 이슈로 상업 배포는 제외하되, "한국어 성능이 얼마나 좋은 모델이 양자화를 견디는가"를 비교하는 참고 실험으로 병행 가치 있음.

---

## 추가 조사 필요

- Acuity **6.12.0**(현재 팀이 실제 쓰는 버전)에서 Sin/Cos/Gather_ND/ScatterND/LayerNorm의 실제 지원 여부 — PDF는 6.21.x 기준이라 버전 차이 검증 필요(가장 빠른 방법: 더미 ONNX로 `pegasus import` 시도).
- Midm-2.0-Mini / kanana-nano-2.1b / EXAONE-3.5의 정확한 GQA head 수, RoPE base, 레이어 수 — 공식 HF 카드에 없어 arXiv 기술보고서 원문 확인 필요.
- Trillion-1.4B라는 모델명의 실존 여부(Tri-1.9B의 오기일 가능성).
- SmoothQuant를 Acuity import 이전 ONNX 그래프에 실제로 삽입하는 구체적 스크립트 사례(오픈소스에 T527향 사례는 없음 — 우리가 처음 만들어야 할 가능성 높음).
- RK3588→T527 성능 외삽치의 정확도 — 1차 실측(예: SmolLM2-135M decode 속도 실측)이 나오면 이 표 전체를 재계산해야 함.
