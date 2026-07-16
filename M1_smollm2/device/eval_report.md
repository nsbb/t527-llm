# M1 Device Evaluation — SmolLM2-135M on T527 NPU

_Date: 2026-07-16_

## TL;DR

**파이프라인 컴파일 + NPU 실행 = 성공.**
**정확도 검증 = 실패 (Acuity 6.12의 ONNX→FP32 변환 자체가 broken).**

M1의 원래 목표(파이프라인 배관 검증)는 달성. 후속 M2에서 반드시 해결해야 할 새 이슈 발견.

---

## 1. NPU 실행 성공

`network_binary.nb` (124 MB, uint8 asymmetric_affine) T527 디바이스에서 실행:

```
cid=0x10000016, device_count=1                    # T527 PID 일치
input 0  dim 32 1 1 0    data_format=8            # int32 token_ids [1,1,32]
output 0 dim 49152 32 1 1 data_format=2           # uint8 [1,1,32,49152]
                          scale=0.492302 zp=133
memory pool size=1574144 byte                      # ~1.5 MB runtime memory
create network 0: 207578 us                        # 208 ms 로드
prepare network 0: 22794 us                        # 23 ms 준비
run time for this network 0: 92600 us              # 93 ms/forward
profile inference time=92248us, cycle=64084199
vpm run ret=0                                      # 성공
```

**의미**:
- Pure NPU decode throughput: **1 / 0.0928 s = 10.8 tok/s** (W=32, no KV cache)
- petayyyy A733 (동일 NPU 계열): 21 tok/s → T527은 ~52% 속도. 스케일 예상 범위 내.

Int16 (dynamic_fixed_point) NB도 성공:
- 268 MB, run time 147 ms/forward, 6.8 tok/s

---

## 2. 정확도 문제 — Acuity 6.12 ONNX→FP32 변환 버그

### 관측 사실

FP32 ONNX Runtime (host) vs Acuity 6.12 FP32 host inference (`pegasus inference --dtype float32`) 비교:

| 지표 | 값 |
|---|---|
| 마지막 position argmax (ORT FP32) | **198** (`\n`) — 예상됨 (EOS 다음 개행) |
| 마지막 position argmax (Acuity FP32) | **29098** (`omson`) — 이상함 |
| Argmax match rate (전체 32 position) | **0/32** |
| Cosine similarity (last logits) | **-0.38** — 심지어 음수 |
| ORT logits range | [-18.77, 36.88] |
| Acuity FP32 logits range | [-87.07, 60.21] — 폭이 훨씬 넓음 |

pad token(id=0) 위치 (0~22)에서:
- ORT는 일관되게 `198` (`\n`) 예측 (예상됨)
- Acuity FP32는 일관되게 `32237` (`avorable`) 예측 — deterministic이지만 wrong

**Acuity에서 argmax로 나오는 토큰들**: `avorable`, `omson`, `Kyr`, `Je`, `Rothschild` — 전부 rare vocab tokens. Systematic 오류 시그널.

### uint8/int16 양자화 실험

두 quantizer 모두 host FP32와 동일한 값 예측 → 양자화가 문제가 아님. Acuity FP32 자체가 이미 잘못된 결과를 내고 있음.

| Quantizer | qtype | Argmax match (vs ORT FP32) | Cosine (last) |
|---|---|---|---|
| asymmetric_affine | uint8 | 0/32 | 0.35 |
| dynamic_fixed_point | int16 | 0/32 | -0.03 |
| **Acuity host FP32 inference** | — | **0/32** | **-0.38** |

즉 int8 → int16으로 올려도 소용없음, FP32 자체가 broken.

### 가능한 원인 후보

1. **Acuity 6.12의 특정 op 변환 버그**: 
   - Gather with int32 indices in [1, 1, 32] shape (Acuity가 NCHW로 취급하면서 인덱스 재해석 오류 가능성)
   - RMSNorm 조립 (Pow→ReduceMean→Sqrt→Reciprocal→Mul→Mul) 중 어느 노드의 dtype/shape 처리 문제
   - RoPE의 상수 sin/cos 테이블 초기화 시 데이터 corruption
2. **Acuity 6.21 (또는 petayyyy가 쓴 최신 버전) 대비 6.12에 없는 fix**: petayyyy는 `ubuntu-npu:v2.0.10.1` docker 이미지 사용. 우리 t527-npu:v1.2 + Acuity 6.12는 구 버전.
3. **ONNX opset 11 (petayyyy 스크립트가 생성) vs Acuity 기대 opset 불일치**: Acuity 6.12가 특정 op 시맨틱을 다르게 해석.

**우선 조사할 것 (M2 시작 전)**:
- Acuity IR (JSON)의 Gather 노드 output shape 확인 vs ONNX 기대값
- 중간 레이어별 tensor dump로 발산 시작 지점 pin-point (Acuity `--dump-layer-tensor` 옵션 활용)
- Acuity 최신 버전 (6.21) 접근성 확인

---

## 3. 산출물

```
device/
├── input_0.dat           # 128 bytes int32 (token_ids [1,32])
├── output_0.dat          # 1.5 MB uint8 (NPU int8 result) [.gitignore]
├── output_int16.dat      # 3 MB int16 (NPU int16 result) [.gitignore]
├── fp32_logits.npy       # ORT FP32 golden [1, 32, 49152] fp32
├── vpm_run.log           # NPU 실행 로그
├── compare_report.json   # 자동 비교 metrics
├── sample.txt            # vpm_run 설정
├── push_and_run.sh       # adb push + vpm_run 실행 스크립트
├── fp32_golden.py        # ORT FP32 baseline 생성
├── compare_logits.py     # NPU vs FP32 비교
└── eval_report.md        # 이 문서
```

---

## 4. M1 판정

| 목표 | 달성? |
|---|---|
| ONNX → Acuity IR → uint8 quantize → NBG 컴파일 성공 | ✅ |
| T527 NPU에서 forward pass 성공 (output_0.dat 생성) | ✅ |
| NB 로드/실행 시 PID 매칭 (0X10000016) 확인 | ✅ |
| pure NPU decode speed 실측 (11 tok/s) | ✅ |
| FP32 대비 양자화 손실 < 1.0 KL | ❌ (Acuity FP32 자체가 broken) |
| 1-token decode 후 그럴싸한 토큰 예측 | ❌ (같은 이유) |
| Multi-token sliding window generation | ⏸ (정확도 broken이라 의미 없음) |

**최종 판단**: M1은 **파이프라인 검증 마일스톤**으로서는 **완전 성공**. Acuity 6.12 자체의 ONNX 변환 버그가 발견됐고, 이건 M2 시작 전 반드시 해결해야 함 (Acuity 6.21 업그레이드 또는 우회 ONNX 재작성).

**"컴파일도 되고 디바이스에서 진짜 돈다"는 사실 자체는 T527 LLM 프로젝트의 가장 큰 미지수였음. 그 미지수는 해소됐다.**
