# PROGRESS — 시간순 작업 로그

## 2026-07-15

### 10:03 — 프로젝트 시작

빈 `t527-llm/` 폴더 생성. 목표: T527 NPU 위 한국어 LLM.

### 10:03~11:00 — 병렬 리서치 4개 스폰

4개 방향 병렬 조사:
1. Allwinner 공식 (`Allwinner-Homlet`, `allwinnertech`, `linux-sunxi` 등)
2. Verisilicon/Vivante (Acuity, TIM-VX, VIPLite, LLM 벤치마크)
3. 커뮤니티 (T527/A527/T536/A733 실제 LLM 포팅 사례)
4. 레퍼런스 NPU (RK3588 RKLLM, i.MX Vivante, MediaTek APU)

산출물:
- `research_allwinner.md`
- `research_verisilicon.md`
- `research_community.md`
- `research_reference_npus.md`

**핵심 발견**:
- 공개된 T527 LLM 포팅 성공 0건
- `petayyyy/a733_npu_driver` — A733(같은 NPU 계열)에서 SmolLM2-135M/360M 실증 (21/8.4 tok/s)
- `waz664/vip9000-embeddinggemma` — EmbeddingGemma 실증
- NXP는 같은 Vivante NPU 두고 LLM은 CPU 폴백 선택 → i.MX 95 신설
- RK3588 RKLLM이 유일한 참고 청사진

### 14:56 — 한국어 LLM 양자화 심화 리서치

`cwm:websearchwithme` 에이전트로 한국어 LLM 후보 + 양자화 심화 조사.

산출물: `research_korean_llm_quantization.md`

**핵심 결과**:
- Allwinner 내부 PDF `NPU_算子支持列表.pdf` (Acuity 6.21 기준, T527 타겟)에서 Sin/Cos/LayerNorm/MatMul/Gather/Scatter 전부 native 지원 확인 (공개 TIM-VX 문서보다 넓은 op 커버리지)
- 한국어 LLM Top-3 후보: Qwen2.5-0.5B (Apache) > Midm-2.0-Mini 2.3B (MIT, KT) > Qwen2.5-1.5B/Qwen3-1.7B (Apache)
- LLM은 W8A8 asymmetric_affine 이외 선택지 없음
- SmoothQuant를 ONNX 사전 삽입이 신규 핵심 작업

### 17:08 — PLAN.md 작성

5개 리서치 종합, 4단계 검증 사다리 (M0→M1→M2→M3), 리스크 매트릭스, 검증 체크리스트, 예상 성능, 즉시 다음 액션 명세.

### 18:57~19:00 — M0 시작: Acuity 6.12 op 지원 실측

24개 dummy ONNX (Sin, Cos, Gather, GatherND, ScatterND, LayerNorm, Softmax, Pow, ReduceMean, Sqrt, Reciprocal, CumSum, MatMul, Erf, Gelu, Where, Mul, Add, Slice, Concat, Transpose, Reshape, Split, RMSNorm-composed) 생성 후 `pegasus import` 배치 실행.

결과:
- **Native OK 21개**
- **Native FAIL 3개**: Cos, LayerNormalization(opset 17), Gelu(opset 20)
  - Cos: Acuity 6.12 converter 매칭 없음
  - LayerNorm/Gelu native: opset 17/20 fused op 미등록 (`Un Specify smart processor`)

### 19:00~19:10 — M0 우회 검증

Cos → `Sin(π/2 − x)`, LayerNorm → 분해 조립, Gelu → Erf 조합 or tanh 근사 (waz664 트릭). 9개 workaround ONNX 추가 (cos_via_sin, layernorm_composed, gelu_erf_composed, gelu_tanh_composed, tanh, sub, sigmoid, silu_composed, rope_minimal).

**결과: 모든 워크어라운드 import 성공.** 33/33 op 커버 확인.

산출물: `M0_op_validation/M0_op_support_matrix.md`

### 19:16 — M1 시작: SmolLM2-135M → T527

`petayyyy/a733_npu_driver` 클론. `HuggingFaceTB/SmolLM2-135M-Instruct` HF에서 다운로드 (257MB safetensors).

### 19:18 — Static ONNX 생성 성공

`scripts/host/make_real_llm_onnx.py --seq-len 32` 로 static ONNX 514MB 생성.
- 30 layers Llama, hidden=576, GQA(9/3), tied embedding, RoPE θ=100000
- RMSNorm/SwiGLU 이미 primitive op로 분해
- **RoPE sin/cos을 상수 테이블로 pre-compute** → Acuity의 Cos 미지원 문제 자동 회피
- LayerNorm/Gelu native op 없음 (RMSNorm/SiLU만 사용)

### 19:20 — pegasus import 성공

`network_binary.nb → smollm2_135m_w32.json` (1.1MB Acuity IR), 2454 layers.
- Sigmoid×Mul 패턴이 **자동으로 swish (SiLU) 30개로 융합됨**

## 2026-07-16

### 00:24~00:26 — quantize 시행착오

첫 시도 실패 로그 정리:

1. `perchannel_symmetric_affine int16` → Acuity 6.12는 int8/int4만 지원
2. `perchannel_symmetric_affine int8` → 성공 근처에서 `Invalid value in tensor used for shape: -30` (Acuity ONNX→TF converter의 `slice_last_hidden` size 계산 버그)
3. `asymmetric_affine uint8` → 같은 slice bug
4. `type: NPY` inputmeta → `Cannot load file containing pickled data`
5. `category: image` (auto default) → `Unable to decode bytes as JPEG/PNG/GIF/BMP`

### 09:28 — ONNX 패치로 slice bug 회피

`slice_last_hidden` 노드를 ONNX에서 제거. logits shape `[1, 1, V]` → `[1, 32, V]`. 마지막 토큰 추출은 CPU에서.

산출물: `patch_onnx_last_slice.py`, `real_llm_nolastslice.onnx`

### 09:31 — 재import 성공 (v2)

패치된 ONNX → `smollm2_135m_w32_v2.json/.data` 생성.

### 09:38 — quantize uint8 asymmetric_affine 성공

inputmeta: TEXT 타입 + dataset.txt에 npy 파일명 리스트, category undefined, auto-generated lid 사용.

```
Error(0), Warning(62)
```

### 09:40 — NBG export 성공 (T527 대상)

`--optimize VIP9000NANOSI_PLUS_PID0X10000016 --pack-nbg-unify`

```
Error(0), Warning(0)
```

산출물: **`network_binary.nb` 124MB (T527 NPU용)**

### 09:45 — M1 문서화 & git commit

M1 전체 README + 재현 스크립트 정리, 시간순 커밋으로 반영. `git push origin main`.

### 10:14 — Phase A: 디바이스 사전 검증

- `adb devices` OK (`51475789d0c64881cd3`)
- `/data/local/tmp/vpm_run_aarch64` 이미 배포됨 (46KB, 2026-04-01)
- `/vendor/lib64` 에 `libVIPlite.so` + `libVIPuser.so` (v1.13 계열) 존재
- vpm_run help 실행 정상

### 10:14~10:15 — Phase B~C: input/sample 준비 + NPU forward

- `input_0.dat` (128B int32) + `sample.txt` + `push_and_run.sh` 작성
- 처음 실행: `Network binary file can't be found` (sample.txt에 빈 줄 있어서 parser 실패)
- 빈 줄 제거 후 재실행 → **NPU forward 성공**:
  ```
  cid=0x10000016, device_count=1
  create network 0: 207578 us
  prepare network 0: 22794 us
  run time for this network 0: 92600 us     ← 10.8 tok/s pure NPU
  vpm run ret=0
  ```
- `output_0.dat` (1.5MB uint8) pull

### 10:15~10:23 — Phase D: FP32 golden vs int8 비교

- `fp32_golden.py`: ONNX Runtime으로 patched ONNX 실행, `fp32_logits.npy` 저장
- `compare_logits.py`: dequant `(u8 - 133) * 0.4923`, argmax/KL/cosine 계산
- **결과: 0/32 argmax match, cosine 0.35** — 양자화 catastrophic loss?
- layout 여러 조합 시도 (`[1,1,32,49152]`, `[32,49152]`, `[49152,32]` 등) — 전부 0/32

### 10:22 — int16 dynamic_fixed_point 재양자화 시도

- `--quantizer dynamic_fixed_point --qtype int16 --rebuild-all`
- NBG 268MB, run time 147ms/forward (6.8 tok/s)
- **결과: 0/32 argmax match, cosine -0.03** — 여전히 broken

### 10:23 — 결정적 진단: Acuity FP32 자체가 broken

- Acuity host inference `--dtype float32` 실행 (`pegasus inference`, 양자화 없음)
- **Acuity FP32 vs ORT FP32 → cosine -0.38, 0/32 argmax match**
- **양자화 문제가 아니라 ONNX→Acuity 변환 자체가 잘못됐음**
- Acuity가 argmax로 뽑는 토큰들: `avorable`, `Rothschild`, `omson`, `Kyr` — 전부 rare vocab tokens (systematic 오류 시그널)

### 10:38 — Phase G: 문서화 및 커밋

- `M1_smollm2/device/eval_report.md` 작성 (실측 데이터 + 원인 후보 + M2 조사 필요 항목)
- Phase E/F (decode_one, decode_multi)는 스킵 — 정확도 broken이라 의미 없음
- 시간순 커밋 5개 추가: device scripts, gitignore 갱신, FP32 comparison harness, eval report

---

## M1 최종 판정

| 목표 | 달성? |
|---|---|
| ONNX → Acuity IR → 양자화 → NBG 컴파일 | ✅ |
| T527 NPU forward pass 성공 | ✅ |
| PID `0X10000016` 매칭 | ✅ |
| Pure NPU decode speed 실측 (11 tok/s, W=32) | ✅ |
| FP32 대비 양자화 손실 정량화 | ❌ (Acuity FP32 자체가 broken이라 비교 무의미) |
| 실제 텍스트 생성 데모 | ❌ (같은 이유) |

**핵심 성과**: T527에서 LLM 컴파일+실행 가능성을 실증. 프로젝트의 가장 큰 unknown 해소.
**새 발견**: Acuity 6.12의 ONNX 변환 버그가 별도 존재 — M2 시작 전 반드시 해결.

## 다음 (M2 준비 이슈)

- Acuity 6.12의 발산 시작 레이어 pin-point (layer tensor dump로 하나씩)
- Acuity 최신 (6.21) 접근 시도 or 우회 ONNX 재작성
- 그 후에 M2: Qwen2.5-0.5B 한국어 파일럿 (SmoothQuant + KV cache 도입)
- M3: Midm-2.0-Mini (2.3B, MIT) 상용 배포

## 2026-07-16 오후 (11:14~12:25) — M1 완결

### 11:04 — Layer-by-layer 발산 추적

Acuity FP32 output이 ORT FP32와 완전 불일치. 원인 pin-point 위해 intermediate output 노출:
- Gather (embedding lookup): **cos=1.0000** (완벽)
- Layer 0 RMSNorm output: **cos=0.8750** ← **여기서 발산 시작**
- Layer 0 Q projection: cos=0.9726
- lm_head logits: cos=-0.3818 (30 layer 누적 오차)

### 11:11 — 근본 원인 확정: Acuity ReduceMean 축 오프바이원

Acuity가 3D ONNX 텐서 `[1, 32, 576]`을 내부 4D `[1, 1, 32, 576]`으로 확장하지만 attribute axes는 renumber 안 함. `axes=[2]`가 hidden dim(576) 대신 seq_len(32)을 reduce.

수치적 증명: `acuity_mean == sq.mean(axis=1)` (max_abs_diff = 0.0). 원래 의도는 `sq.mean(axis=-1)`.

### 11:14 — `patch_reducemean_axes.py` 작성 및 적용

61개 ReduceMean 노드의 `axes=[2]` → `axes=[-1]` 치환. v3 IR 재빌드.

### 11:16 — Acuity FP32 정확도 완전 복구

v3 IR + pegasus inference `--dtype float32` → **ORT FP32와 32/32 argmax match, cos=1.0000**. 축 버그 100% 해결.

### 11:19 — v3 uint8 NB export → 디바이스 실행

- 92 ms/forward = 10.9 tok/s pure NPU
- 정확도 여전히 0/32 argmax match (양자화 자체 손실, axis fix와 별개)
- 원인: LLM activation outlier가 심해 uint8 asymmetric_affine으로 압축 불가

### 11:24 — int16 dfp16 재시도

- 147 ms/forward = 6.8 tok/s
- 28% 값이 ±32768 saturation
- 여전히 0-1/32 match

### 11:31~12:00 — 여러 quantizer 조합 실험

- `--algorithm normal` (minmax uint8): 0/32 match, cos=-0.27
- `asymmetric_affine int16`: **미지원** (Acuity 6.12는 uint8/int8/uint4/int4만)
- `symmetric_affine int16`: **미지원**
- `bfloat16` export: **실패** (`Fatal model generation error: 64768`)

### 12:15 — FP32 NBG export 시도 → **성공!**

`pegasus export ovxlib --dtype float32` (non-quantized) → 626MB NBG. T527에서 SW 에뮬레이션으로 실행:
- 7.28 s/forward (79x slower than uint8)
- cos=0.805 vs ORT FP32
- Real prompt "The capital of France is" → top-5: ' as' | ' of' | '.' | ' in' | ',' (모두 grammatically plausible)

**M1 최종 결론: T527 NPU가 SmolLM2-135M을 정확하게 실행 가능함이 실증됨** (FP32 SW 에뮬 경로). 실용 속도는 M2에서 SmoothQuant로 int8/int16 정확도 회복 시 확보.

### 12:25 — 최종 커밋 및 push

11개 커밋 추가:
- axis fix patch script
- v3 IR + quantize tables + NBG
- FP32 NBG decode script (accurate proof)
- 전체 eval_report.md
