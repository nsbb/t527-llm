# t527-llm

Allwinner T527 SoC (Vivante VIP9000-NanoSI-Plus NPU, `PID0X10000016`, ~2 TOPS) 위에 **한국어 LLM**을 올리기 위한 워크스페이스.

**현재 상태 (2026-07-16)**: M1 (SmolLM2-135M 컴파일 실증) 완료. 다음 M2 (한국어 파일럿 — Qwen2.5-0.5B) 준비 중.

---

## 폴더 구조

```
t527-llm/
├── README.md                 # 이 파일
├── PLAN.md                   # 전체 계획서 (5개 리서치 종합)
├── PROGRESS.md               # 시간순 작업 로그
│
├── research_*.md             # 5개 조사 리포트 (2026-07-15)
│   ├── research_allwinner.md
│   ├── research_verisilicon.md
│   ├── research_community.md
│   ├── research_reference_npus.md
│   └── research_korean_llm_quantization.md
│
├── M0_op_validation/         # M0: Acuity 6.12 op 지원 실측
│   ├── M0_op_support_matrix.md
│   ├── make_dummy_onnx.py
│   ├── make_workarounds.py
│   ├── batch_import.sh
│   └── results.csv           # 33개 op 실측 결과
│
└── M1_smollm2/               # M1: SmolLM2-135M → T527 NB 컴파일 (완료)
    ├── README.md             # M1 전체 문서
    ├── patch_onnx_last_slice.py
    ├── run_pegasus_import.sh
    ├── run_pegasus_quantize.sh
    ├── run_pegasus_export.sh
    └── nbg_meta.json         # NBG I/O 스펙
```

## 마일스톤

| 단계 | 상태 | 결과 |
|---|---|---|
| 리서치 (5 리포트) | ✅ 완료 | 공개된 T527 LLM 포팅 0건. 참고: petayyyy/a733_npu_driver (A733 SmolLM2 21 tok/s) |
| PLAN.md 종합 | ✅ 완료 | 4단계 검증 사다리 (M0→M1→M2→M3) 확정 |
| **M0**: Acuity 6.12 op 지원 실측 | ✅ 완료 | 33/33 op 커버 (native 30 + 조립 3) |
| **M1**: SmolLM2-135M → T527 NB | ✅ **컴파일 + 디바이스 실행 성공** | uint8 NB 124MB @ 10.9 tok/s, FP32 NB 626MB @ 0.14 tok/s |
| **M1**: Acuity 6.12 ReduceMean 축 버그 발견 & 수정 | ✅ **M1 대박 finding** | 61 ReduceMean axes=[2]→[-1], FP32 32/32 match 복구 |
| **M1**: FP32 정확도 검증 | ✅ | ORT FP32 vs Acuity FP32 argmax 32/32 match, T527 NPU FP32 real prompt에서 coherent tokens |
| M1: uint8/int16 양자화 정확도 | ⚠️ | 양자화 손실 심각 (LLM outlier activation), M2에서 SmoothQuant로 해결 |
| **M2**: Qwen2.5-0.5B 한국어 파일럿 | ⏳ | SmoothQuant + KV cache 전략 도입 |
| **M3**: Midm-2.0-Mini (2.3B, MIT) | ⏳ | 상용 배포 가능한 한국어 LLM |

## 하드웨어 & 툴체인 (고정 상수)

| 항목 | 값 |
|---|---|
| SoC | Allwinner T527 |
| NPU | Vivante VIP9000-NanoSI-Plus |
| Optimize PID | `VIP9000NANOSI_PLUS_PID0X10000016` |
| Acuity Toolkit | 6.12.0 binary |
| VivanteIDE | 5.7.2 |
| Docker image | `t527-npu:v1.2` |
| 양자화 (M1 검증) | `asymmetric_affine uint8` + `kl_divergence` (Conformer 레시피 계승) |

## 참고

- 팀의 Conformer/CitriNet 파이프라인 문서: `../t527-stt/`
- petayyyy 참고 리포: [github.com/petayyyy/a733_npu_driver](https://github.com/petayyyy/a733_npu_driver) — A733(T527과 동일 NPU 계열)에서 SmolLM2 실증
- Allwinner Acuity 6.21 op 지원 PDF: `../docs/acuity_toolkit/NPU模块开发指南/NPU_算子支持列表.pdf`
