# M1: SmolLM2-135M-Instruct → T527 NPU

**목표**: petayyyy/a733_npu_driver의 A733 실증(SmolLM2-135M @ 21 tok/s)을 T527 NPU (`PID0X10000016`)용으로 재빌드해 **T527에서 LLM이 실제로 컴파일되는지** 실증.

**결과**: **성공.** `network_binary.nb` (124MB) 생성 완료. Acuity 6.12 파이프라인이 30-layer Llama 아키텍처 전체를 컴파일할 수 있음을 확인.

---

## 최종 산출물

```
acuity_out/smollm2_135m_w32_v2/
├── smollm2_135m_w32_v2.json           # Acuity IR (1.1MB)
├── smollm2_135m_w32_v2.data           # FP32 weights (622MB) [.gitignore]
├── smollm2_135m_w32_v2_uint8.quantize # 양자화 테이블
├── smollm2_135m_w32_v2_inputmeta.yml
├── token_ids.npy, token_ids.raw       # calibration 샘플
├── dataset_tokens.txt                  # calibration 리스트
└── wksp_nbg_nbg_unify/
    ├── network_binary.nb              # ★ T527용 NBG (124MB) [.gitignore]
    ├── nbg_meta.json                  # I/O 스펙 (shape/scale/zp)
    ├── vnn_*.c/h, main.c, makefile.linux
    └── BUILD, .cproject, .project
```

---

## 모델 스펙 (SmolLM2-135M-Instruct)

| 속성 | 값 |
|---|---|
| 아키텍처 | Llama (30 layers) |
| hidden_size | 576 |
| num_attention_heads | 9 (GQA 9Q / 3KV) |
| head_dim | 64 |
| intermediate_size | 1536 |
| vocab_size | 49152 |
| rms_norm_eps | 1e-5 |
| RoPE θ | 100000 |
| tie_word_embeddings | true |
| dtype | bf16 (원본), uint8 (양자화 후) |

**Static ONNX shape**: input `token_ids [1, 32]` int32 → output `logits [1, 32, 49152]` fp32

---

## 실행 파이프라인 (성공한 최종 레시피)

### Step 1: HF 체크포인트 다운로드

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM2-135M-Instruct',
    local_dir='work/models/smollm2-135m-instruct',
    allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
                    'generation_config.json','model.safetensors'])
"
```

### Step 2: Static ONNX 생성 (petayyyy 스크립트 재사용)

```bash
python3 scripts/host/make_real_llm_onnx.py \
  --model-dir work/models/smollm2-135m-instruct \
  --output-dir work/generated/smollm2_135m_w32 \
  --seq-len 32
```

- 출력: `real_llm.onnx` (514MB), `token_ids.npy`, `model_info.json`
- 특징: RMSNorm/SwiGLU/RoPE 전부 primitive op(Pow/ReduceMean/Sqrt/Reciprocal/Mul/Add/Sin·Cos 상수 테이블/Sigmoid+Mul)로 이미 분해되어 있음
- Cos native op를 사용하지 않고 sin/cos 값을 **상수 텐서로 미리 계산**해서 그래프에 임베딩 — Acuity 6.12의 Cos 미지원 문제 회피
- LayerNormalization/Gelu native op 사용 안 함 — Acuity 6.12의 opset 17/20 fused op 미지원 문제 회피

### Step 3: ONNX 패치 — last-token slice 제거

Acuity 6.12의 ONNX→TF converter 버그: `Slice_slice_last_hidden`의 size 계산이 `[1, -30, 32, 576]`로 잘못 되어 `Invalid value in tensor used for shape: -30` 에러 발생.

해결: ONNX에서 `slice_last_hidden` 노드 제거, `final_last_token` 참조를 `final_rms_out`로 리와이어. logits shape `[1, 1, V]` → `[1, 32, V]`. 마지막 토큰 로그잇 추출은 CPU에서 `logits[:, -1, :]`로 처리 (NPU에선 낭비되지만 컴파일 통과가 우선).

```python
# patch_onnx_last_slice.py 참고
target = next(n for n in m.graph.node if n.name == 'slice_last_hidden')
src, dst = target.input[0], target.output[0]
for other in m.graph.node:
    for k, inp in enumerate(other.input):
        if inp == dst: other.input[k] = src
m.graph.node.remove(target)
# logits output shape을 [1, 32, 49152]로 갱신
```

### Step 4: Acuity import

```bash
docker run --rm -v $WORK:/work -v $ACUITY612:/acuity612:ro \
  -w /work/acuity_out/$NAME t527-npu:v1.2 bash -c "
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
    /acuity612/bin/pegasus import onnx \
      --model /work/work/generated/smollm2_135m_w32/real_llm_nolastslice.onnx \
      --output-model $NAME.json --output-data $NAME.data \
      --inputs token_ids --input-size-list 1,32 --input-dtype-list int32 \
      --outputs logits
  "
```

결과: `Error(0)`, Acuity IR 2454 layers 생성. Sigmoid×Mul 패턴이 자동으로 `swish`(SiLU) 30개로 융합됨.

### Step 5: inputmeta 자동 생성 + 수정

```bash
$PEG generate inputmeta --model $NAME.json --input-meta-output ${NAME}_inputmeta.yml
```

**수정 필요 항목:**
- `path: dataset.txt` → `path: dataset_tokens.txt`
- `type: NPY`가 있으면 → `type: TEXT` (Conformer 스타일 dataset listing 방식)
- `category: image`(default) → `category: undefined`
- **lid는 유지** (Acuity 자동 부여, 예: `token_ids_2334`)

`dataset_tokens.txt` 내용:
```
token_ids.npy
```

### Step 6: Quantize (uint8 asymmetric_affine, Conformer 레시피)

```bash
$PEG quantize \
  --model $NAME.json --model-data $NAME.data \
  --device CPU \
  --with-input-meta ${NAME}_inputmeta.yml \
  --rebuild \
  --model-quantize ${NAME}_uint8.quantize \
  --quantizer asymmetric_affine \
  --qtype uint8 \
  --algorithm kl_divergence
```

결과: `Error(0), Warning(62)`. 62개 warnings는 대형 그래프에서 KL 캘리브레이션 관련 정보 warning.

**시도했으나 실패한 조합:**
- `perchannel_symmetric_affine int16` → Acuity 6.12는 이 quantizer가 int8/int4만 지원 (petayyyy는 최신 Acuity 사용)
- `perchannel_symmetric_affine int8` → 동일 slice bug로 실패 (ONNX 패치 전)

### Step 7: NBG export (T527 PID)

```bash
docker run --rm -v $WORK:/work -v $VIVANTE57:/vivante57:ro -v $ACUITY612:/acuity612:ro \
  -w /work/acuity_out/$NAME t527-npu:v1.2 bash -c '
    VSIM=/vivante57/cmdtools/vsimulator
    COMMON=/vivante57/cmdtools/common
    export REAL_GCC=/usr/bin/gcc
    export VIVANTE_VIP_HOME=/vivante57
    export VIVANTE_SDK_DIR=$VSIM
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:$VSIM/lib:$COMMON/lib:$VSIM/lib/x64_linux:$VSIM/lib/x64_linux/vsim:$LD_LIBRARY_PATH
    export EXTRALFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$VSIM/lib -Wl,-rpath,$COMMON/lib -Wl,-rpath,$VSIM/lib/x64_linux -Wl,-rpath,$VSIM/lib/x64_linux/vsim"
    cd /acuity612/bin
    ./pegasus export ovxlib \
      --model /work/acuity_out/$NAME/$NAME.json \
      --model-data /work/acuity_out/$NAME/$NAME.data \
      --dtype quantized \
      --model-quantize /work/acuity_out/$NAME/${NAME}_uint8.quantize \
      --with-input-meta /work/acuity_out/$NAME/${NAME}_inputmeta.yml \
      --pack-nbg-unify \
      --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
      --viv-sdk $VSIM --target-ide-project linux64 --batch-size 1 \
      --output-path /work/acuity_out/$NAME/wksp_nbg/
  '
```

결과: `Error(0), Warning(0)`. `network_binary.nb` 124MB 생성.

---

## Acuity IR op 분포 (2454 layers)

| Op | 개수 | 의미 |
|---|---|---|
| reshape | 662 | 텐서 변형 |
| multiply | 363 | Mul (scaling, gating) |
| permute | 272 | Transpose |
| add | 211 | 잔차 연결 등 |
| fullconnect | 211 | MatMul + bias fuse |
| variable | 128 | weight/bias tensors |
| slice | 121 | RoPE rotate_half, GQA head split |
| divide | 61 | RMSNorm (1/x from Reciprocal) |
| sqrt | 61 | RMSNorm |
| reducemean | 61 | RMSNorm |
| matmul | 60 | Attention Q·K, S·V |
| tile | 60 | GQA head repeat (3x for 9/3) |
| concat | 60 | RoPE rotate concat |
| neg | 60 | RoPE rotate_half |
| softmax | 30 | Attention (30 layers) |
| **swish** | **30** | **Sigmoid×Mul → SiLU 자동 융합** |
| gather | 1 | Token embedding |
| output/input | 1/1 | I/O |

---

## 함정 총정리 (재발 방지)

| # | 함정 | 원인 | 해결 |
|---|---|---|---|
| 1 | `perchannel_symmetric_affine int16` 안 됨 | Acuity 6.12는 이 quantizer가 int8/int4만 지원 | `asymmetric_affine uint8` (Conformer 레시피) |
| 2 | `Invalid value in tensor used for shape: -30` | `slice_last_hidden` ONNX→Acuity 변환 시 size 잘못 계산 (`[1, -30, 32, 576]`) | ONNX에서 last-token slice 제거, logits `[1, seq, V]` 유지, CPU 후처리 |
| 3 | `Cannot load file containing pickled data` | inputmeta `type: NPY` 로더 문제 | `type: TEXT` + dataset.txt에 npy 파일명 리스트 (Conformer/hubert 스타일) |
| 4 | `Unable to decode bytes as JPEG/PNG/GIF/BMP` | `pegasus generate inputmeta` 기본값 `category: image` | `category: undefined`로 수정 |
| 5 | `Network doesn't have a valid input meta, exit.` | lid가 ONNX input name(`token_ids`) 이면 안 됨 | `pegasus generate inputmeta`로 실제 Acuity input lid(`token_ids_2334`) 사용 |
| 6 | `pegasus import` 후 `TypeError: NoneType`은 Cos op | Acuity 6.12 Cos op converter 미구현 | ONNX 생성 시 Sin/Cos를 상수 테이블로 pre-compute (petayyyy 스크립트가 이미 이렇게 함) |
| 7 | Docker `exit 137` 후 파일 생성 여부 확인 안 하면 오해 | tail 파이프가 SIGPIPE로 죽어도 pegasus는 이미 성공한 상태일 수 있음 | `ls`로 산출물 파일 존재 확인 |

---

## 재현 스크립트

파이프라인 전체를 자동화한 스크립트:
- `run_pegasus_import.sh` — Step 4 (ONNX → Acuity IR)
- `run_pegasus_quantize.sh` — Step 6 (uint8 KL divergence)
- `run_pegasus_export.sh` — Step 7 (NBG for T527)

```bash
bash run_pegasus_import.sh    # (원본 ONNX용, 실제로는 실패함 — 패치 필요)
python3 patch_onnx_last_slice.py  # last-token slice 제거
bash run_pegasus_import.sh    # 패치된 ONNX 재import
bash run_pegasus_quantize.sh
bash run_pegasus_export.sh
```

---

## 다음 단계 (M1 마무리)

- [ ] `network_binary.nb` + `nbg_meta.json`을 T527 디바이스에 push
- [ ] `vpm_run` 으로 forward 성공 확인 (output_0.dat 생성)
- [ ] FP32 (ONNX Runtime) vs uint8 NB logits KL divergence 측정 — 정확도 손실 정량화
- [ ] petayyyy의 `npu_lm_runner` 이식 or JNI/awnn 래퍼로 Android 앱에서 W=32 sliding window decode
- [ ] tok/s 실측 (135M 목표: ≥ 10 tok/s, A733는 20.7 tok/s)

## M2로의 시사점 (한국어 LLM)

- **파이프라인 배관 검증됨** — Qwen2.5-0.5B / Midm-2.0-Mini 그대로 이 레시피 적용 가능
- **SmoothQuant 추가 필요**: SmolLM2는 영어 학습이라 outlier 심하지 않을 것. 실제 한국어 LLM (Qwen/Midm)에서 outlier 튀면 SmoothQuant 사전 삽입 필요
- **KV-cache**: 현재 W=32 sliding-window (no KV cache). Qwen 계열은 KV cache 있으면 훨씬 빨라짐 — 하지만 static shape NB에서는 재계산 방식이 안전
