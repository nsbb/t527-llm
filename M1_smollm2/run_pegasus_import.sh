#!/bin/bash
# Step 4: pegasus import (ONNX → Acuity IR)
# 입력 ONNX는 patch_onnx_last_slice.py 로 slice_last_hidden 제거된 파일을 권장.
set -euo pipefail
WORK=/home/nsbb/travail/claude/T527/t527-llm/M1_smollm2
ACUITY612=/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0
NAME=${NAME:-smollm2_135m_w32_v2}
ONNX_FILE=${ONNX_FILE:-work/generated/smollm2_135m_w32/real_llm_nolastslice.onnx}

mkdir -p "$WORK/acuity_out/$NAME"

docker run --rm \
  -v "$WORK:/work" \
  -v "$ACUITY612:/acuity612:ro" \
  -w "/work/acuity_out/$NAME" \
  t527-npu:v1.2 \
  bash -c "
    PEG=/acuity612/bin/pegasus
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
    \$PEG import onnx \
      --model /work/${ONNX_FILE} \
      --output-model ${NAME}.json \
      --output-data  ${NAME}.data \
      --inputs token_ids \
      --input-size-list 1,32 \
      --input-dtype-list int32 \
      --outputs logits 2>&1 | tail -20
  "
