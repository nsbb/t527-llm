#!/bin/bash
# Step 6: pegasus quantize (uint8 asymmetric_affine, Conformer 레시피)
# 사전 준비: inputmeta.yml, dataset_tokens.txt, token_ids.npy 가 $OUTDIR에 있어야 함
set -euo pipefail
WORK=/home/nsbb/travail/claude/T527/t527-llm/M1_smollm2
ACUITY612=/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0
NAME=${NAME:-smollm2_135m_w32_v2}
OUTDIR="$WORK/acuity_out/$NAME"

docker run --rm \
  -v "$WORK:/work" \
  -v "$ACUITY612:/acuity612:ro" \
  -w "/work/acuity_out/$NAME" \
  t527-npu:v1.2 \
  bash -c "
    PEG=/acuity612/bin/pegasus
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
    \$PEG quantize \
      --model ${NAME}.json \
      --model-data ${NAME}.data \
      --device CPU \
      --with-input-meta ${NAME}_inputmeta.yml \
      --rebuild \
      --model-quantize ${NAME}_uint8.quantize \
      --quantizer asymmetric_affine \
      --qtype uint8 \
      --algorithm kl_divergence 2>&1 | tail -20
  "
