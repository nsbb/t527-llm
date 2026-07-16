#!/bin/bash
# Step 7: pegasus export ovxlib (NBG for T527 PID0X10000016)
set -euo pipefail
WORK=/home/nsbb/travail/claude/T527/t527-llm/M1_smollm2
ACUITY612=/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0
VIVANTE57=${VIVANTE57:-/home/nsbb/VeriSilicon/VivanteIDE5.7.2}
NAME=${NAME:-smollm2_135m_w32_v2}
TARGET=${TARGET:-VIP9000NANOSI_PLUS_PID0X10000016}   # T527. A733 는 VIP9000NANODI_PLUS_PID0X1000003B

docker run --rm \
  -v "$WORK:/work" \
  -v "$VIVANTE57:/vivante57:ro" \
  -v "$ACUITY612:/acuity612:ro" \
  -w "/work/acuity_out/$NAME" \
  t527-npu:v1.2 \
  bash -c "
    VSIM=/vivante57/cmdtools/vsimulator
    COMMON=/vivante57/cmdtools/common
    export REAL_GCC=/usr/bin/gcc
    export VIVANTE_VIP_HOME=/vivante57
    export VIVANTE_SDK_DIR=\$VSIM
    export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$VSIM/lib:\$COMMON/lib:\$VSIM/lib/x64_linux:\$VSIM/lib/x64_linux/vsim:\$LD_LIBRARY_PATH
    export EXTRALFLAGS=\"-Wl,--disable-new-dtags -Wl,-rpath,\$VSIM/lib -Wl,-rpath,\$COMMON/lib -Wl,-rpath,\$VSIM/lib/x64_linux -Wl,-rpath,\$VSIM/lib/x64_linux/vsim\"
    cd /acuity612/bin
    ./pegasus export ovxlib \
      --model /work/acuity_out/${NAME}/${NAME}.json \
      --model-data /work/acuity_out/${NAME}/${NAME}.data \
      --dtype quantized \
      --model-quantize /work/acuity_out/${NAME}/${NAME}_uint8.quantize \
      --with-input-meta /work/acuity_out/${NAME}/${NAME}_inputmeta.yml \
      --pack-nbg-unify \
      --optimize ${TARGET} \
      --viv-sdk \$VSIM \
      --target-ide-project linux64 \
      --batch-size 1 \
      --output-path /work/acuity_out/${NAME}/wksp_nbg/ 2>&1 | tail -20
  "
