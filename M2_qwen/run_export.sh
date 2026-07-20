#!/bin/bash
set -euo pipefail
WORK=/home/nsbb/travail/claude/T527/t527-llm/M2_qwen
NAME=${NAME:-qwen25_0_5b_w32}
QUANT=${QUANT:-uint8}
VIVANTE57=/home/nsbb/VeriSilicon/VivanteIDE5.7.2
ACUITY612=/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0
docker run --rm -v "$WORK:/work" -v "$VIVANTE57:/vivante57:ro" -v "$ACUITY612:/acuity612:ro" -w "/work/acuity_out/$NAME" t527-npu:v1.2 bash -c "
  VSIM=/vivante57/cmdtools/vsimulator
  COMMON=/vivante57/cmdtools/common
  export REAL_GCC=/usr/bin/gcc VIVANTE_VIP_HOME=/vivante57 VIVANTE_SDK_DIR=\$VSIM
  export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$VSIM/lib:\$COMMON/lib:\$VSIM/lib/x64_linux:\$VSIM/lib/x64_linux/vsim:\$LD_LIBRARY_PATH
  export EXTRALFLAGS=\"-Wl,--disable-new-dtags -Wl,-rpath,\$VSIM/lib -Wl,-rpath,\$COMMON/lib -Wl,-rpath,\$VSIM/lib/x64_linux -Wl,-rpath,\$VSIM/lib/x64_linux/vsim\"
  cd /acuity612/bin
  ./pegasus export ovxlib \
    --model /work/acuity_out/${NAME}/${NAME}.json --model-data /work/acuity_out/${NAME}/${NAME}.data \
    --dtype quantized --model-quantize /work/acuity_out/${NAME}/${NAME}_${QUANT}.quantize \
    --with-input-meta /work/acuity_out/${NAME}/${NAME}_inputmeta.yml \
    --pack-nbg-unify --optimize VIP9000NANOSI_PLUS_PID0X10000016 \
    --viv-sdk \$VSIM --target-ide-project linux64 --batch-size 1 \
    --output-path /work/acuity_out/${NAME}/wksp_nbg_${QUANT}/ > /work/acuity_out/${NAME}/export_${QUANT}.log 2>&1
  echo exit=\$?
"
tail -3 $WORK/acuity_out/$NAME/export_${QUANT}.log
ls -lh $WORK/acuity_out/$NAME/wksp_nbg_${QUANT}_nbg_unify/network_binary.nb 2>&1
