#!/bin/bash
# M0: 각 dummy ONNX를 pegasus import에 태워 지원 여부 확인
# 결과: import_results/<op>.log, results.csv

set -u
WORK=/home/nsbb/travail/claude/T527/t527-llm/M0_op_validation
ACUITY612=/home/nsbb/travail/T527/acuity-toolkit-binary-6.12.0

RESULTS_DIR=$WORK/import_results
mkdir -p "$RESULTS_DIR"

CSV=$WORK/results.csv
echo "op,status,error_snippet" > "$CSV"

ONNX_LIST=$(ls "$WORK/onnx_dummies"/*.onnx | sort)

for onnx in $ONNX_LIST; do
    name=$(basename "$onnx" .onnx)
    log="$RESULTS_DIR/${name}.log"
    echo "===== $name =====" | tee -a "$log"

    docker run --rm \
      -v "$WORK:/work" \
      -v "$ACUITY612:/acuity612:ro" \
      t527-npu:v1.2 \
      bash -c "
        PEG=/acuity612/bin/pegasus
        export LD_LIBRARY_PATH=/acuity612/bin/lib:/acuity612/bin/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH
        cd /work/import_results
        rm -f ${name}.json ${name}.data
        \$PEG import onnx --model /work/onnx_dummies/${name}.onnx --output-model ${name}.json --output-data ${name}.data 2>&1
      " > "$log" 2>&1

    if [ -f "$RESULTS_DIR/${name}.json" ] && [ -f "$RESULTS_DIR/${name}.data" ]; then
        status="OK"
        err=""
    else
        status="FAIL"
        # 마지막 에러 라인 뽑기
        err=$(grep -iE "(error|not supported|unsupported|failed|traceback|exception)" "$log" | tail -2 | tr '\n' '|' | sed 's/,/;/g' | cut -c1-200)
    fi

    echo "$name,$status,\"$err\"" >> "$CSV"
    echo "  → $status" | tee -a "$log"
done

echo ""
echo "======================================================"
echo "SUMMARY"
echo "======================================================"
column -t -s',' "$CSV"
