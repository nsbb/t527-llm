#!/bin/bash
# Push SmolLM2-135M NB + input to T527 and run vpm_run.
# Uses Windows-side adb.exe (WSL). Reproduces Conformer's push/run pattern.
set -euo pipefail

ADB=${ADB:-/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe}
DEV_DIR=/data/local/tmp/smollm2_llm
NB=${NB:-/home/nsbb/travail/claude/T527/t527-llm/M1_smollm2/acuity_out/smollm2_135m_w32_v2/wksp_nbg_nbg_unify/network_binary.nb}
LOOP=${LOOP:-1}
BYPASS=${BYPASS:-0}   # 0=save output+top5, 1=bypass

HERE=$(cd "$(dirname "$0")" && pwd)

echo "=== device: create workdir ==="
$ADB shell "mkdir -p $DEV_DIR"

echo "=== push nb (124MB) ==="
$ADB push "$NB" "$DEV_DIR/network_binary.nb"

echo "=== push input + sample ==="
$ADB push "$HERE/input_0.dat" "$DEV_DIR/input_0.dat"
$ADB push "$HERE/sample.txt" "$DEV_DIR/sample.txt"

echo "=== run vpm_run ==="
$ADB shell "cd $DEV_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -l $LOOP -b $BYPASS --show_top5 1" 2>&1 | tee "$HERE/vpm_run.log"

echo "=== pull output_0.dat ==="
$ADB pull "$DEV_DIR/output_0.dat" "$HERE/output_0.dat" 2>&1 || echo "(output pull failed — check vpm_run.log)"
ls -lh "$HERE/output_0.dat" 2>/dev/null || true
