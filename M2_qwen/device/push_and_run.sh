#!/bin/bash
set -euo pipefail
ADB=${ADB:-/mnt/c/Users/nsbb/AppData/Local/Android/Sdk/platform-tools/adb.exe}
DEV_DIR=/data/local/tmp/qwen_llm
NB=${NB:?set NB path}
HERE=$(cd "$(dirname "$0")" && pwd)
$ADB shell "mkdir -p $DEV_DIR"
$ADB push "$NB" "$DEV_DIR/network_binary.nb"
$ADB push "$HERE/input_0.dat" "$DEV_DIR/input_0.dat"
$ADB push "$HERE/sample.txt" "$DEV_DIR/sample.txt"
$ADB shell "cd $DEV_DIR && LD_LIBRARY_PATH=/vendor/lib64 /data/local/tmp/vpm_run_aarch64 -s sample.txt -l 1 -b 0" 2>&1
$ADB pull "$DEV_DIR/output_0.dat" "$HERE/output_0.dat"
