#!/usr/bin/env bash
# Run only inside a separately granted Intel GPU scheduling window.
# This lock prevents overlap; acquiring it is not a scheduling grant.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 BUILD_DIRECTORY tiny|golden|rpc [MODEL_DIRECTORY]" >&2
  exit 64
fi
build=$(realpath "$1")
mode=$2
repo=$(cd "$(dirname "$0")/.." && pwd)
: "${UOCR_INTEL_TEST_LOCK:?set the shared GPU scheduling lock path}"
exec 9>"$UOCR_INTEL_TEST_LOCK"
flock -n 9 || { echo 'Intel GPU scheduling lock busy' >&2; exit 75; }
export UOCR_REQUIRE_INTEL=1
date -u +%FT%TZ
git -C "$repo" rev-parse HEAD
uptime
case "$mode" in
  tiny)
    unset UOCR_MODELS_DIR
    for test in intel_numeric_guard_test intel_ops_test intel_native_ops_test intel_native_vision_test intel_native_decoder_test; do
      timeout --kill-after=10s 300s "$build/$test"
    done
    ;;
  golden|rpc)
    [[ $# -eq 3 ]] || { echo 'model directory required' >&2; exit 64; }
    export UOCR_MODELS_DIR
    UOCR_MODELS_DIR=$(realpath "$3")
    (cd "$UOCR_MODELS_DIR" && awk '!/^#/ && NF>=3 {print $2 "  " $1}' "$repo/models/MANIFEST" | sha256sum -c -)
    "$build/intel_memory_check"
    test=intel_native_golden_test
    if [[ "$mode" == rpc ]]; then
      export GRPC_UOCR_BACKEND=intel-gpu GRPC_UOCR_WEIGHT_FORMAT=bf16 GRPC_UOCR_DEVICE_MEMORY_CAP=12884901888
      sha256sum "$build/server_model_stream_test"
      timeout --kill-after=10s 1800s nice -n10 "$build/server_model_stream_test"
      "$build/intel_memory_check"
      test=intel_native_rpc_test
    fi
    sha256sum "$build/$test"
    timeout --kill-after=10s 1800s nice -n10 "$build/$test"
    ;;
  *) echo 'unknown qualification mode' >&2; exit 64;;
esac
