#!/bin/bash
# Run several matrix cells sequentially on one pod, then self-stop the pod
# regardless of per-cell outcome. One cell's failure does not abort the
# remaining cells -- each is independent (see run_cell_pipeline.sh).
#
# Usage: run_pod_matrix_pipeline.sh <pod_id> "<config_dir1>:<prefix1>" ["<config_dir2>:<prefix2>" ...]
#   e.g. run_pod_matrix_pipeline.sh 6oacyl2x0oykjz \
#          "diving48_dismo_linear_probe:diving48_dismo_c32" \
#          "diving48_vjepa_linear_probe:diving48_vjepa_c32" \
#          "hmdb51_dismo_linear_probe:hmdb51_dismo_full_split1"
#
# Expects RUNPOD_API_KEY in the environment (used only at the end, to self-stop).
set -uo pipefail
ulimit -n 65536 2>/dev/null || true

export TORCH_HOME=/workspace/vrl/.torch_cache
export HF_HOME=/workspace/vrl/.hf_cache

POD_ID="$1"
shift
cd /workspace/vrl || exit 1

LOG="pod_matrix_pipeline.log"
exec >> "$LOG" 2>&1
echo "=== $(date -u) starting multi-cell pipeline on pod ${POD_ID}: $* ==="

robust_pod_stop() {
  local pod_id="$1"
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
  if ! runpodctl pod stop "$pod_id" >> "$LOG" 2>&1; then
    echo "'runpodctl pod stop' failed, trying legacy 'runpodctl stop pod' syntax" >> "$LOG"
    runpodctl stop pod "$pod_id" >> "$LOG" 2>&1
  fi
  rm -f /root/.runpod/config.toml
}

cleanup() {
  echo "=== $(date -u) all cells attempted, stopping pod ${POD_ID} ==="
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    robust_pod_stop "$POD_ID"
  else
    echo "no RUNPOD_API_KEY set, cannot self-stop -- pod will keep billing until manually stopped" >&2
  fi
}
trap cleanup EXIT

FAILED=()
for spec in "$@"; do
  config_dir="${spec%%:*}"
  prefix="${spec##*:}"
  echo "=== $(date -u) launching cell ${spec} ==="
  bash scripts/run_cell_pipeline.sh "$config_dir" "$prefix"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! cell ${spec} FAILED (exit ${rc}), continuing with remaining cells !!!"
    FAILED+=("$spec")
  else
    echo "=== cell ${spec} succeeded ==="
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "=== $(date -u) finished with failures: ${FAILED[*]} ==="
else
  echo "=== $(date -u) all cells succeeded ==="
fi
