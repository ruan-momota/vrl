#!/bin/bash
# Run smoke tests + full extraction + evaluation for one Kinetics model cell,
# then stop this pod via the RunPod API regardless of outcome.
#
# Usage: run_kinetics_extraction_pipeline.sh <config_dir> <file_prefix> <pod_id>
#   e.g. run_kinetics_extraction_pipeline.sh kinetics_vjepa_linear_probe kinetics_vjepa_c50 6oacyl2x0oykjz
#
# Expects RUNPOD_API_KEY in the environment (used only at the very end, to
# self-stop; never written to a config file until the trap actually fires).
set -uo pipefail

# num_workers=8 (see the Kinetics run configs) opens far more file
# descriptors/shared-memory handles than PyTorch's default DataLoader
# sharing strategy expects at the container's default ulimit -- raise it
# here rather than lower num_workers, since the slowdown from num_workers=0
# is the thing that made this multi-model run worth parallelizing at all.
ulimit -n 65536 2>/dev/null || true

# Everything outside /workspace (the persistent volume) is reset on every
# pod stop/start cycle, including /root/.cache -- redirect torch.hub's and
# Hugging Face's model caches onto the volume so a mid-download interruption
# (or a capacity-forced restart) doesn't force re-downloading multi-GB
# checkpoints from scratch.
export TORCH_HOME=/workspace/vrl/.torch_cache
export HF_HOME=/workspace/vrl/.hf_cache

CONFIG_DIR="$1"
PREFIX="$2"
POD_ID="$3"
cd /workspace/vrl || exit 1

LOG="pipeline_${PREFIX}.log"
exec >> "$LOG" 2>&1
echo "=== $(date -u) starting pipeline for $PREFIX (pod $POD_ID) ==="
echo "runpodctl version: $(runpodctl version 2>&1)"

# A prior crashed attempt on this same pod can leave partial artifacts behind,
# and src.pipeline.extract refuses to silently overwrite an existing output
# (by design). This pod only ever runs one model's pipeline, so a full wipe
# of both output roots at the start of every attempt is safe and makes every
# run (including retries) start from a clean slate.
echo "clearing any stale outputs from a previous attempt"
rm -rf outputs/runs-smoke outputs/runs

# runpodctl's subcommand syntax has changed across versions (noun-first
# "pod stop <id>" on newer CLIs, verb-first "stop pod <id>" on older ones
# bundled in some pod images) -- try both rather than assume, since a
# self-stop that silently no-ops is exactly the failure mode this trap
# exists to prevent.
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
  status=$?
  echo "=== $(date -u) pipeline exiting with status $status, stopping pod $POD_ID ==="
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    robust_pod_stop "$POD_ID"
  else
    echo "no RUNPOD_API_KEY set, cannot self-stop -- pod will keep billing until manually stopped" >&2
  fi
}
trap cleanup EXIT

run() {
  echo "--- $(date -u) $* ---"
  "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! command failed (exit $rc): $* !!!"
    exit $rc
  fi
}

PY=/workspace/vrl/.venv/bin/python

run "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

for smoke in smoke_train smoke_heldout_original smoke_temporal_shuffle_mid smoke_color_mid; do
  run "$PY" -m src.pipeline.extract \
    --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_${smoke}.json" --limit 1
done
echo "=== smoke tests passed ==="

run "$PY" -m src.pipeline.extract \
  --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_train_original.json"
run "$PY" -m src.pipeline.extract \
  --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_heldout_original.json"

for pert in temporal_shuffle_mid freeze_tail_low freeze_tail_mid freeze_tail_high \
            color_low color_mid color_high spatial_blur_mid; do
  run "$PY" -m src.pipeline.extract \
    --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_heldout_${pert}.json"
done
echo "=== full extraction complete ==="

run "$PY" -m src.pipeline.evaluate \
  --config "configs/runs/${CONFIG_DIR}/${PREFIX}_linear_probe_evaluation.json"
echo "=== evaluation complete, pipeline succeeded ==="
