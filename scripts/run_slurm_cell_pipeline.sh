#!/bin/bash
#SBATCH --job-name=vrl-cell
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Run smoke tests + full extraction (14-perturbation matrix) + evaluation for
# one matrix cell on the LMU IfI CIP-pool SLURM cluster. Unlike the RunPod
# variant, there is no pod to self-stop -- the job just exits when done.
#
# Usage: sbatch --partition=NvidiaAll scripts/run_slurm_cell_pipeline.sh <config_dir> <file_prefix>
#   e.g. sbatch --partition=NvidiaAll scripts/run_slurm_cell_pipeline.sh \
#          hmdb51_dinov2_linear_probe hmdb51_dinov2_full_split1
set -uo pipefail
ulimit -n 65536 2>/dev/null || true

if [[ -n "${VRL_ROOT:-}" ]]; then
  ROOT_DIR="${VRL_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
cd "${ROOT_DIR}" || exit 1

PY="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
CONFIG_DIR="$1"
PREFIX="$2"

LOG="pipeline_${PREFIX}.log"
exec >> "${LOG}" 2>&1
echo "=== $(date -u) starting pipeline for ${PREFIX} on $(hostname) ==="

run() {
  echo "--- $(date -u) $* ---"
  "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! command failed (exit $rc): $* !!!"
    return $rc
  fi
  return 0
}

RUN_ID=$("${PY}" -c "import json; print(json.load(open('configs/runs/${CONFIG_DIR}/${PREFIX}_train_original.json'))['run_id'])") || exit 1
echo "run_id=${RUN_ID}"
rm -rf "outputs/runs/${RUN_ID}" "outputs/runs-smoke/${RUN_ID}-smoke"

run "${PY}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

for smoke in smoke_train smoke_heldout_original smoke_temporal_shuffle_mid smoke_color_mid; do
  run "${PY}" -m src.pipeline.extract \
    --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_${smoke}.json" --limit 1 || exit 1
done
echo "=== smoke tests passed for ${PREFIX} ==="

run "${PY}" -m src.pipeline.extract \
  --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_train_original.json" || exit 1
run "${PY}" -m src.pipeline.extract \
  --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_heldout_original.json" || exit 1

for cfg in "configs/runs/${CONFIG_DIR}/${PREFIX}_heldout_"*.json; do
  case "$cfg" in
    *_heldout_original.json) continue ;;
  esac
  run "${PY}" -m src.pipeline.extract --run-config "$cfg" || exit 1
done
echo "=== full extraction complete for ${PREFIX} ==="

run "${PY}" -m src.pipeline.evaluate \
  --config "configs/runs/${CONFIG_DIR}/${PREFIX}_linear_probe_evaluation.json" || exit 1
echo "=== evaluation complete, ${PREFIX} succeeded ==="
exit 0
