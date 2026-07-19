#!/bin/bash
# Run smoke tests + full extraction (14-perturbation matrix) + evaluation for
# one matrix cell. Does not stop the pod and does not touch other cells'
# outputs -- only wipes this cell's own run_id directory before starting, so
# it is safe to call repeatedly for different cells on the same pod.
#
# Usage: run_cell_pipeline.sh <config_dir> <file_prefix>
#   e.g. run_cell_pipeline.sh diving48_dismo_linear_probe diving48_dismo_c32
set -uo pipefail

CONFIG_DIR="$1"
PREFIX="$2"
cd /workspace/vrl || exit 1

PY=/workspace/vrl/.venv/bin/python
LOG="pipeline_${PREFIX}.log"
exec >> "$LOG" 2>&1
echo "=== $(date -u) starting pipeline for $PREFIX ==="

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

RUN_ID=$("$PY" -c "import json; print(json.load(open('configs/runs/${CONFIG_DIR}/${PREFIX}_train_original.json'))['run_id'])") || exit 1
echo "run_id=${RUN_ID}"
rm -rf "outputs/runs/${RUN_ID}" "outputs/runs-smoke/${RUN_ID}-smoke"

run "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))" || exit 1

for smoke in smoke_train smoke_heldout_original smoke_temporal_shuffle_mid smoke_color_mid; do
  run "$PY" -m src.pipeline.extract \
    --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_${smoke}.json" --limit 1 || exit 1
done
echo "=== smoke tests passed for ${PREFIX} ==="

run "$PY" -m src.pipeline.extract \
  --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_train_original.json" || exit 1
run "$PY" -m src.pipeline.extract \
  --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_heldout_original.json" || exit 1

for pert in temporal_shuffle_mid freeze_tail_low freeze_tail_mid freeze_tail_high \
            color_low color_mid color_high blur_mid \
            rgb_quantization_low rgb_quantization_mid rgb_quantization_high \
            solarization_low solarization_mid solarization_high; do
  run "$PY" -m src.pipeline.extract \
    --run-config "configs/runs/${CONFIG_DIR}/${PREFIX}_heldout_${pert}.json" || exit 1
done
echo "=== full extraction complete for ${PREFIX} ==="

run "$PY" -m src.pipeline.evaluate \
  --config "configs/runs/${CONFIG_DIR}/${PREFIX}_linear_probe_evaluation.json" || exit 1
echo "=== evaluation complete, ${PREFIX} succeeded ==="
exit 0
