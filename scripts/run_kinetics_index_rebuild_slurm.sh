#!/bin/bash
#SBATCH --job-name=vrl-kinetics-index
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -uo pipefail

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
exec >> "index_rebuild.log" 2>&1
echo "=== $(date -u) rebuilding kinetics index with probe-decode on $(hostname) ==="

"${PY}" -m src.data.kinetics_index \
  --video-root data/kinetics/raw/c50 \
  --output-dir data/kinetics/subsets/c50_train100_heldout30 \
  --subset-id c50_train100_heldout30 \
  --max-classes 50 --train-per-class 100 --heldout-per-class 30 --seed 0 --probe-decode

echo "=== $(date -u) done, exit code $? ==="
