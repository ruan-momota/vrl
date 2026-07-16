#!/usr/bin/env bash
#SBATCH --job-name=quan-solar

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
MODE="${MODE:-audit}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

CONFIG_DIRS=(
  "configs/runs/ssv2_videomae_linear_probe"
  "configs/runs/ssv2_slowfast_linear_probe"
  "configs/runs/ssv2_dinov2_linear_probe"
  "configs/runs/ucf101_videomae_linear_probe"
  "configs/runs/ucf101_slowfast_linear_probe"
  "configs/runs/ucf101_dinov2_linear_probe"
  "configs/runs/diving48_videomae_linear_probe"
  "configs/runs/diving48_slowfast_linear_probe"
  "configs/runs/diving48_dinov2_linear_probe"
)

SMOKE_DIRS=(
  "configs/runs/diving48_videomae_linear_probe"
  "configs/runs/diving48_slowfast_linear_probe"
  "configs/runs/diving48_dinov2_linear_probe"
)

array_index() {
  local maximum="$1"
  local index="${SLURM_ARRAY_TASK_ID:-${CELL_INDEX:-0}}"
  if (( index < 0 || index >= maximum )); then
    echo "Array index ${index} is outside [0, $((maximum - 1))]" >&2
    exit 1
  fi
  printf '%s\n' "${index}"
}

single_match() {
  local pattern="$1"
  local matches=()
  shopt -s nullglob
  matches=( ${pattern} )
  shopt -u nullglob
  if (( ${#matches[@]} != 1 )); then
    echo "Expected one config for pattern ${pattern}, found ${#matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

extract_config() {
  local config="$1"
  local args=("${PYTHON}" -m src.pipeline.extract --run-config "${config}")
  if [[ "${OVERWRITE:-0}" == "1" ]]; then
    args+=(--overwrite)
  fi
  echo "Running extraction: ${config}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "${args[@]}"
    printf '\n'
    return
  fi
  "${args[@]}"
}

case "${MODE}" in
  test)
    "${PYTHON}" -m pytest -q
    ;;

  audit)
    # CPU/video-decode intensive, but does not load any model or read held-out results.
    "${PYTHON}" scripts/audit_quan_solar.py
    ;;

  smoke)
    index="$(array_index "${#SMOKE_DIRS[@]}")"
    directory="${SMOKE_DIRS[${index}]}"
    extract_config "$(single_match "${directory}/*_smoke_rgb_quantization_mid.json")"
    extract_config "$(single_match "${directory}/*_smoke_solarization_mid.json")"
    ;;

  extract)
    index="$(array_index "${#CONFIG_DIRS[@]}")"
    directory="${CONFIG_DIRS[${index}]}"

    # Mid first, then low/high. Each array task owns one run_id, avoiding manifest races.
    extract_config "$(single_match "${directory}/*_heldout_rgb_quantization_mid.json")"
    extract_config "$(single_match "${directory}/*_heldout_solarization_mid.json")"
    extract_config "$(single_match "${directory}/*_heldout_rgb_quantization_low.json")"
    extract_config "$(single_match "${directory}/*_heldout_solarization_low.json")"
    extract_config "$(single_match "${directory}/*_heldout_rgb_quantization_high.json")"
    extract_config "$(single_match "${directory}/*_heldout_solarization_high.json")"
    ;;

  evaluate)
    index="$(array_index "${#CONFIG_DIRS[@]}")"
    directory="${CONFIG_DIRS[${index}]}"
    evaluation_config="$(single_match "${directory}/*_linear_probe_evaluation.json")"
    echo "Running evaluation: ${evaluation_config}"
    evaluation_args=(
      "${PYTHON}" -m src.pipeline.evaluate --config "${evaluation_config}" --overwrite
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'DRY RUN:'
      printf ' %q' "${evaluation_args[@]}"
      printf '\n'
    else
      "${evaluation_args[@]}"
    fi
    ;;

  summary)
    "${PYTHON}" scripts/build_diving48_3x3_summary.py
    ;;

  *)
    echo "Unsupported MODE=${MODE}; use test, audit, smoke, extract, evaluate, or summary" >&2
    exit 1
    ;;
esac
