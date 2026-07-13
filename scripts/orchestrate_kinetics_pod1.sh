#!/bin/bash
# Runs entirely on pod1 (the V-JEPA2 pod), independent of the local machine.
# 1. Waits for the already-running Kinetics fetch to finish.
# 2. Builds the subset index.
# 3. Spins up two more pods (VideoMAE, DisMo), ships them the code + dataset,
#    and launches their extraction pipelines (each self-stops on completion).
# 4. Runs this pod's own V-JEPA2 pipeline via run_kinetics_extraction_pipeline.sh,
#    which self-stops this pod on completion (success or failure).
#
# Expects RUNPOD_API_KEY and RUNPOD_SSH_KEY_PATH in the environment.
set -uo pipefail
cd /workspace/vrl || exit 1

POD1_ID="6oacyl2x0oykjz"
LOG="orchestrate.log"
exec >> "$LOG" 2>&1
echo "=== $(date -u) orchestrator starting ==="
echo "runpodctl version: $(runpodctl version 2>&1)"

# runpodctl's subcommand syntax has changed across versions (noun-first
# "pod stop <id>" vs. older verb-first "stop pod <id>") -- try both.
robust_pod_stop() {
  local pod_id="$1"
  runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1
  if ! runpodctl pod stop "$pod_id" >> "$LOG" 2>&1; then
    echo "'runpodctl pod stop' failed, trying legacy 'runpodctl stop pod' syntax" >> "$LOG"
    runpodctl stop pod "$pod_id" >> "$LOG" 2>&1
  fi
  rm -f /root/.runpod/config.toml
}

self_stop_on_failure() {
  status=$?
  if [ $status -ne 0 ]; then
    echo "=== $(date -u) orchestration failed (status $status), self-stopping pod1 ==="
    robust_pod_stop "$POD1_ID"
  fi
}
trap self_stop_on_failure EXIT

run() {
  echo "--- $(date -u) $* ---"
  "$@"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! command failed (exit $rc): $* !!!"
    exit $rc
  fi
}

echo "waiting for fetch_kinetics_cvdf_subset to finish..."
while pgrep -f fetch_kinetics_cvdf_subset > /dev/null 2>&1; do
  sleep 30
done
echo "fetch process exited"

if ! tail -100 fetch_kinetics.log | grep -q "^Done\."; then
  echo "!!! fetch did not report success -- aborting orchestration !!!"
  tail -30 fetch_kinetics.log
  exit 1
fi

run /workspace/vrl/.venv/bin/python -m src.data.kinetics_index \
  --video-root data/kinetics/raw/c50 \
  --output-dir data/kinetics/subsets/c50_train100_heldout30 \
  --subset-id c50_train100_heldout30 \
  --max-classes 50 --train-per-class 100 --heldout-per-class 30 --seed 0

runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1

create_pod_once() {
  local name="$1"
  runpodctl create pod \
    --name "$name" \
    --imageName "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404" \
    --templateId runpod-torch-v280 \
    --secureCloud \
    --gpuType "NVIDIA RTX A5000" \
    --gpuCount 1 \
    --volumeSize 80 \
    --volumePath /workspace \
    --containerDiskSize 20 \
    --startSSH 2>> "$LOG" | grep -oP '(?<=pod ")[a-z0-9]+(?=")'
}

# Capacity errors ("no instances available") are often transient -- retry a
# few times with a short backoff before giving up on this pod. Each pod's
# outcome is independent: one failing must never affect the other.
create_pod() {
  local name="$1" id=""
  for attempt in 1 2 3; do
    id=$(create_pod_once "$name")
    if [ -n "$id" ]; then
      echo "$id"
      return 0
    fi
    echo "create_pod($name) attempt $attempt failed, retrying in 60s..." >> "$LOG"
    sleep 60
  done
  return 1
}

echo "creating pod2 (videomae)..."
POD2_ID=$(create_pod kinetics-c50-videomae) || POD2_ID=""
echo "pod2 id: ${POD2_ID:-<failed>}"
echo "creating pod3 (dismo)..."
POD3_ID=$(create_pod kinetics-c50-dismo) || POD3_ID=""
echo "pod3 id: ${POD3_ID:-<failed>}"

wait_for_ssh() {
  local ip="$1" port="$2"
  for i in $(seq 1 40); do
    if ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        -i "$RUNPOD_SSH_KEY_PATH" -p "$port" root@"$ip" "echo ready" 2>/dev/null; then
      return 0
    fi
    sleep 15
  done
  return 1
}

fail_pod() {
  local pod_id="$1"
  echo "stopping $pod_id due to setup failure so it doesn't idle-bill"
  robust_pod_stop "$pod_id"
}

setup_and_launch() {
  local pod_id="$1" config_dir="$2" prefix="$3"
  local ssh_info ip port
  ssh_info=$(runpodctl ssh info "$pod_id" 2>>"$LOG")
  ip=$(echo "$ssh_info" | /workspace/vrl/.venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['ip'])")
  port=$(echo "$ssh_info" | /workspace/vrl/.venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['port'])")
  echo "pod $pod_id at $ip:$port"

  if ! wait_for_ssh "$ip" "$port"; then
    echo "!!! SSH never came up for $pod_id !!!"
    fail_pod "$pod_id"
    return 1
  fi

  local ssh_opts=(-o StrictHostKeyChecking=accept-new -i "$RUNPOD_SSH_KEY_PATH" -p "$port")
  local rsync_ssh="ssh -i $RUNPOD_SSH_KEY_PATH -p $port -o StrictHostKeyChecking=accept-new"

  run_ssh() { ssh "${ssh_opts[@]}" root@"$ip" "$@"; }

  rsync -az -e "$rsync_ssh" --no-owner --no-group \
    --exclude='/.git/' --exclude='/.venv/' --exclude='/data/' --exclude='/outputs/' \
    --exclude='**/__pycache__/' \
    /workspace/vrl/ root@"$ip":/workspace/vrl/
  rsync -az -e "$rsync_ssh" --no-owner --no-group \
    /workspace/vrl/data/kinetics/raw/c50/ root@"$ip":/workspace/vrl/data/kinetics/raw/c50/
  rsync -az -e "$rsync_ssh" --no-owner --no-group \
    /workspace/vrl/data/kinetics/subsets/c50_train100_heldout30/ \
    root@"$ip":/workspace/vrl/data/kinetics/subsets/c50_train100_heldout30/

  run_ssh "cd /workspace/vrl && uv sync >> setup.log 2>&1 && uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu124 torch >> setup.log 2>&1; .venv/bin/python -c 'import torch; assert torch.cuda.is_available()' >> setup.log 2>&1"
  if [ $? -ne 0 ]; then
    echo "!!! env setup failed on $pod_id !!!"
    fail_pod "$pod_id"
    return 1
  fi

  run_ssh "cd /workspace/vrl && chmod +x scripts/run_kinetics_extraction_pipeline.sh"
  if [ $? -ne 0 ]; then
    echo "!!! chmod failed on $pod_id, pipeline script missing? !!!"
    fail_pod "$pod_id"
    return 1
  fi

  run_ssh "cd /workspace/vrl && RUNPOD_API_KEY='$RUNPOD_API_KEY' nohup bash scripts/run_kinetics_extraction_pipeline.sh $config_dir $prefix $pod_id > /dev/null 2>&1 &"
  echo "launched pipeline on $pod_id"
  return 0
}

if [ -n "$POD2_ID" ]; then
  setup_and_launch "$POD2_ID" kinetics_videomae_linear_probe kinetics_videomae_c50 \
    || echo "!!! pod2 setup/launch failed, continuing !!!"
fi
if [ -n "$POD3_ID" ]; then
  setup_and_launch "$POD3_ID" kinetics_dismo_linear_probe kinetics_dismo_c50 \
    || echo "!!! pod3 setup/launch failed, continuing !!!"
fi

echo "=== $(date -u) pod2/pod3 handoff complete, starting vjepa pipeline on pod1 ==="
trap - EXIT
RUNPOD_API_KEY="$RUNPOD_API_KEY" exec bash scripts/run_kinetics_extraction_pipeline.sh \
  kinetics_vjepa_linear_probe kinetics_vjepa_c50 "$POD1_ID"
