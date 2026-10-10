#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"
cd "$NAVSIM_DEVKIT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nav-v2/bin/python}"
CANDIDATE_CACHE_PATH="${CANDIDATE_CACHE_PATH:-$NAVSIM_EXP_ROOT/drivor_nav2_gt_like/2026.09.28.10.51.11/oracle_candidates.npz}"
GT_CACHE_PATH="${GT_CACHE_PATH:-$NAVSIM_EXP_ROOT/drivor_nav2_gt_like_fde/2026.10.08.11.09.52/oracle_fde.npz}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter not found: %s (set PYTHON_BIN to override)\n' "$PYTHON_BIN" >&2
    exit 1
fi
for CACHE_PATH in "$CANDIDATE_CACHE_PATH" "$GT_CACHE_PATH"; do
    if [[ ! -f "$CACHE_PATH" ]]; then
        printf 'Experiment cache not found: %s\n' "$CACHE_PATH" >&2
        exit 1
    fi
done

LOG_DIR="$NAVSIM_EXP_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drivor_gt_two_stage_$(date +%Y%m%d_%H%M%S)_$$.log"

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m navsim.planning.script.run_pdm_score \
    train_test_split=navhard_two_stage \
    experiment_name=drivor_gt_two_stage \
    agent=drivoR \
    "metric_cache_path=$DRIVOR_NAV2_CACHE_PATH" \
    "synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs" \
    "synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles" \
    gt_two_stage.enabled=true \
    "gt_two_stage.candidate_cache_path=$CANDIDATE_CACHE_PATH" \
    "gt_two_stage.gt_cache_path=$GT_CACHE_PATH" \
    gpu_inference=true \
    gpu_device=cuda:0 \
    gpu_num_devices=4 \
    gpu_batch_size=16 \
    gpu_num_workers=8 \
    worker.threads_per_node=72 \
    max_scenarios_per_task=32 \
    "$@" >"$LOG_FILE" 2>&1 < /dev/null &

printf 'PID: %s\nLog: %s\n' "$!" "$LOG_FILE"
printf 'Candidates are reused from cache; GPU inference parameters are not exercised in this experiment.\n'