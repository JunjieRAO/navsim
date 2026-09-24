#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"
cd "$NAVSIM_DEVKIT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivor-nav2/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter not found: %s (set PYTHON_BIN to override)\n' "$PYTHON_BIN" >&2
    exit 1
fi

LOG_DIR="$NAVSIM_EXP_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drivor_nav2_$(date +%Y%m%d_%H%M%S)_$$.log"

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
    train_test_split=navhard_two_stage \
    experiment_name=drivor_nav2_gpu \
    agent=drivoR \
    "metric_cache_path=$DRIVOR_NAV2_CACHE_PATH" \
    "synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs" \
    "synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles" \
    worker.threads_per_node=72 \
    max_scenarios_per_task=32 \
    "$@" >"$LOG_FILE" 2>&1 < /dev/null &

printf 'PID: %s\nLog: %s\n' "$!" "$LOG_FILE"