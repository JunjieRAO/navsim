#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"
cd "$NAVSIM_DEVKIT_ROOT"

MODE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$MODE" in
    smoke) MAX_MAPPINGS=1 ;;
    full) MAX_MAPPINGS=0 ;;
    *)
        printf 'Usage: %s [smoke|full] [Hydra overrides...]\n' "$0" >&2
        exit 2
        ;;
esac

ORACLE_EC_MODE="${ORACLE_EC_MODE:-none}"
for override in "$@"; do
    case "$override" in
        oracle_ec_mode=*) ORACLE_EC_MODE="${override#oracle_ec_mode=}" ;;
    esac
done
case "$ORACLE_EC_MODE" in
    none) EXPERIMENT_NAME=drivor_nav2_oracle/woEC ;;
    fixed_history) EXPERIMENT_NAME=drivor_nav2_oracle/fixedHistoryEC ;;
    *)
        printf 'Invalid ORACLE_EC_MODE: %s (use none or fixed_history)\n' "$ORACLE_EC_MODE" >&2
        exit 2
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nav-v2/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter not found: %s (set PYTHON_BIN to override)\n' "$PYTHON_BIN" >&2
    exit 1
fi

GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
GPU_NUM_DEVICES="${GPU_NUM_DEVICES:-4}"
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-16}"
GPU_NUM_WORKERS="${GPU_NUM_WORKERS:-8}"
ORACLE_WORKERS="${ORACLE_WORKERS:-72}"

LOG_DIR="$NAVSIM_EXP_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drivor_two_stage_oracle_${MODE}_$(date +%Y%m%d_%H%M%S)_$$.log"

nohup env PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_two_stage_oracle.py" \
    "experiment_name=$EXPERIMENT_NAME" \
    "oracle_ec_mode=$ORACLE_EC_MODE" \
    "gpu_device=$GPU_DEVICE" \
    "gpu_num_devices=$GPU_NUM_DEVICES" \
    "gpu_batch_size=$GPU_BATCH_SIZE" \
    "gpu_num_workers=$GPU_NUM_WORKERS" \
    "oracle_workers=$ORACLE_WORKERS" \
    "metric_cache_path=$DRIVOR_NAV2_CACHE_PATH" \
    "synthetic_sensor_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs" \
    "synthetic_scenes_path=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles" \
    "oracle_max_mappings=$MAX_MAPPINGS" \
    "$@" >"$LOG_FILE" 2>&1 < /dev/null &

printf 'PID: %s\nLog: %s\n' "$!" "$LOG_FILE"