#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"
cd "$NAVSIM_DEVKIT_ROOT"

if [[ $# -lt 1 ]]; then
    printf 'Usage: bash run_drivor_v2.sh {index|cache|infer|score|analyze} [Hydra overrides]\n' >&2
    exit 2
fi
STAGE="$1"
shift
case "$STAGE" in
    index|cache|infer|score|analyze) ;;
    *) printf 'Unknown V2 stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nav-v2/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter not found: %s (set PYTHON_BIN to override)\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [[ "$STAGE" == "cache" || "$STAGE" == "infer" || "$STAGE" == "score" ]]; then
    export OMP_NUM_THREADS="${V2_THREADS_PER_WORKER:-1}"
    export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
    export MKL_NUM_THREADS="$OMP_NUM_THREADS"
    export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
fi

OVERRIDES=(
    "stage=$STAGE"
    "v2_root=${V2_ROOT:-$NAVSIM_EXP_ROOT/drivor_v2/navtrain}"
    "max_scenes=${V2_MAX_SCENES:-0}"
    "max_k=${V2_MAX_K:-64}"
    "v2_workers=${V2_WORKERS:-72}"
    "gpu_device=${V2_GPU_DEVICE:-cuda:0}"
    "gpu_num_devices=${V2_GPU_NUM_DEVICES:-4}"
    "gpu_batch_size=${V2_GPU_BATCH_SIZE:-16}"
    "gpu_num_workers=${V2_GPU_NUM_WORKERS:-8}"
    "gpu_inference_chunk_scenes=${V2_GPU_CHUNK_SCENES:-1024}"
)
if [[ -n "${V2_V1_CACHE_ROOT:-}" ]]; then
    OVERRIDES+=("v1_cache_root=$V2_V1_CACHE_ROOT")
fi
if [[ -n "${V2_K_VALUES:-}" ]]; then
    OVERRIDES+=("k_values=$V2_K_VALUES")
fi

exec "$PYTHON_BIN" navsim/planning/script/run_continuation_value.py "${OVERRIDES[@]}" "$@"