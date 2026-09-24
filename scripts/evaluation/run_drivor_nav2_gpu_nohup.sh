#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nav-v2/bin/python}"
exec "$SCRIPT_DIR/run_drivor_nav2_nohup.sh" \
    gpu_inference=true \
    gpu_device=cuda:0 \
    gpu_num_devices=4 \
    gpu_batch_size=16 \
    gpu_num_workers=8 \
    "$@"