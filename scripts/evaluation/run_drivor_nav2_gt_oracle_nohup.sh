#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_drivor_nav2_gpu_nohup.sh" \
    experiment_name=drivor_nav2_gt_like \
    oracle_gt.enabled=true \
    "$@"