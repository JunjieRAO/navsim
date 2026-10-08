#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"

LOG_DIR="$NAVSIM_EXP_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drivor_v2_$(date +%Y%m%d_%H%M%S)_$$.log"

nohup env PYTHONUNBUFFERED=1 bash "$SCRIPT_DIR/run_drivor_v2_pipeline.sh" "$@" >"$LOG_FILE" 2>&1 < /dev/null &

printf 'PID: %s\nLog: %s\n' "$!" "$LOG_FILE"