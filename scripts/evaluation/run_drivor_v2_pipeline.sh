#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for STEP in 01_index 02_cache 03_infer 04_score 05_analyze; do
    printf '[%s] Starting V2 %s\n' "$(date -Is)" "$STEP"
    bash "$SCRIPT_DIR/run_drivor_v2_${STEP}.sh" "$@"
    printf '[%s] Completed V2 %s\n' "$(date -Is)" "$STEP"
done