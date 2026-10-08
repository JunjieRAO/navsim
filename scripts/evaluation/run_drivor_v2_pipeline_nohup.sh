#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"

SPLIT="${V2_SPLIT:-navhard_two_stage}"
OVERRIDES=("train_test_split=$SPLIT")
if [[ "$SPLIT" == "navhard_two_stage" ]]; then
	PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nav-v2/bin/python}"
	STAGE_ONE_TOKENS="$("$PYTHON_BIN" - "$NAVSIM_DEVKIT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

config = OmegaConf.load(
	Path(sys.argv[1]) / "navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"
)
tokens = list(dict.fromkeys(str(entry[0]) for entry in config.reactive_all_mapping))
if not tokens:
	raise ValueError("No original stage-one tokens in navhard_two_stage mapping")
print(json.dumps(tokens, separators=(",", ":")))
PY
	)"
	OVERRIDES+=(
		"train_test_split.scene_filter.include_synthetic_scenes=false"
		"train_test_split.scene_filter.num_future_frames=10"
		"train_test_split.scene_filter.tokens=$STAGE_ONE_TOKENS"
		"navsim_log_path=$OPENSCENE_DATA_ROOT/navsim_logs/test"
		"original_sensor_path=$OPENSCENE_DATA_ROOT/sensor_blobs/test"
		"v2_root=${V2_ROOT:-$NAVSIM_EXP_ROOT/drivor_v2/navhard_stage1_top25_full}"
	)
fi

LOG_DIR="$NAVSIM_EXP_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/drivor_v2_${SPLIT}_$(date +%Y%m%d_%H%M%S)_$$.log"

nohup env PYTHONUNBUFFERED=1 bash "$SCRIPT_DIR/run_drivor_v2_pipeline.sh" "${OVERRIDES[@]}" "$@" >"$LOG_FILE" 2>&1 < /dev/null &

printf 'PID: %s\nLog: %s\nSplit: %s\n' "$!" "$LOG_FILE" "$SPLIT"