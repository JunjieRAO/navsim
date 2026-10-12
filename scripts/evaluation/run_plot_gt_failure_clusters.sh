#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../env_drivor_nav2.sh"
cd "$NAVSIM_DEVKIT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/nav-v2/bin/python}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$NAVSIM_EXP_ROOT/drivor_gt_two_stage/2026.10.09.16.48.12}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter not found: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$EXPERIMENT_DIR/actual_candidate_scores.csv" ]]; then
    printf 'Candidate scores not found in: %s\n' "$EXPERIMENT_DIR" >&2
    exit 1
fi

MAX_CLUSTERS=""
PLOT_ARGS=()
while (( $# > 0 )); do
    case "$1" in
        --max-clusters)
            if (( $# < 2 )) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s\n' '--max-clusters requires a positive integer.' >&2
                exit 1
            fi
            MAX_CLUSTERS="$2"
            shift 2
            ;;
        --max-clusters=*)
            MAX_CLUSTERS="${1#*=}"
            if [[ ! "$MAX_CLUSTERS" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s\n' '--max-clusters requires a positive integer.' >&2
                exit 1
            fi
            shift
            ;;
        *)
            PLOT_ARGS+=("$1")
            shift
            ;;
    esac
done

SELECTION_ARGS=()
if [[ -n "$MAX_CLUSTERS" ]]; then
    SELECTION_ARGS+=(--max-clusters "$MAX_CLUSTERS")
fi
"$PYTHON_BIN" -m scripts.evaluation.list_gt_failure_clusters "$EXPERIMENT_DIR" "${SELECTION_ARGS[@]}"
TOKEN_FILE="$EXPERIMENT_DIR/cluster_visualizations/failure_cluster_tokens.txt"
mapfile -t CLUSTER_TOKENS < "$TOKEN_FILE"
if [[ -n "$MAX_CLUSTERS" ]]; then
    CLUSTER_TOKENS=("${CLUSTER_TOKENS[@]:0:$MAX_CLUSTERS}")
    printf 'Cluster limit: %s; processing %s clusters in safety-coverage-first order.\n' "$MAX_CLUSTERS" "${#CLUSTER_TOKENS[@]}"
fi
if [[ "${LIST_ONLY:-0}" == "1" ]]; then
    printf 'List-only mode: selected %s clusters; no images generated.\n' "${#CLUSTER_TOKENS[@]}"
    exit 0
fi

LOG_DIR="$EXPERIMENT_DIR/cluster_visualizations/batch_logs/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$LOG_DIR"
FAILED_TOKENS=()
for INDEX in "${!CLUSTER_TOKENS[@]}"; do
    TOKEN="${CLUSTER_TOKENS[$INDEX]}"
    printf '[%s/%s] Plotting cluster %s\n' "$((INDEX + 1))" "${#CLUSTER_TOKENS[@]}" "$TOKEN"
    if "$PYTHON_BIN" -m scripts.evaluation.plot_gt_stage2_cluster \
        "$EXPERIMENT_DIR" "${PLOT_ARGS[@]}" --first-token "$TOKEN" >"$LOG_DIR/$TOKEN.log" 2>&1; then
        printf 'Completed: %s\n' "$TOKEN"
    else
        FAILED_TOKENS+=("$TOKEN")
        printf 'Failed: %s; log: %s/%s.log\n' "$TOKEN" "$LOG_DIR" "$TOKEN" >&2
    fi
done
printf 'Finished: %s succeeded, %s failed. Logs: %s\n' \
    "$((${#CLUSTER_TOKENS[@]} - ${#FAILED_TOKENS[@]}))" "${#FAILED_TOKENS[@]}" "$LOG_DIR"
if (( ${#FAILED_TOKENS[@]} > 0 )); then
    printf '%s\n' "${FAILED_TOKENS[@]}" >"$LOG_DIR/failed_cluster_tokens.txt"
    exit 1
fi