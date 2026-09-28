# Two-stage oracle experiment

This offline experiment uses DrivoR's 64 proposals per scene and analyzes only
the original `now` scene and each mapped follow-up `now` scene. The evaluator,
not DrivoR's predicted scorer, selects the best follow-up proposal. It is
**not** the official two-branch leaderboard metric. There are two scoring modes:

- `none` (default) excludes two-frame extended comfort (EC) and renormalizes
  the other metrics, including history comfort (HC). Previous frames are unused.
- `fixed_history` selects each `now` scene's mapped `previous` frame using
  DrivoR's own predicted top-1. That single previous trajectory is simulated
  once and fixed while all 64 current proposals receive both a no-EC score
  and a score with EC computed against that previous simulated trajectory.
  No previous frame is selected using evaluator scores or included as a
  separate analysis sample.

For follow-up scene `j`, compute `E2[j] = max_k evaluator(j, k)` once per
scoring definition. `fixed_history` takes separate maxima for no-EC and
fixed-history-EC; their best proposal indices can differ.
For original proposal `i`, use its raw trajectory endpoint transformed to
global rear-axle coordinates and each follow-up scene's global starting
rear axle. The existing NAVSIM Gaussian proximity routine computes one
normalized row `W[i, :]` with `sigma_squared=0.1`, including its uniform
fallback for a near-zero weight sum. Set `V2 = W @ E2` independently for each
scoring definition, reusing the same `W`.

The short selection is `argmax_i E1[i]`; the long selection is
`argmax_i E1[i] * V2[i]`. Both are compared on the same two-stage scale:
`T_short = E1[i_short] * V2[i_short]`, `T_long = E1[i_long] * V2[i_long]`.
The gain is `T_long - T_short`, not `T_long - max(E1)`.

The model may infer several independent scenes per GPU batch. Evaluator
scoring is separate: each proposal runs through the official evaluator with
its reference PDM trajectory and its own reactive traffic rollout. Scoring
all 64 proposals in a single scorer batch would change progress normalization.

## Run

From the workspace root, with the paths in `env_drivor_nav2.sh` available:

```bash
bash scripts/evaluation/run_drivor_two_stage_oracle.sh smoke
ORACLE_EC_MODE=fixed_history bash scripts/evaluation/run_drivor_two_stage_oracle.sh smoke oracle_workers=4 gpu_num_devices=1 gpu_batch_size=4 gpu_num_workers=0
bash scripts/evaluation/run_drivor_two_stage_oracle.sh full oracle_ec_mode=fixed_history
```

The launcher runs in the background and prints its PID and log. `smoke` selects
the first entire original-now mapping, including every mapped follow-up-now
scene; `full` selects all mappings. The launcher defaults to 4 GPUs, a model
batch size of 16 per GPU, 8 DataLoader workers per GPU, and 72 CPU evaluation
workers. Set `GPU_DEVICE`, `GPU_NUM_DEVICES`, `GPU_BATCH_SIZE`,
`GPU_NUM_WORKERS`, `ORACLE_WORKERS` or `ORACLE_EC_MODE` in the environment to
change these defaults; trailing Hydra overrides take precedence. `none` writes
to `exp/drivor_nav2_oracle/woEC/<timestamp>/`, and `fixed_history` writes to
`exp/drivor_nav2_oracle/fixedHistoryEC/<timestamp>/`. For a synchronous run
(useful for errors):

```bash
source env_drivor_nav2.sh
/root/miniconda3/envs/nav-v2/bin/python navsim/planning/script/run_two_stage_oracle.py \
  oracle_ec_mode=fixed_history oracle_max_mappings=1 oracle_workers=4 \
  experiment_name=drivor_nav2_oracle/fixedHistoryEC "metric_cache_path=$DRIVOR_NAV2_CACHE_PATH"
```

On H20 (sm_90) GPUs, the older `drivor-nav2` Python environment has PyTorch
2.0.1+cu117 and cannot execute CUDA kernels; the launcher defaults to the
sm_90-compatible `nav-v2` environment. Set `PYTHON_BIN` only if the alternative
environment supports the installed GPU.

To resume, pass the same `experiment_uid` or `output_dir` Hydra override used
for the first run. Previous simulated top-1 and current token artifacts are
reused only when the experiment fingerprint and the corresponding metric-cache
stamps match; each fixed-history current token also checks its previous token
and cache stamp. A different fingerprint in the same directory is rejected;
use a new output directory after changing a checkpoint, scorer, code, split,
or data paths. Previously computed no-EC artifacts lack the fixed-history
states and cannot simply be relabeled as full scores.

Each run writes:

- `manifest.json`: fingerprint, EC mode, branch and proposal count.
- `tokens/<token>.npz`: 64 proposals, no-EC evaluator scores, per-proposal
  submetrics, global endpoints, and global start point. In fixed-history mode
  it also contains `fixed_ec_scores` and `extended_comfort` for all 64.
- `histories/<previous_token>.npz` (fixed-history only): selected DrivoR top-1,
  its 41 simulated states at 0.1 s, and its start time.
- `mappings/<original_token>.npz`: ordered follow-up tokens, `W`, follow-up
  maxima and their indices, 64 downstream values, and combined scores. In
  fixed-history mode it also includes `fixed_` versions of the oracle
  indices, scores and downstream values (the same `W` applies to both).
- `mappings.csv`: per-original short/long indices, `E1`, `V2`, `T`, gain,
  disagreement, and Gaussian fallback diagnostics for no-EC.
- `summary.json`: no-EC disagreement fraction, paired gains, coverage,
  failures, and distance diagnostics across complete original-now mappings.
- `mappings_fixed_history.csv`, `summary_fixed_history.json`, and
  `comparisons.csv` (fixed-history only): independent short/long oracle on
  EC-inclusive scores and paired changes relative to no-EC on the same scenes.
- `failures.json`: missing or failed tokens and invalid mappings.

Each token must have all 64 finite proposal scores. A missing or failed
follow-up invalidates its entire original mapping; the run does not silently
drop it and renormalize `W`. Equal scores choose the lowest original proposal
index. `E1_top_gap` and `near_tie_E1_rate` indicate how often the short choice
is a tie: when multiple proposals share the best `E1`, a positive gain can
reflect this deterministic tie-break rather than a strict immediate-score
tradeoff. This is an offline, generator-conditional recoverability upper bound,
not a deployable policy or a leaderboard result.