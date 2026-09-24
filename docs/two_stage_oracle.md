# Two-stage oracle experiment

This offline experiment uses DrivoR's 64 proposals per scene. It is **not** the
official full two-stage EPDMS metric: it removes two-frame extended comfort
(EC) and renormalizes the remaining weights, including history comfort (HC).
It analyzes only the original `now` scene and the `now` token of each mapped
follow-up pair. The evaluator, not DrivoR's predicted scorer, selects the
best follow-up proposal. No previous-frame trajectories are evaluated.

For follow-up scene `j`, compute `E2[j] = max_k evaluator_no_EC(j, k)` once.
For original proposal `i`, use its raw trajectory endpoint transformed to
global rear-axle coordinates and each follow-up scene's global starting
rear axle. The existing NAVSIM Gaussian proximity routine computes one
normalized row `W[i, :]` with `sigma_squared=0.1`, including its uniform
fallback for a near-zero weight sum. Set `V2 = W @ E2`.

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
bash scripts/evaluation/run_drivor_two_stage_oracle.sh full oracle_workers=4 gpu_num_devices=2
```

The launcher runs in the background and prints its PID and log. `smoke` selects
the first entire original-now mapping, including every mapped follow-up-now
scene; `full` selects all mappings. The launcher defaults to 4 GPUs, a model
batch size of 4 per GPU, 4 DataLoader workers per GPU, and 4 CPU evaluation
workers. Set `GPU_DEVICE`, `GPU_NUM_DEVICES`, `GPU_BATCH_SIZE`,
`GPU_NUM_WORKERS`, or `ORACLE_WORKERS` in the environment to change these
defaults; trailing Hydra overrides take precedence. For a synchronous run
(useful for errors):

```bash
source env_drivor_nav2.sh
/root/miniconda3/envs/nav-v2/bin/python navsim/planning/script/run_two_stage_oracle.py \
  oracle_max_mappings=1 "metric_cache_path=$DRIVOR_NAV2_CACHE_PATH"
```

On H20 (sm_90) GPUs, the older `drivor-nav2` Python environment has PyTorch
2.0.1+cu117 and cannot execute CUDA kernels; the launcher defaults to the
sm_90-compatible `nav-v2` environment. Set `PYTHON_BIN` only if the alternative
environment supports the installed GPU.

To resume, pass the same `experiment_uid` or `output_dir` Hydra override used
for the first run. Completed token artifacts are reused only when the
experiment fingerprint and the token's metric-cache file stamp match. A
different fingerprint in the same directory is rejected; use a new output
directory after changing a checkpoint, scorer, code, split, or data paths.

The default output is under `exp/drivor_nav2_two_stage_oracle/<timestamp>/`:

- `manifest.json`: fingerprint, score definition, branch and proposal count.
- `tokens/<token>.npz`: 64 proposals, no-EC evaluator scores, per-proposal
  submetrics, global endpoints, and global start point.
- `mappings/<original_token>.npz`: ordered follow-up tokens, `W`, follow-up
  maxima and their indices, 64 downstream values, and combined scores.
- `mappings.csv`: per-original short/long indices, `E1`, `V2`, `T`, gain,
  disagreement, and Gaussian fallback diagnostics.
- `summary.json`: disagreement fraction, paired gains, coverage, failures,
  and distance diagnostics across complete original-now mappings.
- `failures.json`: missing or failed tokens and invalid mappings.

Each token must have all 64 finite proposal scores. A missing or failed
follow-up invalidates its entire original mapping; the run does not silently
drop it and renormalize `W`. Equal scores choose the lowest original proposal
index. `E1_top_gap` and `near_tie_E1_rate` indicate how often the short choice
is a tie: when multiple proposals share the best `E1`, a positive gain can
reflect this deterministic tie-break rather than a strict immediate-score
tradeoff. This is an offline, generator-conditional recoverability upper bound,
not a deployable policy or a leaderboard result.