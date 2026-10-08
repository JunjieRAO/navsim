# DrivoR continuation V2 labels

This offline experiment evaluates 64 DrivoR proposals on original navtrain
scenes. Each 4-second proposal is simulated to its endpoint; 64 deterministic
4-second continuations are generated from that simulated state and the original
route centerline. Only scoring uses the following four seconds of ground-truth
agents and traffic lights. No synthetic stage-two scenes or cameras are needed
for continuation generation or scoring; DrivoR proposal inference still needs
the original camera blobs.

The label for proposal `i` and prefix length `K` is the average of the best
`ceil(K / 4)` of its first K continuation scores: K=16 uses top-4, K=32 uses
top-8, and K=64 uses top-16. This favors endpoints with several safe, useful
follow-up actions rather than a single lucky one. Scores use the NAVSIM-v2 PDM
scorer with the two-frame extended comfort weight removed and the remaining
weights renormalized. Each continuation is scored against the *same* route
following reference, one reference/candidate pair at a time, so progress
normalization cannot change with K. History comfort uses the simulated last
1.5 seconds of the first-stage proposal, not the logged ego history at t+4.
At-fault overlap with a ground-truth agent at the endpoint sets all that
proposal's continuation scores to zero. This is non-reactive log-replay
supervision, not the official synthetic-scene two-stage metric or an exact
closed-loop optimal value.

## Prerequisites

- `OPENSCENE_DATA_ROOT/navsim_logs/trainval/*.pkl` with at least eight seconds
  of continuous 2 Hz future data for selected current scenes.
- nuPlan maps at `NUPLAN_MAPS_ROOT`, original trainval sensor blobs for step 3,
  and `DRIVOR_NAV2_CHECKPOINT` plus the model backbone weights.
- A Python environment with NAVSIM-v2, PyTorch compatible with the GPU, and
  the dependencies already used by the oracle runner. The launchers default
  to `/root/miniconda3/envs/nav-v2/bin/python`; set `PYTHON_BIN` to override.

Run from the `navsim-v2` workspace root. For a one-scene smoke run, first set
`V2_ROOT` to an isolated directory and `V2_MAX_SCENES=1`; unset them or choose a
different `V2_ROOT` for the full trainval run. A previously built *NAVSIM-v2*
V1 metric cache can be reused at both t0 and t+4 by setting
`V2_V1_CACHE_ROOT=/path/to/navtrain_metric_cache`. The older DrivoR
`train_metric_cache` has a different pickle type and cannot be reused here.
Without a compatible V1 cache, step 2 builds only the required original and
shifted caches under `V2_ROOT`, leaving all existing cache roots untouched.

```bash
bash scripts/evaluation/run_drivor_v2_01_index.sh
bash scripts/evaluation/run_drivor_v2_02_cache.sh
bash scripts/evaluation/run_drivor_v2_03_infer.sh
bash scripts/evaluation/run_drivor_v2_04_score.sh
bash scripts/evaluation/run_drivor_v2_05_analyze.sh
```

Wait for each command to finish successfully before starting the next. The
background launcher defaults to the original stage-one scenes in
`navhard_two_stage`, using the mapping's original `now` tokens and continuous
original test logs, not synthetic stage-two branches. It writes to
`exp/drivor_v2/navhard_stage1_top25_full/` and runs all five stages in order:

```bash
bash scripts/evaluation/run_drivor_v2_pipeline_nohup.sh
```

The checked test logs contain all 225 stage-one tokens; 223 pass the existing
eight-second continuity checks and two are skipped for discontinuous timestamps.
The skip reasons are recorded in `index.json`. Test-set values are future-GT
oracle diagnostics only, not training labels or deployable selection inputs.
For a one-scene smoke run, use a separate output directory:

```bash
V2_MAX_SCENES=1 V2_GPU_NUM_DEVICES=1 V2_WORKERS=4 \
   V2_ROOT="$PWD/exp/drivor_v2/navhard_stage1_top25_smoke" \
   bash scripts/evaluation/run_drivor_v2_pipeline_nohup.sh
```

To retain the original navtrain background workflow:

```bash
V2_SPLIT=navtrain bash scripts/evaluation/run_drivor_v2_pipeline_nohup.sh
```

The numbered stage scripts still default to navtrain. When resuming an individual
navhard stage, supply the same resolved split, stage-one token filter, paths and
output directory as the background run; these are saved in its Hydra config.

The launcher reads `env_drivor_nav2.sh` (including `OPENSCENE_DATA_ROOT`),
defaults to four GPUs with batch size 16 and eight data workers per GPU,
and caches/scores up to 72 logs/scenes in parallel on the 96-CPU host.
Each parallel process uses one BLAS/OMP thread. Set `V2_GPU_NUM_DEVICES`,
`V2_GPU_BATCH_SIZE`, `V2_GPU_NUM_WORKERS`, `V2_GPU_CHUNK_SCENES` (default 1024),
`V2_WORKERS` (default 72), or `V2_THREADS_PER_WORKER` to adjust these limits.
Inference keeps one model process per GPU across log files, batching scenes
across logs; cache processing reads each log once per CPU task. For a single
log/scene smoke test, only the GPUs and CPU workers with assigned work start.
Each stage is resumable; when a stage fails, check the printed log and run
only that step again after fixing the cause.

1. `01_index` scans original navtrain/trainval logs once. It writes
   `V2_ROOT/index.json` with source stamps, token and raw frame offsets,
   t0/t+4 timestamps and reasons for skipped windows. Standard navtrain scenes
   need five seconds for the original loader; V2 additionally requires t+8.
   A t+4 cache needs four seconds of GT, not nine seconds of log future.
2. `02_cache` reuses compatible V1 caches and creates any missing
   `original_metric_cache/` and `shifted_metric_cache/` entries. The shifted
   cache includes t+4..t+8 GT observation, traffic lights and map. Per-scene
   errors are reported in `cache_status.json`; rerunning skips completed caches.
3. `03_infer` uses the saved index and the fixed checkpoint to cache `[64,8,3]`
   proposals under `proposals/tokens/`. This is the only GPU/camera step.
4. `04_score` simulates stage one, scores each continuation using GT replay,
   and saves `[64,max_k]` scores, per-mode metrics, terminal-overlap flags,
   endpoints and E1 scores under `scores/tokens/`. The default is 64 modes:
   a 4-by-5 speed/lateral grid, four special continuations (including emergency
   braking and straight constant speed), eight additional modes and 32 denser
   modes. Use `V2_WORKERS=4` to allow four CPU scoring processes when memory
   permits. Scoring up to K=64 entails up to 4096 evaluations per scene.
5. `05_analyze` reads existing scores, writes `analysis/top25/k_16_32_64/tokens/`
   label arrays (`v2_values`, `top_counts`, `e1_v2_values`), plus
   `sensitivity.csv` and `summary.json`. These report V2 and E1*V2 top-1
   agreement against the largest K, rank correlation, ties, saturation and
   value changes. `best_mode_indices` and `best_mode_scores` retain the
   single highest-scoring continuation solely for diagnostics; neither is
   the training label. When all 64 proposals tie, top-1 agreement is only an
   argmax tie-break and cannot establish value-ranking stability; Spearman is
   reported as null for constant targets.

`V2_MAX_K=64` controls how many modes step 4 scores. Keep it at 64 to compare
K=16,32,64 without recomputation. Changing only the analysis K list creates
a separate report from the same score files. If you already scored V2 with the
old max target, run only step 5: it validates the saved scorer configuration and
cache stamps, and writes the new top-25% labels without overwriting
`analysis/k_16_32_64/` or repeating inference and scoring.

```bash
V2_K_VALUES='[16,24,32,64]' bash scripts/evaluation/run_drivor_v2_05_analyze.sh
```

Every stage also accepts Hydra overrides after the script name, for example
`v2_root=/path/to/run`, `gpu_batch_size=4`, or `v2_workers=2`. Manifests reject
stale data when the log, checkpoint, scoring configuration or code changes;
start a new `V2_ROOT` after changing the checkpoint or score generator. An
incomplete scene is reported as a failure, not silently omitted from ranking
statistics. Targets are conditional on the checkpoint's proposal geometry;
do not use them unchanged to supervise an updated proposal generator.