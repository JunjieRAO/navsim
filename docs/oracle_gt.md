# DrivoR GT Oracle Reranking

This is a privileged, offline diagnostic, not a deployable policy. The model
and checkpoint are unchanged. Each scene's final 64 proposals and original
`pdm_score` are retained after one GPU inference pass.

## Score and ADE Units

With the default DrivoR configuration the ranking score is

```text
S = 10 log(p_NC) + 13 log(p_DAC) + 6 log(p_DDC)
    + log(14 p_TTC + 15 p_EP + 2 p_comfort)
```

Each probability is a sigmoid head output. The score has upper bound
`log(31) = 3.434` and no finite lower bound. It is not a normalized PDM metric.
Reranking uses `S - lambda * ADE` directly, without exponentiating or
per-scene min-max normalization. ADE is mean Euclidean XY displacement in
meters over the eight future samples at 0.5, 1.0, ..., 4.0 seconds. Heading
is excluded. GT and proposals use the current ego rear-axle coordinate frame.

For each scene with GT, let `best` be the original score argmax. Its first
switching threshold is the minimum of
`(S_best - S_candidate) / (ADE_best - ADE_candidate)` over closer candidates.
The global reference lambda is the median of strictly positive, finite
thresholds. The default scan multiplies it by
`[0, 0.03, 0.1, 0.3, 1, 3, 10, 30]`, then adds a separate minimum-ADE control.
Tied-score zero thresholds and already-minimum-ADE scenes are counted separately.
If no positive threshold exists, an explicit lambda grid is required.

## GT Coverage

Original scenes use `Scene.get_future_trajectory`. Synthetic metric caches
explicitly have no human trajectory; stored synthetic ego poses are not
silently interpreted as human GT, and the PDM reference planner is not used
as a substitute. The default `oracle_gt.missing_gt=error` fails before inference
when human GT is unavailable.

For an all-scene oracle, provide `oracle_gt.gt_path` pointing to a NumPy NPZ
archive with these arrays (no pickled objects):

- `tokens`: unique scene token strings, not synthetic pickle filename stems.
- `poses`: finite `[scenes, 8, 3]` local GT poses at the proposal sample times.
- `interval_length`: scalar `0.5`.

External GT overrides original-scene GT for matching tokens. Its provenance,
coordinate frame, and timestamp alignment are the experimenter's responsibility.

Alternatively, explicitly choose `oracle_gt.missing_gt=baseline`. Only scenes
with GT are reranked; other scenes retain their baseline selection in every
setting, including the minimum-ADE control. Results are labeled
`partial_oracle_missing_gt_keeps_baseline`, and ADE statistics cover only scenes
with GT. This is not an all-scene oracle.

## Run

The new launcher inherits the existing four-GPU setup (batch size 16 and
eight data workers per GPU). It leaves the ordinary evaluation launcher unchanged.

```bash
# All-scene oracle, with externally supplied, aligned synthetic GT:
bash scripts/evaluation/run_drivor_nav2_gt_oracle_nohup.sh \
  oracle_gt.gt_path=/absolute/path/to/gt.npz

# Explicit partial oracle: rerank human-GT scenes, preserve others:
bash scripts/evaluation/run_drivor_nav2_gt_oracle_nohup.sh \
  oracle_gt.missing_gt=baseline

# Cache candidates and inspect calibration without running simulation:
bash scripts/evaluation/run_drivor_nav2_gt_oracle_nohup.sh \
  oracle_gt.missing_gt=baseline oracle_gt.prepare_only=true

# Reuse the candidate cache and supply lambda values in log-score per meter:
bash scripts/evaluation/run_drivor_nav2_gt_oracle_nohup.sh \
  oracle_gt.missing_gt=baseline \
  oracle_gt.cache_path=/absolute/path/to/oracle_candidates.npz \
  'oracle_gt.lambdas=[0,0.01,0.03,0.1,0.3,1,3]'
```

Each setting runs the existing simulator/scorer and recomputes endpoint weights
and two-frame extended comfort. Missing metric caches, incomplete mapping groups,
failed scenarios, and nonfinite scores are errors, not silent population changes.
Select complete two-stage groups for small evaluations. GPU export asserts that
score argmax exactly reproduces the model-returned baseline trajectory.

## Outputs

Outputs are stored in the Hydra experiment directory under
`exp/drivor_nav2_gt_oracle/`:

- `oracle_candidates.npz`: tokens, proposals, raw scores, and cache identity.
- `oracle_ade.npz`: aligned GT, per-proposal ADE, coverage, and GT sources.
- `oracle_diagnostics.json`: score/ADE quantiles, scene score spans, switching
  thresholds, reference lambda, and oracle scope.
- `lambda_XX_selection.csv` and `min_ade_selection.csv`: selected proposal,
  baseline index, selected ADE and raw score for each scene.
- `*_scores.csv`, `*_scores.raw.pkl`, `*_scores.pkl`: scene results before and
  after aggregation. Raw results are retained even if coverage validation fails.
- `oracle_summary.csv`: actual lambda, combined/stage-one/stage-two score,
  delta versus lambda zero, mean selected ADE, change rates, and valid scene count.
  It is updated after each completed setting. The minimum-ADE row has no finite lambda.

Reusing a candidate cache checks agent config, checkpoint path/size/mtime,
data paths, and exact tokens. It does not hash sensor data contents; regenerate
the cache after changing data in place. GT is reloaded and ADE recalculated on
each run, so an updated GT archive does not require another model forward pass.

Compare lambda zero with the ordinary evaluator using the same checkpoint,
scene population, scorer and comfort configuration. A sweep optimized on this
evaluation split is exploratory, not an unbiased estimate of deployment benefit.
Minimum ADE is not an oracle upper bound on PDM score.

## Tests

```bash
/root/miniconda3/envs/nav-v2/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```