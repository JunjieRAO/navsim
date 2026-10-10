# GT two-stage diagnostic experiment

This experiment uses the actual GT trajectory for every stage-one scene and
every stage-two scene with available GT. Stage-one GT is mandatory. For scenes
without GT it compares two selections from the same cached proposal set:

- `gt_model_top1`: maximum model score.
- `gt_actual_top1`: maximum actual per-scene `pdm_score`, evaluated separately
  for every proposal with the same reference trajectory and reactive traffic
  policy as normal evaluation. Ties select the lowest proposal index.

The actual top1 selection excludes two-frame extended comfort. That metric is
computed after selecting both adjacent-frame trajectories. Thus this is not an
oracle maximizing final EPDMS or the joint two-stage score. Exhaustive selection
can require roughly one simulation per missing-GT scene per proposal.

## Run

Use the same agent, checkpoint, dataset paths, and split as the candidate cache.
The experiment validates cache identity and requires complete metric caches and
two-stage mappings; failed scenes abort instead of silently changing populations.
No GPU inference is needed because candidates are read from the existing cache.

```bash
cd /mnt/workspace/roa7sgh/navsim-v2
python -m navsim.planning.script.run_pdm_score \
  agent=drivoR train_test_split=navhard_two_stage \
  experiment_name=drivor_gt_two_stage \
  metric_cache_path=/mnt/workspace/roa7sgh/navsim-v2/exp/navhard_two_stage_metric_cache \
  synthetic_sensor_path=/mnt/workspace/hru4sgh/NAVSIM/dataset/navhard_two_stage/sensor_blobs \
  synthetic_scenes_path=/mnt/workspace/hru4sgh/NAVSIM/dataset/navhard_two_stage/synthetic_scene_pickles \
  gt_two_stage.enabled=true \
  gt_two_stage.candidate_cache_path=/mnt/workspace/roa7sgh/navsim-v2/exp/drivor_nav2_gt_like/2026.09.28.10.51.11/oracle_candidates.npz \
  gt_two_stage.gt_cache_path=/mnt/workspace/roa7sgh/navsim-v2/exp/drivor_nav2_gt_like_fde/2026.10.08.11.09.52/oracle_fde.npz \
  max_scenarios_per_task=8
```

Set the usual NAVSIM data environment variables before running, including the
original logs and original sensor paths. If cache validation reports a mismatch,
use the original experiment's resolved agent and dataset configuration rather
than disabling validation. Worker concurrency can be configured as in normal
PDM evaluation; each task evaluates all missing-GT candidates in its token batch.
Launch as a module from this repository root so another installed NAVSIM checkout
does not take precedence over this experiment's code.

`gt_cache_path` is optional: original scene GT is loaded when not supplied in the
cache. An external GT archive can contain `tokens`, `poses` with shape
`[scenes, num_poses, 3]`, and scalar `interval_length`; an optional `has_gt` mask
marks valid entries. Poses must be in each scene's local rear-axle coordinates,
including synthetic scenes. The existing oracle archive (`ground_truth` and
`has_gt`) is also supported. Unavailable synthetic GT is never fabricated.

## Outputs

- `gt_two_stage_summary.json`: both selection definitions and stage-one,
  stage-two, combined scores for all, non-fallback, and fallback branches;
  zero-score branch counts and safety-zero scene counts plus weighted mass.
- `gt_{model,actual}_top1_branches.csv`: one row per first-stage frame and its
  follow-up set, with kernel sum, nearest start distance, fallback status, and
  all three scores. Counts are computed from GT endpoints, not hard-coded to
  377/73. There are two branches per mapping.
- `gt_{model,actual}_top1_safety_zero_causes.csv`: individual zero-score scenes
  attributed to collision, drivable-area, direction, or traffic-light zero
  multipliers. Multiple causes can overlap. Zero scores without these causes
  are recorded separately. TTC and comfort are weighted metrics, not safety
  multipliers that alone force zero final score.
- `selection.csv`: GT availability and selected indices for both modes; `-1`
  means GT, not a proposal.
- `actual_candidate_scores.csv`: actual PDM scores for every missing-GT
  scene/proposal, including the model top1.
- Per-setting raw and final pickles and final scene-score CSVs.

The combined score averages the product of each branch's stage-one score and
weighted stage-two score, not the product of aggregate stage means. Safety
counts include zero-weight scenes; weighted mass per branch distinguishes their
contribution. The fallback/non-fallback groups are identical in both settings
because their first-stage GT endpoints are identical.