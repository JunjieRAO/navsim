# Visualize a stage-two cluster

A cluster here means all stage-two scenes associated with one stage-one frame.
The original and previous stage-one frames in each mapping define separate
clusters. Cluster sizes may differ. All members are drawn, including successful
scenes, sorted by distance to the corresponding stage-one GT endpoint.

Run from the repository root with the NAVSIM environment:

```bash
cd /mnt/workspace/roa7sgh/navsim-v2
source env_drivor_nav2.sh
/root/miniconda3/envs/nav-v2/bin/python -m scripts.evaluation.plot_gt_stage2_cluster \
  exp/drivor_gt_two_stage/2026.10.09.16.48.12
```

Without a token, the script chooses a cluster containing an all-candidate-zero
scene with the largest saved stage-two weight. It reads existing scores and
candidate poses; it does not run model inference or candidate scoring again.

Select a cluster by its first-stage token:

```bash
/root/miniconda3/envs/nav-v2/bin/python -m scripts.evaluation.plot_gt_stage2_cluster \
  exp/drivor_gt_two_stage/2026.10.09.16.48.12 \
  --first-token c296ded44f9a509a
```

Alternatively, pass `--scene-token` with any stage-two member to select its whole
parent cluster. A cluster must contain at least one all-candidate-zero scene.
Use `--radius 60` to expand each plot, `--per-page 6 --columns 2` for smaller
pages, or `--dpi 200` for more detail. `--candidate-cache`, `--metric-cache`, and
`--output` override paths stored in the experiment's resolved Hydra config.

## Outputs and interpretation

To enumerate and plot every cluster containing an all-candidate-zero scene
whose own saved GT-endpoint weight is strictly greater than 0.001:

```bash
bash scripts/evaluation/run_plot_gt_failure_clusters.sh
```

The default experiment is `exp/drivor_gt_two_stage/2026.10.09.16.48.12`.
`EXPERIMENT_DIR` and `PYTHON_BIN` environment variables override those defaults.
The launcher sources the data environment and invokes the configured interpreter
directly, so prior conda activation is not required. Preview the selection only:

```bash
LIST_ONLY=1 bash scripts/evaluation/run_plot_gt_failure_clusters.sh
```

Selection requires both all 64 actual scores (indices 0..63) to be zero and
that same scene's `weight > 0.001` in `gt_actual_top1_scores.csv`, not baseline
weights or the cluster's total weight. Weight exactly 0.001 is excluded.
The CSV retains total all-zero counts and adds `qualifying_scenes` and
`qualifying_scene_tokens` for failures meeting the weight threshold.
The script saves `failure_cluster_tokens.txt` and
`failure_clusters.csv` under `cluster_visualizations/`. It calls the plotter
with each first-stage token, sequentially, and draws all cluster members.
Limit the batch to N clusters, prioritizing coverage of all four safety-zero categories:

```bash
bash scripts/evaluation/run_plot_gt_failure_clusters.sh \
  --max-clusters 10 --per-page 6 --columns 2
```

`--max-clusters` requires a positive integer; omitting it processes all selected
clusters. Values larger than the available count process all available clusters.
The full token list is ordered with a smallest representative set covering all
available NC=0, DAC=0, DDC=0, and TLC=0 categories first, followed by remaining
clusters in original mapping order. One cluster can cover multiple categories.
Category counts refer to actual top1 metrics of scenes satisfying both the
64-candidate-zero and weight > 0.001 conditions, not unrelated cluster members.
The selection CSV adds `NC_zero_scenes`, `DAC_zero_scenes`, `DDC_zero_scenes`,
and `TLC_zero_scenes`. The launcher prints category coverage for the limited
selection and warns if the limit is too small or qualifying data lacks a category.
Coverage is not a balanced per-category quota and does not require four distinct
clusters. Plotting failures may prevent a selected representative from rendering;
check the batch failure report before relying on the rendered coverage.
The limit also applies in `LIST_ONLY=1` mode, without truncating the saved full
token list. The limit counts attempted clusters, including any failed plots.

Forward plotting options such as `--per-page 6 --columns 2 --dpi 160` after the
shell script name. Do not pass a shared `--output` directory or token selection
options when plotting all clusters: default per-token directories keep outputs
separate. Each batch gets individual cluster logs under `batch_logs/`; failed
clusters are listed in `failed_cluster_tokens.txt`, and the launcher exits
nonzero after attempting the remaining clusters. This batch can create thousands
of images and does not run candidate scoring again.

Outputs are saved under `cluster_visualizations/<first-token>/` in the experiment:

- `cluster_overview.png`: map and all stage-two initial positions in a common
  coordinate frame centered on the GT endpoint, using the first-stage initial
  heading. Red points have all candidate PDM scores zero; green points have at
  least one nonzero candidate. Arrows show initial heading. Numbers correspond
  to the distance-sorted scene plots.
- `cluster_page_*.png`: all members in paginated grids, with each map centered
  and oriented on that scene's ego rear axle. Each panel shows initial objects,
  the 64 network proposals, GT endpoint, initial speed, weight, candidate-zero
  count, maximum candidate PDM, and safety metrics for actual selected top1.
- `scene_<number>_<token>.png`: larger individual panels for inspecting each
  member without the grid.
- `scene_manifest.csv`: tokens, distances, weights, zero counts, maximum PDM,
  and filenames; `notes.json` records drawing limitations.

Candidate line colors: red for zero PDM, green for nonzero PDM. Blue highlights
model top1, black dashed highlights actual PDM top1; these can overlap. A gold
star marks the stage-one GT endpoint. Background box colors use the project's
existing BEV object styles; their colors do not indicate candidate PDM scores.
Distances are global XY start-to-GT-endpoint distances. Each subplot reverses
the lateral axis to match the project's forward-up BEV convention.

NC, DAC, DDC, TLC denote collision, drivable-area, driving-direction, and
traffic-light compliance. These labels describe only the selected actual top1,
not all 64 candidates. All-zero ties select proposal index zero. Network poses
are plotted, not the controller-simulated ego trajectory used during scoring.
Brown dotted tracks show cached actor futures, not candidate-dependent reactive
rollouts. Thus static plots help locate failure patterns but do not prove the
time or cause of a collision. Different speeds and headings can matter even
when scene starts are close in XY.