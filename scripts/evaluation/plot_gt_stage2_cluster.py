import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
from navsim.visualization.bev import add_map_to_bev_ax, add_oriented_box_to_bev_ax
from navsim.visualization.config import AGENT_CONFIG, BEV_PLOT_CONFIG
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


SAFETY_METRICS = {
    "no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "DDC", "traffic_light_compliance": "TLC",
}


def select_cluster(config, candidate_scores, scores, first_token=None, scene_token=None):
    maxima = candidate_scores.groupby("token").pdm_score.max()
    clusters = {}
    for original, previous, pairs in config["train_test_split"]["reactive_all_mapping"]:
        for token, pair_index in ((original, 0), (previous, 1)):
            if token in clusters:
                raise ValueError("Repeated stage-one token in mappings")
            clusters[token] = [pair[pair_index] for pair in pairs]
    if scene_token:
        matches = [token for token, members in clusters.items() if scene_token in members]
        if len(matches) != 1:
            raise ValueError("Scene token must belong to exactly one stage-two cluster")
        first_token = matches[0]
    if first_token is None:
        failed = scores.loc[maxima.index[maxima == 0]].sort_values("weight", ascending=False)
        if failed.empty:
            raise ValueError("No cluster containing an all-candidate-zero scene")
        target = failed.index[0]
        first_token = next(token for token, members in clusters.items() if target in members)
    if first_token not in clusters:
        raise ValueError("Unknown stage-one token")
    members = clusters[first_token]
    if not any(maxima.loc[token] == 0 for token in members):
        raise ValueError("Selected cluster contains no all-candidate-zero scene")
    first = scores.loc[first_token]
    distances = {
        token: float(np.hypot(scores.loc[token, "start_point_x"] - first.endpoint_x,
                             scores.loc[token, "start_point_y"] - first.endpoint_y))
        for token in members
    }
    return first_token, sorted(members, key=distances.get), distances


def local_poses(origin, poses):
    return convert_absolute_to_relative_se2_array(origin, np.asarray(poses, dtype=float))


def map_for_cache(cache):
    parameters = cache.map_parameters
    return get_maps_api(parameters.map_root, parameters.map_version, parameters.map_name)


def prepare_axes(ax, radius):
    ax.set_aspect("equal")
    ax.set_xlim(radius, -radius)
    ax.set_ylim(-radius, radius)
    ax.set_xlabel("Lateral / m")
    ax.set_ylabel("Forward / m")
    ax.tick_params(labelsize=7)


def add_objects(ax, cache, origin):
    detections = cache.current_tracked_objects[0].tracked_objects.tracked_objects
    for tracked in detections:
        box = tracked.box
        center = local_poses(origin, [[box.center.x, box.center.y, box.center.heading]])[0]
        local_box = OrientedBox(StateSE2(*center), box.length, box.width, box.height)
        style = AGENT_CONFIG.get(tracked.tracked_object_type, AGENT_CONFIG[TrackedObjectType.VEHICLE])
        add_oriented_box_to_bev_ax(ax, local_box, style)
    ego_box = cache.ego_state.car_footprint.oriented_box
    ego_center = local_poses(origin, [[ego_box.center.x, ego_box.center.y, ego_box.center.heading]])[0]
    local_ego_box = OrientedBox(StateSE2(*ego_center), ego_box.length, ego_box.width, ego_box.height)
    add_oriented_box_to_bev_ax(ax, local_ego_box, AGENT_CONFIG[TrackedObjectType.EGO], add_heading=False)


def add_object_futures(ax, cache, origin):
    tracks = {}
    for detections in cache.future_tracked_objects[::5]:
        for tracked in detections.tracked_objects.tracked_objects:
            token = tracked.track_token
            if token is None:
                continue
            center = tracked.box.center
            tracks.setdefault(token, []).append([center.x, center.y, center.heading])
    for poses in tracks.values():
        if len(poses) > 1:
            relative = local_poses(origin, poses)
            ax.plot(relative[:, 1], relative[:, 0], color="#8d5e28", alpha=.45,
                    linewidth=.8, linestyle=":", zorder=3)


def draw_scene(ax, cache, map_api, proposals, pdm_scores, selected, row, first, distance, number, radius):
    origin = cache.ego_state.rear_axle
    add_map_to_bev_ax(ax, map_api, origin)
    add_objects(ax, cache, origin)
    add_object_futures(ax, cache, origin)
    curves = np.concatenate([np.zeros((len(proposals), 1, 2)), proposals[:, :, :2]], axis=1)
    colors = ["#cf3e45" if score == 0 else "#568475" for score in pdm_scores]
    ax.add_collection(LineCollection(curves[:, :, [1, 0]], colors=colors,
                                     linewidths=.7, alpha=.35, zorder=4))
    for index, color, style, width in (
        (int(selected.model_index), "#1676c4", "-", 2.1),
        (int(selected.actual_pdm_index), "#111111", "--", 1.6),
    ):
        ax.plot(curves[index, :, 1], curves[index, :, 0], color=color,
                linestyle=style, linewidth=width, zorder=6)
    gt = local_poses(origin, [[first.endpoint_x, first.endpoint_y, 0]])[0]
    ax.scatter(gt[1], gt[0], marker="*", s=115, c="#e7b718", edgecolors="black", zorder=8)
    zero_count = int((pdm_scores == 0).sum())
    speed = float(cache.ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude())
    metric_text = " ".join(f"{label}={row[metric]:g}" for metric, label in SAFETY_METRICS.items())
    ax.set_title(
        f"{number:02d}  {row.name}\n"
        f"PDM=0: {zero_count}/{len(pdm_scores)} | max={pdm_scores.max():.3f}\n"
        f"GT distance={distance:.2f}m | weight={row.weight:.3f} | v={speed:.2f}m/s\n"
        f"Actual #{int(selected.actual_pdm_index)}: {metric_text}",
        fontsize=8, color="#a7202b" if zero_count == len(pdm_scores) else "#222222",
    )
    prepare_axes(ax, radius)


def plot_cluster(args):
    experiment = args.experiment.resolve()
    config = yaml.safe_load((experiment / "code/hydra/config.yaml").read_text())
    scores = pd.read_csv(experiment / "gt_actual_top1_scores.csv").set_index("token")
    selected = pd.read_csv(experiment / "selection.csv").set_index("token")
    candidates = pd.read_csv(experiment / "actual_candidate_scores.csv")
    first_token, members, distances = select_cluster(
        config, candidates, scores, args.first_token, args.scene_token
    )
    candidate_groups = candidates.groupby("token")
    candidate_cache = args.candidate_cache or Path(config["gt_two_stage"]["candidate_cache_path"])
    with np.load(candidate_cache, allow_pickle=False) as archive:
        indices = {token: index for index, token in enumerate(archive["tokens"].tolist())}
        cached_proposals = archive["proposals"]
        proposals = {token: cached_proposals[indices[token]].copy() for token in members}
        del cached_proposals
    cache_loader = MetricCacheLoader(args.metric_cache or Path(config["metric_cache_path"]))
    first_cache = cache_loader.get_from_token(first_token)
    first = scores.loc[first_token]
    if not selected.loc[first_token, "has_gt"]:
        raise ValueError("Stage-one endpoint must come from GT")
    output = args.output or experiment / "cluster_visualizations" / first_token
    output.mkdir(parents=True, exist_ok=True)
    BEV_PLOT_CONFIG["figure_margin"] = (2 * args.radius, 2 * args.radius)
    origin = StateSE2(first.endpoint_x, first.endpoint_y, first_cache.ego_state.rear_axle.heading)
    map_api = map_for_cache(first_cache)
    overview, overview_ax = plt.subplots(figsize=(10, 10))
    add_map_to_bev_ax(overview_ax, map_api, origin)
    gt = local_poses(origin, [[first.endpoint_x, first.endpoint_y, 0]])[0]
    overview_ax.scatter(gt[1], gt[0], marker="*", c="#e7b718", s=180,
                        edgecolors="black", zorder=8, label="Stage 1 GT endpoint")
    manifest = []
    for page_start in range(0, len(members), args.per_page):
        page_members = members[page_start:page_start + args.per_page]
        rows = math.ceil(len(page_members) / args.columns)
        figure, axes = plt.subplots(rows, args.columns, figsize=(5.5 * args.columns, 7 * rows), squeeze=False)
        for offset, token in enumerate(page_members):
            number = page_start + offset + 1
            cache = cache_loader.get_from_token(token)
            pdm = candidate_groups.get_group(token).sort_values("proposal_index")
            expected = np.arange(len(proposals[token]))
            if not np.array_equal(pdm.proposal_index.to_numpy(), expected) or not np.isfinite(pdm.pdm_score).all():
                raise ValueError(f"Incomplete candidate scores for {token}")
            values = pdm.pdm_score.to_numpy()
            scene_origin = cache.ego_state.rear_axle
            actual_origin = scores.loc[token, ["start_point_x", "start_point_y"]].to_numpy(dtype=float)
            if not np.allclose([scene_origin.x, scene_origin.y], actual_origin, atol=1e-5, rtol=0):
                raise ValueError(f"Metric cache initial pose mismatch for {token}")
            position = local_poses(origin, [[scene_origin.x, scene_origin.y, scene_origin.heading]])[0]
            color = "#cf3e45" if values.max() == 0 else "#27846d"
            overview_ax.scatter(position[1], position[0], c=color, s=65, zorder=6)
            overview_ax.annotate(str(number), (position[1], position[0]),
                                 xytext=((18 if number % 2 else -24), 8 + 10 * (number % 3)),
                                 textcoords="offset points", fontsize=8, zorder=9,
                                 arrowprops={"arrowstyle": "-", "color": color, "linewidth": .6})
            overview_ax.arrow(position[1], position[0], 1.5 * np.sin(position[2]),
                              1.5 * np.cos(position[2]), color=color, width=.06, zorder=6)
            ax = axes.flat[offset]
            draw_scene(ax, cache, map_api, proposals[token], values, selected.loc[token], scores.loc[token],
                       first, distances[token], number, args.radius)
            individual, individual_ax = plt.subplots(figsize=(8, 8))
            draw_scene(individual_ax, cache, map_api, proposals[token], values, selected.loc[token], scores.loc[token],
                       first, distances[token], number, args.radius)
            individual.tight_layout()
            image_name = f"scene_{number:02d}_{token}.png"
            individual.savefig(output / image_name, dpi=args.dpi)
            plt.close(individual)
            manifest.append({"number": number, "token": token, "first_token": first_token,
                             "distance_to_gt_endpoint_m": distances[token], "weight": float(scores.loc[token, "weight"]),
                             "zero_candidates": int((values == 0).sum()), "max_pdm": float(values.max()),
                             "image": image_name})
        for unused in list(axes.flat)[len(page_members):]:
            unused.axis("off")
        figure.suptitle(f"Stage 2 cluster: {first_token} | distance-sorted | page {page_start // args.per_page + 1}", fontsize=12)
        legend = [Line2D([], [], color=color, linestyle=style, label=label) for color, style, label in (
            ("#cf3e45", "-", "Zero-PDM candidate"), ("#568475", "-", "Nonzero candidate"),
            ("#1676c4", "-", "Model top1"), ("#111111", "--", "Actual PDM top1"),
            ("#8d5e28", ":", "Cached actor future (not candidate-reactive)"),
        )]
        figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8)
        figure.subplots_adjust(left=.055, right=.985, bottom=.055, top=.94, hspace=.55, wspace=.24)
        figure.savefig(output / f"cluster_page_{page_start // args.per_page + 1:02d}.png", dpi=args.dpi)
        plt.close(figure)
    overview_ax.set_title(f"All {len(members)} stage 2 starts | cluster {first_token}\n"
                          "Red: all candidates zero; green: at least one nonzero; arrows: initial heading")
    prepare_axes(overview_ax, args.radius)
    overview_ax.legend(loc="upper left")
    overview.tight_layout()
    overview.savefig(output / "cluster_overview.png", dpi=args.dpi)
    plt.close(overview)
    pd.DataFrame(manifest).to_csv(output / "scene_manifest.csv", index=False)
    (output / "notes.json").write_text(json.dumps({
        "first_token": first_token, "scenes": len(members),
        "all_zero_scenes": sum(row["max_pdm"] == 0 for row in manifest),
        "trajectory_type": "network proposal poses, not controller-simulated ego states",
        "actor_future": "cached actors, not reactive rollout for each candidate",
        "safety_metrics": "actual selected candidate only; not all candidate causes",
    }, indent=2), encoding="utf-8")
    print(f"Cluster {first_token}: {len(members)} scenes. Images and manifest: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--first-token")
    selection.add_argument("--scene-token")
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--metric-cache", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--radius", type=float, default=40)
    parser.add_argument("--per-page", type=int, default=12)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    if min(args.radius, args.per_page, args.columns, args.dpi) <= 0:
        parser.error("Plot dimensions and pagination values must be positive")
    plot_cluster(args)


if __name__ == "__main__":
    main()