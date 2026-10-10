import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SAFETY_METRICS = [
    "no_at_fault_collisions", "drivable_area_compliance",
    "driving_direction_compliance", "traffic_light_compliance",
]


def distribution(frame, grouping):
    result = []
    for label, group in frame.groupby(grouping, observed=True):
        failed = group[group.all_candidates_zero]
        result.append({
            "distance_group": str(label), "scenes": len(group), "all_zero_scenes": len(failed),
            "all_zero_rate": len(failed) / len(group),
            "all_zero_weight_sum": float(failed.weight.sum()),
            "weight_sum": float(group.weight.sum()),
            **{f"selected_zero_{metric}": int((failed[metric] == 0).sum()) for metric in SAFETY_METRICS},
        })
    return pd.DataFrame(result)


def analyze(experiment):
    config = yaml.safe_load((experiment / "code/hydra/config.yaml").read_text())
    scores = pd.read_csv(experiment / "gt_actual_top1_scores.csv").set_index("token")
    candidates = pd.read_csv(experiment / "actual_candidate_scores.csv")
    maxima = candidates.groupby("token").pdm_score.agg(["max", "count"])
    if not np.isfinite(candidates.pdm_score).all() or not (maxima["count"] == 64).all():
        raise ValueError("Expected finite scores for 64 candidates per stage-two scene")
    branches = pd.read_csv(experiment / "gt_actual_top1_branches.csv").set_index("first_token")
    records = []
    for original, previous, pairs in config["train_test_split"]["reactive_all_mapping"]:
        for first_token, pair_index in ((original, 0), (previous, 1)):
            first = scores.loc[first_token]
            for pair in pairs:
                token = pair[pair_index]
                second = scores.loc[token]
                if token not in maxima.index:
                    raise ValueError("Stage-two scene without exhaustive candidate scores")
                records.append({
                    "token": token, "first_token": first_token, "mapping_original": original,
                    "distance_to_gt_endpoint_m": float(np.hypot(
                        second.start_point_x - first.endpoint_x, second.start_point_y - first.endpoint_y
                    )),
                    "time_gap_s": float(second.start_time - first.start_time),
                    "fallback": bool(branches.loc[first_token, "fallback"]),
                    "weight": float(second.weight), "max_candidate_pdm_score": float(maxima.loc[token, "max"]),
                    "all_candidates_zero": bool(maxima.loc[token, "max"] == 0),
                    "selected_final_score": float(second.score),
                    **{metric: float(second[metric]) for metric in SAFETY_METRICS},
                })
    frame = pd.DataFrame(records).sort_values("distance_to_gt_endpoint_m").reset_index(drop=True)
    if frame.token.duplicated().any() or set(frame.token) != set(maxima.index):
        raise ValueError("Mappings must cover each candidate scene exactly once")
    frame.insert(0, "distance_rank", np.arange(1, len(frame) + 1))
    frame["distance_bin"] = pd.cut(frame.distance_to_gt_endpoint_m,
        [0, .5, 1, 2, 3, 5, 10, 20, np.inf], right=False)
    frame["distance_decile"] = pd.qcut(frame.distance_to_gt_endpoint_m, 10)
    bins = distribution(frame, "distance_bin")
    deciles = distribution(frame, "distance_decile")
    frame["within_branch_rank"] = frame.groupby("first_token").distance_to_gt_endpoint_m.rank(method="first")
    frame["within_branch_rank_fraction"] = (
        (frame.within_branch_rank - .5) / frame.groupby("first_token").token.transform("count")
    )
    frame["within_branch_quartile"] = pd.cut(
        frame.within_branch_rank_fraction, [0, .25, .5, .75, 1], include_lowest=True
    )
    quartiles = distribution(frame, "within_branch_quartile")
    mixed_differences = []
    for _, group in frame.groupby("first_token"):
        if group.all_candidates_zero.nunique() == 2:
            mixed_differences.append(
                group.loc[group.all_candidates_zero, "distance_to_gt_endpoint_m"].mean()
                - group.loc[~group.all_candidates_zero, "distance_to_gt_endpoint_m"].mean()
            )
    quantiles = {}
    for label, subset in (("all", frame), ("all_zero", frame[frame.all_candidates_zero]),
                          ("nonzero", frame[~frame.all_candidates_zero])):
        distances = subset.distance_to_gt_endpoint_m
        quantiles[label] = {
            "scenes": len(subset), "mean_m": float(distances.mean()),
            "quantiles_m": {str(percentile): float(distances.quantile(percentile))
                            for percentile in (0, .1, .25, .5, .75, .9, .95, .99, 1)},
        }
    failed = frame[frame.all_candidates_zero]
    summary = {
        "distance_definition": "global XY stage-two initial rear axle to corresponding stage-one GT endpoint",
        "cause_limit": "safety metrics describe selected top1 only; candidate archive stores total PDM only",
        "quantiles": quantiles,
        "failed_cumulative_counts": {str(threshold): int((failed.distance_to_gt_endpoint_m < threshold).sum())
                                     for threshold in (.5, 1, 2, 3, 5, 10, 20)},
        "failure_distance_spearman": float(frame.distance_to_gt_endpoint_m.corr(
            frame.all_candidates_zero.astype(float), method="spearman")),
        "failed_selected_safety_zero_counts": {metric: int((failed[metric] == 0).sum()) for metric in SAFETY_METRICS},
        "failed_selected_collision_values": failed.no_at_fault_collisions.value_counts().to_dict(),
        "nearest_scene_all_zero_count": int(frame.groupby("first_token").head(1).all_candidates_zero.sum()),
        "mixed_branch_comparison": {
            "branches": len(mixed_differences),
            "failed_closer_branches": int(sum(value < 0 for value in mixed_differences)),
            "mean_failed_minus_nonzero_distance_m": float(np.mean(mixed_differences)) if mixed_differences else None,
        },
        "failed_selected_safety_combinations": failed.apply(
            lambda row: "+".join(metric for metric in SAFETY_METRICS if row[metric] == 0), axis=1
        ).value_counts().to_dict(),
        "failed_weight_mass_by_distance": {str(threshold): float(
            failed.loc[failed.distance_to_gt_endpoint_m < threshold, "weight"].sum() / failed.weight.sum()
        ) for threshold in (1, 2, 3, 5, 10)},
    }
    output = experiment / "distance_diagnostics"
    output.mkdir(exist_ok=True)
    frame.to_csv(output / "all_scenes_sorted_by_gt_distance.csv", index=False)
    failed.to_csv(output / "all_zero_scenes_sorted_by_gt_distance.csv", index=False)
    bins.to_csv(output / "distance_bins.csv", index=False)
    deciles.to_csv(output / "distance_deciles.csv", index=False)
    quartiles.to_csv(output / "within_branch_distance_quartiles.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nDISTANCE BINS\n", bins.to_string(index=False))
    print("\nSORTED DISTANCE DECILES\n", deciles.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    analyze(parser.parse_args().experiment)