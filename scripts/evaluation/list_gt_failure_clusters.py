import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SAFETY_METRICS = {
    "NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
    "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
}


def coverage_order(clusters):
    labels = list(SAFETY_METRICS)
    paths = {0: ()}
    for index, row in clusters.iterrows():
        mask = sum(1 << position for position, label in enumerate(labels) if row[f"{label}_zero_scenes"] > 0)
        for covered, path in list(paths.items()):
            combined = covered | mask
            proposal = path + (index,)
            if combined not in paths or len(proposal) < len(paths[combined]):
                paths[combined] = proposal
    available = max(paths, key=lambda mask: mask.bit_count())
    representatives = list(paths[available])
    remaining = [index for index in clusters.index if index not in representatives]
    return clusters.loc[representatives + remaining].reset_index(drop=True)


def failure_clusters(config, candidates, scene_scores):
    if not np.isfinite(candidates.pdm_score).all() or (candidates.pdm_score < 0).any():
        raise ValueError("Candidate scores must be finite and nonnegative")
    if candidates.duplicated(["token", "proposal_index"]).any():
        raise ValueError("Duplicate scene/proposal scores")
    grouped = candidates.groupby("token")
    complete = grouped.proposal_index.agg(["count", "min", "max"])
    if not ((complete["count"] == 64) & (complete["min"] == 0) & (complete["max"] == 63)).all():
        raise ValueError("Expected exactly 64 proposal indices (0..63) per candidate scene")
    maxima = grouped.pdm_score.max()
    failed = set(maxima.index[maxima == 0])
    if scene_scores.token.duplicated().any():
        raise ValueError("Duplicate scene scores")
    weights = scene_scores.set_index("token").weight
    if not set(maxima.index).issubset(weights.index):
        raise ValueError("Candidate scenes missing from scene weights")
    candidate_weights = weights.loc[maxima.index]
    if not np.isfinite(candidate_weights).all() or (candidate_weights < 0).any():
        raise ValueError("Scene weights must be finite and nonnegative")
    qualifying = {token for token in failed if weights.loc[token] > 0.001}
    safety = scene_scores.set_index("token")[list(SAFETY_METRICS.values())]
    if not np.isfinite(safety.loc[maxima.index].to_numpy()).all():
        raise ValueError("Safety metrics must be finite")
    rows = []
    seen = set()
    mapped_scenes = set()
    for original, previous, pairs in config["train_test_split"]["reactive_all_mapping"]:
        for token, index in ((original, 0), (previous, 1)):
            if token in seen:
                raise ValueError("Repeated first-stage token in mappings")
            seen.add(token)
            members = [pair[index] for pair in pairs]
            mapped_scenes.update(members)
            failures = [member for member in members if member in failed]
            weighted_failures = [member for member in members if member in qualifying]
            if weighted_failures:
                rows.append({"first_token": token, "mapping_original": original,
                             "mapping_previous": previous, "scenes": len(members),
                             "all_zero_scenes": len(failures),
                             "all_zero_scene_tokens": ";".join(failures),
                             "qualifying_scenes": len(weighted_failures),
                                      "qualifying_scene_tokens": ";".join(weighted_failures),
                                      **{f"{label}_zero_scenes": int((safety.loc[weighted_failures, metric] == 0).sum())
                                          for label, metric in SAFETY_METRICS.items()}})
    if not set(maxima.index).issubset(mapped_scenes):
        raise ValueError("Candidate scenes missing from mappings")
    return coverage_order(pd.DataFrame(rows, columns=["first_token", "mapping_original", "mapping_previous",
                                       "scenes", "all_zero_scenes", "all_zero_scene_tokens",
                                       "qualifying_scenes", "qualifying_scene_tokens",
                                       *[f"{label}_zero_scenes" for label in SAFETY_METRICS]]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--max-clusters", type=int)
    args = parser.parse_args()
    if args.max_clusters is not None and args.max_clusters < 1:
        parser.error("--max-clusters must be positive")
    config = yaml.safe_load((args.experiment / "code/hydra/config.yaml").read_text())
    candidates = pd.read_csv(args.experiment / "actual_candidate_scores.csv")
    scene_scores = pd.read_csv(args.experiment / "gt_actual_top1_scores.csv")
    clusters = failure_clusters(config, candidates, scene_scores)
    output = args.experiment / "cluster_visualizations"
    output.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output / "failure_clusters.csv", index=False)
    (output / "failure_cluster_tokens.txt").write_text(
        "".join(f"{token}\n" for token in clusters.first_token), encoding="utf-8"
    )
    print(
        f"Selected {len(clusters)} clusters containing {int(clusters.qualifying_scenes.sum())} "
        "all-candidate-zero scenes with weight > 0.001 (GT actual-top1 weights)."
    )
    print(f"Token list: {output / 'failure_cluster_tokens.txt'}")
    selected = clusters if args.max_clusters is None else clusters.head(args.max_clusters)
    coverage = {label: int((selected[f"{label}_zero_scenes"] > 0).sum()) for label in SAFETY_METRICS}
    print(f"First {len(selected)} clusters: safety-category cluster coverage {coverage}")
    missing = [label for label, count in coverage.items() if count == 0]
    if missing:
        print(f"WARNING: categories not covered: {', '.join(missing)}; increase --max-clusters "
              "or check whether qualifying scenes of these categories exist.")


if __name__ == "__main__":
    main()