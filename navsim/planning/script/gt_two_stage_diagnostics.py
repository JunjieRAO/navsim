import numpy as np
import pandas as pd


SAFETY_METRICS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
)


def summarize_branches(scores, mappings, sigma_squared=0.1):
    frame = scores.set_index("token")
    branches = []
    safety = []
    for (original, previous), pairs in mappings.items():
        for first_token, pair_index in ((original, 0), (previous, 1)):
            first = frame.loc[first_token]
            second = frame.loc[[pair[pair_index] for pair in pairs]]
            squared_distance = (
                (second.start_point_x - first.endpoint_x) ** 2
                + (second.start_point_y - first.endpoint_y) ** 2
            )
            kernel_sum = float(np.exp(-squared_distance / (2 * sigma_squared)).sum())
            fallback = bool(np.isclose(kernel_sum, 0) or np.isnan(kernel_sum))
            weights = second.weight.to_numpy(dtype=float)
            stage_two = float(np.average(second.score, weights=weights))
            branches.append({
                "mapping_original": original, "mapping_previous": previous,
                "first_token": first_token, "fallback": fallback,
                "kernel_sum": kernel_sum,
                "nearest_start_distance_m": float(np.sqrt(squared_distance.min())),
                "stage_one_score": float(first.score), "stage_two_score": stage_two,
                "combined_score": float(first.score * stage_two),
            })
            for stage, group, group_weights in (
                ("stage_one", frame.loc[[first_token]], np.ones(1)),
                ("stage_two", second, weights / weights.sum()),
            ):
                zero = np.isclose(group.score.to_numpy(dtype=float), 0, atol=1e-12, rtol=0)
                for metric in SAFETY_METRICS:
                    metric_zero = group[metric].to_numpy(dtype=float) == 0
                    for row_index, token in enumerate(group.index):
                        if zero[row_index] and metric_zero[row_index]:
                            safety.append({
                                "first_token": first_token, "token": token,
                                "fallback": fallback, "stage": stage, "metric": metric,
                                "weight": float(group_weights[row_index]),
                            })
                unattributed = zero & ~np.any(
                    group[list(SAFETY_METRICS)].to_numpy(dtype=float) == 0, axis=1
                )
                for row_index in np.flatnonzero(unattributed):
                    safety.append({
                        "first_token": first_token, "token": group.index[row_index],
                        "fallback": fallback, "stage": stage,
                        "metric": "zero_without_safety_multiplier",
                        "weight": float(group_weights[row_index]),
                    })
    branches = pd.DataFrame(branches)
    safety = pd.DataFrame(safety, columns=[
        "first_token", "token", "fallback", "stage", "metric", "weight"
    ])
    summary = {}
    for label, subset in (
        ("all", branches),
        ("non_fallback", branches[~branches.fallback]),
        ("fallback", branches[branches.fallback]),
    ):
        causes = safety[safety.first_token.isin(subset.first_token)]
        summary[label] = {
            "branches": len(subset),
            **{metric: float(subset[metric].mean()) if len(subset) else None
               for metric in ("stage_one_score", "stage_two_score", "combined_score")},
            "zero_score_branches": {
                metric: int(np.isclose(subset[metric], 0, atol=1e-12, rtol=0).sum())
                for metric in ("stage_one_score", "stage_two_score", "combined_score")
            },
            "safety_zero_causes": {
                stage: {
                    metric: {
                        "zero_scene_count": int(len(group)),
                        "weighted_zero_mass_per_branch": float(group.weight.sum() / len(subset)),
                    }
                    for metric, group in causes[causes.stage == stage].groupby("metric")
                }
                for stage in ("stage_one", "stage_two")
            },
        }
    return summary, branches, safety