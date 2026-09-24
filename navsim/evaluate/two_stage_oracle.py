from copy import copy, deepcopy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy


METRIC_COLUMNS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
)


def score_without_extended_comfort(result: pd.DataFrame) -> float:
    if len(result) != 1:
        raise ValueError("Expected exactly one candidate result")

    row = result.iloc[0]
    values = np.asarray(row["weighted_metrics"], dtype=np.float64)
    weights = np.asarray(row["weighted_metrics_array"], dtype=np.float64)
    if values.shape != (len(WeightedMetricIndex),) or weights.shape != values.shape:
        raise ValueError("Unexpected weighted metric shape")

    mask = np.arange(len(weights)) != WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    multiplier = float(row["multiplicative_metrics_prod"])
    if not np.isfinite(multiplier) or not np.isfinite(values[mask]).all() or not np.isfinite(weights[mask]).all():
        raise ValueError("Non-finite evaluator metrics")
    if (weights[mask] < 0).any() or weights[mask].sum() <= 0:
        raise ValueError("Expected positive retained metric weight")

    return float(multiplier * np.dot(values[mask], weights[mask]) / weights[mask].sum())


def evaluate_scene_proposals(
    metric_cache: MetricCache,
    proposals: np.ndarray,
    model_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    expected_shape = (64, model_sampling.num_poses, 3)
    if proposals.shape != expected_shape or not np.isfinite(proposals).all():
        raise ValueError(f"Expected finite scene proposals with shape {expected_shape}, got {proposals.shape}")

    scores = np.empty(64, dtype=np.float64)
    metrics: List[Dict[str, float]] = []
    for proposal_index, poses in enumerate(proposals):
        candidate_cache = copy(metric_cache)
        candidate_cache.observation = deepcopy(metric_cache.observation)
        result, _ = pdm_score(
            metric_cache=candidate_cache,
            model_trajectory=Trajectory(poses, model_sampling),
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy,
        )
        scores[proposal_index] = score_without_extended_comfort(result)
        metric_values = {column: float(result.iloc[0][column]) for column in METRIC_COLUMNS}
        if not np.isfinite(list(metric_values.values())).all():
            raise ValueError(f"Non-finite metrics for proposal {proposal_index}")
        metrics.append(metric_values)

    return scores, metrics


def proposal_endpoints(proposals: np.ndarray, initial_pose: StateSE2) -> np.ndarray:
    if proposals.ndim != 3 or proposals.shape[1] < 1 or proposals.shape[2] != 3 or not np.isfinite(proposals).all():
        raise ValueError("Expected finite proposal poses of shape (N, T, 3)")
    local_endpoints = [StateSE2.deserialize(pose) for pose in proposals[:, -1]]
    absolute_endpoints = relative_to_absolute_poses(initial_pose, local_endpoints)
    return np.array([[pose.x, pose.y] for pose in absolute_endpoints], dtype=np.float64)


def stage_two_oracles(proposal_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if (
        proposal_scores.ndim != 2
        or proposal_scores.shape[0] < 1
        or proposal_scores.shape[1] != 64
        or not np.isfinite(proposal_scores).all()
    ):
        raise ValueError("Expected a complete finite (J, 64) stage-two score matrix")
    best_indices = np.argmax(proposal_scores, axis=1)
    return proposal_scores[np.arange(proposal_scores.shape[0]), best_indices], best_indices


def gaussian_weights(
    endpoints: np.ndarray,
    followup_tokens: List[str],
    followup_starts: np.ndarray,
    proposal_sampling: TrajectorySampling,
    sigma_squared: float = 0.1,
) -> np.ndarray:
    if (
        endpoints.ndim != 2
        or endpoints.shape[1] != 2
        or endpoints.shape[0] < 1
        or followup_starts.shape != (len(followup_tokens), 2)
        or not followup_tokens
        or len(set(followup_tokens)) != len(followup_tokens)
        or not np.isfinite(endpoints).all()
        or not np.isfinite(followup_starts).all()
        or sigma_squared <= 0
    ):
        raise ValueError("Expected finite endpoints and unique nonempty follow-up scenes")

    followup_rows = pd.DataFrame(
        followup_starts,
        columns=["start_point_x", "start_point_y"],
        index=pd.Index(followup_tokens, name="token"),
    )
    aggregator = SceneAggregator("", "", pd.DataFrame(), proposal_sampling, sigma_squared=sigma_squared)
    weights = [
        aggregator.calculate_pseudo_closed_loop_weights(
            pd.Series({"endpoint_x": endpoint[0], "endpoint_y": endpoint[1]}), followup_rows
        )["weight"].to_numpy(dtype=np.float64)
        for endpoint in endpoints
    ]
    return np.stack(weights)


def compare_oracles(stage_one_scores: np.ndarray, stage_two_scores: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    if (
        stage_one_scores.ndim != 1
        or stage_one_scores.size < 1
        or stage_two_scores.ndim != 1
        or stage_two_scores.size < 1
        or weights.shape != (stage_one_scores.size, stage_two_scores.size)
        or not np.isfinite(stage_one_scores).all()
        or not np.isfinite(stage_two_scores).all()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.allclose(weights.sum(axis=1), 1.0)
    ):
        raise ValueError("Expected complete finite scores and normalized stage-two weights")

    downstream_values = weights @ stage_two_scores
    total_scores = stage_one_scores * downstream_values
    short_index = int(np.argmax(stage_one_scores))
    long_index = int(np.argmax(total_scores))
    short_score = float(total_scores[short_index])
    long_score = float(total_scores[long_index])
    return {
        "downstream_values": downstream_values,
        "short_index": short_index,
        "long_index": long_index,
        "short_score": short_score,
        "long_score": long_score,
        "gain": long_score - short_score,
        "disagreement": short_index != long_index,
    }


def select_now_mappings(raw_mapping: Sequence, max_mappings: int = 0) -> List[Tuple[str, List[str]]]:
    if max_mappings < 0:
        raise ValueError("max_mappings must be non-negative")

    selected = []
    for original_token, _, followup_pairs in raw_mapping[: max_mappings or None]:
        followup_tokens = [pair[0] for pair in followup_pairs]
        if not followup_tokens or len(set(followup_tokens)) != len(followup_tokens):
            raise ValueError(f"Expected unique nonempty follow-up scenes for {original_token}")
        selected.append((original_token, followup_tokens))

    if not selected or len({original_token for original_token, _ in selected}) != len(selected):
        raise ValueError("Expected unique nonempty original now scenes")
    return selected