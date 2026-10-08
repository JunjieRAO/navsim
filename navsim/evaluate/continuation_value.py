from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import Point
from scipy.stats import rankdata

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.evaluate.two_stage_oracle import METRIC_COLUMNS, score_without_extended_comfort
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_ego_state,
    state_array_to_ego_states,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


@dataclass(frozen=True)
class ContinuationMode:
    acceleration: float
    offset: float
    steering_gain: float = 1.0
    straight: bool = False


CONTINUATION_MODES = (
    tuple(
        ContinuationMode(acceleration, offset)
        for offset in (0.0, -0.5, 0.5, -1.0, 1.0)
        for acceleration in (0.0, -1.5, 1.0, -3.0)
    )
    + (
        ContinuationMode(-6.0, 0.0),
        ContinuationMode(0.0, 0.0, 1.6),
        ContinuationMode(0.0, 0.0, straight=True),
        ContinuationMode(1.5, 0.0),
    )
    + tuple(
        ContinuationMode(acceleration, offset)
        for offset in (-1.5, 1.5)
        for acceleration in (0.0, -2.0, 1.0, 2.0)
    )
    + tuple(
        ContinuationMode(acceleration, offset)
        for offset in (-2.0, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 2.0)
        for acceleration in (-4.0, -0.75, 0.5, 2.5)
    )
)


def generate_continuations(
    endpoint_state: np.ndarray,
    centerline: PDMPath,
    sampling: TrajectorySampling,
    count: int = 64,
) -> np.ndarray:
    """Build local (x, y, heading) paths from a proposal endpoint and the route alone."""
    endpoint_state = np.asarray(endpoint_state, dtype=np.float64)
    if endpoint_state.shape != (StateIndex.size(),) or not np.isfinite(endpoint_state).all():
        raise ValueError("Expected a finite simulated endpoint state")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(CONTINUATION_MODES):
        raise ValueError("Continuation count must be between 1 and 64")
    if sampling.num_poses < 1 or sampling.interval_length <= 0 or centerline.length <= 0:
        raise ValueError("Expected a nonempty route and positive trajectory sampling")

    start_x, start_y, start_heading = endpoint_state[StateIndex.STATE_SE2]
    initial_speed = max(0.0, float(
        endpoint_state[StateIndex.VELOCITY_X] * np.cos(start_heading)
        + endpoint_state[StateIndex.VELOCITY_Y] * np.sin(start_heading)
    ))
    substeps = max(1, int(np.ceil(sampling.interval_length / 0.1)))
    step = sampling.interval_length / substeps
    trajectories = np.empty((count, sampling.num_poses, 3), dtype=np.float64)

    for mode_index, mode in enumerate(CONTINUATION_MODES[:count]):
        x, y, heading, speed = start_x, start_y, start_heading, initial_speed
        for pose_index in range(sampling.num_poses):
            for _ in range(substeps):
                speed = np.clip(speed + mode.acceleration * step, 0.0, 25.0)
                if not mode.straight:
                    progress = centerline.project(Point(x, y))
                    preview = max(4.0, speed * 0.8 + 2.0)
                    reference = centerline.interpolate(np.array([progress + preview]), as_array=True)[0]
                    target_x = reference[0] - mode.offset * np.sin(reference[2])
                    target_y = reference[1] + mode.offset * np.cos(reference[2])
                    beyond_route = max(0.0, progress + preview - centerline.length)
                    target_x += beyond_route * np.cos(reference[2])
                    target_y += beyond_route * np.sin(reference[2])
                    target_distance = max(3.0, np.hypot(target_x - x, target_y - y))
                    error = np.arctan2(target_y - y, target_x - x) - heading
                    curvature = np.clip(
                        mode.steering_gain * 2.0 * np.sin(error) / target_distance, -0.16, 0.16,
                    )
                    heading += speed * curvature * step
                x += speed * np.cos(heading) * step
                y += speed * np.sin(heading) * step

            delta_x, delta_y = x - start_x, y - start_y
            trajectories[mode_index, pose_index, 0] = delta_x * np.cos(start_heading) + delta_y * np.sin(start_heading)
            trajectories[mode_index, pose_index, 1] = -delta_x * np.sin(start_heading) + delta_y * np.cos(start_heading)
            trajectories[mode_index, pose_index, 2] = np.arctan2(
                np.sin(heading - start_heading), np.cos(heading - start_heading),
            )

    return trajectories


def _trajectory_states(poses, sampling, simulator, initial_state):
    trajectory = transform_trajectory(Trajectory(poses, sampling), initial_state)
    return get_trajectory_as_array(trajectory, simulator.proposal_sampling, initial_state.time_point)


def simulate_stage_one_proposals(proposals, original_cache, sampling, simulator):
    proposals = np.asarray(proposals, dtype=np.float64)
    if proposals.shape != (64, sampling.num_poses, 3) or not np.isfinite(proposals).all():
        raise ValueError("Expected 64 finite four-second DrivoR proposals")
    if not np.isclose(sampling.time_horizon, simulator.proposal_sampling.time_horizon):
        raise ValueError("Stage-one simulation must end at the proposal endpoint")
    states = np.stack([
        _trajectory_states(poses, sampling, simulator, original_cache.ego_state)
        for poses in proposals
    ])
    simulated = simulator.simulate_proposals(states, original_cache.ego_state)
    if simulated.shape != (64, simulator.proposal_sampling.num_poses + 1, StateIndex.size()):
        raise ValueError("Unexpected stage-one simulation shape")
    if not np.isfinite(simulated).all():
        raise ValueError("Stage-one simulation produced non-finite states")
    return simulated


def _proposal_history(stage_one_states, start_time, vehicle_parameters, interval_length):
    history_step = int(round(0.5 / interval_length))
    history_indices = np.arange(len(stage_one_states) - 1 - 3 * history_step, len(stage_one_states), history_step)
    if len(history_indices) != 4 or history_indices[0] < 0:
        raise ValueError("Stage-one history must cover the last 1.5 seconds")
    time_points = [
        TimePoint(start_time.time_us + int(round((index - len(stage_one_states) + 1) * interval_length * 1e6)))
        for index in history_indices
    ]
    states = state_array_to_ego_states(stage_one_states[history_indices], time_points, vehicle_parameters)
    return InterpolatedTrajectory(states)


def _terminal_agent_overlap(ego_state, observation):
    ego_box = ego_state.car_footprint.geometry
    return any(
        tracked.tracked_object_type in AGENT_TYPES and tracked.box.geometry.intersects(ego_box)
        for tracked in observation.detections_tracks[0].tracked_objects.tracked_objects
    )


def score_proposal_continuations(
    stage_one_states,
    original_cache,
    shifted_cache,
    sampling,
    simulator,
    scorer,
    count=64,
):
    """Score each future-GT continuation from its own simulated t+4 ego state."""
    stage_one_states = np.asarray(stage_one_states, dtype=np.float64)
    expected_shape = (simulator.proposal_sampling.num_poses + 1, StateIndex.size())
    if (stage_one_states.ndim != 3 or stage_one_states.shape[1:] != expected_shape
            or not stage_one_states.shape[0] or not np.isfinite(stage_one_states).all()):
        raise ValueError("Expected finite simulated stage-one states")
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise ValueError("Simulator and scorer sampling must match")
    if not np.isclose(sampling.time_horizon, 4.0) or not np.isclose(
        shifted_cache.timepoint.time_s - original_cache.timepoint.time_s, 4.0, atol=0.02,
    ):
        raise ValueError("GT observation must start four seconds after the original scene")
    if not 1 <= count <= len(CONTINUATION_MODES):
        raise ValueError("Continuation count must be between 1 and 64")

    scores = np.empty((stage_one_states.shape[0], count), dtype=np.float64)
    metrics = np.zeros((stage_one_states.shape[0], count, len(METRIC_COLUMNS)), dtype=np.float64)
    terminal_overlaps = np.zeros(stage_one_states.shape[0], dtype=bool)
    vehicle_parameters = original_cache.ego_state.car_footprint.vehicle_parameters
    for proposal_index, proposal_states in enumerate(stage_one_states):
        endpoint = proposal_states[-1]
        ego_state = state_array_to_ego_state(endpoint, shifted_cache.timepoint, vehicle_parameters)
        terminal_overlaps[proposal_index] = _terminal_agent_overlap(ego_state, shifted_cache.observation)
        if terminal_overlaps[proposal_index]:
            scores[proposal_index] = 0.0
            continue

        history = _proposal_history(
            proposal_states, shifted_cache.timepoint, vehicle_parameters, simulator.proposal_sampling.interval_length,
        )
        continuations = generate_continuations(endpoint, original_cache.centerline, sampling, count)
        trajectory_states = np.stack([
            _trajectory_states(poses, sampling, simulator, ego_state) for poses in continuations
        ])
        simulated_candidates = simulator.simulate_proposals(trajectory_states, ego_state)
        reference_simulated = simulated_candidates[0]

        for continuation_index, simulated in enumerate(simulated_candidates):
            result = scorer.score_proposals(
                np.stack([reference_simulated, simulated]),
                shifted_cache.observation,
                original_cache.centerline,
                original_cache.route_lane_ids,
                shifted_cache.drivable_area_map,
                shifted_cache.map_parameters,
                human_past_trajectory=history,
            )[1]
            scores[proposal_index, continuation_index] = score_without_extended_comfort(result)
            metrics[proposal_index, continuation_index] = [float(result.iloc[0][key]) for key in METRIC_COLUMNS]

    if not np.isfinite(scores).all() or not np.isfinite(metrics).all():
        raise ValueError("Non-finite continuation metrics")
    return scores, metrics, terminal_overlaps


def continuation_values(scores: np.ndarray, k_values: Sequence[int]) -> np.ndarray:
    """Average the highest quarter of each proposal's K-prefix scores."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not all(scores.shape) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty finite (proposals, continuations) score matrix")

    k_values = tuple(k_values)
    if not k_values or any(
        isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= scores.shape[1]
        for k in k_values
    ) or len(set(k_values)) != len(k_values):
        raise ValueError("K values must be unique integers between 1 and the number of continuations")

    values = np.empty((scores.shape[0], len(k_values)), dtype=np.float64)
    for column, k in enumerate(k_values):
        top_count = (k + 3) // 4
        values[:, column] = np.partition(scores[:, :k], k - top_count, axis=1)[:, -top_count:].mean(axis=1)
    return values


def analyze_continuation_scores(scores: np.ndarray, e1_scores: np.ndarray, k_values: Sequence[int]):
    """Compare nested top-quarter targets without recomputing scores."""
    k_values = tuple(k_values)
    values = continuation_values(scores, k_values)
    e1_scores = np.asarray(e1_scores, dtype=np.float64)
    if (e1_scores.shape != (values.shape[0],) or not np.isfinite(e1_scores).all()
            or tuple(sorted(k_values)) != k_values):
        raise ValueError("Expected finite E1 values and strictly increasing K prefixes")

    scores = np.asarray(scores, dtype=np.float64)
    best_modes = np.stack([np.argmax(scores[:, :k], axis=1) for k in k_values], axis=1)
    best_mode_scores = np.stack([np.max(scores[:, :k], axis=1) for k in k_values], axis=1)
    top_counts = np.asarray([(k + 3) // 4 for k in k_values], dtype=np.int64)
    combined = e1_scores[:, None] * values
    v2_winners = np.argmax(values, axis=0)
    combined_winners = np.argmax(combined, axis=0)
    reference = values[:, -1]
    rows = []
    for column, k in enumerate(k_values):
        current = values[:, column]
        if np.ptp(current) == 0 or np.ptp(reference) == 0:
            spearman = None
        else:
            spearman = float(np.corrcoef(rankdata(current), rankdata(reference))[0, 1])
        rows.append({
            "k": int(k),
            "top_count": int(top_counts[column]),
            "v2_top_index": int(v2_winners[column]),
            "e1_v2_top_index": int(combined_winners[column]),
            "v2_top_agrees_with_max_k": bool(v2_winners[column] == v2_winners[-1]),
            "e1_v2_top_agrees_with_max_k": bool(combined_winners[column] == combined_winners[-1]),
            "v2_top_ties": int(np.count_nonzero(np.isclose(current, current.max(), rtol=0, atol=1e-8))),
            "v2_value_span": float(np.ptp(current)),
            "v2_saturated_fraction": float(np.mean(current >= 1.0 - 1e-8)),
            "max_abs_v2_change": float(np.max(np.abs(current - reference))),
            "mean_abs_v2_change": float(np.mean(np.abs(current - reference))),
            "spearman_vs_max_k": spearman,
        })
    return {
        "k_values": np.asarray(k_values, dtype=np.int64),
        "top_counts": top_counts,
        "v2_values": values,
        "best_mode_indices": best_modes,
        "best_mode_scores": best_mode_scores,
        "e1_scores": e1_scores,
        "e1_v2_values": combined,
        "v2_top_indices": v2_winners,
        "e1_v2_top_indices": combined_winners,
    }, rows