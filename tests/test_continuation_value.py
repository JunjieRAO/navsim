import numpy as np
import pytest
import pandas as pd
from types import SimpleNamespace
from unittest.mock import Mock, patch
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import box

from navsim.evaluate.continuation_value import (
    CONTINUATION_MODES, _proposal_history, analyze_continuation_scores, continuation_values, generate_continuations,
    score_proposal_continuations, simulate_stage_one_proposals,
)
from navsim.evaluate.two_stage_oracle import METRIC_COLUMNS
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def test_k_prefixes_share_scores_and_can_change_proposal_ranking():
    scores = np.full((64, 64), 0.1)
    scores[0, 15] = 0.9
    scores[1, 31] = 0.95
    scores[0, 63] = 0.99

    values = continuation_values(scores, (16, 32, 64))

    assert values.shape == (64, 3)
    np.testing.assert_array_equal(values.argmax(axis=0), [0, 1, 0])
    np.testing.assert_allclose(values[0], [0.3, 0.2, (0.9 + 0.99 + 14 * 0.1) / 16])
    assert values[0, 1] < values[0, 0]


def test_top_quarter_rewards_broad_high_quality_continuations():
    scores = np.zeros((2, 32))
    scores[0, :8] = 0.92
    scores[1, 0] = 0.91
    scores[1, 1:8] = 0.4

    values = continuation_values(scores, (32,))

    np.testing.assert_allclose(values[:, 0], [0.92, (0.91 + 7 * 0.4) / 8])
    np.testing.assert_allclose(continuation_values(scores, (1, 5))[:, 0], [0.92, 0.91])
    assert continuation_values(scores, (5,))[1, 0] == pytest.approx((0.91 + 0.4) / 2)


@pytest.mark.parametrize("ks", [(), (0,), (65,), (16, 16), (16.5,), (True,)])
def test_rejects_invalid_k(ks):
    with pytest.raises(ValueError):
        continuation_values(np.ones((64, 64)), ks)


@pytest.mark.parametrize("scores", [np.zeros((0, 16)), np.full((64, 16), np.nan), np.zeros((16,))])
def test_rejects_incomplete_scores(scores):
    with pytest.raises(ValueError):
        continuation_values(scores, (16,))


def test_analysis_tracks_ranking_flips_and_best_continuation_indices():
    scores = np.full((64, 64), 0.1)
    scores[0, 15] = 0.9
    scores[1, 31] = 0.95
    scores[0, 63] = 0.99

    targets, rows = analyze_continuation_scores(scores, np.ones(64), (16, 32, 64))

    np.testing.assert_array_equal(targets["v2_top_indices"], [0, 1, 0])
    np.testing.assert_array_equal(targets["top_counts"], [4, 8, 16])
    np.testing.assert_allclose(targets["best_mode_scores"][0], [0.9, 0.9, 0.99])
    np.testing.assert_array_equal(targets["best_mode_indices"][0], [15, 15, 63])
    assert [row["top_count"] for row in rows] == [4, 8, 16]
    assert [row["v2_top_agrees_with_max_k"] for row in rows] == [True, False, True]
    assert [row["e1_v2_top_index"] for row in rows] == [0, 1, 0]
    assert rows[0]["mean_abs_v2_change"] > 0
    assert rows[-1]["mean_abs_v2_change"] == 0
    with pytest.raises(ValueError, match="increasing"):
        analyze_continuation_scores(scores, np.ones(64), (32, 16))


def test_constant_scores_report_undefined_rank_correlation():
    _, rows = analyze_continuation_scores(np.zeros((64, 32)), np.ones(64), (16, 32))
    assert [row["spearman_vs_max_k"] for row in rows] == [None, None]
    assert rows[0]["v2_top_ties"] == 64
    assert rows[0]["v2_value_span"] == 0
    assert rows[0]["v2_saturated_fraction"] == 0


def test_continuations_are_kinematic_route_modes_with_stable_prefixes():
    route = PDMPath([StateSE2(float(position), 0.0, 0.0) for position in range(0, 205, 5)])
    state = np.zeros(11)
    state[3] = 6.0
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)

    full = generate_continuations(state, route, sampling)
    first_sixteen = generate_continuations(state, route, sampling, count=16)

    assert len(CONTINUATION_MODES) == 64
    assert full.shape == (64, 8, 3)
    assert np.isfinite(full).all()
    np.testing.assert_allclose(full[:16], first_sixteen)
    assert np.all(full[:, 0, 0] >= 0)
    assert full[0, -1, 0] > full[3, -1, 0]
    assert full[4, -1, 1] < full[8, -1, 1]


def test_continuation_rejects_invalid_endpoint():
    route = PDMPath([StateSE2(0, 0, 0), StateSE2(30, 0, 0)])
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    with pytest.raises(ValueError):
        generate_continuations(np.zeros(3), route, sampling)


def test_shifted_gt_scoring_keeps_reference_fixed_and_penalizes_terminal_overlap():
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    scoring_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = SimpleNamespace(proposal_sampling=scoring_sampling)
    simulator.simulate_proposals = Mock(side_effect=lambda states, initial: states)
    reference_states = []
    histories = []

    def score_pair(states, observation, *args, human_past_trajectory):
        reference_states.append(states[0].copy())
        histories.append(human_past_trajectory)
        row = {key: 1.0 for key in METRIC_COLUMNS}
        row.update({
            "weighted_metrics": np.array([0.8, 1.0, 1.0, 1.0, np.nan]),
            "weighted_metrics_array": np.array([5.0, 5.0, 2.0, 2.0, 2.0]),
            "multiplicative_metrics_prod": 1.0,
        })
        return [pd.DataFrame([row]), pd.DataFrame([row])]

    scorer = SimpleNamespace(proposal_sampling=scoring_sampling, score_proposals=score_pair)
    empty = SimpleNamespace(tracked_objects=SimpleNamespace(tracked_objects=[]))
    observation = SimpleNamespace(detections_tracks=[empty])
    route = PDMPath([StateSE2(0, 0, 0), StateSE2(100, 0, 0)])
    original = SimpleNamespace(
        timepoint=TimePoint(10000000),
        ego_state=SimpleNamespace(car_footprint=SimpleNamespace(vehicle_parameters=object())),
        centerline=route, route_lane_ids=["lane"],
    )
    shifted = SimpleNamespace(
        timepoint=TimePoint(14000000), observation=observation,
        drivable_area_map=object(), map_parameters=object(),
    )
    simulated_stage_one = np.zeros((1, 41, 11))
    simulated_stage_one[0, -1, 3] = 5.0
    fake_ego = SimpleNamespace(car_footprint=SimpleNamespace(geometry=box(0, 0, 2, 2)))
    def simulated_candidate(poses, *args):
        states = np.zeros((41, 11))
        states[-1, 0] = poses[-1, 0]
        return states

    with patch("navsim.evaluate.continuation_value.state_array_to_ego_state", return_value=fake_ego), patch(
        "navsim.evaluate.continuation_value._proposal_history", return_value=object(),
    ), patch("navsim.evaluate.continuation_value._trajectory_states", side_effect=simulated_candidate):
        scores, metrics, overlaps = score_proposal_continuations(
            simulated_stage_one, original, shifted, sampling, simulator, scorer, count=3,
        )

    assert scores.shape == (1, 3)
    np.testing.assert_allclose(scores, (5 * 0.8 + 5 + 2 + 2) / 14)
    assert metrics.shape == (1, 3, len(METRIC_COLUMNS))
    assert not overlaps[0]
    assert len(histories) == 3
    assert simulator.simulate_proposals.call_count == 1
    assert all(np.array_equal(reference_states[0], states) for states in reference_states)

    colliding_agent = SimpleNamespace(
        tracked_object_type=TrackedObjectType.VEHICLE,
        box=SimpleNamespace(geometry=box(0, 0, 2, 2)),
    )
    shifted.observation.detections_tracks[0].tracked_objects.tracked_objects.append(colliding_agent)
    with patch("navsim.evaluate.continuation_value.state_array_to_ego_state", return_value=fake_ego):
        zero_scores, _, collision_flags = score_proposal_continuations(
            simulated_stage_one, original, shifted, sampling, simulator, scorer, count=3,
        )
    np.testing.assert_array_equal(zero_scores, 0)
    assert collision_flags[0]

    shifted.timepoint = TimePoint(13500000)
    with pytest.raises(ValueError, match="four seconds"):
        score_proposal_continuations(simulated_stage_one, original, shifted, sampling, simulator, scorer, count=3)


def test_stage_one_history_uses_proposal_states_at_the_shifted_timestamps():
    states = np.zeros((41, 11))
    states[:, 0] = np.arange(41) * 0.5
    states[:, 3] = 5.0

    history = _proposal_history(states, TimePoint(14000000), get_pacifica_parameters(), 0.1)

    assert history.start_time.time_us == 12500000
    assert history.end_time.time_us == 14000000
    assert history.get_state_at_time(TimePoint(13500000)).rear_axle.x == pytest.approx(17.5)


def test_stage_one_simulates_all_proposals_and_rejects_missing_candidates():
    sampling = TrajectorySampling(num_poses=8, interval_length=0.5)
    evaluator_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    cache = SimpleNamespace(ego_state=SimpleNamespace(time_point=TimePoint(10000000)))
    simulator = SimpleNamespace(proposal_sampling=evaluator_sampling)
    simulator.simulate_proposals = lambda states, initial: states
    proposals = np.zeros((64, 8, 3))
    proposals[:, 0, 0] = np.arange(64)

    with patch("navsim.evaluate.continuation_value._trajectory_states", side_effect=lambda poses, *args: np.full(
        (41, 11), poses[0, 0], dtype=np.float64,
    )) as transform:
        states = simulate_stage_one_proposals(proposals, cache, sampling, simulator)

    assert states.shape == (64, 41, 11)
    assert transform.call_count == 64
    assert states[-1, -1, 0] == 63
    with pytest.raises(ValueError, match="64"):
        simulate_stage_one_proposals(proposals[:-1], cache, sampling, simulator)