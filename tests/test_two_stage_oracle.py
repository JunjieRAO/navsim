from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.evaluate.two_stage_oracle import (
    METRIC_COLUMNS,
    compare_oracles,
    evaluate_scene_proposals,
    evaluate_scene_proposals_with_history,
    gaussian_weights,
    proposal_endpoints,
    score_with_extended_comfort,
    score_without_extended_comfort,
    select_history_pairs,
    select_now_mappings,
    simulate_fixed_history,
    stage_two_oracles,
    two_frame_extended_comfort,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex


class TestNoExtendedComfortScore(TestCase):
    def test_uses_post_filter_metrics_and_excludes_extended_comfort(self):
        weights = np.zeros(len(WeightedMetricIndex))
        values = np.ones(len(WeightedMetricIndex))
        weights[WeightedMetricIndex.PROGRESS] = 5
        weights[WeightedMetricIndex.TTC] = 5
        weights[WeightedMetricIndex.LANE_KEEPING] = 2
        weights[WeightedMetricIndex.HISTORY_COMFORT] = 2
        weights[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = 2
        values[WeightedMetricIndex.TTC] = 0.4
        values[WeightedMetricIndex.HISTORY_COMFORT] = 0.5
        values[WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT] = np.nan
        result = pd.DataFrame([{
            "weighted_metrics": values,
            "weighted_metrics_array": weights,
            "multiplicative_metrics_prod": 0.8,
            "pdm_score": 0.0,
        }])

        score = score_without_extended_comfort(result)

        self.assertAlmostEqual(score, 0.8 * (5 + 5 * 0.4 + 2 + 2 * 0.5) / 14)

    def test_rejects_incomplete_and_nonfinite_evaluator_results(self):
        with self.assertRaises(ValueError):
            score_without_extended_comfort(pd.DataFrame())

        values = np.ones(len(WeightedMetricIndex))
        values[WeightedMetricIndex.PROGRESS] = np.nan
        result = pd.DataFrame([{
            "weighted_metrics": values,
            "weighted_metrics_array": np.ones(len(WeightedMetricIndex)),
            "multiplicative_metrics_prod": 1.0,
        }])
        with self.assertRaises(ValueError):
            score_without_extended_comfort(result)

    def test_fixed_history_score_matches_official_weighted_reduction(self):
        values = np.array([0.8, 0.5, 1.0, 0.75, np.nan])
        result = pd.DataFrame([{
            "weighted_metrics": values,
            "weighted_metrics_array": np.array([5.0, 5.0, 2.0, 2.0, 2.0]),
            "multiplicative_metrics_prod": 0.6,
            "pdm_score": -1.0,
        }])

        score = score_with_extended_comfort(result, 0.25)

        self.assertAlmostEqual(score, 0.6 * (5 * 0.8 + 5 * 0.5 + 2 + 2 * 0.75 + 2 * 0.25) / 16)
        self.assertTrue(np.isnan(values[-1]))
        with self.assertRaises(ValueError):
            score_with_extended_comfort(result, np.nan)

    def test_two_frame_comfort_matches_official_aggregator(self):
        from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import SceneAggregator

        sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        previous = np.zeros((41, 11))
        current = np.zeros((41, 11))
        previous[:, 0] = np.arange(41) * 0.1
        current[:, 0] = 0.5 + np.arange(41) * 0.1
        frame_scores = pd.DataFrame.from_dict({
            "now": {"ego_simulated_states": current, "start_time": 10.0},
            "previous": {"ego_simulated_states": previous, "start_time": 9.5},
        }, orient="index")
        official = SceneAggregator("now", "previous", frame_scores, sampling)._compute_two_frame_comfort(
            "now", "previous",
        )

        self.assertEqual(two_frame_extended_comfort(current, previous, 10.0, 9.5, sampling), official)
        with self.assertRaises(ValueError):
            two_frame_extended_comfort(current, previous, 9.5, 10.0, sampling)


class TestSceneEvaluation(TestCase):
    def test_scores_each_proposal_separately_without_mutating_cache(self):
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        evaluator_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        simulator = SimpleNamespace(proposal_sampling=evaluator_sampling)
        proposals = np.zeros((64, 2, 3))
        proposals[:, 0, 0] = np.arange(64) / 64
        metric_cache = SimpleNamespace(observation=SimpleNamespace(calls=0))
        seen = []

        def evaluate_candidate(**kwargs):
            candidate_cache = kwargs["metric_cache"]
            self.assertEqual(candidate_cache.observation.calls, 0)
            candidate_cache.observation.calls += 1
            self.assertEqual(kwargs["future_sampling"], evaluator_sampling)
            self.assertEqual(kwargs["model_trajectory"].trajectory_sampling, sampling)
            value = float(kwargs["model_trajectory"].poses[0, 0])
            seen.append(value)
            row = {column: 1.0 for column in METRIC_COLUMNS}
            row.update({
                "weighted_metrics": np.array([value, 1, 1, 1, np.nan]),
                "weighted_metrics_array": np.array([5, 5, 2, 2, 2]),
                "multiplicative_metrics_prod": 1.0,
                "pdm_score": -1.0,
            })
            return pd.DataFrame([row]), None

        with patch("navsim.evaluate.two_stage_oracle.pdm_score", side_effect=evaluate_candidate):
            scores, metrics = evaluate_scene_proposals(metric_cache, proposals, sampling, simulator, None, None)
            reversed_scores, _ = evaluate_scene_proposals(metric_cache, proposals[::-1], sampling, simulator, None, None)

        self.assertEqual(len(seen), 128)
        self.assertEqual(len(metrics), 64)
        self.assertEqual(metric_cache.observation.calls, 0)
        np.testing.assert_allclose(scores, reversed_scores[::-1])
        self.assertAlmostEqual(scores[0], 9 / 14)
        self.assertAlmostEqual(scores[63], (9 + 5 * 63 / 64) / 14)

    def test_rejects_missing_candidate_and_nonfinite_poses(self):
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        metric_cache = SimpleNamespace(observation=SimpleNamespace())
        with self.assertRaises(ValueError):
            evaluate_scene_proposals(metric_cache, np.zeros((63, 2, 3)), sampling, None, None, None)

        proposals = np.zeros((64, 2, 3))
        proposals[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            evaluate_scene_proposals(metric_cache, proposals, sampling, None, None, None)

    def test_fixed_history_produces_both_scores_in_one_evaluator_pass(self):
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        evaluator_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        simulator = SimpleNamespace(proposal_sampling=evaluator_sampling)
        cache = SimpleNamespace(observation=SimpleNamespace(calls=0), timepoint=SimpleNamespace(time_s=10.0))
        proposals = np.zeros((64, 2, 3))
        proposals[:, 0, 0] = np.arange(64)
        seen = []

        def evaluate_candidate(**kwargs):
            self.assertEqual(kwargs["metric_cache"].observation.calls, 0)
            kwargs["metric_cache"].observation.calls += 1
            value = int(kwargs["model_trajectory"].poses[0, 0])
            seen.append(value)
            row = {column: 1.0 for column in METRIC_COLUMNS}
            row.update({
                "weighted_metrics": np.array([1.0, 1.0, 1.0, 1.0, np.nan]),
                "weighted_metrics_array": np.array([5.0, 5.0, 2.0, 2.0, 2.0]),
                "multiplicative_metrics_prod": 1.0,
                "pdm_score": -1.0,
            })
            states = np.zeros((41, 11))
            states[0, 0] = value
            return pd.DataFrame([row]), states

        with patch("navsim.evaluate.two_stage_oracle.pdm_score", side_effect=evaluate_candidate), patch(
            "navsim.evaluate.two_stage_oracle.two_frame_extended_comfort",
            side_effect=lambda states, *args: float(states[0, 0] % 2),
        ):
            scores, metrics, fixed_scores, comfort = evaluate_scene_proposals_with_history(
                cache, proposals, np.zeros((41, 11)), 9.5, sampling, simulator, None, None,
            )

        self.assertEqual(seen, list(range(64)))
        self.assertEqual(cache.observation.calls, 0)
        self.assertEqual(len(metrics), 64)
        np.testing.assert_array_equal(scores, np.ones(64))
        np.testing.assert_array_equal(comfort, np.arange(64) % 2)
        np.testing.assert_array_equal(fixed_scores, (14 + 2 * comfort) / 16)

    def test_previous_top_one_is_simulated_without_scoring(self):
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
        evaluator_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        selected = Trajectory(np.zeros((2, 3)), sampling)
        cache = SimpleNamespace(ego_state=SimpleNamespace(time_point=object()))
        simulator = SimpleNamespace(proposal_sampling=evaluator_sampling)
        simulator.simulate_proposals = lambda states, initial: states
        reference_states = np.zeros((41, 11))

        with patch("navsim.evaluate.two_stage_oracle.transform_trajectory", return_value=object()) as transform, patch(
            "navsim.evaluate.two_stage_oracle.get_trajectory_as_array", return_value=reference_states,
        ) as get_states, patch("navsim.evaluate.two_stage_oracle.pdm_score") as score:
            simulated = simulate_fixed_history(cache, selected, simulator)

        transform.assert_called_once_with(selected, cache.ego_state)
        self.assertEqual(get_states.call_args.args[1], evaluator_sampling)
        score.assert_not_called()
        np.testing.assert_array_equal(simulated, reference_states)


class TestOracleReduction(TestCase):
    def test_stage_two_selects_evaluator_max_once_per_scene(self):
        scores = np.zeros((3, 64))
        scores[0, 0], scores[0, 10] = 0.95, 0.9
        scores[1, 8], scores[1, 9] = 0.6, 0.6
        scores[2, 63] = 0.3

        maxima, indices = stage_two_oracles(scores)

        np.testing.assert_array_equal(maxima, [0.95, 0.6, 0.3])
        np.testing.assert_array_equal(indices, [0, 8, 63])

    def test_short_and_long_selections_match_example(self):
        weights = np.array([[0.8, 0.15, 0.05], [0.1, 0.3, 0.6]])
        result = compare_oracles(np.array([0.8, 0.9]), np.array([0.95, 0.6, 0.3]), weights)

        np.testing.assert_allclose(result["downstream_values"], [0.865, 0.455])
        self.assertEqual((result["short_index"], result["long_index"]), (1, 0))
        self.assertAlmostEqual(result["short_score"], 0.4095)
        self.assertAlmostEqual(result["long_score"], 0.692)
        self.assertAlmostEqual(result["gain"], 0.2825)
        self.assertTrue(result["disagreement"])

    def test_gaussian_weights_match_kernel_and_uniform_fallback(self):
        weights = gaussian_weights(
            np.array([[0.0, 0.0], [10.0, 10.0]]),
            ["close", "far"],
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            TrajectorySampling(num_poses=2, interval_length=0.5),
        )

        np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0])
        self.assertAlmostEqual(weights[0, 0] / weights[0, 1], np.exp(1 / 0.2))
        np.testing.assert_array_equal(weights[1], [0.5, 0.5])

    def test_endpoints_use_global_rear_axle_reference(self):
        proposals = np.zeros((2, 2, 3))
        proposals[0, -1, :2] = [1, 0]
        proposals[1, -1, :2] = [0, 1]
        endpoints = proposal_endpoints(proposals, StateSE2(10, 20, np.pi / 2))
        np.testing.assert_allclose(endpoints, [[10, 21], [9, 20]], atol=1e-10)

    def test_zero_values_and_exact_ties_choose_first_proposal(self):
        result = compare_oracles(np.zeros(64), np.array([0.9]), np.ones((64, 1)))
        self.assertEqual((result["short_index"], result["long_index"]), (0, 0))
        self.assertEqual(result["gain"], 0)
        np.testing.assert_array_equal(result["downstream_values"], np.full(64, 0.9))

        with self.assertRaises(ValueError):
            stage_two_oracles(np.zeros((1, 63)))
        with self.assertRaises(ValueError):
            gaussian_weights(np.zeros((64, 2)), [], np.zeros((0, 2)),
                             TrajectorySampling(num_poses=2, interval_length=0.5))

    def test_selects_complete_now_mappings_before_limiting(self):
        mappings = [
            ["now-a", "prev-a", [["follow-a", "follow-prev-a"], ["follow-b", "follow-prev-b"]]],
            ["now-b", "prev-b", [["follow-c", "follow-prev-c"]]],
        ]
        self.assertEqual(select_now_mappings(mappings, 1), [("now-a", ["follow-a", "follow-b"])])
        self.assertEqual(len(select_now_mappings(mappings)), 2)
        with self.assertRaises(ValueError):
            select_now_mappings(mappings, -1)
        with self.assertRaises(ValueError):
            select_now_mappings([["now", "prev", []]])

    def test_fixed_history_pairs_preserve_original_and_followup_previous(self):
        mappings = [
            ["now-a", "prev-a", [["follow-a", "follow-prev-a"], ["follow-b", "follow-prev-b"]]],
            ["now-b", "prev-b", [["follow-c", "follow-prev-c"]]],
        ]
        self.assertEqual(select_history_pairs(mappings, 1), {
            "now-a": "prev-a", "follow-a": "follow-prev-a", "follow-b": "follow-prev-b",
        })
        with self.assertRaisesRegex(ValueError, "Conflicting fixed-history"):
            select_history_pairs([
                ["now-a", "prev-a", [["follow-a", "prev-a"]]],
                ["now-b", "prev-b", [["follow-a", "prev-b"]]],
            ])

    def test_scorer_progress_depends_on_proposal_group(self):
        scorer = PDMScorer(TrajectorySampling(num_poses=2, interval_length=0.5))

        def score_for_progress(progress):
            scorer._multi_metrics = np.ones((len(MultiMetricIndex), len(progress)))
            scorer._weighted_metrics = np.ones((len(WeightedMetricIndex), len(progress)))
            scorer._progress_raw = np.asarray(progress)
            return scorer._aggregate_pdm_scores()[1]

        self.assertNotAlmostEqual(score_for_progress([8, 4]), score_for_progress([8, 4, 20]))