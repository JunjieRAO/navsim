from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.evaluate.two_stage_oracle import (
    METRIC_COLUMNS,
    compare_oracles,
    evaluate_scene_proposals,
    gaussian_weights,
    proposal_endpoints,
    score_without_extended_comfort,
    select_now_mappings,
    stage_two_oracles,
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

    def test_scorer_progress_depends_on_proposal_group(self):
        scorer = PDMScorer(TrajectorySampling(num_poses=2, interval_length=0.5))

        def score_for_progress(progress):
            scorer._multi_metrics = np.ones((len(MultiMetricIndex), len(progress)))
            scorer._weighted_metrics = np.ones((len(WeightedMetricIndex), len(progress)))
            scorer._progress_raw = np.asarray(progress)
            return scorer._aggregate_pdm_scores()[1]

        self.assertNotAlmostEqual(score_for_progress([8, 4]), score_for_progress([8, 4, 20]))