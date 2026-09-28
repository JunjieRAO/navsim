from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.evaluate.two_stage_oracle import METRIC_COLUMNS, stage_two_oracles
from navsim.planning.script.run_two_stage_oracle import (
    aggregate_mapping_results,
    evaluate_history_group,
    evaluate_token_group,
    load_history_artifact,
    load_token_artifact,
    save_history_artifact,
    save_token_artifact,
    summarize_results,
    token_artifact_path,
    verify_manifest,
    write_report,
)


class TestOracleArtifacts(TestCase):
    def test_resume_requires_complete_artifact_matching_fingerprint_and_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "metric_cache.pkl"
            cache_path.write_bytes(b"original cache")
            sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
            result = {
                "proposals": np.ones((64, 2, 3)),
                "scores": np.full(64, 0.5),
                "metrics": np.ones((64, len(METRIC_COLUMNS))),
                "endpoints": np.zeros((64, 2)),
                "start": np.zeros(2),
            }

            save_token_artifact(root, "scene-1", result, "fingerprint-a", cache_path)

            restored = load_token_artifact(root, "scene-1", "fingerprint-a", cache_path, sampling)
            np.testing.assert_array_equal(restored["scores"], result["scores"])
            self.assertIsNone(load_token_artifact(root, "scene-1", "fingerprint-b", cache_path, sampling))
            cache_path.write_bytes(b"changed metric cache")
            self.assertIsNone(load_token_artifact(root, "scene-1", "fingerprint-a", cache_path, sampling))

            with self.assertRaises(ValueError):
                token_artifact_path(root, "../escape")

    def test_manifest_rejects_mixed_experiments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            verify_manifest(root, "fingerprint-a")
            verify_manifest(root, "fingerprint-a")
            with self.assertRaisesRegex(ValueError, "different configuration"):
                verify_manifest(root, "fingerprint-b")

    def test_fixed_history_cache_requires_matching_previous_and_complete_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            now_cache = root / "now.pkl"
            prev_cache = root / "prev.pkl"
            now_cache.write_bytes(b"now")
            prev_cache.write_bytes(b"previous")
            model_sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
            evaluator_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
            history = {
                "selected_trajectory": np.zeros((2, 3)),
                "simulated_states": np.zeros((41, 11)),
                "start_time": np.array(9.5),
            }
            result = {
                "proposals": np.zeros((64, 2, 3)),
                "scores": np.ones(64),
                "metrics": np.ones((64, len(METRIC_COLUMNS))),
                "endpoints": np.zeros((64, 2)),
                "start": np.zeros(2),
                "fixed_ec_scores": np.full(64, 0.8),
                "extended_comfort": np.ones(64),
            }
            save_history_artifact(root, "prev", history, "fixed", prev_cache)
            save_token_artifact(root, "now", result, "fixed", now_cache, "prev", prev_cache)

            self.assertIsNotNone(load_history_artifact(
                root, "prev", "fixed", prev_cache, model_sampling, evaluator_sampling,
            ))
            self.assertEqual(load_token_artifact(
                root, "now", "fixed", now_cache, model_sampling, "prev", prev_cache,
            )["fixed_ec_scores"].shape, (64,))
            self.assertIsNone(load_token_artifact(
                root, "now", "fixed", now_cache, model_sampling, "other", prev_cache,
            ))
            prev_cache.write_bytes(b"previous changed")
            self.assertIsNone(load_history_artifact(
                root, "prev", "fixed", prev_cache, model_sampling, evaluator_sampling,
            ))
            self.assertIsNone(load_token_artifact(
                root, "now", "fixed", now_cache, model_sampling, "prev", prev_cache,
            ))

    def test_cached_fixed_history_skips_top_one_inference_and_simulation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "previous.pkl"
            cache_path.write_bytes(b"previous")
            model_sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
            evaluator_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
            history = {
                "selected_trajectory": np.zeros((2, 3)),
                "simulated_states": np.zeros((41, 11)),
                "start_time": np.array(9.5),
            }
            save_history_artifact(root, "previous", history, "fixed", cache_path)
            history_results = {}

            with patch("navsim.planning.script.run_two_stage_oracle.infer_previous_trajectories") as inference, patch(
                "navsim.planning.script.run_two_stage_oracle.score_history_job"
            ) as simulation:
                evaluate_history_group(
                    ["previous"], SimpleNamespace(), SimpleNamespace(tokens=["previous"]),
                    {"previous": cache_path}, root, "fixed", model_sampling, evaluator_sampling,
                    history_results, {},
                )

            inference.assert_not_called()
            simulation.assert_not_called()
            np.testing.assert_array_equal(history_results["previous"]["simulated_states"], history["simulated_states"])

    def test_fixed_history_token_requires_valid_previous_and_uses_both_cached_scores(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            now_cache = root / "now.pkl"
            prev_cache = root / "previous.pkl"
            now_cache.write_bytes(b"now")
            prev_cache.write_bytes(b"previous")
            sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
            result = {
                "proposals": np.zeros((64, 2, 3)),
                "scores": np.ones(64),
                "fixed_ec_scores": np.full(64, 0.8),
                "extended_comfort": np.ones(64),
                "metrics": np.ones((64, len(METRIC_COLUMNS))),
                "endpoints": np.zeros((64, 2)),
                "start": np.zeros(2),
            }
            save_token_artifact(root, "now", result, "fixed", now_cache, "previous", prev_cache)
            current = {}
            errors = {}
            with patch("navsim.planning.script.run_two_stage_oracle.infer_proposals") as inference:
                evaluate_token_group(
                    ["now"], SimpleNamespace(), SimpleNamespace(tokens=["now"]),
                    {"now": now_cache, "previous": prev_cache}, root, "fixed", sampling,
                    current, errors, {"now": "previous"}, {},
                )
                self.assertIn("previous", errors["now"])
                evaluate_token_group(
                    ["now"], SimpleNamespace(), SimpleNamespace(tokens=["now"]),
                    {"now": now_cache, "previous": prev_cache}, root, "fixed", sampling,
                    current, errors, {"now": "previous"}, {"previous": {}},
                )
                inference.assert_not_called()

            self.assertEqual(current["now"]["fixed_ec_scores"].shape, (64,))

    def test_shared_stage_two_scene_has_one_oracle_and_missing_scene_invalidates_mapping(self):
        starts = {"follow-good": [0.0, 0.0], "follow-poor": [1.0, 0.0]}
        stage_one_scores = np.zeros(64)
        stage_one_scores[:2] = [0.8, 0.9]
        endpoints = np.zeros((64, 2))
        endpoints[1, 0] = 1
        token_results = {
            "now-a": {"scores": stage_one_scores, "endpoints": endpoints},
            "now-b": {"scores": stage_one_scores, "endpoints": endpoints},
        }
        for token, score in (("follow-good", 0.95), ("follow-poor", 0.3)):
            scores = np.zeros(64)
            scores[4] = score
            token_results[token] = {"scores": scores, "start": np.array(starts[token])}
        mappings = [
            ("now-a", ["follow-good", "follow-poor"]),
            ("now-b", ["follow-good"]),
            ("now-c", ["follow-poor"]),
        ]
        sampling = TrajectorySampling(num_poses=2, interval_length=0.5)

        with patch("navsim.planning.script.run_two_stage_oracle.stage_two_oracles", wraps=stage_two_oracles) as stage_two_max:
            rows, invalid, details = aggregate_mapping_results(mappings, token_results, {}, sampling, 0.1)

        self.assertEqual(stage_two_max.call_count, 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0]["missing_tokens"], ["now-c"])
        self.assertEqual((rows[0]["short_index"], rows[0]["long_index"]), (1, 0))
        self.assertGreater(rows[0]["gain"], 0)
        np.testing.assert_allclose(details["now-a"]["weights"].sum(axis=1), np.ones(64))

    def test_dual_oracle_ranks_each_mode_separately_and_reuses_weights(self):
        original_scores = np.zeros(64)
        original_scores[:2] = [1.0, 0.8]
        original_fixed = np.zeros(64)
        original_fixed[:2] = [0.7, 0.9]
        followup_scores = np.zeros(64)
        followup_scores[:2] = [0.95, 0.3]
        followup_fixed = np.zeros(64)
        followup_fixed[:2] = [0.5, 0.85]
        token_results = {
            "now": {
                "scores": original_scores, "fixed_ec_scores": original_fixed,
                "endpoints": np.zeros((64, 2)),
            },
            "follow": {
                "scores": followup_scores, "fixed_ec_scores": followup_fixed,
                "start": np.zeros(2),
            },
        }
        mappings = [("now", ["follow"])]
        sampling = TrajectorySampling(num_poses=40, interval_length=0.1)

        with patch("navsim.planning.script.run_two_stage_oracle.gaussian_weights", wraps=__import__(
            "navsim.evaluate.two_stage_oracle", fromlist=["gaussian_weights"]
        ).gaussian_weights) as proximity:
            rows, _, details = aggregate_mapping_results(mappings, token_results, {}, sampling, 0.1)
            fixed_rows, _, fixed_details = aggregate_mapping_results(
                mappings, token_results, {}, sampling, 0.1, "fixed_ec_scores",
                {token: result["weights"] for token, result in details.items()},
            )

        self.assertEqual(proximity.call_count, 1)
        self.assertEqual(details["now"]["stage_two_oracle_indices"].tolist(), [0])
        self.assertEqual(fixed_details["now"]["stage_two_oracle_indices"].tolist(), [1])
        self.assertEqual(rows[0]["short_index"], 0)
        self.assertEqual(fixed_rows[0]["short_index"], 1)
        np.testing.assert_array_equal(details["now"]["weights"], fixed_details["now"]["weights"])

    def test_complete_cached_token_skips_inference_and_scoring(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cache_path = root / "metric_cache.pkl"
            cache_path.write_bytes(b"cache")
            sampling = TrajectorySampling(num_poses=2, interval_length=0.5)
            result = {
                "proposals": np.zeros((64, 2, 3)),
                "scores": np.zeros(64),
                "metrics": np.ones((64, len(METRIC_COLUMNS))),
                "endpoints": np.zeros((64, 2)),
                "start": np.zeros(2),
            }
            save_token_artifact(root, "scene", result, "fingerprint", cache_path)
            token_results = {}
            errors = {}

            with patch("navsim.planning.script.run_two_stage_oracle.infer_proposals") as inference, patch(
                "navsim.planning.script.run_two_stage_oracle.score_scene_job"
            ) as scoring:
                evaluate_token_group(
                    ["scene"], SimpleNamespace(), SimpleNamespace(tokens=["scene"]),
                    {"scene": cache_path}, root, "fingerprint", sampling, token_results, errors,
                )

            inference.assert_not_called()
            scoring.assert_not_called()
            self.assertEqual(errors, {})
            self.assertEqual(token_results["scene"]["scores"].shape, (64,))

    def test_report_records_paired_gain_and_failure_coverage(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{
                "original_token": "now-a", "gain": 0.25, "disagreement": True,
                "T_short": 0.4, "T_long": 0.65, "E1_top_gap": 0.01,
                "uniform_fallback_rows": 0, "mean_nearest_followup_distance_m": 1.0,
            }]
            invalid = [{"original_token": "now-b", "missing_tokens": ["follow-b"]}]
            details = {"now-a": {"weights": np.ones((64, 1))}}
            summary = summarize_results(rows, invalid, ["now-a", "follow-a", "now-b", "follow-b"],
                                        {"now-a": {}, "follow-a": {}, "now-b": {}}, {"follow-b": "missing"})

            write_report(root, rows, invalid, details, summary, {"follow-b": "missing"})

            self.assertEqual(summary["disagreement_rate"], 1.0)
            self.assertEqual(summary["mean_gain"], 0.25)
            self.assertEqual(summary["failed_tokens"], 1)
            self.assertTrue((root / "mappings.csv").exists())
            self.assertTrue((root / "mappings" / "now-a.npz").exists())
            self.assertTrue((root / "failures.json").exists())

    def test_summary_counts_fixed_histories_without_counting_them_as_proposals(self):
        summary = summarize_results(
            [], [{"original_token": "now", "missing_tokens": ["now"]}],
            ["now", "previous"], {}, {"now": "Missing previous"},
            {"previous": {"simulated_states": np.zeros((41, 11))}},
        )
        self.assertEqual(summary["complete_tokens"], 1)
        self.assertEqual(summary["failed_tokens"], 1)
        self.assertEqual(summary["complete_fixed_histories"], 1)
        self.assertEqual(summary["complete_proposal_scores"], 0)

    def test_fixed_history_report_writes_both_oracles_and_comparison(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            no_ec = {"original_token": "now", "short_index": 0, "long_index": 1, "gain": 0.2}
            fixed = {"original_token": "now", "short_index": 2, "long_index": 3, "gain": 0.3}
            details = {"now": {"weights": np.ones((64, 1)), "stage_two_oracle_indices": np.array([4])}}
            fixed_details = {"now": {
                "stage_two_oracle_indices": np.array([6]),
                "stage_two_oracle_scores": np.array([0.9]),
                "stage_one_scores": np.ones(64),
                "downstream_values": np.ones(64),
                "combined_scores": np.ones(64),
            }}

            write_report(root, [no_ec], [], details, {"metric": "EPDMS-no-EC"}, {},
                         [fixed], fixed_details, {"metric": "EPDMS-fixed-history-EC"})

            self.assertTrue((root / "mappings_fixed_history.csv").exists())
            self.assertTrue((root / "summary_fixed_history.json").exists())
            self.assertTrue((root / "comparisons.csv").exists())
            with np.load(root / "mappings" / "now.npz", allow_pickle=False) as artifact:
                np.testing.assert_array_equal(artifact["fixed_stage_two_oracle_indices"], [6])
                np.testing.assert_array_equal(artifact["weights"], details["now"]["weights"])