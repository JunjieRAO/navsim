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
    evaluate_token_group,
    load_token_artifact,
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