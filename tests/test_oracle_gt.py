from unittest import TestCase

import numpy as np

from navsim.planning.script.oracle_gt import calibrate_lambda, proposal_distance, select_proposal, switching_threshold


class TestOracleGT(TestCase):
    def test_ade_uses_xy_and_all_future_points(self):
        proposals = np.array([[[3, 4, 8], [0, 0, 9]], [[0, 0, 10], [0, 0, 11]]])
        np.testing.assert_allclose(proposal_distance(proposals, np.zeros((2, 3)), "ade"), [2.5, 0])
        with self.assertRaises(ValueError):
            proposal_distance(proposals, np.zeros((3, 3)), "ade")

    def test_fde_uses_xy_of_final_point_only(self):
        proposals = np.array([[[3, 4, 8], [0, 0, 9]], [[0, 0, 10], [6, 8, 11]]])
        np.testing.assert_allclose(proposal_distance(proposals, np.zeros((2, 3)), "fde"), [0, 10])
        with self.assertRaises(ValueError):
            proposal_distance(proposals, np.zeros((2, 3)), "lateral")

    def test_baseline_switch_and_monotonic_ade(self):
        scores = np.array([3.0, 2.0, -4.0])
        ade = np.array([5.0, 3.0, 0.0])
        self.assertEqual(select_proposal(scores, ade, 0), np.argmax(scores))
        self.assertEqual(switching_threshold(scores, ade), 0.5)
        self.assertEqual(select_proposal(scores, ade, 0.49), 0)
        self.assertEqual(select_proposal(scores, ade, 0.51), 1)
        selected_ade = [ade[select_proposal(scores, ade, strength)] for strength in [0, 0.1, 1, 3, 10]]
        self.assertTrue(np.all(np.diff(selected_ade) <= 0))
        self.assertEqual(select_proposal(scores, ade, min_distance=True), 2)

    def test_calibration_handles_ties_and_already_closest(self):
        reference, thresholds = calibrate_lambda([[3, 2], [1, 1], [2, 1]], [[5, 3], [2, 1], [0, 1]])
        self.assertEqual(reference, 0.5)
        np.testing.assert_equal(thresholds, [0.5, 0, np.inf])
        reference, _ = calibrate_lambda([[1, 1]], [[2, 1]])
        self.assertIsNone(reference)

    def test_rejects_invalid_values(self):
        for scores, ade, strength in [([np.nan], [0], 1), ([0], [-1], 1), ([0], [1], -1)]:
            with self.assertRaises(ValueError):
                select_proposal(scores, ade, strength)