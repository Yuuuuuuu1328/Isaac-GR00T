import unittest

import numpy as np

from deployment_scripts.ant.quality_metrics import (
    compute_action_error_metrics,
    compute_action_smoothness,
    compute_proxy_success,
)


class QualityMetricsTest(unittest.TestCase):
    def test_compute_action_error_metrics_reports_rmse_and_first_step(self):
        predicted = np.array([[[0.0], [1.0], [2.0]]], dtype=np.float32)
        ground_truth = np.array([[[0.0], [2.0], [1.0]]], dtype=np.float32)

        metrics = compute_action_error_metrics(predicted, ground_truth)

        self.assertAlmostEqual(metrics["open_loop_mse"], 0.6667, places=4)
        self.assertAlmostEqual(metrics["open_loop_rmse"], 0.8165, places=4)
        self.assertAlmostEqual(metrics["open_loop_mae"], 0.6667, places=4)
        self.assertAlmostEqual(metrics["open_loop_first_step_rmse"], 0.0, places=4)
        self.assertAlmostEqual(metrics["open_loop_horizon_pos_1_rmse"], 1.0, places=4)

    def test_compute_action_smoothness_reports_pullback_ratio(self):
        predicted = np.array([[[0.0], [1.0], [0.2]]], dtype=np.float32)

        metrics = compute_action_smoothness(predicted)

        self.assertGreater(metrics["smoothness_pullback_ratio"], 0.0)
        self.assertGreater(metrics["smoothness_second_diff_norm_mean"], 0.0)

    def test_compute_proxy_success_requires_both_error_thresholds(self):
        metrics = compute_proxy_success(
            {
                "open_loop_rmse": 0.04,
                "open_loop_first_step_rmse": 0.03,
                "smoothness_sign_flip_ratio": 0.05,
            },
            rmse_threshold=0.05,
            first_step_threshold=0.05,
            sign_flip_threshold=0.10,
        )
        self.assertEqual(metrics["proxy_success_rate"], 1.0)

        failed_metrics = compute_proxy_success(
            {
                "open_loop_rmse": 0.08,
                "open_loop_first_step_rmse": 0.03,
                "smoothness_sign_flip_ratio": 0.05,
            },
            rmse_threshold=0.05,
            first_step_threshold=0.05,
            sign_flip_threshold=0.10,
        )
        self.assertEqual(failed_metrics["proxy_success_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
