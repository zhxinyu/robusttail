import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

ENVIRONMENT_PREFIX = Path(sys.executable).resolve().parent.parent
os.environ.setdefault("R_HOME", str(ENVIRONMENT_PREFIX / "lib" / "R"))

from experiments.run_scripts.tail_probability import (
    benchmark_tail_probability_estimation as benchmark,
)


class _RResult:
    def tolist(self):
        return [0.01, 0.02]


class BenchmarkTailProbabilityTest(unittest.TestCase):
    def test_logit_delta_interval_is_bounded_and_matches_formula(self):
        common_utils = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "run_scripts"
            / "evtr"
            / "common_utils.R"
        )
        benchmark.ro.r(f"source('{common_utils.as_posix()}')")
        probability = 0.01
        standard_error = 0.02
        alpha = 0.05

        result = np.asarray(
            benchmark.ro.r(
                "logit_delta_interval("
                f"{probability}, {standard_error}, {alpha})"
            ),
            dtype=float,
        )
        critical_value = norm.ppf(1 - alpha / 2)
        transformed_standard_error = standard_error / (
            probability * (1 - probability)
        )
        expected = expit(
            logit(probability)
            + np.asarray([-critical_value, critical_value])
            * transformed_standard_error
        )

        self.assertLess(probability - critical_value * standard_error, 0)
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 1)

    def test_explicit_threshold_is_forwarded_to_retained_r_estimator(self):
        with patch.object(benchmark.ro, "r", return_value=_RResult()) as mocked_r:
            result = benchmark.benchmark_estimate_tail_probability(
                input_data=np.asarray([1.0, 2.0, 3.0]),
                left_end_point_objective=2.5,
                right_end_point_objective=np.inf,
                alpha=0.05,
                method="pot",
                threshold_percentage=0.70,
                random_state=123,
            )

        self.assertEqual(result, [0.01, 0.02])
        r_code = mocked_r.call_args.args[0]
        self.assertIn("u <- as.numeric(quantile(data, 0.7))", r_code)
        self.assertIn("gpdTIP(data, lhs, rhs, conf=conf, u=u)", r_code)
        self.assertIn("set.seed(123)", r_code)
        self.assertNotIn("grDevices::pdf(NULL)", r_code)

    def test_adaptive_threshold_behavior_remains_default(self):
        with patch.object(benchmark.ro, "r", return_value=_RResult()) as mocked_r:
            benchmark.benchmark_estimate_tail_probability(
                input_data=np.asarray([1.0, 2.0, 3.0]),
                left_end_point_objective=2.5,
                right_end_point_objective=np.inf,
                alpha=0.05,
                method="pot_bt",
            )

        r_code = mocked_r.call_args.args[0]
        self.assertNotIn("u <- as.numeric(quantile(data", r_code)
        self.assertNotIn("set.seed(", r_code)
        self.assertIn("gpdTIP(data, lhs, rhs, conf=conf)", r_code)
        self.assertNotIn("grDevices::pdf(NULL)", r_code)

    def test_adaptive_mle_v1_uses_headless_graphics_device(self):
        with patch.object(benchmark.ro, "r", return_value=_RResult()) as mocked_r:
            benchmark.benchmark_estimate_tail_probability(
                input_data=np.asarray([1.0, 2.0, 3.0]),
                left_end_point_objective=2.5,
                right_end_point_objective=np.inf,
                alpha=0.05,
                method="pot",
            )

        r_code = mocked_r.call_args.args[0]
        self.assertIn("grDevices::pdf(NULL)", r_code)
        self.assertIn("grDevices::dev.off()", r_code)


if __name__ == "__main__":
    unittest.main()
