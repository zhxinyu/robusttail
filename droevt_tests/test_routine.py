import unittest
from unittest.mock import patch

import numpy as np

from droevt.routine import optimization_with_ellipsodial_constraint


class RoutineTest(unittest.TestCase):
    def test_ellipsoidal_moments_include_observation_equal_to_threshold(self):
        data = np.arange(5.0)

        with patch(
            "droevt.routine._optimization_convex_chi_square",
            return_value=0.25,
        ) as optimize:
            result = optimization_with_ellipsodial_constraint(
                D=2,
                input_data=data,
                threshold_percentage=0.5,
                alpha=0.05,
                left_end_point_objective=3.0,
                right_end_point_objective=np.inf,
                g_ellipsoidal_dimension=2,
                bootstrapping_size=10,
                bootstrapping_seed=1,
            )

        self.assertEqual(result, 0.25)
        call = optimize.call_args.kwargs
        np.testing.assert_allclose(
            call["mu"],
            np.array([3.0 / 5.0, 9.0 / 5.0]),
        )
        expected_rows = np.vstack(
            [
                np.array([0.0, 0.0, 1.0, 1.0, 1.0]),
                np.array([0.0, 0.0, 2.0, 3.0, 4.0]),
            ]
        )
        np.testing.assert_allclose(call["Sigma"], np.cov(expected_rows))


if __name__ == "__main__":
    unittest.main()
