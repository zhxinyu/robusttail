import unittest
import numpy as np
from scipy.stats import chi2, kstwobign
import logging

import droevt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestCalibration(unittest.TestCase):

    def setUp(self):
        # Simple synthetic data for testing
        # For standard normal distribution N(0,1):
        # PDF: f(x) = (1/sqrt(2π)) * exp(-x^2/2)
        # PDF derivative: f'(x) = -(x/sqrt(2π)) * exp(-x^2/2)
        
        # At x=1:
        # f(1) = (1/sqrt(2π)) * exp(-1/2)
        # f'(1) = -(1/sqrt(2π)) * exp(-1/2)
        
        self.density_at_1 = 1/np.sqrt(2*np.pi) * np.exp(-1/2)  # ≈ 0.2420
        self.density_derivative_at_1 = -1/np.sqrt(2*np.pi) * np.exp(-1/2)  # ≈ -0.2420
        self.threshold = 1
        self.alpha = 0.05
        self.bootstrapping_size = 1_000
        self.bootstrapping_seed = 42
        self.D_riser_number = 1
        self.num_multi_threshold = 1

        np.random.seed(self.bootstrapping_seed)  # Set fixed seed for reproducible tests
        self.data = np.random.normal(loc=0, scale=1, size=1_000_000)

    def test_eta_generation_no_bootstrap(self):
        result = droevt.calibration.eta_generation(
            data=self.data,
            point_estimate=self.threshold,
            bootstrapping_flag=False,
            bootstrapping_size=-1,
            bootstrapping_seed=-1
        )
        self.assertIsInstance(result, float)
        self.assertLess(abs(result - self.density_at_1), 1e-3)
        logger.info(f"eta_generation_no_bootstrap: {result:.3f}, density_at_1: {self.density_at_1:.3f}")

    def test_eta_generation_with_bootstrap(self):
        result = droevt.calibration.eta_generation(
            data=self.data,
            point_estimate=self.threshold,
            bootstrapping_flag=True,
            bootstrapping_size=self.bootstrapping_size,
            bootstrapping_seed=self.bootstrapping_seed
        )
        # Compute quantiles
        lower = np.quantile(result, self.alpha/2)
        upper = np.quantile(result, 1 - self.alpha/2)
        self.assertTrue(lower <= self.density_at_1 <= upper)
        self.assertLess((upper-lower)/self.density_at_1, 0.23)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), self.bootstrapping_size)
        self.assertTrue(all(isinstance(x, float) for x in result))
        logger.info(
            f"eta_generation_with_bootstrap: "
            f"lower: {lower:.3f}, "
            f"upper: {upper:.3f}, "
            f"density_at_1: {self.density_at_1:.3f}"
        )

    def test_eta_specification(self):
        result = droevt.calibration.eta_specification(
            data=self.data,
            threshold=self.threshold,
            alpha=self.alpha,
            bootstrapping_size=self.bootstrapping_size,
            bootstrapping_seed=self.bootstrapping_seed,
            D_riser_number=self.D_riser_number,
            num_multi_threshold=self.num_multi_threshold
        )
        self.assertIsInstance(result, float)
        upper = np.quantile(result, 1 - self.alpha/2)
        self.assertTrue(self.density_at_1 <= upper)
        logger.info(f"eta_specification: upper: {upper:.3f}, density_at_1: {self.density_at_1:.3f}")

    def test_negative_nu_generation_no_bootstrap(self):
        result = droevt.calibration.negative_nu_generation(
            data=self.data,
            point_estimate=self.threshold,
            bootstrapping=False,
            bootstrap_size=self.bootstrapping_size,
            bootstrap_seed=self.bootstrapping_seed
        )
        self.assertIsInstance(result, float)
        logger.info(
            f"negative_nu_generation_no_bootstrap: {result:.3f}, "
            f"density_derivative_at_1: {self.density_derivative_at_1:.3f}, "
            f"diff: {abs(result - self.density_derivative_at_1):.3f}"
        )
        self.assertLess(abs(result - self.density_derivative_at_1), 7e-3)

    def test_negative_nu_generation_with_bootstrap(self):
        result = droevt.calibration.negative_nu_generation(
            data=self.data,
            point_estimate=self.threshold,
            bootstrapping=True,
            bootstrap_size=self.bootstrapping_size,
            bootstrap_seed=self.bootstrapping_seed
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), self.bootstrapping_size)
        self.assertTrue(all(isinstance(x, float) for x in result))

        lower = np.quantile(result, self.alpha/2)
        self.assertTrue(lower <= self.density_derivative_at_1)
        logger.info(
            f"negative_nu_generation_with_bootstrap: "
            f"lower: {lower:.3f}, "
            f"density_derivative_at_1: {self.density_derivative_at_1:.3f}, "
            f"diff: {abs(self.density_derivative_at_1 - lower):.3f}"
        )

    def test_nu_specification(self):
        result = droevt.calibration.nu_specification(
            data=self.data,
            threshold=self.threshold,
            alpha=self.alpha,
            bootstrapping_size=self.bootstrapping_size,
            bootstrapping_seed=self.bootstrapping_seed,
            num_multi_threshold=self.num_multi_threshold
        )
        self.assertIsInstance(result, float)
        self.assertTrue(self.density_derivative_at_1 >= -result)
        logger.info(
            f"nu_specification: "
            f"result: {result:.3f}, "
            f"density_derivative_at_1: {self.density_derivative_at_1:.3f}, "
            f"diff: {abs(self.density_derivative_at_1 - -result):.3f}"
        )

    def test_z_of_chi_square(self):
        z = droevt.calibration.z_of_chi_square(
            alpha=self.alpha,
            D_riser_number=1,
            g_dimension=2,
            num_multi_threshold=self.num_multi_threshold
        )
        self.assertIsInstance(z, float)
        self.assertGreater(z, 0)
        self.assertEqual(z, chi2.ppf(q=1-self.alpha/(self.num_multi_threshold+1), df=2))
        logger.info(f"z_of_chi_square: {z:.3f}")

    def test_z_of_kolmogorov(self):
        z = droevt.calibration.z_of_kolmogorov(
            alpha=self.alpha,
            D_riser_number=1,
            num_multi_threshold=self.num_multi_threshold
        )
        self.assertIsInstance(z, float)
        self.assertGreater(z, 0)
        self.assertEqual(z, kstwobign.ppf(q=1-self.alpha/(self.num_multi_threshold+1)))
        logger.info(f"z_of_kolmogorov: {z:.3f}")

if __name__ == '__main__':
    unittest.main()
