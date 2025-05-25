import unittest
import numpy as np
from droevt.engine import optimization, PolynomialFunction
import logging
import inspect
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import droevt

class TestPolynomialFunctionIntegrationRiser(unittest.TestCase):

    def test_indicator_functions(self):
        # \mathbb{I}(x\geq a)
        poly1 = droevt.engine.PolynomialFunction([1, np.inf], [[1]])
        self.assertEqual(poly1.integration_riser(), droevt.engine.PolynomialFunction([1, np.inf], [[-1, 1]]))
        self.assertEqual(poly1.integration_riser(1), droevt.engine.PolynomialFunction([1, np.inf], [[-1, 1]]))
        self.assertEqual(poly1.integration_riser(2), droevt.engine.PolynomialFunction([1, np.inf], [[1/2, -1, 1/2]]))
        self.assertEqual(poly1.integration_riser(3), droevt.engine.PolynomialFunction([1, np.inf], [[-1/6, 1/2, -1/2, 1/6]]))
        self.assertEqual(poly1.integration_riser(4), droevt.engine.PolynomialFunction([1, np.inf], [[1/24, -1/6, 1/4, -1/6, 1/24]]))
        for _ in range(10):
            a = np.random.rand(1)[0]
            polya = droevt.engine.PolynomialFunction([a, np.inf], [[1]])
            self.assertEqual(polya.integration_riser(), droevt.engine.PolynomialFunction([a, np.inf], [[-a, 1]]))
            self.assertEqual(polya.integration_riser(1), droevt.engine.PolynomialFunction([a, np.inf], [[-a, 1]]))
            self.assertEqual(polya.integration_riser(2), droevt.engine.PolynomialFunction([a, np.inf], [[a**2/2, -a, 1/2]]))
            self.assertEqual(polya.integration_riser(3), droevt.engine.PolynomialFunction([a, np.inf], [[-a**3/6, a**2/2, -a/2, 1/6]]))
            self.assertEqual(polya.integration_riser(4), droevt.engine.PolynomialFunction([a, np.inf], [[a**4/24, -a**3/6, a**2/4, -a/6, 1/24]]))

    def test_interval_indicator_functions(self):
        # \mathbb{I}(b\leq x\leq c)
        poly2 = droevt.engine.PolynomialFunction([1, 2, np.inf], [[1], [0]])
        self.assertEqual(poly2.integration_riser(1), droevt.engine.PolynomialFunction([1, 2, np.inf], [[-1, 1], [1, 0]]))
        self.assertEqual(poly2.integration_riser(2), droevt.engine.PolynomialFunction([1, 2, np.inf], [[1/2, -1, 1/2], [-3/2, 1, 0]]))
        for _ in range(10):
            LR = np.random.rand(2)
            L = np.sort(LR)[0]
            R = np.sort(LR)[1]
            polyLR = droevt.engine.PolynomialFunction([L, R, np.inf], [[1], [0]])
            self.assertEqual(polyLR.integration_riser(1), droevt.engine.PolynomialFunction([L, R, np.inf], [[-L, 1], [R-L, 0]]))
            self.assertEqual(polyLR.integration_riser(2), droevt.engine.PolynomialFunction([L, R, np.inf], [[L**2/2, -L, 1/2], [-(R-L)*(R+L)/2, R-L, 0]]))

    def test_power_functions(self):
        # x^i\mathbb{I}(x\geq a),i>=0
        for i in range(0, 10):
            a = np.random.rand(1)[0]
            polyi = droevt.engine.PolynomialFunction([a, np.inf], [[0]*i+[1]])
            self.assertEqual(polyi.integration_riser(1), droevt.engine.PolynomialFunction([a, np.inf], [[-a**(i+1)/(i+1)]+[0]*i+[1/(i+1)]]))
            self.assertEqual(polyi.integration_riser(2), droevt.engine.PolynomialFunction([a, np.inf], [[a**(i+2)/(i+2)]+[-a**(i+1)/(i+1)]+[0]*(i)+[1/(i+1)/(i+2)]]))

class TestOptimizationEngine(unittest.TestCase):

    def test_0_ks_scenario_1(self):
        """
        Optimization problem:

        Maximize: h(x) = I(1 <= x <= 2)
        Subject to: g(x) = I(x >= threshold_level)
                    0.6 <= E[g(X)] <= 0.8

        Where:
        - I() is the indicator function
        - threshold_level is a random value between 0 and 1
        - X is a random variable
        """
        function_name = inspect.currentframe().f_code.co_name
        mu_lb_value = np.array([0.6])
        mu_ub_value = np.array([0.8])
        D_riser_number = 0
        L, R = 1, 2
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = np.random.rand()
        g_Rs = [PolynomialFunction([threshold_level, np.inf], [[1]])]
        calculated_value = optimization(D_riser_number=D_riser_number,
                                        threshold_level=threshold_level,
                                        h=h,
                                        g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value)
        expected_value = 0.8
        assert np.abs(expected_value-calculated_value) < 1e-6
        logger.info(f"{function_name} optimization optimal value: {calculated_value:.2E}")


    def test_0_ks_scenario_2(self):
        """
        Test the optimization engine with multiple moment constraints.

        This test case demonstrates the behavior of the optimization function
        as we progressively add more moment constraints. The objective is to
        maximize h(x) = I(2 <= x <= 3) subject to various moment constraints.

        The test performs the following steps:
        1. Start with one moment constraint.
        2. Progressively add three more moment constraints.
        3. For each step, calculate the optimal value using the optimization function.
        4. Verify that the optimal values decrease as more constraints are added.
        5. Check that the first optimal value is close to the upper bound of the first constraint.

        The moment constraints are of the form E[x^i | x >= 1] for i = 0, 1, 2, 3.
        """
        function_name = inspect.currentframe().f_code.co_name
        mu_lb_value = np.array([0.14])
        mu_ub_value = np.array([0.17])
        D_riser_number = 0
        L = 2
        R = 3
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = 1
        g_Rs = [PolynomialFunction([threshold_level, np.inf], [[0]*0+[1]])]
        calculated_values = []
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                              h=h,
                                              g_Rs=g_Rs,
                                              mu_lb_value=mu_lb_value,
                                              mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.23)
        mu_ub_value = np.append(mu_ub_value, 0.26)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*1+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs,
                                            mu_lb_value=mu_lb_value,
                                            mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.38)
        mu_ub_value = np.append(mu_ub_value, 0.41)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*2+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                              h=h,
                                              g_Rs=g_Rs,
                                              mu_lb_value=mu_lb_value,
                                              mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.72)
        mu_ub_value = np.append(mu_ub_value, 0.75)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*3+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                              h=h,
                                              g_Rs=g_Rs,
                                              mu_lb_value=mu_lb_value,
                                              mu_ub_value=mu_ub_value))
        self.assertLess(np.abs(0.17-np.array(calculated_values)[0]), 1e-6)
        self.assertTrue(np.all(np.diff(np.array(calculated_values)) < 0))
        logger.info("{:} optimization optimal value: {:}".format(function_name,
            '/'.join([f"{calculated_value:.2E}" for calculated_value in calculated_values])))


    def test_1_ks_scenario_1(self):
        function_name = inspect.currentframe().f_code.co_name
        mu_lb_value = np.array([0.14])
        mu_ub_value = np.array([0.17])
        D_riser_number = 1
        eta = 0.25
        L = 2
        R = 3
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = 1
        g_Rs = [PolynomialFunction([threshold_level, np.inf], [[0]*0+[1]])]
        calculated_values = []
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.23)
        mu_ub_value = np.append(mu_ub_value, 0.26)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*1+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.38)
        mu_ub_value = np.append(mu_ub_value, 0.41)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*2+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.72)
        mu_ub_value = np.append(mu_ub_value, 0.75)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*3+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))

        self.assertLess(np.abs(8.50E-02-calculated_values[0]), 1e-12)
        self.assertTrue(np.all(np.diff(np.array(calculated_values)) < 0))
        logger.info("{:} optimization optimal value: {:}".format(function_name,
            '/'.join([f"{calculated_value:.2E}" for calculated_value in calculated_values])))


    def test_2_ks_scenario_1(self):
        function_name = inspect.currentframe().f_code.co_name
        mu_lb_value = np.array([0.14])
        mu_ub_value = np.array([0.17])
        D_riser_number = 2
        eta_lb = 0.23
        eta_ub = 0.27
        nu = 0.24
        L = 2
        R = 3
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = 1
        g_Rs = [PolynomialFunction([threshold_level, np.inf], [[0]*0+[1]])]
        calculated_values = []
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.23)
        mu_ub_value = np.append(mu_ub_value, 0.26)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*1+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.38)
        mu_ub_value = np.append(mu_ub_value, 0.41)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*2+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))
        ### add another moment function
        mu_lb_value = np.append(mu_lb_value, 0.72)
        mu_ub_value = np.append(mu_ub_value, 0.75)
        g_Rs += [PolynomialFunction([threshold_level, np.inf], [[0]*3+[1]])]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value))

        self.assertLess(np.abs(4.18E-02-np.array(calculated_values)[0]), 51e-5)
        self.assertTrue(np.all(np.diff(np.array(calculated_values)) < 0))
        logger.info("{:} optimization optimal value: {:}".format(function_name,
            '/'.join([f"{calculated_value:.2E}" for calculated_value in calculated_values])))

    def test_0_chi2_scenario_1(self):
        function_name = inspect.currentframe().f_code.co_name
        L = 2
        R = 3
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = 1
        radius = 10
        d_E = 4
        D_riser_number = 0
        calculated_values = []
        mu_value = np.array([0.1593, 0.2435, 0.4044, 0.7362])
        Sigma = np.array([[0.1339369, 0.20471848, 0.34000757, 0.61899594],
                        [0.20471848, 0.34514292, 0.63781269, 1.29513989],
                        [0.34000757, 0.63781269, 1.31086431, 2.94225469],
                        [0.61899594, 1.29513989, 2.94225469, 7.21695721]])/10000
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0]*i+[1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 3
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0]*i+[1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 2
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0]*i+[1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 1
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0]*i+[1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, threshold_level=threshold_level,
                                              h=h,
                                              g_Es=g_Es,
                                              mu_value=mu_value, Sigma=Sigma, radius=radius))
        calculated_values = calculated_values[::-1]
        self.assertTrue(np.all(np.diff(np.array(calculated_values)) < 1e-9))
        logger.info("{:} optimization optimal value: {:}".format(function_name,
            '/'.join([f"{calculated_value:.2E}" for calculated_value in calculated_values])))

    def test_1_chi2_scenario_1(self):
        function_name = inspect.currentframe().f_code.co_name
        L = 2
        R = 3
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = 1
        radius = 10
        d_E = 4
        D_riser_number = 1
        eta = 0.24
        calculated_values = []
        mu_value = np.array([0.1593, 0.2435, 0.4044, 0.7362])
        Sigma = np.array([[0.1339369, 0.20471848, 0.34000757, 0.61899594],
                        [0.20471848, 0.34514292, 0.63781269, 1.29513989],
                        [0.34000757, 0.63781269, 1.31086431, 2.94225469],
                        [0.61899594, 1.29513989, 2.94225469, 7.21695721]]) / 10000
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 3
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 2
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 1
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta=eta,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        calculated_values = calculated_values[::-1]
        self.assertTrue(np.all(np.diff(np.array(calculated_values)) < 1e-9))
        logger.info("{:} optimization optimal value: {:}".format(function_name,
            '/'.join([f"{calculated_value:.2E}" for calculated_value in calculated_values])))


    def test_2_chi2_scenario_1(self):
        function_name = inspect.currentframe().f_code.co_name
        L = 2
        R = 3
        h = PolynomialFunction([L, R, np.inf], [[1], [0]])
        threshold_level = 1
        radius = 10
        d_E = 4
        D_riser_number = 2
        eta_lb = 0.23
        eta_ub = 0.27
        nu = 0.24
        calculated_values = []
        mu_value = np.array([0.1593, 0.2435, 0.4044, 0.7362])
        Sigma = np.array([[0.1339369, 0.20471848, 0.34000757, 0.61899594],
                        [0.20471848, 0.34514292, 0.63781269, 1.29513989],
                        [0.34000757, 0.63781269, 1.31086431, 2.94225469],
                        [0.61899594, 1.29513989, 2.94225469, 7.21695721]]) / 10000
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 3
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 2
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        d_E = 1
        mu_value = mu_value[:d_E]
        Sigma = Sigma[:d_E, :d_E]
        g_Es = [PolynomialFunction([threshold_level, np.inf], [
                                [0] * i + [1]]) for i in range(d_E)]
        calculated_values.append(optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                                            threshold_level=threshold_level,
                                            h=h,
                                            g_Es=g_Es,
                                            mu_value=mu_value, Sigma=Sigma, radius=radius))

        calculated_values = calculated_values[::-1]
        self.assertTrue(np.all(np.diff(np.array(calculated_values)) < 1e-9))
        logger.info("{:} optimization optimal value: {:}".format(function_name,
            '/'.join([f"{calculated_value:.2E}" for calculated_value in calculated_values])))

if __name__ == '__main__':
    unittest.main()