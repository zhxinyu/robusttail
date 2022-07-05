from optimization_engine import optimization, PolynomialFunction
import numpy as np


def test_0_ks_ex0():
    mu_lb_value = np.array([0.6])
    mu_ub_value = np.array([0.8])
    D_riser_number = 0
    L = 1
    R = 2
    h = PolynomialFunction([L, R, np.inf], [[1], [0]])
    threshold_level = np.random.rand(1)[0]
    g_Rs = [PolynomialFunction([threshold_level, np.inf], [[0]*0+[1]])]
    calculated_value = optimization(D_riser_number=D_riser_number,
                                    threshold_level=threshold_level,
                                    h=h,
                                    g_Rs=g_Rs, mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value)
    expected_value = 0.8
    assert np.abs(expected_value-calculated_value) < 1e-6
    print("0_ks_ex1 optimization optimal value: {:}".format(calculated_value))


def test_0_ks_ex1():
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

    # expected_value = 0.8
    assert np.abs(0.17-calculated_values[0]) < 1e-6
    assert np.all(np.diff(np.array(calculated_values)) < 0)
    print("0_ks_ex2 optimization optimal value: {:}".format(
        '/'.join([str(calculated_value) for calculated_value in calculated_values])))


def test_1_ks_ex1():
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

    # expected_value = 0.8
    # assert np.abs(0.17-calculated_values[0])<1e-6
    assert np.all(np.diff(np.array(calculated_values)) < 0)
    print("1_ks_ex1 optimization optimal value: {:}".format(
        '/'.join([str(calculated_value) for calculated_value in calculated_values])))


def test_2_ks_ex1():
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

    # expected_value = 0.8
    # assert np.abs(0.17-calculated_values[0])<1e-6
    assert np.all(np.diff(np.array(calculated_values)) < 0)
    print("2_ks_ex1 optimization optimal value: {:}".format(
        '/'.join([str(calculated_value) for calculated_value in calculated_values])))


def test_0_chi2_ex1():
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

    # print(np.all(np.diff(np.array(calculated_values)) > 0))
    print("0_chi2_ex1 optimization optimal value: {:}".format(
        '/'.join([str(calculated_value) for calculated_value in calculated_values[::-1]])))


def test_1_chi2_ex1():
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

    # print(np.all(np.diff(np.array(calculated_values)) > 0))
    print("1_chi2_ex1 optimization optimal value: {:}".format(
        '/'.join([str(calculated_value) for calculated_value in calculated_values[::-1]])))


def test_2_chi2_ex1():
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

    # print(np.all(np.diff(np.array(calculated_values)) > 0))
    print("2_chi2_ex1 optimization optimal value: {:}".format(
        '/'.join([str(calculated_value) for calculated_value in calculated_values[::-1]])))


if __name__ == '__main__':
    test_0_ks_ex1()
    test_1_ks_ex1()
    test_2_ks_ex1()
    test_0_chi2_ex1()
    test_1_chi2_ex1()
    test_2_chi2_ex1()
    # .integration_riser(2)
    # G_Es   = [PolynomialFunction([threshold_level],[[0]*i+[1]]) for i in range(d_E)]
    # xi_obs = np.random.rand(d_R-1)*3+threshold_level
    # xi_obs = [threshold_level]+np.sort(xi_obs).tolist()
    # G_Rs   = [PolynomialFunction([xi_obs[i],xi_obs[i+1]],[[1],[0]]) for i in range(d_R-1)]+[PolynomialFunction([xi_obs[-1]],[[1]])]
