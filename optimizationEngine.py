from __future__ import annotations
from typing import List
import mosek.fusion as mf
import mosek
import numpy as np
from scipy.special import comb
from scipy.linalg import sqrtm
import bisect
import copy


class PolynomialFunction:
    def __init__(self, input_endpoints: List[float], coefficients: List[List[float]]):
        '''
        Polynomial function for x\geq input_endpoints[0]. 
        
        input_endpoints -> [a_0,...,a_I] and the last one a_I = np.inf
        the ith coefficients -> [y0,y1,...,yk] for polynomial_i(x)= y0+y1*x+y2*x^2+...+yk*x^k with a_i<= x \leq a_{i+1}, i = 0, ..., I-1
        REQUIRE:
            len(input_endpoints)-1 == len(coefficients)
            input_endpoints is sorted, in increasing order
            
        EXAMPLES:
            indicator functions: \mathbb{I}(x\geq a)          -> PolynomialFunction([a,np.inf],[[1]])
                                 \mathbb{I}(b\leq x\leq c)    -> PolynomialFunction([b,c,np.inf],[[1],[0]])
            power functions:     x\mathbb{I}(x\geq a)         -> PolynomialFunction([a,np.inf],[[0, 1]])
                                 x^i\mathbb{I}(x\geq a),i>=0  -> PolynomialFunction([a,np.inf],[[0...0 (#i), 1]])
        '''
        assert len(input_endpoints)-1 == len(coefficients)
        assert input_endpoints[-1] == np.inf
        assert input_endpoints == sorted(input_endpoints)
        self._input_endpoints = copy.deepcopy(input_endpoints)
        self._coefficients = copy.deepcopy(coefficients)
        self._threshold_level = self._input_endpoints[0]

    @property
    def input_endpoints(self):
        return self._input_endpoints

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def threshold_level(self):
        return self._threshold_level

    def multiply(self, multiple: float) -> None:
        for coefficient in self.coefficients:
            for i in range(len(coefficient)):
                coefficient[i] *= multiple

    def integration_riser(self, order: int = 1) -> 'PolynomialFunction':

        assert order >= 0
        if order == 0:
            return self
        if order > 1:
            return self.integration_riser(order-1).integration_riser(1)

        running_integral = 0
        riser_coefficients = []
        for coefficient_index in range(len(self.input_endpoints)-1):
            interval_left = self.input_endpoints[coefficient_index]
            interval_right = self.input_endpoints[coefficient_index+1]
            _k = len(self.coefficients[coefficient_index])-1
            riser_coefficient = np.zeros(_k+2)
            riser_coefficient[0] = running_integral-np.sum(interval_left**np.arange(
                1, _k+2)/np.arange(1, _k+2)*np.array(self.coefficients[coefficient_index]))
            riser_coefficient[1:] = np.array(
                self.coefficients[coefficient_index])/np.arange(1, _k+2)
            if interval_right < np.inf:
                running_integral += np.sum(interval_right**np.arange(1, _k+2)/np.arange(1, _k+2)*np.array(self.coefficients[coefficient_index])) -\
                    np.sum(interval_left**np.arange(1, _k+2)/np.arange(1,
                           _k+2)*np.array(self.coefficients[coefficient_index]))
            riser_coefficients.append(riser_coefficient.tolist())
#         print(self.input_endpoints[:-1],riser_coefficients)
        return PolynomialFunction(self.input_endpoints, riser_coefficients)

    def __eq__(self, other) -> bool:
        #         print(np.max(np.abs(np.array(self.input_endpoints[:-1])-np.array(other.input_endpoints[:-1]))),
        #               np.max(np.abs(np.array(self.coefficients)-np.array(other.coefficients)) ))
        return np.all(np.abs(np.array(self.input_endpoints[:-1])-np.array(other.input_endpoints[:-1])) < 1e-12) and \
            np.all(np.abs(np.array(self.coefficients) -
                   np.array(other.coefficients)) < 1e-12)


def off_diag_auxmat(k: int, sum_index: int) -> List[List[int]]:
    auxmat = np.zeros(shape=(k+1, k+1))
    for i in range(k+1):
        if sum_index-i < 0 or sum_index-i > k:
            continue
        auxmat[i][sum_index-i] = 1
    return auxmat.tolist()


def infinite_constraint(M: mf.Model, H: PolynomialFunction, G_Es: List[PolynomialFunction], G_Rs: List[PolynomialFunction]) -> None:
    input_endpoints = H.input_endpoints[:-1]
    if len(G_Es) == 0:
        G_Es = None
    if len(G_Rs) == 0:
        G_Rs = None
    if G_Es and G_Rs:
        for G_E in G_Es:
            for G_R in G_Rs:
                input_endpoints += G_R.input_endpoints[::-1]
            input_endpoints += G_E.input_endpoints[:-1]
    elif G_Es:
        for G_E in G_Es:
            input_endpoints += G_E.input_endpoints[:-1]
    elif G_Rs:
        for G_R in G_Rs:
            input_endpoints += G_R.input_endpoints[:-1]
    input_endpoints = list(set(input_endpoints))
    input_endpoints.sort()
    input_endpoints += [np.inf]
    if G_Es:
        constraint_ellipsoid_coefficient = mf.Expr.mul(
            M.getParameter('rSigma_sqrtminv'), M.getVariable('u'))
    if G_Rs:
        constraint_rectangle_coefficient = mf.Expr.sub(
            M.getVariable('lambda1'), M.getVariable('lambda2'))

    for interval_index in range(len(input_endpoints)-1):
        interval_left = input_endpoints[interval_index]
        interval_right = input_endpoints[interval_index+1]
        H_bisect_rpt = bisect.bisect_right(H.input_endpoints, interval_left)
        H_each_interval = H.coefficients[H_bisect_rpt -
                                         1] if H_bisect_rpt >= 1 else [0]
        if G_Es:
            G_E_bisect_rpts = [bisect.bisect_right(
                G_E.input_endpoints, interval_left) for G_E in G_Es]
            G_Es_each_interval = [G_E.coefficients[G_E_bisect_rpt-1] if G_E_bisect_rpt >=
                                  1 else [0] for G_E, G_E_bisect_rpt in zip(G_Es, G_E_bisect_rpts)]
        if G_Rs:
            G_R_bisect_rpts = [bisect.bisect_right(
                G_R.input_endpoints, interval_left) for G_R in G_Rs]
            G_Rs_each_interval = [G_R.coefficients[G_R_bisect_rpt-1] if G_R_bisect_rpt >=
                                  1 else [0] for G_R, G_R_bisect_rpt in zip(G_Rs, G_R_bisect_rpts)]

        if G_Es and G_Rs:
            k = max([len(H_each_interval),
                     max([len(G_E_each_interval)
                         for G_E_each_interval in G_Es_each_interval]),
                     max([len(G_R_each_interval) for G_R_each_interval in G_Rs_each_interval])])-1
        elif G_Es:
            k = max([len(H_each_interval),
                     max([len(G_E_each_interval)
                         for G_E_each_interval in G_Es_each_interval])
                     ])-1
        elif G_Rs:
            k = max([len(H_each_interval),
                     max([len(G_R_each_interval)
                         for G_R_each_interval in G_Rs_each_interval])
                     ])-1

        if G_Es:
            G_Es_each_interval_np = np.zeros(shape=(len(G_Es), k+1))
            for G_E_iter in range(len(G_Es)):
                G_Es_each_interval_np[G_E_iter, :len(
                    G_Es_each_interval[G_E_iter])] = G_Es_each_interval[G_E_iter]
        if G_Rs:
            G_Rs_each_interval_np = np.zeros(shape=(len(G_Rs), k+1))
            for G_R_iter in range(len(G_Rs)):
                G_Rs_each_interval_np[G_R_iter, :len(
                    G_Rs_each_interval[G_R_iter])] = G_Rs_each_interval[G_R_iter]
        y = [mf.Expr.zeros(1)]*(k+1)
        y[0] = mf.Expr.add(y[0], M.getVariable('kappa'))
        for k_iteration in range(k+1):
            y[k_iteration] = mf.Expr.add(y[k_iteration],
                                         -H_each_interval[k_iteration] if len(H_each_interval) > k_iteration else 0)
            if G_Es:
                y[k_iteration] = mf.Expr.add(y[k_iteration],
                                             mf.Expr.sum(mf.Expr.mulElm(
                                                 G_Es_each_interval_np[:, k_iteration].tolist(), constraint_ellipsoid_coefficient))
                                             )
            if G_Rs:
                y[k_iteration] = mf.Expr.add(y[k_iteration],
                                             mf.Expr.sum(mf.Expr.mulElm(
                                                 G_Rs_each_interval_np[:, k_iteration].tolist(), constraint_rectangle_coefficient))
                                             )
        infinite_constraint_basis(M, y, interval_left, interval_right)


def infinite_constraint_basis(M: mf.Model, y: List[mf.Expression], b: float, c: float = np.inf) -> None:
    '''
    A semi-definite representation of     
        if c==inf:
            sum_i=0^k y_{i}x^i\geq 0, x\geq b
        if c < inf:
            sum_i=0^k y_{i}x^i\geq 0, b\leq x \leq c
    '''
    k = len(y)-1
    if c == np.inf:
        V = M.variable('PSDmatrix_{:}_{:}'.format(
            b, c), mf.Domain.inPSDCone(k+1))
        for l in range(1, k+1):
            M.constraint('V_sumIndex=2*{:}-1_interval_left={:}'.format(
                l, b), mf.Expr.dot(off_diag_auxmat(k, 2*l-1), V), mf.Domain.equalsTo(0))
        for l in range(0, k+1):
            RHS_expr = mf.Expr.sum(mf.Expr.vstack(
                [mf.Expr.mul(comb(r, l)*b**(r-l), y[r]) for r in range(l, k+1)]))
            M.constraint('V_sumIndex=2*{:}_interval_left={:}'.format(l, b), mf.Expr.sub(
                mf.Expr.dot(off_diag_auxmat(k, 2*l), V), RHS_expr), mf.Domain.equalsTo(0))
    else:
        W = M.variable('PSDmatrix_{:}_{:}'.format(
            b, c), mf.Domain.inPSDCone(k+1))
        for l in range(1, k+1):
            M.constraint('W_sumIndex=2*{:}-1_interval_left={:}_interval_right={:}'.format(
                l, b, c), mf.Expr.dot(off_diag_auxmat(k, 2*l-1), W), mf.Domain.equalsTo(0))
        for l in range(0, k+1):
            RHS_expr = mf.Expr.sum(mf.Expr.vstack([mf.Expr.mul(comb(
                r, m)*comb(k-r, l-m)*b**(r-m)*c**m, y[r]) for m in range(0, l+1) for r in range(m, k+m-l+1)]))
            M.constraint('W_sumIndex=2*{:}_interval_left={:}_interval_right={:}'.format(
                l, b, c), mf.Expr.sub(mf.Expr.dot(off_diag_auxmat(k, 2*l), W), RHS_expr), mf.Domain.equalsTo(0))
    return


def optimization(D_riser_number: int = None, eta: float = None, eta_lb: float = None, eta_ub: float = None, nu: float = None,
                 threshold_level: float = None,
                 h: 'PolynomialFunction' = None,
                 g_Es: List['PolynomialFunction'] = None, mu_value: np.ndarray = None, Sigma: np.ndarray = None, radius: float = None,
                 g_Rs: List['PolynomialFunction'] = None, mu_lb_value: np.ndarray = None, mu_ub_value: np.ndarray = None) -> float:

    if D_riser_number is None or threshold_level is None or h is None or (g_Es is None and g_Rs is None):
        assert False
    if D_riser_number == 1:
        assert (eta is not None)
    if D_riser_number == 2:
        assert (eta_ub is not None) and (
            eta_lb is not None) and (nu is not None)
    if g_Es is None:
        g_Es = []
    if g_Rs is None:
        g_Rs = []

    # with mf.Model("base_problem") as M:
    M = mf.Model("optimization_problem")
    H = h.integration_riser(D_riser_number)

    ## probabilitiy distribution
    if D_riser_number == 0:
        kappa = M.variable("kappa", 1, mf.Domain.greaterThan(0))
    else:
        kappa = M.variable("kappa", 1, mf.Domain.unbounded(1))
        if D_riser_number == 1:
            H.multiply(eta)
        if D_riser_number == 2:
            H.multiply(nu)
            if mu_lb_value is None:
                mu_lb_value = np.array([eta_lb])
            else:
                mu_lb_value = np.append(mu_lb_value, eta_lb)
            if mu_ub_value is None:
                mu_ub_value = np.array([eta_ub])
            else:
                mu_ub_value = np.append(mu_ub_value, eta_ub)
    if len(g_Es) == 0:
        ellipsoid_obj_expr = mf.Expr.zeros(1)
        G_Es = []
    else:
        if D_riser_number == 0:
            G_Es = g_Es
        elif D_riser_number == 1:
            G_Es = [g_E.integration_riser(D_riser_number) for g_E in g_Es]
            for G in G_Es:
                G.multiply(eta)
        elif D_riser_number == 2:
            G_Es = [g_E.integration_riser(D_riser_number) for g_E in g_Es]
            for G in G_Es:
                G.multiply(nu)
        if Sigma.size == 1:
            rSigma_sqrtminv_value = 1/np.sqrt(radius * Sigma)
        else:
            rSigma_sqrtminv_value = np.linalg.inv(sqrtm(radius * Sigma))

        mu = M.parameter("mu_value", len(mu_value))
        mu.setValue(mu_value.tolist())
        if Sigma.size == 1:
            rSigma_sqrtminv = M.parameter("rSigma_sqrtminv", 1, 1)
        else:
            rSigma_sqrtminv = M.parameter(
                "rSigma_sqrtminv", rSigma_sqrtminv_value.shape[0], rSigma_sqrtminv_value.shape[1])
        rSigma_sqrtminv.setValue(rSigma_sqrtminv_value.tolist())
        lambda_ = M.variable("lambda", mf.Domain.greaterThan(0))
        u = M.variable("u", mf.Domain.unbounded(mu.getSize()))
        M.constraint("second-order-cone",
                     mf.Var.vstack(lambda_, u), mf.Domain.inQCone())
        if Sigma.size == 1:
            ellipsoid_obj_expr = mf.Expr.add(lambda_,
                                             mf.Expr.mul(mf.Expr.mul(mf.Expr.reshape(u, 1, 1),
                                                                     rSigma_sqrtminv), mu))
        else:
            ellipsoid_obj_expr = mf.Expr.add(lambda_,
                                             mf.Expr.mul(mf.Expr.mul(mf.Expr.reshape(u, 1, u.getSize()),
                                                                     rSigma_sqrtminv),
                                                         mu
                                                         )
                                             )
    if len(g_Rs) == 0 and D_riser_number != 2:
        rectangle_obj_expr = mf.Expr.zeros(1)
        G_Rs = []
    elif len(g_Rs) == 0 and D_riser_number == 2:
        G_Rs = [PolynomialFunction([threshold_level, np.inf], [
                                   [-threshold_level * nu, nu]])]
        mu_lb = M.parameter("mu_lb", len(mu_lb_value))
        mu_lb.setValue(mu_lb_value.tolist())
        mu_ub = M.parameter("mu_ub", len(mu_ub_value))
        mu_ub.setValue(mu_ub_value.tolist())
        lambda1 = M.variable(
            "lambda1", mf.Domain.greaterThan(0, mu_ub.getSize()))
        lambda2 = M.variable(
            "lambda2", mf.Domain.greaterThan(0, mu_lb.getSize()))
        rectangle_obj_expr = mf.Expr.sub(mf.Expr.dot(
            lambda1, mu_ub), mf.Expr.dot(lambda2, mu_lb))
    else:
        if D_riser_number == 0:
            G_Rs = g_Rs
        elif D_riser_number == 1:
            G_Rs = [g_R.integration_riser(D_riser_number) for g_R in g_Rs]
            for G in G_Rs:
                G.multiply(eta)
        elif D_riser_number == 2:
            G_Rs = [g_R.integration_riser(D_riser_number) for g_R in g_Rs]
            for G in G_Rs:
                G.multiply(nu)
        mu_lb = M.parameter("mu_lb", len(mu_lb_value))
        mu_lb.setValue(mu_lb_value.tolist())
        mu_ub = M.parameter("mu_ub", len(mu_ub_value))
        mu_ub.setValue(mu_ub_value.tolist())
        lambda1 = M.variable(
            "lambda1", mf.Domain.greaterThan(0, mu_ub.getSize()))
        lambda2 = M.variable(
            "lambda2", mf.Domain.greaterThan(0, mu_lb.getSize()))
        rectangle_obj_expr = mf.Expr.sub(mf.Expr.dot(
            lambda1, mu_ub), mf.Expr.dot(lambda2, mu_lb))
        if D_riser_number == 2:
            G_Rs += [PolynomialFunction([threshold_level, np.inf],
                                        [[-threshold_level * nu, nu]])]
    ## objective function
    M.objective('obj', mf.ObjectiveSense.Minimize, mf.Expr.add(
        mf.Expr.add(kappa, ellipsoid_obj_expr), rectangle_obj_expr))
    ## infinite constraint
    infinite_constraint(M, H, G_Es, G_Rs)
    
    ## solve!
    M.solve()
    ## solution
    return M.primalObjValue()


def test_PolynomialFunction_integration_riser():
    ### indicator functions
    ####  \mathbb{I}(x\geq a)       -> PolynomialFunction([a],[[1]])
    poly1 = PolynomialFunction([1, np.inf], [[1]])
    assert poly1.integration_riser() == PolynomialFunction([
        1, np.inf], [[-1, 1]])
    assert poly1.integration_riser(
        1) == PolynomialFunction([1, np.inf], [[-1, 1]])
    assert poly1.integration_riser(2) == PolynomialFunction(
        [1, np.inf], [[1/2, -1, 1/2]])
    assert poly1.integration_riser(3) == PolynomialFunction(
        [1, np.inf], [[-1/6, 1/2, -1/2, 1/6]])
    assert poly1.integration_riser(4) == PolynomialFunction(
        [1, np.inf], [[1/24, -1/6, 1/4, -1/6, 1/24]])
    for _ in range(10):
        a = np.random.rand(1)[0]
        polya = PolynomialFunction([a, np.inf], [[1]])
        assert polya.integration_riser() == PolynomialFunction([
            a, np.inf], [[-a, 1]])
        assert polya.integration_riser(
            1) == PolynomialFunction([a, np.inf], [[-a, 1]])
        assert polya.integration_riser(2) == PolynomialFunction(
            [a, np.inf], [[a**2/2, -a, 1/2]])
        assert polya.integration_riser(3) == PolynomialFunction(
            [a, np.inf], [[-a**3/6, a**2/2, -a/2, 1/6]])
        assert polya.integration_riser(4) == PolynomialFunction(
            [a, np.inf], [[a**4/24, -a**3/6, a**2/4, -a/6, 1/24]])
    #### \mathbb{I}(b\leq x\leq c) -> PolynomialFunction([b,c],[[1],[0]])
    poly2 = PolynomialFunction([1, 2, np.inf], [[1], [0]])
    assert poly2.integration_riser(1) == PolynomialFunction(
        [1, 2, np.inf], [[-1, 1], [1, 0]])
    assert poly2.integration_riser(2) == PolynomialFunction(
        [1, 2, np.inf], [[1/2, -1, 1/2], [-3/2, 1, 0]])
    for _ in range(10):
        LR = np.random.rand(2)
        L = np.sort(LR)[0]
        R = np.sort(LR)[1]
        polyLR = PolynomialFunction([L, R, np.inf], [[1], [0]])
        assert polyLR.integration_riser(1) == PolynomialFunction(
            [L, R, np.inf], [[-L, 1], [R-L, 0]])
        assert polyLR.integration_riser(2) == PolynomialFunction(
            [L, R, np.inf], [[L**2/2, -L, 1/2], [-(R-L)*(R+L)/2, R-L, 0]])
    ## power functions
    ### x^i\mathbb{I}(x\geq a),i>=0
    for i in range(0, 10):
        a = np.random.rand(1)[0]
        polyi = PolynomialFunction([a, np.inf], [[0]*i+[1]])
        assert polyi.integration_riser(1) == PolynomialFunction(
            [a, np.inf], [[-a**(i+1)/(i+1)]+[0]*i+[1/(i+1)]])
        assert polyi.integration_riser(2) == PolynomialFunction(
            [a, np.inf], [[a**(i+2)/(i+2)]+[-a**(i+1)/(i+1)]+[0]*(i)+[1/(i+1)/(i+2)]])
    print("All unit tests on PolynomialFunction_integration_riser pass.")


if __name__ == '__main__':
    test_PolynomialFunction_integration_riser()
