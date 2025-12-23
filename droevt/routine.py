from copy import deepcopy
import typing

import numpy as np

from .engine import optimization, PolynomialFunction
from .calibration import eta_specification, nu_specification, z_of_chi_square, z_of_kolmogorov

def _optimization_plain_chi_square(data: np.ndarray,
                                   threshold: float,
                                   objective_function: PolynomialFunction,
                                   moment_constraint_functions: typing.List[PolynomialFunction],
                                   mu: np.ndarray,
                                   Sigma: np.ndarray,
                                   bootstrapping_size: int,
                                   bootstrapping_seed: int,
                                   alpha: float,
                                   num_multi_threshold: int,
                                   right_endpoint: float=np.inf) -> float:
    """
    Perform optimization with plain chi-square constraints.

    (0, chi2)

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    threshold : float
        Threshold value for optimization.
    objective_function : PolynomialFunction
        Objective function to optimize.
    moment_constraint_functions : List[PolynomialFunction]
        List of moment constraint functions.
    mu : np.ndarray
        Mean vector.
    Sigma : np.ndarray
        Covariance matrix.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Random seed for bootstrapping.
    alpha : float
        Significance level.
    num_multi_threshold : int
        Number of multiple thresholds.
    right_endpoint : float
        Right end point of probability distribution.

    Returns
    -------
    float
        Optimal value from optimization.
    """
    D_riser_number = 0
    g_ellipsoidal_dimension = len(moment_constraint_functions)
    z = z_of_chi_square(alpha=alpha,
                        D_riser_number=D_riser_number,
                        g_dimension=g_ellipsoidal_dimension,
                        num_multi_threshold=num_multi_threshold)
    radius = z/data.shape[0]
    return optimization(D_riser_number=D_riser_number,
                        threshold_level=threshold,
                        h=objective_function,
                        g_Es=moment_constraint_functions,
                        mu_value=mu, Sigma=Sigma, radius=radius,
                        right_endpoint=right_endpoint)

def _optimization_monotone_chi_square(data: np.ndarray,
                                      threshold: float,
                                      objective_function: PolynomialFunction,
                                      moment_constraint_functions: typing.List[PolynomialFunction],
                                      mu: np.ndarray,
                                      Sigma: np.ndarray,
                                      bootstrapping_size: int,
                                      bootstrapping_seed: int,
                                      alpha: float,
                                      num_multi_threshold: int,
                                      right_endpoint: float=np.inf) -> float:
    """
    Perform optimization with monotone chi-square constraints.

    (1, chi2)
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
    threshold : float
        Threshold value for optimization.
    objective_function : PolynomialFunction
        Objective function to optimize.
    moment_constraint_functions : List[PolynomialFunction]
        List of moment constraint functions.
    mu : np.ndarray
        Mean vector.
    Sigma : np.ndarray
        Covariance matrix.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Random seed for bootstrapping.
    alpha : float
        Significance level.
    num_multi_threshold : int
        Number of multiple thresholds.
    right_endpoint : float
        Right end point of probability distribution.
    Returns
    -------
    float
        Optimal value from optimization with monotone chi-square constraints.

    """
    D_riser_number = 1
    g_ellipsoidal_dimension = len(moment_constraint_functions)
    eta = eta_specification(data=data,
                           threshold=threshold,
                           alpha=alpha,
                           bootstrapping_size=bootstrapping_size,
                           bootstrapping_seed=bootstrapping_seed,
                           D_riser_number=D_riser_number,
                           num_multi_threshold=num_multi_threshold)
    z = z_of_chi_square(alpha=alpha,
                        D_riser_number=D_riser_number,
                        g_dimension=g_ellipsoidal_dimension,
                        num_multi_threshold=num_multi_threshold)
    radius = z/data.shape[0]
    return optimization(D_riser_number=D_riser_number,
                        eta=eta,
                        threshold_level=threshold,
                        h=objective_function,
                        g_Es=moment_constraint_functions,
                        mu_value=mu, Sigma=Sigma, radius=radius,
                        right_endpoint=right_endpoint)

def _optimization_convex_chi_square(data: np.ndarray,
                                threshold: float,
                                objective_function: PolynomialFunction,
                                moment_constraint_functions: typing.List[PolynomialFunction],
                                mu: np.ndarray,
                                Sigma: np.ndarray,
                                bootstrapping_size: int,
                                bootstrapping_seed: int,
                                alpha: float,
                                num_multi_threshold: int,
                                right_endpoint: float=np.inf) -> float:
    """
    Perform optimization with convex chi-square constraints.

    (2, chi2)

    Parameters
    ----------
    data : np.ndarray
        Input data.
    threshold : float
        Threshold value.
    objective_function : PolynomialFunction
        Objective function to optimize.
    moment_constraint_functions : List[PolynomialFunction]
        List of moment constraint functions.
    mu : np.ndarray
        Mean vector.
    Sigma : np.ndarray
        Covariance matrix.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Random seed for bootstrapping.
    alpha : float
        Significance level.
    num_multi_threshold : int
        Number of multiple thresholds.
    right_endpoint : float
        Right end point of probability distribution.
    Returns
    -------
    float
        Optimal value from optimization with convex chi-square constraints.
    """
    D_riser_number = 2
    g_ellipsoidal_dimension = len(moment_constraint_functions)

    [eta_lb, eta_ub] = eta_specification(data=data,
                                        alpha=alpha,
                                        threshold=threshold,
                                        bootstrapping_size=bootstrapping_size,
                                        bootstrapping_seed=bootstrapping_seed,
                                        D_riser_number=D_riser_number,
                                        num_multi_threshold=num_multi_threshold)

    nu = nu_specification(data=data,
                          threshold=threshold,
                          alpha=alpha,
                          bootstrapping_size=bootstrapping_size,
                          bootstrapping_seed=bootstrapping_seed,
                          num_multi_threshold=num_multi_threshold)
    z = z_of_chi_square(alpha=alpha,
                        D_riser_number=D_riser_number,
                        g_dimension=g_ellipsoidal_dimension,
                        num_multi_threshold=num_multi_threshold)

    radius = z/data.shape[0]
    return optimization(D_riser_number=D_riser_number, eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                        threshold_level=threshold,
                        h=objective_function,
                        g_Es=moment_constraint_functions,
                        mu_value=mu, Sigma=Sigma, radius=radius,
                        right_endpoint=right_endpoint)

def _optimization_plain_kolmogorov(data: np.ndarray,
                                   threshold: float,
                                   objective_function: PolynomialFunction,
                                   bootstrapping_size: int,
                                   bootstrapping_seed: int,
                                   alpha: float,
                                   num_multi_threshold: int,
                                   right_endpoint: float=np.inf) -> float:
    """
    Optimization with plain Kolmogorov-Smirnov constraints.

    (0, ks)

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    threshold : float
        Threshold value.
    objective_function : PolynomialFunction
        Objective function to optimize.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Random seed for bootstrapping.
    alpha : float
        Significance level.
    num_multi_threshold : int
        Number of multiple thresholds.
    right_endpoint : float
        Right end point of probability distribution.
    Returns
    -------
    float
        Optimal value from optimization with plain Kolmogorov-Smirnov constraints.
    """
    new_objective_function = deepcopy(objective_function)
    D_riser_number = 0
    data_over_threshold = np.sort(data[data > threshold])
    size_over_threshold = np.sum(data > threshold)
    size_on_data = data.shape[0]

    z = z_of_kolmogorov(alpha=alpha,
                        D_riser_number=D_riser_number,
                        num_multi_threshold=num_multi_threshold)
    mu_lb_value = np.maximum(0, (size_on_data+1-np.arange(
        size_on_data-size_over_threshold+1, size_on_data+1))/size_on_data-z/np.sqrt(size_on_data))
    mu_ub_value = np.minimum(1, (size_on_data-np.arange(
        size_on_data-size_over_threshold+1, size_on_data+1))/size_on_data+z/np.sqrt(size_on_data))

    constraint_functions = [PolynomialFunction(
        [xi, np.inf], [[0]*0+[1]]) for xi in data_over_threshold]

    return optimization(D_riser_number=D_riser_number,
                        threshold_level=threshold,
                        h=new_objective_function,
                        g_Rs=constraint_functions,
                        mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value,
                        right_endpoint=right_endpoint)


def _optimization_monotone_kolmogorov(data: np.ndarray,
                                      threshold: float,
                                      objective_function: PolynomialFunction,
                                      bootstrapping_size: int,
                                      bootstrapping_seed: int,
                                      alpha: float,
                                      num_multi_threshold: int,
                                      right_endpoint: float=np.inf) -> float:
    """
    Optimization with monotone Kolmogorov-Smirnov constraints.

    (1, ks)

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    threshold : float
        Threshold value.
    objective_function : PolynomialFunction
        Objective function to optimize.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Random seed for bootstrapping.
    alpha : float
        Significance level.
    num_multi_threshold : int
        Number of multiple thresholds.
    right_endpoint : float
        Right end point of probability distribution.
    Returns
    -------
    float
        Optimal value from optimization with monotone Kolmogorov-Smirnov constraints.
    """
    new_objective_function = deepcopy(objective_function)
    D_riser_number = 1
    data_over_threshold = np.sort(data[data > threshold])
    size_over_threshold = np.sum(data > threshold)
    size_on_data = data.shape[0]

    z = z_of_kolmogorov(alpha=alpha,
                        D_riser_number=D_riser_number,
                        num_multi_threshold=num_multi_threshold)
    mu_lb_value = np.maximum(0, (size_on_data+1-np.arange(
        size_on_data-size_over_threshold+1, size_on_data+1))/size_on_data-z/np.sqrt(size_on_data))
    mu_ub_value = np.minimum(1, (size_on_data-np.arange(
        size_on_data-size_over_threshold+1, size_on_data+1))/size_on_data+z/np.sqrt(size_on_data))

    constraint_functions = [PolynomialFunction(
        [xi, np.inf], [[0]*0+[1]]) for xi in data_over_threshold]

    eta = eta_specification(data=data,
                           threshold=threshold,
                           alpha=alpha,
                           D_riser_number=D_riser_number,
                           bootstrapping_size=bootstrapping_size,
                           bootstrapping_seed=bootstrapping_seed,
                           num_multi_threshold=num_multi_threshold)

    return optimization(D_riser_number=D_riser_number,
                        eta=eta,
                        threshold_level=threshold,
                        h=new_objective_function,
                        g_Rs=constraint_functions,
                        mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value,
                        right_endpoint=right_endpoint)


def _optimization_convex_kolmogorov(data: np.ndarray,
                                    threshold: float,
                                    objective_function: PolynomialFunction,
                                    bootstrapping_size: int,
                                    bootstrapping_seed: int,
                                    alpha: float,
                                    num_multi_threshold: int,
                                    right_endpoint: float=np.inf) -> float:
    """
    Optimization with convex Kolmogorov-Smirnov constraints.

    (2, ks)
    
    Parameters
    ----------
    data : np.ndarray
        Input data array.
    threshold : float
        Threshold value.
    objective_function : PolynomialFunction
        Objective function to optimize.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.
    alpha : float
        Significance level.
    num_multi_threshold : int
        Number of multiple thresholds.
    right_endpoint : float
        Right end point of probability distribution.
    Returns
    -------
    float
        Optimal value from optimization with convex Kolmogorov-Smirnov constraints.
    """
    new_objective_function = deepcopy(objective_function)
    D_riser_number = 2
    data_over_threshold = np.sort(data[data > threshold])
    size_over_threshold = np.sum(data > threshold)
    size_on_data = data.shape[0]

    z = z_of_kolmogorov(alpha=alpha,
                        D_riser_number=D_riser_number,
                        num_multi_threshold=num_multi_threshold)
    mu_lb_value = np.maximum(0, (size_on_data+1-np.arange(
        size_on_data-size_over_threshold+1, size_on_data+1))/size_on_data-z/np.sqrt(size_on_data))
    mu_ub_value = np.minimum(1, (size_on_data-np.arange(
        size_on_data-size_over_threshold+1, size_on_data+1))/size_on_data+z/np.sqrt(size_on_data))

    constraint_functions = [PolynomialFunction(
        [xi, np.inf], [[0]*0+[1]]) for xi in data_over_threshold]

    [eta_lb, eta_ub] = eta_specification(data=data,
                                        threshold=threshold,
                                        alpha=alpha,
                                        D_riser_number=D_riser_number,
                                        bootstrapping_size=bootstrapping_size,
                                        bootstrapping_seed=bootstrapping_seed,
                                        num_multi_threshold=num_multi_threshold)

    nu = nu_specification(data=data,
                          threshold=threshold,
                          alpha=alpha,
                          bootstrapping_size=bootstrapping_size,
                          bootstrapping_seed=bootstrapping_seed,
                          num_multi_threshold=num_multi_threshold)

    return optimization(D_riser_number=D_riser_number,
                        eta_lb=eta_lb, eta_ub=eta_ub, nu=nu,
                        threshold_level=threshold,
                        h=new_objective_function,
                        g_Rs=constraint_functions,
                        mu_lb_value=mu_lb_value, mu_ub_value=mu_ub_value,
                        right_endpoint=right_endpoint)


def optimization_with_rectangular_constraint(D: int, input_data: np.ndarray,
                                             threshold_percentage: typing.Union[float, typing.List[float]],
                                             alpha: float,
                                             left_end_point_objective: float, right_end_point_objective: float,
                                             bootstrapping_size: int, bootstrapping_seed: int,
                                             right_endpoint: float=np.inf,
                                             is_max: bool = True) -> float:
    """
    Perform optimization with rectangular constraint.

    Parameters
    ----------
    D : int
        Order of derivative constraint (0=plain, 1=monotone, 2=convex).
    input_data : np.ndarray
        Input data array for optimization.
    threshold_percentage : float or List[float]
        Single threshold percentage or list of threshold percentages.
    alpha : float
        Significance level (confidence level = 1-alpha).
    left_end_point_objective : float
        Left endpoint of objective function interval.
    right_end_point_objective : float
        Right endpoint of objective function interval.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.
    right_endpoint : float
        Right end point of probability distribution.
    is_max : bool, optional
        Whether the objective function is maximized.
    Returns
    -------
    float
        Minimum value across optimizations at different thresholds.

    Notes
    -----
    The confidence level for the rectangular constraint is set to 1-alpha.
    """
        
    if right_end_point_objective == np.inf:
        h = PolynomialFunction(
            [left_end_point_objective, np.inf], [[1]])
    else:
        h = PolynomialFunction(
            [left_end_point_objective, right_end_point_objective, np.inf], [[1], [0]])
    if not is_max:
        h.multiply(-1)

    if isinstance(threshold_percentage, float):
        num_multi_threshold = 1
        thresholds = [np.quantile(input_data, threshold_percentage)]
    else:
        assert isinstance(threshold_percentage, list) and len(
            threshold_percentage) >= 2
        assert all(isinstance(
            each_threshold_percentage, float) for each_threshold_percentage in threshold_percentage)
        num_multi_threshold = len(threshold_percentage)
        thresholds = [np.quantile(input_data, each_threshold_percentage)
                      for each_threshold_percentage in threshold_percentage]

    optimization_functions = {
        0: _optimization_plain_kolmogorov,
        1: _optimization_monotone_kolmogorov,
        2: _optimization_convex_kolmogorov
    }.get(D, None)

    if optimization_functions is None:
        raise ValueError(f"Invalid D value: {D}")

    value = float(np.min([optimization_functions(data=input_data,
                                          threshold=threshold,
                                          objective_function=h,
                                          bootstrapping_size=bootstrapping_size,
                                          bootstrapping_seed=bootstrapping_seed,
                                          alpha=alpha,
                                          num_multi_threshold=num_multi_threshold,
                                          right_endpoint=right_endpoint) for threshold in thresholds]))
    if not is_max:
        value = min(max(- value, 0), 1) 
    return value

def optimization_with_ellipsodial_constraint(D: int, input_data: np.ndarray,
                                             threshold_percentage: typing.Union[float, typing.List[float]],
                                             alpha: float,
                                             left_end_point_objective: float, right_end_point_objective: float,
                                             g_ellipsoidal_dimension: int,
                                             bootstrapping_size: int, bootstrapping_seed: int,
                                             right_endpoint: float = np.inf,
                                             is_max: bool = True, ) -> float:
    """
    Optimization with ellipsoidal constraint.

    Parameters
    ----------
    D : int
        Order of derivative (0: plain, 1: monotone, 2: convex).
    input_data : np.ndarray
        Input data array.
    threshold_percentage : float or List[float]
        Percentile(s) for threshold calculation.
    alpha : float
        Significance level (confidence level = 1-alpha).
    left_end_point_objective : float
        Left endpoint for objective function.
    right_end_point_objective : float
        Right endpoint for objective function.
    g_ellipsoidal_dimension : int
        Dimension of ellipsoidal constraint.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.
    right_endpoint : float
        Right end point of probability distribution.
    is_max : bool, optional
        Whether the objective function is maximized.
    Returns
    -------
    float
        Minimum value across optimizations at different thresholds.

    Notes
    -----
    This function performs optimization with an ellipsoidal constraint where
    the confidence level is set to 1-alpha.
    """
    if right_end_point_objective == np.inf:
        h = PolynomialFunction(
            [left_end_point_objective, np.inf], [[1]])
    else:
        h = PolynomialFunction(
            [left_end_point_objective, right_end_point_objective, np.inf], [[1], [0]])

    if not is_max:
        h.multiply(-1)

    if isinstance(threshold_percentage, float):
        num_multi_threshold = 1
        thresholds = [np.quantile(input_data, threshold_percentage)]
        g_EsList = [[PolynomialFunction([thresholds[0], np.inf], [[0] * i + [1]])
                     for i in range(g_ellipsoidal_dimension)]]
        muList = [np.array([np.sum(input_data**power*(input_data > thresholds[0])) /
                            input_data.size for power in range(g_ellipsoidal_dimension)])]
        SigmaList = [np.cov(np.vstack(
            [(input_data > thresholds[0])*1.0*input_data**power for power in range(g_ellipsoidal_dimension)]))]
    else:
        assert isinstance(threshold_percentage, list) and len(
            threshold_percentage) >= 2
        assert all(isinstance(
            each_threshold_percentage, float) for each_threshold_percentage in threshold_percentage)

        num_multi_threshold = len(threshold_percentage)
        thresholds = [np.quantile(input_data, each_threshold_percentage)
                      for each_threshold_percentage in threshold_percentage]
        g_EsList = [[PolynomialFunction([threshold, np.inf], [[0] * i + [1]])
                     for i in range(g_ellipsoidal_dimension)] for threshold in thresholds]
        muList = [np.array([np.sum(input_data**power*(input_data > threshold)) /
                            input_data.size for power in range(g_ellipsoidal_dimension)]) for threshold in thresholds]
        SigmaList = [np.cov(np.vstack(
            [(input_data > threshold)*1.0*input_data**power for power in range(g_ellipsoidal_dimension)])) for threshold in thresholds]

    optimization_functions = {
        0: _optimization_plain_chi_square,
        1: _optimization_monotone_chi_square,
        2: _optimization_convex_chi_square
    }.get(D, None)

    if optimization_functions is None:
        raise ValueError(f"Invalid D value: {D}")

    value = float(np.min([optimization_functions(data=input_data,
                                          threshold=threshold,
                                          objective_function=h,
                                          moment_constraint_functions=g_Es,
                                          mu=mu,
                                          Sigma=Sigma,
                                          bootstrapping_size=bootstrapping_size,
                                          bootstrapping_seed=bootstrapping_seed,
                                          alpha=alpha,
                                          num_multi_threshold=num_multi_threshold,
                                          right_endpoint=right_endpoint) for threshold, g_Es, mu, Sigma in zip(thresholds, g_EsList, muList, SigmaList)])
                                          )
    if not is_max:
        value = min(max(- value, 0), 1)
    return value

if __name__ == '__main__':
    pass
