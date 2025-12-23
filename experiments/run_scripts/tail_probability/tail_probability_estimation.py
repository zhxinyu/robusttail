import droevt.utils.synthetic_data_generator as data_utils
import droevt.routine as droevt_routine
import typing
from scipy.stats import gamma
import numpy as np 

"""
This module provides functions for estimating tail probabilities using both rectangular
and ellipsoidal constraint optimization methods. It includes functionality for different
derivative constraints:

- D=0: No shape constraint on the tail probability function.
- D=1: Monotone decreasing constraint in the tail region.
- D=2: Convex constraint in the tail region.

These constraints help in producing more realistic and stable estimates of the tail probability,
especially for different types of distributions and tail behaviors.
"""

def _estimate_tail_probability_rectangular_constraint(data_module,
                                                      percentage_lhs: float, percentage_rhs: float,
                                                      data_size: int, threshold_percentage: typing.Union[float, typing.List[float]],
                                                      alpha: float,
                                                      random_state: int,
                                                      bootstrapping_size: int) -> typing.List[float]:
    """
    Estimate tail probability using rectangular constraint optimization.

    This function estimates the tail probability of a given distribution using rectangular
    constraint optimization for three different derivative constraints: D=0 (no shape constraint), D=1 (monotone decreasing in the tail region), and D=2 (convex in the tail region).

    Parameters:
    -----------
    data_module : module
        Module containing the data generation and quantile functions.
    percentage_lhs : float
        Left-hand side percentage for quantile calculation.
    percentage_rhs : float
        Right-hand side percentage for quantile calculation.
    data_size : int
        Size of the synthetic data to generate.
    threshold_percentage : float or List[float]
        Threshold percentage(s) for optimization.
    alpha : float
        Significance level for the optimization.
    random_state : int
        Random seed for reproducibility.
    bootstrapping_size: int
        Size of the bootstrap samples.
    Returns:
    --------
    List[float]
        A list containing three tail probability estimates, one for each derivative constraint (D=0, D=1, D=2).
    """
    left_end_point_objective = data_utils.get_quantile(
        data_module, percentage_lhs, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
    right_end_point_objective = data_utils.get_quantile(
        data_module, percentage_rhs, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
    input_data = data_utils.generate_synthetic_data(
        data_module, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name], data_size, random_state)
    tail_probability_estimates = [0]*3
    tail_probability_estimates[0] = droevt_routine.optimization_with_rectangular_constraint(D=0,
                                                                                            input_data=input_data,
                                                                                            threshold_percentage=threshold_percentage,
                                                                                            alpha=alpha,
                                                                                            left_end_point_objective=left_end_point_objective, 
                                                                                            right_end_point_objective=right_end_point_objective,
                                                                                            bootstrapping_size=bootstrapping_size, bootstrapping_seed=7*random_state+1)
    tail_probability_estimates[1] = droevt_routine.optimization_with_rectangular_constraint(D=1,
                                                                                            input_data=input_data,
                                                                                            threshold_percentage=threshold_percentage,
                                                                                            alpha=alpha,
                                                                                            left_end_point_objective=left_end_point_objective, 
                                                                                            right_end_point_objective=right_end_point_objective,
                                                                                            bootstrapping_size=bootstrapping_size, bootstrapping_seed=7*random_state+1)
    tail_probability_estimates[2] = droevt_routine.optimization_with_rectangular_constraint(D=2,
                                                                                            input_data=input_data,
                                                                                            threshold_percentage=threshold_percentage,
                                                                                            alpha=alpha,
                                                                                            left_end_point_objective=left_end_point_objective, 
                                                                                            right_end_point_objective=right_end_point_objective,
                                                                                            bootstrapping_size=bootstrapping_size, bootstrapping_seed=7*random_state+1)
    return tail_probability_estimates


def _estimate_tail_probability_ellipsodial_constraint(
        data_module,
        percentage_lhs: float, percentage_rhs: float,
        data_size: int, threshold_percentage: typing.Union[float, typing.List[float]],
        g_ellipsoidal_dimension: int,
        alpha: float,
        random_state: int,
        bootstrapping_size: int) -> typing.List[float]:
    """
    Estimate tail probability using ellipsoidal constraint optimization.

    This function estimates the tail probability of a given distribution using ellipsoidal
    constraint optimization for three different derivative constraints: D=0 (no shape constraint),
    D=1 (monotone decreasing in the tail region), and D=2 (convex in the tail region).

    Parameters:
    -----------
    data_module : module
        Module containing the data generation and quantile functions.
    percentage_lhs : float
        Left-hand side percentage for quantile calculation.
    percentage_rhs : float
        Right-hand side percentage for quantile calculation.
    data_size : int
        Size of the synthetic data to generate.
    threshold_percentage : float or List[float]
        Threshold percentage(s) for optimization.
    g_ellipsoidal_dimension : int
        Dimension of the ellipsoidal constraint.
    alpha : float
        Significance level for the optimization.
    random_state : int
        Random seed for reproducibility.
    bootstrapping_size: int
        Size of the bootstrap samples.
    Returns:
    --------
    List[float]
        A list containing three tail probability estimates, one for each derivative constraint:
        - D=0: No shape constraint on the tail probability function.
        - D=1: Monotone decreasing constraint in the tail region.
        - D=2: Convex constraint in the tail region.

    Notes:
    ------
    The D parameter in the optimization function represents the order of the derivative constraint:
    - D=0: No additional constraint, allowing for flexible estimation.
    - D=1: Enforces monotone decreasing behavior in the tail, which is often a realistic assumption.
    - D=2: Enforces convexity in the tail, which can provide more stable estimates for heavy-tailed distributions.
    These constraints help in producing more realistic and stable estimates of the tail probability.
    """
    left_end_point_objective = data_utils.get_quantile(
        data_module, percentage_lhs, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
    right_end_point_objective = data_utils.get_quantile(
        data_module, percentage_rhs, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
    input_data = data_utils.generate_synthetic_data(
        data_module, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name], data_size, random_state)
    tail_probability_estimates = [0]*3
    tail_probability_estimates[0] = droevt_routine.optimization_with_ellipsodial_constraint(D=0,
                                                                                            input_data=input_data,
                                                                                            threshold_percentage=threshold_percentage,
                                                                                            alpha=alpha,
                                                                                            left_end_point_objective=left_end_point_objective, 
                                                                                            right_end_point_objective=right_end_point_objective,
                                                                                            g_ellipsoidal_dimension=g_ellipsoidal_dimension,
                                                                                            bootstrapping_size=bootstrapping_size, bootstrapping_seed=7*random_state+1)

    tail_probability_estimates[1] = droevt_routine.optimization_with_ellipsodial_constraint(D=1,
                                                                                            input_data=input_data,
                                                                                            threshold_percentage=threshold_percentage,
                                                                                            alpha=alpha,
                                                                                            left_end_point_objective=left_end_point_objective, 
                                                                                            right_end_point_objective=right_end_point_objective,
                                                                                            g_ellipsoidal_dimension=g_ellipsoidal_dimension,
                                                                                            bootstrapping_size=bootstrapping_size, bootstrapping_seed=7*random_state+1)

    tail_probability_estimates[2] = droevt_routine.optimization_with_ellipsodial_constraint(D=2,
                                                                                            input_data=input_data,
                                                                                            threshold_percentage=threshold_percentage,
                                                                                            alpha=alpha,
                                                                                            left_end_point_objective=left_end_point_objective, 
                                                                                            right_end_point_objective=right_end_point_objective,
                                                                                            g_ellipsoidal_dimension=g_ellipsoidal_dimension,
                                                                                            bootstrapping_size=bootstrapping_size, bootstrapping_seed=7*random_state+1)
    return tail_probability_estimates

def estimate_tail_probability_with_data_module(
        data_module,
        percentage_lhs: float, percentage_rhs: float,
        data_size: int, threshold_percentage: typing.Union[float, typing.List[float]],
        g_ellipsoidal_dimension: int,
        alpha: float,
        random_state: int,
        bootstrapping_size: int) -> typing.List[float]:
    """
    Estimate tail probabilities using both rectangular and ellipsoidal constraints.

    Parameters
    ----------
    data_module : module
        Module containing the data generation and utility functions.
    percentage_lhs : float
        Left percentile for defining the objective function interval.
    percentage_rhs : float
        Right percentile for defining the objective function interval.
    data_size : int
        Number of samples to generate.
    threshold_percentage : float or List[float]
        Single threshold percentage or list of threshold percentages.
    g_ellipsoidal_dimension : int
        Dimension parameter for ellipsoidal constraint.
    alpha : float
        Significance level (confidence level = 1-alpha).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    List[float]
        Combined list of tail probability estimates from both rectangular and 
        ellipsoidal constraints. The first three values are from rectangular
        constraints (D=0,1,2) and the next three from ellipsoidal constraints
        (D=0,1,2).
    """
    tail_probability_estimates = _estimate_tail_probability_rectangular_constraint(
        data_module=data_module,
        percentage_lhs=percentage_lhs, 
        percentage_rhs=percentage_rhs,
        data_size=data_size, 
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        random_state=random_state,
        bootstrapping_size=bootstrapping_size)
    tail_probability_estimates.extend(
        _estimate_tail_probability_ellipsodial_constraint(
            data_module=data_module,
            percentage_lhs=percentage_lhs, 
            percentage_rhs=percentage_rhs,
            data_size=data_size, 
            threshold_percentage=threshold_percentage,
            g_ellipsoidal_dimension=g_ellipsoidal_dimension,
            alpha=alpha,
            random_state=random_state,
            bootstrapping_size=bootstrapping_size))
    return tail_probability_estimates

def estimate_tail_probability(input_data: np.ndarray, 
                              left_end_point_objective: float, right_end_point_objective: float,
                              threshold_percentage: typing.Union[float, typing.List[float]],
                              g_ellipsoidal_dimension: int,
                              alpha: float,
                              random_state: int,
                              bootstrapping_size: int) -> typing.List[float]:
    """
    Estimate tail probabilities using both rectangular and ellipsoidal constraints.

    This function calculates tail probability estimates for given input data using
    both rectangular and ellipsoidal constraint methods. It computes estimates
    for three different values of D (0, 1, 2) for each constraint type, where:
    - D=0: No shape constraint on the tail probability function.
    - D=1: Monotone decreasing constraint in the tail region.
    - D=2: Convex constraint in the tail region.

    Parameters
    ----------
    input_data : np.ndarray
        The input data array for which tail probabilities are to be estimated.
    left_end_point_objective : float
        Left end point of the objective function interval.
    right_end_point_objective : float
        Right end point of the objective function interval.
    threshold_percentage : float or List[float]
        Single threshold percentage or list of threshold percentages.
    g_ellipsoidal_dimension : int
        Dimension parameter for ellipsoidal constraint.
    alpha : float
        Significance level (confidence level = 1-alpha).
    random_state : int
        Random seed for reproducibility.
    bootstrapping_size: int
        Size of the bootstrap samples.

    Returns
    -------
    List[float]
        A list of six tail probability estimates. The first three values are from
        rectangular constraints (D=0,1,2) and the next three from ellipsoidal
        constraints (D=0,1,2).

    Notes
    -----
    The function uses the droevt_routine module for optimization calculations.
    It applies both rectangular and ellipsoidal constraints with varying D values.
    """
    tail_probability_estimates = [0]*6
    # rectangular_constraint
    tail_probability_estimates[0] = droevt_routine.optimization_with_rectangular_constraint(
        D=0,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1)
    tail_probability_estimates[1] = droevt_routine.optimization_with_rectangular_constraint(
        D=1,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1)
    tail_probability_estimates[2] = droevt_routine.optimization_with_rectangular_constraint(
        D=2,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1)
    # ellipsoidal_constraint
    tail_probability_estimates[3] = droevt_routine.optimization_with_ellipsodial_constraint(
        D=0,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        g_ellipsoidal_dimension=g_ellipsoidal_dimension,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1)

    tail_probability_estimates[4] = droevt_routine.optimization_with_ellipsodial_constraint(
        D=1,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        g_ellipsoidal_dimension=g_ellipsoidal_dimension,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1)

    tail_probability_estimates[5] = droevt_routine.optimization_with_ellipsodial_constraint(
        D=2,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        g_ellipsoidal_dimension=g_ellipsoidal_dimension,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1)
    
    return tail_probability_estimates

def estimate_tail_probability_D2_chi2_only(input_data: np.ndarray, 
                                          left_end_point_objective: float, right_end_point_objective: float,
                                          threshold_percentage: typing.Union[float, typing.List[float]],
                                          g_ellipsoidal_dimension: int,
                                          alpha: float,
                                          random_state: int,
                                          bootstrapping_size: int,
                                          right_endpoint: float) -> typing.List[float]:
    """
    Estimate tail probabilities using both rectangular and ellipsoidal constraints.

    This function calculates tail probability estimates for given input data using
    both rectangular and ellipsoidal constraint methods. It computes estimates
    for two different values of D (2) for each constraint type, where:
    - D=2: Convex constraint in the tail region.

    Parameters
    ----------
    input_data : np.ndarray
        The input data array for which tail probabilities are to be estimated.
    left_end_point_objective : float
        Left end point of the objective function interval.
    right_end_point_objective : float
        Right end point of the objective function interval.
    threshold_percentage : float or List[float]
        Single threshold percentage or list of threshold percentages.
    g_ellipsoidal_dimension : int
        Dimension parameter for ellipsoidal constraint.
    alpha : float
        Significance level (confidence level = 1-alpha).
    random_state : int
        Random seed for reproducibility.
    bootstrapping_size: int
        Size of the bootstrap samples.
    right_endpoint: float
        Right end point of probability distribution.
    Returns
    -------
    List[List[float]]
        A list of two lists, each containing two tail probability estimates that form
        an estimated interval. The first list contains estimates from rectangular 
        constraints (D=2) and the second list contains estimates from ellipsoidal
        constraints (D=2). Each inner list contains [lower_bound, upper_bound] for
        the estimated interval.

    Notes
    -----
    The function uses the droevt_routine module for optimization calculations.
    It applies both rectangular and ellipsoidal constraints with varying D values.
    """
    # ellipsoidal_constraint
    tail_probability_estimates = [droevt_routine.optimization_with_ellipsodial_constraint(
        D=2,
        input_data=input_data,
        threshold_percentage=threshold_percentage,
        alpha=alpha,
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective,
        g_ellipsoidal_dimension=g_ellipsoidal_dimension,
        bootstrapping_size=bootstrapping_size, 
        bootstrapping_seed=7*random_state+1, is_max=is_max, right_endpoint=right_endpoint) for is_max in [False, True]]
    
    return tail_probability_estimates

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    data_size = 500
    true_value = 0.005
    percentage_lhs = 0.9
    percentage_rhs = percentage_lhs + true_value
    threshold_percentage = 0.7
    alpha = 0.05
    random_state = 20220222
    g_ellipsoidal_dimension = 3
    logger.info("A small example on tail probability estimation--single threshold.")
    result = estimate_tail_probability_with_data_module(
        gamma, percentage_lhs, percentage_rhs, data_size, threshold_percentage, g_ellipsoidal_dimension, alpha, random_state)
    logger.info(f"{[f'{x:.2E}' for x in result]}")
    logger.info("A small example on tail probability estimation--multiple thresholds.")
    threshold_percentage = [0.65, 0.7, 0.75, 0.8]
    result = estimate_tail_probability_with_data_module(
        gamma, percentage_lhs, percentage_rhs, data_size, threshold_percentage, g_ellipsoidal_dimension, alpha, random_state)
    logger.info(f"{[f'{x:.2E}' for x in result]}")
