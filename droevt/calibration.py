"""
This module provides functions for calibration and statistical analysis.

It includes functions for generating parameters in optimization problems,
ellipsoidal specification using chi-square distribution, and rectangular
specification using Kolmogorov-Smirnov test.

The module uses rpy2 to interface with R for kernel density estimation.
"""

from typing import Union, List
import random
import numpy as np
from scipy.stats import norm, chi2, kstwobign
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError

numpy2ri.activate()
importr('base')
utils = importr('utils')
importr('stats')
try:
    importr('ks')
except RRuntimeError:
    utils.install_packages('ks', contriburl="https://cran.microsoft.com/")
    importr('ks')

def eta_generation(data: np.ndarray,
                   point_estimate: float,
                   bootstrapping_flag: bool,
                   bootstrapping_size: int,
                   bootstrapping_seed: int) -> Union[float, List[float]]:
    """
    Generate eta based on observations using kernel density estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    point_estimate : float
        Point at which to evaluate the density.
    bootstrapping_flag : bool
        Whether to use bootstrapping.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.

    Returns
    -------
    float or List[float]
        Estimated eta value(s).
    """
    # Generate eta based on observations.
    if not bootstrapping_flag:
        return ro.r('''
            data = c({:})
            (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
        '''.format(','.join([str(each_data) for each_data in data.tolist()]), point_estimate))[0]
    else:
        random.seed(bootstrapping_seed)
        output_list = []
        for _ in range(bootstrapping_size):
            bootstrapping_data = random.choices(data, k=bootstrapping_size)
            output_list.append(ro.r('''
                data = c({:})
                (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
            '''.format(','.join([str(each_data) for each_data in bootstrapping_data]), point_estimate)
            )[0])
        return output_list


def eta_specification(data: np.ndarray,
                      threshold: float,
                      alpha: float,
                      bootstrapping_size: int,
                      bootstrapping_seed: int,
                      D_riser_number: int,
                      num_multi_threshold: int) -> Union[float, np.ndarray]:
    """
    Specify eta values based on bootstrapped estimates.

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    threshold : float
        Threshold value.
    alpha : float
        Significance level.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.
    D_riser_number : int
        Number of D-risers (1 or 2).
    num_multi_threshold : int
        Number of multiple thresholds.

    Returns
    -------
    float or np.ndarray
        Specified eta value(s).
    """
    eta_bootstrapping = eta_generation(data,
                                       threshold,
                                       bootstrapping_flag=True,
                                       bootstrapping_size=bootstrapping_size,
                                       bootstrapping_seed=bootstrapping_seed)
    
    if D_riser_number not in [1, 2]:
        raise ValueError("D_riser_number must be 1 or 2")
    
    if D_riser_number == 1:
        quantile = 1 - alpha / (num_multi_threshold + 1)
        return np.quantile(a=eta_bootstrapping, q=quantile)
    else:
        quantiles = [alpha / (4 * num_multi_threshold + 2), 1 - alpha / (4 * num_multi_threshold + 2)]
        return np.quantile(a=eta_bootstrapping, q=quantiles)

def nu_generation(data: np.ndarray,
                  point_estimate: float,
                  bootstrapping: bool,
                  bootstrap_size: int,
                  bootstrap_seed: int) -> Union[float, List[float]]:
    """
    Generate nu based on observations using kernel density derivative estimation.

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    point_estimate : float
        Point at which to evaluate the density derivative.
    bootstrapping : bool
        Whether to use bootstrapping.
    bootstrap_size : int
        Size of bootstrap samples.
    bootstrap_seed : int
        Seed for random number generation in bootstrapping.

    Returns
    -------
    float or List[float]
        Estimated nu value(s).
    """
    if not bootstrapping:
        return _estimate_nu(data, point_estimate)
    else:
        np.random.seed(bootstrap_seed)
        return [_estimate_nu(np.random.choice(data, size=bootstrap_size, replace=True), point_estimate) 
                for _ in range(bootstrap_size)]

def _estimate_nu(data: np.ndarray, point: float) -> float:
    """Helper function to estimate nu using R's kdde function."""
    data_str = ','.join(map(str, data))
    return ro.r(f'''
        data <- c({data_str})
        (kdde(x = data, deriv.order = 1, eval.points = {point}))$estimate
    ''')[0]

def nu_specification(data: np.ndarray,
                     threshold: float,
                     alpha: float,
                     bootstrapping_size: int,
                     bootstrapping_seed: int,
                     num_multi_threshold: int) -> float:
    """
    Specify nu value based on bootstrapped estimates.

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    threshold : float
        Threshold value.
    alpha : float
        Significance level.
    bootstrapping_size : int
        Size of bootstrap samples.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.
    num_multi_threshold : int
        Number of multiple thresholds.

    Returns
    -------
    float
        Specified nu value.
    """
    quantile = alpha / (2 * num_multi_threshold + 1)
    nu_bootstrapping = nu_generation(data,
                                     threshold,
                                     bootstrapping=True,
                                     bootstrap_size=bootstrapping_size,
                                     bootstrap_seed=bootstrapping_seed)
    return -np.quantile(a=nu_bootstrapping, q=quantile)


############################################################################################################
############################################################################################################
# Ellipsoidal Specification (chi square)
############################################################################################################
############################################################################################################

def z_of_chi_square(alpha: float,
                    D_riser_number: int,
                    g_dimension: int,
                    num_multi_threshold: int) -> float:
    """
    Generate z value under chi-square distribution.

    Parameters
    ----------
    alpha : float
        Significance level.
    D_riser_number : int
        Number of D-risers (0, 1, or 2).
    g_dimension : int
        Degrees of freedom for chi-square distribution.
    num_multi_threshold : int
        Number of multiple thresholds.

    Returns
    -------
    float
        z value under chi-square distribution.
    """
    assert D_riser_number in [0, 1, 2], "D_riser_number must be 0, 1, or 2"
    
    if D_riser_number == 1:
        quantile = 1 - alpha / (num_multi_threshold + 1)
    elif D_riser_number == 2:
        quantile = 1 - alpha / (2 * num_multi_threshold + 1)
    else:  # D_riser_number == 0
        quantile = 1 - alpha / num_multi_threshold
    
    return chi2.ppf(q=quantile, df=g_dimension)

############################################################################################################
############################################################################################################
# Rectangular Specification (Kolmogorov–Smirnov test)
############################################################################################################
############################################################################################################

def z_of_kolmogorov(alpha: float,
                    D_riser_number: int,
                    num_multi_threshold: int) -> float:
    """
    Generate z value under Kolmogorov distribution.

    Parameters
    ----------
    alpha : float
        Significance level.
    D_riser_number : int
        Number of D-risers (0, 1, or 2).
    num_multi_threshold : int
        Number of multiple thresholds.

    Returns
    -------
    float
        z value under Kolmogorov distribution.
    """
    assert D_riser_number in [0, 1, 2], "D_riser_number must be 0, 1, or 2"
    
    if D_riser_number == 1:
        quantile = 1 - alpha / (num_multi_threshold + 1)
    elif D_riser_number == 2:
        quantile = 1 - alpha / (2 * num_multi_threshold + 1)
    else:  # D_riser_number == 0
        quantile = 1 - alpha / num_multi_threshold
    
    return kstwobign.ppf(q=quantile)


if __name__ == '__main__':
    pass
