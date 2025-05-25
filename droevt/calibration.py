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
from scipy.stats import chi2, kstwobign
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
# from rpy2.rinterface_lib.embedded import RRuntimeError

numpy2ri.activate()
# Import R packages
importr('base')
importr('utils')
importr('stats')
importr('ks')

def eta_generation(data: np.ndarray,
                   point_estimate: float,
                   bootstrapping_flag: bool,
                   bootstrapping_size: int,
                   bootstrapping_seed: int) -> Union[float, List[float]]:
    """
    Generate eta, which represents the kernel density estimate of the probability density function.
    
    This function uses kernel density estimation to approximate the probability density function
    of the input data at a given point. The eta value represents the height/value of the 
    estimated density function at the specified point_estimate.

    Parameters
    ----------
    data : np.ndarray
        Input data for density estimation.
    point_estimate : float
        Point at which to evaluate the probability density.
    bootstrapping_flag : bool
        Whether to use bootstrapping to get multiple density estimates.
    bootstrapping_size : int
        Size of bootstrap samples if bootstrapping is used.
    bootstrapping_seed : int
        Seed for random number generation in bootstrapping.

    Returns
    -------
    float or List[float]
        If bootstrapping_flag is False, returns a single density estimate at point_estimate.
        If bootstrapping_flag is True, returns a list of density estimates from bootstrap samples.
    """
    # Generate eta based on observations.
    if not bootstrapping_flag:
        np_float = ro.r('''
            data = c({:})
            (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
        '''.format(','.join([str(each_data) for each_data in data.tolist()]), point_estimate))[0]
        return float(np_float)
    else:
        # Bootstrapping is used to estimate the sampling distribution of the kernel density estimate
        # by repeatedly:
        # 1. Resampling with replacement from the original data (creating bootstrap samples)
        # 2. Computing the kernel density estimate for each bootstrap sample
        # This gives us a distribution of density estimates that helps quantify uncertainty
        
        # Set random seed for reproducibility
        random.seed(bootstrapping_seed)
        output_list = []
        
        for _ in range(bootstrapping_size):
            bootstrapping_data = np.random.choice(data, size=data.shape[0], replace=True)
            output_list.append(float(ro.r('''
                data = c({:})
                (kdde(x = data, deriv.order = 0, eval.points = {:}))$estimate
            '''.format(','.join([str(each_data) for each_data in bootstrapping_data]), point_estimate)
            )[0]))
            
        # Return list of density estimates from all bootstrap samples
        # This distribution can be used to:
        # - Estimate confidence intervals for the true density
        # - Assess variability in the density estimate
        # - Calculate standard errors
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

    This function uses eta_generation() with bootstrapping enabled to generate multiple density 
    estimates, then computes appropriate quantiles based on the significance level alpha and 
    number of D-risers. While eta_generation() provides the raw density estimates, 
    eta_specification() determines the critical values needed for statistical inference.

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
        Specified eta value(s) representing critical values derived from the bootstrapped
        density estimates. Returns a single value for D_riser_number=1 or two values for
        D_riser_number=2.
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

def negative_nu_generation(data: np.ndarray,
                  point_estimate: float,
                  bootstrapping: bool,
                  bootstrap_size: int,
                  bootstrap_seed: int) -> Union[float, List[float]]:
    """
    Generate negative nu (the first derivative of the kernel density estimate) based on observations 
    using kernel density derivative estimation. This estimates how quickly the density is 
    changing at a given point.

    Parameters
    ----------
    data : np.ndarray
        Input data for estimation.
    point_estimate : float
        Point at which to evaluate the first derivative of the kernel density estimate.
    bootstrapping : bool
        Whether to use bootstrapping.
    bootstrap_size : int
        Size of bootstrap samples.
    bootstrap_seed : int
        Seed for random number generation in bootstrapping.

    Returns
    -------
    float or List[float]
        Estimated value(s) of negative nu (the first derivative of the kernel density estimate).
        Returns a single float when bootstrapping=False, or a list of floats when 
        bootstrapping=True containing bootstrap replicates.
    """
    if not bootstrapping:
        return _estimate_negative_nu(data, point_estimate)
    else:
        np.random.seed(bootstrap_seed)
        return [_estimate_negative_nu(np.random.choice(data, size=data.shape[0], replace=True), point_estimate) 
                for _ in range(bootstrap_size)]

def _estimate_negative_nu(data: np.ndarray, point: float) -> float:
    """
    Helper function to estimate negative nu (the first derivative of the kernel density estimate) using R's kdde function.
    
    Negative nu represents the rate of change of the probability density at a given point. A positive nu means the density
    is increasing, while a negative nu means it's decreasing. The magnitude indicates how rapidly the density changes.
    
    This is calculated using kernel density derivative estimation (kdde) from R, which estimates the first derivative 
    of the kernel density function at the specified point.
    """
    data_str = ','.join(map(str, data))
    return float(ro.r(f'''
        data <- c({data_str})
        (kdde(x = data, deriv.order = 1, eval.points = {point}))$estimate
    ''')[0])

def nu_specification(data: np.ndarray,
                     threshold: float,
                     alpha: float,
                     bootstrapping_size: int,
                     bootstrapping_seed: int,
                     num_multi_threshold: int) -> float:
    """
    Specify nu (the negative first derivative of the kernel density estimate) based on bootstrapped estimates.

    While negative_nu_generation() calculates the first derivative of the kernel density estimate 
    (either a single value or bootstrap replicates), this function uses those bootstrap 
    replicates to determine a specification bound. Specifically, it:

    1. Generates multiple nu estimates through bootstrapping using nu_generation()
    2. Calculates a quantile based on the significance level alpha and number of thresholds
    3. Returns the negative quantile of the bootstrap estimates as the specification bound

    This specification bound can be used to make statistical inferences about the derivative
    of the density function, accounting for multiple testing through num_multi_threshold.

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
        The specification bound computed as the negative quantile of bootstrapped nu estimates.
    """
    quantile = alpha / (2 * num_multi_threshold + 1)
    negative_nu_bootstrapping = negative_nu_generation(data,
                                                       threshold,
                                                       bootstrapping=True,
                                                       bootstrap_size=bootstrapping_size,
                                                       bootstrap_seed=bootstrapping_seed)
    return -np.quantile(a=negative_nu_bootstrapping, q=quantile)


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

    The z value represents the critical value or threshold for the chi-square distribution,
    used to determine the confidence region for statistical inference. This confidence
    region has a level of 1 - alpha, adjusted for multiple testing.

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
        z value under chi-square distribution, which is the critical value
        used to define the confidence region of level 1 - alpha (adjusted for multiple testing).
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

    The z value represents the critical value or threshold for the Kolmogorov distribution,
    used to determine the confidence region for statistical inference. This confidence
    region has a level of 1 - alpha, adjusted for multiple testing.

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
        z value under Kolmogorov distribution, which is the critical value
        used to define the confidence region of level 1 - alpha (adjusted for multiple testing).
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
