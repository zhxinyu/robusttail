"""
synthetic_data_generator.py

This module provides utility functions for generating synthetic data samples
from various probability distributions and calculating quantiles. It is designed
to support experiments and simulations that require controlled, reproducible
data generation from specific statistical distributions.

The module includes:
- Default parameters for common distributions (gamma, lognorm, pareto, genpareto)
- A function to generate synthetic data samples
- A function to calculate quantiles for given distributions and probabilities

This can be useful for testing statistical methods, benchmarking algorithms,
or creating controlled datasets for numerical experiments.
"""

import numpy as np
from scipy.stats import gamma, lognorm, pareto, genpareto

DISTRIBUTION_DEFAULT_PARAMETERS = {
    "normal_truncated": {
        "loc": 0,      # mean
        "scale": 1,     # standard deviation  
        "a": 0,
        "b": np.inf
    },
    "gamma": {
        "a": 0.5,      # shape parameter
        "scale": 1     # scale parameter
    },
    "lognorm": {
        "loc": 0,      # mean
        "s": 1         # standard deviation  
    },
    "pareto": {
        "b": 1.5,      # shape parameter
        "scale": 1     # scale parameter
    }, # Quantile points for 0.99: [3.317, 10.24, 21.54]
    "genpareto": {
        "c": -0.1,     # c < 0 gives bounded tail (Weibull). The right endpoint is 10.
    } 
}

def generate_synthetic_data(data_module: object, 
                            param_dict: dict, 
                            data_size: int, 
                            random_state: int) -> np.ndarray:
    """
    Generate synthetic data samples from a probability distribution.
    
    Parameters
    ----------
    data_module : object
        Statistical distribution object (e.g. scipy.stats distribution)
    param_dict : dict
        Dictionary of distribution parameters
    data_size : int
        Number of samples to generate  
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Array of randomly generated samples
    """
    return data_module.rvs(size=data_size,
                           random_state=random_state, 
                           **param_dict)

def get_quantile(distribution, probability: float, params: dict) -> float:
    """
    Generate endpoint quantiles for objective function indicator 1_{lhs<=x<=rhs}.
    
    Parameters
    ----------
    distribution : object
        Statistical distribution object (e.g. scipy.stats distribution)
    probability : float
        Probability level for quantile
    params : dict
        Dictionary of distribution parameters
        
    Returns
    -------
    float
        Quantile value at specified probability level
    """
    return distribution.ppf(q=probability, **params)
