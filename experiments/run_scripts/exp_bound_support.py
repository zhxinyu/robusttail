import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import genpareto

import sys
import os
import tempfile
import shutil
os.environ['R_HOME'] = sys.executable.replace('bin/python', 'lib/R')
from tail_probability.tail_probability_estimation import estimate_tail_probability_D2_chi2_only


def get_data():
    input_data = genpareto.rvs(size=500, c=-0.01, loc=0, scale=1)
    return input_data

def _parallel_run(pool_param: tuple):
    input_data, lhs, right_end_point_objective, kwargs = pool_param
    results = estimate_tail_probability_D2_chi2_only(input_data=input_data, 
                                                     left_end_point_objective=lhs, 
                                                     right_end_point_objective=right_end_point_objective, 
                                                     **kwargs)
    return results

if __name__ == "__main__":
    input_data = get_data()

    alpha=0.05
    right_end_point_objective=np.inf
    g_ellipsoidal_dimension=3
    threshold_percentage=0.7
    random_state=20220222
    bootstrapping_size=500
    kwargs = {
        "threshold_percentage": threshold_percentage,
        "g_ellipsoidal_dimension": g_ellipsoidal_dimension,
        "alpha": alpha,
        "random_state": random_state,
        "bootstrapping_size": bootstrapping_size,
    }

    np.random.seed(random_state)
    quantiles = ["0.95", "0.99", "0.995"]
    right_end_points = [1.90, 1.91, 1.93, 1.95, 1.98, 2, 2.01, 2.03, 2.05, 2.08, 2.10, np.inf]
    c = - 0.5
    for quantile in quantiles:
        raw_results = []
        for right_endpoint in right_end_points:
            lhs = genpareto.ppf(q=float(quantile),  c=c, loc=0, scale=1)
            pool_param_list = [(genpareto.rvs(size=500, c=c, loc=0, scale=1), lhs, right_end_point_objective, kwargs |
                                {"right_endpoint": right_endpoint}) for _ in range(200)]
            with Pool() as pool:
                bootstrap_results = pool.map(_parallel_run, pool_param_list)
            print(bootstrap_results)

