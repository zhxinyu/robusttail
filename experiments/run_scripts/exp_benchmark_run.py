from multiprocessing import Pool
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd
import numpy as np
import os
import itertools
import pathlib
import tqdm
import logging
import typing
from typing import Union
import inspect
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import droevt.utils.synthetic_data_generator as data_utils
from tail_probability.benchmark_tail_probability_estimation import benchmark_estimate_tail_probability

string_to_data_module = {"gamma": gamma,
                         "lognorm": lognorm,
                         "pareto": pareto,
                         "genpareto": genpareto}

def _parallelRun(pool_param: tuple) ->  typing.List[float]:
    input_data = pool_param['input_data']
    left_end_point_objective = pool_param['left_end_point_objective']
    right_end_point_objective = pool_param['right_end_point_objective']
    method = pool_param['method']
    alpha = pool_param['alpha']

    ro_result = benchmark_estimate_tail_probability(
        input_data=input_data, 
        left_end_point_objective=left_end_point_objective, 
        right_end_point_objective=right_end_point_objective, 
        alpha=alpha,
        method=method)
        
    return ro_result

def benchmark_base_runner_tail_probability(
    exp_name: str,
    methods: list[str],
    data_distributions: list[str],
    data_sizes: list[int],
    percentage_lhs_values: list[Union[float, str]],
    true_value: Union[float, str],
    random_seed: int,
    n_experiment_repetitions: int) -> None:
    
    for data_distribution, data_size, percentage_lhs, method in tqdm.tqdm(itertools.product(*[data_distributions, 
                                                                                              data_sizes, 
                                                                                              percentage_lhs_values,
                                                                                              methods])):
        
        data_module = string_to_data_module[data_distribution]
        percentage_rhs = float(percentage_lhs) + float(true_value)
        left_end_point_objective = data_utils.get_quantile(
            data_module, float(percentage_lhs), data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
        right_end_point_objective = data_utils.get_quantile(
            data_module, float(percentage_rhs), data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
        pool_param_list = [{"input_data": data_utils.generate_synthetic_data(data_module, 
                                                                             data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_distribution], 
                                                                             data_size, random_seed+ nnrep),
                            "left_end_point_objective": left_end_point_objective,
                            "right_end_point_objective": right_end_point_objective,
                            "method": method,
                            "alpha": 0.05} 
                            for nnrep in range(n_experiment_repetitions)
        ]

        try:
            if False: # set to True to run in parallel
                with Pool() as p:
                    df = pd.DataFrame(np.asarray(p.map(_parallelRun, pool_param_list)),
                                        columns=["Lower Bound", "Upper Bound"])
                    # Right-align each column with width of 12 characters to ensure alignment between headers and values
                    logger.info(" ".join([f"{col:>12}" for col in df.columns]))
                    logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
                    print(df)
            else:
                for pool_param in pool_param_list:
                    result = _parallelRun(pool_param)
                    print(result)
        except Exception as e:
            logger.error("Exception: %s", e)
            
def benchmark_exp_tail_probability():
    # Single threshold percentage and multi-threshold percentages
    exp_name = inspect.currentframe().f_code.co_name
    logger.info("Experiment name: %s", exp_name)
    methods = ['pot', 'pot_bt', 'pl', 'bayesian', 'pwm']
    data_distributions = ['gamma', 'lognorm', 'pareto', 'genpareto']
    data_sizes = [500]
    percentage_lhs_values = ["0.9", "0.95", "0.99"]
    true_value = "0.005"
    n_experiment_repetitions = 200
    random_seed = 20220222
    benchmark_base_runner_tail_probability(exp_name=exp_name, 
                                           methods=methods,
                                           data_distributions=data_distributions, 
                                           data_sizes=data_sizes, 
                                           percentage_lhs_values=percentage_lhs_values, 
                                           true_value=true_value, 
                                           random_seed=random_seed, 
                                           n_experiment_repetitions=n_experiment_repetitions)

def benchmark_exp_tail_probability_real_data():
    # Single threshold percentage and multi-threshold percentages
    exp_name = inspect.currentframe().f_code.co_name
    logger.info("Experiment name: %s", exp_name)
    methods = ['pot', 'pot_bt', 'pl', 'bayesian', 'pwm']
    
    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]
    left_end_point_objectives = [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 
                                 7.6, 7.7, 7.8, 7.9, 8.0,
                                 7.25]
    right_end_point_objective = 1e9

    result = []
    pool_param_list = []
    for method in methods:
        for region in regions:
            input_data = np.loadtxt(os.path.join(pathlib.Path(__file__).parents[1], 
                                                "input_data", 
                                                "cmt", 
                                                "parsed_data", 
                                                region+".csv"))
            for left_end_point_objective in left_end_point_objectives:
                pool_param = {"input_data": input_data,
                              "left_end_point_objective": left_end_point_objective,
                              "right_end_point_objective": right_end_point_objective,
                              "method": method,
                              "alpha": 0.05}
                pool_param_list.append(pool_param)
    
    try:
        if False: # set to True to run in parallel
            with Pool() as p:
                ro_result_list = p.map(_parallelRun, pool_param_list)
                result = [[pool_param['method'], 
                        pool_param['region'], 
                        pool_param['left_end_point_objective'], 
                        ro_result[0], 
                        ro_result[1]] for (pool_param, ro_result) in zip(pool_param_list, ro_result_list)]            
            print(result)                            
        else:
            for pool_param in pool_param_list:
                ro_result = _parallelRun(pool_param)
                result.append([pool_param['method'], 
                               pool_param['region'], 
                               pool_param['left_end_point_objective'], 
                               ro_result[0], 
                               ro_result[1]])
                print(result[-1])
    except Exception as e:
        logger.error("Exception: %s", e)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run benchmark experiments for tail probability estimation.")
    parser.add_argument('--exp', choices=['synthetic', 'real'], 
                        help="Which experiment(s) to run: 'synthetic', 'real', or 'both' (default: both)")
    args = parser.parse_args()

    logger.info("Starting experiment of tail probability estimation")

    if args.exp == 'synthetic':
        benchmark_exp_tail_probability()
    elif args.exp == 'real':
        benchmark_exp_tail_probability_real_data()
    else:
        logger.error("Invalid experiment: %s", args.exp)
        exit(1)
