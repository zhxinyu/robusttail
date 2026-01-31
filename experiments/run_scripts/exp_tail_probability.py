from multiprocessing import Pool
import tail_probability.tail_probability_estimation as tpe
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

def base_meta_data_dict():
    
    meta_data_dict = {"data_size": 500,
                      "percentage_lhs": 0.99,
                      "percentage_rhs": 0.995,
                      "threshold_percentage": 0.7,
                      "alpha": 0.05,
                      "g_ellipsoidal_dimension": 3}

    string_to_data_module = {"gamma": gamma,
                            "lognorm": lognorm,
                            "pareto": pareto,
                            "genpareto": genpareto}

    return meta_data_dict, string_to_data_module


def _parallelRun(pool_param: tuple) ->  typing.List[float]:
    _, string_to_data_module = base_meta_data_dict()

    _data_distribution, _meta_data_dict, _random_state = pool_param
    _meta_data_dict = _meta_data_dict.copy()
    _meta_data_dict["random_state"] = _random_state
    return tpe.estimate_tail_probability_with_data_module(
        string_to_data_module[_data_distribution], **_meta_data_dict)

def base_runner_tail_probability(
    exp_name: str,
    data_distributions: list[str],
    data_sizes: list[int],
    percentage_lhs_values: list[Union[float, str]],
    threshold_percentages: list[Union[float, str]] | list[list[Union[float, str]]],
    true_value: Union[float, str],
    random_seed: int,
    n_experiment_repetitions: int,
    bootstrapping_size: int | None = None
) -> None:
    """
    Run tail probability estimation experiments for various configurations.

    This function performs tail probability estimation experiments for different
    data distributions, sizes, and threshold values.

    Parameters:
    -----------
    exp_name : str
        Name of the experiment, used for creating the output directory.
    data_distributions : list[str]
        List of data distribution names to be used in the experiments.
    data_sizes : list[int]
        List of data sizes to be used in the experiments.
    percentage_lhs_values : list[Union[float, str]]
        List of left-hand side percentage values for quantile calculation.
    threshold_percentages : list[Union[float, str]]
        List of threshold percentages for optimization.
    true_value : Union[float, str]
        True value to be added to percentage_lhs for right-hand side calculation.
    random_seed : int
        Seed for random number generation to ensure reproducibility.
    n_experiment_repetitions : int
        Number of times to repeat each experiment configuration.
    bootstrapping_size: int, optional. Default: None
        Size of the bootstrap samples.

    Returns:
    --------
    None
    """

    meta_data_dict, _ = base_meta_data_dict()

    for data_distribution, data_size, percentage_lhs, threshold_percentage in tqdm.tqdm(itertools.product(*[data_distributions, 
                                                                                                            data_sizes, 
                                                                                                            percentage_lhs_values, 
                                                                                                            threshold_percentages])):
        meta_data_dict["data_size"] = data_size
        meta_data_dict["percentage_lhs"] = float(percentage_lhs)
        meta_data_dict["percentage_rhs"] = float(percentage_lhs) + float(true_value)
        meta_data_dict["bootstrapping_size"] = data_size if bootstrapping_size is None else bootstrapping_size
        if isinstance(threshold_percentage, list):
            meta_data_dict["threshold_percentage"] = [float(threshold_percentage_i) for threshold_percentage_i in threshold_percentage]
        else:
            meta_data_dict["threshold_percentage"] = float(threshold_percentage)
        assert "random_state" not in meta_data_dict
        pool_param_list = [(data_distribution, meta_data_dict, random_state + random_seed)
                         for random_state in range(n_experiment_repetitions)]
        logger.info(f"Running experiment {exp_name} with parameters:")

        try:
            if False: # set to True to run in parallel
                with Pool() as p:
                    df = pd.DataFrame(np.asarray(p.map(_parallelRun, pool_param_list)),
                                        columns=["(0,KS)", "(1,KS)", "(2,KS)",
                                                "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"])
                    # Right-align each column with width of 12 characters to ensure alignment between headers and values
                    logger.info(" ".join([f"{col:>12}" for col in df.columns]))
                    logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
            else:
                for pool in pool_param_list:
                    result = _parallelRun(pool)
                    logger.info(" ".join([f"{val:>12.2E}" for val in result]))
        except Exception as e:
            logger.error("Exception: %s", e)

def exp_tail_probability_quick_run():
    exp_name = inspect.currentframe().f_code.co_name # type: ignore
    logger.info("Experiment name: %s", exp_name)
    data_distributions = ['gamma', 'lognorm', 'pareto', 'genpareto']
    data_sizes = [500]
    # Single threshold percentage and multi-threshold percentages
    threshold_percentages = [0.7]
    percentage_lhs_values = [0.99]
    true_value = 0.005
    n_experiment_repetitions = 1
    random_seed = 20220222
    base_runner_tail_probability(exp_name=exp_name,
                data_distributions=data_distributions, 
                data_sizes=data_sizes, 
                percentage_lhs_values=percentage_lhs_values, 
                threshold_percentages=threshold_percentages, 
                true_value=true_value, 
                random_seed=random_seed, 
                n_experiment_repetitions=n_experiment_repetitions)

def exp_tail_probability_thresholds():
    # Single threshold percentage and multi-threshold percentages
    exp_name = inspect.currentframe().f_code.co_name # type: ignore
    logger.info("Experiment name: %s", exp_name)
    data_distributions = ['gamma', 'lognorm', 'pareto', 'genpareto']
    data_sizes = [500]
    # Single threshold percentage and multi-threshold percentages
    threshold_percentages = [0.6, 0.7, 0.8, 0.9, [0.6, 0.7, 0.8, 0.9]]
    percentage_lhs_values = [0.99]
    true_value = "0.005"
    n_experiment_repetitions = 200
    random_seed = 20220222
    base_runner_tail_probability(exp_name=exp_name,
                data_distributions=data_distributions, 
                data_sizes=data_sizes, 
                percentage_lhs_values=percentage_lhs_values, 
                threshold_percentages=threshold_percentages, 
                true_value=true_value, 
                random_seed=random_seed, 
                n_experiment_repetitions=n_experiment_repetitions)

def exp_tail_probability_percentage_lhs():
    # Single threshold percentage and multi-threshold percentages
    exp_name = inspect.currentframe().f_code.co_name
    logger.info("Experiment name: %s", exp_name)
    data_distributions = ['gamma', 'lognorm', 'pareto', 'genpareto']
    data_sizes = [500]
    # Single threshold percentage and multi-threshold percentages
    threshold_percentages = [0.7]
    percentage_lhs_values = np.linspace(0.9, 0.99, 10).tolist()
    true_value = "0.005"
    n_experiment_repetitions = 200
    random_seed = 20220222
    base_runner_tail_probability(exp_name=exp_name,
                data_distributions=data_distributions, 
                data_sizes=data_sizes, 
                percentage_lhs_values=percentage_lhs_values, 
                threshold_percentages=threshold_percentages, 
                true_value=true_value, 
                random_seed=random_seed, 
                n_experiment_repetitions=n_experiment_repetitions)

def exp_tail_probability_scarce_data():
    # Experiment on scarce data
    exp_name = inspect.currentframe().f_code.co_name
    logger.info("Experiment name: %s", exp_name)
    data_distributions = ['gamma', 'lognorm', 'pareto', 'genpareto']
    data_sizes = [30]
    threshold_percentages = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    percentage_lhs_values = [0.9]
    true_value = "0.005"
    n_experiment_repetitions = 200
    random_seed = 20220222
    bootstrapping_size=500
    base_runner_tail_probability(exp_name=exp_name,
                data_distributions=data_distributions, 
                data_sizes=data_sizes, 
                percentage_lhs_values=percentage_lhs_values, 
                threshold_percentages=threshold_percentages, 
                true_value=true_value, 
                random_seed=random_seed, 
                n_experiment_repetitions=n_experiment_repetitions,
                bootstrapping_size=bootstrapping_size)

def exp_tail_probability_real_data():
    # Experiment on real data

    exp_name = inspect.currentframe().f_code.co_name

    meta_data_dict = {"alpha": 0.05,
                      "g_ellipsoidal_dimension": 3}

    logger.info("Experiment name: %s", exp_name)
    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]
    threshold_percentages = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

    bootstrapping_size=500
    random_seed = 20220222

    left_end_point_objectives = [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 
                                 7.6, 7.7, 7.8, 7.9, 8.0,
                                 7.25]

    meta_data_dict["right_end_point_objective"] = np.inf
    meta_data_dict["bootstrapping_size"] = bootstrapping_size

    for region in regions:
        inputData = np.loadtxt(os.path.join(pathlib.Path(__file__).parents[1], 
                                            "input_data", 
                                            "cmt", 
                                            "parsed_data", 
                                            region+".csv"))

        for threshold_percentage in threshold_percentages:
            meta_data_dict["threshold_percentage"] = threshold_percentage
            meta_data_dict["random_state"] = random_seed
            for left_end_point_objective in left_end_point_objectives:
                meta_data_dict["left_end_point_objective"] = left_end_point_objective
                try:                         
                    df = pd.DataFrame([tpe.estimate_tail_probability(input_data=inputData, **meta_data_dict)],
                                       columns=["(0,KS)", "(1,KS)", "(2,KS)",
                                                "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"])

                    logger.info(" ".join([f"{col:>12}" for col in df.columns]))
                    logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
                except Exception as e:
                    logger.error("Exception: %s", e)

import argparse

def main():
    logger.info("Starting experiment of tail probability estimation")
    parser = argparse.ArgumentParser(description="Tail Probability Estimation Experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="quick_run",
        choices=[
            "quick_run",
            "thresholds", 
            "percentage_lhs", 
            "scarce_data", 
            "real_data", 
        ],
        help="Which experiment(s) to run"
    )
    args = parser.parse_args()

    if args.experiment == "quick_run":
        exp_tail_probability_quick_run()
    elif args.experiment == "thresholds":
        exp_tail_probability_thresholds()
    elif args.experiment == "percentage_lhs":
        exp_tail_probability_percentage_lhs()
    elif args.experiment == "scarce_data":
        exp_tail_probability_scarce_data()
    elif args.experiment == "real_data":
        exp_tail_probability_real_data()
    else:
        logger.error("Invalid experiment: %s", args.experiment)
        exit(1)
        
if __name__ == '__main__':
    main()