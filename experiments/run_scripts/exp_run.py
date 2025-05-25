from multiprocessing import Pool
import tail_probability.tail_probability_estimation as tpe
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd
import numpy as np
import os
import itertools
import sys
import traceback
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

def _parallel_run_data_module(pool_param: tuple) ->  typing.List[float]:
    _, string_to_data_module = base_meta_data_dict()

    _data_distribution, _meta_data_dict, _random_state = pool_param
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
    bootstrapping_size: int = None
) -> None:
    """
    Run tail probability estimation experiments for various configurations.

    This function performs tail probability estimation experiments for different
    data distributions, sizes, and threshold values. It saves the results to CSV files.

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
        Results are saved to CSV files in the specified output directory.
    """
        
    # generate a folder to store the results
    FOLDER_DIR = os.path.join(pathlib.Path(__file__).parents[1], "raw_output", exp_name)
    if not os.path.isdir(FOLDER_DIR):
       os.mkdir(FOLDER_DIR)

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
        FILE_NAME = ["tail_probability_estimation"]
        FILE_NAME += ["data_distribution="+data_distribution]
        FILE_NAME += [key+"="+str(meta_data_dict[key]) for key in meta_data_dict if key != "bootstrapping_size"]
        FILE_NAME += ["random_seed="+str(random_seed)]
        FILE_NAME += ["n_experiment_repetitions="+str(n_experiment_repetitions)]
        FILE_NAME = '_'.join(FILE_NAME)+".csv"
        parameter_pairs = FILE_NAME.removeprefix('tail_probability_estimation_').removesuffix('.csv').split('=')
        names = [x.split('_', maxsplit=1)[1] if idx !=0 else x for idx, x in enumerate(parameter_pairs[:-1]) ]
        values = [x.split('_', maxsplit=1)[0] for idx, x in enumerate(parameter_pairs[1:]) ]
        logger.info(f"Running experiment {exp_name} with parameters:")
        # Using blue (94) for parameter names and green (92) for values
        logger.info(" ".join([f"\033[94m{name:>{max(len(name), len(value))}}\033[0m" for name, value in zip(names, values)]))
        logger.info(" ".join([f"\033[92m{value:>{max(len(name), len(value))}}\033[0m" for name, value in zip(names, values)]))

        if os.path.exists(os.path.join(FOLDER_DIR, FILE_NAME)):
            # Using yellow (93) color for "Already exists!"
            logger.info("Note: \033[93mAlready exists!\033[0m Write: " +
                  os.path.join(FOLDER_DIR, FILE_NAME))
            df = pd.read_csv(os.path.join(FOLDER_DIR, FILE_NAME),
                             index_col="Experiment Repetition Index")
            # Right-align each column with width of 12 characters to ensure alignment between headers and values
            logger.info(" ".join([f"{col:>12}" for col in df.columns]))
            logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
            continue 

        logger.info("Writing: " + os.path.join(FOLDER_DIR, FILE_NAME))
        try:
            with Pool() as p:
                df = pd.DataFrame(np.asarray(p.map(_parallel_run_data_module, pool_param_list)),
                                  columns=["(0,KS)", "(1,KS)", "(2,KS)",
                                           "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"])
                # Right-align each column with width of 12 characters to ensure alignment between headers and values
                logger.info(" ".join([f"{col:>12}" for col in df.columns]))
                logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
                df.to_csv(os.path.join(FOLDER_DIR, FILE_NAME),
                            index=True,
                            index_label="Experiment Repetition Index")
                del df
            logger.info("\033[92mSuccess!\033[0m")
        except BaseException as ex:
            logger.error("Fail on "+os.path.join(FOLDER_DIR, FILE_NAME))
            if os.path.exists(os.path.join(FOLDER_DIR, FILE_NAME)):
                os.remove(os.path.join(FOLDER_DIR, FILE_NAME))

            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()

            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                    trace[0], trace[1], trace[2], trace[3]))

            logger.error("Exception type : %s " % ex_type.__name__)
            logger.error("Exception message : %s" % ex_value)
            for i, trace in enumerate(stack_trace):
                logger.error("Stack trace %d: %s" % (i+1, trace))

def exp_tail_probability_thresholds():
    # Single threshold percentage and multi-threshold percentages
    exp_name = inspect.currentframe().f_code.co_name
    logger.info("Experiment name: %s", exp_name)
    data_distributions = ['gamma', 'lognorm', 'pareto', 'genpareto']
    data_sizes = [500]
    # Single threshold percentage and multi-threshold percentages
    threshold_percentages = ["0.6", "0.7", "0.8", "0.9", ["0.6", "0.7", "0.8", "0.9"]]
    percentage_lhs_values = ["0.99"]
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
    threshold_percentages = ["0.7"]
    percentage_lhs_values = [f"{x:.2f}" for x in np.linspace(0.9, 0.99, 10)]
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
    threshold_percentages = ["0.6","0.65","0.7","0.75","0.8","0.85"]
    percentage_lhs_values = ["0.9"]
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

def _parallel_real_data_run(pool_param: tuple) ->  typing.List[float]:
    intput_data_region, meta_data_dict, FOLDER_DIR, FILE_NAME, exp_name = pool_param
    parameter_pairs = FILE_NAME.removeprefix(exp_name + "_").removesuffix('.csv').split('=')
    names = [x.split('_', maxsplit=1)[1] if idx !=0 else x for idx, x in enumerate(parameter_pairs[:-1])]
    values = [x.split('_', maxsplit=1)[0] for _, x in enumerate(parameter_pairs[1:]) ]

    logger.info(f"Running experiment {exp_name} with parameters:")
    # Using blue (94) for parameter names and green (92) for values
    logger.info(" ".join([f"\033[94m{name:>{max(len(name), len(value))}}\033[0m" for name, value in zip(names,values)]))
    logger.info(" ".join([f"\033[92m{value:>{max(len(name), len(value))}}\033[0m" for name, value in zip(names,values)]))

    file_path = os.path.join(FOLDER_DIR, FILE_NAME)
    if os.path.exists(file_path):
        # Using yellow (93) color for "Already exists!"
        logger.info("Note: \033[93mAlready exists!\033[0m Write: " + file_path)
        df = pd.read_csv(file_path, index_col="Experiment Repetition Index")
        logger.info(" ".join([f"{col:>12}" for col in df.columns]))
        logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
        return FILE_NAME, df.mean(axis=0).to_dict(), "exists"
    
    logger.info("Writing: " + file_path)
    try:                         
        df = pd.DataFrame([tpe.estimate_tail_probability(input_data=intput_data_region, **meta_data_dict)],
                            columns=["(0,KS)", "(1,KS)", "(2,KS)",
                                        "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"])

        # Right-align each column with width of 12 characters to ensure alignment between headers and values
        logger.info(" ".join([f"{col:>12}" for col in df.columns]))
        logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
        df.to_csv(file_path, index=True, index_label="Experiment Repetition Index")
        logger.info("\033[92mSuccess!\033[0m")
        return (FILE_NAME, df.mean(axis=0).to_dict(), "success")
    
    except BaseException as ex:
        logger.error("Fail on " + file_path)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Get current system exception
        ex_type, ex_value, ex_traceback = sys.exc_info()

        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                trace[0], trace[1], trace[2], trace[3]))

        logger.error("Exception type : %s " % ex_type.__name__)
        logger.error("Exception message : %s" % ex_value)
        for i, trace in enumerate(stack_trace):
            logger.error("Stack trace %d: %s" % (i+1, trace))

        return (FILE_NAME, str(ex), "fail")

def exp_tail_probability_real_data():
    # Experiment on real data

    # generate a folder to store the results
    exp_name = inspect.currentframe().f_code.co_name
    FOLDER_DIR = os.path.join(pathlib.Path(__file__).parents[1], "raw_output", exp_name)
    if not os.path.isdir(FOLDER_DIR):
       os.mkdir(FOLDER_DIR)
    logger.info("Experiment name: %s", exp_name)

    meta_data_dict = {"alpha": 0.05,
                      "g_ellipsoidal_dimension": 3}
    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]
    threshold_percentages = ["0.6","0.65","0.7","0.75","0.8","0.85"]
    left_end_point_objectives = ["7.0", "7.1", "7.2", "7.3", "7.4", "7.5", 
                                 "7.6", "7.7", "7.8", "7.9", "8.0",
                                 "7.25"]
    bootstrapping_size=500
    random_seed = 20220222
    meta_data_dict["bootstrapping_size"] = bootstrapping_size
    meta_data_dict["random_state"] = random_seed
    meta_data_dict["right_end_point_objective"] = np.inf

    intput_data_regions = [np.loadtxt(os.path.join(pathlib.Path(__file__).parents[1], 
                                            "input_data", 
                                            "cmt", 
                                            "parsed_data", 
                                            region+".csv"))
                                            for region in regions]

    # Prepare parameter list
    param_list = []
    for region, intput_data_region, threshold_percentage, left_end_point_objective in itertools.product(
            regions, intput_data_regions, threshold_percentages, left_end_point_objectives):
        
        _meta_data_dict = meta_data_dict.copy()
        _meta_data_dict["left_end_point_objective"] = float(left_end_point_objective)
        _meta_data_dict["threshold_percentage"] = float(threshold_percentage)
        FILE_NAME = [exp_name + f"_region={region.replace('_', '-')}"]
        FILE_NAME += [key+"="+str(_meta_data_dict[key]) for key in _meta_data_dict if key != "bootstrapping_size"]
        FILE_NAME = '_'.join(FILE_NAME)+".csv"


        param_list.append((intput_data_region, _meta_data_dict, FOLDER_DIR, FILE_NAME, exp_name))

    with Pool() as pool:
        results = pool.map(_parallel_real_data_run, param_list)
        
    # Optionally, log results
    for FILE_NAME, result, status in results:
        if status == "success":
            logger.info(f"Success: {FILE_NAME} {result}")
        elif status == "exists":
            logger.info(f"Already exists: {FILE_NAME}")
        else:
            logger.error(f"Failed: {FILE_NAME} {result}")
        
if __name__ == '__main__':
    logger.info("Starting experiment of tail probability estimation")
    # exp_tail_probability_thresholds()
    # exp_tail_probability_percentage_lhs()
    # exp_tail_probability_scarce_data()
    exp_tail_probability_real_data()