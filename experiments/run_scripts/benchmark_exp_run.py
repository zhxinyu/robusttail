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

import droevt.utils.synthetic_data_generator as data_utils
import tail_probability.benchmark_tail_probability_estimation as benchmark_tpe

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

    ro_result = benchmark_tpe.benchmark_estimate_tail_probability(
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
    
    # generate a folder to store the results
    FOLDER_DIR = os.path.join(pathlib.Path(__file__).parents[1], "raw_output", exp_name)
    if not os.path.isdir(FOLDER_DIR):
       os.mkdir(FOLDER_DIR)
    
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

        FILE_NAME = f"{method}_tail_probability_estimation_"\
                    f"data_distribution={data_distribution}_"\
                    f"data_size={data_size}_percentage_lhs={percentage_lhs}_"\
                    f"true_value={true_value}_"\
                    f"random_seed={random_seed}_"\
                    f"n_experiment_repetitions={n_experiment_repetitions}.csv"

        parameter_pairs = FILE_NAME.removeprefix(f"{method}_tail_probability_estimation_").removesuffix('.csv').split('=')
        names = ["method"] + [x.split('_', maxsplit=1)[1] if idx !=0 else x for idx, x in enumerate(parameter_pairs[:-1]) ]
        values = [method] + [x.split('_', maxsplit=1)[0] for idx, x in enumerate(parameter_pairs[1:]) ]
        logger.info(f"Running experiment {exp_name} with parameters:")
        # Using blue (94) for parameter names and green (92) for values
        logger.info(" ".join([f"\033[94m{name:>{max(len(name), len(value))}}\033[0m" for name, value in zip(names, values)]))
        logger.info(" ".join([f"\033[92m{value:>{max(len(name), len(value))}}\033[0m" for name, value in zip(names, values)]))

        if os.path.exists(os.path.join(FOLDER_DIR, FILE_NAME)):
            # Using yellow (93) color for "Already exists!"
            logger.info("Note: \033[93mAlready exists!\033[0m Write: " +
                  os.path.join(FOLDER_DIR, FILE_NAME))
            df = pd.read_csv(os.path.join(FOLDER_DIR, FILE_NAME), index_col="Experiment Repetition Index")
            # Right-align each column with width of 12 characters to ensure alignment between headers and values
            logger.info(" ".join([f"{col:>12}" for col in df.columns]))
            logger.info(" ".join([f"{val:>12.2E}" for val in df.mean(axis=0)]))
            continue 

        logger.info("Writing: " + os.path.join(FOLDER_DIR, FILE_NAME))
        try:
            with Pool() as p:
                df = pd.DataFrame(np.asarray(p.map(_parallelRun, pool_param_list)),
                                    columns=["Lower Bound", "Upper Bound"])
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
    
    meta_data_dict = {}
    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]
    meta_data_dict["left_end_point_objective"] = ["7.0", "7.1", "7.2", "7.3", "7.4", "7.5", 
                                                  "7.6", "7.7", "7.8", "7.9", "8.0",
                                                  "7.25"]
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
            for left_end_point_objective in meta_data_dict["left_end_point_objective"]:
                pool_param = {"input_data": input_data,
                              "left_end_point_objective": left_end_point_objective,
                              "right_end_point_objective": right_end_point_objective,
                              "method": method,
                              "alpha": 0.05}
                pool_param_list.append(pool_param)

    FILE_NAME = os.path.join(pathlib.Path(__file__).parents[1], "raw_output", exp_name, "result.csv")

    if os.path.exists(FILE_NAME):
        # Using yellow (93) color for "Already exists!"
        logger.info("Note: \033[93mAlready exists!\033[0m Write: " + FILE_NAME)
        df = pd.read_csv(FILE_NAME, index_col="Experiment Repetition Index")
        # Right-align each column with width of 12 characters to ensure alignment between headers and values
        logger.info(" ".join([f"{col:>12}" for col in df.columns]))
        logger.info(df)
        return
    
    logger.info("Writing: " + FILE_NAME)
    try:
        with Pool() as p:
            ro_result_list = p.map(_parallelRun, pool_param_list)
            result = [[pool_param['method'], 
                       pool_param['region'], 
                       pool_param['left_end_point_objective'], 
                       ro_result[0], 
                       ro_result[1]] for (pool_param, ro_result) in zip(pool_param_list, ro_result_list)]            
        df = pd.DataFrame(result, columns=["Method", "Region", "Left End Point Objective", "Lower Bound", "Upper Bound"])
        df.to_csv(FILE_NAME, index=False)
        logger.info("\033[92mSuccess!\033[0m")
    except BaseException as ex:
        logger.error("Fail on "+FILE_NAME)
        if os.path.exists(FILE_NAME):
            os.remove(FILE_NAME)

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

if __name__ == '__main__':
    logger.info("Starting experiment of tail probability estimation")
    benchmark_exp_tail_probability()
    # benchmark_exp_tail_probability_real_data()