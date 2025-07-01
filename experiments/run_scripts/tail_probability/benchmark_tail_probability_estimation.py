import typing
import pathlib
from scipy.stats import gamma
import rpy2.robjects as ro
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
numpy2ri.activate()

importr('base')
importr('utils')
importr('nloptr')
importr('MASS')
importr('POT')
importr('QRM')
importr('eva')

def benchmark_estimate_tail_probability(input_data: np.ndarray, 
                                        left_end_point_objective: float, right_end_point_objective: float,
                                        alpha: float,
                                        method: str) -> typing.List[float]:

    RCodeLib = "\n".join([
        "rm(list=ls())",
        f"source('{pathlib.Path(__file__).parents[1]}/evtr/common_utils.R')",
        "" # empty line
    ])
    if method == 'pot':
        with open(f'{pathlib.Path(__file__).parents[1]}/evtr/gpdTIP_pot.R','r') as f:
            RCodeLib += f.read()    
    elif method == 'pot_bt':
        with open(f'{pathlib.Path(__file__).parents[1]}/evtr/gpdTIP_pot_bt.R','r') as f:
            RCodeLib += f.read()
    elif method == 'pl':
        with open(f'{pathlib.Path(__file__).parents[1]}/evtr/gpdTIP_pl.R','r') as f:
            RCodeLib += f.read()
    elif method == 'bayesian':
        with open(f'{pathlib.Path(__file__).parents[1]}/evtr/gpdTIP_bayesian.R','r') as f:
            RCodeLib += f.read()
    elif method == 'pwm':
        with open(f'{pathlib.Path(__file__).parents[1]}/evtr/gpdTIP_pwm.R','r') as f:
            RCodeLib += f.read()
    else:
        raise NotImplementedError()

    RCodeApply = f'''
lhs <- {left_end_point_objective}

rhs <- if('{right_end_point_objective}' == 'inf') Inf else {right_end_point_objective}

data <- c({','.join(map(str, input_data.tolist()))})

conf <- 1 - {alpha}

out <- tryCatch(
	gpdTIP(data, lhs, rhs, conf=conf), 
	error = function(e) e
)

bbd <- if ("error" %in% class(out)) NA else{{
	out$CI
}}
bbd 
    '''
    try: 
        ro_result = ro.r(RCodeLib+RCodeApply).tolist()
        return ro_result
    except Exception as e:
        return [0,0]


if __name__ == '__main__':
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # data_size = 500
    # true_value = 0.005
    # percentage_lhs = 0.9
    # percentage_rhs = percentage_lhs + true_value
    # threshold_percentage = 0.7
    # alpha = 0.05
    # random_state = 20220222
    # g_ellipsoidal_dimension = 3
    # logger.info("A small example on tail probability estimation--single threshold.")
    # result = estimate_tail_probability_with_data_module(
    #     gamma, percentage_lhs, percentage_rhs, data_size, threshold_percentage, g_ellipsoidal_dimension, alpha, random_state)
    # logger.info(f"{[f'{x:.2E}' for x in result]}")
    # logger.info("A small example on tail probability estimation--multiple thresholds.")
    # threshold_percentage = [0.65, 0.7, 0.75, 0.8]
    # result = estimate_tail_probability_with_data_module(
    #     gamma, percentage_lhs, percentage_rhs, data_size, threshold_percentage, g_ellipsoidal_dimension, alpha, random_state)
    # logger.info(f"{[f'{x:.2E}' for x in result]}")
