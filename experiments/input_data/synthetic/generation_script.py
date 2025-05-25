import os
import numpy as np
from scipy.stats import gamma, lognorm, pareto, genpareto
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import droevt.utils.synthetic_data_generator as data_utils

def runner(data_size: int):
    random_seed = 20220222
    
    data_module_map = {"gamma": gamma,
                       "lognorm": lognorm,
                       "pareto": pareto,
                       "genpareto": genpareto}
    meta_data_dict = {"data_size": data_size}
    data_sources = ["gamma", "lognorm", "pareto", "genpareto"]
    
    nrep = 200
    for data_source in data_sources:
        file_dir = f"{os.path.dirname(__file__)}/n{data_size}/{data_source}/default"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        data_module = data_module_map[data_source]
        data_param_dict = data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_source]
        logger.info(f"Generating {data_source} data with {nrep} repetitions")
        logger.info(f"The quantile points for {data_source} are lhs={data_utils.get_quantile(data_module, 0.99, data_param_dict)}, "
                    f"rhs={data_utils.get_quantile(data_module, 0.995, data_param_dict)}")
        for nnrep in range(nrep):
            meta_data_dict['random_state'] = random_seed + nnrep
            input_data = data_utils.generate_synthetic_data(data_module, 
                                                            data_param_dict, 
                                                            meta_data_dict['data_size'], 
                                                            meta_data_dict['random_state'])
            np.savetxt(f"{file_dir}/random_seed={random_seed + nnrep}.csv", input_data, delimiter=",")
                
if __name__ == '__main__':
    runner(data_size=500)
    runner(data_size=30)