
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd

import droevt.utils.synthetic_data_generator as data_utils
import tail_probability.benchmark_tail_probability_estimation as benchmark_tpe


def runner(args):
    random_seed = 20220222
    string_to_data_module = {"gamma": gamma,
                             "lognorm": lognorm,
                             "pareto": pareto,
                             "genpareto": genpareto}
    true_value = "0.005"
    meta_data_dict = {"data_size": 500}
    list_percentage_lhs = [args.lhs]
    data_sources = [ args.ds ]
    method = args.method
    alpha = 0.05
    nrep = 200
    result = []
    columns = ["Data Source", 'nData',"percentageLHS", "Lower Bound","Upper Bound", "True Value", "Repetition Index"]
    for percentage_lhs in list_percentage_lhs:
        percentage_lhs = float(percentage_lhs)
        percentage_rhs = percentage_lhs + float(true_value)
        for data_source in data_sources:
            data_module = string_to_data_module[data_source]
            data_size = meta_data_dict['data_size']
            left_end_point_objective = data_utils.get_quantile(
                data_module, percentage_lhs, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])
            right_end_point_objective = data_utils.get_quantile(
                data_module, percentage_rhs, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_module.name])

            for nnrep in range(nrep):
                print(f"Working on {percentage_lhs}_{data_source}_{nnrep}")
                try:
                    input_data = data_utils.generate_synthetic_data(
                        data_module, data_utils.DISTRIBUTION_DEFAULT_PARAMETERS[data_source], data_size, random_seed+nnrep)                    
                    ro_result = benchmark_tpe.benchmark_estimate_tail_probability(
                        input_data=input_data, 
                        left_end_point_objective=left_end_point_objective, 
                        right_end_point_objective=right_end_point_objective, 
                        alpha=alpha,
                        method=method)
                    result.append([data_source, meta_data_dict['data_size'], percentage_lhs, ro_result[0], ro_result[1], true_value, nnrep])
                    print(result[-1])
                except Exception as e: 
                    print(e)
                    result.append([data_source, meta_data_dict['data_size'], 
                                   percentage_lhs, 0, 
                                   0, 
                                   true_value, nnrep])
            # assert False
            print(f"Finish experiments on {percentage_lhs}-{data_source}")
            df = pd.DataFrame(data=result, columns=columns)
            # df.to_csv(os.path.join(file_dir, f'table5_{percentage_lhs}_{data_source}.csv'), header=columns, index = False)

if __name__ == '__main__':

    """Usage
    
        ```
        #!/bin/bash 
        for lhs in 0.9 0.95 0.99
        do
            echo "Running" ${lhs} gamma pl
            python table5_ab.py ${lhs} gamma pl &
            echo "Running" ${lhs} lognorm pl
            python table5_ab.py ${lhs} lognorm pl &
            echo "Running" ${lhs} pareto pl
            python table5_ab.py ${lhs} pareto pl &
        done
        wait 
        echo "done all processes."
        ``` 
    """
    # Import argparse at the top of the file
    import argparse
    
    parser = argparse.ArgumentParser(description='TIP estimation with user-specific methods.')
    parser.add_argument('--lhs', type=float, default=0.9, required=False, help='LHS in the objective function')
    parser.add_argument('--ds', type=str, default='gamma', required=False, help='Data source for simulation')
    parser.add_argument('--method', type=str, default='pwm', required=False, help='choose the method for tail probability estimation: pot, pl, bayesian and pwm')
    args = parser.parse_args()
    
    runner(args)