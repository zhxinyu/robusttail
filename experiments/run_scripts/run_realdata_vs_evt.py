import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import zscore
import sys
import os
import math
sys.path.append("/Users/sam/ws/robusttail")
sys.path.append("/Users/sam/ws/robusttail/experiments/run_scripts")
os.environ['R_HOME'] = sys.executable.replace('bin/python', 'lib/R')
import tail_probability.benchmark_tail_probability_estimation as btpe
import tail_probability.tail_probability_estimation as tpe
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

scale = 1

def get_data():
    raw_ndk = ""
    with open('/Users/sam/ws/robusttail/experiments/input_data/cmt/raw_data/jan76_dec20.ndk') as f:
        raw_ndk = f.read()
    
    raw_ndk = raw_ndk.rstrip("\n").split("\n")
    num_events = len(raw_ndk)//5
    
    events_raw = []
    for num_iter in range(num_events):
        # date: 1,6-15
        # location: 1,57-80
        # exponent: 4, 1-2
        # scalar moment: 5, 50-56
        # Reference: https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/allorder.ndk_explained
        # Mw = 2/3 * (lgM0  - 16.1), 
        # Reference: https://www.globalcmt.org/CMTsearch.html
        an_event = raw_ndk[num_iter * 5 : num_iter * 5 + 5]
        date = an_event[0][5:16].strip()
        location = an_event[0][56:81].strip()
        exponent = float(an_event[3][0:2].strip())
        scalar = float(an_event[4][49:57].strip())
        mw = float(2/3* (np.log10(scalar) + exponent - 16.1))
        events_raw.append([date, location, scalar, exponent, mw])
    df = pd.DataFrame(events_raw, columns=["date", "location", "scalar", "exponent", "Mw"]).set_index('date')
    df.loc[:, 'location'] = df['location'].str.replace(',', '').str.replace(' ', '_').str.upper()
    return df

def _parallel_run(pool_param: tuple):
    method, input_data, lhs, right_end_point_objective, kwargs, alpha = pool_param
    if 'opt' in method:
        if len(method.split('_')) == 2:
            right_endpoint = float(method.split('_')[1])/scale
        else:
            right_endpoint = np.inf
        kwargs['right_endpoint'] = right_endpoint
        results = tpe.estimate_tail_probability_D2_chi2_only(input_data=input_data, 
                                                            left_end_point_objective=lhs, 
                                                            right_end_point_objective=right_end_point_objective, **kwargs)
    else:
        results = btpe.benchmark_estimate_tail_probability(input_data=input_data, 
                                                            left_end_point_objective=lhs,
                                                            right_end_point_objective=right_end_point_objective,
                                                            method=method, 
                                                            alpha=alpha)
    return results

def run_plain_estimation():
    df = get_data()

    alpha=0.05
    right_end_point_objective=np.inf
    g_ellipsoidal_dimension=5
    threshold_percentage=0.7
    random_state=20220222
    bootstrapping_size=500
    kwargs = {
        "threshold_percentage": threshold_percentage,
        "g_ellipsoidal_dimension": g_ellipsoidal_dimension,
        "alpha": alpha,
        "random_state": random_state,
        "bootstrapping_size": bootstrapping_size,
        "right_endpoint": np.inf
    }

    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", 
               "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", 
               "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]

    # set seed
    for data_sample in [0]:
        table_per_region = {}
        for region in regions:
            input_data_full = df.loc[df['location'] == region, 'Mw']
            input_data_full = input_data_full.to_frame().apply(zscore, axis=0).squeeze()
            np.random.seed(random_state)
            # input_data = np.random.choice(input_data_full, size=data_sample, replace=False)
            input_data = input_data_full
            print(f"Running {region} with length {len(input_data)}")
            methods = ['opt', 'pot_bt', 'pl', 'bayesian', 'pwm']
            quantiles = list(np.arange(0.991, 0.999, 0.001)) + list(np.arange(0.9991, 0.9995, 0.0001))
            setup = []
            for quantile in quantiles:
                lhs = np.quantile(input_data_full, q=quantile)
                kwargs['threshold_percentage'] = np.max([1 - 60 / len(input_data_full), threshold_percentage])
                setup.extend([(method, input_data, lhs, right_end_point_objective, kwargs, alpha) for method in methods])
            with Pool() as pool:
                results = pool.map(_parallel_run, setup)
            all_results = []
            for result in results:
                all_results.append(f"[{result[0]:.2E},{result[1]:.2E}]")
            # reshape to len(quantiles) x len(methods)
            all_results = np.array(all_results).reshape(len(quantiles), len(methods))
            df_result = pd.DataFrame(all_results, columns=pd.Index(["(2, CHI)", "pot_bt", "pl", "bayesian", "pwm"]),
                                    index=pd.Index([np.round(quantile, 4) for quantile in quantiles], name="quantile"))
            table_per_region[region] = df_result
        # print as csv
        df = pd.concat(table_per_region, axis=0)
        df.to_csv(f"/Users/sam/ws/robusttail/real_data_vs_evt_resample_{data_sample}.csv")
        print(df.to_markdown())
    #         results = tpe.estimate_tail_probability_D2_chi2_only(input_data=input_data, 
    #                                     left_end_point_objective=lhs, 
    #                                     right_end_point_objective=right_end_point_objective, **kwargs)        
    # all_results = [[f"{x:.2E}" for x in results]]
    # for method in methods:
    #     benchmark_result = btpe.benchmark_estimate_tail_probability(input_data=input_data, 
    #                                                             left_end_point_objective=lhs,
    #                                                             right_end_point_objective=right_end_point_objective,
    #                                                             method=method, 
    #                                                             alpha=alpha)
    #     all_results.append([f"{x:.2E}" for x in benchmark_result])
    # table_raw.append([region, quantile] + all_results)
    # df_result = pd.DataFrame(table_raw, columns=["region", "quantile", "(2, CHI)", "pot_bt", "pl", "bayesian", "pwm"])
    # df_result = df_result.set_index(["region", "quantile"])
    # print(df_result)            
    # for row_idx in range(len(table_raw)):
    #     for element_idx in range(2, 8):
    #         if table_raw[row_idx][element_idx][0][0] == '-':
    #             table_raw[row_idx][element_idx][0] = '0.00E+00'
    #         table_raw[row_idx][element_idx] = '[' + ','.join(table_raw[row_idx][element_idx]) + ']'
    #         table_raw[row_idx][element_idx] = table_raw[row_idx][element_idx].replace('0.00E+00', '0')
    #         table_raw[row_idx][element_idx] = table_raw[row_idx][element_idx].replace(',', ', ')


    # ranks = []
    # for row in table_raw:
    #     width = []
    #     for element in row:
    #         a, b = element[1:-1].split(',')
    #         width.append(float(b) - float(a))
    #     # get the rank for each element
    #     rank = np.argsort(np.argsort(width)) + 1 # Sort in descending order and get indices
    #     ranks.append('|'.join([str(idx) for idx in rank]))
        
    # df_result = pd.DataFrame(table_raw, columns=["region", "quantile", "(2, CHI)", "pot_bt", "pl", "bayesian", "pwm"])
    # df_result = df_result.set_index(["region", "quantile"])


def run_different_threshold_percentages():
    df = get_data()

    alpha=0.05
    right_end_point_objective=np.inf
    g_ellipsoidal_dimension=3
    threshold_percentages = np.linspace(0.6, 0.9, 31).tolist()
    random_state=20220222
    bootstrapping_size=500
    kwargs = {
        "g_ellipsoidal_dimension": g_ellipsoidal_dimension,
        "alpha": alpha,
        "random_state": random_state,
        "bootstrapping_size": bootstrapping_size,
        "right_endpoint": np.inf
    }

    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", 
               "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", 
               "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]

    # set seed
    for data_sample in [0]:
        table_per_region = {}
        for region in regions:
            input_data_full = df.loc[df['location'] == region, 'Mw']
            input_data_full = input_data_full.to_frame().apply(zscore, axis=0).squeeze()
            np.random.seed(random_state)
            # input_data = np.random.choice(input_data_full, size=data_sample, replace=False)
            input_data = input_data_full
            print(f"Running {region} with length {len(input_data)}")
            # methods = ['opt', 'pot_bt', 'pl', 'bayesian', 'pwm']
            methods = ['opt']
            columns = ["(2, CHI)"]
            # columns = ["(2, CHI)", "pot_bt", "pl", "bayesian", "pwm"]
            quantile = 0.999
            setup = []
            lhs = np.quantile(input_data_full, q=quantile)
            for threshold_percentage in threshold_percentages:
                kwargs_copy = kwargs.copy()
                kwargs_copy['threshold_percentage'] = threshold_percentage
                setup.extend([(method, input_data, lhs, right_end_point_objective, kwargs_copy, alpha) for method in methods])
            with Pool(processes=16) as pool:
                results = pool.map(_parallel_run, setup)
            all_results = []
            for result in results:
                all_results.append(f"[{result[0]:.2E},{result[1]:.2E}]")
            # reshape to len(threshold_percentages) x len(methods)
            all_results = np.array(all_results).reshape(len(threshold_percentages), len(methods))
            df_result = pd.DataFrame(all_results, columns=pd.Index(columns),
                                     index=pd.Index([np.round(threshold_percentage, 4) for threshold_percentage in threshold_percentages], 
                                                    name="threshold_percentage")
                                    )
            table_per_region[region] = df_result
        # print as csv
        df = pd.concat(table_per_region, axis=0)
        df.to_csv(f"/Users/sam/ws/robusttail/real_data_vs_evt_threshold_percentage_{data_sample}.csv")
        print(df.to_markdown())


def run_bootstrap_estimation():
    df = get_data()

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
        "right_endpoint": np.inf
    }

    # Bootstrap Stability
    # Use bootstrapping to assess:
    # 	•	Stability of interval bounds across resamples
    # 	•	CI width consistency
    # 	•	Whether empirical coverage remains within the bounds across samples
    # For your method:
    # 	•	Bootstrap each regional dataset
    # 	•	Report variation in the lower and upper bounds of your intervals
    regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA", 
               "TURKEY", "HOKKAIDO_JAPAN_REGION", 
               "BANDA_SEA", "KURIL_ISLANDS", 
               "SOLOMON_ISLANDS", "FIJI_ISLANDS_REGION"]
    # regions = ["ECUADOR", "OFF_COAST_OF_NORTHERN_CA"]
    
    table_raw = []
    # set random seed for reproducibility
    np.random.seed(random_state)
    for region in regions:
        input_data = df.loc[df['location'] == region, 'Mw']/scale
        # set one: size = len(input_data)
        # set two: size = 60
        # set three: various sizes = 50, 75, 100, 125, 150, 200, various quantiles = 0.95, 0.98, 0.99, 0.991, 0.993, 0.995
        for resample_size in [50, 60, 70, 80, 90, 100]:
        # for resample_size in [50]:
            input_data_bootstrap = [np.random.choice(input_data, size=resample_size, replace=True) for _ in range(200)]
            for method in ['opt', 'pot_bt', 'pl', 'bayesian', 'pwm']:
                # quantiles = [0.95, 0.98, 0.99, 0.991, 0.993, 0.995]
                quantiles = [0.99]
                for quantile in quantiles:
                    lhs = np.quantile(input_data, q=float(quantile))
                    
                    if os.path.exists(f'/Users/sam/ws/robusttail/run_realdata_result_set3/real_data_bootstrap_{method}_{region}_{quantile}_{resample_size}.csv'):
                        print(f"Skipping {region} {quantile} {method} because it already exists")
                        continue 
                    print(f"Running {region} {quantile} {method} {resample_size}")

                    pool_param_list = [(method, 
                                        input_data_bootstrap[i], 
                                        lhs, 
                                        right_end_point_objective, 
                                        kwargs, alpha) 
                                        for i in range(200)]

                    with Pool() as pool:
                        bootstrap_results = pool.map(_parallel_run, pool_param_list)
                    # add region, quantile to the bootstrap results
                    bootstrap_results = [[method, region, float(quantile), resample_size] + bootstrap_result for bootstrap_result in bootstrap_results]
                    # save as csv. 
                    print(f"Saving {region} {quantile} {method} {resample_size}")
                    bootstrap_results_df = pd.DataFrame(bootstrap_results, columns=['method', 'region', 'quantile', 'resample_size', 'lb', 'ub'])
                    bootstrap_results_df.to_csv(f'/Users/sam/ws/robusttail/run_realdata_result_set3/real_data_bootstrap_{method}_{region}_{quantile}_{resample_size}.csv', index=False)
                    print("lb and ub are (mean):")
                    print(bootstrap_results_df[['lb', 'ub']].mean().to_frame().T)

if __name__ == "__main__":
    # run_plain_estimation()
    run_different_threshold_percentages()