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
    df = pd.DataFrame(events_raw, columns=pd.Index(["date", "location", "scalar", "exponent", "Mw"])).set_index('date')
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

def run_different_critical_values():
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
            quantiles = list(np.arange(0.990, 0.999, 0.001)) + list(np.arange(0.9991, 0.9995, 0.0001))
            setup = []
            for quantile in quantiles:
                lhs = np.quantile(input_data_full, q=quantile)
                # kwargs_copy = kwargs.copy()
                # kwargs_copy['threshold_percentage'] = np.max([1 - 60 / len(input_data_full), threshold_percentage])
                setup.extend([(method, input_data, lhs, right_end_point_objective, kwargs, alpha) for method in methods])
            with Pool() as pool:
                results = pool.map(_parallel_run, setup)
            all_results = []
            for result in results:
                all_results.append(f"[{result[0]},{result[1]}]")
            # reshape to len(quantiles) x len(methods)
            all_results = np.array(all_results).reshape(len(quantiles), len(methods))
            df_result = pd.DataFrame(all_results, columns=pd.Index(["(2, CHI)", "pot_bt", "pl", "bayesian", "pwm"]),
                                    index=pd.Index([np.round(quantile, 4) for quantile in quantiles], name="quantile"))
            table_per_region[region] = df_result
        # print as csv
        df = pd.concat(table_per_region, axis=0)
        df.to_csv(f"/Users/sam/ws/robusttail/real_data_vs_evt_critical_values_{data_sample}.csv")
        # Print DataFrame as scientific notation with 2 decimals (e.g. 1.23E+04)
        with pd.option_context('display.float_format', '{:.2E}'.format):
            print(df)

def run_different_confidence_levels():
    df = get_data()

    alphas = np.linspace(0.01, 0.1, 10).tolist()[::-1]
    right_end_point_objective=np.inf
    g_ellipsoidal_dimension=3
    threshold_percentage=0.7
    quantile = 0.999
    random_state=20220222
    bootstrapping_size=500
    kwargs = {
        "threshold_percentage": threshold_percentage,
        "g_ellipsoidal_dimension": g_ellipsoidal_dimension,
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
            setup = []
            lhs = np.quantile(input_data_full, q=quantile)
            for alpha in alphas:
                kwargs_copy = kwargs.copy()
                kwargs_copy['alpha'] = alpha
                setup.extend([(method, input_data, lhs, right_end_point_objective, kwargs_copy, alpha) for method in methods])

            with Pool() as pool:
                results = pool.map(_parallel_run, setup)
            all_results = []
            for result in results:
                all_results.append(f"[{result[0]},{result[1]}]")
            # reshape to len(quantiles) x len(methods)
            all_results = np.array(all_results).reshape(len(alphas), len(methods))
            df_result = pd.DataFrame(all_results, columns=pd.Index(["(2, CHI)", "pot_bt", "pl", "bayesian", "pwm"]),
                                     index=pd.Index([np.round(1 - alpha, 2) for alpha in alphas], name="confidence_level"))
            table_per_region[region] = df_result
        # print as csv
        df = pd.concat(table_per_region, axis=0)
        df.to_csv(f"/Users/sam/ws/robusttail/real_data_vs_evt_confidence_levels_{data_sample}.csv")
        # Print DataFrame as scientific notation with 2 decimals (e.g. 1.23E+04)
        with pd.option_context('display.float_format', '{:.2E}'.format):
            print(df)

def run_bootstrap_estimation():

    # bootstrap
    # variation: number of bootstrap samples (50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150)
    #            for each bootstrap sample, vary the critical value list(np.arange(0.990, 0.999, 0.001)) + list(np.arange(0.9991, 0.9995, 0.0001))

    df = get_data()

    alpha = 0.05
    right_end_point_objective=np.inf
    g_ellipsoidal_dimension=3
    threshold_percentage=0.7
    quantiles = [0.99, 0.992, 0.994, 0.996, 0.998, 0.9991, 0.9993, 0.9995]
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
    repeats = 200
    for region in regions:
        # for data_sample in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]:
        for data_sample in [60, 80, 100, 110, 120, 130, 140, 150]:
            input_data_full = df.loc[df['location'] == region, 'Mw']
            input_data_full = input_data_full.to_frame().apply(zscore, axis=0).squeeze()
            print(f"Running {region}... Resample size: {data_sample}, Repeats: {repeats}")
            methods = ['opt', 'pot_bt', 'pl', 'bayesian', 'pwm']

            setup = []
            for quantile in quantiles:
                lhs = np.quantile(input_data_full, q=quantile)
                # set seed
                np.random.seed(random_state)
                for _ in range(repeats):
                    input_data = np.random.choice(input_data_full, size=data_sample, replace=True)
                    setup.extend([(method, input_data, lhs, right_end_point_objective, kwargs, alpha) for method in methods])

            with Pool() as pool:
                results = pool.map(_parallel_run, setup)
            # reshape to len(quantiles) x len(methods) x repeats
            all_results = np.array(results).reshape(len(quantiles), repeats, len(methods), 2)
            # true_value = 1 - np.array(quantiles)
            # hit_case = (all_results[..., 0] <= true_value[:, np.newaxis, np.newaxis]) * (all_results[..., 1] >= true_value[:, np.newaxis, np.newaxis])
            # width = all_results[..., 1] - all_results[..., 0]
            # width_mean, width_std = np.nanmean(width, axis=1), np.nanstd(width, axis=1, ddof=1)
            # hit_rate, hit_rate_std = hit_case.astype(float).mean(axis=1), hit_case.astype(float).std(axis=1, ddof=1))
            
            # print as csv
            import pickle
            with open(f"/Users/sam/ws/robusttail/real_data_vs_evt_bootstrap_results_{region}_{data_sample}.pkl", 'wb') as f:
                pickle.dump(all_results, f)
            print(all_results)

if __name__ == "__main__":
    import argparse
    # implement argparse to choose the function to run
    parser = argparse.ArgumentParser(description='Run real data vs evt estimation')
    parser.add_argument('--function', type=str, default='run_bootstrap_estimation_v3', help='Function to run')
    args = parser.parse_args()
    logger.info(f"Running {args.function}")
    if args.function == 'run_different_threshold_percentages':
        run_different_threshold_percentages()
    elif args.function == 'run_different_critical_values':
        run_different_critical_values()
    elif args.function == 'run_different_confidence_levels':
        run_different_confidence_levels()
    elif args.function == 'run_bootstrap_estimation':
        run_bootstrap_estimation()
    else:
        print("Invalid function")
        exit(1)