import numpy as np
import pandas as pd
from multiprocessing import Pool

import sys
import os
sys.path.append("/Users/sam/ws/robusttail")
sys.path.append("/Users/sam/ws/robusttail/experiments/run_scripts")
os.environ['R_HOME'] = sys.executable.replace('bin/python', 'lib/R')
import tail_probability.benchmark_tail_probability_estimation as btpe
import tail_probability.tail_probability_estimation as tpe


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
        an_event = raw_ndk[num_iter*5:num_iter*5+5]
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
    if method == 'opt':
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

if __name__ == "__main__":
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
        "bootstrapping_size": bootstrapping_size
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
    table_raw = []
    # set random seed for reproducibility
    np.random.seed(random_state)
    for method in ['opt', 'pot_bt', 'pl', 'bayesian', 'pwm']:
        for region in regions:
            input_data = df.loc[df['location'] == region, 'Mw']
            quantiles = [0.95, 0.99, 0.995]
            for quantile in quantiles:
                if os.path.exists(f'/Users/sam/ws/robusttail/run_realdata_result/real_data_bootstrap_{method}_{region}_{quantile}.csv'):
                    print(f"Skipping {region} {quantile} {method} because it already exists")
                    continue 
                print(f"Running {region} {quantile} {method}")
                lhs = np.quantile(input_data, q=quantile)

                pool_param_list = [(method, 
                                    np.random.choice(input_data, size=len(input_data), replace=True), 
                                    lhs, 
                                    right_end_point_objective, 
                                    kwargs, alpha) 
                                    for _ in range(100)]

                with Pool() as pool:
                    bootstrap_results = pool.map(_parallel_run, pool_param_list)
                # add region, quantile to the bootstrap results
                bootstrap_results = [[method, region, quantile] + bootstrap_result for bootstrap_result in bootstrap_results]
                # save as csv. 
                print(f"Saving {region} {quantile} {method}")
                bootstrap_results_df = pd.DataFrame(bootstrap_results, columns=['method', 'region', 'quantile', 'lb', 'ub'])
                bootstrap_results_df.to_csv(f'/Users/sam/ws/robusttail/run_realdata_result/real_data_bootstrap_{method}_{region}_{quantile}.csv', index=False)
                print("lb and ub are (mean):")
                print(bootstrap_results_df[['lb', 'ub']].mean().to_frame().T)

            
