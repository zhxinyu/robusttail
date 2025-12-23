import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.stats import genpareto

import sys
import os
sys.path.append("/Users/sam/ws/robusttail")
sys.path.append("/Users/sam/ws/robusttail/experiments/run_scripts")
os.environ['R_HOME'] = sys.executable.replace('bin/python', 'lib/R')
import tail_probability.benchmark_tail_probability_estimation as btpe
import tail_probability.tail_probability_estimation as tpe


def get_data():
    input_data = genpareto.rvs(size=500, c=-0.01, loc=0, scale=1)
    return input_data

def _parallel_run(pool_param: tuple):
    input_data, lhs, right_end_point_objective, kwargs = pool_param
    results = tpe.estimate_tail_probability_D2_chi2_only(input_data=input_data, 
                                                        left_end_point_objective=lhs, 
                                                        right_end_point_objective=right_end_point_objective, 
                                                        **kwargs)
    return results

if __name__ == "__main__":
    input_data = get_data()

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
    }

    np.random.seed(random_state)
    quantiles = ["0.95", "0.99", "0.995"]
    right_end_points = [1.90, 1.91, 1.93, 1.95, 1.98, 2, 2.01, 2.03, 2.05, 2.08, 2.10, np.inf]
    c = - 0.5
    for quantile in quantiles:
        raw_results = []
        for right_endpoint in right_end_points:
            lhs = genpareto.ppf(q=float(quantile),  c=c, loc=0, scale=1)
            pool_param_list = [(genpareto.rvs(size=500, c=c, loc=0, scale=1), lhs, right_end_point_objective, kwargs | 
                                {"right_endpoint": right_endpoint}) for _ in range(200)]
            with Pool() as pool:
                bootstrap_results = pool.map(_parallel_run, pool_param_list)
            raw_result = [[float(quantile), right_endpoint, idx] + bootstrap_result for idx, bootstrap_result in enumerate(bootstrap_results)]
            raw_results.extend(raw_result)
            print(pd.DataFrame(raw_result).mean(axis=0).to_frame().T)
            df = pd.DataFrame(raw_result, columns=['quantile', 'right_endpoint', 'bootstrap_idx', 'lb', 'ub'])

        bootstrap_results_df = pd.DataFrame(raw_results, columns=['quantile', 'right_endpoint', 'bootstrap_idx', 'lb', 'ub'])
        bootstrap_results_df.to_csv(f'/Users/sam/ws/robusttail/run_realdata_result/real_bound_support_{quantile}.csv', index=False)

    # read all csv files in the directory and concatenate them into one csv file
    df_list = []
    for quantile in ["0.95", "0.99", "0.995"]:
        df = pd.read_csv(f"/Users/sam/ws/robusttail/run_realdata_result/real_bound_support_{quantile}.csv")
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    df.to_csv(f"/Users/sam/ws/robusttail/run_realdata_result/real_bound_support.csv", index=False)


    # analyze and plot the results

    # read all csv files in the directory
    df_list = []
    for quantile in ["0.95", "0.99", "0.995"]:
        df = pd.read_csv(f"/Users/sam/ws/robusttail/run_realdata_result/real_bound_support_{quantile}.csv")
        df_list.append(df)
    df = pd.concat(df_list, axis=0)
    # mask = (df['lb'] == 0) & (df['ub'] == 0)
    # df.loc[mask, 'lb'] = 1
    # df.loc[mask, 'ub'] = 0

    grouped = df.groupby(by=["quantile", "right_endpoint"])

    coverage_rate_mean = grouped.apply(
        lambda group: np.mean(group['lb'].le(1-group.name[0]) & \
                            group['ub'].ge(1-group.name[0])),
                            include_groups=False
    )

    coverage_rate_std = grouped.apply(
        lambda group: np.std(group['lb'].le(1-group.name[0]) & \
                            group['ub'].ge(1-group.name[0])),
                            include_groups=False
    )
    coverage_rate_count = grouped['lb'].count()
    coverage_rate_interval_length = 1.96 * coverage_rate_std / np.sqrt(coverage_rate_count)

    lb_mean = grouped['lb'].mean()
    lb_std = grouped['lb'].std()
    lb_count = grouped['lb'].count()  # number of non-NA values
    lb_interval_length = 1.96 * lb_std / np.sqrt(lb_count)  # z_0.975 * sigma/sqrt(n)

    ub_mean = grouped['ub'].mean()
    ub_std = grouped['ub'].std()
    ub_count = grouped['ub'].count()  # number of non-NA values
    ub_interval_length = 1.96 * ub_std / np.sqrt(ub_count)  # z_0.975 * sigma/sqrt(n)

    width_mean = grouped.apply(lambda group: np.mean(group['ub'] - group['lb']), include_groups=False)
    width_std = grouped.apply(lambda group: np.std(group['ub'] - group['lb']), include_groups=False)
    width_count = grouped['ub'].count()  # number of non-NA values
    width_interval_length = 1.96 * width_std / np.sqrt(width_count)  # z_0.975 * sigma/sqrt(n)


    ## formatting  -> {:.2E} and then (mean, interval_length)
    coverage_rate_mean = coverage_rate_mean.map('{:.2f}'.format)
    coverage_rate_interval_length = coverage_rate_interval_length.map('{:.2f}'.format)
    srs_coverage_rate = pd.Series(zip(coverage_rate_mean, coverage_rate_interval_length), index=coverage_rate_mean.index, name='coverage_rate')

    lb_mean = lb_mean.map('{:.2E}'.format)
    lb_interval_length = lb_interval_length.map('{:.2E}'.format)
    srs_lb = pd.Series(zip(lb_mean, lb_interval_length), index=lb_mean.index, name='lb')

    ub_mean = ub_mean.map('{:.2E}'.format)
    ub_interval_length = ub_interval_length.map('{:.2E}'.format)
    srs_ub = pd.Series(zip(ub_mean, ub_interval_length), index=ub_mean.index, name='ub')

    width_mean = width_mean.map('{:.2E}'.format)
    width_interval_length = width_interval_length.map('{:.2E}'.format)
    srs_width = pd.Series(zip(width_mean, width_interval_length), index=width_mean.index, name='width')

    df_analysis = pd.DataFrame({'lb': srs_lb, 'ub': srs_ub, 'width': srs_width, "cp": srs_coverage_rate})

    def number_to_latex(x, format: str) -> str:
        if format == "D":
            if isinstance(x, float):
                return "${:.2f}$".format(x)
            else:
                return "${} (\\pm {})$".format(x[0], x[1])
        elif format == "E":
            return "${}".format(x[0].replace("E-0", "E-").replace("E", "\\times 10^{") + "}") + \
                "(\\pm {})$".format(x[1].replace("E-0", "E-").replace("E", "\\times 10^{") + "}")
        else:
            raise ValueError(f"Invalid format: {format}")
        
    def get_table_latex(df_analysis):
        df_analysis = df_analysis.reset_index()
        drop_rows = df_analysis.right_endpoint.isin([1.9, 2.01])
        df_analysis = df_analysis[~drop_rows]

        for row in df_analysis.itertuples():
            row_in_latex = " & ".join([number_to_latex(row.right_endpoint, 'D'), number_to_latex(row.cp, 'D'), number_to_latex(row.lb, 'E'),
                        number_to_latex(row.ub, 'E'), number_to_latex(row.width, 'E')])
            row_in_latex = " & " + row_in_latex + " \\\\"
            row_in_latex = row_in_latex.replace("inf", "\\infty")
            if row.right_endpoint == 1.91:
                row_in_latex =f"\\multirow{{10}}{{*}}{{$P(X\\geq q_{{{row.quantile}}})$}}" + row_in_latex
            if row.right_endpoint == np.inf:
                row_in_latex += "\\hline"
            print(row_in_latex)

    get_table_latex(df_analysis)

    import matplotlib.pyplot as plt
    import numpy as np

    # Helper to extract the mean and interval length from the tuple string
    def extract_mean_and_interval(s):
        try:
            return float(s[0]), float(s[1])
        except Exception:
            return np.nan, np.nan

    df_plot = df_analysis.copy()
    for col in ['lb', 'ub', 'width', 'cp']:
        df_plot[f'{col}_mean'], df_plot[f'{col}_interval'] = zip(*df_plot[col].apply(extract_mean_and_interval))

    df_plot = df_plot.reset_index()
    inf_replacement = 2.12
    df_plot['right_end_point_numeric'] = df_plot['right_endpoint'].replace(np.inf, inf_replacement).astype(float)
    drop_rows = df_plot.right_endpoint.isin([1.9, 2.01])
    df_plot = df_plot[~drop_rows]

    quantiles = ["0.95", "0.99", "0.995"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), sharey='row')

    for i, q in enumerate(quantiles):
        q = float(q)

        ax = axes[0, i]
        data = df_plot[df_plot['quantile'] == q].sort_values('right_end_point_numeric')
        # Plot mean values with error bars (confidence intervals)
        ax.errorbar(data['right_end_point_numeric'], data['cp_mean'], 
                    yerr=data['cp_interval'], label='Coverage Rate', marker='o', capsize=5, capthick=2, color='blue')
        
        # Add horizontal line 
        ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.7, linewidth=2, label="1-$\\alpha$")
        
        if i == 0:
            ax.set_title(f'Coverage Probability for $P(X \\geq q_{{{q}}})$')
        else:
            ax.set_title(f'$P(X \\geq q_{{{q}}})$')
        if i == 2:
            ax.legend(loc='lower right', fontsize=12) # larger font
        ax.grid(True)
        
        # Replace inf with a readable value
        xticks = data['right_end_point_numeric'].replace(np.inf, inf_replacement)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) if x != inf_replacement else '∞' for x in xticks], fontsize=12)

        # add 0.95 to y tick labels and make it larger
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.95, 1])
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)

        # Bold the x = 2 tick label
        for j, tick in enumerate(ax.get_xticklabels()):
            if tick.get_text() == '2.0':
                tick.set_weight('bold')

        ax = axes[1, i]
        data = df_plot[df_plot['quantile'] == q].sort_values('right_end_point_numeric')
        # Plot mean values with error bars (confidence intervals)
        ax.errorbar(data['right_end_point_numeric'], data['lb_mean'], 
                    yerr=data['lb_interval'], label='Lower Bound', marker='o', capsize=5, capthick=2, color='orange')
        ax.errorbar(data['right_end_point_numeric'], data['ub_mean'], 
                    yerr=data['ub_interval'], label='Upper Bound', marker='^', capsize=5, capthick=2, color='green')
        # ax.errorbar(data['right_end_point_numeric'], data['width_mean'], 
        #             yerr=data['width_interval'], label='Width', marker='o', capsize=5, capthick=2)
        
        # Add horizontal line at y = 1-q
        ax.axhline(y=1-q, color='red', linestyle=':', alpha=0.7, linewidth=2, label="True Value")
        if i == 0:
            ax.set_title(f'Estimated Tail Probability for $P(X \\geq q_{{{q}}})$')
        else:
            ax.set_title(f'$P(X \\geq q_{{{q}}})$')
        if i == 2:
            ax.legend(loc='upper right', fontsize=12) # larger font
        ax.grid(True)
        
        # Replace inf with a readable value
        xticks = data['right_end_point_numeric'].replace(np.inf, inf_replacement)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) if x != inf_replacement else '∞' for x in xticks], fontsize=12)

        # Bold the x = 2 tick label
        for j, tick in enumerate(ax.get_xticklabels()):
            if tick.get_text() == '2.0':
                tick.set_weight('bold')

        # add 0.005 to y tick labels and make it larger
        ax.set_yticks([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)

        ax = axes[2, i]
        data = df_plot[df_plot['quantile'] == q].sort_values('right_end_point_numeric')
        
        # Plot mean values with error bars (confidence intervals)
        ax.errorbar(data['right_end_point_numeric'], data['width_mean'], 
                    yerr=data['width_interval'], label='Width', marker='o', capsize=5, capthick=2, color='purple')
        
        if i == 0:
            ax.set_title(f'Width of Estimated Tail Probability for $P(X \\geq q_{{{q}}})$')
        else:
            ax.set_title(f'$P(X \\geq q_{{{q}}})$')
        if i == 0:
            ax.set_xlabel('Right End Point of Distribution Support', fontsize=12)
        if i == 2:
            ax.legend(loc='upper right', fontsize=12) # larger font
        ax.grid(True)
        
        # Replace inf with a readable value
        xticks = data['right_end_point_numeric'].replace(np.inf, inf_replacement)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) if x != inf_replacement else '∞' for x in xticks], fontsize=12)

        # Bold the x = 2 tick label
        for j, tick in enumerate(ax.get_xticklabels()):
            if tick.get_text() == '2.0':
                tick.set_weight('bold')

        # add 0.005 to y tick labels and make it larger
        ax.set_yticks([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)

    plt.tight_layout()
    # plt.show()
    # save the figure
    plt.savefig('./bounded_support.png', dpi=300, bbox_inches='tight')