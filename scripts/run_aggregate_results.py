import os
import re
import argparse

import pandas as pd
import numpy as np
from scipy.stats import norm

from dinamo.configs import models_configs, plots_configs
from dinamo.utils import matplotlib_setup, add_hist
import matplotlib.pyplot as plt
matplotlib_setup()

def sort_files_by_number(file_list):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')
    return sorted(file_list, key=extract_number)

def plot_integrated_results(
        bins: np.ndarray,
        counts_historical: np.ndarray, 
        counts_continual: np.ndarray, 
        mean_historical: float,
        mean_continual: float,
        color_historical: str,
        color_continual: str,
        xlabel: str,
        base_path: str,
        approach_type: str,
        plot_name: str,
        ylim: float,
    ) -> None:
    fig, ax = plt.subplots(figsize = (8, 5))
    
    add_hist(ax, counts_historical, bins, True, color_historical, 'Historical regime', alpha=0.8)
    ax.axvline(mean_historical, ls='--', color=color_historical, zorder=-1, linewidth=2.5, alpha=0.8)

    add_hist(ax, counts_continual, bins, True, color_continual, 'Continual regime', alpha=0.8)
    ax.axvline(mean_continual, ls='--', color=color_continual, zorder=-1, linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized number of runs")
    ax.set_ylim(0.0, ylim)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(f'{base_path}/plots/{approach_type}/integrated_results_{plot_name}.pdf')
    plt.close()

def plot_approach_comparison(
        bins: np.ndarray,
        counts_standard: np.ndarray,
        counts_ml: np.ndarray,
        median_standard: float,
        median_ml: float,
        lower_bound_standard: float,
        lower_bound_ml: float,
        upper_bound_standard: float,
        upper_bound_ml: float,
        color_standard: str,
        color_ml: str,
        xlabel: str,
        base_path: str,
        plot_name: str,
        ylim: float
    ) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    add_hist(ax, counts_standard, bins, True, color_standard, 'DINAMO-S', alpha=0.8)
    ax.axvline(median_standard, ls='--', color=color_standard, zorder=-1, linewidth=2.5, alpha=0.8)

    add_hist(ax, counts_ml, bins, True, color_ml, 'DINAMO-ML', alpha=0.8)
    ax.axvline(median_ml, ls='--', color=color_ml, zorder=-1, linewidth=2.5, alpha=0.8)

    ax.text(0.86, 0.765, "Continual regime", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=12, ha="center", va="center", transform=ax.transAxes)

    ax.text(0.225, 0.95, "DINAMO-S, 95% CI: "+ f"({lower_bound_standard:.3f}, {upper_bound_standard:.3f})", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="#ebebeb", alpha=1),  
            fontsize=10, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.225, 0.88, "DINAMO-ML, 95% CI: "+ f"({lower_bound_ml:.3f}, {upper_bound_ml:.3f})", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="#d7b1d7", alpha=1),  
            fontsize=10, ha="center", va="center", transform=ax.transAxes)

    ax.plot([0], [0], color='black', linestyle='--', linewidth=2, label="Median")

    handles, labels = ax.get_legend_handles_labels()                
    legend1 = ax.legend(handles[:2], labels[:2], frameon=1, ncol=1, fontsize=14, loc=(0.7, 0.825), framealpha=1)
    legend2 = ax.legend(handles[2:], labels[2:], frameon=1, ncol=1, fontsize=12, loc=(0.5, 0.9), framealpha=1)
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized number of runs")
    ax.set_ylim(0.0, ylim)
    fig.tight_layout()
    fig.savefig(f'{base_path}/plots/models_comparison_{plot_name}_continual.pdf')
    plt.close()

parser = argparse.ArgumentParser(
    description="""Run results aggregation for both models. Produces .csv tables with different statistics per each metric:
                   mean, standard deviation, median, max, min, etc. Plots the aggregated result per each metric approach by approach
                   and the comparison between the two. Table are saved in the results. The plots are saved in the plots folder.""")
parser.add_argument("--base_path_to_results", type=str, default=models_configs['base_path_to_results'],
                    help="""Base path to a folder to load the models' resulted metrics .csv files.
                            Used to save the aggregated results as well.""")
parser.add_argument("--base_path_to_plots", type=str, default=plots_configs['base_path_to_plots'], 
                    help='Base path to a folder to save the final plots')
args = parser.parse_args()

bad_runs_color = plots_configs["bad_runs_color"]
good_runs_color = plots_configs["good_runs_color"]
accuracy_metrics = [
    "balanced_accuracy_continual", 
    "accuracy_continual", 
    "specificity_continual", 
    "sensitivity_continual", 
    "precision_continual"
]
xlims_to_plot = {
    "balanced_accuracy_continual": [0.875, 1.025],
    "accuracy_continual": [0.875, 1.025],
    "specificity_continual": [0.875, 1.025],
    "sensitivity_continual": [0.85, 1.025],
    "precision_continual": [0.29, 1.0],
    "jaccard_distance_continual": [0.068, 0.21],
    "nsteps_to_adapt_continual": [0.0, 5.0],
    "adaptivity_index_continual": [0.35, 2.1],
}

percentiles = []
for σ in [1, 2, 3]:
    lb, ub = norm(0, 1).cdf(-σ), norm(0, 1).cdf(σ)
    percentiles.append(lb)
    percentiles.append(ub)

results_dict = {}
for approach_type in ["standard_approach", "ml_approach"]:
    all_files = os.listdir(f"{args.base_path_to_results}/results/{approach_type}/metrics/")
    
    # Filter files that match the pattern "metrics_*.csv"
    metrics_df_names = [f for f in all_files if re.match(r"metrics_\d+\.csv$", f)]
    metrics_df_names = sort_files_by_number(metrics_df_names)

    metrics_dfs = []
    for metrics_df_name in metrics_df_names:
        metrics_df = pd.read_csv(f"{args.base_path_to_results}/results/{approach_type}/metrics/{metrics_df_name}")
        metrics_dfs.append(metrics_df)

    metrics_df_all_datasets = pd.concat(metrics_dfs, ignore_index=True)
    metrics_df_all_datasets_describe = metrics_df_all_datasets.describe(
        percentiles=percentiles+[0.025, 0.05, 0.25, 0.75, 0.95, 0.975])
    
    mins = metrics_df_all_datasets_describe.loc['0.1%']
    maxs = metrics_df_all_datasets_describe.loc['99.9%']
    
    metrics_df_all_datasets.to_csv(
        f"{args.base_path_to_results}/results/{approach_type}/metrics/all_metrics_table_{approach_type}.csv", 
        index=True)
    metrics_df_all_datasets_describe.to_csv(
        f"{args.base_path_to_results}/results/{approach_type}/metrics/aggregated_metrics_{approach_type}.csv", 
        index=True)
    
    results_dict[approach_type] = {
        "all_metrics": metrics_df_all_datasets,
        "aggregated_metrics": metrics_df_all_datasets_describe,
        "mins": mins,
        "maxs": maxs,
    }

    for metric_name_historical, metric_name_continual in zip(metrics_df_all_datasets.columns[:8], 
                                                             metrics_df_all_datasets.columns[8:]):
        
        median_historical = metrics_df_all_datasets_describe[metric_name_historical]['50%']
        median_continual = metrics_df_all_datasets_describe[metric_name_continual]['50%']
    
        if "nsteps_to_adapt" in metric_name_continual: 
            xlabel = "Number of steps to adapt"
        else:
            xlabel = " ".join(metric_name_continual.split("_")[:-1]).capitalize()
        plot_name = "_".join(metric_name_continual.split("_")[:-1])
        
        mini = min(mins[metric_name_historical], mins[metric_name_continual])
        maxi = max(maxs[metric_name_historical], maxs[metric_name_continual])
    
        if metric_name_continual in accuracy_metrics:
            mini = max(mini, 0.0)
            maxi = min(maxi, 1.0)
    
        bins = np.linspace(mini, maxi, 40)
    
        counts_historical, _ = np.histogram(metrics_df_all_datasets[metric_name_historical], bins=bins)
        counts_historical = counts_historical / counts_historical.sum()
        
        counts_continual, _ = np.histogram(metrics_df_all_datasets[metric_name_continual], bins=bins)
        counts_continual = counts_continual / counts_continual.sum()
    
        ylim = 1.5 * max(counts_continual.max(), counts_historical.max())

        plot_integrated_results(
            bins, counts_historical, counts_continual,
            median_historical, median_continual,
            bad_runs_color, good_runs_color, xlabel,
            args.base_path_to_plots, approach_type, plot_name, ylim
        )

# Generate comparison plots for both approaches
for metric_name_continual in results_dict["standard_approach"]["all_metrics"].columns[8:]:

    # Extract metrics for standard and ml approaches
    standard_approach_aggregated_metrics = results_dict["standard_approach"]['aggregated_metrics']
    ml_approach_aggregated_metrics = results_dict["ml_approach"]['aggregated_metrics']

    # Extract aggregated metrics for standard and ml approaches
    standard_approach_all_metrics = results_dict["standard_approach"]['all_metrics']
    ml_approach_all_metrics = results_dict["ml_approach"]['all_metrics']

    # Extract mins and maxs for plotting for standard and ml approaches
    standard_approach_mini = results_dict["standard_approach"]['mins']
    ml_approach_mini = results_dict["ml_approach"]['mins']

    standard_approach_maxi = results_dict["standard_approach"]['maxs']
    ml_approach_maxi = results_dict["ml_approach"]['maxs']

    median_standard_continual = standard_approach_aggregated_metrics[metric_name_continual]['50%']
    median_ml_continual = ml_approach_aggregated_metrics[metric_name_continual]['50%']

    lower_bound_standard_continual = standard_approach_aggregated_metrics[metric_name_continual]['2.5%']
    lower_bound_ml_continual = ml_approach_aggregated_metrics[metric_name_continual]['2.5%']

    upper_bound_standard_continual = standard_approach_aggregated_metrics[metric_name_continual]['97.5%']
    upper_bound_ml_continual = ml_approach_aggregated_metrics[metric_name_continual]['97.5%']

    if "nsteps_to_adapt" in metric_name_continual: 
        xlabel = "Number of steps to adapt"
    else:
        xlabel = " ".join(metric_name_continual.split("_")[:-1]).capitalize()
    plot_name = "_".join(metric_name_continual.split("_")[:-1])

    mini, maxi = xlims_to_plot[metric_name_continual]

    bins = np.linspace(mini, maxi, 40)

    counts_standard_continual, _ = np.histogram(standard_approach_all_metrics[metric_name_continual], bins=bins)
    counts_standard_continual = counts_standard_continual / counts_standard_continual.sum()

    counts_ml_continual, _ = np.histogram(ml_approach_all_metrics[metric_name_continual], bins=bins)
    counts_ml_continual = counts_ml_continual / counts_ml_continual.sum()

    ylim = 1.5 * max(counts_standard_continual.max(), counts_ml_continual.max())

    # Plot comparison between standard and ml approaches
    plot_approach_comparison(
        bins, counts_standard_continual, counts_ml_continual,
        median_standard_continual, median_ml_continual,
        lower_bound_standard_continual, lower_bound_ml_continual,
        upper_bound_standard_continual, upper_bound_ml_continual,
        "lightslategrey", "indigo", xlabel, args.base_path_to_plots, 
        plot_name, ylim
    )
