import argparse
import numpy as np
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score

from dinamo.configs import models_configs
from dinamo.datasets import load_data
from dinamo.models.standard import DinamoS
from dinamo.utils import threshold_opt

from dinamo.metrics import adaptivity_index
from dinamo.utils import jaccard_distance_evaluation

from dinamo.utils import matplotlib_setup
import matplotlib.pyplot as plt
matplotlib_setup()

def plot_metrics(αs, balanced_acc_mean, balanced_acc_std, jds_mean, jds_std,
                 y3_mean, y3_std, y3_label, y3_ylim, y3_log, bbox_to_anchor_x, 
                 filename):
    fig, ax1 = plt.subplots(figsize=(15, 5))
    
    # Left y-axis: Balanced accuracy
    color1 = 'tab:blue'
    ax1.set_xlabel("α")
    ax1.set_ylabel("Balanced accuracy")
    ax1.plot(αs, balanced_acc_mean, linestyle='-', label="Balanced accuracy", color=color1)
    ax1.fill_between(αs, balanced_acc_mean-balanced_acc_std, balanced_acc_mean+balanced_acc_std, fc=color1, alpha=0.3)
    ax1.tick_params(axis='y')
    ax1.set_ylim(0.675, 1.025)
    ax1.set_xlim(0.0, 1.0)
    
    # Right y-axis 1: Jaccard distance
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel("Jaccard distance", color=color2)
    ax2.plot(αs, jds_mean, linestyle='-', label="Jaccard distance", color=color2)
    ax2.fill_between(αs, jds_mean-jds_std, jds_mean+jds_std, fc=color2, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.05, 0.75)
    
    # Right y-axis 2: Third metric (Adaptivity Index or Steps to Adapt)
    ax3 = ax1.twinx()
    color3 = 'tab:red'
    ax3.spines['right'].set_position(('outward', 80))
    ax3.spines['right'].set_visible(True)
    ax3.spines['right'].set_color(color3)
    ax3.set_ylabel(y3_label, color=color3)
    ax3.plot(αs, y3_mean, linestyle='-', label=y3_label, color=color3)
    ax3.fill_between(αs, y3_mean-y3_std, y3_mean+y3_std, fc=color3, alpha=0.3)
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.set_yscale("log" if y3_log else "linear")
    ax3.set_ylim(*y3_ylim)
    
    # Add text label
    ax1.text(0.1, 0.91, "Continual regime", color="black",
             bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
             fontsize=14, ha="center", va="center", transform=ax1.transAxes)
    
    # Legend
    lines, labels = [], []
    for ax in [ax1, ax2, ax3]:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    fig.legend(lines, labels, loc="upper center", ncol=3, bbox_to_anchor=(bbox_to_anchor_x, 0.925))
    
    fig.tight_layout()
    fig.savefig(filename)

parser = argparse.ArgumentParser(
    description="""Run a scan over the alpha parameter for the DinamoS model on different datasets.
                   Compute the metrics (balanced accuracy, Jaccard distance, adaptivity index, and number of steps to adapt), 
                   then visualize and save the final plots.""")
parser.add_argument("--n_datasets", type=int, default=200,
                     help='Number of different toy datasets')
args = parser.parse_args()

n_datasets = args.n_datasets
base_path_to_data = "./"
n_historical = models_configs["n_historical"]
adaptivity_patience = models_configs['adaptivity_patience']

αs = np.arange(0.0, 1.0, 0.01)

# define the model
dinamo_s = DinamoS(stat_debiasing=False)

balanced_accuracy_scores_historical_all = []
balanced_accuracy_scores_continual_all = []
n_steps_to_adapt_historical_all = []
n_steps_to_adapt_continual_all = []
adaptivity_indices_historical_all = []
adaptivity_indices_continual_all = []
jds_historical_all = []
jds_continual_all = []

for dataset_seed in tqdm(range(n_datasets)):
    # load the data
    dataset = load_data(seed=dataset_seed, base_path_to_data=base_path_to_data)
    x_data, y_data = dataset['x'], dataset['y']
    μ, σ = dataset['mu'], dataset['sigma']
    inds_for_rapid_changes = np.union1d(
        dataset['inds_for_rapid_changes_μ'], 
        dataset['inds_for_rapid_changes_σ']
    )
    inds_for_rapid_changes_historical = inds_for_rapid_changes[inds_for_rapid_changes < n_historical]
    inds_for_rapid_changes_continual = inds_for_rapid_changes[inds_for_rapid_changes >= n_historical]
    x_historical = x_data[:n_historical, :]
    y_historical = y_data[:n_historical]

    dataset_inputs = dataset['inputs'].item()
    n_bins = dataset_inputs['n_bins']
    n_runs = dataset_inputs['n_runs']
    bin_centers = dataset_inputs['bin_centers']
    bin_edges = dataset_inputs['bin_edges']

    balanced_accuracy_scores_historical = []
    balanced_accuracy_scores_continual = []
    n_steps_to_adapt_historical = []
    n_steps_to_adapt_continual = []
    adaptivity_indices_historical = []
    adaptivity_indices_continual = []
    jds_historical = []
    jds_continual = []

    for α in tqdm(αs, leave=False):
        # find optimal threshold using historical data
        results = dinamo_s.run(α, x_historical, y_historical)
        log_red_χ2 = np.log(results['χ2'] / n_bins)
        threshold_best = threshold_opt(
            y_historical, log_red_χ2,
            approach_type="standard_approach",
            dataset_seed=dataset_seed, plot=False, 
            base_path_to_data=base_path_to_data)

        results = dinamo_s.run(α, x_data, y_data)
        χ2, pull, references, x, σ_references, σ_p_x = results.values()
        log_red_χ2 = np.log(χ2 / n_bins)

        ## Calculate balanced accuracy
        y_predict = (log_red_χ2 > threshold_best).astype(int)

        y_predict_historical = y_predict[:n_historical]
        y_historical = y_data[:n_historical]
        balanced_accuracy_scores_historical.append(
            balanced_accuracy_score(y_historical, y_predict_historical))

        y_predict_continual = y_predict[n_historical:]
        y_continual = y_data[n_historical:]
        balanced_accuracy_scores_continual.append(
            balanced_accuracy_score(y_continual, y_predict_continual))

        ## Calculate adaptivity metrics
        nsteps_to_adapt_historical, adaptivity_index_historical = adaptivity_index(
            log_red_χ2, y_data, inds_for_rapid_changes_historical, threshold_best, adaptivity_patience)
        n_steps_to_adapt_historical.append(nsteps_to_adapt_historical)
        adaptivity_indices_historical.append(adaptivity_index_historical)

        nsteps_to_adapt_continual, adaptivity_index_continual = adaptivity_index(
            log_red_χ2, y_data, inds_for_rapid_changes_continual, threshold_best, adaptivity_patience)
        n_steps_to_adapt_continual.append(nsteps_to_adapt_continual)
        adaptivity_indices_continual.append(adaptivity_index_continual)

        ## Jaccard distance for uncertainty
        jds, _ = jaccard_distance_evaluation(
            bin_centers=bin_centers,
            data_μ=μ, 
            data_σ=σ, 
            x=x, 
            y=y_data, 
            references=references, 
            σ_references=σ_references, 
            n_historical=n_historical,
        )
        jd_historical, jd_continual = jds
        jds_historical.append(jd_historical)
        jds_continual.append(jd_continual)

    balanced_accuracy_scores_historical_all.append(balanced_accuracy_scores_historical)
    balanced_accuracy_scores_continual_all.append(balanced_accuracy_scores_continual)
    n_steps_to_adapt_historical_all.append(n_steps_to_adapt_historical)
    n_steps_to_adapt_continual_all.append(n_steps_to_adapt_continual)
    adaptivity_indices_historical_all.append(adaptivity_indices_historical)
    adaptivity_indices_continual_all.append(adaptivity_indices_continual)
    jds_historical_all.append(jds_historical)
    jds_continual_all.append(jds_continual)

balanced_accuracy_scores_historical_all = np.array(balanced_accuracy_scores_historical_all)
balanced_accuracy_scores_continual_all = np.array(balanced_accuracy_scores_continual_all)
n_steps_to_adapt_historical_all = np.array(n_steps_to_adapt_historical_all)
n_steps_to_adapt_continual_all = np.array(n_steps_to_adapt_continual_all)
adaptivity_indices_historical_all = np.array(adaptivity_indices_historical_all)
adaptivity_indices_continual_all = np.array(adaptivity_indices_continual_all)
jds_historical_all = np.array(jds_historical_all)
jds_continual_all = np.array(jds_continual_all)

balanced_accuracy_scores_historical_all_mean = balanced_accuracy_scores_historical_all.mean(axis=0)
balanced_accuracy_scores_continual_all_mean = balanced_accuracy_scores_continual_all.mean(axis=0)
n_steps_to_adapt_historical_all_mean = n_steps_to_adapt_historical_all.mean(axis=0)
n_steps_to_adapt_continual_all_mean = n_steps_to_adapt_continual_all.mean(axis=0)
adaptivity_indices_historical_all_mean = adaptivity_indices_historical_all.mean(axis=0)
adaptivity_indices_continual_all_mean = adaptivity_indices_continual_all.mean(axis=0)
jds_historical_all_mean = jds_historical_all.mean(axis=0)
jds_continual_all_mean = jds_continual_all.mean(axis=0)

balanced_accuracy_scores_historical_all_std = balanced_accuracy_scores_historical_all.std(axis=0)
balanced_accuracy_scores_continual_all_std = balanced_accuracy_scores_continual_all.std(axis=0)
n_steps_to_adapt_historical_all_std = n_steps_to_adapt_historical_all.std(axis=0)
n_steps_to_adapt_continual_all_std = n_steps_to_adapt_continual_all.std(axis=0)
adaptivity_indices_historical_all_std = adaptivity_indices_historical_all.std(axis=0)
adaptivity_indices_continual_all_std = adaptivity_indices_continual_all.std(axis=0)
jds_historical_all_std = jds_historical_all.std(axis=0)
jds_continual_all_std = jds_continual_all.std(axis=0)

plot_metrics(αs, balanced_accuracy_scores_continual_all_mean, balanced_accuracy_scores_continual_all_std,
             jds_continual_all_mean, jds_continual_all_std,
             adaptivity_indices_continual_all_mean, adaptivity_indices_continual_all_std,
             "Adaptivity index", (1e-2, 1e2), True, 0.5,
             './plots/standard_approach/metrics_vs_alpha_adi_continual_w_stds.pdf')

plot_metrics(αs, balanced_accuracy_scores_continual_all_mean, balanced_accuracy_scores_continual_all_std,
             jds_continual_all_mean, jds_continual_all_std,
             n_steps_to_adapt_continual_all_mean, n_steps_to_adapt_continual_all_std,
             "Number of steps to adapt", (1e-1, 1e2), True, 0.525,
             './plots/standard_approach/metrics_vs_alpha_nsteps_continual_w_stds.pdf')
