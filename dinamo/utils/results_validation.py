import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import balanced_accuracy_score

from ..datasets import load_data
from ..configs import models_configs, plots_configs
from ..metrics import adaptivity_index
from .plot_utils import plot_results_hist, plot_results_scatter
from .plot_utils import plot_uncertainty_evaluation, plot_uncertainty_evaluation_continual
from .plot_utils import plot_refs_examples
from .jaccard_distance_evaluation import jaccard_distance_evaluation

def results_validation(
        dataset_seed: int, 
        approach_type: str, 
        base_path_to_data: str,
        base_path_to_results: str,
        base_path_to_plots: str
    ) -> None:
    n_historical = models_configs['n_historical']
    adaptivity_patience = models_configs['adaptivity_patience']

    bad_runs_color = plots_configs["bad_runs_color"]
    good_runs_color = plots_configs["good_runs_color"]
    xllim, xrlim = plots_configs['xllim'], plots_configs['xrlim']

    dataset = load_data(seed=dataset_seed, base_path_to_data=base_path_to_data)
    y = dataset['y']
    μ, σ = dataset['mu'], dataset['sigma']
    inds_for_rapid_changes = np.union1d(dataset['inds_for_rapid_changes_μ'], dataset['inds_for_rapid_changes_σ'])
    inds_for_rapid_changes_historical = inds_for_rapid_changes[inds_for_rapid_changes < n_historical]
    inds_for_rapid_changes_continual = inds_for_rapid_changes[inds_for_rapid_changes >= n_historical]

    dataset_inputs = dataset['inputs'].item()
    n_bins = dataset_inputs['n_bins']
    n_runs = dataset_inputs['n_runs']
    bin_centers = dataset_inputs['bin_centers']
    bin_edges = dataset_inputs['bin_edges']

    results_path = f'{base_path_to_results}/results/{approach_type}/results_{dataset_seed}.npz'
    results = np.load(results_path, allow_pickle=True)
    χ2, pull, references, x, σ_references, σ_p_x = results.values()
    log_red_χ2 = np.log(χ2 / n_bins)

    hpopt_path = f'{base_path_to_results}/results/{approach_type}/hyperparameters_{dataset_seed}.npz'
    hpopt = np.load(hpopt_path, allow_pickle=True)
    *_, threshold_best = hpopt.values()

    # Calculate metrics
    ## define the model's outputs
    y_predict = (log_red_χ2 > threshold_best).astype(int)
    y_predict_historical = y_predict[:n_historical]
    y_historical = y[:n_historical]
    y_predict_continual = y_predict[n_historical:]
    y_continual = y[n_historical:]
    df_confusion_matrix_continual = pd.DataFrame(
        confusion_matrix(y_continual, y_predict_continual))

    ## Calculate adaptivity metrics
    nsteps_to_adapt_historical, adaptivity_index_historical = adaptivity_index(
        log_red_χ2, y, inds_for_rapid_changes_historical, threshold_best, adaptivity_patience)
    nsteps_to_adapt_continual, adaptivity_index_continual = adaptivity_index(
        log_red_χ2, y, inds_for_rapid_changes_continual, threshold_best, adaptivity_patience)

    ## Jaccard distance for uncertainty
    jds, pdfs_to_plot = jaccard_distance_evaluation(
        bin_centers=bin_centers,
        data_μ=μ, 
        data_σ=σ, 
        x=x, 
        y=y, 
        references=references, 
        σ_references=σ_references, 
        n_historical=n_historical,
    )
    jd_historical, jd_continual = jds

    df_metrics = dict(
        balanced_accuracy_historical=[],
        accuracy_historical=[],
        specificity_historical=[],
        sensitivity_historical=[],
        precision_historical=[],
        jaccard_distance_historical=[],
        nsteps_to_adapt_historical=[],
        adaptivity_index_historical=[],

        balanced_accuracy_continual=[],
        accuracy_continual=[],
        specificity_continual=[],
        sensitivity_continual=[],
        precision_continual=[],
        jaccard_distance_continual=[],
        nsteps_to_adapt_continual=[],
        adaptivity_index_continual=[],
    )

    ## balanced accuracy, accuracy, specificity, sensitivity, precision
    df_metrics['balanced_accuracy_historical'].append(
        balanced_accuracy_score(y_historical, y_predict_historical))
    df_metrics['accuracy_historical'].append(
        accuracy_score(y_historical, y_predict_historical))
    df_metrics['specificity_historical'].append(
        recall_score(y_historical, y_predict_historical, pos_label=0))
    df_metrics['sensitivity_historical'].append(
        recall_score(y_historical, y_predict_historical, pos_label=1))
    df_metrics['precision_historical'].append(
        precision_score(y_historical, y_predict_historical))
    df_metrics['jaccard_distance_historical'].append(jd_historical)
    df_metrics['nsteps_to_adapt_historical'].append(nsteps_to_adapt_historical) 
    df_metrics['adaptivity_index_historical'].append(adaptivity_index_historical)  

    df_metrics['balanced_accuracy_continual'].append(
        balanced_accuracy_score(y_continual, y_predict_continual))
    df_metrics['accuracy_continual'].append(
        accuracy_score(y_continual, y_predict_continual))
    df_metrics['specificity_continual'].append(
        recall_score(y_continual, y_predict_continual, pos_label=0))
    df_metrics['sensitivity_continual'].append(
        recall_score(y_continual, y_predict_continual, pos_label=1))
    df_metrics['precision_continual'].append(
        precision_score(y_continual, y_predict_continual))
    df_metrics['jaccard_distance_continual'].append(jd_continual)
    df_metrics['nsteps_to_adapt_continual'].append(nsteps_to_adapt_continual)
    df_metrics['adaptivity_index_continual'].append(adaptivity_index_continual)

    df_metrics = pd.DataFrame(df_metrics)
    df_metrics.to_csv(
        f'{base_path_to_results}/results/{approach_type}/metrics/metrics_{dataset_seed}.csv', 
        index=False
    )

    # Plot results
    plot_results_hist(
        dataset_seed, 
        df_confusion_matrix_continual,
        y, 
        log_red_χ2, 
        n_historical, 
        xllim, 
        xrlim, 
        good_runs_color, 
        bad_runs_color, 
        threshold_best,
        base_path_to_plots,
        approach_type,
        save=True
    )   

    plot_results_scatter(
        dataset_seed,
        n_runs, 
        y, 
        log_red_χ2, 
        n_historical, 
        xllim,
        xrlim,
        good_runs_color, 
        bad_runs_color, 
        threshold_best, 
        base_path_to_plots,
        approach_type,
        save=True
    )

    plot_uncertainty_evaluation(
        dataset_seed, 
        bin_centers, 
        *pdfs_to_plot, 
        *jds,
        good_runs_color, 
        base_path_to_plots, 
        approach_type,
        save=True
    )

    plot_uncertainty_evaluation_continual(
        dataset_seed, 
        bin_centers, 
        *pdfs_to_plot[4:], 
        jd_continual,
        good_runs_color, 
        base_path_to_plots, 
        approach_type,
        save=True
    )

    ## refs examples
    N = 5
    colors = ["royalblue", "royalblue", "darkred", "darkred"]
    labels = ["Good", "Misclass. good", "Bad", "Misclass. bad"]
    rng = np.random.default_rng(22)
    indexes = np.arange(n_runs)

    ### Define conditions
    conditions = {
        "good_runs": np.logical_and(y_predict == 0, y == 0),
        "misc_good_runs": np.logical_and(y_predict == 1, y == 0),
        "bad_runs": np.logical_and(y_predict == 1, y == 1),
        "misc_bad_runs": np.logical_and(y_predict == 0, y == 1),
    }

    while N > 0: # if less than N examples in one of the categories, reduce N until 1
        try:
            ### Select random indices for each category
            inds_to_plot = {key: rng.choice(cond.sum(), N, replace=False) for key, cond in conditions.items()}
            break
        except ValueError: 
            N -= 1
    if N == 0:
        raise ValueError("No examples in one of the categories to plot")

    #### Extract data into matrices
    references_matrix = np.stack([references[conditions[key]][inds_to_plot[key]] for key in conditions], axis=1)
    log_red_χ2_matrix = np.column_stack([log_red_χ2[conditions[key]][inds_to_plot[key]] for key in conditions])
    pull_matrix = np.stack([pull[conditions[key]][inds_to_plot[key]] for key in conditions], axis=1)
    x_matrix = np.stack([x[conditions[key]][inds_to_plot[key]] for key in conditions], axis=1)
    σ_references_matrix = np.stack([σ_references[conditions[key]][inds_to_plot[key]] for key in conditions], axis=1)
    σ_p_x_matrix = np.stack([σ_p_x[conditions[key]][inds_to_plot[key]] for key in conditions], axis=1)
    indexes_matrix = np.column_stack([indexes[conditions[key]][inds_to_plot[key]] for key in conditions])
    
    plot_refs_examples(
        labels,
        colors,
        bin_edges,
        bin_centers,
        log_red_χ2_matrix, 
        pull_matrix, 
        x_matrix, 
        σ_p_x_matrix,
        references_matrix, 
        σ_references_matrix,
        indexes_matrix,
        threshold_best, 
        approach_type,
        base_path_to_plots,
        dataset_seed,
        save=True
    )
