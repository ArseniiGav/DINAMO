import numpy as np

from .plot_utils import plot_threshold_opt_results
from ..metrics import calculate_metrics_with_bootstrap
from ..configs import models_configs, plots_configs

def threshold_opt(y, log_red_χ2, approach_type, dataset_seed=0, plot=False, base_path_to_data='../'):
    n_metrics_bootstrap_trials = models_configs["n_metrics_bootstrap_trials"]
    bootstrap_seed = models_configs["bootstrap_seed"]
    ba_weight = models_configs["ba_weight"]
    sensitivity_color = plots_configs["bad_runs_color"]
    specificity_color = plots_configs["good_runs_color"]
    xllim, xrlim = plots_configs['xllim'], plots_configs['xrlim']
    T_values = np.linspace(xllim, xrlim, 1000)
    n_samples = len(log_red_χ2)
    
    sensitivity_list, specificity_list, balanced_accuracy_weighed_list = calculate_metrics_with_bootstrap(
        y, log_red_χ2, T_values, n_samples, n_metrics_bootstrap_trials, ba_weight, bootstrap_seed)
    
    median_balanced_accuracy_weighed = np.quantile(balanced_accuracy_weighed_list, [0.5], axis=0)
    T_value_best = T_values[np.argmax(median_balanced_accuracy_weighed)]

    if plot:
        plot_threshold_opt_results(
            sensitivity_list, specificity_list, balanced_accuracy_weighed_list,
            T_values, xllim, xrlim, dataset_seed, sensitivity_color, specificity_color,
            approach_type, base_path_to_data
    )

    return T_value_best
    