import optuna
import numpy as np
from sklearn import metrics

from .dinamo_s import DinamoS
from ...metrics import calculate_metrics_with_bootstrap

def objective(
        x: np.ndarray,
        y: np.ndarray,
        trial: optuna.trial.Trial, 
        st_shifter: DinamoS,
        metric_mode: str,
        **kwargs
    ) -> float:
    """
    Objective function for hyperparameter optimization using Optuna.

    Parameters:
    trial (optuna.trial.Trial): A trial object for suggesting hyperparameters.
    - x (np.ndarray): The array with the histograms that represent the input data.
    - y (np.ndarray): The array with the labels of the runs.
    - st_shifter (StShifter): The object of the StShifter class.
    - metric_mode (str): Mode of evaluation, either "auc" or "wba".
    - **kwargs: Additional keyword arguments for specific options:
        - For "wba" mode:
            - β (float): Weight for the positive class in the weighted balanced accuracy.
            - n_metrics_bootstrap_trials (int): Number of bootstrap trials.

    Returns:
    float: The evaluation metric to be optimized.
    """
    # Suggest a value for α using Optuna
    α = trial.suggest_float("α", 0., 1.)
    results = st_shifter.run(α, x, y)
    χ2 = results['χ2']
    log_red_χ2 = np.log(χ2 / x.shape[1])

    if metric_mode == "auc":
        fpr, tpr, _ = metrics.roc_curve(y, log_red_χ2)
        return np.abs(metrics.auc(fpr, tpr) - 0.5) + 0.5
    elif metric_mode == "wba":
        β = kwargs['β']
        n_metrics_bootstrap_trials = kwargs['n_metrics_bootstrap_trials']
        
        T_values = np.linspace(min(log_red_χ2), max(log_red_χ2), 200)
        n_samples = len(log_red_χ2)
        _, _, balanced_accuracy_weighed_list = calculate_metrics_with_bootstrap(
            y, log_red_χ2, T_values, n_samples, n_metrics_bootstrap_trials, β)
        
        median_balanced_accuracy_weighed = np.median(balanced_accuracy_weighed_list, axis=0)
        return np.max(median_balanced_accuracy_weighed)

def hyperopt_dinamo_s(
        x: np.ndarray,
        y: np.ndarray, 
        dinamo_s: DinamoS,
        metric_mode: str, 
        n_sts_hpopt_trials: int, 
        sampler_seed: int, 
        **kwargs
    ) -> float:
    """
    Run hyperparameter optimization for the standard algorithm using Optuna.

    Parameters:
    - x (np.ndarray): The array with the histograms that represent the input data.
    - y (np.ndarray): The array with the labels of the runs.
    - dinamo_s (DinamoS): The object of the DinamoS class.
    - metric_mode (str): Mode of evaluation, either "auc" or "wba".
    - n_sts_hpopt_trials (int): Number of trials for hyperparameter optimization.
    - sampler_seed (int): Seed for the sampler.
    - **kwargs: Additional keyword arguments for specific options:
        - For "wba" mode:
            - β (float): Weight for the positive class in the weighted balanced accuracy.
            - n_metrics_bootstrap_trials (int): Number of bootstrap trials.

    Returns:
    float: The best hyperparameter value for α.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(study_name="EWMA", direction="maximize", sampler=sampler)

    if metric_mode == "auc":
        obj_function = lambda trial: objective(
            x, y, trial, dinamo_s, metric_mode)
    elif metric_mode == "wba":
        β = kwargs['β']
        n_metrics_bootstrap_trials = kwargs['n_metrics_bootstrap_trials']
        obj_function = lambda trial: objective(
            x, y, trial, dinamo_s, metric_mode,
            β=β, n_metrics_bootstrap_trials=n_metrics_bootstrap_trials)

    study.optimize(obj_function, n_trials=n_sts_hpopt_trials)
    trial = study.best_trial
    α = trial.params['α']
    return α
