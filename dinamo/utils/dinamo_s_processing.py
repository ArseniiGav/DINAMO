import os

import numpy as np

from ..configs import models_configs
from ..datasets import load_data
from ..models.standard import DinamoS, hyperopt_dinamo_s
from .threshold_opt import threshold_opt

def dinamo_s_processing(
        dataset_seed: int, 
        stat_debiasing: bool, 
        alpha_opt: bool, 
        t_opt: bool, 
        metric_mode: str, 
        base_path_to_data: str, 
        base_path_to_results: str
    ) -> None:
    n_historical = models_configs["n_historical"]
    n_sts_hpopt_trials = models_configs["n_sts_hpopt_trials"]

    dataset = load_data(seed=dataset_seed, base_path_to_data=base_path_to_data)
    x, y = dataset['x'], dataset['y']
    x_historical = x[:n_historical, :]
    y_historical = y[:n_historical]

    dataset_inputs = dataset['inputs'].item()
    n_bins = dataset_inputs['n_bins']

    if stat_debiasing:
        n_resamples = models_configs["n_resamples"]
        debiasing_seed = models_configs["debiasing_seed"]
        dinamo_s = DinamoS(stat_debiasing=stat_debiasing,
                               n_resamples=n_resamples,
                               debiasing_seed=debiasing_seed)
    else:
        dinamo_s = DinamoS(stat_debiasing=stat_debiasing)

    if alpha_opt:
        α_best = hyperopt_dinamo_s(
            x_historical, y_historical, # using historical data only
            dinamo_s,
            metric_mode=metric_mode,
            n_sts_hpopt_trials=n_sts_hpopt_trials,
            sampler_seed=models_configs["sampler_seed"],
        )
    else:
        α_best = models_configs['α_default']

    results = dinamo_s.run(α_best, x, y)

    if t_opt:
        log_red_χ2 = np.log(results['χ2'] / n_bins)
        threshold_best = threshold_opt(
            y_historical, log_red_χ2[:n_historical], # using historical data only
            approach_type="standard_approach",
            dataset_seed=dataset_seed, plot=True, 
            base_path_to_data=base_path_to_data)
    else:
        threshold_best = models_configs['threshold_default']

    np.savez(
        os.path.join(base_path_to_results, 
                     f"results/standard_approach/results_{dataset_seed}.npz"),
        **results
    )
    np.savez(
        os.path.join(base_path_to_results, 
                     f"results/standard_approach/hyperparameters_{dataset_seed}.npz"),
        α_best=α_best,
        threshold_best=threshold_best
    )
