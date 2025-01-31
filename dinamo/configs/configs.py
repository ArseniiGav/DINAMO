data_configs = {
    "n_datasets": 200,
    "n_runs": 5000,
    "n_anomalous_runs": 500,
    "min_stats": 2000,
    "max_stats": 20000,
    "mu_slow_change_period": 500,
    "mu_rapid_changes_shift_lims": (0.5, 1.5),
    "sigma_rapid_changes_shift_lims": (0.1, 0.4),
    "mu_anomaly_shift_lims": (0.25, 0.75),
    "sigma_anomaly_shift_lims": (0.05, 0.2),
    "rapid_change_p": 0.005,
    "binom_p": 0.4,
    "anomaly_p": 0.9,
    "base_path_to_data": "./",
    "single_dataset": 0,
    "dataset_seed": 0
}

models_configs = {
    # common configs
    "n_historical": 1000,
    "ba_weight": 1,
    "t_opt": 1,
    "n_resamples": 100,
    "stat_debiasing": 0,
    "debiasing_seed": 22,
    "threshold_default": 1.0,
    "n_metrics_bootstrap_trials": 100,
    "bootstrap_seed": 22,
    "adaptivity_patience": 1,
    "base_path_to_results": "./",

    # Standard model configs
    "alpha_opt": 1,
    "metric_mode": "auc",
    "α_default": 0.5,
    "sampler_seed": 22,
    "n_sts_hpopt_trials": 100,

    # ML model configs
    ## transformer encoder configs
    "n_bins": 100,
    "M": 20,
    "K": 10,
    "d_model": 100,
    "nhead": 10,
    "num_layers": 3,
    "dim_feedforward": 100,
    "dropout": 0.15,

    ## DinamoML configs
    "lr": 5e-4,
    "n_epochs": 1000,
    "early_stopping_patience": 5,
    "training_seed": 22
}

plots_configs = {
    "bad_runs_color": "#C3161B", 
    "good_runs_color": "#1C6AB0",
    "xllim": -3.25,
    "xrlim": 7.5,
    "base_path_to_plots": "./",
}
