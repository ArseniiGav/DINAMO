import argparse

from tqdm import tqdm
from dinamo.configs import data_configs
from dinamo.datasets import ToyDataset

parser = argparse.ArgumentParser(description='Produce the toy datasets with different seeds')
parser.add_argument("--single_dataset", type=int, default=data_configs['single_dataset'], 
                    help='Process only a single dataset? Either 1 or 0')
parser.add_argument("--dataset_seed", type=int, default=data_configs['dataset_seed'], 
                    help='Seed of the toy dataset')
parser.add_argument("--n_datasets", type=int, 
                    default=data_configs['n_datasets'],
                    help='Number of different toy datasets')
parser.add_argument("--n_runs", type=int, 
                    default=data_configs['n_runs'],
                    help='Number of runs in each dataset')
parser.add_argument("--n_anomalous_runs", type=int, 
                    default=data_configs['n_anomalous_runs'],
                    help='Number of bad runs in each dataset')
parser.add_argument("--min_stats", type=int, 
                    default=data_configs['min_stats'],
                    help='Minimal event statistics per run')
parser.add_argument("--max_stats", type=int, 
                    default=data_configs['max_stats'],
                    help='Maximal event statistics per run')
parser.add_argument("--mu_slow_change_period", type=int, 
                    default=data_configs['mu_slow_change_period'],
                    help="""Period of the slow change in μ of the gaussian distribution. 
                            Follows sine function with this period.""")
parser.add_argument("--mu_rapid_changes_shift_lims", nargs='+', type=float, 
                    default=data_configs['mu_rapid_changes_shift_lims'],
                    help='Left and right limits of the uniform distribution for the μ rapid changes shift')
parser.add_argument("--sigma_rapid_changes_shift_lims", nargs='+', type=float, 
                    default=data_configs['sigma_rapid_changes_shift_lims'],
                    help='Left and right limits of the uniform distribution for the σ rapid changes shift')
parser.add_argument("--mu_anomaly_shift_lims", nargs='+', type=float, 
                    default=data_configs['mu_anomaly_shift_lims'],
                    help='Left and right limits of the uniform distribution for the μ anomaly changes shift')
parser.add_argument("--sigma_anomaly_shift_lims", nargs='+', type=float, 
                    default=data_configs['sigma_anomaly_shift_lims'], 
                    help='Left and right limits of the uniform distribution for the σ anomaly changes shift')
parser.add_argument("--rapid_change_p", type=float, 
                    default=data_configs['rapid_change_p'],
                    help='Probability to have a rapid change for each run')
parser.add_argument("--binom_p", type=float, 
                    default=data_configs['binom_p'],
                    help="""Probability parameter of a binomial distribution used to apply additional uncertainty.
                            If set to 0.0, no additional uncertainty will be applied""")
parser.add_argument("--anomaly_p", type=float, 
                    default=data_configs['anomaly_p'],
                    help='Probability to have an anomaly in μ/σ parameters of the gaussian distribution')
parser.add_argument("--base_path_to_data", type=str, 
                    default=data_configs['base_path_to_data'], 
                    help=('Base path to a folder to save the output '
                          '.npz files'))
args = parser.parse_args()

toy_dataset = ToyDataset(
    n_runs=args.n_runs,
    n_anomalous_runs=args.n_anomalous_runs,
    min_stats=args.min_stats,
    max_stats=args.max_stats,
    mu_slow_change_period=args.mu_slow_change_period,
    rapid_change_p=args.rapid_change_p,
    mu_rapid_changes_shift_lims=args.mu_rapid_changes_shift_lims,
    sigma_rapid_changes_shift_lims=args.sigma_rapid_changes_shift_lims,
    anomaly_p=args.anomaly_p,
    mu_anomaly_shift_lims=args.mu_anomaly_shift_lims,
    sigma_anomaly_shift_lims=args.sigma_anomaly_shift_lims,
    binom_p=args.binom_p,
    base_path_to_data=args.base_path_to_data,
)

if args.single_dataset:
    # Generate and save datasets with a specific seed
    toy_dataset.build_and_save_dataset(seed=args.dataset_seed)
else:
    # Generate and save datasets with different seeds
    for seed in tqdm(range(args.n_datasets)):
        toy_dataset.build_and_save_dataset(seed=seed)
