import argparse

from tqdm import tqdm
from dinamo.configs import data_configs, models_configs, plots_configs
from dinamo.utils import results_validation

parser = argparse.ArgumentParser(
    description="""Run results validation (metrics computation and plots production) of a model
                   on the toys dataset with different seeds""")
parser.add_argument("--approach_type", type=str, default='standard_approach',
                    help='Type of the approach: standard_approach or ml_approach')
parser.add_argument("--single_dataset", type=int, default=data_configs['single_dataset'], 
                    help='Process only on a single dataset? Either 1 or 0')
parser.add_argument("--dataset_seed", type=int, default=data_configs['dataset_seed'], 
                    help='Seed of the toy dataset')
parser.add_argument("--n_datasets", type=int, default=data_configs['n_datasets'], 
                    help='Number of different toy datasets')
parser.add_argument("--base_path_to_data", type=str, default=data_configs['base_path_to_data'], 
                    help='Base path to a folder to load the toy datasets')
parser.add_argument("--base_path_to_results", type=str, default=models_configs['base_path_to_results'],
                    help="""Base path to a folder to load the model's output .npz files 
                            (results and hyperparameters) and save the final metrics.""")
parser.add_argument("--base_path_to_plots", type=str, default=plots_configs['base_path_to_plots'], 
                    help='Base path to a folder to save the final plots')
args = parser.parse_args()

if args.single_dataset:
    results_validation(
        dataset_seed=args.dataset_seed, 
        approach_type=args.approach_type,
        base_path_to_data=args.base_path_to_data,
        base_path_to_results=args.base_path_to_results,
        base_path_to_plots=args.base_path_to_plots
    )
else:
    for dataset_seed in tqdm(range(args.n_datasets)):
        results_validation(
            dataset_seed=dataset_seed,
            approach_type=args.approach_type,
            base_path_to_data=args.base_path_to_data,
            base_path_to_results=args.base_path_to_results,
            base_path_to_plots=args.base_path_to_plots
        )
