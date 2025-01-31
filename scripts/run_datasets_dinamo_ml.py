import argparse

from tqdm import tqdm
from dinamo.configs import data_configs, models_configs
from dinamo.utils import dinamo_ml_processing

parser = argparse.ArgumentParser(description='Dinamo-ML analysis for toy datasets with different seeds')
parser.add_argument("--single_dataset", type=int, default=data_configs['single_dataset'], 
                    help="""Process only a single dataset? Either 1 or 0. 
                            If set to 0, will process all datasets""")
parser.add_argument("--dataset_seed", type=int, default=data_configs['dataset_seed'], 
                    help='Seed of the toy dataset')
parser.add_argument("--n_datasets", type=int, default=data_configs['n_datasets'], 
                    help='Number of different toy datasets')
parser.add_argument("--t_opt", type=int, default=models_configs['t_opt'], 
                    help="""Optimize the threshold T for the model's outputs to calculate the accuracy metrics?
                            Either 1 or 0. If set to 0, will use the default value of T from the configs""")
parser.add_argument("--base_path_to_data", type=str, default=data_configs['base_path_to_data'], 
                    help='Base path to a folder to load the .npz files with the data')
parser.add_argument("--base_path_to_results", type=str, default=models_configs['base_path_to_results'], 
                    help='Base path to a folder to save the output .npz files with the results')
parser.add_argument("--device", type=str, default='cuda', 
                    help='What device to use for training the model? Either "cpu" or "cuda". Default is "cuda".')
parser.add_argument("--n_cpus", type=str, default=1, 
                    help='If device is "cpu", how many CPUs to use for training the model? Default is 1.')
args = parser.parse_args()

if args.single_dataset:
    dinamo_ml_processing(
        dataset_seed=args.dataset_seed, 
        t_opt=args.t_opt, 
        base_path_to_data=args.base_path_to_data, 
        base_path_to_results=args.base_path_to_results,
        device=args.device,
        n_cpus=args.n_cpus
    )
else:
    for dataset_seed in tqdm(range(args.n_datasets)):
        dinamo_ml_processing(
            dataset_seed=dataset_seed, 
            t_opt=args.t_opt, 
            base_path_to_data=args.base_path_to_data, 
            base_path_to_results=args.base_path_to_results,
            device=args.device,
            n_cpus=args.n_cpus
        )
