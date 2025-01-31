import argparse

from tqdm import tqdm
from dinamo.configs import data_configs
from dinamo.utils import eda_processing

parser = argparse.ArgumentParser(description='EDA for toy datasets with different seeds')
parser.add_argument("--single_dataset", type=int, default=data_configs['single_dataset'], 
                    help='Process only a single dataset? Either 1 or 0')
parser.add_argument("--dataset_seed", type=int, default=data_configs['dataset_seed'], 
                    help='Seed of the toy dataset')
parser.add_argument("--n_datasets", type=int, default=data_configs['n_datasets'], 
                    help='Number of different toy datasets')
parser.add_argument("--base_path_to_data", type=str, default=data_configs['base_path_to_data'], 
                    help='Base path to a folder to save the output .npz files')
args = parser.parse_args()

if args.single_dataset:
    eda_processing(dataset_seed=args.dataset_seed, base_path_to_data=args.base_path_to_data)
else:
    for i in tqdm(range(args.n_datasets)):
        eda_processing(dataset_seed=i, base_path_to_data=args.base_path_to_data)
