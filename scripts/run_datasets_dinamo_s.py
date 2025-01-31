import argparse

from tqdm import tqdm
from dinamo.configs import data_configs, models_configs
from dinamo.utils import dinamo_s_processing

parser = argparse.ArgumentParser(description='Dinamo-S analysis for toy datasets with different seeds')
parser.add_argument("--single_dataset", type=int, default=data_configs['single_dataset'], 
                    help="""Process only a single dataset? Either 1 or 0. 
                            If set to 0, will process all datasets""")
parser.add_argument("--dataset_seed", type=int, default=data_configs['dataset_seed'], 
                    help='Seed of the toy dataset')
parser.add_argument("--n_datasets", type=int, default=data_configs['n_datasets'], 
                    help='Number of different toy datasets')
parser.add_argument("--alpha_opt", type=int, default=models_configs['alpha_opt'], 
                    help="""Optimize the alpha hyperparameter of the model? Either 1 or 0. 
                            If set to 0, will use the default value of alpha from the configs""")
parser.add_argument("--t_opt", type=int, default=models_configs['t_opt'], 
                    help="""Optimize the threshold T for the model's outputs to calculate the accuracy metrics?
                            Either 1 or 0. If set to 0, will use the default value of T from the configs""")
parser.add_argument("--stat_debiasing", type=int, default=models_configs['stat_debiasing'], 
                    help="""Use run statistics debiasing technique?
                            Either 1 or 0. If set to 1, will use the default values of n_resamples
                            and the corresponding seed from the configs""")
parser.add_argument("--metric_mode", type=str, default=models_configs['metric_mode'],
                    help="""Metric to optimize for the hyperparameter optimization. 
                            Either 'auc' or 'wba'""")
parser.add_argument("--base_path_to_data", type=str, default=data_configs['base_path_to_data'], 
                    help='Base path to a folder to load the .npz files with the data')
parser.add_argument("--base_path_to_results", type=str, default=models_configs['base_path_to_results'], 
                    help='Base path to a folder to save the output .npz files with the results')
args = parser.parse_args()

if args.single_dataset:
    dinamo_s_processing(
        dataset_seed=args.dataset_seed, 
        stat_debiasing=args.stat_debiasing, 
        alpha_opt=args.alpha_opt, 
        t_opt=args.t_opt, 
        metric_mode=args.metric_mode, 
        base_path_to_data=args.base_path_to_data, 
        base_path_to_results=args.base_path_to_results
    )
else:
    for dataset_seed in tqdm(range(args.n_datasets)):
        dinamo_s_processing(
            dataset_seed=dataset_seed, 
            stat_debiasing=args.stat_debiasing, 
            alpha_opt=args.alpha_opt, 
            t_opt=args.t_opt, 
            metric_mode=args.metric_mode, 
            base_path_to_data=args.base_path_to_data, 
            base_path_to_results=args.base_path_to_results
        )
