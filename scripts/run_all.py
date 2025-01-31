import os
import argparse
from tqdm.contrib.concurrent import process_map

def run_script(script_name, dataset_seed, approach_type=None, device=None):
    """
    Runs a script with the given dataset_seed and optional arguments for approach_type and device.
    """
    command = f"python3 scripts/{script_name} --single_dataset 1 --dataset_seed {dataset_seed}"
    if approach_type:
        command += f" --approach_type {approach_type}"
    if device and "dinamo_ml" in script_name:
        command += f" --device {device}"
    os.system(command)

def run_all_scripts_for_dataset(args):
    """
    Runs all scripts for a single dataset_seed in sequence.
    """
    dataset_seed, device = args
    scripts = [
        ("run_datasets_production.py", None),               # No approach type or device
        ("run_datasets_eda.py", None),                      # No approach type or device
        ("run_datasets_dinamo_s.py", None),                 # No approach type or device
        ("run_results_validation.py", "standard_approach"), # Standard approach
        ("run_datasets_dinamo_ml.py", device),              # Needs device
        ("run_results_validation.py", "ml_approach"),       # ML approach
    ]
    for script_name, additional_arg in scripts:
        if "dinamo_ml" in script_name:
            run_script(script_name, dataset_seed, device=additional_arg)
        else:
            run_script(script_name, dataset_seed, approach_type=additional_arg)

if __name__ == "__main__":
    print("Running all scripts...")

    print("Installing the package...")
    os.system("pip3 install .")

    from dinamo.configs import data_configs
    parser = argparse.ArgumentParser(
        description="""Run all: produces toy datasets, runs EDA on them, and runs both the approaches on them, 
                       and runs the results validation. 
                       Number of datasets can be controlled, and can be parallelized; 
                       for the others uses the default values from the configs.
                       """)
    parser.add_argument("--n_datasets", type=int, default=data_configs['n_datasets'],
                        help='Number of different toy datasets')
    parser.add_argument("--parallelized", type=int, default=1,
                    help='Using parallelization or not')
    parser.add_argument("--n_cpus", type=int, default=50,
                    help='If parallelized, number of CPUs to use')
    parser.add_argument("--device", type=str, default="cpu",
                    help='What device to use for training the model? Either "cpu" or "cuda". Default is "cuda".')
    args = parser.parse_args()

    max_cpus = os.cpu_count()
    if args.n_cpus > max_cpus:
        print(f"Requested {args.n_cpus} CPUs, but only {max_cpus} are available. Using {max_cpus}.")
        args.n_cpus = max_cpus

    # Create necessary directories
    base_dirs = [
        "./data",
        "./results",
        "./results/ml_approach",
        "./results/ml_approach/metrics",
        "./results/standard_approach",
        "./results/standard_approach/metrics",
        "./plots",
        "./plots/data_eda",
        "./plots/data_eda/class_ratio",
        "./plots/data_eda/dead_bins",
        "./plots/data_eda/eda",
        "./plots/data_eda/gaussian_mu",
        "./plots/data_eda/gaussian_sigma",
        "./plots/data_eda/run_statistics",
        "./plots/data_eda/unc_vis",
        "./plots/ml_approach",
        "./plots/standard_approach"
    ]

    for base_dir in base_dirs:
        os.makedirs(base_dir, exist_ok=True)

    for dataset_seed in range(args.n_datasets):
        os.makedirs(
            f"./plots/standard_approach/dataset_{dataset_seed}", exist_ok=True
        )
        os.makedirs(
            f"./plots/ml_approach/dataset_{dataset_seed}", exist_ok=True
        )

    if args.parallelized:
        print("Running all tasks in parallel...")
        # Use `process_map` for parallel execution with progress bar
        process_map(
            run_all_scripts_for_dataset,
            [(dataset_seed, args.device) for dataset_seed in range(args.n_datasets)],
            max_workers=args.n_cpus,
            desc="Processing datasets"
        )
    else:
        print("Running sequentially...")
        for dataset_seed in range(args.n_datasets):
            run_all_scripts_for_dataset((dataset_seed, args.device))

    print("All tasks completed.")
