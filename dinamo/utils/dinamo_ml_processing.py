import os

import torch
import torch.nn as nn
import numpy as np

from ..configs import models_configs
from ..datasets import load_data
from ..models.ml import DinamoML, TransformerEncoderDML
from .threshold_opt import threshold_opt

def dinamo_ml_processing(
        dataset_seed: int, 
        t_opt: bool,
        base_path_to_data: str, 
        base_path_to_results: str,
        device: str,
        n_cpus: int = None,
    ) -> None:
    n_bins = models_configs['n_bins']
    M = models_configs['M']
    K = models_configs['K']
    d_model = models_configs['d_model']
    nhead = models_configs['nhead']
    num_layers = models_configs['num_layers']
    dim_feedforward = models_configs['dim_feedforward']
    dropout = models_configs['dropout']

    lr = models_configs['lr']
    n_epochs = models_configs['n_epochs']
    early_stopping_patience = models_configs['early_stopping_patience']
    training_seed = models_configs['training_seed']

    torch.manual_seed(training_seed)
    torch.cuda.manual_seed(training_seed)

    if device == 'cpu':
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        
        torch.set_num_threads(n_cpus)
        torch.set_num_interop_threads(n_cpus)

    n_historical = models_configs["n_historical"]
    dataset = load_data(seed=dataset_seed, base_path_to_data=base_path_to_data)
    x, y = dataset['x'], dataset['y']

    dataset_inputs = dataset['inputs'].item()
    n_bins = dataset_inputs['n_bins']

    transformer_encoder = TransformerEncoderDML(
        n_bins=n_bins, d_model=d_model, nhead=nhead, num_layers=num_layers, 
        dim_feedforward=dim_feedforward, dropout=dropout
    ).to(device).to(torch.float32)

    loss_function = nn.GaussianNLLLoss(full=False, reduction='mean', eps=1e-9)
    optimizer = torch.optim.AdamW(transformer_encoder.parameters(), lr=lr, weight_decay=1e-4)

    dinamo_ml = DinamoML(
        model=transformer_encoder, loss_function=loss_function, optimizer=optimizer, 
        M=M, K=K, lr=lr, n_epochs=n_epochs, early_stopping_patience=early_stopping_patience, 
        device=device
    )
    results = dinamo_ml.run(x, y)

    if t_opt:
        log_red_χ2 = np.log(results['χ2'] / n_bins)
        threshold_best = threshold_opt(
            y[:n_historical], log_red_χ2[:n_historical], # using historical data only
            approach_type="ml_approach",
            dataset_seed=dataset_seed, plot=True, 
            base_path_to_data=base_path_to_data)
    else:
        threshold_best = models_configs['threshold_default']

    np.savez(
        os.path.join(base_path_to_results, 
                     f"results/ml_approach/results_{dataset_seed}.npz"),
        **results
    )
    np.savez(
        os.path.join(base_path_to_results, 
                     f"results/ml_approach/hyperparameters_{dataset_seed}.npz"),
        threshold_best=threshold_best
    )
