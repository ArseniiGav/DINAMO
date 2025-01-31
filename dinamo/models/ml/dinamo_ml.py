import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinamoML:
    def __init__(
            self,
            model: nn.Module,
            loss_function: nn.Module,
            optimizer: torch.optim.Optimizer,
            M: int,
            K: int,
            lr: float,
            n_epochs=1000,
            early_stopping_patience=5,
            device='cuda'
        ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.M = M
        self.K = K

        self.lr = lr

        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.device = device

        self.good_runs = []
    
    def _estim_poisson_unc(self, x, x_integral):
        if x_integral < 1e-9:
            return np.zeros_like(x)
        σ_p_x = np.sqrt(x / x_integral - x**2 / x_integral)
        σ_p_x[x == 0] = 1 / x_integral
        return σ_p_x

    def _build_context_tensor(self, t: int):
        context_x = []
        context_t = []

        for good_run in self.good_runs[::-1]: # start iterating from the recent ones
            good_run_x, good_run_t = good_run
            if good_run_t < t:
                context_x.append(good_run_x) # normalized to unity
                context_t.append(good_run_t - t) # relative time wrt to the current t

                if len(context_t) == self.M: # no more than last M good runs
                    break

        if len(context_x) == 0:
            context_x.append(0.01 * np.ones(self.model.n_bins))
            context_t.append(-100)
            
        context_x = context_x[::-1] # old first
        context_t = context_t[::-1] # old first

        context_x_tensor = torch.tensor(
            context_x, dtype=torch.float32, device=self.device)
        context_t_tensor = torch.tensor(
            context_t, dtype=torch.float32, device=self.device)

        context_x_tensor = context_x_tensor.unsqueeze(0)
        context_x_tensor = F.pad(
            context_x_tensor, 
            (0, 0, self.M-context_x_tensor.shape[1], 0),
            value=0.01
        ) # pad to M if number of runs insufficient

        context_t_tensor = context_t_tensor.unsqueeze(0)
        context_t_tensor = F.pad(
            context_t_tensor, 
            (self.M-context_t_tensor.shape[1], 0), 
            value=-100
        ) # pad to M if number of runs insufficient
        return context_x_tensor, context_t_tensor # (1, M, n_bins), (1, M,)

    def _train_step(self):
        N = len(self.good_runs)
        if N < 1:
            return
        K_eff = min(self.K, N) # in the beginning, can be smaller than K

        self.model.train()
        self.optimizer.zero_grad()

        good_run_x_stacked, context_x_stacked, context_t_stacked = [], [], []
        for good_run in self.good_runs[::-1]: # start iterating from the recent ones
            good_run_x, good_run_t = good_run
            good_run_x_tensor = torch.tensor(
                good_run_x, dtype=torch.float32, device=self.device)
            good_run_x_stacked.append(good_run_x_tensor.unsqueeze(0))

            context_x, context_t = self._build_context_tensor(good_run_t)
            context_x_stacked.append(context_x)
            context_t_stacked.append(context_t)

            if len(context_x_stacked) == K_eff: # no more than last K good runs
                break

        good_run_x_stacked = torch.cat(good_run_x_stacked, dim=0)
        context_x_stacked = torch.cat(context_x_stacked, dim=0)
        context_t_stacked = torch.cat(context_t_stacked, dim=0)

        μ_out, σ_μ_out = self.model(context_x_stacked, context_t_stacked)  # (K, n_bins), (K, n_bins)

        loss = self.loss_function(μ_out, good_run_x_stacked, σ_μ_out**2) #input, target, var 
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _calculate_χ2(self, x_norm: np.ndarray, x_integral: float, t: int):
        σ_p_x = self._estim_poisson_unc(x_norm, x_integral)
        context_x, context_t = self._build_context_tensor(t)

        self.model.eval()
        with torch.no_grad():
            μ_out, σ_μ_out = self.model(context_x, context_t)

        μ = μ_out.squeeze(0).cpu().numpy()
        σ_μ = σ_μ_out.squeeze(0).cpu().numpy()

        σ = np.sqrt(σ_p_x**2 + σ_μ**2)
        χ2 = np.sum((x_norm - μ)**2 / σ**2)
        pull = (x_norm - μ) / σ
        return χ2, pull, μ, x_norm, σ_μ, σ_p_x

    def run(self, x, y):
        results = {
            "χ2": [], "pull": [], "μ": [], "x": [],  "σ_μ": [], "σ_p_x": []
        }
        
        x = x.copy()
        y = y.copy()
        integrals = x.sum(axis=1)
        x_norm = x / integrals[:, None]

        n_runs = x.shape[0]
        for i in tqdm(range(n_runs)):
            χ2, pull, μ, current_x, σ_μ, σ_p_current_x = self._calculate_χ2(
                x_norm[i], integrals[i], i)

            results["χ2"].append(χ2)
            results["pull"].append(pull)
            results["μ"].append(μ)
            results["x"].append(current_x)
            results["σ_μ"].append(σ_μ)
            results["σ_p_x"].append(σ_p_current_x)

            if y[i] == 0:
                self.good_runs.append((x_norm[i], i))

                best_mean_loss = np.inf
                epochs_no_improve = 0
                for _ in range(self.n_epochs):
                    loss = self._train_step()
                    
                    if loss < best_mean_loss:
                        best_mean_loss = loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= self.early_stopping_patience:
                        break

        return {key: np.array(value) for key, value in results.items()}
