import numpy as np


class ToyDataset:
    def __init__(
            self,
            n_runs: int,
            n_anomalous_runs: int,
            min_stats: int,
            max_stats: int,
            mu_slow_change_period: int,
            rapid_change_p: float,
            mu_rapid_changes_shift_lims: tuple,
            sigma_rapid_changes_shift_lims: tuple,
            anomaly_p: float,
            mu_anomaly_shift_lims: tuple,
            sigma_anomaly_shift_lims: tuple,
            binom_p: float,
            base_path_to_data: str,
        ):
        self.n_runs = n_runs
        self.n_anomalous_runs = n_anomalous_runs
        self.min_stats = min_stats
        self.max_stats = max_stats
        self.base_stats = max_stats - min_stats

        self.mu_slow_change_period = mu_slow_change_period
        self.rapid_change_p = rapid_change_p
        self.mu_rapid_changes_shift_lims = mu_rapid_changes_shift_lims
        self.sigma_rapid_changes_shift_lims = sigma_rapid_changes_shift_lims

        self.anomaly_p = anomaly_p
        self.mu_anomaly_shift_lims = mu_anomaly_shift_lims
        self.sigma_anomaly_shift_lims = sigma_anomaly_shift_lims
        
        self.binom_p = binom_p
        
        self.bin_edges = np.arange(-5.0, 5.1, 0.1)
        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        self.n_bins = len(self.bin_edges) - 1

        self.base_path_to_data = base_path_to_data

        self.inputs = {
            "n_runs": self.n_runs,
            "n_anomalous_runs": self.n_anomalous_runs,
            "min_stats": self.min_stats,
            "max_stats": self.max_stats,
            "base_stats": self.base_stats,
            "mu_slow_change_period": self.mu_slow_change_period,
            "rapid_change_p": self.rapid_change_p,
            "mu_rapid_changes_shift_lims": self.mu_rapid_changes_shift_lims,
            "sigma_rapid_changes_shift_lims": self.sigma_rapid_changes_shift_lims,
            "anomaly_p": self.anomaly_p,
            "mu_anomaly_shift_lims": self.mu_anomaly_shift_lims,
            "sigma_anomaly_shift_lims": self.sigma_anomaly_shift_lims,
            "binom_p": self.binom_p,
            "bin_edges": self.bin_edges,
            "bin_centers": self.bin_centers,
            "n_bins": self.n_bins,
            "base_path_to_data": self.base_path_to_data
        }

    def _apply_rapid_changes(
            self,
            rng: np.random.Generator, 
            n_runs: int,
            rapid_change_p: float,
            base_values: np.ndarray,
            shifts: np.ndarray
        ) -> np.ndarray:
        """
        Apply shifts to values of either μ or σ of the Gaussian distribution. 
        The array base_values is modified in place.
        Shifts are applied randomly with either negative or positive sign.

        Parameters:
        - rng (np.random.Generator): The random number generator for reproducibility.
        - n_runs (int): The total number of runs.
        - rapid_change_p (float): The probability of a rapid change for each run.
        - base_values (np.ndarray): The array with base values of μ or σ to be modified.
        - shifts (np.ndarray): The array with shifts to be applied.

        Returns:
        - inds_for_rapid_changes (np.ndarray): The indexes of the rapid changes.
        """
        rapid_change_indices = np.flatnonzero(rng.random(n_runs) < rapid_change_p)
        n_changes = len(rapid_change_indices)
        for i in range(n_changes):
            start_idx = rapid_change_indices[i]
            if i + 1 < n_changes:
                end_idx = rapid_change_indices[i+1]
            else:
                end_idx = n_runs

            sign = 2 * rng.binomial(n=1, p=0.5) - 1
            base_values[start_idx:end_idx] += (sign * shifts[start_idx])
        return rapid_change_indices 

    def build_and_save_dataset(self, seed: int) -> None:
        """
        Build and save a synthetic dataset of Gaussian distributions with anomalies for DINAMO.
        Multiple effects are modeled: 
            1) Slow changes for μ parameter;
            2) Rapid changes for both μ and σ parameters;
            3) Additional systematic uncertainties;
            4) Randomized "bad" behaviour in μ and σ parameters;
            5) Dead bins for some bad runs.

        Parameters:
        - seed (int): Seed for the random number generator to control reproducibility.

        Returns:
        - None: The function saves the generated dataset to a compressed .npz file.

        The saved .npz file contains:
        -----------------------------
        - x: np.ndarray
            Histogram counts of the runs.
        - y: np.ndarray
            Labels indicating good (0) or bad (1) runs.
        - inputs: dict
            Inputs used for generating the dataset.
        - mu: np.ndarray
            Means of the Gaussian distributions.
        - sigma: np.ndarray
            Standard deviations of the Gaussian distributions.
        - I0: np.ndarray
            Initial event statistics for each run.
        - I: np.ndarray
            Final event statistics for each run.
        - min_I: int
            Minimum event statistics across all runs.
        - max_I: int
            Maximum event statistics across all runs.
        - x_add: np.ndarray
            Additional systematic uncertainties applied to the histograms.
        - inds_for_rapid_changes_μ: np.ndarray
            Indices of runs with rapid changes in mean (μ).
        - inds_for_rapid_changes_σ: np.ndarray
            Indices of runs with rapid changes in standard deviation (σ).
        - anomaly_inds: np.ndarray
            Indices of anomalous runs.
        - anomaly_inds_for_μ_change: np.ndarray
            Indices of anomalous runs with changes in mean (μ).
        - anomaly_inds_for_σ_change: np.ndarray
            Indices of anomalous runs with changes in standard deviation (σ).
        - anomaly_inds_for_dead_bins: np.ndarray
            Indices of anomalous runs with dead bins.
        - number_of_dead_bins: np.ndarray
            Number of dead bins for each run.
        """
        # Create a random number generator with the seed for reproducibility
        rng = np.random.default_rng(seed)

        # Initialize labels and bases for parameters of the gaussian distribution
        labels = np.zeros(self.n_runs, dtype=np.int32)
        μ0 = np.zeros(self.n_runs)
        σ = np.ones(self.n_runs)

        # Set indeces for anomalous runs
        anomaly_inds = rng.choice(self.n_runs, size=self.n_anomalous_runs, replace=False)
        labels[anomaly_inds] = 1

        # Apply slow changes for μ of the gaussian distribution
        μ_t = np.arange(self.n_runs)
        μ = μ0 + 0.5 * np.sin(μ_t / self.mu_slow_change_period * np.pi)
        
        # Prepare random shifts for rapid changes
        μ_shifts = rng.uniform(*self.mu_rapid_changes_shift_lims, size=self.n_runs)
        σ_shifts = rng.uniform(*self.sigma_rapid_changes_shift_lims, size=self.n_runs)
        
        # Apply rapid changes 
        inds_for_rapid_changes_μ = self._apply_rapid_changes(
            rng, self.n_runs, self.rapid_change_p, μ, μ_shifts)
        inds_for_rapid_changes_σ = self._apply_rapid_changes(
            rng, self.n_runs, self.rapid_change_p, σ, σ_shifts)

        # Apply anomaly shifts
        anomaly_inds_for_μ_change = rng.choice(
            anomaly_inds, size=int(self.anomaly_p * self.n_anomalous_runs), replace=False)
        μ_signs = 2 * rng.binomial(p=0.5, n=1, size=int(self.anomaly_p * self.n_anomalous_runs)) - 1
        μ[anomaly_inds_for_μ_change] += (
            μ_signs * rng.uniform(*self.mu_anomaly_shift_lims,
                                        size=int(self.anomaly_p * self.n_anomalous_runs)))

        anomaly_inds_for_σ_change = rng.choice(
            anomaly_inds, size=int(self.anomaly_p * self.n_anomalous_runs), replace=False)
        σ_signs = 2 * rng.binomial(p=0.5, n=1, size=int(self.anomaly_p * self.n_anomalous_runs)) - 1
        σ[anomaly_inds_for_σ_change] += (
            σ_signs * rng.uniform(*self.sigma_anomaly_shift_lims, 
                                        size=int(self.anomaly_p * self.n_anomalous_runs)))

        # Sample from gaussian distributions with the corresponding parameters 
        # with initial event statistics of I0 and build the histograms
        I0 = rng.uniform(size=self.n_runs, low=self.min_stats, high=self.max_stats).astype(int)
        run_distrs_counts = []
        for i in range(len(I0)):
            run_distr = rng.normal(μ[i], σ[i], size=I0[i])
            run_distr_counts = np.histogram(run_distr, bins=self.bin_edges)[0]
            run_distrs_counts.append(run_distr_counts)
        run_distrs_counts = np.array(run_distrs_counts)

        # Apply additional systematic uncertainty
        signs_add = 2 * rng.binomial(n=1, p=0.5, size=(labels.shape[0], 1)) - 1
        run_distrs_counts_add = np.zeros(shape=run_distrs_counts.shape)
        run_distrs_counts_add[:, 50:] = signs_add * rng.binomial(run_distrs_counts[:, 50:], self.binom_p)
        run_distrs_counts = run_distrs_counts + run_distrs_counts_add

        # Apply dead bins for anomaly runs
        anomaly_inds_for_dead_bins = rng.choice(
            anomaly_inds, size=int(self.n_anomalous_runs), replace=False)
        number_of_dead_bins = np.zeros(self.n_runs).astype(int)
        number_of_dead_bins[anomaly_inds_for_dead_bins] = rng.integers(
            1, 20, size=anomaly_inds_for_dead_bins.shape[0])
        for anomaly_ind_for_dead_bins in anomaly_inds_for_dead_bins:
            inds_for_dead_bins = rng.choice(
                self.n_bins, size=number_of_dead_bins[anomaly_ind_for_dead_bins], replace=False)
            run_distrs_counts[anomaly_ind_for_dead_bins, inds_for_dead_bins] *= 0

        # Save the final statistics of the runs
        I = run_distrs_counts.sum(axis=1)
        min_I = I.min()
        max_I = I.max()
        
        # Save the dataset
        dataset_path = f'{self.base_path_to_data}/data/1d_gaussians_{seed}.npz'
        np.savez_compressed(dataset_path,
                            x=run_distrs_counts, y=labels, inputs=self.inputs, mu=μ, sigma=σ,
                            I0=I0, I=I, min_I=min_I, max_I=max_I, x_add=run_distrs_counts_add,
                            inds_for_rapid_changes_μ=inds_for_rapid_changes_μ, 
                            inds_for_rapid_changes_σ=inds_for_rapid_changes_σ,
                            anomaly_inds=anomaly_inds, 
                            anomaly_inds_for_μ_change=anomaly_inds_for_μ_change,
                            anomaly_inds_for_σ_change=anomaly_inds_for_σ_change,
                            anomaly_inds_for_dead_bins=anomaly_inds_for_dead_bins,
                            number_of_dead_bins=number_of_dead_bins)
