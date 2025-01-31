import numpy as np


class DinamoS:
    def __init__(
            self,
            stat_debiasing: bool,
            eps: float = 1e-9,
            **kwargs
        ):
        """
        Standard non-ML algorithm for DINAMO: DINAMO-S. 

        Parameters:
        - stat_debiasing (bool): Whether to use statistical debiasing (True or False).
        - eps (float, optional): Small value to avoid division by zero. Default is 1e-9.
        - **kwargs: Additional keyword arguments:
            - For stat_debiasing == True:
                - n_resamples (int): Number of resamples for the statistical debiasing.
                - debiasing_seed (int): Random seed for the statistical debiasing.
        
        Results of the algorithm:
        - χ2 (np.ndarray): Array of the chi-squared values.
        - pull (np.ndarray): Array of the pull terms.
        - μ (np.ndarray): Array of the references.
        - x (np.ndarray): Array of the normalized input histograms.
        - σ_μ (np.ndarray): Array of the references' uncertainties.
        - σ_p_x (np.ndarray): Array of the Poisson uncertainties for the historgams x.
        """
        self.stat_debiasing = stat_debiasing
        if self.stat_debiasing:
            self.n_resamples = kwargs['n_resamples']
            self.debiasing_seed = kwargs['debiasing_seed']
            self.rng = np.random.default_rng(self.debiasing_seed)
        self.eps = eps

    def _calculate_χ2(
            self, 
            x: np.ndarray, 
            σ_p_x: np.ndarray, 
            x_integral: float,
            μ: np.ndarray, 
            σ_μ: np.ndarray,
            μ_integral: float,
            n_bins: int,
        ) -> tuple:
        """
        Parameters:
        - x (np.ndarray): The current run.
        - σ_p_x (np.ndarray): The uncertainties associated with the current run.
        - x_integral (float): The integral of the current run.
        - μ (np.ndarray): The reference.
        - σ_μ (np.ndarray): The uncertainties of the reference.
        - μ_integral (float): The integral of the reference.
        - n_bins (int): The number of bins in the histograms.

        Returns:
        - tuple: A tuple containing:
            - χ2 (float): The chi-squared value.
            - pull (np.ndarray): The pull term.
        """
        if self.stat_debiasing:
            if μ_integral >= x_integral:
                μ_resampled = self.rng.normal(
                    μ[:, None], np.sqrt(σ_μ**2 + σ_p_x**2)[:, None],
                    size=(n_bins, self.n_resamples)
                )
                χ2 = np.sum(
                    (x[:, None] - μ_resampled)**2/(2*σ_p_x**2 + σ_μ**2)[:, None], axis=0
                ).mean()
                pull = np.mean(
                    (x[:, None] - μ_resampled)/(np.sqrt(2*σ_p_x**2 + σ_μ**2))[:, None], axis=1
                )
            else:
                σ_p_μ = np.sqrt(μ / μ_integral - μ**2 / μ_integral)
                σ_p_μ[μ  == 0] = 1 / μ_integral

                x1_resampled = self.rng.normal(
                    x[:, None], np.sqrt(σ_μ**2 + σ_p_μ**2)[:, None], 
                    size=(n_bins, self.n_resamples)
                )
                χ2 = np.sum(
                    (x1_resampled - μ[:, None])**2/(σ_p_x**2 + σ_μ**2 + σ_p_μ**2)[:, None], axis=0
                ).mean()
                pull = np.mean(
                        (x1_resampled - μ[:, None])/(np.sqrt(σ_p_x**2 + σ_μ**2 + σ_p_μ**2)[:, None]), axis=1
                )
        else:
            σ = np.sqrt(σ_p_x**2 + σ_μ**2)
            χ2 = np.sum((x - μ)**2 / σ**2)
            pull = (x - μ) / σ
        return χ2, pull

    def _reference_init(self, n_bins: int) -> tuple:
        """
        Method initializes a reference histogram as uniform
        and calculates its Poisson uncertainty for each bin.

        Parameters:
            - n_bins (float): The number of bins in the histograms.

        Returns:
            tuple: A tuple containing:
                - μ (numpy.ndarray): The normalized reference histogram.
                - σ_p_μ (numpy.ndarray): The Poisson uncertainty for each bin in the reference histogram.
                - μ_integral (float): The integral of the reference histogram.
        """
        μ = np.ones(n_bins) * 100 # initialize reference as uniform
        μ_integral = μ.sum()
        μ /= μ_integral
        σ_p_μ = np.sqrt(μ / μ_integral - μ**2 / μ_integral)
        return μ, σ_p_μ, μ_integral
    
    def _emwa_init(
            self,
            α: float,
            μ0: np.ndarray,
            σ_p_μ0: np.ndarray,
        ) -> tuple:
        """
        Method initializes the variables for the EWMA algorithm for the 
        initial reference and its uncertainty.

        Parameters:
            - α (float): The smoothing factor for the EWMA algorithm.
            - μ0 (numpy.ndarray): The initial reference.
            - σ_p_μ0 (numpy.ndarray): The uncertainty of the initial reference.

        Returns:
            tuple: A tuple containing:
                - W0 (numpy.ndarray): The weight array of the EWMA algorithm.
                - S_μ0 (numpy.ndarray): The weighted sum for the reference computation.
                - S_σ_μ0 (numpy.ndarray): The weighted sum for the reference's uncertainty computation.
        """
        ω = 1 / (σ_p_μ0**2 + self.eps)
        W0 = (1 - α) * ω
        S_μ0 = (1 - α) * ω * μ0
        S_σ_μ0 = (1 - α) * ω * σ_p_μ0**2
        return W0, S_μ0, S_σ_μ0

    def _update_reference(
            self, 
            α: float, 
            x_good: np.ndarray, 
            x_good_integral: float,
            σ_x_good: np.ndarray,
            μ: np.ndarray,
            μ_integral: float,
            W: np.ndarray, 
            S_μ: np.ndarray, 
            S_σ_μ: np.ndarray, 
        ) -> tuple:
        """
        The method to update the reference and its uncertainty with the EWMA algorithm.

        Parameters:
            - α (float): The weighting factor of the EWMA algorithm.
            - x_good (np.ndarray): The array of the current good run.
            - x_good_integral (float): The integral of the current good run.
            - σ_x_good (np.ndarray): The uncertainty of the current good run.
            - μ (np.ndarray): The current reference.
            - μ_integral (float): The current integral of the reference.
            - W (np.ndarray): The current weight array of the EWMA algorithm.
            - S_μ (np.ndarray): The current weighted sum for the reference computation.
            - S_σ_μ (np.ndarray): The current weighted sum for the reference's uncertainty computation.

        Returns:
            - tuple: A tuple containing:
                - μ (np.ndarray): The updated reference.
                - σ_μ (np.ndarray): The updated uncertainty of the reference.
                - μ_integral (float): The updated integral of the reference.
                - W (np.ndarray): The updated weight array of the EWMA algorithm.
                - S_μ (np.ndarray): The updated weighted sum for the reference computation.
                - S_σ_μ (np.ndarray): The updated weighted sum for the reference's uncertainty computation.
        """
        ω = 1 / (σ_x_good**2 + self.eps)
        W = α * W + (1 - α) * ω

        S_σ_μ = α * S_σ_μ + (1 - α) * ω * (x_good - μ)**2
        σ_μ = np.sqrt(S_σ_μ / W)

        S_μ = α * S_μ + (1 - α) * ω * x_good
        μ = S_μ / W

        μ_integral = α * μ_integral + (1 - α) * x_good_integral
        return μ, σ_μ, μ_integral, W, S_μ, S_σ_μ

    def run(
            self, 
            α: float, 
            x: np.ndarray, 
            y: np.ndarray, 
        ) -> dict:
        """
        Run the algorithm.

        Parameters:
        - α (float): The weighting factor of the EWMA algorithm.
        - x (np.ndarray): The 2D array where each row represents a run and each column represents a bin.
        - y (np.ndarray): A 1D array that represents the label of each run (0 for good, 1 for bad).

        Returns:
        - dict: A dictionary containing the following keys:
            - "χ2" (np.ndarray): The χ2 for each run.
            - "pull" (np.ndarray): The pull terms for each run.
            - "μ" (np.ndarray): The corresponding references for each run.
            - "x" (np.ndarray): The normalized histograms x.
            - "σ_μ" (np.ndarray): The uncertainties on the references.
            - "σ_p_x" (np.ndarray): The Poisson uncertainties on the normalized histograms x.
        """
        x = x.copy()
        y = y.copy()
        n_runs, n_bins = x.shape
        μ, σ_μ, μ_integral = self._reference_init(n_bins)
        W, S_μ, S_σ_μ = self._emwa_init(α, μ, σ_μ)
        
        results = {
            "χ2": [], "pull": [], "μ": [], "x": [],  "σ_μ": [], "σ_p_x": []
        }

        for i in range(n_runs):
            current_x = x[i]
            current_x_integral = current_x.sum()

            if current_x_integral > self.eps:
                current_x /= current_x_integral
                σ_p_current_x = np.sqrt(current_x / current_x_integral - current_x**2 / current_x_integral)
                σ_p_current_x[current_x == 0] = 1 / current_x_integral
            else:
                σ_p_current_x = np.zeros_like(current_x)

            χ2, pull = self._calculate_χ2(
                current_x, σ_p_current_x, current_x_integral,
                μ, σ_μ, μ_integral, n_bins
            )

            # save results
            results["χ2"].append(χ2)
            results["pull"].append(pull)
            results["μ"].append(μ)
            results["x"].append(current_x)
            results["σ_μ"].append(σ_μ)
            results["σ_p_x"].append(σ_p_current_x)

            # update the reference if the current run is good
            if y[i] == 0 and i + 1 < n_runs:
                μ, σ_μ, μ_integral, W, S_μ, S_σ_μ = self._update_reference(
                    α, current_x, current_x_integral, σ_p_current_x, 
                    μ, μ_integral, W, S_μ, S_σ_μ
                )

        # Convert results' lists to numpy arrays
        return {key: np.array(value) for key, value in results.items()}
