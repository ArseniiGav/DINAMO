import numpy as np

def calculate_metrics_with_bootstrap(
        y_val: np.ndarray, 
        y_pred_probs: np.ndarray, 
        threshold_values: np.ndarray, 
        n_samples: int, 
        n_trials: int, 
        β: float, 
        bootstrap_seed: int
    ) -> tuple[list[float], list[float], list[float]]:
    rng = np.random.default_rng(bootstrap_seed)

    sensitivity_list = []
    specificity_list = []
    balanced_accuracy_weighed_list = []

    for _ in range(n_trials):
        indx = rng.choice(y_val.shape[0], size=n_samples, replace=True)
        y_pred_probs_sample = y_pred_probs[indx]

        y_preds = (y_pred_probs_sample[:, np.newaxis] > threshold_values).astype(np.int32)
        true_positives = np.sum((y_val[indx][:, np.newaxis] == 1) & (y_preds == 1), axis=0)
        true_negatives = np.sum((y_val[indx][:, np.newaxis] == 0) & (y_preds == 0), axis=0)
        false_positives = np.sum((y_val[indx][:, np.newaxis] == 0) & (y_preds == 1), axis=0)
        false_negatives = np.sum((y_val[indx][:, np.newaxis] == 1) & (y_preds == 0), axis=0)

        sensitivity = true_positives / (true_positives + false_negatives)
        specificity = true_negatives / (true_negatives + false_positives)
        balanced_accuracy_weighed = (sensitivity + β * specificity) / (1 + β)

        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        balanced_accuracy_weighed_list.append(balanced_accuracy_weighed)
    return sensitivity_list, specificity_list, balanced_accuracy_weighed_list

def jaccard_distance(
        true_lower: np.ndarray, 
        true_upper: np.ndarray, 
        predicted_lower: np.ndarray, 
        predicted_upper: np.ndarray, 
    ) -> float:
    # Calculate intersection area
    intersection_upper = np.minimum(true_upper, predicted_upper)
    intersection_lower = np.maximum(true_lower, predicted_lower)
    intersection_area = np.sum(np.clip(intersection_upper - intersection_lower, a_min=0, a_max=None))

    # Calculate union area
    union_upper = np.maximum(true_upper, predicted_upper)
    union_lower = np.minimum(true_lower, predicted_lower)
    union_area = np.sum((union_upper - union_lower))

    # Handle potential division by zero
    if union_area == 0.0:
        return 1.0 # empty shapes

    # Compute Jaccard Index and Distance
    iou = intersection_area / union_area
    jd = 1 - iou
    return jd

def adaptivity_index(
        log_red_χ2: np.ndarray,
        y: np.ndarray,
        inds_for_rapid_changes: np.ndarray,
        threshold: float,
        patience: int, 
        reduction: str = 'mean'
    ) -> tuple[float, float]:
    if not isinstance(patience, int) or patience < 1:
        raise ValueError("Patience must be positive integer and >= 1")

    n_rapid_changes = len(inds_for_rapid_changes)

    adaptivity_indices = []
    nsteps_to_adapt = []
    for i, change_idx in enumerate(inds_for_rapid_changes):
        if i < n_rapid_changes-1:
            next_change_idx = inds_for_rapid_changes[i+1]
        else:
            next_change_idx = len(log_red_χ2)

        current_log_red_χ2_values = log_red_χ2[change_idx:next_change_idx]
        current_y = y[change_idx:next_change_idx]
        start_log_red_χ2_value = log_red_χ2[change_idx]

        nsteps = 0
        recovered_height = 0.0
        below_threshold_count = 0
        converged = False

        for j, log_red_χ2_value in enumerate(current_log_red_χ2_values):
            if current_y[j] == 0:
                if log_red_χ2_value <= threshold:
                    below_threshold_count += 1
                else:
                    below_threshold_count = 0  # Reset if it goes back above the threshold

                if below_threshold_count >= patience:  # Model has stabilized below the threshold
                    recovered_height = abs(start_log_red_χ2_value - log_red_χ2_value)
                    converged = True
                    break

                nsteps += 1

        if nsteps == 0:
            adaptivity_index = 0.0
            adaptivity_indices.append(adaptivity_index)
            nsteps_to_adapt.append(nsteps)
        else:
            if not converged:
                # Partial recovery if no full convergence before next change, take the last value
                recovered_height = abs(start_log_red_χ2_value - log_red_χ2_value)
            adaptivity_index = recovered_height / nsteps
            adaptivity_indices.append(adaptivity_index)
            nsteps_to_adapt.append(nsteps)

    if adaptivity_indices:
        if reduction == 'mean':
            aggr_adaptivity_index = np.mean(adaptivity_indices)
            aggr_nsteps_to_adapt = np.mean(nsteps_to_adapt)
        elif reduction == 'median':
            aggr_adaptivity_index = np.median(adaptivity_indices)
            aggr_nsteps_to_adapt = np.median(nsteps_to_adapt)
    else:
        aggr_adaptivity_index = None
        aggr_nsteps_to_adapt = None

    return aggr_nsteps_to_adapt, aggr_adaptivity_index
