import numpy as np

def z_score_transform_for_hists(bin_centers, x, μ, σ, I):
    z_transformed_bins = np.transpose((bin_centers[:, None] - μ) / σ)
    pdf_estimations = x / (I[:, None] * np.diff(z_transformed_bins, axis=1)[:, :1]) * np.diff(bin_centers)[0]
    return pdf_estimations, z_transformed_bins

def unify_binning(new_bin_centers, transformed_bins, counts):
    interp_counts_list = []
    for i in range(counts.shape[0]):
        interp_counts = np.interp(new_bin_centers, transformed_bins[i], counts[i])
        interp_counts_list.append(interp_counts)
    interp_counts_list = np.array(interp_counts_list)
    return interp_counts_list
