import numpy as np

from .z_score_transform import z_score_transform_for_hists, unify_binning   
from ..metrics import jaccard_distance

def jaccard_distance_evaluation(
        bin_centers: np.ndarray,
        data_μ: np.ndarray, 
        data_σ: np.ndarray, 
        x: np.ndarray, 
        y: np.ndarray, 
        references: np.ndarray, 
        σ_references: np.ndarray, 
        n_historical: int,
    ) -> tuple:
    mask_good = (y == 0)
    mask_good_historical = mask_good[:n_historical]
    mask_good_continual = mask_good[n_historical:]

    ############################################################################
    x_historical = x[:n_historical]
    references_historical = references[:n_historical]
    σ_references_historical = σ_references[:n_historical]
    data_μ_historical = data_μ[:n_historical]
    data_σ_historical = data_σ[:n_historical]

    I_historical = np.ones(shape=mask_good_historical.sum())
    x_good_historical = x_historical[mask_good_historical]
    references_good_historical = references_historical[mask_good_historical]
    references_σ_historical = σ_references_historical[mask_good_historical]
    data_μ_good_historical = data_μ_historical[mask_good_historical]
    data_σ_good_historical = data_σ_historical[mask_good_historical]

    ############################################################################
    x_continual = x[n_historical:]
    references_continual = references[n_historical:]
    σ_references_continual = σ_references[n_historical:]
    data_μ_continual = data_μ[n_historical:]
    data_σ_continual = data_σ[n_historical:]

    I_continual = np.ones(shape=mask_good_continual.sum())
    x_good_continual = x_continual[mask_good_continual]
    references_good_continual = references_continual[mask_good_continual]
    references_σ_continual = σ_references_continual[mask_good_continual]
    data_μ_good_continual = data_μ_continual[mask_good_continual]
    data_σ_good_continual = data_σ_continual[mask_good_continual]

    ######################################################
    x_pdf_estimations_historical, x_transformed_bins_historical = z_score_transform_for_hists(
        bin_centers, x_good_historical, data_μ_good_historical, data_σ_good_historical, I_historical)
    x_pdf_estimations_unified_binning_historical = unify_binning(
        bin_centers, x_transformed_bins_historical, x_pdf_estimations_historical)

    x_μ_historical = np.median(x_pdf_estimations_unified_binning_historical, axis=0)
    x_full_σ_historical = np.sqrt(x_pdf_estimations_unified_binning_historical.var(0))
    true_lower_historical = x_μ_historical - x_full_σ_historical
    true_upper_historical = x_μ_historical + x_full_σ_historical

    ##########

    pdf_estimations_refs_historical, refs_transformed_bins_historical = z_score_transform_for_hists(
        bin_centers, references_good_historical, data_μ_good_historical, data_σ_good_historical, I_historical)
    pdf_estimations_refs_unified_binning_historical = unify_binning(
        bin_centers, refs_transformed_bins_historical, pdf_estimations_refs_historical)
    refs_μ_historical = np.median(pdf_estimations_refs_unified_binning_historical, axis=0)

    pdf_σ_refs_historical, σ_refs_transformed_bins_historical = z_score_transform_for_hists(
        bin_centers, references_σ_historical, data_μ_good_historical, data_σ_good_historical, I_historical)
    pdf_σ_refs_unified_binning_historical = unify_binning(
        bin_centers, σ_refs_transformed_bins_historical, pdf_σ_refs_historical)
    refs_full_σ_historical = (pdf_σ_refs_unified_binning_historical).mean(0)
    predicted_lower_historical = refs_μ_historical - refs_full_σ_historical
    predicted_upper_historical = refs_μ_historical + refs_full_σ_historical

    jd_historical = jaccard_distance(
        true_lower_historical, true_upper_historical, predicted_lower_historical, predicted_upper_historical)

    ######################################################

    x_pdf_estimations_continual, x_transformed_bins_continual = z_score_transform_for_hists(
        bin_centers, x_good_continual, data_μ_good_continual, data_σ_good_continual, I_continual)
    x_pdf_estimations_unified_binning_continual = unify_binning(
        bin_centers, x_transformed_bins_continual, x_pdf_estimations_continual)

    x_μ_continual = np.median(x_pdf_estimations_unified_binning_continual, axis=0)
    x_full_σ_continual = np.sqrt(x_pdf_estimations_unified_binning_continual.var(0))
    true_lower_continual = x_μ_continual - x_full_σ_continual
    true_upper_continual = x_μ_continual + x_full_σ_continual

    ##########

    pdf_estimations_refs_continual, refs_transformed_bins_continual = z_score_transform_for_hists(
        bin_centers, references_good_continual, data_μ_good_continual, data_σ_good_continual, I_continual)
    pdf_estimations_refs_unified_binning_continual = unify_binning(
        bin_centers, refs_transformed_bins_continual, pdf_estimations_refs_continual)
    refs_μ_continual = np.median(pdf_estimations_refs_unified_binning_continual, axis=0)

    pdf_σ_refs_continual, σ_refs_transformed_bins_continual = z_score_transform_for_hists(
        bin_centers, references_σ_continual, data_μ_good_continual, data_σ_good_continual, I_continual)
    pdf_σ_refs_unified_binning_continual = unify_binning(
        bin_centers, σ_refs_transformed_bins_continual, pdf_σ_refs_continual)
    refs_full_σ_continual = pdf_σ_refs_unified_binning_continual.mean(0)
    predicted_lower_continual = refs_μ_continual - refs_full_σ_continual
    predicted_upper_continual = refs_μ_continual + refs_full_σ_continual

    jd_continual = jaccard_distance(
        true_lower_continual, true_upper_continual, predicted_lower_continual, predicted_upper_continual)

    return (jd_historical, jd_continual), \
           (x_μ_historical, x_full_σ_historical, refs_μ_historical, refs_full_σ_historical, 
            x_μ_continual, x_full_σ_continual, refs_μ_continual, refs_full_σ_continual)

