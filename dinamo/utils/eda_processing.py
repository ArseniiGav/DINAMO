import numpy as np
import matplotlib.pyplot as plt

from ..datasets import load_data
from ..configs import models_configs, plots_configs
from .matplotlib_setup import matplotlib_setup
from .z_score_transform import z_score_transform_for_hists, unify_binning
matplotlib_setup()

def eda_processing(dataset_seed, base_path_to_data):
    dataset = load_data(seed=dataset_seed, base_path_to_data=base_path_to_data)
    x, y = dataset['x'], dataset['y']
    μ, σ = dataset['mu'], dataset['sigma']
    I = dataset['I']
    min_I, max_I = dataset['min_I'], dataset['max_I']
    number_of_dead_bins = dataset['number_of_dead_bins']
    x_add = dataset['x_add']

    dataset_inputs = dataset['inputs'].item()
    bin_centers = dataset_inputs['bin_centers']
    n_runs = dataset_inputs['n_runs']

    n_historical = models_configs["n_historical"]
    bad_runs_color = plots_configs["bad_runs_color"]
    good_runs_color = plots_configs["good_runs_color"]
    
    mask_good = (y == 0)
    mask_bad = (y == 1)
    run_numbers = np.arange(n_runs)
    
    #################################################### class ratio ####################################################
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.bar(
        x=["Good runs", "Bad runs"],
        height=[len(y)-sum(y), sum(y)],
        color=[good_runs_color, bad_runs_color],
        label=['Good runs', 'Bad runs']
    )
    
    ax.set_ylabel("Number of runs")
    ax.set_xlabel("Class label")
    fig.tight_layout()
    fig.savefig(f'{base_path_to_data}/plots/data_eda/class_ratio/dataset_{dataset_seed}_class_ratio.pdf')
    plt.close(fig) 
    
    #################################################### run statistics ####################################################
    
    I_bins = np.arange(min_I, max_I+1, (max_I - min_I) // 50)
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.hist(
        I[mask_good],
        bins=I_bins,
        fc=good_runs_color,
        alpha=0.7,
        zorder=-1,
        label='Good runs',
        histtype='stepfilled'
    )
    
    ax.hist(
        I[mask_bad],
        bins=I_bins,
        fc=bad_runs_color,
        alpha=0.7,
        zorder=-1,
        label='Bad runs',
        histtype='stepfilled'
    )
    
    ax.set_ylabel("Number of runs")
    ax.set_xlabel("Run statistics")
    ax.set_ylim(0, 200)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{base_path_to_data}/plots/data_eda/run_statistics/dataset_{dataset_seed}_run_statistics.pdf') 
    plt.close(fig) 

    #################################################### uncertainty ####################################################
    
    x_good = x[mask_good]
    μ_good = μ[mask_good]
    σ_good = σ[mask_good]
    I_good = I[mask_good]
    x_add_good = x_add[mask_good]
    
    x_pdf_estimations, x_transformed_bins = z_score_transform_for_hists(bin_centers, x_good, μ_good, σ_good, I_good)
    x_pdf_estimations_unified_binning = unify_binning(bin_centers, x_transformed_bins, x_pdf_estimations)

    pdf_μ = np.median(x_pdf_estimations_unified_binning, axis=0)
    pdf_full_σ = x_pdf_estimations_unified_binning.std(0)
    pdf_σ_poisson_term = np.sqrt(pdf_μ / I_good.mean())
    
    fig, ax = plt.subplots(figsize = (8, 6))
    
    ax.plot(bin_centers, pdf_μ, color='black', alpha=0.9, linewidth=1.5, label='Median of z-score transformed good runs',)
    ax.plot(bin_centers, pdf_μ + pdf_full_σ, color='black', alpha=0.9, linewidth=1.5, linestyle='--', label='Full unc.: ' + r'$σ_{\rm full}$')
    ax.plot(bin_centers, pdf_μ - pdf_full_σ, color='black', alpha=0.9, linewidth=1.5, linestyle='--')
    ax.fill_between(bin_centers, pdf_μ-pdf_full_σ, pdf_μ+pdf_full_σ, alpha=0.2, color=good_runs_color,
                    label='Syst. unc.: ' + r'$σ_{\rm syst}$')
    ax.fill_between(bin_centers, pdf_μ-pdf_σ_poisson_term, pdf_μ+pdf_σ_poisson_term, alpha=0.4, color="grey", 
                    label='Stat. unc.: ' + r'$σ_{\rm stat}$')
    
    ax.set_ylabel("Normalized values")
    ax.set_xlabel("z")
    ax.set_ylim(0, 0.07)
    ax.set_xlim(bin_centers[0], bin_centers[-1])
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{base_path_to_data}/plots/data_eda/unc_vis/dataset_{dataset_seed}_unc_vis.pdf')
    plt.close(fig) 
    
    #################################################### dead bins ####################################################
    
    fig, ax = plt.subplots(figsize = (8, 6))
    
    ax.axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax.axvspan(0, n_historical, color='green', alpha=0.1,)
    ax.text(0.1, 0.8, "Historical regime", color="black", rotation=90,
               bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
               fontsize=13, ha="center", va="center", transform=ax.transAxes)
    
    ax.scatter(run_numbers[mask_good], number_of_dead_bins[mask_good], c=good_runs_color, alpha=0.5, label='Good runs')
    ax.scatter(run_numbers[mask_bad], number_of_dead_bins[mask_bad], c=bad_runs_color, alpha=0.5, label='Bad runs')
    
    ax.set_xlabel('Run number')
    ax.set_ylabel('Number of dead bins')
    ax.set_xlim(0, n_runs)
    ax.set_ylim(-2, 25)
    ax.legend(loc='upper right')
    fig.tight_layout()    
    fig.savefig(f'{base_path_to_data}/plots/data_eda/dead_bins/dataset_{dataset_seed}_dead_bins.pdf')
    plt.close(fig) 

    #################################################### gaussian mu ####################################################
    
    fig, ax = plt.subplots(figsize = (8, 6))
    
    ax.axhline(0, color='black', linewidth=1.5, alpha=1)
    ax.axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax.axvspan(0, n_historical, color='green', alpha=0.1,)
    ax.text(0.1, 0.8, "Historical regime", color="black", rotation=90,
               bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
               fontsize=13, ha="center", va="center", transform=ax.transAxes)
    
    ax.scatter(run_numbers[mask_good], μ[mask_good], c=good_runs_color, alpha=0.5, label='Good runs')
    ax.scatter(run_numbers[mask_bad], μ[mask_bad], c=bad_runs_color, alpha=0.5, label='Bad runs')
    
    ax.set_xlabel('Run number')
    ax.set_ylabel('Mean of the gaussian distribution: μ')
    ax.set_xlim(0, n_runs)
    ax.set_ylim(-4.0, 6.0)
    ax.legend(loc='upper right', fontsize=13)
    fig.tight_layout()   
    fig.savefig(f'{base_path_to_data}/plots/data_eda/gaussian_mu/dataset_{dataset_seed}_gaussian_mu.pdf')
    plt.close(fig)
    
    #################################################### gaussian sigma ####################################################
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.axhline(1, color='black', linewidth=1.5, alpha=1)
    ax.scatter(run_numbers[mask_good], σ[mask_good], c=good_runs_color, alpha=0.5, label='Good runs')
    ax.scatter(run_numbers[mask_bad], σ[mask_bad], c=bad_runs_color, alpha=0.5, label='Bad runs')
    
    ax.set_xlabel('Run number')
    ax.set_ylabel('Std. dev. of the gaussian distribution: σ')
    ax.set_ylim(0.2, 2.5)
    ax.set_xlim(0, n_runs)
    ax.axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax.axvspan(0, n_historical, color='green', alpha=0.1,)
    ax.text(0.1, 0.8, "Historical regime", color="black", rotation=90,
               bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
               fontsize=13, ha="center", va="center", transform=ax.transAxes)
    ax.legend(loc='upper right')
    
    fig.tight_layout()
    fig.savefig(f'{base_path_to_data}/plots/data_eda/gaussian_sigma/dataset_{dataset_seed}_gaussian_sigma.pdf')
    plt.close(fig)
    
    #################################################### all together ####################################################
    
    fig, ax = plt.subplots(2, 3, figsize = (18, 10))
    ax = ax.flatten()
    
    ####################################################
    
    ax[0].bar(
        x=["Good runs", "Bad runs"],
        height=[len(y)-sum(y), sum(y)],
        color=[good_runs_color, bad_runs_color],
        label=['Good runs', 'Bad runs'], 
        alpha=0.7, zorder=-1,
    )
    
    ax[0].set_ylabel("Number of runs", fontsize=16)
    ax[0].set_xlabel("Class label")
    
    ####################################################
    
    ax[1].hist(I[mask_good], bins=I_bins, fc=good_runs_color,
               alpha=0.7, zorder=-1, label='Good runs', histtype='stepfilled')
    
    ax[1].hist(I[mask_bad], bins=I_bins, fc=bad_runs_color,
               alpha=0.7, zorder=-1, label='Bad runs', histtype='stepfilled')
    
    ax[1].set_ylabel("Number of runs", fontsize=16)
    ax[1].set_xlabel("Run statistics")
    ax[1].set_ylim(0, 200)
    ax[1].legend(loc='upper right', fontsize=13)
    
    ####################################################
    
    ax[2].plot(bin_centers, pdf_μ, color='black', alpha=0.9, linewidth=1.5, label='Median of z-score transformed good runs',)
    ax[2].plot(bin_centers, pdf_μ + pdf_full_σ, color='black', alpha=0.9, linewidth=1.5, linestyle='--', label='Full unc.: ' + r'$σ_{\rm full}$')
    ax[2].plot(bin_centers, pdf_μ - pdf_full_σ, color='black', alpha=0.9, linewidth=1.5, linestyle='--')
    ax[2].fill_between(bin_centers, pdf_μ-pdf_full_σ, pdf_μ+pdf_full_σ, alpha=0.2, color=good_runs_color,
                    label='Syst. unc.: ' + r'$σ_{\rm syst}$')
    ax[2].fill_between(bin_centers, pdf_μ-pdf_σ_poisson_term, pdf_μ+pdf_σ_poisson_term, alpha=0.4, color="grey", 
                    label='Stat. unc.: ' + r'$σ_{\rm stat}$')
    
    ax[2].set_ylabel("Normalized values", fontsize=16)
    ax[2].set_xlabel("z")
    ax[2].set_ylim(0, 0.07)
    ax[2].set_xlim(bin_centers[0], bin_centers[-1])
    ax[2].legend(loc='upper right', fontsize=11)
    
    ####################################################
    
    ax[3].axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax[3].axvspan(0, n_historical, color='green', alpha=0.1,)
    ax[3].text(0.1, 0.8, "Historical regime", color="black", rotation=90,
               bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
               fontsize=11, ha="center", va="center", transform=ax[3].transAxes)
    
    ax[3].scatter(run_numbers[mask_good], number_of_dead_bins[mask_good], c=good_runs_color, alpha=0.9, label='Good runs', s=20)
    ax[3].scatter(run_numbers[mask_bad], number_of_dead_bins[mask_bad], c=bad_runs_color, alpha=0.4, label='Bad runs', s=20)
    
    ax[3].set_xlabel('Run number')
    ax[3].set_ylabel('Number of dead bins', fontsize=16)
    ax[3].set_xlim(0, n_runs)
    ax[3].set_ylim(-2, 25)
    ax[3].legend(loc='upper right', fontsize=13)
    
    ####################################################
    
    ax[4].axhline(0, color='black', linewidth=1, alpha=1)
    ax[4].axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax[4].axvspan(0, n_historical, color='green', alpha=0.1,)
    ax[4].text(0.1, 0.8, "Historical regime", color="black", rotation=90,
               bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
               fontsize=11, ha="center", va="center", transform=ax[4].transAxes)
    
    ax[4].scatter(run_numbers[mask_good], μ[mask_good], c=good_runs_color, alpha=0.9, label='Good runs', s=20)
    ax[4].scatter(run_numbers[mask_bad], μ[mask_bad], c=bad_runs_color, alpha=0.4, label='Bad runs', s=20)
    
    ax[4].set_xlabel('Run number')
    ax[4].set_ylabel('Mean of the gaussian distribution: μ', fontsize=15)
    ax[4].set_xlim(0, n_runs)
    ax[4].set_ylim(-4.0, 6.0)
    ax[4].legend(loc='upper right', fontsize=13)
    
    ####################################################
    
    ax[5].axhline(1, color='black', linewidth=1, alpha=1)
    ax[5].scatter(run_numbers[mask_good], σ[mask_good], c=good_runs_color, alpha=0.9, label='Good runs', s=20)
    ax[5].scatter(run_numbers[mask_bad], σ[mask_bad], c=bad_runs_color, alpha=0.4, label='Bad runs', s=20)
    
    ax[5].set_xlabel('Run number')
    ax[5].set_ylabel('Std. dev. of the gaussian distribution: σ', fontsize=15)
    ax[5].set_ylim(0.2, 2.5)
    ax[5].set_xlim(0, n_runs)
    ax[5].axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax[5].axvspan(0, n_historical, color='green', alpha=0.1,)
    ax[5].text(0.1, 0.8, "Historical regime", color="black", rotation=90,
               bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
               fontsize=11, ha="center", va="center", transform=ax[5].transAxes)
    ax[5].legend(loc='upper right', fontsize=13)
    
    fig.tight_layout()
    fig.savefig(f'{base_path_to_data}/plots/data_eda/eda/dataset_{dataset_seed}_eda.pdf') 
    plt.close(fig)
    