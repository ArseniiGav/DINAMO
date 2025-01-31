import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from .matplotlib_setup import matplotlib_setup
matplotlib_setup()

import warnings
warnings.filterwarnings("ignore")

def plot_refs_examples(
        labels: list,
        colors: list,
        bin_edges: np.ndarray,
        bin_centers: np.ndarray,
        log_red_χ2: np.ndarray, 
        pulls: np.ndarray, 
        xs: np.ndarray, 
        σ_p_xs: np.ndarray,
        references: np.ndarray, 
        σ_references: np.ndarray,
        indexes: np.ndarray,
        threshold_best: float, 
        approach_type: str,
        base_path_to_plots: str,
        dataset_seed: int,
        save=True,
    ) -> None:
    N = log_red_χ2.shape[0]
    K = log_red_χ2.shape[1]
    for i in range(N):
        fig, ax = plt.subplots(2, K, figsize=(22, 6), gridspec_kw={'height_ratios':(3, 2)})
        st = fig.suptitle(f"Threshold: {threshold_best:.2f}", fontsize=20)
        ax = ax.flatten()

        y_lim = 1.75 * max(
            max([max(references[i][j]) for j in range(K)]),
            max([max(xs[i][j]) for j in range(K)]))
        y_lim = max(y_lim, 0.01)

        for j in range(K):
            sub_name = f"Run: {indexes[i][j]}, " + f"log(χ2/ν): {log_red_χ2[i][j]:.2f}"
            
            reference = references[i][j]
            x = xs[i][j]
                
            σ_reference = σ_references[i][j]
            σ_p_x = σ_p_xs[i][j]
            
            pull = pulls[i][j]
        
            ax[j].stairs(
                reference, bin_edges, color='black', alpha=0.9, 
                label=r'$μ \pm σ_{\mu}$', linewidth=1.5
            )
            ax[j].fill_between(
                bin_centers, 
                reference-σ_reference, 
                reference+σ_reference, 
                fc='grey', alpha=0.3, step='mid'
            )
        
            ax[j].stairs(
                x, bin_edges, color=colors[j], alpha=0.9, 
                label=f'{labels[j]} run: ' + r'$x \pm σ_{x, p}$', linewidth=1.5
            )
            ax[j].fill_between(
                bin_centers, 
                x-σ_p_x, 
                x+σ_p_x, 
                fc=colors[j], alpha=0.3, step='mid'
            )

            ax[j].set_title(sub_name, fontsize=15, y=1.05)
            ax[j].set_ylabel("Normalized values")

            ax[j].legend(loc='upper right')
            ax[j].set_ylim(0.0, y_lim)
            ax[j].set_xlim(-5, 5)
            ax[j].set_xticklabels([])

            ax[j+K].plot(
                bin_centers, pull, 
                color='black', alpha=0.9, drawstyle='steps-mid'
            )

            ax[j+K].set_xlabel("u")
            ax[j+K].set_ylabel(r"$\frac{x-μ}{\sqrt{σ_{\mu}^2 + σ_{x, p}^2}}$")
            ax[j+K].set_ylim(-14, 14)
            ax[j+K].set_yticks([-10, -6, -2, 2, 6, 10])
            ax[j+K].set_xlim(-5, 5)

            fig.tight_layout()
            fig.subplots_adjust(hspace=0)
            if save:
                fig.savefig(f'{base_path_to_plots}plots/{approach_type}/dataset_{dataset_seed}/refs_examples_seed_{dataset_seed}_{i}.pdf')
        if save:
            plt.close()
        else:
            plt.show()
            plt.close()

def plot_threshold_opt_results(
        sensitivity_list,
        specificity_list,
        balanced_accuracy_weighed_list,
        T_values,
        xllim,
        xrlim,
        dataset_seed,
        sensitivity_color,
        specificity_color,
        approach_type,
        base_path_to_plots='../',
        save=True
    ):
    low_sensitivity, median_sensitivity, up_sensitivity = np.quantile(sensitivity_list, [0.5-0.34, 0.5, 0.5+0.34], axis=0)
    low_specificity, median_specificity, up_specificity = np.quantile(specificity_list, [0.5-0.34, 0.5, 0.5+0.34], axis=0)
    low_balanced_accuracy_weighed, median_balanced_accuracy_weighed, up_balanced_accuracy_weighed = np.quantile(
        balanced_accuracy_weighed_list, [0.5-0.34, 0.5, 0.5+0.34], axis=0)
    
    T_value_best = T_values[np.argmax(median_balanced_accuracy_weighed)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(T_values, median_sensitivity, color=sensitivity_color, alpha=0.7, label="Sensitivity")
    ax.fill_between(T_values, low_sensitivity, up_sensitivity, fc=sensitivity_color, alpha=0.3)
    
    ax.plot(T_values, median_specificity, color=specificity_color, alpha=0.7, label="Specificity")
    ax.fill_between(T_values, low_specificity, up_specificity, fc=specificity_color, alpha=0.3)
    
    ax.plot(T_values, median_balanced_accuracy_weighed, color='green', alpha=0.7, label="Balanced accuracy")
    ax.fill_between(T_values, low_balanced_accuracy_weighed, up_balanced_accuracy_weighed, fc='g', alpha=0.3)
    
    plt.vlines(T_value_best, 0, 2, ls='--', color='black', linewidth=4)
    
    plt.xlabel('log(χ2/ν) threshold')
    plt.ylabel('Metric value')
    ax.set_ylim(0.0, 1.01)
    ax.set_xlim(xllim, xrlim)
    ax.annotate(r"$\rm T_{best} = %.2f$" % T_value_best,
                xy=(T_value_best+0.05, 0.3),
                xytext=(T_value_best+0.5, 0.2),
                arrowprops=dict(facecolor='black', shrink=0.1),
                bbox=dict(boxstyle="round, pad=0.3", edgecolor="black", facecolor="white"),
                fontsize=14)
    
    ax.legend(loc="upper right", fontsize=14)
    fig.tight_layout()
    if save:
        fig.savefig(f'{base_path_to_plots}plots/{approach_type}/dataset_{dataset_seed}/best_threshold_seed_{dataset_seed}.pdf')
        plt.close()
    else:
        plt.show()
        plt.close()

def add_hist(
        ax: plt.Axes, 
        counts: np.ndarray, 
        bins: np.ndarray, 
        fill: bool, 
        color: str, 
        label: str, 
        alpha: float = 0.7, 
        linestyle: str = '-'
    ) -> None:
    ax.stairs(
        counts,
        bins,
        fill=fill,
        color=color,
        alpha=alpha,
        label=label,
        linestyle=linestyle,
        zorder=-1,
    )

def plot_results_hist(
        dataset_seed: int, 
        df_confusion_matrix: pd.DataFrame,
        y: np.ndarray, 
        log_red_χ2: np.ndarray, 
        n_historical: int, 
        xllim: float, 
        xrlim: float, 
        good_runs_color: str, 
        bad_runs_color: str, 
        threshold_best: float,
        base_path_to_plots: str,
        approach_type: str,
        save=True
    ) -> None:

    mask_good_historical = (y[:n_historical] == 0)
    mask_bad_historical = (y[:n_historical] == 1)
    log_red_χ2_historical = log_red_χ2[:n_historical]

    mask_good_continual = (y[n_historical:] == 0)
    mask_bad_continual = (y[n_historical:] == 1)
    log_red_χ2_continual = log_red_χ2[n_historical:]

    bins = np.linspace(xllim, xrlim, 51)
    log_red_χ2_continual_good_counts, _ = np.histogram(log_red_χ2_continual[mask_good_continual], bins=bins)
    log_red_χ2_continual_good_counts = log_red_χ2_continual_good_counts / log_red_χ2_continual_good_counts.sum()
    log_red_χ2_continual_bad_counts, _ = np.histogram(log_red_χ2_continual[mask_bad_continual], bins=bins)
    log_red_χ2_continual_bad_counts = log_red_χ2_continual_bad_counts / log_red_χ2_continual_bad_counts.sum()

    log_red_χ2_historical_good_counts, _ = np.histogram(log_red_χ2_historical[mask_good_historical], bins=bins)
    log_red_χ2_historical_good_counts = log_red_χ2_historical_good_counts / log_red_χ2_historical_good_counts.sum()
    log_red_χ2_historical_bad_counts, _ = np.histogram(log_red_χ2_historical[mask_bad_historical], bins=bins)
    log_red_χ2_historical_bad_counts = log_red_χ2_historical_bad_counts / log_red_χ2_historical_bad_counts.sum()

    fig, ax = plt.subplots(figsize = (8, 5))
    add_hist(ax, log_red_χ2_continual_good_counts, bins, True, good_runs_color, 'Good runs')
    add_hist(ax, log_red_χ2_continual_bad_counts, bins, True, bad_runs_color, 'Bad runs')

    add_hist(ax, log_red_χ2_historical_good_counts, bins, False, good_runs_color, None, linestyle='--')
    add_hist(ax, log_red_χ2_historical_bad_counts, bins, False, bad_runs_color, None, linestyle='--')

    ax.axvline(threshold_best, ls='--', color='black', zorder=-1, linewidth=4)
    ax.set_xlabel('log(χ2/ν)')
    ax.set_ylabel('Normalized number of runs')
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim(xllim, xrlim)
    ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])

    ax.stairs([0], [0, 1], color='black', fill=True, label="Continual")
    ax.plot([0], [0], color='black', linestyle='--', linewidth=2, label="Historical")

    handles, labels = ax.get_legend_handles_labels()                
    legend1 = ax.legend(handles[:2], labels[:2], frameon=1, ncol=1, fontsize=14, loc="upper right",)
    legend2 = ax.legend(handles[2:], labels[2:], frameon=1, ncol=1, fontsize=11, loc=(0.52, 0.85))
    ax.add_artist(legend1)
    ax.add_artist(legend2)

    a = plt.axes([0.73, 0.55, .23, .23])
    ax = sns.heatmap(
        df_confusion_matrix, annot=True, fmt='d', alpha=0.6,
        cbar=False, annot_kws={"size": 14}, cmap="Blues",
        xticklabels=["Good", "Bad"],
        yticklabels=["Good", "Bad"]
    )

    ax.set_xlabel("Predicted", fontsize=14, labelpad=10)
    ax.set_ylabel("True", fontsize=14, labelpad=10)
    ax.tick_params(left=False, bottom=False, right=False, top=False, which='both', labelsize=14)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)

    fig.tight_layout()
    if save:
        fig.savefig(f'{base_path_to_plots}/plots/{approach_type}/dataset_{dataset_seed}/results_hist_dataset_seed_{dataset_seed}.pdf') 
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_results_scatter(
        dataset_seed: int,
        n_runs: int, 
        y: np.ndarray, 
        log_red_χ2: np.ndarray, 
        n_historical: int, 
        xllim: float,
        xrlim: float,
        good_runs_color: str, 
        bad_runs_color: str, 
        threshold_best: float, 
        base_path_to_plots: str,
        approach_type: str,
        save=True
    ) -> None:
    x_scatter_plot = np.arange(n_runs)
    fig, ax = plt.subplots(figsize = (14, 4))

    ax.scatter(
        x_scatter_plot[y==0],
        log_red_χ2[y==0],
        color=good_runs_color,
        label='Good runs',
        alpha=0.5
    )

    ax.scatter(
        x_scatter_plot[y==1],
        log_red_χ2[y==1],
        color=bad_runs_color,
        label='Bad runs',
        alpha=0.5
    )

    ax.axhline(threshold_best, ls='--', color='black', linewidth=4)
    ax.axvline(n_historical, ls='--', color='green',  alpha=0.7, linewidth=2)
    ax.axvspan(0, n_historical, color='green', alpha=0.1,)
    ax.text(0.1, 0.9, "Historical regime", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=13, ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel('Run number')
    ax.set_ylabel('log(χ2/ν)')
    ax.set_xlim(-10, n_runs+10)
    ax.set_ylim(xllim, xrlim)
    ax.legend(loc="upper right")

    fig.tight_layout()
    if save:
        fig.savefig(f'{base_path_to_plots}/plots/{approach_type}/dataset_{dataset_seed}/results_scatter_dataset_seed_{dataset_seed}.pdf')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_uncertainty_evaluation(
        dataset_seed: int,
        bin_centers: np.ndarray,
        x_μ_historical: np.ndarray,
        x_full_σ_historical: np.ndarray,
        refs_μ_historical: np.ndarray,
        refs_full_σ_historical: np.ndarray,
        x_μ_continual: np.ndarray,
        x_full_σ_continual: np.ndarray,
        refs_μ_continual: np.ndarray,
        refs_full_σ_continual: np.ndarray,
        jd_historical: float,
        jd_continual: float,
        good_runs_color: str,
        base_path_to_plots: str,
        approach_type: str,
        save=True
    ) -> None:

    # Both cases
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax = ax.flatten()

    # Historical Case
    ax[0].plot(bin_centers, x_μ_historical, alpha=0.8, color="darkblue", linewidth=1.5, label='Median of z-score transformed good runs')
    ax[0].fill_between(bin_centers, 
                    x_μ_historical - x_full_σ_historical, 
                    x_μ_historical + x_full_σ_historical, 
                    alpha=0.3, color=good_runs_color,
                        label='Full unc. of the good runs: ' + r'$σ_{\rm full}$')
    ax[0].plot(bin_centers, refs_μ_historical, alpha=0.8, color='darkgreen', linewidth=1.5, label='Median of z-score transformed references')
    ax[0].fill_between(bin_centers, 
                    refs_μ_historical - refs_full_σ_historical, 
                    refs_μ_historical + refs_full_σ_historical, 
                    alpha=0.3, color='palegreen',
                    label='Modeled unc. on the references: ' + r'$σ_{\rm model}$')

    ax[0].text(0.17, 0.85, r"$D_J^H = %.3f$" %jd_historical, color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=10, ha="center", va="center", transform=ax[0].transAxes)
    ax[0].text(0.78, 0.6, "Historical regime", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=10, ha="center", va="center", transform=ax[0].transAxes)

    ax[0].set_ylabel("Density", fontsize=13)
    ax[0].set_xlabel("z", fontsize=13)
    ax[0].tick_params(axis='both', labelsize=13)
    ax[0].set_ylim(0, 0.07)
    ax[0].set_xlim(bin_centers[0], bin_centers[-1])
    ax[0].legend(loc='upper right', fontsize=8.5)

    # Continual Case
    ax[1].plot(bin_centers, x_μ_continual, alpha=0.8, color="darkblue", linewidth=1.5, label='Median of z-score transformed good runs')
    ax[1].fill_between(bin_centers, 
                    x_μ_continual - x_full_σ_continual, 
                    x_μ_continual + x_full_σ_continual, 
                    alpha=0.3, color=good_runs_color,
                    label='Full unc. of the good runs: ' + r'$σ_{\rm full}$')
    ax[1].plot(bin_centers, refs_μ_continual, alpha=0.8, color='darkgreen', linewidth=1.5, label='Median of z-score transformed references')
    ax[1].fill_between(bin_centers, 
                    refs_μ_continual - refs_full_σ_continual, 
                    refs_μ_continual + refs_full_σ_continual, 
                    alpha=0.3, color='palegreen',
                    label='Modeled unc. on the references: ' + r'$σ_{\rm model}$')

    ax[1].text(0.17, 0.85, r"$D_J^C = %.3f$" %jd_continual, color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=10, ha="center", va="center", transform=ax[1].transAxes)
    ax[1].text(0.78, 0.6, "Continual regime", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=10, ha="center", va="center", transform=ax[1].transAxes)

    ax[1].set_xlabel("z", fontsize=13)
    ax[1].tick_params(axis='both', labelsize=13)
    ax[1].set_xlim(bin_centers[0], bin_centers[-1])
    ax[1].legend(loc='upper right', fontsize=8.5)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    if save:
        fig.savefig(f'{base_path_to_plots}/plots/{approach_type}/dataset_{dataset_seed}/unc_vis_dataset_seed_{dataset_seed}.pdf') 
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_uncertainty_evaluation_continual(
        dataset_seed: int,
        bin_centers: np.ndarray,
        x_μ_continual: np.ndarray,
        x_full_σ_continual: np.ndarray,
        refs_μ_continual: np.ndarray,
        refs_full_σ_continual: np.ndarray,
        jd_continual: float,
        good_runs_color: str,
        base_path_to_plots: str,
        approach_type: str,
        save=True
    ) -> None:
    # Continual only
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), sharey=True)

    ax.plot(
        bin_centers, 
        x_μ_continual, 
        alpha=0.8, 
        color="darkblue", 
        linewidth=1.5, 
        label='Median of z-score transformed good runs'
    )
    ax.fill_between(
        bin_centers, 
        x_μ_continual - x_full_σ_continual, 
        x_μ_continual + x_full_σ_continual, 
        alpha=0.3, 
        color=good_runs_color,
        label='Full unc. of the good runs: ' + r'$σ_{\rm full}$'
    )

    ax.plot(
        bin_centers, 
        refs_μ_continual, 
        alpha=0.8, 
        color='darkgreen', 
        linewidth=1.5, 
        label='Median of z-score transformed references'
    )
    ax.fill_between(
        bin_centers, 
        refs_μ_continual - refs_full_σ_continual, 
        refs_μ_continual + refs_full_σ_continual, 
        alpha=0.3, 
        color='palegreen',
        label='Modeled unc. on the references: ' + r'$σ_{\rm model}$'
    )

    ax.text(0.175, 0.85, r"$D_J^C = %.3f$" %jd_continual, color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=11, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.78, 0.6, "Continual regime", color="black",
            bbox=dict(boxstyle="round, pad=0.4", edgecolor="black", facecolor="white"),  
            fontsize=11, ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("Density", fontsize=13)
    ax.set_xlabel("z", fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_ylim(0, 0.07)
    ax.set_xlim(bin_centers[0], bin_centers[-1])
    ax.legend(loc='upper right', fontsize=9.5)

    fig.tight_layout()
    if save:
        fig.savefig(f'{base_path_to_plots}/plots/{approach_type}/dataset_{dataset_seed}/unc_vis_dataset_seed_{dataset_seed}_continual.pdf') 
        plt.close()
    else:
        plt.show()
        plt.close()
