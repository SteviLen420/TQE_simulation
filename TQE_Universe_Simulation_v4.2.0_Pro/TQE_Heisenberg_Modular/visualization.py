# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# visualization.py - All Plotting Functions
# ==========================================================================================
# Publication-quality visualizations for TQE Heisenberg Fluctuation Analysis
# ==========================================================================================

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_all_plots(
    tlist_agg, mean_E_no_law, std_E_no_law, mean_E_with_law, std_E_with_law,
    mean_S_no_law, mean_S_with_law, mean_C_no_law, mean_C_with_law,
    mean_U_no_law, std_U_no_law, mean_U_with_law, std_U_with_law,
    mean_DX_no_law, mean_DX_with_law,
    final_energies_no_law, final_energies_with_law,
    results_no_law, results_with_law,
    results_with_law_emergent, results_with_law_inherent, results_with_law_threshold,
    stats_comparison, config, figdir, T_FINAL
):
    """
    Generate all visualization plots.
    
    Returns:
    --------
    int
        Number of plots generated
    """
    n_plots = 0
    
    # Figure 1: Energy Evolution Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist_agg, mean_E_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
    ax.fill_between(tlist_agg, mean_E_no_law - std_E_no_law, mean_E_no_law + std_E_no_law,
                    alpha=0.3, color='red')
    ax.plot(tlist_agg, mean_E_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
    ax.fill_between(tlist_agg, mean_E_with_law - std_E_with_law, mean_E_with_law + std_E_with_law,
                    alpha=0.3, color='blue')
    ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Mean Energy ⟨E(t)⟩', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Energy Evolution: NO-LAW vs WITH-LAW', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "01_energy_comparison.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 2: Variance Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist_agg, std_E_no_law**2, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
    ax.plot(tlist_agg, std_E_with_law**2, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
    ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Energy Variance σ²(t)', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Variance Evolution: Fluctuation Suppression', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "02_variance_comparison.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 3: Entropy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist_agg, mean_S_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
    ax.plot(tlist_agg, mean_S_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
    ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Mean von Neumann Entropy', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Entropy Evolution: NO-LAW vs WITH-LAW', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "03_entropy_comparison.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 4: Coherence Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist_agg, mean_C_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
    ax.plot(tlist_agg, mean_C_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
    ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Mean Quantum Coherence', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Coherence Evolution: NO-LAW vs WITH-LAW', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "04_coherence_comparison.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 5: Final Energy Distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_energies_no_law, bins=30, alpha=0.6, label='NO-LAW (Pre-law)', color='red', density=True)
    ax.hist(final_energies_with_law, bins=30, alpha=0.6, label='WITH-LAW (Stable laws)', color='blue', density=True)
    ax.set_xlabel('Final Energy', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Probability Density', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Final Energy Distribution Comparison', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "05_final_energy_dist.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 6: Suppression Summary Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Variance\nRatio', 'Std Dev\nRatio', 'Max Energy\nRatio']
    values = [
        stats_comparison['SUPPRESSION_RATIOS']['variance_ratio'],
        stats_comparison['SUPPRESSION_RATIOS']['std_ratio'],
        stats_comparison['SUPPRESSION_RATIOS']['max_energy_ratio'],
    ]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='No suppression (ratio = 1)')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Ratio (WITH-LAW / NO-LAW)', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Fluctuation Suppression Summary\n(Values < 1.0 indicate suppression)', 
                 fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "06_suppression_summary.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 7: Heisenberg Uncertainty Evolution
    hbar_half = config["HBAR"] / 2.0
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist_agg, mean_U_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
    ax.fill_between(tlist_agg, mean_U_no_law - std_U_no_law, mean_U_no_law + std_U_no_law,
                    alpha=0.3, color='red')
    ax.plot(tlist_agg, mean_U_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
    ax.fill_between(tlist_agg, mean_U_with_law - std_U_with_law, mean_U_with_law + std_U_with_law,
                    alpha=0.3, color='blue')
    ax.axhline(y=hbar_half, color='black', linestyle='--', linewidth=2, 
               label=f'Heisenberg Limit (ℏ/2 = {hbar_half:.2f})', alpha=0.7)
    ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Uncertainty Product Δx·Δp', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Heisenberg Uncertainty Evolution', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "07_heisenberg_uncertainty.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 8: Phase Space (Energy vs Entropy)
    fig, ax = plt.subplots(figsize=(10, 8))
    n_sample = min(100, len(results_no_law), len(results_with_law))
    for i in range(n_sample):
        if i < len(results_no_law):
            ax.plot(results_no_law[i]["energies"], results_no_law[i]["entropy"], 
                   'r-', alpha=0.05, linewidth=0.5)
        if i < len(results_with_law):
            ax.plot(results_with_law[i]["energies"], results_with_law[i]["entropy"], 
                   'b-', alpha=0.05, linewidth=0.5)
    ax.plot(mean_E_no_law, mean_S_no_law, 'r-', linewidth=3, label='NO-LAW mean', alpha=0.9)
    ax.plot(mean_E_with_law, mean_S_with_law, 'b-', linewidth=3, label='WITH-LAW mean', alpha=0.9)
    ax.set_xlabel('Energy ⟨E⟩', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('von Neumann Entropy S', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Phase Space: Energy vs Entropy', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax.legend(fontsize=config['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "08_phase_space_E_vs_S.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figure 9: Multi-Dimensional Tracking
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    ax1.plot(tlist_agg, mean_E_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
    ax1.plot(tlist_agg, mean_E_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
    ax1.set_ylabel('Energy ⟨E⟩', fontsize=11)
    ax1.set_title('(A) Energy Evolution', fontsize=12, pad=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax2.plot(tlist_agg, mean_S_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
    ax2.plot(tlist_agg, mean_S_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
    ax2.set_ylabel('Entropy S', fontsize=11)
    ax2.set_title('(B) von Neumann Entropy', fontsize=12, pad=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax3.plot(tlist_agg, mean_C_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
    ax3.plot(tlist_agg, mean_C_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Coherence C (normalized)', fontsize=11)
    ax3.set_title('(C) Quantum Coherence', fontsize=12, pad=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax4.plot(tlist_agg, mean_U_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
    ax4.plot(tlist_agg, mean_U_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
    ax4.axhline(y=hbar_half, color='black', linestyle='--', linewidth=2, alpha=0.7, label='ℏ/2')
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Δx·Δp', fontsize=11)
    ax4.set_title('(D) Heisenberg Uncertainty', fontsize=12, pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    plt.suptitle('Multi-Dimensional Quantum State Tracking', fontsize=config['PLOT_FONTSIZE_TITLE'], y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, "09_multidimensional_tracking.png"), dpi=config['PLOT_DPI'])
    plt.close()
    n_plots += 1
    
    # Figures 11-13: I Evolution for ALL 3 I-Modes
    I_mode_results = {
        "emergent": results_with_law_emergent,
        "inherent": results_with_law_inherent,
        "threshold": results_with_law_threshold
    }
    I_mode_colors = {
        "emergent": "dodgerblue",
        "inherent": "forestgreen",
        "threshold": "darkorange"
    }
    I_mean_all_modes = {}
    
    for mode_name, results_mode in I_mode_results.items():
        if results_mode and len(results_mode) > 0:
            I_evolutions = [r.get('I_evolution') for r in results_mode if r.get('I_evolution') is not None]
            
            if I_evolutions:
                fig, ax = plt.subplots(figsize=(10, 6))
                color = I_mode_colors[mode_name]
                for I_evo in I_evolutions:
                    if len(I_evo) > 0:
                        t_I = np.linspace(0, T_FINAL, len(I_evo))
                        ax.plot(t_I, I_evo, alpha=0.1, color=color, linewidth=0.8)
                
                max_len = max([len(I_evo) for I_evo in I_evolutions])
                I_matrix = np.full((len(I_evolutions), max_len), np.nan)
                for i, I_evo in enumerate(I_evolutions):
                    I_matrix[i, :len(I_evo)] = I_evo
                I_mean = np.nanmean(I_matrix, axis=0)
                t_I_mean = np.linspace(0, T_FINAL, max_len)
                ax.plot(t_I_mean, I_mean, color='red', linewidth=2.5, label='Mean I(t)', zorder=10)
                I_mean_all_modes[mode_name] = (t_I_mean, I_mean)
                
                ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Information I', fontsize=config['PLOT_FONTSIZE_LABEL'])
                ax.set_title(f'Information Evolution: {mode_name.capitalize()} Model', 
                            fontsize=config['PLOT_FONTSIZE_TITLE'], pad=15)
                ax.legend(loc='best', fontsize=config['PLOT_FONTSIZE_LEGEND'])
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if mode_name == "emergent":
                    filename = "11_I_evolution_emergent.png"
                elif mode_name == "inherent":
                    filename = "12_I_evolution_inherent.png"
                elif mode_name == "threshold":
                    filename = "13_I_evolution_threshold.png"
                else:
                    filename = f"I_evolution_{mode_name}.png"
                
                plt.savefig(os.path.join(figdir, filename), dpi=config['PLOT_DPI'])
                plt.close()
                n_plots += 1
    
    # Figure 14: I-Mode Comparison
    if len(I_mean_all_modes) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        for mode_name, (t_I, I_mean) in I_mean_all_modes.items():
            color = I_mode_colors[mode_name]
            ax.plot(t_I, I_mean, color=color, linewidth=2.5, label=f'{mode_name.capitalize()}', alpha=0.85)
        ax.set_xlabel('Time', fontsize=config['PLOT_FONTSIZE_LABEL'])
        ax.set_ylabel('Mean Information I(t)', fontsize=config['PLOT_FONTSIZE_LABEL'])
        ax.set_title('Information Origin Models Comparison', 
                    fontsize=config['PLOT_FONTSIZE_TITLE'], pad=15)
        ax.legend(loc='best', fontsize=config['PLOT_FONTSIZE_LEGEND'])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figdir, "14_I_mode_comparison.png"), dpi=config['PLOT_DPI'])
        plt.close()
        n_plots += 1
    
    return n_plots

def generate_parameter_sweep_plot(sweep_var, sweep_values, sweep_results, config, figdir):
    """Generate parameter sweep visualization."""
    if not sweep_results:
        return
    
    import pandas as pd
    df_sweep = pd.DataFrame(sweep_results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(df_sweep[sweep_var], df_sweep['mean_energy'], 'o-', linewidth=2, markersize=8, color='purple')
    ax1.set_xlabel(f'{sweep_var}', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax1.set_ylabel('Mean Final Energy', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax1.set_title(f'Parameter Sweep: Mean Energy vs {sweep_var}', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax1.grid(True, alpha=0.3)
    ax2.plot(df_sweep[sweep_var], df_sweep['variance'], 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel(f'{sweep_var}', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax2.set_ylabel('Energy Variance', fontsize=config['PLOT_FONTSIZE_LABEL'])
    ax2.set_title(f'Parameter Sweep: Variance vs {sweep_var}', fontsize=config['PLOT_FONTSIZE_TITLE'], pad=20)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f"10_parameter_sweep_{sweep_var}.png"), dpi=config['PLOT_DPI'])
    plt.close()

