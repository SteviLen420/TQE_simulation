# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Comparative analysis functions

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.ioff()

def compare_ei_definitions(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """
    Compare all E+I definitions (11 I-parameters: including Jensen-Shannon and KL-Shannon-Entanglement).
    
    Generates:
    - Stability rates comparison
    - Goldilocks zones comparison
    - Planck fit comparison
    - Ranking table
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: E+I DEFINITIONS COMPARISON")
    print("="*70)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping E+I comparison")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    # 1. Stability rates comparison
    print("\n1.1 Stability Rates Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Stable %
    df_ei_sorted = df_ei.sort_values("stable_percent", ascending=False)
    axes[0].barh(df_ei_sorted["i_definition"], df_ei_sorted["stable_percent"], color='green', alpha=0.7)
    axes[0].set_xlabel("Stable %", fontsize=12, fontweight='bold')
    axes[0].set_title("Stability Rate", fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Unstable %
    df_ei_sorted = df_ei.sort_values("unstable_percent", ascending=True)
    axes[1].barh(df_ei_sorted["i_definition"], df_ei_sorted["unstable_percent"], color='red', alpha=0.7)
    axes[1].set_xlabel("Unstable %", fontsize=12, fontweight='bold')
    axes[1].set_title("Instability Rate", fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Lock-in %
    df_ei_sorted = df_ei.sort_values("lockin_percent", ascending=False)
    axes[2].barh(df_ei_sorted["i_definition"], df_ei_sorted["lockin_percent"], color='blue', alpha=0.7)
    axes[2].set_xlabel("Lock-in %", fontsize=12, fontweight='bold')
    axes[2].set_title("Law Lock-in Rate", fontsize=14, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_rates_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ stability_rates_comparison.png")
    
    # 2. Goldilocks zones comparison
    print("\n1.2 Goldilocks Zones Comparison")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, row in df_ei.iterrows():
        y_pos = i
        x_peak = row["X_peak"]
        x_unc = row["X_peak_uncertainty"]
        x_low = row["X_low"]
        x_high = row["X_high"]
        
        # Goldilocks window
        ax.barh(y_pos, x_high - x_low, left=x_low, height=0.6, 
                alpha=0.3, color='yellow', edgecolor='green', linewidth=2)
        
        # Peak with uncertainty
        ax.errorbar(x_peak, y_pos, xerr=x_unc*1.96, fmt='o', 
                   color='red', markersize=10, capsize=5, capthick=2, linewidth=2)
    
    ax.set_yticks(range(len(df_ei)))
    ax.set_yticklabels(df_ei["i_definition"])
    ax.set_xlabel("X (E·I coupling)", fontsize=12, fontweight='bold')
    ax.set_title("Goldilocks Zones Comparison (Yellow=Zone, Red=Peak±σ)", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "goldilocks_zones_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ goldilocks_zones_comparison.png")
    
    # 3. Chi-squared comparison
    print("\n1.3 Planck χ² Fit Comparison")
    # Resolve chi2 column name: prefer 'chi_squared_reduced', fallback to 'planck_chi2_reduced'
    chi2_col_primary = "chi_squared_reduced"
    chi2_col_fallback = "planck_chi2_reduced"
    chi2_col_resolved = None
    if chi2_col_primary in df_ei.columns and df_ei[chi2_col_primary].notna().any():
        chi2_col_resolved = chi2_col_primary
    elif chi2_col_fallback in df_ei.columns and df_ei[chi2_col_fallback].notna().any():
        chi2_col_resolved = chi2_col_fallback
    else:
        chi2_col_resolved = None
    df_ei_chi = df_ei.dropna(subset=[chi2_col_resolved]) if chi2_col_resolved else pd.DataFrame(columns=df_ei.columns)
    
    if len(df_ei_chi) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_ei_chi_sorted = df_ei_chi.sort_values(chi2_col_resolved)
        ax.barh(df_ei_chi_sorted["i_definition"], df_ei_chi_sorted[chi2_col_resolved], 
                color='purple', alpha=0.7)
        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect fit (χ²/dof=1)')
        ax.set_xlabel("χ²/dof", fontsize=12, fontweight='bold')
        ax.set_title("Planck Validation: χ² Fit Quality (lower is better)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "planck_chi2_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ planck_chi2_comparison.png")
    else:
        print("   ⚠️  No χ² data available")
    
    # 4. Ranking table
    print("\n1.4 Ranking Table")
    # Build ranking with resolved chi2 column
    if chi2_col_resolved is None:
        ranking_df = df_ei[["i_definition", "stable_percent", "lockin_percent", "X_peak_uncertainty"]].copy()
    else:
        ranking_df = df_ei[["i_definition", "stable_percent", "lockin_percent", 
                            "X_peak_uncertainty", chi2_col_resolved]].copy()
        # Normalize column name to 'chi_squared_reduced' in output if fallback was used
        if chi2_col_resolved != "chi_squared_reduced":
            ranking_df = ranking_df.rename(columns={chi2_col_resolved: "chi_squared_reduced"})
    ranking_df = ranking_df.sort_values("stable_percent", ascending=False)
    ranking_df.to_csv(os.path.join(output_dir, "ei_ranking_table.csv"), index=False)
    print("   ✅ ei_ranking_table.csv")
    
    print("\n✅ E+I Definitions Comparison Complete!")


def compare_eonly_vs_ei(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """
    PHASE 3B: Compare E-only vs E+I (if E-only data available in batch_all mode).
    
    Analyzes improvements from adding I-coupling:
    - Stability rate improvements (stable %, lock-in %)
    - Goldilocks peak shifts (E-only baseline vs E+I peaks)
    - Detailed improvement metrics for each I-definition
    
    Generates:
    - stability_improvement.png (bar charts showing Δ% vs E-only)
    - goldilocks_shift.png (peak position changes with arrows)
    - eonly_vs_ei_metrics.json (detailed improvement data)
    """
    print("\n" + "="*70)
    print("PHASE 3B: E-ONLY vs E+I COMPARISON")
    print("="*70)
    
    df_eonly = df_metrics[df_metrics["run_type"] == "E-only"]
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    if len(df_eonly) == 0:
        print("⚠️  No E-only data found, skipping E-only vs E+I comparison")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    # Get E-only baseline
    eonly_stable = df_eonly["stable_percent"].iloc[0]
    eonly_lockin = df_eonly["lockin_percent"].iloc[0]
    eonly_X_peak = df_eonly["X_peak"].iloc[0]
    
    # 1. Stability improvement
    print("\n2.1 Stability Improvement")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stable %
    improvements_stable = df_ei["stable_percent"] - eonly_stable
    df_plot = df_ei.copy()
    df_plot["improvement_stable"] = improvements_stable
    df_plot_sorted = df_plot.sort_values("improvement_stable", ascending=False)
    
    colors_stable = ['green' if x > 0 else 'red' for x in df_plot_sorted["improvement_stable"]]
    axes[0].barh(df_plot_sorted["i_definition"], df_plot_sorted["improvement_stable"], color=colors_stable, alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel("Improvement in Stable % (vs E-only)", fontsize=12, fontweight='bold')
    axes[0].set_title("Stability Improvement", fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Lock-in %
    improvements_lockin = df_ei["lockin_percent"] - eonly_lockin
    df_plot["improvement_lockin"] = improvements_lockin
    df_plot_sorted = df_plot.sort_values("improvement_lockin", ascending=False)
    
    colors_lockin = ['blue' if x > 0 else 'red' for x in df_plot_sorted["improvement_lockin"]]
    axes[1].barh(df_plot_sorted["i_definition"], df_plot_sorted["improvement_lockin"], color=colors_lockin, alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel("Improvement in Lock-in % (vs E-only)", fontsize=12, fontweight='bold')
    axes[1].set_title("Law Lock-in Improvement", fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_improvement.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ stability_improvement.png")
    
    # 2. Goldilocks shift
    print("\n2.2 Goldilocks Peak Shift")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # E-only baseline
    ax.axhline(-1, color='gray', linestyle='--', linewidth=2)
    ax.scatter([eonly_X_peak], [-1], s=200, color='gray', marker='s', 
               label='E-only baseline', zorder=10, edgecolors='black', linewidths=2)
    
    # E+I peaks
    for i, row in df_ei.iterrows():
        y_pos = i
        x_peak = row["X_peak"]
        x_unc = row["X_peak_uncertainty"]
        
        ax.errorbar(x_peak, y_pos, xerr=x_unc*1.96, fmt='o', 
                   color='blue', markersize=10, capsize=5, capthick=2, linewidth=2)
        
        # Arrow from E-only to E+I
        ax.annotate('', xy=(x_peak, y_pos), xytext=(eonly_X_peak, y_pos),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.5))
    
    ax.set_yticks(list(range(-1, len(df_ei))))
    ax.set_yticklabels(['E-only'] + list(df_ei["i_definition"]))
    ax.set_xlabel("X_peak (E·I coupling)", fontsize=12, fontweight='bold')
    ax.set_title("Goldilocks Peak Position Comparison", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "goldilocks_shift.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ goldilocks_shift.png")
    
    # 3. Metrics JSON
    print("\n2.3 Metrics Summary")
    comparison_metrics = {
        "eonly_baseline": {
            "stable_percent": float(eonly_stable),
            "lockin_percent": float(eonly_lockin),
            "X_peak": float(eonly_X_peak)
        },
        "ei_improvements": {
            i_def: {
                "stable_improvement": float(row["stable_percent"] - eonly_stable),
                "lockin_improvement": float(row["lockin_percent"] - eonly_lockin),
                "X_peak_shift": float(row["X_peak"] - eonly_X_peak)
            }
            for i_def, row in df_ei.set_index("i_definition").iterrows()
        }
    }
    
    with open(os.path.join(output_dir, "eonly_vs_ei_metrics.json"), 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    print("   ✅ eonly_vs_ei_metrics.json")
    
    print("\n✅ E-only vs E+I Comparison Complete!")

