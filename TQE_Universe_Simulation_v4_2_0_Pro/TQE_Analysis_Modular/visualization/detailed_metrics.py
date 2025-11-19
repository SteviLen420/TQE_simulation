# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Detailed metrics visualization functions

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
plt.ioff()

def generate_detailed_metrics(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """
    PHASE 3C: Generate detailed statistical metrics analysis.
    
    Comprehensive statistical analysis including:
    - Full metrics export (CSV with all runs and all metrics)
    - Correlation matrix (identify interdependencies)
    - Distribution analysis (box plots showing quartiles, outliers)
    - Comparison against E-only baseline (if available)
    
    Generates:
    - all_runs_metrics.csv (comprehensive table)
    - correlation_matrix.png (heatmap of metric correlations)
    - distributions_boxplot.png (stability, lock-in, X_peak, uncertainty distributions)
    """
    print("\n" + "="*70)
    print("PHASE 3C: DETAILED STATISTICAL METRICS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    # 1. Save comprehensive metrics CSV
    print("\n3.1 Comprehensive Metrics Table")
    df_metrics.to_csv(os.path.join(output_dir, "all_runs_metrics.csv"), index=False)
    print("   ✅ all_runs_metrics.csv")
    
    # 2. Correlation matrix (E+I only)
    print("\n3.2 Correlation Matrix")
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    if len(df_ei) > 2:
        numeric_cols = ["stable_percent", "lockin_percent", "X_peak", "X_peak_uncertainty", 
                       "goldilocks_width", "chi_squared_reduced"]
        df_corr = df_ei[numeric_cols].dropna()
        
        if len(df_corr) > 2:
            corr_matrix = df_corr.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Correlation Matrix (E+I runs)", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=figure_dpi, bbox_inches='tight')
            plt.close()
            print("   ✅ correlation_matrix.png")
        else:
            print("   ⚠️  Insufficient data for correlation matrix")
    
    # 3. Box plots
    print("\n3.3 Distribution Box Plots")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    # Stable %
    axes[0, 0].boxplot([df_ei["stable_percent"]], labels=['E+I'])
    if len(df_metrics[df_metrics["run_type"] == "E-only"]) > 0:
        eonly_stable = df_metrics[df_metrics["run_type"] == "E-only"]["stable_percent"].iloc[0]
        axes[0, 0].axhline(eonly_stable, color='red', linestyle='--', label='E-only baseline')
        axes[0, 0].legend()
    axes[0, 0].set_ylabel("Stable %", fontweight='bold')
    axes[0, 0].set_title("Stability Rate Distribution", fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Lock-in %
    axes[0, 1].boxplot([df_ei["lockin_percent"]], labels=['E+I'])
    if len(df_metrics[df_metrics["run_type"] == "E-only"]) > 0:
        eonly_lockin = df_metrics[df_metrics["run_type"] == "E-only"]["lockin_percent"].iloc[0]
        axes[0, 1].axhline(eonly_lockin, color='red', linestyle='--', label='E-only baseline')
        axes[0, 1].legend()
    axes[0, 1].set_ylabel("Lock-in %", fontweight='bold')
    axes[0, 1].set_title("Lock-in Rate Distribution", fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # X_peak
    axes[1, 0].boxplot([df_ei["X_peak"]], labels=['E+I'])
    axes[1, 0].set_ylabel("X_peak", fontweight='bold')
    axes[1, 0].set_title("Goldilocks Peak Distribution", fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # X_peak_uncertainty
    axes[1, 1].boxplot([df_ei["X_peak_uncertainty"]], labels=['E+I'])
    axes[1, 1].set_ylabel("X_peak uncertainty (σ)", fontweight='bold')
    axes[1, 1].set_title("Peak Uncertainty Distribution", fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distributions_boxplot.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ distributions_boxplot.png")
    
    print("\n✅ Detailed Metrics Complete!")

