# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Advanced visualization functions

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
plt.ioff()

def generate_advanced_visualizations(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """
    PHASE 4: Generate advanced multi-dimensional visualizations.
    
    Creates publication-quality visualizations for comparative analysis:
    - Radar chart: Multi-metric comparison on normalized 0-100 scale (top 5 models)
    - Performance heatmap: Color-coded strength/weakness matrix
    - Scatter plots: Goldilocks peak vs stability rate relationships
    
    Generates:
    - radar_chart_top5.png (spider plot showing top 5 I-definitions)
    - heatmap_performance.png (green=good, red=poor performance matrix)
    - scatter_X_peak_vs_stability.png (peak position vs stability correlation)
    """
    print("\n" + "="*70)
    print("PHASE 4: ADVANCED VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    # 1. Radar chart
    print("\n4.1 Radar Chart (Spider Plot)")
    
    # Normalize metrics to 0-1 scale for radar chart
    df_radar = df_ei[["i_definition", "stable_percent", "lockin_percent"]].copy()
    df_radar["precision"] = 100 - (df_ei["X_peak_uncertainty"] / df_ei["X_peak"] * 100).clip(0, 100)  # Higher is better
    
    # Add chi-squared if available (invert: lower is better → higher score)
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_min = df_ei["chi_squared_reduced"].min()
        chi_max = df_ei["chi_squared_reduced"].max()
        if chi_max > chi_min:
            df_radar["planck_fit"] = 100 * (1 - (df_ei["chi_squared_reduced"] - chi_min) / (chi_max - chi_min))
        else:
            df_radar["planck_fit"] = 50.0
    else:
        df_radar["planck_fit"] = 50.0  # Neutral if no data
    
    # Select top 5 models for clarity
    df_radar_top5 = df_radar.nlargest(5, "stable_percent")
    
    categories = ['Stability', 'Lock-in', 'Precision', 'Planck Fit']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_radar_top5)))
    
    for i, (idx, row) in enumerate(df_radar_top5.iterrows()):
        values = [row["stable_percent"], row["lockin_percent"], row["precision"], row["planck_fit"]]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row["i_definition"], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title("Top 5 I-Definitions: Multi-Metric Radar Chart", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart_top5.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ radar_chart_top5.png")
    
    # 2. Performance heatmap
    print("\n4.2 Performance Heatmap")
    
    # Normalize all metrics to 0-100 scale
    df_heatmap = pd.DataFrame()
    df_heatmap['I-Definition'] = df_ei["i_definition"]
    df_heatmap['Stability'] = df_ei["stable_percent"]
    df_heatmap['Lock-in'] = df_ei["lockin_percent"]
    df_heatmap['Precision'] = 100 - (df_ei["X_peak_uncertainty"] / df_ei["X_peak"] * 100).clip(0, 100)
    
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_min = df_ei["chi_squared_reduced"].min()
        chi_max = df_ei["chi_squared_reduced"].max()
        if chi_max > chi_min:
            df_heatmap['Planck Fit'] = 100 * (1 - (df_ei["chi_squared_reduced"] - chi_min) / (chi_max - chi_min))
        else:
            df_heatmap['Planck Fit'] = 50.0
    
    df_heatmap = df_heatmap.set_index('I-Definition')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_heatmap.T, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
               square=False, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Performance Heatmap (0-100 scale, higher is better)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metrics", fontsize=12, fontweight='bold')
    ax.set_xlabel("I-Definitions", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_performance.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ heatmap_performance.png")
    
    # 3. Scatter: X_peak vs Stability
    print("\n4.3 Scatter: X_peak vs Stability Rate")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, row in df_ei.iterrows():
        ax.errorbar(row["X_peak"], row["stable_percent"], 
                   xerr=row["X_peak_uncertainty"]*1.96, 
                   fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2,
                   label=row["i_definition"], alpha=0.7)
    
    ax.set_xlabel("X_peak (Goldilocks Peak)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Stability Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Goldilocks Peak vs Stability Rate", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_X_peak_vs_stability.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ scatter_X_peak_vs_stability.png")
    
    print("\n✅ Advanced Visualizations Complete!")


def generate_complexity_analysis(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """
    PHASE 4B: Generate complexity and life-compatibility analysis.
    
    Critical insight: More stable ≠ Better!
    - E-only may be stable but chaotic (no complexity)
    - E+I may be less stable but more complex (life-compatible)
    
    Generates:
    - complexity_vs_stability.png (scatter: complexity score vs stability %)
    - life_compatibility_comparison.png (bar chart across I-definitions)
    """
    print("\n" + "="*70)
    print("PHASE 4B: COMPLEXITY & LIFE-COMPATIBILITY ANALYSIS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    # 1. Complexity vs Stability scatter
    print("\n4B.1 Complexity vs Stability Scatter")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, row in df_metrics.iterrows():
        color = 'red' if row['run_type'] == 'E-only' else 'blue'
        marker = 's' if row['run_type'] == 'E-only' else 'o'
        ax.scatter(row['stable_percent'], row.get('complexity_score', 0), 
                  color=color, marker=marker, s=100, alpha=0.7, 
                  label=row['i_definition'] if row['run_type'] == 'E+I' else 'E-only')
    
    ax.set_xlabel("Stability Rate (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Complexity Score", fontsize=12, fontweight='bold')
    ax.set_title("Complexity vs Stability (Higher complexity = Better for TQE)", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complexity_vs_stability.png"), dpi=figure_dpi, bbox_inches='tight')
    plt.close()
    print("   ✅ complexity_vs_stability.png")
    
    # 2. Life compatibility comparison
    print("\n4B.2 Life Compatibility Comparison")
    if "life_compatibility_score" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_sorted = df_ei.sort_values("life_compatibility_score", ascending=False)
        ax.barh(df_sorted["i_definition"], df_sorted["life_compatibility_score"], color='green', alpha=0.7)
        ax.set_xlabel("Life Compatibility Score", fontsize=12, fontweight='bold')
        ax.set_title("Life Compatibility Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "life_compatibility_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ life_compatibility_comparison.png")
    
    print("\n✅ Complexity Analysis Complete!")

