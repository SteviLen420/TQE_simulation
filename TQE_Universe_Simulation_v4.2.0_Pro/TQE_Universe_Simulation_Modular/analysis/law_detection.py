# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Law detection module
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext

def _detect_conservation_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect conservation laws (energy, momentum, charge)."""
    laws = []
    
    # Energy conservation law
    if 'E' in df.columns:
        E_conservation = df['E'].std() / df['E'].mean()
        laws.append({
            'law_type': 'energy_conservation',
            'law_strength': 1.0 / (1.0 + E_conservation),
            'law_quality': 'excellent' if E_conservation < 0.1 else 'good' if E_conservation < 0.2 else 'fair',
            'statistical_significance': 1.0 - E_conservation,
            'universe_count': len(df)
        })
    
    return laws


def _detect_symmetry_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect symmetry breaking laws."""
    laws = []
    
    # E-I symmetry
    if 'E' in df.columns and 'I' in df.columns:
        E_I_correlation = df['E'].corr(df['I'])
        symmetry_breaking = abs(E_I_correlation)
        laws.append({
            'law_type': 'E_I_symmetry',
            'law_strength': symmetry_breaking,
            'law_quality': 'excellent' if symmetry_breaking > 0.8 else 'good' if symmetry_breaking > 0.6 else 'fair',
            'statistical_significance': symmetry_breaking,
            'universe_count': len(df)
        })
    
    return laws


def _detect_scaling_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect scaling laws and power laws."""
    laws = []
    
    # X scaling law
    if 'X' in df.columns and 'stable' in df.columns:
        from scipy.stats import spearmanr
        correlation, p_value = spearmanr(df['X'], df['stable'])
        laws.append({
            'law_type': 'X_stability_scaling',
            'law_strength': abs(correlation),
            'law_quality': 'excellent' if abs(correlation) > 0.8 else 'good' if abs(correlation) > 0.6 else 'fair',
            'statistical_significance': 1.0 - p_value,
            'universe_count': len(df)
        })
    
    return laws


def _detect_emergent_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect emergent behavior laws."""
    laws = []
    
    # Lock-in emergence law
    if 'lock_epoch' in df.columns:
        lockin_rate = (df['lock_epoch'] >= 0).mean()
        laws.append({
            'law_type': 'lock_in_emergence',
            'law_strength': lockin_rate,
            'law_quality': 'excellent' if lockin_rate > 0.7 else 'good' if lockin_rate > 0.5 else 'fair',
            'statistical_significance': lockin_rate,
            'universe_count': len(df)
        })
    
    return laws


def _detect_quantum_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect quantum mechanical laws."""
    laws = []
    
    # Quantum uncertainty principle
    if 'E' in df.columns and 'I' in df.columns:
        uncertainty_product = (df['E'] * df['I']).mean()
        laws.append({
            'law_type': 'quantum_uncertainty',
            'law_strength': uncertainty_product,
            'law_quality': 'excellent' if uncertainty_product > 0.1 else 'good' if uncertainty_product > 0.05 else 'fair',
            'statistical_significance': uncertainty_product,
            'universe_count': len(df)
        })
    
    return laws


def _detect_thermodynamic_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect thermodynamic laws."""
    laws = []
    
    # Entropy increase law
    if 'entropy' in df.columns:
        entropy_increase = df['entropy'].diff().mean()
        laws.append({
            'law_type': 'entropy_increase',
            'law_strength': max(0, entropy_increase),
            'law_quality': 'excellent' if entropy_increase > 0.01 else 'good' if entropy_increase > 0.005 else 'fair',
            'statistical_significance': max(0, entropy_increase),
            'universe_count': len(df)
        })
    
    return laws


def _detect_statistical_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect statistical mechanics laws."""
    laws = []
    
    # Boltzmann distribution
    if 'E' in df.columns:
        E_std = df['E'].std()
        E_mean = df['E'].mean()
        boltzmann_quality = 1.0 / (1.0 + abs(E_std - E_mean * 0.1))
        laws.append({
            'law_type': 'boltzmann_distribution',
            'law_strength': boltzmann_quality,
            'law_quality': 'excellent' if boltzmann_quality > 0.9 else 'good' if boltzmann_quality > 0.7 else 'fair',
            'statistical_significance': boltzmann_quality,
            'universe_count': len(df)
        })
    
    return laws


def _detect_field_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect field theory laws."""
    laws = []
    
    # Field correlation law
    if 'E' in df.columns and 'I' in df.columns:
        field_correlation = abs(df['E'].corr(df['I']))
        laws.append({
            'law_type': 'field_correlation',
            'law_strength': field_correlation,
            'law_quality': 'excellent' if field_correlation > 0.8 else 'good' if field_correlation > 0.6 else 'fair',
            'statistical_significance': field_correlation,
            'universe_count': len(df)
        })
    
    return laws


def _detect_geometric_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect geometric and topological laws."""
    laws = []
    
    # Geometric scaling law
    if 'X' in df.columns:
        X_geometric_mean = df['X'].apply(lambda x: np.sqrt(x) if x > 0 else 0).mean()
        laws.append({
            'law_type': 'geometric_scaling',
            'law_strength': X_geometric_mean,
            'law_quality': 'excellent' if X_geometric_mean > 5.0 else 'good' if X_geometric_mean > 3.0 else 'fair',
            'statistical_significance': X_geometric_mean / 10.0,
            'universe_count': len(df)
        })
    
    return laws


def _detect_information_laws(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect information theory laws."""
    laws = []
    
    # Information conservation law
    if 'I' in df.columns:
        I_conservation = df['I'].std() / df['I'].mean()
        laws.append({
            'law_type': 'information_conservation',
            'law_strength': 1.0 / (1.0 + I_conservation),
            'law_quality': 'excellent' if I_conservation < 0.1 else 'good' if I_conservation < 0.2 else 'fair',
            'statistical_significance': 1.0 - I_conservation,
            'universe_count': len(df)
        })
    
    return laws

# ======================================================
# VISUALIZATION HELPER FUNCTIONS
# ======================================================


def _create_law_detection_plots(law_df: pd.DataFrame, ctx: PipelineContext):
    """Create comprehensive law detection visualization plots."""
    if law_df.empty:
        return
    
    # Apply consistent plot style
    setup_scientific_plotting_style(ctx.config)
    
    # 1. Law Type Analysis
    # PUBLICATION: Larger figsize for 2x2 subplots (was: 12,10)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14),
                                                   constrained_layout=True)  # Auto-adjust spacing
    
    # Law type distribution
    law_counts = law_df['law_type'].value_counts()
    # Clean labels: remove underscores
    clean_labels = [label.replace('_', ' ') for label in law_counts.index]
    ax1.bar(range(len(law_counts)), law_counts.values, color='lightgreen', edgecolor='black', alpha=0.7)
    ax1.set_xticks(range(len(law_counts)))
    ax1.set_xticklabels(clean_labels, rotation=45, ha='right')
    apply_consistent_plot_style(ax1, "Law Type Distribution", "Law Type", "Count", ctx.config)
    
    # Law quality distribution
    quality_counts = law_df['law_quality'].value_counts()
    colors = ['green' if q == 'excellent' else 'orange' if q == 'good' else 'red' for q in quality_counts.index]
    ax2.bar(quality_counts.index, quality_counts.values, color=colors, edgecolor='black', alpha=0.7)
    apply_consistent_plot_style(ax2, "Law Quality Distribution", "Quality Level", "Count", ctx.config)
    
    # Law strength distribution
    ax3.hist(law_df['law_strength'], bins=20, color='lightblue', edgecolor='black', alpha=0.7)
    apply_consistent_plot_style(ax3, "Law Strength Distribution", "Law Strength", "Frequency", ctx.config)
    
    # Statistical significance vs law strength
    ax4.scatter(law_df['statistical_significance'], law_df['law_strength'], 
               c=law_df['law_strength'], cmap='plasma', alpha=0.7, s=50)
    apply_consistent_plot_style(ax4, "Statistical Significance vs Law Strength", "Statistical Significance", "Law Strength", ctx.config)
    plt.colorbar(ax4.collections[0], ax=ax4, label='Law Strength')
    
    # NOTE: constrained_layout=True already set, no need for plt.tight_layout()
    
    # Save plot with categorization
    ctx.save_fig("advanced_law_detection_analysis.png", category="laws")
    
    if ctx.config.get("VERBOSE", True):
        print(f"[LAW PLOTS] Analysis plot saved with categorization")


# ==========================================================================================
# BAYESIAN MODEL SELECTION (PRO FEATURES)
# ==========================================================================================


