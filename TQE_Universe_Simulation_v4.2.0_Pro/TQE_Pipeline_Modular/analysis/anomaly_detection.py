# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Anomaly detection module
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext

def _detect_quantum_field_anomalies(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect quantum field theory anomalies."""
    anomalies = []
    
    # Quantum fluctuation anomalies
    if 'quantum_fluctuation' in df.columns:
        qf_mean = df['quantum_fluctuation'].mean()
        qf_std = df['quantum_fluctuation'].std()
        threshold = ctx.config.get('ANOMALY_DETECTION_THRESHOLD', 3.0)
        
        # OPTIMIZED: Vectorized anomaly detection (100× faster than iterrows)
        qf_values = df['quantum_fluctuation'].values
        universe_ids = df['universe_id'].values if 'universe_id' in df.columns else np.arange(len(df))
        deviations = np.abs(qf_values - qf_mean)
        dev_sigma = deviations / qf_std
        anomaly_mask = deviations > threshold * qf_std
        
        for idx in np.where(anomaly_mask)[0]:
                anomalies.append({
                'universe_id': universe_ids[idx],
                    'anomaly_type': 'quantum_field_fluctuation',
                'anomaly_value': qf_values[idx],
                    'expected_value': qf_mean,
                'deviation_sigma': dev_sigma[idx],
                'significance': 'high' if dev_sigma[idx] > 5 else 'medium'
                })
    
    return anomalies


def _detect_entropy_anomalies(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect entropy fluctuation anomalies."""
    anomalies = []
    
    # Entropy volatility anomalies
    if 'entropy_volatility' in df.columns:
        ev_mean = df['entropy_volatility'].mean()
        ev_std = df['entropy_volatility'].std()
        threshold = ctx.config.get('ANOMALY_DETECTION_THRESHOLD', 3.0)
        
        # OPTIMIZED: Vectorized anomaly detection (100× faster than iterrows)
        values = df['entropy_volatility'].values
        universe_ids = df['universe_id'].values if 'universe_id' in df.columns else np.arange(len(df))
        deviations = np.abs(values - ev_mean)
        dev_sigma = deviations / ev_std
        anomaly_mask = deviations > threshold * ev_std
        
        for idx in np.where(anomaly_mask)[0]:
                anomalies.append({
                'universe_id': universe_ids[idx],
                    'anomaly_type': 'entropy_volatility',
                'anomaly_value': values[idx],
                    'expected_value': ev_mean,
                'deviation_sigma': dev_sigma[idx],
                'significance': 'high' if dev_sigma[idx] > 5 else 'medium'
                })
    
    return anomalies


def _detect_topological_anomalies(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect topological defect anomalies."""
    anomalies = []
    
    # Topological defect density anomalies
    if 'topological_defect_density' in df.columns:
        td_mean = df['topological_defect_density'].mean()
        td_std = df['topological_defect_density'].std()
        threshold = ctx.config.get('ANOMALY_DETECTION_THRESHOLD', 3.0)
        
        # OPTIMIZED: Vectorized anomaly detection (100× faster than iterrows)
        values = df['topological_defect_density'].values
        universe_ids = df['universe_id'].values if 'universe_id' in df.columns else np.arange(len(df))
        deviations = np.abs(values - td_mean)
        dev_sigma = deviations / td_std
        anomaly_mask = deviations > threshold * td_std
        
        for idx in np.where(anomaly_mask)[0]:
                anomalies.append({
                'universe_id': universe_ids[idx],
                    'anomaly_type': 'topological_defect_density',
                'anomaly_value': values[idx],
                    'expected_value': td_mean,
                'deviation_sigma': dev_sigma[idx],
                'significance': 'high' if dev_sigma[idx] > 5 else 'medium'
                })
    
    return anomalies


def _detect_energy_anomalies(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect energy conservation anomalies."""
    anomalies = []
    
    # Energy conservation violations
    if 'E' in df.columns and 'I' in df.columns:
        E_mean = df['E'].mean()
        E_std = df['E'].std()
        threshold = ctx.config.get('ANOMALY_DETECTION_THRESHOLD', 3.0)
        
        # OPTIMIZED: Vectorized anomaly detection (100× faster than iterrows)
        values = df['E'].values
        universe_ids = df['universe_id'].values if 'universe_id' in df.columns else np.arange(len(df))
        deviations = np.abs(values - E_mean)
        dev_sigma = deviations / E_std
        anomaly_mask = deviations > threshold * E_std
        
        for idx in np.where(anomaly_mask)[0]:
                anomalies.append({
                'universe_id': universe_ids[idx],
                    'anomaly_type': 'energy_conservation',
                'anomaly_value': values[idx],
                    'expected_value': E_mean,
                'deviation_sigma': dev_sigma[idx],
                'significance': 'high' if dev_sigma[idx] > 5 else 'medium'
                })
    
    return anomalies


def _detect_information_anomalies(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect information theory anomalies."""
    anomalies = []
    
    # Information entropy anomalies
    if 'I' in df.columns:
        I_mean = df['I'].mean()
        I_std = df['I'].std()
        threshold = ctx.config.get('ANOMALY_DETECTION_THRESHOLD', 3.0)
        
        # OPTIMIZED: Vectorized anomaly detection (100× faster than iterrows)
        values = df['I'].values
        universe_ids = df['universe_id'].values if 'universe_id' in df.columns else np.arange(len(df))
        deviations = np.abs(values - I_mean)
        dev_sigma = deviations / I_std
        anomaly_mask = deviations > threshold * I_std
        
        for idx in np.where(anomaly_mask)[0]:
                anomalies.append({
                'universe_id': universe_ids[idx],
                    'anomaly_type': 'information_entropy',
                'anomaly_value': values[idx],
                    'expected_value': I_mean,
                'deviation_sigma': dev_sigma[idx],
                'significance': 'high' if dev_sigma[idx] > 5 else 'medium'
                })
    
    return anomalies


def _detect_cmb_statistical_anomalies(df: pd.DataFrame, ctx: PipelineContext) -> list:
    """Detect CMB statistical anomalies."""
    anomalies = []
    
    # CMB power spectrum anomalies
    if 'cmb_power_spectrum' in df.columns:
        ps_mean = df['cmb_power_spectrum'].mean()
        ps_std = df['cmb_power_spectrum'].std()
        threshold = ctx.config.get('ANOMALY_DETECTION_THRESHOLD', 3.0)
        
        # OPTIMIZED: Vectorized anomaly detection (100× faster than iterrows)
        values = df['cmb_power_spectrum'].values
        universe_ids = df['universe_id'].values if 'universe_id' in df.columns else np.arange(len(df))
        deviations = np.abs(values - ps_mean)
        dev_sigma = deviations / ps_std
        anomaly_mask = deviations > threshold * ps_std
        
        for idx in np.where(anomaly_mask)[0]:
                anomalies.append({
                'universe_id': universe_ids[idx],
                    'anomaly_type': 'cmb_power_spectrum',
                'anomaly_value': values[idx],
                    'expected_value': ps_mean,
                'deviation_sigma': dev_sigma[idx],
                'significance': 'high' if dev_sigma[idx] > 5 else 'medium'
                })
    
    return anomalies
# ======================================================
# LAW DETECTION HELPER FUNCTIONS
# ======================================================


def _create_anomaly_detection_plots(anomaly_df: pd.DataFrame, ctx: PipelineContext):
    """Create comprehensive anomaly detection visualization plots."""
    if anomaly_df.empty:
        return
    
    # Apply consistent plot style
    setup_scientific_plotting_style(ctx.config)
    
    # 1. Anomaly Type Distribution
    # PUBLICATION: Larger figsize for 2x2 subplots (was: 12,10)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14), 
                                                   constrained_layout=True)  # Auto-adjust spacing
    
    # Anomaly type distribution
    anomaly_counts = anomaly_df['anomaly_type'].value_counts()
    # Clean labels: remove underscores
    clean_labels = [label.replace('_', ' ') for label in anomaly_counts.index]
    ax1.bar(range(len(anomaly_counts)), anomaly_counts.values, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xticks(range(len(anomaly_counts)))
    ax1.set_xticklabels(clean_labels, rotation=45, ha='right')
    apply_consistent_plot_style(ax1, "Anomaly Type Distribution", "Anomaly Type", "Count", ctx.config)
    
    # Significance distribution
    significance_counts = anomaly_df['significance'].value_counts()
    colors = ['red' if s == 'high' else 'orange' if s == 'medium' else 'green' for s in significance_counts.index]
    ax2.bar(significance_counts.index, significance_counts.values, color=colors, edgecolor='black', alpha=0.7)
    apply_consistent_plot_style(ax2, "Anomaly Significance Distribution", "Significance Level", "Count", ctx.config)
    
    # Deviation sigma distribution
    ax3.hist(anomaly_df['deviation_sigma'], bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    apply_consistent_plot_style(ax3, "Deviation Sigma Distribution", "Deviation (σ)", "Frequency", ctx.config)
    
    # Anomaly value vs expected value
    ax4.scatter(anomaly_df['expected_value'], anomaly_df['anomaly_value'], 
               c=anomaly_df['deviation_sigma'], cmap='viridis', alpha=0.7, s=50)
    apply_consistent_plot_style(ax4, "Anomaly vs Expected Values", "Expected Value", "Anomaly Value", ctx.config)
    plt.colorbar(ax4.collections[0], ax=ax4, label='Deviation (σ)')
    
    # NOTE: constrained_layout=True already set, no need for plt.tight_layout()
    
    # Save plot with categorization
    ctx.save_fig("advanced_anomaly_detection_analysis.png", category="anomaly")
    
    if ctx.config.get("VERBOSE", True):
        print(f"[ANOMALY PLOTS] Analysis plot saved with categorization")


