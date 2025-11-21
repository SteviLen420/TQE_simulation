# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Specialized analysis functions with PNG visualization generation

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.ioff()

def analyze_emergent_laws(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze emergent laws (power-laws, phase transitions) with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 1: EMERGENT LAWS COMPARISON")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping emergent laws analysis")
        return
    
    # Phase transitions comparison
    if "n_phase_transitions" in df_ei.columns and df_ei["n_phase_transitions"].notna().any():
        print("\n1.1 Phase Transitions Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["n_phase_transitions"]).sort_values("n_phase_transitions", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["n_phase_transitions"], color='orange', alpha=0.7)
        ax.set_xlabel("Number of Phase Transitions", fontsize=12, fontweight='bold')
        ax.set_title("Emergent Laws: Phase Transitions Detection", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_transitions_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ phase_transitions_comparison.png")
    
    print("\n✅ Emergent Laws Analysis Complete!")

def analyze_friedmann_cosmology(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze Friedmann cosmology consistency with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 2: FRIEDMANN COSMOLOGY")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping Friedmann cosmology analysis")
        return
    
    # Universe age comparison
    if "age_Gyr_mean" in df_ei.columns and df_ei["age_Gyr_mean"].notna().any():
        print("\n2.1 Universe Age Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["age_Gyr_mean"]).sort_values("age_Gyr_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["age_Gyr_mean"], color='blue', alpha=0.7)
        ax.axvline(13.8, color='green', linestyle='--', linewidth=2, label='Planck 2018 (13.8 Gyr)')
        ax.set_xlabel("Mean Universe Age (Gyr)", fontsize=12, fontweight='bold')
        ax.set_title("Friedmann Cosmology: Universe Age Comparison", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "universe_age_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ universe_age_comparison.png")
    else:
        print("   ⚠️  Friedmann cosmology metrics (age_Gyr) not yet extracted")
    
    print("\n✅ Friedmann Cosmology Analysis Complete!")

def analyze_cmb_anomalies(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze CMB anomalies (cold spots, Axis of Evil) with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 3: CMB ANOMALIES")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping CMB anomalies analysis")
        return
    
    # Cold spots comparison
    if "n_coldspots_mean" in df_ei.columns and df_ei["n_coldspots_mean"].notna().any():
        print("\n3.1 Cold Spots Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["n_coldspots_mean"]).sort_values("n_coldspots_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["n_coldspots_mean"], color='blue', alpha=0.7)
        ax.set_xlabel("Mean Number of Cold Spots", fontsize=12, fontweight='bold')
        ax.set_title("CMB Anomalies: Cold Spots Detection", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "coldspots_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ coldspots_comparison.png")
    
    # Physical anomalies comparison
    if "physical_anomaly_count" in df_ei.columns and df_ei["physical_anomaly_count"].notna().any():
        print("\n3.2 Physical Anomalies Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["physical_anomaly_count"]).sort_values("physical_anomaly_count", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["physical_anomaly_count"], color='red', alpha=0.7)
        ax.set_xlabel("Physical Anomaly Count", fontsize=12, fontweight='bold')
        ax.set_title("CMB Anomalies: Physical Anomalies Detection", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "physical_anomalies_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ physical_anomalies_comparison.png")
    
    print("\n✅ CMB Anomalies Analysis Complete!")

def analyze_lockin_dynamics(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze lock-in dynamics and efficiency with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 4: LOCK-IN DYNAMICS")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping lock-in dynamics analysis")
        return
    
    # Lock-in efficiency comparison
    if "lockin_efficiency" in df_ei.columns and df_ei["lockin_efficiency"].notna().any():
        print("\n4.1 Lock-in Efficiency Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["lockin_efficiency"]).sort_values("lockin_efficiency", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["lockin_efficiency"], color='green', alpha=0.7)
        ax.set_xlabel("Lock-in Efficiency", fontsize=12, fontweight='bold')
        ax.set_title("Lock-in Dynamics: Efficiency Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "lockin_efficiency_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ lockin_efficiency_comparison.png")
    
    # Early lock-in rate comparison
    if "early_lockin_rate" in df_ei.columns and df_ei["early_lockin_rate"].notna().any():
        print("\n4.2 Early Lock-in Rate Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["early_lockin_rate"]).sort_values("early_lockin_rate", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["early_lockin_rate"], color='purple', alpha=0.7)
        ax.set_xlabel("Early Lock-in Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title("Lock-in Dynamics: Early Lock-in Rate", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "early_lockin_rate_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ early_lockin_rate_comparison.png")
    
    print("\n✅ Lock-in Dynamics Analysis Complete!")

def analyze_quantum_fields(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze quantum field properties with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 5: QUANTUM FIELDS")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping quantum fields analysis")
        return
    
    # Vacuum energy comparison
    if "vacuum_energy_mean" in df_ei.columns and df_ei["vacuum_energy_mean"].notna().any():
        print("\n5.1 Vacuum Energy Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["vacuum_energy_mean"]).sort_values("vacuum_energy_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["vacuum_energy_mean"], color='purple', alpha=0.7)
        ax.set_xlabel("Mean Vacuum Energy", fontsize=12, fontweight='bold')
        ax.set_title("Quantum Fields: Vacuum Energy Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vacuum_energy_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ vacuum_energy_comparison.png")
    else:
        print("   ⚠️  Quantum field metrics (vacuum_energy) not yet extracted")
    
    print("\n✅ Quantum Fields Analysis Complete!")

def analyze_entanglement(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze entanglement entropy and holographic properties with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 6: ENTANGLEMENT")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping entanglement analysis")
        return
    
    # Entanglement entropy comparison
    if "entanglement_entropy_mean" in df_ei.columns and df_ei["entanglement_entropy_mean"].notna().any():
        print("\n6.1 Entanglement Entropy Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["entanglement_entropy_mean"]).sort_values("entanglement_entropy_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["entanglement_entropy_mean"], color='cyan', alpha=0.7)
        ax.set_xlabel("Mean Entanglement Entropy", fontsize=12, fontweight='bold')
        ax.set_title("Entanglement: Entanglement Entropy Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "entanglement_entropy_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ entanglement_entropy_comparison.png")
    else:
        print("   ⚠️  Entanglement metrics (entanglement_entropy) not yet extracted")
    
    print("\n✅ Entanglement Analysis Complete!")

def analyze_parameter_sensitivity(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze parameter sensitivity (E/I/X) with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 7: PARAMETER SENSITIVITY")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping parameter sensitivity analysis")
        return
    
    # Parameter sensitivity comparison
    has_e_sens = "E_sensitivity" in df_ei.columns and df_ei["E_sensitivity"].notna().any()
    has_i_sens = "I_sensitivity" in df_ei.columns and df_ei["I_sensitivity"].notna().any()
    has_x_sens = "X_sensitivity" in df_ei.columns and df_ei["X_sensitivity"].notna().any()
    
    if has_e_sens or has_i_sens or has_x_sens:
        print("\n7.1 Parameter Sensitivity Comparison")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if has_e_sens:
            df_e = df_ei.dropna(subset=["E_sensitivity"]).sort_values("E_sensitivity", ascending=False)
            axes[0].barh(df_e["i_definition"], df_e["E_sensitivity"], color='red', alpha=0.7)
            axes[0].set_xlabel("E Sensitivity", fontsize=12, fontweight='bold')
            axes[0].set_title("E Parameter Sensitivity", fontsize=14, fontweight='bold')
            axes[0].grid(axis='x', alpha=0.3)
        
        if has_i_sens:
            df_i = df_ei.dropna(subset=["I_sensitivity"]).sort_values("I_sensitivity", ascending=False)
            axes[1].barh(df_i["i_definition"], df_i["I_sensitivity"], color='blue', alpha=0.7)
            axes[1].set_xlabel("I Sensitivity", fontsize=12, fontweight='bold')
            axes[1].set_title("I Parameter Sensitivity", fontsize=14, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
        
        if has_x_sens:
            df_x = df_ei.dropna(subset=["X_sensitivity"]).sort_values("X_sensitivity", ascending=False)
            axes[2].barh(df_x["i_definition"], df_x["X_sensitivity"], color='green', alpha=0.7)
            axes[2].set_xlabel("X Sensitivity", fontsize=12, fontweight='bold')
            axes[2].set_title("X Parameter Sensitivity", fontsize=14, fontweight='bold')
            axes[2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_sensitivity_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ parameter_sensitivity_comparison.png")
    else:
        print("   ⚠️  Parameter sensitivity metrics (E/I/X_sensitivity) not yet extracted")
    
    print("\n✅ Parameter Sensitivity Analysis Complete!")

def analyze_topology(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze topological properties with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 8: TOPOLOGY")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping topology analysis")
        return
    
    # Topological defects comparison
    if "topological_defect_rate" in df_ei.columns and df_ei["topological_defect_rate"].notna().any():
        print("\n8.1 Topological Defects Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["topological_defect_rate"]).sort_values("topological_defect_rate", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["topological_defect_rate"], color='brown', alpha=0.7)
        ax.set_xlabel("Topological Defect Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title("Topology: Topological Defects Detection", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "topological_defects_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ topological_defects_comparison.png")
    
    print("\n✅ Topology Analysis Complete!")

def analyze_i_definitions_direct(df_metrics: pd.DataFrame, collected_data: dict, output_dir: str, config: dict):
    """Analyze I-definitions directly from comparison data with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 9: I-DEFINITIONS DIRECT")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping I-definitions direct analysis")
        return
    
    # I-value comparison
    if "I_value_mean" in df_ei.columns and df_ei["I_value_mean"].notna().any():
        print("\n9.1 I-Value Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["I_value_mean"]).sort_values("I_value_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["I_value_mean"], color='cyan', alpha=0.7)
        ax.set_xlabel("Mean I Value", fontsize=12, fontweight='bold')
        ax.set_title("I-Definitions: Mean I Value Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "i_value_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ i_value_comparison.png")
    
    print("\n✅ I-Definitions Direct Analysis Complete!")

def analyze_planck_fit(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze Planck fit quality with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 10: PLANCK FIT")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping Planck fit analysis")
        return
    
    # Planck validation chi2 comparison
    if "planck_validation_chi2_mean" in df_ei.columns and df_ei["planck_validation_chi2_mean"].notna().any():
        print("\n10.1 Planck Validation χ² Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["planck_validation_chi2_mean"]).sort_values("planck_validation_chi2_mean", ascending=True)
        ax.barh(df_plot["i_definition"], df_plot["planck_validation_chi2_mean"], color='purple', alpha=0.7)
        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect fit (χ²=1)')
        ax.set_xlabel("Mean Planck Validation χ²", fontsize=12, fontweight='bold')
        ax.set_title("Planck Fit: Validation χ² Comparison", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "planck_validation_chi2_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ planck_validation_chi2_comparison.png")
    
    print("\n✅ Planck Fit Analysis Complete!")

def analyze_life_top_universes(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze life-compatible top universes with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 11: LIFE TOP UNIVERSES")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping life top universes analysis")
        return
    
    # Life compatibility score comparison
    if "life_compatibility_score" in df_ei.columns and df_ei["life_compatibility_score"].notna().any():
        print("\n11.1 Life Compatibility Score Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["life_compatibility_score"]).sort_values("life_compatibility_score", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["life_compatibility_score"], color='green', alpha=0.7)
        ax.set_xlabel("Life Compatibility Score", fontsize=12, fontweight='bold')
        ax.set_title("Life Top Universes: Compatibility Score Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "life_compatibility_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ life_compatibility_comparison.png")
    
    print("\n✅ Life Top Universes Analysis Complete!")

def analyze_entropy_volatility(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze entropy volatility patterns with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 12: ENTROPY VOLATILITY")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping entropy volatility analysis")
        return
    
    # Entropy volatility comparison
    if "entropy_volatility_global_mean" in df_ei.columns and df_ei["entropy_volatility_global_mean"].notna().any():
        print("\n12.1 Entropy Volatility Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["entropy_volatility_global_mean"]).sort_values("entropy_volatility_global_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["entropy_volatility_global_mean"], color='orange', alpha=0.7)
        ax.set_xlabel("Global Mean Entropy Volatility", fontsize=12, fontweight='bold')
        ax.set_title("Entropy Volatility: Global Mean Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "entropy_volatility_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ entropy_volatility_comparison.png")
    
    print("\n✅ Entropy Volatility Analysis Complete!")

def analyze_physical_anomalies(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """Analyze physical anomalies with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 13: PHYSICAL ANOMALIES")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping physical anomalies analysis")
        return
    
    # Advanced anomaly sigma comparison
    if "advanced_anomaly_sigma_mean" in df_ei.columns and df_ei["advanced_anomaly_sigma_mean"].notna().any():
        print("\n13.1 Advanced Anomaly Sigma Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["advanced_anomaly_sigma_mean"]).sort_values("advanced_anomaly_sigma_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["advanced_anomaly_sigma_mean"], color='red', alpha=0.7)
        ax.set_xlabel("Mean Advanced Anomaly Sigma", fontsize=12, fontweight='bold')
        ax.set_title("Physical Anomalies: Advanced Anomaly Detection", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "advanced_anomaly_sigma_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ advanced_anomaly_sigma_comparison.png")
    
    print("\n✅ Physical Anomalies Analysis Complete!")

def analyze_statistical_finetuning(df_metrics: pd.DataFrame, collected_data: dict, output_dir: str, config: dict):
    """Analyze statistical fine-tuning with PNG generation."""
    print("\n" + "="*70)
    print("CATEGORY 14: STATISTICAL FINETUNING")
    print("="*70)
    os.makedirs(output_dir, exist_ok=True)
    figure_dpi = config.get("FIGURE_DPI", 300)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping statistical finetuning analysis")
        return
    
    # Statistical finetuning rate comparison
    if "statistical_finetuning_rate_mean" in df_ei.columns and df_ei["statistical_finetuning_rate_mean"].notna().any():
        print("\n14.1 Statistical Finetuning Rate Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df_ei.dropna(subset=["statistical_finetuning_rate_mean"]).sort_values("statistical_finetuning_rate_mean", ascending=False)
        ax.barh(df_plot["i_definition"], df_plot["statistical_finetuning_rate_mean"], color='orange', alpha=0.7)
        ax.set_xlabel("Mean Statistical Finetuning Rate (%)", fontsize=12, fontweight='bold')
        ax.set_title("Statistical Finetuning: Mean Rate Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "statistical_finetuning_comparison.png"), dpi=figure_dpi, bbox_inches='tight')
        plt.close()
        print("   ✅ statistical_finetuning_comparison.png")
    else:
        print("   ⚠️  Statistical finetuning metrics not yet extracted")
        print("   ℹ️  Waiting for statistical_finetuning_summary.csv data")
    
    print("\n✅ Statistical Finetuning Analysis Complete!")
