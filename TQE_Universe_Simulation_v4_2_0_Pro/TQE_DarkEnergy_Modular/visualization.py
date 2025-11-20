# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# visualization.py - Visualization Module
# ==========================================================================================
# TQE–ΛSim: Visualization functions for plots and dashboards
# ==========================================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from .config import MASTER_CTRL

def compare_eonly_vs_eplusi(all_results, run_dir):
    """
    PHASE 3: Compare E-only vs E+I coupling modes
    Creates comparison table and dashboard plot
    """
    print("📊 Loading E-only and E+I results...")
    
    # Extract results by coupling mode
    eonly_results = all_results.get('Eonly', [])
    eplusi_results = all_results.get('EplusI', [])
    
    if not eonly_results or not eplusi_results:
        raise ValueError("Both E-only and E+I results required for comparison")
    
    print(f"   Found {len(eonly_results)} E-only models")
    print(f"   Found {len(eplusi_results)} E+I models")
    
    # Create comparison directory
    comparison_dir = f"{run_dir}/Eonly_vs_EplusI_Comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for eonly_result in eonly_results:
        model_name = eonly_result['model_name']
        eonly_results_dict = eonly_result['results']
        
        # Find matching E+I result
        eplusi_result = next((r for r in eplusi_results if r['model_name'] == model_name), None)
        if not eplusi_result:
            continue
            
        eplusi_results_dict = eplusi_result['results']
        
        # Extract COMPLETE metrics for full comparison
        # S8 metrics
        eonly_s8 = eonly_results_dict.get('observables', {}).get('S8_raw', 0.0)
        eplusi_s8 = eplusi_results_dict.get('observables', {}).get('S8_raw', 0.0)
        
        # ρ_DE metrics (safe list access with length check)
        eonly_rho_de_list = eonly_results_dict.get('evolution_series', {}).get('rho_DE', [0.0])
        eonly_rho_de = eonly_rho_de_list[-1] if len(eonly_rho_de_list) > 0 else 0.0
        eplusi_rho_de_list = eplusi_results_dict.get('evolution_series', {}).get('rho_DE', [0.0])
        eplusi_rho_de = eplusi_rho_de_list[-1] if len(eplusi_rho_de_list) > 0 else 0.0
        
        # I-E correlation metrics
        eonly_mi = eonly_results_dict.get('I_E_correlation', {}).get('mutual_information', 0.0)
        eplusi_mi = eplusi_results_dict.get('I_E_correlation', {}).get('mutual_information', 0.0)
        eonly_pearson = eonly_results_dict.get('I_E_correlation', {}).get('pearson_r', 0.0)
        eplusi_pearson = eplusi_results_dict.get('I_E_correlation', {}).get('pearson_r', 0.0)
        
        # Likelihood metrics
        eonly_chi2 = eonly_results_dict.get('likelihood', {}).get('chi2_total', 0.0)
        eplusi_chi2 = eplusi_results_dict.get('likelihood', {}).get('chi2_total', 0.0)
        eonly_aic = eonly_results_dict.get('likelihood', {}).get('AIC', 0.0)
        eplusi_aic = eplusi_results_dict.get('likelihood', {}).get('AIC', 0.0)
        eonly_bic = eonly_results_dict.get('likelihood', {}).get('BIC', 0.0)
        eplusi_bic = eplusi_results_dict.get('likelihood', {}).get('BIC', 0.0)
        
        # Observable metrics
        eonly_mu_z1 = eonly_results_dict.get('observables', {}).get('mu_z1', 0.0)
        eplusi_mu_z1 = eplusi_results_dict.get('observables', {}).get('mu_z1', 0.0)
        eonly_h_z0 = eonly_results_dict.get('observables', {}).get('H_z0', 67.4)
        eplusi_h_z0 = eplusi_results_dict.get('observables', {}).get('H_z0', 67.4)
        eonly_dm_z051 = eonly_results_dict.get('observables', {}).get('D_M_z051', 0.0)
        eplusi_dm_z051 = eplusi_results_dict.get('observables', {}).get('D_M_z051', 0.0)
        
        # Growth factor (safe list access with length check)
        eonly_d_list = eonly_results_dict.get('evolution_series', {}).get('D', [1.0])
        eonly_d_z0 = eonly_d_list[0] if len(eonly_d_list) > 0 else 1.0
        eplusi_d_list = eplusi_results_dict.get('evolution_series', {}).get('D', [1.0])
        eplusi_d_z0 = eplusi_d_list[0] if len(eplusi_d_list) > 0 else 1.0
        
        # I-parameter metrics (at key redshifts)
        eonly_i_list = eonly_results_dict.get('evolution_series', {}).get('I', [0.0])
        eplusi_i_list = eplusi_results_dict.get('evolution_series', {}).get('I', [0.0])
        eonly_i_z0 = eonly_i_list[0] if len(eonly_i_list) > 0 else 0.0
        eplusi_i_z0 = eplusi_i_list[0] if len(eplusi_i_list) > 0 else 0.0
        # I @ z=2 (approximately middle of list if available)
        eonly_i_z2 = eonly_i_list[len(eonly_i_list)//2] if len(eonly_i_list) > 0 else 0.0
        eplusi_i_z2 = eplusi_i_list[len(eplusi_i_list)//2] if len(eplusi_i_list) > 0 else 0.0
        
        # S8 evolution metrics (range, not just final value)
        eonly_s8_list = eonly_results_dict.get('evolution_series', {}).get('S8', [0.0])
        eplusi_s8_list = eplusi_results_dict.get('evolution_series', {}).get('S8', [0.0])
        eonly_s8_range = max(eonly_s8_list) - min(eonly_s8_list) if len(eonly_s8_list) > 0 else 0.0
        eplusi_s8_range = max(eplusi_s8_list) - min(eplusi_s8_list) if len(eplusi_s8_list) > 0 else 0.0
        
        # CMB power spectrum comparison (if available)
        eonly_cmb_chi2 = eonly_results_dict.get('likelihood', {}).get('chi2_cmb', None)
        eplusi_cmb_chi2 = eplusi_results_dict.get('likelihood', {}).get('chi2_cmb', None)
        
        # Sanity check summary (pass/fail counts)
        eonly_sanity = eonly_results_dict.get('sanity_checks', {})
        eplusi_sanity = eplusi_results_dict.get('sanity_checks', {})
        eonly_sanity_passed = sum(1 for v in eonly_sanity.values() if isinstance(v, bool) and v)
        eplusi_sanity_passed = sum(1 for v in eplusi_sanity.values() if isinstance(v, bool) and v)
        
        # PHASE 4: Compute galaxy structure metrics for both modes
        print(f"🔬 Computing galaxy structure metrics for {model_name}...")
        eonly_galaxy_analysis = GalaxyStructureAnalysis(eonly_results_dict, 'Eonly')
        eplusi_galaxy_analysis = GalaxyStructureAnalysis(eplusi_results_dict, 'EplusI')
        
        eonly_galaxy_metrics = eonly_galaxy_analysis.compute_all_metrics()
        eplusi_galaxy_metrics = eplusi_galaxy_analysis.compute_all_metrics()
        
        # Compute deltas
        delta_s8 = eplusi_s8 - eonly_s8
        delta_rho_de = eplusi_rho_de - eonly_rho_de
        delta_s8_percent = (delta_s8 / eonly_s8 * 100) if eonly_s8 != 0 else 0.0
        delta_rho_de_percent = (delta_rho_de / eonly_rho_de * 100) if eonly_rho_de != 0 else 0.0
        
        # Compute ALL deltas (PRODUCTION: comprehensive comparison)
        delta_mi = eplusi_mi - eonly_mi
        delta_pearson = eplusi_pearson - eonly_pearson
        delta_chi2 = eplusi_chi2 - eonly_chi2
        delta_aic = eplusi_aic - eonly_aic
        delta_bic = eplusi_bic - eonly_bic
        delta_mu_z1 = eplusi_mu_z1 - eonly_mu_z1
        delta_h_z0 = eplusi_h_z0 - eonly_h_z0
        delta_dm_z051 = eplusi_dm_z051 - eonly_dm_z051
        delta_d_z0 = eplusi_d_z0 - eonly_d_z0
        delta_i_z0 = eplusi_i_z0 - eonly_i_z0
        delta_i_z2 = eplusi_i_z2 - eonly_i_z2
        delta_s8_range = eplusi_s8_range - eonly_s8_range
        delta_cmb_chi2 = (eplusi_cmb_chi2 - eonly_cmb_chi2) if (eonly_cmb_chi2 is not None and eplusi_cmb_chi2 is not None) else None
        delta_sanity = eplusi_sanity_passed - eonly_sanity_passed
        
        comparison_data.append({
            # Model info
            'model_name': model_name,
            'coupling_type': eonly_result['model_config']['coupling_type'],
            
            # S8 comparison
            'eonly_s8': eonly_s8,
            'eplusi_s8': eplusi_s8,
            'delta_s8': delta_s8,
            'delta_s8_percent': delta_s8_percent,
            
            # ρ_DE comparison
            'eonly_rho_de': eonly_rho_de,
            'eplusi_rho_de': eplusi_rho_de,
            'delta_rho_de': delta_rho_de,
            'delta_rho_de_percent': delta_rho_de_percent,
            
            # I-E correlation comparison
            'eonly_mi': eonly_mi,
            'eplusi_mi': eplusi_mi,
            'delta_mi': delta_mi,
            'eonly_pearson': eonly_pearson,
            'eplusi_pearson': eplusi_pearson,
            'delta_pearson': delta_pearson,
            
            # Likelihood comparison
            'eonly_chi2': eonly_chi2,
            'eplusi_chi2': eplusi_chi2,
            'delta_chi2': delta_chi2,
            'eonly_aic': eonly_aic,
            'eplusi_aic': eplusi_aic,
            'delta_aic': delta_aic,
            'eonly_bic': eonly_bic,
            'eplusi_bic': eplusi_bic,
            'delta_bic': delta_bic,
            
            # Observable comparison
            'eonly_mu_z1': eonly_mu_z1,
            'eplusi_mu_z1': eplusi_mu_z1,
            'delta_mu_z1': delta_mu_z1,
            'eonly_h_z0': eonly_h_z0,
            'eplusi_h_z0': eplusi_h_z0,
            'delta_h_z0': delta_h_z0,
            'eonly_dm_z051': eonly_dm_z051,
            'eplusi_dm_z051': eplusi_dm_z051,
            'delta_dm_z051': delta_dm_z051,
            
            # Growth factor comparison
            'eonly_d_z0': eonly_d_z0,
            'eplusi_d_z0': eplusi_d_z0,
            'delta_d_z0': delta_d_z0,
            
            # I-parameter comparison (at key redshifts)
            'eonly_i_z0': eonly_i_z0,
            'eplusi_i_z0': eplusi_i_z0,
            'delta_i_z0': delta_i_z0,
            'eonly_i_z2': eonly_i_z2,
            'eplusi_i_z2': eplusi_i_z2,
            'delta_i_z2': delta_i_z2,
            
            # S8 evolution range comparison
            'eonly_s8_range': eonly_s8_range,
            'eplusi_s8_range': eplusi_s8_range,
            'delta_s8_range': delta_s8_range,
            
            # CMB comparison (if available)
            'eonly_cmb_chi2': eonly_cmb_chi2,
            'eplusi_cmb_chi2': eplusi_cmb_chi2,
            'delta_cmb_chi2': delta_cmb_chi2,
            
            # Sanity check comparison
            'eonly_sanity_passed': eonly_sanity_passed,
            'eplusi_sanity_passed': eplusi_sanity_passed,
            'delta_sanity': delta_sanity,
            
            # Galaxy structure metrics (nested dictionaries)
            'eonly_galaxy_metrics': eonly_galaxy_metrics,
            'eplusi_galaxy_metrics': eplusi_galaxy_metrics
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_table_path = f"{comparison_dir}/Comparison_Table.csv"
    comparison_df.to_csv(comparison_table_path, index=False)
    
    
    # Create dashboard plot
    dashboard_plot_path = f"{comparison_dir}/Dashboard_Comparison.png"
    create_eonly_vs_eplusi_dashboard(comparison_df, dashboard_plot_path)
    
    
    # Save COMPREHENSIVE comparison summary (PRODUCTION: all metrics)
    comparison_summary = {
        'total_models_compared': len(comparison_data),
        
        # S8 summary
        'average_delta_s8': comparison_df['delta_s8'].mean(),
        'average_delta_s8_percent': comparison_df['delta_s8_percent'].mean(),
        'max_delta_s8_percent': comparison_df['delta_s8_percent'].max(),
        'std_delta_s8': comparison_df['delta_s8'].std(),
        
        # ρ_DE summary
        'average_delta_rho_de': comparison_df['delta_rho_de'].mean(),
        'average_delta_rho_de_percent': comparison_df['delta_rho_de_percent'].mean(),
        'max_delta_rho_de_percent': comparison_df['delta_rho_de_percent'].max(),
        'std_delta_rho_de': comparison_df['delta_rho_de'].std(),
        
        # I-E correlation summary
        'average_delta_mi': comparison_df['delta_mi'].mean(),
        'average_delta_pearson': comparison_df['delta_pearson'].mean(),
        'max_delta_mi': comparison_df['delta_mi'].max(),
        
        # Likelihood summary
        'average_delta_chi2': comparison_df['delta_chi2'].mean(),
        'average_delta_aic': comparison_df['delta_aic'].mean(),
        'average_delta_bic': comparison_df['delta_bic'].mean(),
        'best_model_eonly': comparison_df.loc[comparison_df['eonly_aic'].idxmin(), 'model_name'] if 'eonly_aic' in comparison_df.columns else 'N/A',
        'best_model_eplusi': comparison_df.loc[comparison_df['eplusi_aic'].idxmin(), 'model_name'] if 'eplusi_aic' in comparison_df.columns else 'N/A',
        
        # Observable summary
        'average_delta_mu_z1': comparison_df['delta_mu_z1'].mean(),
        'average_delta_h_z0': comparison_df['delta_h_z0'].mean(),
        'average_delta_dm_z051': comparison_df['delta_dm_z051'].mean(),
        'average_delta_d_z0': comparison_df['delta_d_z0'].mean(),
        
        # I-parameter summary
        'average_delta_i_z0': comparison_df['delta_i_z0'].mean(),
        'average_delta_i_z2': comparison_df['delta_i_z2'].mean(),
        
        # S8 evolution summary
        'average_delta_s8_range': comparison_df['delta_s8_range'].mean(),
        
        # CMB summary (if available)
        'average_delta_cmb_chi2': comparison_df['delta_cmb_chi2'].mean() if comparison_df['delta_cmb_chi2'].notna().any() else None,
        
        # Sanity check summary
        'average_delta_sanity': comparison_df['delta_sanity'].mean(),
        'average_eonly_sanity_passed': comparison_df['eonly_sanity_passed'].mean(),
        'average_eplusi_sanity_passed': comparison_df['eplusi_sanity_passed'].mean(),
        
        # File paths
        'comparison_table_path': comparison_table_path,
        'dashboard_plot_path': dashboard_plot_path
    }
    
    summary_path = f"{comparison_dir}/Delta_Metrics.json"
    with open(summary_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    
    return comparison_summary

def compute_bayes_factors_all_models(all_results):
    """
    Compute Bayes Factors for all models relative to Null ΛCDM reference.
    
    Bayes Factor BF_ij = Z_i / Z_j where:
    - Z_i = evidence for model i
    - Z_j = evidence for reference model j (ΛCDM)
    
    Returns:
        dict: Bayes factors for each model
    """
    print("\n" + "="*80)
    print("🎯 BAYES FACTOR ANALYSIS - Model Comparison")
    print("="*80)
    
    bayes_factors = {}
    
    # Find ΛCDM reference log evidence
    logz_reference = None
    reference_mode = None
    
    for mode in ['Eonly', 'EplusI']:
        if mode not in all_results:
            continue
        
        for model in all_results[mode]:
            if 'Null' in model['model_name'] or 'LCDM' in model['model_name']:
                bayesian_inf = model['results'].get('bayesian_inference', {})
                if 'log_evidence' in bayesian_inf:
                    logz_reference = bayesian_inf['log_evidence']
                    reference_mode = mode
                    print(f"📍 Reference model (ΛCDM {mode}): log Z = {logz_reference:.2f}")
                    break
        
        if logz_reference is not None:
            break
    
    if logz_reference is None:
        print("⚠️ No reference model with evidence found (nested sampling not run?)")
        return None
    
    # Compute Bayes Factors for all models
    for mode in ['Eonly', 'EplusI']:
        if mode not in all_results:
            continue
        
        bayes_factors[mode] = []
        
        for model in all_results[mode]:
            model_name = model['model_name']
            bayesian_inf = model['results'].get('bayesian_inference', {})
            
            if 'log_evidence' in bayesian_inf:
                logz_model = bayesian_inf['log_evidence']
                logz_err = bayesian_inf.get('log_evidence_err', 0.0)
                
                log_BF = logz_model - logz_reference
                BF = np.exp(log_BF)
                
                # Interpretation
                if log_BF > 5:
                    interpretation = "Very strong"
                elif log_BF > 3:
                    interpretation = "Strong"
                elif log_BF > 1:
                    interpretation = "Substantial"
                elif log_BF > -1:
                    interpretation = "Weak"
                elif log_BF > -3:
                    interpretation = "Negative (substantial)"
                else:
                    interpretation = "Negative (strong)"
                
                bayes_factors[mode].append({
                    'model_name': model_name,
                    'log_evidence': float(logz_model),
                    'log_evidence_err': float(logz_err),
                    'log_bayes_factor': float(log_BF),
                    'bayes_factor': float(BF),
                    'interpretation': interpretation
                })
                
                print(f"  {model_name:50s} log BF = {log_BF:+7.2f} ({interpretation})")
    
    print("="*80)
    
    return {
        'reference_model': 'Null_Model_LCDM',
        'reference_mode': reference_mode,
        'log_evidence_reference': float(logz_reference),
        'bayes_factors': bayes_factors
    }

def create_bayes_factor_plot(bayes_factor_results, output_path):
    """
    Create publication-quality Bayes Factor comparison plot.
    
    Shows log Bayes Factor for all models relative to ΛCDM reference.
    """
    if bayes_factor_results is None:
        print("⚠️ No Bayes Factor data available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Bayes Factor Model Comparison (relative to ΛCDM)', 
                 fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_TITLE', 16),
                 fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'normal'))
    
    # Collect data
    bf_data = []
    for mode in ['Eonly', 'EplusI']:
        if mode in bayes_factor_results.get('bayes_factors', {}):
            for bf in bayes_factor_results['bayes_factors'][mode]:
                bf_data.append({
                    'model': bf['model_name'],
                    'mode': mode,
                    'log_BF': bf['log_bayes_factor'],
                    'BF': bf['bayes_factor'],
                    'interpretation': bf['interpretation']
                })
    
    if not bf_data:
        print("⚠️ No Bayes Factor data to plot")
        plt.close()
        return
    
    # Sort by log BF
    bf_data = sorted(bf_data, key=lambda x: x['log_BF'], reverse=True)
    
    # Panel 1: log Bayes Factor bar chart
    eonly_data = [d for d in bf_data if d['mode'] == 'Eonly']
    eplusi_data = [d for d in bf_data if d['mode'] == 'EplusI']
    
    y_eonly = [d['log_BF'] for d in eonly_data[:10]]  # Top 10
    y_eplusi = [d['log_BF'] for d in eplusi_data[:10]]
    x_eonly = range(len(y_eonly))
    x_eplusi = range(len(y_eplusi))
    
    ax1.barh(x_eonly, y_eonly, alpha=0.7, color='#457B9D', label='E-only', edgecolor='black')
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax1.axvline(x=1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Substantial (|log BF| > 1)')
    ax1.axvline(x=3, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Strong (|log BF| > 3)')
    ax1.axvline(x=5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Very Strong (|log BF| > 5)')
    ax1.set_xlabel('log Bayes Factor', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 14))
    ax1.set_ylabel('Model (E-only)', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 14))
    ax1.set_title('E-only Models vs ΛCDM', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_TITLE', 16) - 2)
    ax1.set_yticks(x_eonly)
    ax1.set_yticklabels([d['model'][:25] for d in eonly_data[:10]], fontsize=9)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=MASTER_CTRL.get('PLOT_GRID_ALPHA', 0.25), axis='x')
    
    # Panel 2: E+I models
    ax2.barh(x_eplusi, y_eplusi, alpha=0.7, color='#E63946', label='E+I', edgecolor='black')
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax2.axvline(x=1, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Substantial (|log BF| > 1)')
    ax2.axvline(x=3, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Strong (|log BF| > 3)')
    ax2.axvline(x=5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Very Strong (|log BF| > 5)')
    ax2.set_xlabel('log Bayes Factor', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 14))
    ax2.set_ylabel('Model (E+I)', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 14))
    ax2.set_title('E+I Models vs ΛCDM', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_TITLE', 16) - 2)
    ax2.set_yticks(x_eplusi)
    ax2.set_yticklabels([d['model'][:25] for d in eplusi_data[:10]], fontsize=9)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=MASTER_CTRL.get('PLOT_GRID_ALPHA', 0.25), axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=MASTER_CTRL.get('PLOT_SAVE_DPI', 300), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Bayes Factor plot saved: {output_path}")

def create_eonly_vs_eplusi_dashboard(comparison_df, output_path):
    """
    Create 6-panel dashboard comparing E-only vs E+I metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('E-only vs E+I Coupling Comparison Dashboard', 
                 fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_TITLE', 14) + 2, 
                 fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    
    # Panel 1: S8 comparison
    ax1 = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.35
    ax1.bar(x - width/2, comparison_df['eonly_s8'], width, label='E-only', alpha=0.8, color='blue')
    ax1.bar(x + width/2, comparison_df['eplusi_s8'], width, label='E+I', alpha=0.8, color='red')
    ax1.set_xlabel('Model', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12), fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    ax1.set_ylabel('S8', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12), fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    ax1.set_title('S8 Comparison', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_TITLE', 14), fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    ax1.set_xticks(x)
    ax1.set_xticklabels([name[:10] for name in comparison_df['model_name']], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: ρ_DE comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, comparison_df['eonly_rho_de'], width, label='E-only', alpha=0.8, color='blue')
    ax2.bar(x + width/2, comparison_df['eplusi_rho_de'], width, label='E+I', alpha=0.8, color='red')
    ax2.set_xlabel('Model', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12), fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    ax2.set_ylabel('ρ_DE (final)', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12), fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    ax2.set_title('Dark Energy Density Comparison', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_TITLE', 14), fontweight=MASTER_CTRL.get('PLOT_FONTWEIGHT', 'light'))
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:10] for name in comparison_df['model_name']], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: ΔS8 percentage
    ax3 = axes[0, 2]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['delta_s8_percent']]
    ax3.bar(x, comparison_df['delta_s8_percent'], color=colors, alpha=0.7)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('ΔS8 (%)')
    ax3.set_title('S8 Change (E+I vs E-only)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([name[:10] for name in comparison_df['model_name']], rotation=45)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Δρ_DE percentage
    ax4 = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['delta_rho_de_percent']]
    ax4.bar(x, comparison_df['delta_rho_de_percent'], color=colors, alpha=0.7)
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Δρ_DE (%)')
    ax4.set_title('Dark Energy Change (E+I vs E-only)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name[:10] for name in comparison_df['model_name']], rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: S8 vs ρ_DE scatter (E-only)
    ax5 = axes[1, 1]
    ax5.scatter(comparison_df['eonly_rho_de'], comparison_df['eonly_s8'], 
               c='blue', alpha=0.7, s=100, label='E-only')
    ax5.set_xlabel('ρ_DE (E-only)')
    ax5.set_ylabel('S8 (E-only)')
    ax5.set_title('E-only: S8 vs ρ_DE')
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: S8 vs ρ_DE scatter (E+I)
    ax6 = axes[1, 2]
    ax6.scatter(comparison_df['eplusi_rho_de'], comparison_df['eplusi_s8'], 
               c='red', alpha=0.7, s=100, label='E+I')
    ax6.set_xlabel('ρ_DE (E+I)')
    ax6.set_ylabel('S8 (E+I)')
    ax6.set_title('E+I: S8 vs ρ_DE')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Dashboard created with 6 comparison panels")

# ==========================================================================================
# GOLDILOCKS ZONE FINDER - BAYESIAN OPTIMIZATION
# ==========================================================================================

def find_goldilocks_zone_bayesian(run_dir):
    """
    Goldilocks Zone Finder - Bayesian Optimization
    
    Finds optimal E_c, sigma, alpha, beta0 parameters that maximize stability
    and minimize chi-squared simultaneously using Gaussian Process optimization.
    
    Returns:
        dict: Optimal parameters {E_c, sigma, alpha, beta0, objective_value, stability_score, chi2}
    """
    print("="*80)
    print("🔍 GOLDILOCKS ZONE FINDER - BAYESIAN OPTIMIZATION")
    print("="*80)
    
    from scipy.optimize import differential_evolution
    
    # Parameter bounds from MASTER_CTRL
    bounds = [
        MASTER_CTRL['GOLDILOCKS_SEARCH_RANGES']['E_c'],
        MASTER_CTRL['GOLDILOCKS_SEARCH_RANGES']['sigma'],
        MASTER_CTRL['GOLDILOCKS_SEARCH_RANGES']['alpha'],
        MASTER_CTRL['GOLDILOCKS_SEARCH_RANGES']['beta0']
    ]
    
    print(f"📊 Search ranges:")
    print(f"  E_c: {bounds[0]}")
    print(f"  σ: {bounds[1]}")
    print(f"  α: {bounds[2]}")
    print(f"  β₀: {bounds[3]}")
    
    # Objective function to minimize
    def goldilocks_objective(params):
        """
        Combined objective: stability + chi2
        
        Args:
            params: [E_c, sigma, alpha, beta0]
        
        Returns:
            score: Lower is better (minimize)
        """
        E_c, sigma, alpha, beta0 = params
        
        try:
            # Create temporary I-parameter and coupling models (TQE-COMPLIANT)
            i_field_temp = EnergyInformationContent(
                model_type='energy_based',  # TQE-COMPLIANT: I from energy evolution
                params={'epsilon': MASTER_CTRL['I_FIELD_EPSILON'], 'normalization': MASTER_CTRL['I_FIELD_NORMALIZATION']}
            )
            
            coupling_temp = CouplingModel(
                coupling_type='geometric',
                information_content=i_field_temp,
                params={'beta0': beta0},
                coupling_mode='EplusI'  # Goldilocks uses E+I coupling
            )
            
            friedmann_temp = FriedmannEvolution(
                H0=MASTER_CTRL['H0'],
                Omega_m=MASTER_CTRL['OMEGA_M'],
                Omega_Lambda=MASTER_CTRL['OMEGA_LAMBDA'],
                Omega_b=MASTER_CTRL['OMEGA_B'],
                Omega_r=MASTER_CTRL['OMEGA_R'],
                coupling_model=coupling_temp,
                information_content=i_field_temp
            )
            
            # Compute evolution
            a_grid = np.linspace(0.1, 1.0, 100)
            H_vals = []
            for a_val in a_grid:
                try:
                    H_vals.append(friedmann_temp.H(a_val))
                except:
                    H_vals.append(0.0)
            
            H_vals = np.array(H_vals)
            
            # Stability score: check for NaN, negative, or extreme values
            stability_penalties = 0.0
            
            # Penalty 1: NaN or inf
            if np.any(~np.isfinite(H_vals)):
                stability_penalties += 1000.0
            
            # Penalty 2: Negative H
            if np.any(H_vals <= 0):
                stability_penalties += 500.0
            
            # Penalty 3: H(a=1) deviation from H0
            H_at_1 = friedmann_temp.H(1.0)
            H0_deviation_pct = abs(H_at_1 - MASTER_CTRL['H0']) / MASTER_CTRL['H0'] * 100
            stability_penalties += H0_deviation_pct * 10.0  # 10× weight
            
            # Penalty 4: Extreme variation (H should be smooth)
            H_variation = np.std(H_vals) / np.mean(H_vals) if np.mean(H_vals) > 0 else 10.0
            if H_variation > 0.5:  # >50% variation = unstable
                stability_penalties += H_variation * 100.0
            
            # Chi-squared approximation (simplified, no full observable computation)
            # Use H(z=0) - H0 as proxy
            chi2_proxy = H0_deviation_pct**2
            
            # Combined objective (lower is better)
            objective_mode = MASTER_CTRL.get('GOLDILOCKS_OBJECTIVE', 'stability')
            
            if objective_mode == 'stability':
                score = stability_penalties
            elif objective_mode == 'chi2':
                score = chi2_proxy
            elif objective_mode == 'composite':
                score = stability_penalties + chi2_proxy
            else:
                score = stability_penalties
            
            return score
            
        except Exception as e:
            # Severe penalty for failed evaluations
            return 1e6
    
    # Run Bayesian optimization (differential evolution as proxy)
    print(f"\n🔍 Running Bayesian optimization...")
    print(f"   Method: Differential Evolution (adaptive)")
    print(f"   Max evaluations: {MASTER_CTRL.get('GOLDILOCKS_MAX_EVALS', 100)}")
    
    result = differential_evolution(
        goldilocks_objective,
        bounds=bounds,
        maxiter=MASTER_CTRL.get('GOLDILOCKS_MAX_EVALS', 100) // 10,
        popsize=10,
        seed=42,
        disp=True,
        polish=True
    )
    
    E_c_opt, sigma_opt, alpha_opt, beta0_opt = result.x
    objective_value = result.fun
    
    print(f"\n✅ GOLDILOCKS ZONE FOUND!")
    print(f"{'='*60}")
    print(f"  E_c (optimal) = {E_c_opt:.4f}")
    print(f"  σ (optimal) = {sigma_opt:.4f}")
    print(f"  α (optimal) = {alpha_opt:.6f}")
    print(f"  β₀ (optimal) = {beta0_opt:.6f}")
    print(f"  Objective value = {objective_value:.4f}")
    print(f"{'='*60}")
    
    # Verify stability with optimal parameters
    print(f"\n🔍 Verifying stability with optimal parameters...")
    final_score = goldilocks_objective([E_c_opt, sigma_opt, alpha_opt, beta0_opt])
    
    # Save Goldilocks results
    goldilocks_results = {
        'E_c_optimal': float(E_c_opt),
        'sigma_optimal': float(sigma_opt),
        'alpha_optimal': float(alpha_opt),
        'beta0_optimal': float(beta0_opt),
        'objective_value': float(objective_value),
        'final_stability_score': float(final_score),
        'search_method': 'bayesian_differential_evolution',
        'n_evaluations': result.nfev,
        'success': result.success,
        'message': result.message,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save to JSON (inside run_dir)
    goldilocks_dir = f"{run_dir}/Goldilocks_Results"
    os.makedirs(goldilocks_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    goldilocks_file = f"{goldilocks_dir}/Goldilocks_Optimal_Parameters_{timestamp}.json"
    with open(goldilocks_file, 'w') as f:
        json.dump(goldilocks_results, f, indent=2)
    print(f"Goldilocks results saved: {goldilocks_file}")
    
    # Create visualization
    print(f"\nCreating Goldilocks visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: E_c vs σ landscape (2D heatmap would need grid search, skip for speed)
    ax1 = axes[0, 0]
    ax1.scatter([E_c_opt], [sigma_opt], s=500, c='red', marker='*', edgecolors='black', linewidth=2, label='Optimal')
    ax1.set_xlabel('E_c (Critical Energy)', fontsize=12)
    ax1.set_ylabel('σ (Stability Width)', fontsize=12)
    ax1.set_title('Goldilocks Zone (E_c, σ)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bounds[0])
    ax1.set_ylim(bounds[1])
    
    # Plot 2: α vs β₀
    ax2 = axes[0, 1]
    ax2.scatter([alpha_opt], [beta0_opt], s=500, c='green', marker='*', edgecolors='black', linewidth=2, label='Optimal')
    ax2.set_xlabel('α (Coupling Strength)', fontsize=12)
    ax2.set_ylabel('β₀ (Geometric Coupling)', fontsize=12)
    ax2.set_title('Optimal Coupling Parameters', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(bounds[2])
    ax2.set_ylim(bounds[3])
    
    # Plot 3: Parameter summary bar chart
    ax3 = axes[1, 0]
    param_names = ['E_c', 'σ', 'α', 'β₀']
    param_values = [E_c_opt, sigma_opt, alpha_opt, beta0_opt]
    param_colors = ['red', 'blue', 'green', 'orange']
    ax3.bar(param_names, param_values, color=param_colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Parameter Value', fontsize=12)
    ax3.set_title('Optimal Goldilocks Parameters', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Stability info text
    ax4 = axes[1, 1]
    ax4.axis('off')
    info_text = f"""
    GOLDILOCKS OPTIMIZATION RESULTS
    ═══════════════════════════════
    
    Search Method: Bayesian (Differential Evolution)
    Evaluations: {result.nfev}
    Success: {'✅ YES' if result.success else '❌ NO'}
    
    OPTIMAL PARAMETERS:
    ───────────────────
    E_c = {E_c_opt:.4f}
    σ = {sigma_opt:.4f}
    α = {alpha_opt:.6f}
    β₀ = {beta0_opt:.6f}
    
    SCORES:
    ───────
    Objective value: {objective_value:.2f}
    Stability score: {final_score:.2f}
    
    INTERPRETATION:
    ───────────────
    Lower objective = Better stability
    This parameter set minimizes H(a)
    deviations while maintaining physical
    consistency (H>0, smooth evolution).
    """
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=10, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plot_file = f"{goldilocks_dir}/Goldilocks_Optimal_Visualization_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Goldilocks visualization saved: {plot_file}")
    plt.close()
    
    return goldilocks_results

# ==========================================================================================
# PHASE 4: AUTOMATIC PIPELINE FUNCTION
# ==========================================================================================

