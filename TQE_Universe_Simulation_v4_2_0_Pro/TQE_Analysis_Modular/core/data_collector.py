# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Data collection orchestrator

import os
import time
from typing import Dict
from .path_setup import (
    find_latest_mode_directory,
    validate_target_mode,
    detect_eonly_presence,
    collect_run_directories
)
from ..data_loaders import (
    load_summary_json,
    load_tqe_runs_csv,
    load_bayesian_calibration_csv,
    load_emergent_law_summary,
    load_parameter_sensitivity,
    load_cmb_coldspots,
    load_cmb_aoe,
    load_i_definitions_comparison,
    load_life_compatibility_summary,
    load_planck_validation,
    load_entropy_volatility_summary,
    load_stability_sweep,
    load_advanced_anomaly_results,
    load_nested_sampling_samples,
    load_pre_fluctuation_pairs,
    load_universe_seeds,
    load_comprehensive_universe_physics,
    load_advanced_law_detection,
    load_complexity_metrics,
    load_complexity_ranking,
    load_ei_importance_comparison,
    load_feature_importance,
    load_statistical_finetuning,
    load_parameter_correlation,
    load_cmb_power_spectrum,
    load_lockin_statistics,
    load_avg_lockin_curve,
    load_stability_distribution,
    load_all_fl_timeseries
)
from ..utils.helpers import extract_i_definition

def collect_simulation_data(target_mode: str, simulation_root: str, target_timestamp: str = None) -> Dict:
    """
    PHASE 1: Comprehensive data collection function (EXTENDED).
    
    Automatically discovers all simulation runs and loads:
    - summary_full.json (stability, Goldilocks, Bayesian)
    - tqe_runs.csv (ALL columns: E, I, X, cosmology, quantum fields, entropy)
    - bayesian_calibration_*.csv (Goldilocks optimization)
    - emergent_law_summary.csv (power-laws, phase transitions)
    - parameter_sensitivity_analysis.csv (E/I/X sensitivity)
    - cmb_coldspots_summary_*.csv (CMB anomalies)
    - cmb_aoe_summary_*.csv (Axis of Evil)
    - advanced_anomaly_detection_results*.csv (physical anomalies, gaussianity, isotropy)
    - planck_best_fit_summary.json + planck_validation*.csv (Planck proximity)
    - life_compatibility_summary.json (life-score components)
    - stability_by_I_eps_sweep*.csv, stability_by_I_zero*.csv (sensitivity sweeps)
    - entropy_volatility_summary*.csv (information volatility)
    - nested_sampling_samples*.csv (evidence traces)
    - I_Definitions_Comparison.csv (I(E) curves)
    - enhanced_physics CSVs (Friedmann, quantum, entanglement)
    
    Returns:
        dict: Comprehensive dictionary with extended simulation data
              {
                  "metadata": {...},
                  "eonly": {dirname: {summary, tqe_runs, bayesian, extended_data, ...}},
                  "ei": {i_def: {dirname, summary, tqe_runs, bayesian, extended_data, ...}}
              }
    """
    print("\n" + "="*70)
    print("PHASE 1: DATA COLLECTION & VALIDATION")
    print("="*70)
    
    # Validate target mode
    if not validate_target_mode(target_mode):
        raise ValueError(f"Invalid target mode: {target_mode}")
    
    # Find the target directory (latest or specific timestamp)
    target_path = find_latest_mode_directory(simulation_root, target_mode, target_timestamp)
    
    if target_path is None:
        raise FileNotFoundError(f"Could not find target directory for mode '{target_mode}'")
    
    print(f"Target mode: {target_mode}")
    print(f"Target path: {target_path}\n")
    
    # Detect E-only presence
    has_eonly = detect_eonly_presence(target_path)
    print(f"E-only presence: {'✅ YES (batch_all mode)' if has_eonly else '❌ NO (batch_ei mode)'}\n")
    
    # Collect run directories
    run_dirs = collect_run_directories(target_path)
    print(f"Found {len(run_dirs['eonly'])} E-only runs")
    print(f"Found {len(run_dirs['ei'])} E+I runs")
    if len(run_dirs['ei']) > 0:
        print(f"   E+I directories: {', '.join(run_dirs['ei'][:3])}{'...' if len(run_dirs['ei']) > 3 else ''}")
    print()
    
    # Load data from each run
    collected_data = {
        "metadata": {
            "target_mode": target_mode,
            "target_path": target_path,
            "has_eonly": has_eonly,
            "n_eonly_runs": len(run_dirs['eonly']),
            "n_ei_runs": len(run_dirs['ei']),
            "collection_timestamp": time.strftime("%Y%m%d_%H%M%S")
        },
        "eonly": {},
        "ei": {}
    }
    
    # Load E-only data
    if has_eonly:
        print("Loading E-only data...")
        for dirname in run_dirs['eonly']:
            run_path = os.path.join(target_path, dirname)
            summary = load_summary_json(run_path)
            tqe_runs = load_tqe_runs_csv(run_path)
            bayesian = load_bayesian_calibration_csv(run_path)
            
            # Load extended analysis data
            emergent_laws = load_emergent_law_summary(run_path)
            param_sens = load_parameter_sensitivity(run_path)
            i_defs_comp = load_i_definitions_comparison(run_path)
            life_compat = load_life_compatibility_summary(run_path)
            planck_fit = load_planck_validation(run_path)
            entropy_volatility = load_entropy_volatility_summary(run_path)
            stability_eps = load_stability_sweep(run_path, "eps_sweep")
            stability_zero = load_stability_sweep(run_path, "zero")
            advanced_anomalies = load_advanced_anomaly_results(run_path)
            nested_sampling = load_nested_sampling_samples(run_path)
            pre_fluctuation = load_pre_fluctuation_pairs(run_path)
            universe_seeds = load_universe_seeds(run_path)
            comprehensive_physics = load_comprehensive_universe_physics(run_path)
            advanced_laws = load_advanced_law_detection(run_path)
            complexity_metrics = load_complexity_metrics(run_path)
            complexity_ranking = load_complexity_ranking(run_path)
            ei_importance = load_ei_importance_comparison(run_path)
            feature_importance = load_feature_importance(run_path)
            statistical_finetuning = load_statistical_finetuning(run_path)
            parameter_correlation = load_parameter_correlation(run_path)
            cmb_power_spectrum = load_cmb_power_spectrum(run_path)
            lockin_statistics = load_lockin_statistics(run_path)
            avg_lockin_curve = load_avg_lockin_curve(run_path)
            stability_distribution = load_stability_distribution(run_path)
            fl_timeseries = load_all_fl_timeseries(run_path)
            
            if summary:
                collected_data["eonly"][dirname] = {
                    "summary": summary,
                    "tqe_runs": tqe_runs,
                    "bayesian": bayesian,
                    "emergent_laws": emergent_laws,
                    "parameter_sensitivity": param_sens,
                    "i_definitions_comparison": i_defs_comp,
                    "life_compatibility": life_compat,
                    "planck_validation": planck_fit,
                    "entropy_volatility": entropy_volatility,
                    "stability_sweep_eps": stability_eps,
                    "stability_sweep_zero": stability_zero,
                    "advanced_anomalies": advanced_anomalies,
                    "nested_sampling": nested_sampling,
                    "pre_fluctuation_pairs": pre_fluctuation,
                    "universe_seeds": universe_seeds,
                    "comprehensive_physics": comprehensive_physics,
                    "advanced_laws": advanced_laws,
                    "complexity_metrics": complexity_metrics,
                    "complexity_ranking": complexity_ranking,
                    "ei_importance": ei_importance,
                    "feature_importance": feature_importance,
                    "statistical_finetuning": statistical_finetuning,
                    "parameter_correlation": parameter_correlation,
                    "cmb_power_spectrum": cmb_power_spectrum,
                    "lockin_statistics": lockin_statistics,
                    "avg_lockin_curve": avg_lockin_curve,
                    "stability_distribution": stability_distribution,
                    "fl_timeseries": fl_timeseries,
                    "run_path": run_path
                }
                print(f"  ✅ {dirname}")
            else:
                print(f"  ⚠️  SKIPPED: {dirname} (no summary_full.json found)")
    
    # Load E+I data
    print("\nLoading E+I data...")
    for dirname in run_dirs['ei']:
        run_path = os.path.join(target_path, dirname)
        i_def = extract_i_definition(dirname)
        summary = load_summary_json(run_path)
        tqe_runs = load_tqe_runs_csv(run_path)
        bayesian = load_bayesian_calibration_csv(run_path)
        
        # Load extended analysis data
        emergent_laws = load_emergent_law_summary(run_path)
        param_sens = load_parameter_sensitivity(run_path)
        cmb_coldspots = load_cmb_coldspots(run_path, i_def)
        cmb_aoe = load_cmb_aoe(run_path, i_def)
        i_defs_comp = load_i_definitions_comparison(run_path)
        life_compat = load_life_compatibility_summary(run_path)
        planck_fit = load_planck_validation(run_path)
        entropy_volatility = load_entropy_volatility_summary(run_path)
        stability_eps = load_stability_sweep(run_path, "eps_sweep")
        stability_zero = load_stability_sweep(run_path, "zero")
        advanced_anomalies = load_advanced_anomaly_results(run_path)
        nested_sampling = load_nested_sampling_samples(run_path)
        pre_fluctuation = load_pre_fluctuation_pairs(run_path)
        universe_seeds = load_universe_seeds(run_path)
        comprehensive_physics = load_comprehensive_universe_physics(run_path)
        advanced_laws = load_advanced_law_detection(run_path)
        complexity_metrics = load_complexity_metrics(run_path)
        complexity_ranking = load_complexity_ranking(run_path)
        ei_importance = load_ei_importance_comparison(run_path)
        feature_importance = load_feature_importance(run_path)
        statistical_finetuning = load_statistical_finetuning(run_path)
        parameter_correlation = load_parameter_correlation(run_path)
        cmb_power_spectrum = load_cmb_power_spectrum(run_path)
        lockin_statistics = load_lockin_statistics(run_path)
        avg_lockin_curve = load_avg_lockin_curve(run_path)
        stability_distribution = load_stability_distribution(run_path)
        fl_timeseries = load_all_fl_timeseries(run_path)
        
        if summary:
            collected_data["ei"][i_def] = {
                "dirname": dirname,
                "summary": summary,
                "tqe_runs": tqe_runs,
                "bayesian": bayesian,
                "emergent_laws": emergent_laws,
                "parameter_sensitivity": param_sens,
                "cmb_coldspots": cmb_coldspots,
                "cmb_aoe": cmb_aoe,
                "i_definitions_comparison": i_defs_comp,
                "life_compatibility": life_compat,
                "planck_validation": planck_fit,
                "entropy_volatility": entropy_volatility,
                "stability_sweep_eps": stability_eps,
                "stability_sweep_zero": stability_zero,
                "advanced_anomalies": advanced_anomalies,
                "nested_sampling": nested_sampling,
                "pre_fluctuation_pairs": pre_fluctuation,
                "universe_seeds": universe_seeds,
                "comprehensive_physics": comprehensive_physics,
                "advanced_laws": advanced_laws,
                "complexity_metrics": complexity_metrics,
                "complexity_ranking": complexity_ranking,
                "ei_importance": ei_importance,
                "feature_importance": feature_importance,
                "statistical_finetuning": statistical_finetuning,
                "parameter_correlation": parameter_correlation,
                "cmb_power_spectrum": cmb_power_spectrum,
                "lockin_statistics": lockin_statistics,
                "avg_lockin_curve": avg_lockin_curve,
                "stability_distribution": stability_distribution,
                "fl_timeseries": fl_timeseries,
                "run_path": run_path
            }
            print(f"  ✅ {i_def}")
        else:
            print(f"  ⚠️  SKIPPED: {i_def} (no summary_full.json found in {dirname})")
    
    print(f"\n✅ Data collection complete!")
    print(f"   E-only: {len(collected_data['eonly'])} runs loaded")
    print(f"   E+I: {len(collected_data['ei'])} runs loaded")
    
    # Extended data statistics
    print(f"\n📊 Extended data loaded:")
    ext_counts = {
        "emergent_laws": 0,
        "param_sens": 0,
        "cmb_coldspots": 0,
        "cmb_aoe": 0,
        "i_defs_comp": 0,
        "life_compat": 0,
        "planck": 0,
        "entropy": 0,
        "stability_eps": 0,
        "stability_zero": 0,
        "advanced_anomalies": 0,
        "nested_sampling": 0,
        "comprehensive_physics": 0,
        "advanced_laws": 0,
        "complexity_metrics": 0,
        "complexity_ranking": 0,
        "ei_importance": 0,
        "feature_importance": 0,
        "statistical_finetuning": 0,
        "parameter_correlation": 0,
        "cmb_power_spectrum": 0,
        "lockin_statistics": 0,
        "avg_lockin_curve": 0,
        "stability_distribution": 0,
        "fl_timeseries": 0
    }
    for data in list(collected_data['eonly'].values()) + list(collected_data['ei'].values()):
        if data.get("emergent_laws") is not None: ext_counts["emergent_laws"] += 1
        if data.get("parameter_sensitivity") is not None: ext_counts["param_sens"] += 1
        if data.get("cmb_coldspots") is not None: ext_counts["cmb_coldspots"] += 1
        if data.get("cmb_aoe") is not None: ext_counts["cmb_aoe"] += 1
        if data.get("i_definitions_comparison") is not None: ext_counts["i_defs_comp"] += 1
        if data.get("life_compatibility") is not None: ext_counts["life_compat"] += 1
        if data.get("planck_validation") is not None: ext_counts["planck"] += 1
        if data.get("entropy_volatility") is not None: ext_counts["entropy"] += 1
        if data.get("stability_sweep_eps") is not None: ext_counts["stability_eps"] += 1
        if data.get("stability_sweep_zero") is not None: ext_counts["stability_zero"] += 1
        if data.get("advanced_anomalies") is not None: ext_counts["advanced_anomalies"] += 1
        if data.get("nested_sampling") is not None: ext_counts["nested_sampling"] += 1
        if data.get("comprehensive_physics") is not None: ext_counts["comprehensive_physics"] += 1
        if data.get("advanced_laws") is not None: ext_counts["advanced_laws"] += 1
        if data.get("complexity_metrics") is not None: ext_counts["complexity_metrics"] += 1
        if data.get("complexity_ranking") is not None: ext_counts["complexity_ranking"] += 1
        if data.get("ei_importance") is not None: ext_counts["ei_importance"] += 1
        if data.get("feature_importance") is not None: ext_counts["feature_importance"] += 1
        if data.get("statistical_finetuning") is not None: ext_counts["statistical_finetuning"] += 1
        if data.get("parameter_correlation") is not None: ext_counts["parameter_correlation"] += 1
        if data.get("cmb_power_spectrum") is not None: ext_counts["cmb_power_spectrum"] += 1
        if data.get("lockin_statistics") is not None: ext_counts["lockin_statistics"] += 1
        if data.get("avg_lockin_curve") is not None: ext_counts["avg_lockin_curve"] += 1
        if data.get("stability_distribution") is not None: ext_counts["stability_distribution"] += 1
        if data.get("fl_timeseries") is not None: ext_counts["fl_timeseries"] += 1
    
    total_runs = len(collected_data['eonly']) + len(collected_data['ei'])
    print(f"   • Emergent laws: {ext_counts['emergent_laws']}/{total_runs}")
    print(f"   • Parameter sensitivity: {ext_counts['param_sens']}/{total_runs}")
    print(f"   • CMB cold spots: {ext_counts['cmb_coldspots']}/{len(collected_data['ei'])}")
    print(f"   • CMB Axis of Evil: {ext_counts['cmb_aoe']}/{len(collected_data['ei'])}")
    print(f"   • I-definitions comp: {ext_counts['i_defs_comp']}/{total_runs}")
    print(f"   • Life compatibility: {ext_counts['life_compat']}/{total_runs}")
    print(f"   • Planck validation: {ext_counts['planck']}/{total_runs}")
    print(f"   • Entropy volatility: {ext_counts['entropy']}/{total_runs}")
    print(f"   • Stability sweeps (eps/zero): {ext_counts['stability_eps']} / {ext_counts['stability_zero']}")
    print(f"   • Advanced anomalies: {ext_counts['advanced_anomalies']}/{total_runs}")
    print(f"   • Nested sampling traces: {ext_counts['nested_sampling']}/{total_runs}")
    print(f"   • Comprehensive physics: {ext_counts['comprehensive_physics']}/{total_runs}")
    print(f"   • Advanced law detection: {ext_counts['advanced_laws']}/{total_runs}")
    print(f"   • Complexity metrics: {ext_counts['complexity_metrics']}/{total_runs}")
    print(f"   • Complexity ranking: {ext_counts['complexity_ranking']}/{total_runs}")
    print(f"   • E/I importance: {ext_counts['ei_importance']}/{total_runs}")
    print(f"   • Feature importance: {ext_counts['feature_importance']}/{total_runs}")
    print(f"   • Statistical finetuning: {ext_counts['statistical_finetuning']}/{total_runs}")
    print(f"   • Parameter correlation: {ext_counts['parameter_correlation']}/{total_runs}")
    print(f"   • CMB power spectrum: {ext_counts['cmb_power_spectrum']}/{total_runs}")
    print(f"   • Lockin statistics: {ext_counts['lockin_statistics']}/{total_runs}")
    print(f"   • Avg lockin curve: {ext_counts['avg_lockin_curve']}/{total_runs}")
    print(f"   • Stability distribution: {ext_counts['stability_distribution']}/{total_runs}")
    print(f"   • FL timeseries: {ext_counts['fl_timeseries']}/{total_runs}")
    
    return collected_data

