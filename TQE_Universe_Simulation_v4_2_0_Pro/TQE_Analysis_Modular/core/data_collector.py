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
    load_universe_seeds
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
        "nested_sampling": 0
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
    
    print(f"   • Emergent laws: {ext_counts['emergent_laws']}/{len(collected_data['eonly'])+len(collected_data['ei'])}")
    print(f"   • Parameter sensitivity: {ext_counts['param_sens']}/{len(collected_data['eonly'])+len(collected_data['ei'])}")
    print(f"   • CMB cold spots: {ext_counts['cmb_coldspots']}/{len(collected_data['ei'])}")
    print(f"   • CMB Axis of Evil: {ext_counts['cmb_aoe']}/{len(collected_data['ei'])}")
    print(f"   • I-definitions comp: {ext_counts['i_defs_comp']}/{len(collected_data['eonly'])+len(collected_data['ei'])}")
    print(f"   • Life compatibility: {ext_counts['life_compat']}")
    print(f"   • Planck validation: {ext_counts['planck']}")
    print(f"   • Entropy volatility: {ext_counts['entropy']}")
    print(f"   • Stability sweeps (eps/zero): {ext_counts['stability_eps']} / {ext_counts['stability_zero']}")
    print(f"   • Advanced anomalies: {ext_counts['advanced_anomalies']}")
    print(f"   • Nested sampling traces: {ext_counts['nested_sampling']}")
    
    return collected_data

