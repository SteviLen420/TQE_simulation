# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# pipeline.py - Main Pipeline Orchestrator
# ==========================================================================================
# TQE Heisenberg Fluctuation Simulation Pipeline
# ==========================================================================================

import os
import time
import json
import gc
import numpy as np
from datetime import datetime, timezone
from tqdm.auto import tqdm

from .config import MASTER_CTRL
from .core.quantum_system import build_quantum_system
from .core.tqe_functions import sample_info_beta
from .simulation.trajectory import run_single
from .simulation.ensemble import sample_coherent_states
from .visualization import generate_all_plots, generate_parameter_sweep_plot
from .diagnostics import (
    run_preflight_diagnostics,
    report_preflight_results,
    run_postrun_diagnostics,
    report_postrun_results,
)

def run_pipeline(config_override=None):
    """
    Main pipeline orchestrator for TQE Heisenberg Fluctuation Analysis.
    
    Parameters
    ----------
    config_override : dict, optional
        Dictionary to override MASTER_CTRL values
    
    Returns
    -------
    dict
        Results dictionary with all simulation data and metadata
    """
    # Merge config overrides
    config = MASTER_CTRL.copy()
    if config_override:
        config.update(config_override)

    preflight_report = run_preflight_diagnostics(config)
    report_preflight_results(preflight_report)
    if preflight_report["status"] == "error":
        raise RuntimeError(
            "Preflight diagnostics reported blocking issues. Resolve them before rerunning."
        )
    
    # Setup directory structure on Desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    main_dir = os.path.join(desktop_path, "TQE_Heisenberg_Modular_Results")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"TQE_Heisenberg_Fluctuation_{timestamp}"
    run_dir = os.path.join(main_dir, run_folder_name)
    
    figdir = os.path.join(run_dir, "figs")
    datadir = os.path.join(run_dir, "data")
    
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)
    
    print(f"✅ Desktop main folder: {main_dir}")
    print(f"✅ Desktop run folder: {run_dir}")
    
    # Setup reproducibility
    from .utils.setup import setup_reproducibility
    seed, rng = setup_reproducibility(config.get("SEED"), config)
    
    # Build quantum system
    print("\n" + "="*80)
    print("QUANTUM SYSTEM INITIALIZATION")
    print("="*80)
    print(f"[QUANTUM] Building quantum system operators...")
    print(f"[QUANTUM] Hilbert space dimension: {config['N_HILB']} per mode")
    
    quantum_system = build_quantum_system(config)
    two_mode = quantum_system['two_mode']
    
    print("[QUANTUM] ✓ Quantum system operators ready")
    print("="*80)
    
    # Time evolution setup
    T_FINAL = config["T_FINAL"]
    N_T = config["N_T"]
    tlist = np.linspace(0.0, T_FINAL, N_T)
    
    # Initial state sampling
    N_ENSEMBLE = config["N_ENSEMBLE"]
    print("\n" + "="*80)
    print("INITIAL STATE SAMPLING")
    print("="*80)
    print(f"[ENSEMBLE] Sampling {N_ENSEMBLE} initial coherent states...")
    
    alphas1 = sample_coherent_states(N_ENSEMBLE, rng, config)
    alphas2 = sample_coherent_states(N_ENSEMBLE, rng, config) if two_mode else None
    I_samples = sample_info_beta(N_ENSEMBLE, rng, config)
    
    print(f"[ENSEMBLE] ✓ Initial states ready")
    print(f"[ENSEMBLE] I-parameter range: [{I_samples.min():.3f}, {I_samples.max():.3f}]")
    print("="*80)
    
    # Main pipeline
    print("\n" + "="*80)
    print("TQE HEISENBERG FLUCTUATION PIPELINE v4.2.0 PRO")
    print("="*80)
    print(f"Ensemble Size:    {N_ENSEMBLE} trajectories")
    print(f"Time Points:      {N_T} (dt = {T_FINAL/N_T:.4f})")
    print(f"Hilbert Dim:      {config['N_HILB']} per mode")
    print(f"Multiprocessing:  {'Enabled' if config.get('USE_MULTIPROCESSING', False) else 'Disabled'}")
    print(f"Master Seed:      {seed}")
    print("="*80 + "\n")
    
    progress = tqdm(total=7, desc="Pipeline Progress", unit="phase", leave=True)
    
    # ===== SCENARIO 1: NO-LAW (PRE-LAW STATE) =====
    progress.set_description("1/7: NO-LAW Simulation")
    print("\n" + "="*80)
    print("[PHASE 1] NO-LAW SIMULATION")
    print("="*80)
    print("  Scenario: Pre-law quantum fluctuations (no lock-in mechanism)")
    print(f"  Ensemble: {N_ENSEMBLE} trajectories")
    print(f"  Time Points: {N_T} (dt = {T_FINAL/N_T:.4f})")
    
    results_no_law = []
    I_kept_no_law = []
    
    for i in tqdm(range(N_ENSEMBLE), desc="  Simulating", leave=False):
        a2_sample = alphas2[i] if two_mode else None
        r = run_single(alphas1[i], a2_sample, I_samples[i], False, config, quantum_system, tlist)
        
        if r is not None:
            results_no_law.append(r)
            I_kept_no_law.append(I_samples[i])
    
    print(f"[PHASE 1] ✓ Complete: {len(results_no_law)}/{N_ENSEMBLE} valid trajectories")
    progress.update(1)
    gc.collect()
    
    # ===== SCENARIO 2: WITH-LAW (STABLE LAWS) - ALL 3 I-MODES =====
    progress.set_description("2/7: WITH-LAW Simulation (3 I-modes)")
    print("\n" + "="*80)
    print("[PHASE 2] WITH-LAW SIMULATION - ALL 3 I-ORIGIN MODES")
    print("="*80)
    print("  Scenario: TQE lock-in mechanism active (f(E,I) coupling)")
    print(f"  Ensemble: {N_ENSEMBLE} trajectories × 3 I-modes")
    print(f"  I-Modes: emergent, inherent, threshold")
    print(f"  Segments: {config['N_SEGMENTS']} (dynamic adaptation)")
    
    original_I_mode = config.get("I_ORIGIN_MODE", "emergent")
    results_with_law_emergent = []
    results_with_law_inherent = []
    results_with_law_threshold = []
    
    I_modes_to_test = ["emergent", "inherent", "threshold"]
    
    for mode_idx, i_mode in enumerate(I_modes_to_test):
        print(f"\n  [{mode_idx+1}/3] Running I-mode: {i_mode}")
        config["I_ORIGIN_MODE"] = i_mode
        
        results_mode = []
        for i in tqdm(range(N_ENSEMBLE), desc=f"    Simulating ({i_mode})", leave=False):
            a2_sample = alphas2[i] if two_mode else None
            r = run_single(alphas1[i], a2_sample, I_samples[i], True, config, quantum_system, tlist)
            
            if r is not None:
                results_mode.append(r)
        
        if i_mode == "emergent":
            results_with_law_emergent = results_mode
        elif i_mode == "inherent":
            results_with_law_inherent = results_mode
        elif i_mode == "threshold":
            results_with_law_threshold = results_mode
        
        print(f"    ✓ {i_mode}: {len(results_mode)}/{N_ENSEMBLE} valid")
    
    config["I_ORIGIN_MODE"] = original_I_mode
    results_with_law = results_with_law_emergent
    
    print(f"\n[PHASE 2] ✓ Complete: 3 I-modes tested ({N_ENSEMBLE} traj each)")
    progress.update(1)
    gc.collect()
    
    # Validation
    if not results_no_law or not results_with_law:
        raise RuntimeError("Simulation produced no valid results. Cannot continue.")
    
    # ===== DATA AGGREGATION =====
    progress.set_description("3/7: Data Aggregation")
    print("\n" + "="*80)
    print("[PHASE 3] DATA AGGREGATION")
    print("="*80)
    print("  Aggregating time-series data from all trajectories...")
    
    final_energies_no_law = np.array([r["final_energy"] for r in results_no_law])
    final_energies_with_law = np.array([r["final_energy"] for r in results_with_law])
    
    T_len_no_law = min(len(r["energies"]) for r in results_no_law)
    T_len_with_law = min(len(r["energies"]) for r in results_with_law)
    T_len = min(T_len_no_law, T_len_with_law)
    
    E_mat_no_law = np.vstack([r["energies"][:T_len] for r in results_no_law])
    S_mat_no_law = np.vstack([r["entropy"][:T_len] for r in results_no_law])
    C_mat_no_law = np.vstack([r["coherence"][:T_len] for r in results_no_law])
    U_mat_no_law = np.vstack([r["uncertainty_product"][:T_len] for r in results_no_law])
    DX_mat_no_law = np.vstack([r["delta_x"][:T_len] for r in results_no_law])
    
    E_mat_with_law = np.vstack([r["energies"][:T_len] for r in results_with_law])
    S_mat_with_law = np.vstack([r["entropy"][:T_len] for r in results_with_law])
    C_mat_with_law = np.vstack([r["coherence"][:T_len] for r in results_with_law])
    U_mat_with_law = np.vstack([r["uncertainty_product"][:T_len] for r in results_with_law])
    DX_mat_with_law = np.vstack([r["delta_x"][:T_len] for r in results_with_law])
    
    mean_E_no_law = np.mean(E_mat_no_law, axis=0)
    std_E_no_law = np.std(E_mat_no_law, axis=0)
    mean_E_with_law = np.mean(E_mat_with_law, axis=0)
    std_E_with_law = np.std(E_mat_with_law, axis=0)
    mean_S_no_law = np.mean(S_mat_no_law, axis=0)
    mean_S_with_law = np.mean(S_mat_with_law, axis=0)
    mean_C_no_law = np.mean(C_mat_no_law, axis=0)
    mean_C_with_law = np.mean(C_mat_with_law, axis=0)
    mean_U_no_law = np.mean(U_mat_no_law, axis=0)
    std_U_no_law = np.std(U_mat_no_law, axis=0)
    mean_U_with_law = np.mean(U_mat_with_law, axis=0)
    std_U_with_law = np.std(U_mat_with_law, axis=0)
    mean_DX_no_law = np.mean(DX_mat_no_law, axis=0)
    mean_DX_with_law = np.mean(DX_mat_with_law, axis=0)
    
    tlist_agg = tlist[:T_len]
    
    print(f"[PHASE 3] ✓ Complete: {len(results_no_law) + len(results_with_law)} trajectories aggregated")
    progress.update(1)
    
    # ===== STATISTICAL ANALYSIS =====
    progress.set_description("4/7: Statistical Analysis")
    print("\n" + "="*80)
    print("[PHASE 4] STATISTICAL ANALYSIS")
    print("="*80)
    print("  Computing comparative statistics (NO-LAW vs WITH-LAW)...")
    
    stats_comparison = {
        "run_metadata": {
            "run_name": run_folder_name,
            "timestamp_utc": timestamp,
            "master_seed": seed,
            "reproducibility_note": "Use 'master_seed' value in MASTER_CTRL['SEED'] to reproduce this exact run",
        },
        "NO_LAW": {
            "scenario_description": "Pre-law state (pure quantum fluctuations, no lock-in)",
            "n_valid_trajectories": len(results_no_law),
            "final_energy_mean": float(np.mean(final_energies_no_law)),
            "final_energy_std": float(np.std(final_energies_no_law)),
            "final_energy_max": float(np.max(final_energies_no_law)),
            "final_energy_min": float(np.min(final_energies_no_law)),
            "variance_mean": float(np.mean(std_E_no_law**2)),
            "variance_max": float(np.max(std_E_no_law**2)),
            "entropy_final_mean": float(mean_S_no_law[-1]),
            "coherence_final_mean": float(mean_C_no_law[-1]),
            "heisenberg_uncertainty_final_mean": float(mean_U_no_law[-1]),
            "heisenberg_uncertainty_final_std": float(std_U_no_law[-1]),
            "delta_x_final_mean": float(mean_DX_no_law[-1]),
        },
        "WITH_LAW": {
            "scenario_description": "Stable laws active (TQE lock-in mechanism enabled)",
            "n_valid_trajectories": len(results_with_law),
            "final_energy_mean": float(np.mean(final_energies_with_law)),
            "final_energy_std": float(np.std(final_energies_with_law)),
            "final_energy_max": float(np.max(final_energies_with_law)),
            "final_energy_min": float(np.min(final_energies_with_law)),
            "variance_mean": float(np.mean(std_E_with_law**2)),
            "variance_max": float(np.max(std_E_with_law**2)),
            "entropy_final_mean": float(mean_S_with_law[-1]),
            "coherence_final_mean": float(mean_C_with_law[-1]),
            "heisenberg_uncertainty_final_mean": float(mean_U_with_law[-1]),
            "heisenberg_uncertainty_final_std": float(std_U_with_law[-1]),
            "delta_x_final_mean": float(mean_DX_with_law[-1]),
        },
        "SUPPRESSION_RATIOS": {
            "description": "Ratios < 1.0 indicate suppression by stable laws",
            "variance_ratio": float(np.mean(std_E_with_law**2) / (np.mean(std_E_no_law**2) + 1e-15)),
            "std_ratio": float(np.std(final_energies_with_law) / (np.std(final_energies_no_law) + 1e-15)),
            "max_energy_ratio": float(np.max(final_energies_with_law) / (np.max(final_energies_no_law) + 1e-15)),
            "uncertainty_ratio": float(mean_U_with_law[-1] / (mean_U_no_law[-1] + 1e-15)),
            "coherence_ratio": float(mean_C_with_law[-1] / (mean_C_no_law[-1] + 1e-15)),
        },
        "HEISENBERG_COMPLIANCE": {
            "description": "Check if Heisenberg uncertainty principle is satisfied",
            "min_uncertainty_no_law": float(np.min(mean_U_no_law)),
            "min_uncertainty_with_law": float(np.min(mean_U_with_law)),
            "theoretical_minimum": float(config["HBAR"] / 2.0),
            "no_law_compliant": bool(np.min(mean_U_no_law) >= config["HBAR"] / 2.0 * 0.99),
            "with_law_compliant": bool(np.min(mean_U_with_law) >= config["HBAR"] / 2.0 * 0.99),
        },
        "INFORMATION_ORIGIN": {
            "description": "All 3 I-origin models tested: emergent, inherent, threshold",
            "I_initial_mean": float(np.mean(I_samples)),
            "I_initial_std": float(np.std(I_samples)),
            "emergent": {
                "n_trajectories": len(results_with_law_emergent),
                "model": "I_{t+1} = γ·I_t + α·|ΔE_t| + β·corr(ΔE_t, ΔE_{t-1})",
            },
            "inherent": {
                "n_trajectories": len(results_with_law_inherent),
                "model": "I = scale · log(E/E0) or (E/E0)^γ",
            },
            "threshold": {
                "n_trajectories": len(results_with_law_threshold),
                "model": "I = 0 if E < E_c, else I += slope·(E-E_c)",
            },
        }
    }
    
    print(f"[PHASE 4] ✓ Complete: Statistics computed")
    progress.update(1)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARATIVE STATISTICS SUMMARY")
    print("="*80)
    print(f"\nNO-LAW (Pre-law state):")
    print(f"  Final Energy: {stats_comparison['NO_LAW']['final_energy_mean']:.3f} ± {stats_comparison['NO_LAW']['final_energy_std']:.3f}")
    print(f"  Max Energy: {stats_comparison['NO_LAW']['final_energy_max']:.3f}")
    print(f"  Mean Variance: {stats_comparison['NO_LAW']['variance_mean']:.3f}")
    
    print(f"\nWITH-LAW (Stable laws active):")
    print(f"  Final Energy: {stats_comparison['WITH_LAW']['final_energy_mean']:.3f} ± {stats_comparison['WITH_LAW']['final_energy_std']:.3f}")
    print(f"  Max Energy: {stats_comparison['WITH_LAW']['final_energy_max']:.3f}")
    print(f"  Mean Variance: {stats_comparison['WITH_LAW']['variance_mean']:.3f}")
    
    print(f"\nSUPPRESSION RATIOS (WITH-LAW / NO-LAW):")
    print(f"  Variance Ratio: {stats_comparison['SUPPRESSION_RATIOS']['variance_ratio']:.4f}")
    print(f"  Std Dev Ratio: {stats_comparison['SUPPRESSION_RATIOS']['std_ratio']:.4f}")
    print(f"  Max Energy Ratio: {stats_comparison['SUPPRESSION_RATIOS']['max_energy_ratio']:.4f}")
    print(f"  Uncertainty Product Ratio: {stats_comparison['SUPPRESSION_RATIOS']['uncertainty_ratio']:.4f}")
    print(f"  Coherence Ratio: {stats_comparison['SUPPRESSION_RATIOS']['coherence_ratio']:.4f}")
    
    print(f"\nHEISENBERG UNCERTAINTY COMPLIANCE:")
    print(f"  Theoretical Minimum (ℏ/2): {stats_comparison['HEISENBERG_COMPLIANCE']['theoretical_minimum']:.3f}")
    print(f"  NO-LAW Min: {stats_comparison['HEISENBERG_COMPLIANCE']['min_uncertainty_no_law']:.3f} " + 
          f"({'✓' if stats_comparison['HEISENBERG_COMPLIANCE']['no_law_compliant'] else '✗'})")
    print(f"  WITH-LAW Min: {stats_comparison['HEISENBERG_COMPLIANCE']['min_uncertainty_with_law']:.3f} " +
          f"({'✓' if stats_comparison['HEISENBERG_COMPLIANCE']['with_law_compliant'] else '✗'})")
    print("="*80 + "\n")
    
    # ===== SAVE DATA =====
    progress.set_description("5/7: Saving Data Files")
    print("\n" + "="*80)
    print("[PHASE 5] SAVING DATA FILES")
    print("="*80)
    print("  Writing CSV and JSON files to disk...")
    
    # Save comparative analysis JSON
    comparative_filepath = os.path.join(run_dir, "comparative_analysis.json")
    with open(comparative_filepath, 'w') as f:
        json.dump(stats_comparison, f, indent=4)
    print(f"[SAVED] comparative_analysis.json")
    
    # Save summary JSON
    summary_data = {
        "run_info": {
            "run_name": run_folder_name,
            "timestamp_utc": timestamp,
            "seed": seed,
        },
        "parameters": config,
        "results": stats_comparison,
    }
    
    summary_filepath = os.path.join(run_dir, "summary.json")
    with open(summary_filepath, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"[SAVED] summary.json")
    
    # Save time-series data
    if tlist_agg.size > 0:
        csv_no_law = os.path.join(datadir, "no_law_timeseries.csv")
        data_no_law = np.vstack([tlist_agg, mean_E_no_law, std_E_no_law, mean_S_no_law, mean_C_no_law, 
                                  mean_U_no_law, std_U_no_law, mean_DX_no_law]).T
        header_no_law = (f"# TQE Heisenberg Fluctuation - NO-LAW Scenario\n"
                         f"# Run: {run_folder_name}\n"
                         f"# Timestamp: {timestamp}\n"
                         f"# Master Seed: {seed}\n"
                         f"# N_Ensemble: {len(results_no_law)}\n"
                         f"# Reproducibility: Set MASTER_CTRL['SEED']={seed} to reproduce this run\n"
                         f"time,mean_energy,std_energy,mean_entropy,mean_coherence,mean_uncertainty,std_uncertainty,mean_delta_x")
        np.savetxt(csv_no_law, data_no_law, delimiter=",", header=header_no_law, comments="")
        print(f"[SAVED] no_law_timeseries.csv")
        
        csv_with_law = os.path.join(datadir, "with_law_timeseries.csv")
        data_with_law = np.vstack([tlist_agg, mean_E_with_law, std_E_with_law, mean_S_with_law, mean_C_with_law,
                                    mean_U_with_law, std_U_with_law, mean_DX_with_law]).T
        header_with_law = (f"# TQE Heisenberg Fluctuation - WITH-LAW Scenario\n"
                           f"# Run: {run_folder_name}\n"
                           f"# Timestamp: {timestamp}\n"
                           f"# Master Seed: {seed}\n"
                           f"# N_Ensemble: {len(results_with_law)}\n"
                           f"# Reproducibility: Set MASTER_CTRL['SEED']={seed} to reproduce this run\n"
                           f"time,mean_energy,std_energy,mean_entropy,mean_coherence,mean_uncertainty,std_uncertainty,mean_delta_x")
        np.savetxt(csv_with_law, data_with_law, delimiter=",", header=header_with_law, comments="")
        print(f"[SAVED] with_law_timeseries.csv")
    
    # Save final energies
    csv_final = os.path.join(datadir, "ensemble_final_energies.csv")
    data_final = np.vstack([final_energies_no_law, final_energies_with_law]).T
    header_final = (f"# TQE Heisenberg Fluctuation - Final Energies\n"
                    f"# Run: {run_folder_name}\n"
                    f"# Timestamp: {timestamp}\n"
                    f"# Master Seed: {seed}\n"
                    f"# N_Ensemble (NO-LAW): {len(results_no_law)}\n"
                    f"# N_Ensemble (WITH-LAW): {len(results_with_law)}\n"
                    f"# Reproducibility: Set MASTER_CTRL['SEED']={seed} to reproduce this run\n"
                    f"no_law_final_energy,with_law_final_energy")
    np.savetxt(csv_final, data_final, delimiter=",", header=header_final, comments="")
    print(f"[SAVED] ensemble_final_energies.csv")
    
    print(f"\n[PHASE 5] Summary:")
    print(f"  ✓ comparative_analysis.json")
    print(f"  ✓ summary.json")
    print(f"  ✓ no_law_timeseries.csv")
    print(f"  ✓ with_law_timeseries.csv")
    print(f"  ✓ ensemble_final_energies.csv")
    print(f"[PHASE 5] ✓ Complete: 5 data files saved (2 JSON + 3 CSV)")
    progress.update(1)
    
    # ===== VISUALIZATION =====
    progress.set_description("6/7: Generating Visualizations")
    print("\n" + "="*80)
    print("[PHASE 6] GENERATING VISUALIZATIONS")
    print("="*80)
    print("  Creating publication-quality plots...")
    
    n_plots = generate_all_plots(
        tlist_agg, mean_E_no_law, std_E_no_law, mean_E_with_law, std_E_with_law,
        mean_S_no_law, mean_S_with_law, mean_C_no_law, mean_C_with_law,
        mean_U_no_law, std_U_no_law, mean_U_with_law, std_U_with_law,
        mean_DX_no_law, mean_DX_with_law,
        final_energies_no_law, final_energies_with_law,
        results_no_law, results_with_law,
        results_with_law_emergent, results_with_law_inherent, results_with_law_threshold,
        stats_comparison, config, figdir, T_FINAL
    )
    
    print(f"\n[PHASE 6] ✓ Complete: {n_plots} visualization plots generated")
    progress.update(1)
    
    # ===== PARAMETER SWEEP (Optional) =====
    progress.set_description("7/7: Parameter Sweep (optional)")
    
    if config.get("ENABLE_PARAMETER_SWEEP", False):
        print("\n" + "="*80)
        print("[PHASE 7] PARAMETER SWEEP ANALYSIS")
        print("="*80)
        
        sweep_var = config["SWEEP_VARIABLE"]
        sweep_values = config["SWEEP_VALUES"]
        sweep_n_ensemble = config["SWEEP_N_ENSEMBLE"]
        
        print(f"  Sweep Variable: {sweep_var}")
        print(f"  Sweep Range:    {len(sweep_values)} points")
        print(f"  Ensemble/point: {sweep_n_ensemble} trajectories")
        
        sweep_results = []
        orig_val = config[sweep_var]
        
        for val in tqdm(sweep_values, desc=f"  Sweeping {sweep_var}", leave=False):
            config[sweep_var] = val
            
            alphas1_sweep = sample_coherent_states(sweep_n_ensemble, rng, config)
            alphas2_sweep = sample_coherent_states(sweep_n_ensemble, rng, config) if two_mode else None
            I_sweep = sample_info_beta(sweep_n_ensemble, rng, config)
            
            results_sweep = []
            for i in range(sweep_n_ensemble):
                a2_s = alphas2_sweep[i] if two_mode else None
                r = run_single(alphas1_sweep[i], a2_s, I_sweep[i], True, config, quantum_system, tlist)
                if r is not None:
                    results_sweep.append(r)
            
            if results_sweep:
                final_E_sweep = np.array([r["final_energy"] for r in results_sweep])
                variance_sweep = np.var(final_E_sweep)
                mean_E_sweep = np.mean(final_E_sweep)
                
                sweep_results.append({
                    sweep_var: val,
                    "mean_energy": float(mean_E_sweep),
                    "variance": float(variance_sweep),
                    "n_trajectories": len(results_sweep)
                })
            
            config[sweep_var] = orig_val
        
        if sweep_results:
            import pandas as pd
            df_sweep = pd.DataFrame(sweep_results)
            sweep_csv = os.path.join(datadir, f"parameter_sweep_{sweep_var}.csv")
            df_sweep.to_csv(sweep_csv, index=False)
            print(f"[SAVED] parameter_sweep_{sweep_var}.csv")
            
            generate_parameter_sweep_plot(sweep_var, sweep_values, sweep_results, config, figdir)
            print(f"[SAVED] 10_parameter_sweep_{sweep_var}.png")
            print(f"\n[PHASE 7] ✓ Complete: Parameter sweep finished (1 CSV + 1 PNG)")
    else:
        print("\n[PHASE 7] Skipped (ENABLE_PARAMETER_SWEEP = False)")
    
    progress.update(1)
    progress.close()

    print("\n" + "="*80)
    print("PIPELINE COMPLETED: TQE Heisenberg Fluctuation Analysis")
    print("="*80)
    print(f"Ensemble:   {len(results_no_law)}/{N_ENSEMBLE} NO-LAW | {len(results_with_law)}/{N_ENSEMBLE} WITH-LAW")
    print(f"I-Modes:    emergent ({len(results_with_law_emergent)}) | inherent ({len(results_with_law_inherent)}) | threshold ({len(results_with_law_threshold)})")
    print(f"\nSuppression (emergent model):")
    print(f"  Variance: {stats_comparison['SUPPRESSION_RATIOS']['variance_ratio']:.4f} | Δx·Δp: {stats_comparison['SUPPRESSION_RATIOS']['uncertainty_ratio']:.4f} | Coherence: {stats_comparison['SUPPRESSION_RATIOS']['coherence_ratio']:.4f}")
    print(f"  Heisenberg: {'✓ PASSED' if stats_comparison['HEISENBERG_COMPLIANCE']['with_law_compliant'] else '✗ FAILED'}")
    n_plots_final = n_plots + (1 if config.get("ENABLE_PARAMETER_SWEEP", False) else 0)
    print(f"\nSaved: 2 JSON + 3 CSV + {n_plots_final} PNG (including 4 I-origin plots)")
    print(f"Seed:  {seed} | Dir: {run_dir}")
    print("="*80)

    postrun_report = run_postrun_diagnostics(run_dir)
    report_postrun_results(postrun_report)

    return {
        'run_dir': run_dir,
        'config': config,
        'stats_comparison': stats_comparison,
        'results_no_law': results_no_law,
        'results_with_law': results_with_law,
        'results_with_law_emergent': results_with_law_emergent,
        'results_with_law_inherent': results_with_law_inherent,
        'results_with_law_threshold': results_with_law_threshold,
        'diagnostics': {
            'preflight': preflight_report,
            'postrun': postrun_report,
        }
    }

