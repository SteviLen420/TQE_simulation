# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# TQE Analysis Pipeline Modular - Main Entry Point
# Local execution only - saves to Desktop

import os
import sys
import time
import json
import pickle
from typing import Dict
import pandas as pd
from tqdm import tqdm

# Setup package installation first
from .utils.setup import check_and_install_packages
check_and_install_packages()

# Now safe to import other modules
from .config import MASTER_CTRL
from .core.path_setup import setup_paths
from .core.data_collector import collect_simulation_data
from .analysis.metrics_extractor import build_metrics_dataframe
from .analysis.comparative import compare_ei_definitions, compare_eonly_vs_ei
from .analysis.model_selector import select_best_model
from .analysis.specialized import (
    analyze_emergent_laws, analyze_friedmann_cosmology, analyze_cmb_anomalies,
    analyze_lockin_dynamics, analyze_quantum_fields, analyze_entanglement,
    analyze_parameter_sensitivity, analyze_topology, analyze_i_definitions_direct,
    analyze_planck_fit, analyze_life_top_universes, analyze_entropy_volatility,
    analyze_physical_anomalies, analyze_statistical_finetuning
)
from .visualization.detailed_metrics import generate_detailed_metrics
from .visualization.advanced_plots import generate_advanced_visualizations, generate_complexity_analysis
from .visualization.reports import generate_extended_reports


def export_summary_and_metadata(df_metrics: pd.DataFrame, collected_data: Dict, dirs: Dict[str, str], output_root: str, simulation_root: str) -> str:
    """
    PHASE 7: Generate execution summary and metadata artifacts.
    """
    print("\n" + "="*70)
    print("PHASE 7: SUMMARY EXPORT")
    print("="*70)
    
    summary_text = []
    summary_text.append("╔" + "═"*68 + "╗")
    summary_text.append("║  TQE ANALYSIS PIPELINE v4.2.0 PRO - EXECUTION SUMMARY".ljust(69) + "║")
    summary_text.append("╚" + "═"*68 + "╝")
    summary_text.append("")
    summary_text.append(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_text.append(f"Target mode: {collected_data['metadata']['target_mode']}")
    summary_text.append(f"Simulation root: {simulation_root}")
    summary_text.append("")
    summary_text.append("=" * 70)
    summary_text.append("RUNS ANALYZED")
    summary_text.append("=" * 70)
    summary_text.append(f"Total runs: {len(df_metrics)}")
    summary_text.append(f"  • E-only: {collected_data['metadata']['n_eonly_runs']}")
    summary_text.append(f"  • E+I: {collected_data['metadata']['n_ei_runs']}")
    summary_text.append("")
    summary_text.append("=" * 70)
    summary_text.append("OUTPUT STRUCTURE")
    summary_text.append("=" * 70)
    summary_text.append(f"{output_root}/")
    summary_text.append("├── 00_summary/ (overview + metadata + validation)")
    summary_text.append("├── 01_comparative_analysis/ (12+ categories)")
    summary_text.append("├── 02_detailed_metrics/ (extended CSV + plots)")
    summary_text.append("├── 03_visualizations/ (radar + heatmap + complexity)")
    summary_text.append("├── 04_best_model_selection/ (triple rankings + report)")
    summary_text.append("└── 05_raw_data/ (collected_data.pkl + extended_metrics.pkl)")
    summary_text.append("")
    summary_text.append("=" * 70)
    summary_text.append("TRIPLE RANKING SNAPSHOT")
    summary_text.append("=" * 70)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    if len(df_ei) > 0:
        best_stable = df_ei.nlargest(1, "stable_percent").iloc[0]
        best_complex = df_ei.nlargest(1, "complexity_score").iloc[0]
        if "physical_laws_total_score" in df_ei.columns:
            best_physical = df_ei.nlargest(1, "physical_laws_total_score").iloc[0]
            summary_text.append(f"  • Physical-Laws Winner: {best_physical['i_definition']}")
        summary_text.append(f"  • Stability Winner: {best_stable['i_definition']} ({best_stable['stable_percent']:.2f}%)")
        summary_text.append(f"  • Complexity Winner: {best_complex['i_definition']} ({best_complex['complexity_score']:.2f})")
    summary_text.append("")
    summary_text.append("=" * 70)
    summary_text.append("NEXT ACTIONS")
    summary_text.append("=" * 70)
    summary_text.append(f"1. Review recommendation: {os.path.join(dirs['best_model'], 'recommendation_report.md')}")
    summary_text.append(f"2. Inspect detailed metrics: {os.path.join(dirs['detailed_metrics'], 'all_runs_metrics.csv')}")
    summary_text.append(f"3. Validate Planck fit module: {os.path.join(dirs['planck_fit'], 'planck_fit_metrics.csv')}")
    summary_text.append("")
    
    summary_str = "\n".join(summary_text)
    with open(os.path.join(dirs["summary"], "analysis_summary.txt"), 'w') as f:
        f.write(summary_str)
    with open(os.path.join(dirs["summary"], "run_info.json"), 'w') as f:
        json.dump(collected_data["metadata"], f, indent=2)
    
    print("   ✅ analysis_summary.txt and run_info.json written")
    return summary_str


def run_validation_checks(dirs: Dict[str, str]) -> bool:
    """
    PHASE 8: Lightweight validation to ensure critical artifacts exist.
    """
    print("\n" + "="*70)
    print("PHASE 8: VALIDATION & QC")
    print("="*70)
    
    checks = [
        ("All Runs Metrics CSV", os.path.join(dirs["detailed_metrics"], "all_runs_metrics.csv")),
        ("Weighted Ranking CSV", os.path.join(dirs["best_model"], "weighted_ranking.csv")),
        ("Recommendation Report", os.path.join(dirs["best_model"], "recommendation_report.md")),
        ("Extended Report", os.path.join(dirs["summary"], "extended_report.md")),
    ]
    
    lines = []
    overall_pass = True
    for label, path in checks:
        exists = os.path.exists(path)
        overall_pass = overall_pass and exists
        status = "PASS" if exists else "MISSING"
        lines.append(f"{status:7} - {label} ({path})")
        print(f"   {status:7} {label}")
    
    lines.append("")
    lines.append(f"Overall status: {'PASS' if overall_pass else 'CHECK LOGS'}")
    
    report_path = os.path.join(dirs["summary"], "validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"   ✅ Validation report written to {report_path}")
    return overall_pass


def run_analysis_pipeline(config_override: dict = None):
    """
    Main analysis pipeline orchestrator.
    
    Executes complete comparative analysis workflow:
    1. Validates target mode (batch_ei or batch_all)
    2. Collects simulation data from all runs
    3. Extracts and normalizes metrics
    4. Performs comparative analyses
    5. Generates visualizations
    6. Ranks models and produces recommendation
    
    Args:
        config_override: Optional dictionary to override MASTER_CTRL settings
    
    Returns:
        None (all results saved to disk)
    """
    # Merge config
    config = MASTER_CTRL.copy()
    if config_override:
        config.update(config_override)
    
    print("\n" + "="*70)
    print("TQE ANALYSIS PIPELINE v4.2.0 PRO (MODULAR)")
    print("="*70)
    print(f"Analysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target mode: {config['TARGET_MODE']}")
    print("="*70)
    
    # Setup paths (Desktop output)
    simulation_root, analysis_output_root = setup_paths(config)
    print(f"Simulation root: {simulation_root}")
    print(f"Analysis output root: {analysis_output_root}")
    print("="*70)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_name = config['TARGET_MODE'].replace("TQE_Universe_Simulation_", "").replace("_Pipeline_v4.2.0_PRO", "")
    output_root = os.path.join(analysis_output_root, f"{mode_name}_{timestamp}_analysis")
    os.makedirs(output_root, exist_ok=True)
    
    print(f"\nOutput directory: {output_root}\n")
    
    # Create progress tracker (8 phases total)
    progress = tqdm(total=8, desc="Analysis Progress", ncols=100)
    
    # PHASE 1: Collect data
    progress.set_description("Phase 1/8: Comprehensive Data Collection")
    collected_data = collect_simulation_data(
        config['TARGET_MODE'],
        simulation_root,
        config.get('TARGET_TIMESTAMP')
    )
    progress.update(1)
    
    # Save collected data
    os.makedirs(os.path.join(output_root, "05_raw_data"), exist_ok=True)
    with open(os.path.join(output_root, "05_raw_data", "collected_data.pkl"), 'wb') as f:
        pickle.dump(collected_data, f)
    
    # PHASE 2: Build metrics DataFrame
    progress.set_description("Phase 2/8: Extended Metric Extraction")
    print("\n" + "="*70)
    print("PHASE 2: COMPREHENSIVE METRIC EXTRACTION & SCORING (EXTENDED)")
    print("="*70)
    df_metrics = build_metrics_dataframe(collected_data)
    print(f"✅ Comprehensive metrics: complexity, life-compatibility, information richness")
    print(f"✅ Extended metrics: emergent laws, Friedmann, CMB, lock-in, quantum, entanglement, etc.")
    
    with open(os.path.join(output_root, "05_raw_data", "extended_metrics.pkl"), 'wb') as f:
        pickle.dump(df_metrics, f)
    print(f"✅ Saved extended_metrics.pkl to 05_raw_data/")
    
    progress.update(1)
    
    # Create output subdirectories
    dirs = {
        "summary": os.path.join(output_root, "00_summary"),
        "ei_comparison": os.path.join(output_root, "01_comparative_analysis", "basic_metrics"),
        "emergent_laws": os.path.join(output_root, "01_comparative_analysis", "emergent_laws"),
        "friedmann": os.path.join(output_root, "01_comparative_analysis", "friedmann_cosmology"),
        "cmb_anomalies": os.path.join(output_root, "01_comparative_analysis", "cmb_anomalies"),
        "lockin_dynamics": os.path.join(output_root, "01_comparative_analysis", "lockin_dynamics"),
        "quantum_fields": os.path.join(output_root, "01_comparative_analysis", "quantum_fields"),
        "entanglement": os.path.join(output_root, "01_comparative_analysis", "entanglement"),
        "param_sensitivity": os.path.join(output_root, "01_comparative_analysis", "parameter_sensitivity"),
        "finetuning": os.path.join(output_root, "01_comparative_analysis", "finetuning"),
        "topology": os.path.join(output_root, "01_comparative_analysis", "topology"),
        "i_definitions_direct": os.path.join(output_root, "01_comparative_analysis", "i_definitions_direct"),
        "planck_fit": os.path.join(output_root, "01_comparative_analysis", "planck_fit"),
        "life_top": os.path.join(output_root, "01_comparative_analysis", "life_top_universes"),
        "entropy_volatility": os.path.join(output_root, "01_comparative_analysis", "entropy_volatility"),
        "physical_anomalies": os.path.join(output_root, "01_comparative_analysis", "physical_anomalies"),
        "eonly_vs_ei": os.path.join(output_root, "01_comparative_analysis", "eonly_vs_ei"),
        "detailed_metrics": os.path.join(output_root, "02_detailed_metrics"),
        "visualizations": os.path.join(output_root, "03_visualizations"),
        "complexity_analysis": os.path.join(output_root, "03_visualizations", "complexity"),
        "best_model": os.path.join(output_root, "04_best_model_selection"),
        "raw_data": os.path.join(output_root, "05_raw_data"),
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # PHASE 3: Comprehensive Comparative Analysis
    progress.set_description("Phase 3/8: Comparative Analysis")
    
    # 3A: Basic metrics
    compare_ei_definitions(df_metrics, dirs["ei_comparison"], config)
    
    # 3B-K: Extended category analyses
    analyze_emergent_laws(df_metrics, dirs["emergent_laws"], config)
    analyze_friedmann_cosmology(df_metrics, dirs["friedmann"], config)
    analyze_cmb_anomalies(df_metrics, dirs["cmb_anomalies"], config)
    analyze_lockin_dynamics(df_metrics, dirs["lockin_dynamics"], config)
    analyze_quantum_fields(df_metrics, dirs["quantum_fields"], config)
    analyze_entanglement(df_metrics, dirs["entanglement"], config)
    analyze_parameter_sensitivity(df_metrics, dirs["param_sensitivity"], config)
    analyze_statistical_finetuning(df_metrics, collected_data, dirs["finetuning"], config)
    analyze_topology(df_metrics, dirs["topology"], config)
    analyze_i_definitions_direct(df_metrics, collected_data, dirs["i_definitions_direct"], config)
    analyze_planck_fit(df_metrics, dirs["planck_fit"], config)
    analyze_life_top_universes(df_metrics, dirs["life_top"], config)
    analyze_entropy_volatility(df_metrics, dirs["entropy_volatility"], config)
    analyze_physical_anomalies(df_metrics, dirs["physical_anomalies"], config)
    
    # 3L: E-only vs E+I baseline
    if collected_data["metadata"]["has_eonly"]:
        compare_eonly_vs_ei(df_metrics, dirs["eonly_vs_ei"], config)
    
    generate_detailed_metrics(df_metrics, dirs["detailed_metrics"], config)
    progress.update(1)
    
    # PHASE 4: Advanced Visualizations
    progress.set_description("Phase 4/8: Advanced Visualizations")
    generate_advanced_visualizations(df_metrics, dirs["visualizations"], config)
    generate_complexity_analysis(df_metrics, dirs["complexity_analysis"], config)
    progress.update(1)
    
    # PHASE 5: Triple Ranking System
    progress.set_description("Phase 5/8: Triple Model Ranking")
    select_best_model(df_metrics, dirs["best_model"], config)
    progress.update(1)
    
    # PHASE 6: Extended Analysis Reports
    progress.set_description("Phase 6/8: Extended Reports")
    generate_extended_reports(df_metrics, collected_data, dirs["summary"], config)
    progress.update(1)
    
    # PHASE 7: Comprehensive Summary Export
    progress.set_description("Phase 7/8: Summary Export")
    summary_text = export_summary_and_metadata(df_metrics, collected_data, dirs, output_root, simulation_root)
    progress.update(1)
    
    # PHASE 8: Validation & QC
    progress.set_description("Phase 8/8: Validation")
    validation_passed = run_validation_checks(dirs)
    progress.update(1)
    
    progress.close()
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("\n" + summary_text)
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    if validation_passed:
        print("║" + "  ✅ ANALYSIS PIPELINE COMPLETE!".center(68) + "║")
    else:
        print("║" + "  ⚠️ ANALYSIS COMPLETE WITH WARNINGS".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print(f"\n📁 Results directory: {output_root}")
    print(f"📊 Recommendation: {os.path.join(dirs['best_model'], 'recommendation_report.md')}")
    if not validation_passed:
        print("⚠️  See validation_report.txt for missing artifacts.")
    print("")


if __name__ == "__main__":
    # Print startup banner
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TQE ANALYSIS PIPELINE v4.2.0 PRO (MODULAR)".center(68) + "║")
    print("║" + "  Comparative Analysis & Model Selection".center(68) + "║")
    print("║" + "  Local Execution Only - Desktop Output".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    print("\n[LOCAL] Local environment detected.")
    print("[OUTPUT] Results will be saved to Desktop/TQE_Analysis_Modular_Results/")
    
    # Run main analysis pipeline
    try:
        run_analysis_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

