# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Main pipeline orchestrator
#
import os
import sys
import time
import json
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .config.master_ctrl import MASTER_CTRL
from .core.pipeline_context import PipelineContext
from .core.physics_engine import PhysicsEngine
from .simulation.monte_carlo import phase_01_monte_carlo
from .phases.phase_01_10 import (
    phase_02_stability_curve, phase_03_scatter_ei, phase_04_fluctuation_panels,
    phase_05_stability_by_i, phase_06_lockin_histogram, phase_07_stability_distribution,
    phase_08_avg_lockin_curve, phase_09_feature_importance, phase_10_emergent_laws
)
from .phases.phase_11_20 import (
    phase_11_finetuning_detector, phase_12_best_universe_plots, phase_13_generate_missing_cmb_maps,
    phase_14_entropy_volatility, phase_15_planck_validation, phase_16_cmb_anomaly_detection,
    phase_17_ei_importance_comparison, phase_18_multi_mode_goldilocks_comparison,
    phase_19_cmb_analysis_plots, phase_20_comprehensive_correlation_analysis
)
from .phases.phase_21_28 import (
    phase_21_advanced_statistical_analysis, phase_22_cmb_anomaly_analysis_plots,
    phase_23_enhanced_physics_analysis, phase_24_comprehensive_data_extraction,
    phase_25_advanced_anomaly_detection, phase_26_advanced_law_detection,
    phase_27_comprehensive_visualization_extraction, phase_28_final_summary,
    integrate_complexity_analysis
)
from .analysis.bayesian import (
    compute_bayesian_model_selection, run_nested_sampling,
    save_bayesian_metrics_csv, plot_bayesian_comparison
)
from .utils.memory import optimize_for_colab, cleanup_memory

IN_COLAB = ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_BACKEND_VERSION" in os.environ)

def switch_pipeline_type(pipeline_type: str = "E+I"):
    """
    Switch between E+I and E-Only pipeline modes.
    
    Args:
        pipeline_type (str): "E+I" or "E-Only"
    """
    if pipeline_type.upper() == "E+I" or pipeline_type.lower() == "full":
        MASTER_CTRL["PIPELINE_VARIANT"] = "full"
        print("🔄 Switched to E+I (Energy + Information) pipeline mode")
    elif pipeline_type.upper() == "E-ONLY" or pipeline_type.lower() == "energy_only":
        MASTER_CTRL["PIPELINE_VARIANT"] = "energy_only"
        print("🔄 Switched to E-Only (Energy only) pipeline mode")
    else:
        print(f"Invalid pipeline type: {pipeline_type}. Use 'E+I' or 'E-Only'")
        return
    
    if MASTER_CTRL.get("VERBOSE", False):
        print(f" Pipeline variant set to: {MASTER_CTRL['PIPELINE_VARIANT']}")
        print("All generated files will be tagged accordingly")


def run_multi_i_parameter_analysis(i_definitions: list = None, pipeline_variants: list = None) -> dict:
    """
    Run pipeline for multiple I parameter definitions and pipeline variants.
    
    Args:
        i_definitions: List of I parameter definitions to test
        pipeline_variants: List of pipeline variants to test
    
    Returns:
        dict: Comprehensive analysis results
    """
    if i_definitions is None:
        i_definitions = ["kl_shannon", "shannon", "fisher"]
    
    if pipeline_variants is None:
        pipeline_variants = ["full", "energy_only"]
    
    print("=" * 80)
    print(f"STARTING MULTI-I PARAMETER COMPREHENSIVE ANALYSIS ")
    print("=" * 80)
    print(f" I Parameter Definitions: {i_definitions}")
    print(f"🔄 Pipeline Variants: {pipeline_variants}")
    print("=" * 80)
    
    # Create master results directory on Google Drive
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    if IN_COLAB:
        base_dir = "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO"
    else:
        base_dir = os.path.join(os.getcwd(), "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO")
    
    master_save_dir = os.path.join(base_dir, "runs", f"COMPARATIVE_ANALYSIS_{timestamp}")
    os.makedirs(master_save_dir, exist_ok=True)
    
    all_results = {}
    comparative_data = []
    
    # Run each combination
    total_combinations = len(i_definitions) * len(pipeline_variants)
    current_combination = 0
    
    for i_def in i_definitions:
        for variant in pipeline_variants:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"🔄 RUNNING COMBINATION {current_combination}/{total_combinations}")
            print(f" I Definition: {i_def}")
            print(f"🔄 Pipeline Variant: {variant}")
            print(f"{'='*60}")
            
            # Create subdirectory for this I-definition (simple name, not full pipeline name)
            i_param_dir = os.path.join(master_save_dir, i_def)
            os.makedirs(i_param_dir, exist_ok=True)
            
            # Set configuration
            config = MASTER_CTRL.copy()
            config["I_DEFINITION_MODE"] = i_def
            config["PIPELINE_VARIANT"] = variant
            config["MULTI_I_ANALYSIS_MODE"] = True
            config["MULTI_I_SAVE_DIR"] = master_save_dir  # Parent directory, PipelineContext will use run_id as subdirectory
            
            # Create simple run ID (just the I-definition name, will be used as subdirectory)
            run_id = i_def
            
            try:
                # Run pipeline
                result = run_pipeline(config_override=config, run_id_override=run_id)
                
                # Store results
                key = f"{i_def}_{variant}"
                all_results[key] = {
                    "i_definition": i_def,
                    "pipeline_variant": variant,
                    "result": result,
                    "run_id": run_id
                }
                
                # Add to comparative data
                comparative_data.append({
                    "i_definition": i_def,
                    "pipeline_variant": variant,
                    "stability_rate": result.get("stability_rate", 0),
                    "lockin_rate": result.get("lockin_rate", 0),
                    "peak_x": result.get("peak_x", 0),
                    "total_universes": result.get("total_universes", 0),
                    "run_id": run_id
                })
                
                print(f" Completed: {i_def} + {variant}")
                
            except Exception as e:
                print(f"Error in {i_def} + {variant}: {e}")
                all_results[f"{i_def}_{variant}"] = {
                    "i_definition": i_def,
                    "pipeline_variant": variant,
                    "error": str(e),
                    "run_id": run_id
                }
    
    # Create comprehensive analysis
    print(f"\n{'='*80}")
    print(" CREATING COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    
    # Save comparative data
    comparative_df = pd.DataFrame(comparative_data)
    comparative_csv_path = os.path.join(master_save_dir, "multi_i_parameter_comparison.csv")
    comparative_df.to_csv(comparative_csv_path, index=False)
    
    # Create summary analysis
    summary_analysis = create_i_parameter_summary_analysis(comparative_df, master_save_dir)
    
    # Save all results
    results_json_path = os.path.join(master_save_dir, "multi_i_parameter_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"📁 Master results saved to: {master_save_dir}")
    
    return {
        "master_save_dir": master_save_dir,
        "all_results": all_results,
        "comparative_data": comparative_df,
        "summary_analysis": summary_analysis
    }


def create_i_parameter_summary_analysis(comparative_df: pd.DataFrame, save_dir: str) -> str:
    """Create comprehensive summary analysis of I parameter comparisons."""
    try:
        # Create summary plots
        # PUBLICATION: Larger 2x2 with constrained_layout (was: 16,12)
        fig, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
        fig.suptitle('Multi-I Parameter Analysis Summary', fontsize=20, fontweight='bold')
        
        # 1. Stability rate by I definition and variant
        ax1 = axes[0,0]
        pivot_stability = comparative_df.pivot(index='i_definition', columns='pipeline_variant', values='stability_rate')
        pivot_stability.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
        ax1.set_title('Stability Rate by I Definition and Pipeline Variant')
        ax1.set_ylabel('Stability Rate')
        ax1.legend(title='Pipeline Variant')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Lock-in rate by I definition and variant
        ax2 = axes[0,1]
        pivot_lockin = comparative_df.pivot(index='i_definition', columns='pipeline_variant', values='lockin_rate')
        pivot_lockin.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
        ax2.set_title('Lock-in Rate by I Definition and Pipeline Variant')
        ax2.set_ylabel('Lock-in Rate')
        ax2.legend(title='Pipeline Variant')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Peak X comparison
        ax3 = axes[1,0]
        pivot_peak = comparative_df.pivot(index='i_definition', columns='pipeline_variant', values='peak_x')
        pivot_peak.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('Peak X Value by I Definition and Pipeline Variant')
        ax3.set_ylabel('Peak X Value')
        ax3.legend(title='Pipeline Variant')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Total universes comparison
        ax4 = axes[1,1]
        pivot_universes = comparative_df.pivot(index='i_definition', columns='pipeline_variant', values='total_universes')
        pivot_universes.plot(kind='bar', ax=ax4, color=['#FF6B6B', '#4ECDC4'])
        ax4.set_title('Total Universes by I Definition and Pipeline Variant')
        ax4.set_ylabel('Total Universes')
        ax4.legend(title='Pipeline Variant')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # NOTE: constrained_layout=True already set, no need for plt.tight_layout()
        
        # Save plot
        summary_plot_path = os.path.join(save_dir, "multi_i_parameter_summary_analysis.png")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed analysis report
        report_path = os.path.join(save_dir, "i_parameter_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("MULTI-I PARAMETER ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total combinations tested: {len(comparative_df)}\n")
            f.write(f"I definitions: {', '.join(comparative_df['i_definition'].unique())}\n")
            f.write(f"Pipeline variants: {', '.join(comparative_df['pipeline_variant'].unique())}\n\n")
            
            f.write("STABILITY RATE ANALYSIS:\n")
            f.write("-" * 25 + "\n")
            stability_stats = comparative_df.groupby('i_definition')['stability_rate'].agg(['mean', 'std', 'min', 'max'])
            f.write(stability_stats.to_string())
            f.write("\n\n")
            
            f.write("LOCK-IN RATE ANALYSIS:\n")
            f.write("-" * 22 + "\n")
            lockin_stats = comparative_df.groupby('i_definition')['lockin_rate'].agg(['mean', 'std', 'min', 'max'])
            f.write(lockin_stats.to_string())
            f.write("\n\n")
            
            f.write("PEAK X VALUE ANALYSIS:\n")
            f.write("-" * 22 + "\n")
            peak_stats = comparative_df.groupby('i_definition')['peak_x'].agg(['mean', 'std', 'min', 'max'])
            f.write(peak_stats.to_string())
            f.write("\n\n")
            
            f.write("BEST PERFORMING COMBINATIONS:\n")
            f.write("-" * 30 + "\n")
            best_stability = comparative_df.loc[comparative_df['stability_rate'].idxmax()]
            best_lockin = comparative_df.loc[comparative_df['lockin_rate'].idxmax()]
            
            f.write(f"Highest Stability Rate: {best_stability['i_definition']} + {best_stability['pipeline_variant']} ({best_stability['stability_rate']:.3f})\n")
            f.write(f"Highest Lock-in Rate: {best_lockin['i_definition']} + {best_lockin['pipeline_variant']} ({best_lockin['lockin_rate']:.3f})\n")
        
        return summary_plot_path
        
    except Exception as e:
        print(f"⚠️ Error creating summary analysis: {e}")
        return None


def run_single_i_parameter_mode(i_definition: str = "kl_shannon", pipeline_variant: str = "full") -> dict:
    """
    Run pipeline for a single I parameter definition with categorization.
    
    Args:
        i_definition: I parameter definition to use
        pipeline_variant: Pipeline variant to use
    
    Returns:
        dict: Analysis results
    """
    print("=" * 80)
    print(f"STARTING SINGLE I PARAMETER MODE ")
    print("=" * 80)
    print(f" I Parameter Definition: {i_definition}")
    print(f"🔄 Pipeline Variant: {pipeline_variant}")
    print("=" * 80)
    
    # Set configuration
    config = MASTER_CTRL.copy()
    config["I_DEFINITION_MODE"] = i_definition
    config["PIPELINE_VARIANT"] = pipeline_variant
    
    # Create directory structure for single I parameter on Google Drive
    if IN_COLAB:
        base_dir = "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro"
    else:
        base_dir = "/Users/stevilen/Desktop/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_Pro"
    
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    pipeline_name = f"TQE_Universe_Simulation_{i_definition.title()}_Pipeline_v4.2.0_Pro_{run_timestamp}"
    
    single_i_dir = os.path.join(base_dir, pipeline_name)
    os.makedirs(single_i_dir, exist_ok=True)
    
    # Set the save directory for this run
    config["MULTI_I_ANALYSIS_MODE"] = True
    config["MULTI_I_SAVE_DIR"] = single_i_dir
    
    # Create run ID
    run_id = f"SINGLE_I_{i_definition}_{pipeline_variant}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Run pipeline
        result = run_pipeline(config_override=config, run_id_override=run_id)
        
        print(f" Completed single I parameter analysis: {i_definition} + {pipeline_variant}")
        print(f"📁 Results saved with categorization in: {run_id}")
        
        return {
            "i_definition": i_definition,
            "pipeline_variant": pipeline_variant,
            "result": result,
            "run_id": run_id
        }
        
    except Exception as e:
        print(f"Error in single I parameter mode: {e}")
        return {
            "i_definition": i_definition,
            "pipeline_variant": pipeline_variant,
            "error": str(e),
            "run_id": run_id
    }
# ======================================================
# MAIN ORCHESTRATOR
# ======================================================
# MAIN ORCHESTRATOR
# ======================================================


def run_pipeline(config_override: dict = None, run_id_override: str = None) -> dict:
    """
    Main pipeline orchestrator. Executes all 21 phases sequentially.
    
    Returns:
        dict: Summary metrics for comparative analysis
    """
    # Apply Colab optimizations if needed
    if IN_COLAB:
        optimize_for_colab()
    
    # Initialize context
    config = config_override if config_override else MASTER_CTRL.copy()
    ctx = PipelineContext(config, run_id_override)

    # NOTE: PHASE 0 Goldilocks calibration is now built-in (run_builtin_goldilocks_calibration)
    # External calibration available via TQE_GoldiLock_Calibration_Pipeline.py
    
    # Verify deterministic seed setup
    if ctx.config.get("USE_STRICT_SEED", True):
        # Ensure all random number generators are properly seeded
        test_rng = np.random.default_rng(ctx.master_seed)
        test_value = test_rng.random()
        if ctx.config.get("VERBOSE", True):
            print(f"Deterministic seed verification: master_seed={ctx.master_seed}, test_value={test_value:.6f}")
    
    # Physics engine
    physics = PhysicsEngine(ctx.config, ctx.rng)
    
    # One-line summary
    print("-" * 60)
    print(f"Main results directory:\n{ctx.paths['SAVE_DIR']}")
    
    # Pipeline type display
    print(f"TQE (Theory of the Question of Existence) Universe Simulation Pipeline v4.2.0 Professional")
    if ctx.variant == "energy_only":
        print(f"Pipeline Type:      E-ONLY (Energy Only, I disabled)")
        print(f"Analysis Mode:      Energy parameter analysis only")
    else:
        print(f"Pipeline Type:      E+I (Energy + Information)")
        print(f"Analysis Mode:      Full E+I interaction analysis")
    
    # I-Definition Mode (only relevant in E+I mode)
    if ctx.variant == "energy_only":
        print(f"I-Definition Mode:  N/A (Energy only - I parameter disabled)")
    else:
        print(f"I-Definition Mode:  {ctx.config.get('I_DEFINITION_MODE','kl_shannon')}")
    
    print(f"Enhanced Physics: {'Enabled' if ctx.config.get('USE_ENHANCED_PHYSICS', True) else 'Disabled'}")
    
    print(f"Using master seed: {ctx.master_seed}")
    print("-" * 60)
    
    # Progress bar with pipeline type
    pipeline_desc = f"TQE (Theory of the Question of Existence) {'E_only' if ctx.variant == 'energy_only' else 'EI'}_Pipeline_v4.2.0_Pro"
    progress = tqdm(total=28, desc=pipeline_desc)  # PHASE 1-28 (Goldilocks integrated into PHASE 1)
    
    # Memory optimization for Colab
    if ctx.config.get("REDUCE_MEMORY_USAGE", False):
        cleanup_memory()
    
    df = pd.DataFrame()
    X_c_low, X_c_high, peak_x = None, None, None # Variables shared across phases

    # ======================================================
    # OPTIMIZED PIPELINE EXECUTION STRUCTURE
    # ======================================================
    
    # ===== GROUP 1: CORE SIMULATION & DATA GENERATION =====
    # 1. Monte Carlo Simulation + Goldilocks Calibration (INTEGRATED!)
    progress.set_description("1/28: Monte Carlo + Bayesian Goldilocks")
    # NOTE: Goldilocks is computed FROM the same universes generated here
    df, X_c_low_used, X_c_high_used = phase_01_monte_carlo(ctx)
    
    # Save main universe data CSV
    ctx.save_csv(df, os.path.join(ctx.paths["AGGREGATE_DIR"], "tqe_runs.csv"))
    progress.update(1)
    
    # ===== GROUP 2: BASIC ANALYSIS & VISUALIZATION =====
    # 2. Stability curve analysis
    progress.set_description("2/28: Stability Curve Analysis")
    peak_x = phase_02_stability_curve(ctx, df)
    progress.update(1)
    
    # 3. E-I parameter space visualization
    progress.set_description("3/28: E-I Parameter Space")
    phase_03_scatter_ei(ctx, df)
    progress.update(1)
    
    # 4. Fluctuation dynamics
    progress.set_description("4/28: Fluctuation Dynamics")
    phase_04_fluctuation_panels(ctx, df)
    progress.update(1)
    
    # ===== GROUP 3: STABILITY & LOCK-IN ANALYSIS =====
    # 5. Stability-by-I analysis
    progress.set_description("5/28: Stability-by-I Analysis")
    phase_05_stability_by_i(ctx, df)
    progress.update(1)
    
    # 6. Lock-in histogram
    progress.set_description("6/28: Lock-in Histogram")
    phase_06_lockin_histogram(ctx, df)
    progress.update(1)
    
    # 7. Stability distribution
    progress.set_description("7/28: Stability Distribution")
    phase_07_stability_distribution(ctx, df)
    progress.update(1)
    
    # 8. Average lock-in curve
    progress.set_description("8/28: Average Lock-in Curve")
    phase_08_avg_lockin_curve(ctx, df)
    progress.update(1)
    
    # ===== GROUP 4: MACHINE LEARNING & EMERGENT LAWS =====
    # 9. Feature importance analysis
    progress.set_description("9/28: Feature Importance Analysis")
    phase_09_feature_importance(ctx, df)
    progress.update(1)
    
    # 10. Emergent laws detection
    progress.set_description("10/28: Emergent Laws Detection")
    phase_10_emergent_laws(ctx, df)
    progress.update(1)
    
    # 11. Statistical finetuning detector
    progress.set_description("11/28: Statistical Finetuning Detector")
    phase_11_finetuning_detector(ctx, df)
    progress.update(1)
    
    # ===== GROUP 5: CMB GENERATION & VALIDATION =====
    # 12. Best universe plots & CMB map generation (generates simulated CMB FITS files)
    progress.set_description("12/28: Best Universe & CMB Generation")
    phase_12_best_universe_plots(ctx, df)
    progress.update(1)
    
    # 13. Generate missing CMB maps (ensures all lock-in universes have CMB maps)
    progress.set_description("13/28: Complete CMB Map Coverage")
    phase_13_generate_missing_cmb_maps(ctx, df)
    progress.update(1)
    
    # 14. Entropy volatility analysis
    progress.set_description("14/28: Entropy Volatility Analysis")
    phase_14_entropy_volatility(ctx, df)
    progress.update(1)
    
    # 15. Planck validation (ONLY phase that uses Planck observational data for chi-squared comparison)
    progress.set_description("15/28: Planck Observational Comparison")
    df_planck, planck_chi2 = phase_15_planck_validation(ctx, df)
    progress.update(1)
    
    # 16. CMB anomaly detection (coldspots, Axis of Evil detection on simulated maps)
    progress.set_description("16/28: CMB Anomaly Detection")
    phase_16_cmb_anomaly_detection(ctx, df)
    progress.update(1)
    
    # ===== GROUP 6: E+I INTERACTION ANALYSIS =====
    # 17. E+I importance comparison
    progress.set_description("17/28: E+I Importance Comparison")
    phase_17_ei_importance_comparison(ctx, df)
    progress.update(1)
    
    # 18. I-Definitions Goldilocks comparison (Bayesian zones for each I-def)
    progress.set_description("18/28: I-Definitions Goldilocks Zones")
    phase_18_multi_mode_goldilocks_comparison(ctx, df)
    progress.update(1)
    
    # ===== GROUP 7: ADVANCED CMB ANALYSIS =====
    # 19. CMB analysis plots (Gaussianity, Isotropy, Power Spectrum) - aggregates simulated CMB maps
    progress.set_description("19/28: CMB Statistical Analysis")
    phase_19_cmb_analysis_plots(ctx, df)
    progress.update(1)
    
    # 20. Comprehensive correlation analysis
    progress.set_description("20/28: Comprehensive Correlation Analysis")
    phase_20_comprehensive_correlation_analysis(ctx, df)
    progress.update(1)
    
    # ===== GROUP 8: ADVANCED STATISTICAL ANALYSIS =====
    # 21. Advanced statistical analysis
    progress.set_description("21/28: Advanced Statistical Analysis")
    phase_21_advanced_statistical_analysis(ctx, df)
    progress.update(1)
    
    # 22. CMB anomaly analysis plots (aggregate anomaly overlays from Phase 16 detections)
    progress.set_description("22/28: CMB Anomaly Visualization")
    phase_22_cmb_anomaly_analysis_plots(ctx, df)
    progress.update(1)
    
    # ===== GROUP 9: ENHANCED PHYSICS ANALYSIS =====
    # 23. Enhanced physics analysis
    progress.set_description("23/28: Enhanced Physics Analysis")
    phase_23_enhanced_physics_analysis(ctx, df)
    progress.update(1)
    
    # 24. Comprehensive data extraction from all universes
    progress.set_description("24/28: Comprehensive Data Extraction")
    phase_24_comprehensive_data_extraction(ctx, df)
    progress.update(1)
    
    # ===== GROUP 10: ADVANCED ANOMALY & LAW DETECTION =====
    # 25. Advanced anomaly detection
    progress.set_description("25/28: Advanced Anomaly Detection")
    phase_25_advanced_anomaly_detection(ctx, df)
    progress.update(1)
    
    # 26. Advanced law detection
    progress.set_description("26/28: Advanced Law Detection")
    phase_26_advanced_law_detection(ctx, df)
    progress.update(1)
    
    # ===== GROUP 11: COMPREHENSIVE VISUALIZATION =====
    # 27. Comprehensive visualization extraction
    progress.set_description("27/28: Comprehensive Visualization Extraction")
    phase_27_comprehensive_visualization_extraction(ctx, df)
    progress.update(1)
    
    # ===== GROUP 12: FINAL SUMMARY & BAYESIAN =====
    # 28. Final Summary & Bayesian Integration
    progress.set_description("28/28: Final Summary & Bayesian Integration")
    
    # Generate summary FIRST
    summary = phase_28_final_summary(ctx, df, peak_x)
    
    # Bayesian Model Selection (BIC, AIC)
    bayesian_metrics = {}
    if ctx.config.get("ENABLE_BAYESIAN_ANALYSIS", False) and planck_chi2 is not None:
        bayesian_metrics = compute_bayesian_model_selection(ctx, df, planck_chi2)
        save_bayesian_metrics_csv(ctx, bayesian_metrics, {})
        plot_bayesian_comparison(ctx, bayesian_metrics)
    
    # Nested Sampling (Bayesian Evidence)
    nested_results = {}
    if ctx.config.get("ENABLE_NESTED_SAMPLING", False):
        try:
            nested_results = run_nested_sampling(ctx, df)
            if nested_results:
                save_bayesian_metrics_csv(ctx, bayesian_metrics, nested_results)
        except Exception as e:
            # FIX: Don't crash pipeline if nested sampling fails
            print(f"[NESTED SAMPLING] Skipped due to error: {e}")
            nested_results = {}
    
    # Add validation flag IMMEDIATELY (even if Bayesian fails)
    # FIX: Set pipeline_completed flag BEFORE Bayesian integration to prevent false "failed" detection
    if summary and "stability_summary" in summary and summary["stability_summary"].get("total_universes", 0) > 0:
        summary["pipeline_completed"] = True
        summary["pipeline_status"] = "success"
    else:
        summary = summary or {}
        summary["pipeline_completed"] = False
        summary["pipeline_status"] = "partial"
    
    # Add Bayesian metrics to summary (if available)
    if bayesian_metrics or nested_results:
        summary["bayesian_model_selection"] = {
            "BIC": bayesian_metrics.get("BIC", None),
            "AIC": bayesian_metrics.get("AIC", None),
            "AICc": bayesian_metrics.get("AICc", None),
            "log_likelihood": bayesian_metrics.get("log_likelihood", None),
            "chi_squared_reduced": bayesian_metrics.get("chi_squared_reduced", None),
            "log_evidence": nested_results.get("log_evidence", None),
            "log_evidence_error": nested_results.get("log_evidence_error", None),
            "nested_sampling_status": "completed" if nested_results else "disabled"
        }

    # Integrate complexity & life-compatibility analysis
    summary = integrate_complexity_analysis(ctx, df, summary, bayesian_metrics)

    progress.update(1)
    
    progress.close()
    
    # Print standardized completion summary
    _print_pipeline_completion(summary, ctx)
    
    return summary



def _print_pipeline_completion(summary: dict, ctx: PipelineContext):
    """Print standardized completion summary for all pipeline runs."""
    pipeline_type = summary.get("pipeline_type", "E-ONLY" if ctx.variant == "energy_only" else "E+I")
    i_def = summary.get("i_definition", "N/A")
    
    stab_sum = summary.get("stability_summary", {})
    total = int(stab_sum.get("total_universes", 0))
    stable = int(stab_sum.get("stable_count", stab_sum.get("stable_universes", 0)))
    unstable = int(stab_sum.get("unstable_count", stab_sum.get("unstable_universes", 0)))
    lockin = int(stab_sum.get("lockin_count", stab_sum.get("lockin_universes", 0)))
    
    stable_pct = float(stab_sum.get("stable_percent", 0.0))
    unstable_pct = float(stab_sum.get("unstable_percent", 0.0))
    lockin_pct = float(stab_sum.get("lockin_percent", 0.0))
    
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETED: {pipeline_type}")
    print(f"{'='*70}")
    print(f"I-Definition:     {i_def}")
    print(f"Total Universes:  {total}")
    print(f"  ✓ Stable:       {stable} ({stable_pct:.1f}%)")
    print(f"  ✗ Unstable:     {unstable} ({unstable_pct:.1f}%)")
    print(f"  🔒 Lock-in:     {lockin} ({lockin_pct:.1f}%)")
    
    # Goldilocks info
    gold = summary.get("goldilocks_window_used", {})
    if gold.get("X_peak"):
        X_peak = gold.get("X_peak", 0)
        X_unc = gold.get("X_peak_uncertainty", 0)
        X_low = gold.get("X_low_plot_est", 0)
        X_high = gold.get("X_high_plot_est", 0)
        print(f"Goldilocks Peak:  X = {X_peak:.2f} ± {X_unc:.2f}")
        print(f"Goldilocks Zone:  [{X_low:.2f}, {X_high:.2f}]")
    
    # Bayesian info
    if gold.get("mode") == "bayesian_adaptive":
        sampled = gold.get("total_sampled", 0)
        kappa = gold.get("ucb_kappa", 0)
        print(f"Bayesian Method:  GP + UCB (κ={kappa:.1f}, sampled={sampled})")
    
    master_seed = summary.get('master_seed', 0)
    print(f"Master Seed:      {master_seed}")
    print(f"Save Directory:   {ctx.paths['SAVE_DIR']}")
    print(f"{'='*70}\n")

# ======================================================
# MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    
    # Ensure Colab/Windows compatibility for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    
    # Print header
    print("\n" + "="*70)
    print("TQE UNIVERSE SIMULATION PIPELINE v4.2.0 PRO")
    print("="*70)
    
    # ===================================================================
    # RUN MODE ORCHESTRATION (v4.2.0 PRO)
    # ===================================================================
    # 4 execution modes:
    #   • single_eonly: E-only baseline (I disabled, Bayesian Goldilocks integrated in Phase 1)
    #   • single_ei:    Single E+I run with selected I-definition (Bayesian Goldilocks integrated in Phase 1)
    #   • batch_ei:     All 10 I-definitions (independent runs, each with Goldilocks in Phase 1)
    #   • batch_all:    E-only + all 10 I-definitions (11 independent runs)
    # 
    # Each run executes PHASES 1-28 independently with Bayesian Goldilocks integrated into Phase 1.
    # ===================================================================
    
    run_mode = MASTER_CTRL.get("RUN_MODE", "single_ei")
    
    # 10 I-parameter definitions (removed horizon_entropy and phenomenological)
    # NEW: jensen_shannon added (symmetric KL-divergence, validated with Planck 2018 CMB data)
    ALL_I_DEFINITIONS = [
        "kl_divergence", "shannon", "renyi", "mutual_info", 
        "composite", "kl_shannon", "entanglement", "fisher", 
        "fisher_kl_fusion", "jensen_shannon"  # NEW: Symmetric KL-divergence (validated with Planck 2018)
            ]
            
    # Create run-mode-specific directory WITH TIMESTAMP
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if IN_COLAB:
        base_dir = "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO"
    else:
        base_dir = os.path.join(os.getcwd(), "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO")
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create run-mode-specific subdirectory WITH TIMESTAMP (prevents overwriting)
    mode_dir = os.path.join(base_dir, f"TQE_Universe_Simulation_{run_mode}_{timestamp}")
    os.makedirs(mode_dir, exist_ok=True)
    
    if run_mode in ["batch_ei", "batch_all"]:
        # Batch modes: Use timestamped mode_dir for batch runs
        batch_dir = mode_dir
        print(f"Batch directory: {batch_dir}\n")
    else:
        # Single modes: Use timestamped mode_dir for single runs
        single_run_dir = mode_dir
        print(f"Run directory: {single_run_dir}\n")
    
    # ===================================================================
    # RUN MODE EXECUTION
    # ===================================================================
    
    # ===================================================================
    # MODE 1: SINGLE E-ONLY
    # ===================================================================
    # Baseline simulation with I parameter disabled (energy-only coupling)
    # - Executes PHASES 1-28 with Bayesian Goldilocks integrated into Phase 1
    # - Generates simulated CMB maps (Phase 12-13)
    # - Compares to Planck 2018 data (Phase 15 only)
    # - Provides ΛCDM-equivalent baseline for TQE comparison
    # ===================================================================
    if run_mode == "single_eonly":
        print("=" * 70)
        print("RUN MODE: SINGLE E-ONLY (Baseline)")
        print("=" * 70)
        print(f"Run directory: {single_run_dir}\n")
        
        config = MASTER_CTRL.copy()
        config["PIPELINE_VARIANT"] = "energy_only"
        config["DRIVE_BASE_DIR"] = single_run_dir
        
        result = run_pipeline(config_override=config)
        # Summary printed by _print_pipeline_completion()
    
    # ===================================================================
    # MODE 2: SINGLE E+I (one specific I-definition)
    # ===================================================================
    # TQE simulation with Energy-Information coupling
    # - Executes PHASES 1-28 with selected I-definition
    # - Bayesian Goldilocks calibration integrated into Phase 1
    # - Generates simulated CMB maps with E-I coupling (Phase 12-13)
    # - Detects emergent CMB anomalies (Phase 16)
    # - Aggregates CMB statistics from simulated maps (Phase 19)
    # - Compares to Planck 2018 data (Phase 15 only)
    # ===================================================================
    elif run_mode == "single_ei":
        print("=" * 70)
        print("RUN MODE: SINGLE E+I (TQE Coupling)")
        print("=" * 70)
        
        selected_i_def = MASTER_CTRL.get("I_DEFINITION_MODE", "kl_shannon")
        print(f"I-Definition: {selected_i_def}")
        print(f"Run directory: {single_run_dir}\n")
        
        config = MASTER_CTRL.copy()
        config["PIPELINE_VARIANT"] = "full"
        config["I_DEFINITION_MODE"] = selected_i_def
        config["DRIVE_BASE_DIR"] = single_run_dir
        
        result = run_pipeline(config_override=config)
        # Summary printed by _print_pipeline_completion()
    
    # ===================================================================
    # MODE 3: BATCH E+I (all 10 I-definitions, NO E-only)
    # ===================================================================
    # Batch execution: All 10 I-parameter definitions independently
    # - Each I-definition runs PHASES 1-28 independently
    # - Each has its own Bayesian Goldilocks calibration (Phase 1)
    # - Each generates independent simulated CMB maps (Phase 12-13)
    # - Results saved to separate timestamped directories
    # - Use external comparison tool for cross-definition analysis
    # ===================================================================
    elif run_mode == "batch_ei":
        print("=" * 70)
        print("RUN MODE: BATCH E+I (10 I-definitions)")
        print("=" * 70)
        print(f"Batch Directory: {batch_dir}\n")
        
        successful = 0
        failed = 0
        
        for idx, i_def in enumerate(ALL_I_DEFINITIONS):
            print(f"\n{'─'*70}")
            print(f"E+I Run {idx+1}/10: {i_def}")
            print(f"{'─'*70}")
            
            config = MASTER_CTRL.copy()
            config["PIPELINE_VARIANT"] = "full"
            config["I_DEFINITION_MODE"] = i_def
            config["MULTI_I_ANALYSIS_MODE"] = True
            config["MULTI_I_SAVE_DIR"] = batch_dir
            
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_id = f"EplusI_{i_def}_{run_timestamp}"
            
            try:
                result = run_pipeline(config_override=config, run_id_override=run_id)
                # Summary printed by _print_pipeline_completion()
                if result and result.get('pipeline_completed', False):
                    successful += 1
                else:
                    failed += 1
                    print(f"⚠️ '{i_def}' pipeline returned no results\n")
            except Exception as e:
                failed += 1
                print(f"❌ ERROR in '{i_def}': {e}\n")
        
        print(f"\n{'='*70}")
        print(f"BATCH E+I COMPLETED: {successful}/10 successful, {failed}/10 failed")
        print(f"Results saved to: {batch_dir}")
        print(f"{'='*70}")
    
    # ===================================================================
    # MODE 4: BATCH ALL (E-only + all 10 I-definitions)
    # ===================================================================
    # Comprehensive batch: E-only baseline + all 10 I-definitions
    # - Total: 11 independent runs (1 E-only + 10 E+I)
    # - Each runs PHASES 1-28 independently
    # - Each has its own Bayesian Goldilocks calibration (Phase 1)
    # - Each generates independent simulated CMB maps (Phase 12-13)
    # - Each compares to Planck 2018 data (Phase 15)
    # - Results saved to separate timestamped directories
    # - Use external comparison tool for cross-run analysis
    # ===================================================================
    elif run_mode == "batch_all":
        print("=" * 70)
        print("RUN MODE: BATCH ALL (E-only + 10 I-definitions)")
        print("=" * 70)
        print(f"Batch Directory: {batch_dir}\n")
        
        successful = 0
        failed = 0
        
        # 1. Run E-only
        print(f"{'─'*70}")
        print(f"E-only Run (1/11)")
        print(f"{'─'*70}")
        
        config_eonly = MASTER_CTRL.copy()
        config_eonly["PIPELINE_VARIANT"] = "energy_only"
        config_eonly["MULTI_I_ANALYSIS_MODE"] = True
        config_eonly["MULTI_I_SAVE_DIR"] = batch_dir
        config_eonly.pop("DRIVE_BASE_DIR", None)  # Remove to use MULTI_I path
        
        eonly_timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id_eonly = f"Eonly_{eonly_timestamp}"
        
        try:
            result_eonly = run_pipeline(config_override=config_eonly, run_id_override=run_id_eonly)
            # Summary printed by _print_pipeline_completion()
            if result_eonly and result_eonly.get('pipeline_completed', False):
                successful += 1
            else:
                failed += 1
                print(f"⚠️ E-only pipeline returned no results\n")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR in E-only: {e}\n")
        
        # 2. Run all 10 E+I
        for idx, i_def in enumerate(ALL_I_DEFINITIONS):
            print(f"\n{'─'*70}")
            print(f"E+I Run {idx+2}/11: {i_def}")
            print(f"{'─'*70}")
            
            config_ei = MASTER_CTRL.copy()
            config_ei["PIPELINE_VARIANT"] = "full"
            config_ei["I_DEFINITION_MODE"] = i_def
            config_ei["MULTI_I_ANALYSIS_MODE"] = True
            config_ei["MULTI_I_SAVE_DIR"] = batch_dir
            config_ei.pop("DRIVE_BASE_DIR", None)  # Remove to use MULTI_I path
            
            ei_timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_id_ei = f"EplusI_{i_def}_{ei_timestamp}"
            
            try:
                result_ei = run_pipeline(config_override=config_ei, run_id_override=run_id_ei)
                # Summary printed by _print_pipeline_completion()
                if result_ei and result_ei.get('pipeline_completed', False):
                    successful += 1
                else:
                    failed += 1
                    print(f"⚠️ '{i_def}' pipeline returned no results\n")
            except Exception as e:
                failed += 1
                print(f"❌ ERROR in '{i_def}': {e}\n")
        
        print(f"\n{'='*70}")
        print(f"BATCH ALL COMPLETED: {successful}/11 successful, {failed}/11 failed")
        print(f"Results saved to: {batch_dir}")
        print(f"{'='*70}")
    
    else:
        print(f"❌ ERROR: Unknown RUN_MODE '{run_mode}'")
        print(f"   Valid modes: single_eonly, single_ei, batch_ei, batch_all")
        sys.exit(1)
    
    # ===================================================================
    # FINAL MESSAGE
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"TQE Universe Simulation Pipeline v4.2.0 PRO - Execution Complete")
    print(f"Enhanced Physics: {'Enabled' if MASTER_CTRL.get('USE_ENHANCED_PHYSICS', True) else 'Disabled'}")
    print(f"{'='*70}")
