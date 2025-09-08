# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_01_EI_UNIVERSE_SIMULATION_Master_Control.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This module serves as the main execution driver and orchestrator for the
# entire simulation pipeline. When executed directly, it systematically runs
# each stage of the simulation in the predefined order.
#
# It imports the final `ACTIVE` configuration from the
# `TQE_00_..._config.py` module, which serves as the single source of truth
# for all parameters. The core logic iterates through a `STAGES` list,
# dynamically importing and calling each module's main function while
# respecting the on/off switches defined in the `PIPELINE` configuration.
# A `safe_call` utility ensures that the failure of an optional or
# non-critical stage does not halt the entire pipeline.
#
# ===================================================================================

import os
import sys
import traceback
import importlib
from pathlib import Path

# Ensure repository code directory is on sys.path when executed directly
CODE_DIR = Path(__file__).parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

# Import the single source of truth for configuration
from TQE_00_EI_UNIVERSE_SIMULATION_config import ACTIVE

# IO helpers (your real modules; shims acceptable in Colab)
try:
    from TQE_02_EI_UNIVERSE_SIMULATION_Output_io_paths import resolve_output_paths, ensure_colab_drive_mounted
except ImportError:
    # Fallback for simplified environments
    def resolve_output_paths(cfg):
        run_dir = Path.cwd() / "tqe_output"
        return {"primary_run_dir": str(run_dir), "fig_dir": str(run_dir / "figs"), "mirrors": []}
    def ensure_colab_drive_mounted(cfg):
        pass # No-op


def safe_call(module_name: str, func_name: str, kwargs: dict):
    """
    Import module and call function safely; log errors, but do not crash the pipeline.
    This keeps long runs robust even if an optional stage is missing.
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[SKIP] Cannot import module '{module_name}': {e}")
        return None

    fn = getattr(mod, func_name, None)
    if fn is None:
        print(f"[SKIP] Function '{func_name}' not found in '{module_name}'.")
        return None

    try:
        print(f"[RUN] {module_name}.{func_name}()")
        out = fn(**kwargs)
        print(f"[OK ] {module_name}.{func_name} finished.")
        return out
    except Exception as e:
        print(f"[ERR] {module_name}.{func_name} raised: {e}")
        print("".join(traceback.format_exc()))
        return None


def main():
    # Best-effort: mount Drive if enabled (no-op outside Colab)
    try:
        ensure_colab_drive_mounted(ACTIVE)
    except Exception as e:
        print("[WARN] Drive mount skipped:", e)

    # Prepare output folders (primary run dir + figure dir)
    paths   = resolve_output_paths(ACTIVE)
    run_dir = paths["primary_run_dir"]
    fig_dir = paths["fig_dir"]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("========== TQE PIPELINE START ==========")
    print(f"run_dir: {run_dir}")
    print(f"fig_dir: {fig_dir}")

    # This dictionary will store the results from each stage
    results = {}

    # Ordered stage plan. Each tuple: (stage_name, pipeline_flag, module_name, function_name, kwargs_builder)
    STAGES = [
        ("energy_sampling", "run_energy_sampling",
         "TQE_05_EI_UNIVERSE_SIMULATION_E_energy_sampling", "run_energy_sampling",
         lambda data: {"active_cfg": ACTIVE}),

        ("info_bootstrap", "run_info_bootstrap",
         "TQE_06_EI_UNIVERSE_SIMULATION_I_information_bootstrap", "run_information_bootstrap",
         lambda data: {"active_cfg": ACTIVE}),
        
        ("fluctuation", "run_fluctuation",
         "TQE_07_EI_UNIVERSE_SIMULATION_t_lt_0_fluctuation", "run_fluctuation_stage",
         lambda data: {"active_cfg": ACTIVE}),

        ("superposition", "run_superposition",
         "TQE_08_EI_UNIVERSE_SIMULATION_t_lt_0_superposition", "run_superposition_stage",
         lambda data: {"active_cfg": ACTIVE, "arrays": data.get("fluctuation", {}).get("arrays")}),

        ("collapse", "run_collapse",
         "TQE_09_EI_UNIVERSE_SIMULATION_t_eq_0_collapse_LawLockin", "run_lockin_stage",
         lambda data: {"active_cfg": ACTIVE, "arrays": data.get("superposition", {}).get("arrays")}),

        ("expansion", "run_expansion",
         "TQE_10_EI_UNIVERSE_SIMULATION_t_gt_0_expansion", "run_expansion_stage",
         lambda data: {"active_cfg": ACTIVE, "collapse_df": data.get("collapse", {}).get("table")}),
        
        ("montecarlo", "run_montecarlo",
         "TQE_11_EI_UNIVERSE_SIMULATION_montecarlo", "run_montecarlo_stage",
         lambda data: {"active_cfg": ACTIVE, 
                       "collapse_df": data.get("collapse", {}).get("table"), 
                       "expansion_df": data.get("expansion", {}).get("table")}),
        
        ("best_universe", "run_best_universe",
         "TQE_12_EI_UNIVERSE_SIMULATION_best_universe", "run_best_universe",
         lambda data: {"active_cfg": ACTIVE, 
                       "montecarlo_df": data.get("montecarlo", {}).get("table")}),
        
        ("cmb_map", "run_cmb_map",
         "TQE_13_EI_UNIVERSE_SIMULATION_cmb_map_generation", "run_cmb_map_generation",
         lambda data: {"active_cfg": ACTIVE}),

        ("finetune_diag", "run_finetune_diag",
         "TQE_14_EI_UNIVERSE_SIMULATION_finetune_diagnostics", "run_finetune_stage",
         lambda data: {"active_cfg": ACTIVE}),
        
        ("anomaly_cold_spot", "run_anomaly_scan",
         "TQE_15_EI_UNIVERSE_SIMULATION_anomaly_cold_spot", "run_anomaly_cold_spot_stage",
         lambda data: {"active_cfg": ACTIVE}),

        ("anomaly_low_multipole", "run_anomaly_scan",
         "TQE_16_EI_UNIVERSE_SIMULATION_anomaly_low_multipole_alignments", "run_anomaly_low_multipole_alignments_stage",
         lambda data: {"active_cfg": ACTIVE}),
        
        ("anomaly_llac", "run_anomaly_scan",
         "TQE_17_EI_UNIVERSE_SIMULATION_anomaly_LackOfLargeAngleCorrelation", "run_llac_stage",
         lambda data: {"active_cfg": ACTIVE}),

        ("anomaly_hpa", "run_anomaly_scan",
         "TQE_18_EI_UNIVERSE_SIMULATION_anomaly_HemisphericalAsymmetry", "run_hpa",
         lambda data: {"active_cfg": ACTIVE}),

        ("xai", "run_xai",
         "TQE_19_EI_UNIVERSE_SIMULATION_xai", "run_xai",
         lambda data: {"active_cfg": ACTIVE}),

        ("manifest", "run_manifest",
         "TQE_20_EI_UNIVERSE_SIMULATION_results_manifest", "run_results_manifest",
         lambda data: {"active_cfg": ACTIVE, "run_dir": run_dir}),
    ]

    # Execute stages in order, honoring PIPELINE flags and passing data
    for name, flag, module_name, func_name, make_kwargs in STAGES:
        if not ACTIVE["PIPELINE"].get(flag, False):
            print(f"[SKIP] {module_name} ({func_name}) because {flag}=False")
            continue
        
        kwargs = make_kwargs(results)
        output = safe_call(module_name, func_name, kwargs)
        results[name] = output

    print("=========== TQE PIPELINE END ===========")


if __name__ == "__main__":
    main()
