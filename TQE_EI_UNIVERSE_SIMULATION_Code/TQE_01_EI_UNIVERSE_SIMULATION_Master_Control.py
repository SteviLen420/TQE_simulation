# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_01_EI_UNIVERSE_SIMULATION_Master_Control.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This module serves as the central control hub and main execution driver for the
# simulation pipeline. Its primary roles are twofold:
#
# 1.  **Defines the Master Configuration (`MASTER_CTRL`):** It contains the
#     comprehensive default parameters for every stage of the simulation, from
#     energy sampling to XAI analysis. This includes pipeline switches to
#     enable/disable stages, detailed physics and computational parameters, and
#     pre-defined run "profiles" (e.g., "demo", "paper") that provide specific
#     overrides for different use cases.
#
# 2.  **Acts as the Pipeline Orchestrator:** When executed directly, the
#     `if __name__ == "__main__"` block functions as the main execution harness.
#     It dynamically imports and calls each stage of the pipeline in sequence,
#     respecting the on/off switches defined in the active configuration.
#
# It also provides the `_deep_merge` utility function used by the configuration
# resolver to combine the base settings with a selected profile.
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
from io_paths import resolve_output_paths, ensure_colab_drive_mounted


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

    # Tag for EI/E in filenames, based on PIPELINE.use_information
    tag = "EI" if ACTIVE["PIPELINE"].get("use_information", True) else "E"

    # Ordered stage plan. Each tuple: (pipeline_flag, module_name, function_name, kwargs_builder)
    STAGES = [
        ("run_energy_sampling",
         "TQE_05_EI_UNIVERSE_SIMULATION_E_energy_sampling",
         "run_energy_sampling",
         lambda: {"active": ACTIVE, "tag": tag}),

        ("run_info_bootstrap",
         "TQE_06_EI_UNIVERSE_SIMULATION_I_information_bootstrap",
         "run_information_bootstrap",
         lambda: {"active": ACTIVE}),

        ("run_fluctuation",
         "TQE_07_EI_UNIVERSE_SIMULATION_t_lt_0_fluctuation",
         "run_fluctuation_stage",
         lambda: {"active": ACTIVE}),

        ("run_superposition",
         "TQE_08_EI_UNIVERSE_SIMULATION_t_lt_0_superposition",
         "run_superposition_stage",
         lambda: {"active": ACTIVE}),

        ("run_lockin",
         "TQE_09_EI_UNIVERSE_SIMULATION_t_eq_0_collapse_LawLockin",
         "run_lockin_stage",
         lambda: {"active": ACTIVE}),

        ("run_expansion",
         "TQE_10_EI_UNIVERSE_SIMULATION_t_gt_0_expansion",
         "run_expansion_stage",
         lambda: {"active": ACTIVE}),

        ("run_montecarlo",
         "TQE_11_EI_UNIVERSE_SIMULATION_montecarlo",
         "run_montecarlo",
         lambda: {"active": ACTIVE}),

        ("run_best_universe",
         "TQE_12_EI_UNIVERSE_SIMULATION_best_universe",
         "run_best_universe",
         lambda: {"active": ACTIVE}),

        ("run_cmb_map",
         "TQE_13_EI_UNIVERSE_SIMULATION_cmb_map_generation",
         "run_cmb_generation",
         lambda: {"active": ACTIVE}),

        ("run_anomaly_scan",
         "TQE_15_EI_UNIVERSE_SIMULATION_anomaly_cold_spot",
         "run_cold_spot_scan",
         lambda: {"active": ACTIVE}),

        ("run_anomaly_scan",
         "TQE_16_EI_UNIVERSE_SIMULATION_anomaly_low_multipole_alignments",
         "run_low_ell_alignments",
         lambda: {"active": ACTIVE}),

        ("run_anomaly_scan",
         "TQE_17_EI_UNIVERSE_SIMULATION_anomaly_LackOfLargeAngleCorrelation",
         "run_lack_large_angle",
         lambda: {"active": ACTIVE}),

        ("run_anomaly_scan",
         "TQE_18_EI_UNIVERSE_SIMULATION_anomaly_HemisphericalAsymmetry",
         "run_hemi_asymmetry",
         lambda: {"active": ACTIVE}),

        ("run_finetune_diag",
         "TQE_14_EI_UNIVERSE_SIMULATION_finetune_diagnostics",
         "run_finetune_diagnostics",
         lambda: {"active": ACTIVE}),

        ("run_xai",
         "TQE_19_EI_UNIVERSE_SIMULATION_xai",
         "run_xai",
         lambda: {"active": ACTIVE}),

        ("run_manifest",
         "TQE_20_EI_UNIVERSE_SIMULATION_results_manifest",
         "run_results_manifest",
         lambda: {"active_cfg": ACTIVE, "run_dir": run_dir}),
    ]

    # Execute in order, honoring PIPELINE flags
    for flag, module_name, func_name, make_kwargs in STAGES:
        if not ACTIVE["PIPELINE"].get(flag, False):
            print(f"[SKIP] {module_name} ({func_name}) because {flag}=False")
            continue
        safe_call(module_name, func_name, make_kwargs())

    print("=========== TQE PIPELINE END ===========")


if __name__ == "__main__":
    main()
