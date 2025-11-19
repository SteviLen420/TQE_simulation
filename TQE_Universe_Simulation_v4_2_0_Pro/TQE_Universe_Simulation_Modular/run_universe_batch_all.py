#!/usr/bin/env python3
"""
Convenience launcher for the modular TQE Universe pipeline.

- Select run modes: single_eonly, single_ei, batch_ei, batch_all
- Choose number of universes
- Override I-definition (single_ei) and master seed for reproducibility
- Ensures batch runs store outputs under a dedicated mode directory with
  per-run subfolders (E-only + each I-parameter)
"""

import sys
import time
import argparse
from pathlib import Path

MODULAR_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = MODULAR_ROOT.parent  # .../TQE_Universe_Simulation_v4_2_0_Pro
REPO_ROOT = PACKAGE_ROOT.parent     # repository root

# Ensure the package parent is importable
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TQE_Universe_Simulation_v4_2_0_Pro.TQE_Universe_Simulation_Modular import main
from TQE_Universe_Simulation_v4_2_0_Pro.TQE_Universe_Simulation_Modular.config.master_ctrl import MASTER_CTRL

VALID_MODES = ["single_eonly", "single_ei", "batch_ei", "batch_all"]
VALID_I_DEFS = [
    "kl_divergence",
    "shannon",
    "renyi",
    "mutual_info",
    "composite",
    "kl_shannon",
    "entanglement",
    "fisher",
    "fisher_kl_fusion",
    "jensen_shannon",
    "kl_shannon_entanglement",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the TQE Universe Modular pipeline with custom settings."
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default=MASTER_CTRL.get("RUN_MODE", "single_ei"),
        help="Execution mode (default: value from MASTER_CTRL or single_ei)",
    )
    parser.add_argument(
        "--universes",
        type=int,
        default=MASTER_CTRL.get("NUM_UNIVERSES", 300),
        help="Number of universes to simulate (default: value from MASTER_CTRL)",
    )
    parser.add_argument(
        "--i-definition",
        choices=VALID_I_DEFS,
        default=None,
        help="Override I parameter definition (only used when mode=single_ei)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional master seed override (default: use config value or random)",
    )
    return parser.parse_args()


def ensure_results_root() -> Path:
    desktop = Path.home() / "Desktop"
    results_root = desktop / "TQE_Universe_Simulation_Modular_Results"
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def prepare_mode_directory(mode: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_root = ensure_results_root()
    mode_dir = results_root / f"TQE_Universe_Simulation_{mode}_{timestamp}"
    mode_dir.mkdir(parents=True, exist_ok=True)
    return mode_dir


def base_config(args) -> dict:
    cfg = MASTER_CTRL.copy()
    cfg["NUM_UNIVERSES"] = args.universes
    cfg["RUN_MODE"] = args.mode
    cfg.pop("DRIVE_BASE_DIR", None)
    if args.seed is not None:
        cfg["SEED"] = args.seed
    return cfg


def run_single_eonly(args, mode_dir: Path):
    print("=" * 70)
    print("RUN MODE: SINGLE E-ONLY")
    print("=" * 70)
    config = base_config(args)
    config["PIPELINE_VARIANT"] = "energy_only"
    config["MULTI_I_ANALYSIS_MODE"] = True
    config["MULTI_I_SAVE_DIR"] = str(mode_dir)
    run_id = f"Eonly_{time.strftime('%Y%m%d_%H%M%S')}"
    main.run_pipeline(config_override=config, run_id_override=run_id)


def run_single_ei(args, mode_dir: Path):
    print("=" * 70)
    print("RUN MODE: SINGLE E+I")
    print("=" * 70)
    config = base_config(args)
    config["PIPELINE_VARIANT"] = "full"
    selected_i = args.i_definition or config.get("I_DEFINITION_MODE", "kl_shannon")
    if selected_i not in VALID_I_DEFS:
        raise ValueError(
            f"Invalid I-definition '{selected_i}'. Choose from: {', '.join(VALID_I_DEFS)}"
        )
    config["I_DEFINITION_MODE"] = selected_i
    config["MULTI_I_ANALYSIS_MODE"] = True
    config["MULTI_I_SAVE_DIR"] = str(mode_dir)
    run_id = f"EplusI_{selected_i}_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"I-definition: {selected_i}")
    main.run_pipeline(config_override=config, run_id_override=run_id)


def run_batch_ei(args, mode_dir: Path):
    print("=" * 70)
    print("RUN MODE: BATCH E+I (11 definitions)")
    print("=" * 70)
    successes = 0
    failures = 0
    for idx, i_def in enumerate(VALID_I_DEFS, start=1):
        label = f"EplusI_{i_def}_{time.strftime('%Y%m%d_%H%M%S')}_{idx:02d}"
        print(f"\n{'─'*70}\nE+I Run {idx}/11: {i_def}\n{'─'*70}")
        config = base_config(args)
        config["PIPELINE_VARIANT"] = "full"
        config["I_DEFINITION_MODE"] = i_def
        config["MULTI_I_ANALYSIS_MODE"] = True
        config["MULTI_I_SAVE_DIR"] = str(mode_dir)
        try:
            main.run_pipeline(config_override=config, run_id_override=label)
            successes += 1
        except Exception as exc:
            failures += 1
            print(f"❌ ERROR in '{i_def}': {exc}")
    print(f"\n{'='*70}")
    print(f"Batch E+I completed: {successes}/11 successful, {failures} failed")
    print(f"Results saved under: {mode_dir}")
    print(f"{'='*70}")


def run_batch_all(args, mode_dir: Path):
    print("=" * 70)
    print("RUN MODE: BATCH ALL (E-only + 11 I-definitions)")
    print("=" * 70)
    successes = 0
    failures = 0

    # E-only baseline
    print(f"{'─'*70}\nE-only Run (1/12)\n{'─'*70}")
    run_id_eonly = f"Eonly_{time.strftime('%Y%m%d_%H%M%S')}"
    config_eonly = base_config(args)
    config_eonly["PIPELINE_VARIANT"] = "energy_only"
    config_eonly["MULTI_I_ANALYSIS_MODE"] = True
    config_eonly["MULTI_I_SAVE_DIR"] = str(mode_dir)
    try:
        main.run_pipeline(config_override=config_eonly, run_id_override=run_id_eonly)
        successes += 1
    except Exception as exc:
        failures += 1
        print(f"❌ ERROR in E-only: {exc}")

    # E+I runs
    for idx, i_def in enumerate(VALID_I_DEFS, start=2):
        print(f"\n{'─'*70}\nE+I Run {idx}/12: {i_def}\n{'─'*70}")
        run_id = f"EplusI_{i_def}_{time.strftime('%Y%m%d_%H%M%S')}_{idx:02d}"
        config = base_config(args)
        config["PIPELINE_VARIANT"] = "full"
        config["I_DEFINITION_MODE"] = i_def
        config["MULTI_I_ANALYSIS_MODE"] = True
        config["MULTI_I_SAVE_DIR"] = str(mode_dir)
        try:
            main.run_pipeline(config_override=config, run_id_override=run_id)
            successes += 1
        except Exception as exc:
            failures += 1
            print(f"❌ ERROR in '{i_def}': {exc}")

    print(f"\n{'='*70}")
    print(f"BATCH ALL COMPLETED: {successes}/12 successful, {failures} failed")
    print(f"Results saved under: {mode_dir}")
    print(f"{'='*70}")


def main_entry():
    args = parse_args()
    if args.mode != "single_ei" and args.i_definition:
        print("[WARN] --i-definition is only used in single_ei mode. Ignoring override.")

    mode_dir = prepare_mode_directory(args.mode)
    print("=============================================")
    print(" TQE Universe Modular Pipeline Launcher")
    print("=============================================")
    print(f" Run mode:   {args.mode}")
    print(f" Universes:  {args.universes}")
    if args.seed is not None:
        print(f" Master seed: {args.seed}")
    print(f" Output base: {mode_dir}")
    print("=============================================")

    if args.mode == "single_eonly":
        run_single_eonly(args, mode_dir)
    elif args.mode == "single_ei":
        run_single_ei(args, mode_dir)
    elif args.mode == "batch_ei":
        run_batch_ei(args, mode_dir)
    elif args.mode == "batch_all":
        run_batch_all(args, mode_dir)
    else:
        raise ValueError(f"Unsupported run mode: {args.mode}")


if __name__ == "__main__":
    main_entry()

