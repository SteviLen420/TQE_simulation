# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# TQE_Analysis_Pipeline_v4.2.0_PRO.py
# ==========================================================================================
# TQE Comparative Analysis: Batch Simulation Results Analysis and Model Selection
# Based on the Theory of the Question of Existence (TQE)
# ==========================================================================================
#
# AUTHOR: Stefan Len
# DATE: 2025
# VERSION: v4.2.0 PRO
#
# ==========================================================================================
# PURPOSE: COMPARATIVE ANALYSIS OF TQE UNIVERSE SIMULATIONS
# ==========================================================================================
#
# This pipeline provides comprehensive comparative analysis for batch TQE simulations,
# identifying the best-performing I-parameter definition across 12 analysis categories.
#
# SUMMARY:
#   • Automatically discovers the latest batch (`batch_all` or `batch_ei`) and harvests every
#     CSV/JSON artifact emitted by the main simulation (Planck, life, entropy, anomalies, etc.).
#   • Builds an extended metrics table (≈80 columns) and renders 12+ comparative modules,
#     triple model rankings, and advanced visualization sets.
#   • Designed for both local and Colab execution; no extra dependencies beyond the main
#     pipeline requirements.
#
# ==========================================================================================
# For detailed usage instructions and interpretation guide, see README.md
# ==========================================================================================

# ======== CRITICAL: PACKAGE INSTALLATION FIRST ========
import sys
import subprocess

def _ensure(pkg):
    """Ensure a package is installed before importing."""
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ⚡ INSTALL PACKAGES BEFORE ANY IMPORTS ⚡
print("[SETUP] Checking and installing dependencies...")
essential_packages = ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "tqdm"]

for pkg in essential_packages:
    _ensure(pkg)

print("[SETUP] All dependencies ready!")

# ======== NOW SAFE TO IMPORT ========
import os
import time
import json
import glob
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from tqdm import tqdm

# Configure matplotlib for proper figure generation
import matplotlib
matplotlib.use('Agg')
plt.ioff()

warnings.filterwarnings('ignore')

# ==========================================================================================
# MASTER CONTROL PANEL
# ==========================================================================================
# Configure analysis target, ranking weights, visualization settings, and statistical options.
# Modify these parameters to customize the analysis behavior.
# ==========================================================================================
MASTER_CTRL = {
    
    # === TARGET SELECTION ===
    "TARGET_MODE": "batch_all",
    
    # TARGET_TIMESTAMP: Specify which simulation batch to analyze
    #   - None: Auto-detect latest batch (recommended for most recent results)
    #   - "YYYYMMDD_HHMMSS": Specific timestamp (e.g., "20251105_070107")
    #   
    #   Current: "20251105_070107" - batch_all with 10 E+I runs including jensen_shannon
    #   This batch contains all I-definitions for comprehensive comparison.
    "TARGET_TIMESTAMP": "20251105_065930",
    
    # === PATH CONFIGURATION ===
    # AUTO-DETECT: These are set automatically, override only if needed
    "SIMULATION_ROOT": None,          # Auto-detected (Colab or local)
    "ANALYSIS_OUTPUT_ROOT": None,     # Auto-detected (Colab or local)
    
    # === RANKING WEIGHTS (Model Selection) ===
    # TWO RANKING SYSTEMS:
    
    # RANKING SYSTEM 1: STABILITY-FOCUSED (Traditional)
    "RANKING_WEIGHTS_STABILITY": {
        "stability_rate": 0.30,           # 30% - Stable universe %
        "lockin_rate": 0.20,              # 20% - Law emergence frequency
        "planck_chi2_fit": 0.20,          # 20% - Observational validation
        "goldilocks_precision": 0.15,     # 15% - Peak uncertainty
        "cmb_anomaly_match": 0.10,        # 10% - Anomaly detection
        "bayesian_efficiency": 0.05,      # 5% - GP performance
    },
    
    # RANKING SYSTEM 2: COMPLEXITY-FOCUSED (TQE-Consistent)
    "RANKING_WEIGHTS_COMPLEXITY": {
        "complexity_score": 0.35,         # 35% - Structural complexity
        "life_compatibility": 0.25,       # 25% - Life-readiness indicators
        "information_richness": 0.20,     # 20% - I-parameter effectiveness
        "stability_quality": 0.10,        # 10% - Stability (not quantity!)
        "observational_match": 0.10,      # 10% - Planck fit
    },
    
    # RANKING SYSTEM 3: PHYSICAL-LAWS-FOCUSED
    "RANKING_WEIGHTS_PHYSICAL_LAWS": {
        "emergent_laws_quality": 0.30,    # 30% - Power-law, phase transitions
        "friedmann_consistency": 0.25,    # 25% - age, H0, Omegas vs Planck 2018
        "cmb_anomaly_match": 0.20,        # 20% - Cold spots, Axis of Evil
        "lockin_efficiency": 0.15,        # 15% - Fast, decisive law formation
        "quantum_field_realism": 0.10,    # 10% - Vacuum energy, fluctuations
    },
    
    # Active ranking system
    "RANKING_MODE": "triple",  # "stability" | "complexity" | "physical_laws" | "triple"
    
    # === VISUALIZATION SETTINGS ===
    "FIGURE_DPI": 300,                    # Plot resolution (300 for publication quality)
    "FIGURE_FORMAT": "png",               # Output format
    "PLOT_STYLE": "seaborn-v0_8-darkgrid",  # Matplotlib style
    "COLOR_PALETTE": "husl",              # Seaborn color palette
    "FONT_SIZE_TITLE": 14,                # Title font size
    "FONT_SIZE_LABEL": 12,                # Axis label font size
    "FONT_SIZE_TICK": 10,                 # Tick label font size
    
    # === ANALYSIS OPTIONS ===
    "INCLUDE_BAYESIAN_DATA": True,        # Include Bayesian calibration CSV analysis
    "GENERATE_RADAR_CHART": True,         # Generate spider plot
    "GENERATE_HEATMAP": True,             # Generate performance heatmap
    "GENERATE_CORRELATION_MATRIX": True,  # Generate correlation analysis
    "TOP_N_MODELS": 3,                    # Number of top models to highlight
    "VERBOSE": True,                      # Print detailed progress messages
    
    # === STATISTICAL SETTINGS ===
    "CONFIDENCE_LEVEL": 0.95,             # Confidence level for uncertainty bands
    "OUTLIER_THRESHOLD": 3.0,             # Z-score threshold for outlier detection
    "MIN_RUNS_FOR_CORRELATION": 3,        # Minimum runs required for correlation matrix
    
    # === PLANCK TARGET REFERENCES ===
    "PLANCK_TARGET_E": float(os.environ.get("PLANCK_TARGET_E", 0.7619)),
    "PLANCK_TARGET_I": float(os.environ.get("PLANCK_TARGET_I", 0.1309)),
}

# ==========================================================================================
# AUTO-DETECTION: ENVIRONMENT
# ==========================================================================================

# Detect if running in IPython/Jupyter/Colab
try:
    get_ipython()
    IN_COLAB = True
except NameError:
    IN_COLAB = False

# ==========================================================================================
# AUTO-DETECTION: PATHS (function to be called after Drive mount)
# ==========================================================================================

def setup_paths():
    """
    Setup SIMULATION_ROOT and ANALYSIS_OUTPUT_ROOT based on environment.
    This should be called AFTER Drive is mounted in Colab.
    """
    global SIMULATION_ROOT, ANALYSIS_OUTPUT_ROOT
    
    if IN_COLAB:
        # Google Colab or Jupyter environment
        if os.path.exists("/content/drive/MyDrive"):
            # Colab with mounted Drive
            SIMULATION_ROOT = "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO"
            ANALYSIS_OUTPUT_ROOT = "/content/drive/MyDrive/TQE_Analysis_Pipeline_v4.2.0_PRO/analysis_results"
        else:
            # Colab without mounted drive or local Jupyter - use current directory
            SIMULATION_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO"))
            ANALYSIS_OUTPUT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "analysis_results"))
    else:
        # Running as script (not in IPython/Jupyter)
        SIMULATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO"))
        ANALYSIS_OUTPUT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "analysis_results"))
    
    # Override with MASTER_CTRL if specified
    if MASTER_CTRL.get("SIMULATION_ROOT") is not None:
        SIMULATION_ROOT = MASTER_CTRL["SIMULATION_ROOT"]
    if MASTER_CTRL.get("ANALYSIS_OUTPUT_ROOT") is not None:
        ANALYSIS_OUTPUT_ROOT = MASTER_CTRL["ANALYSIS_OUTPUT_ROOT"]
    
    return SIMULATION_ROOT, ANALYSIS_OUTPUT_ROOT

# Initialize paths (will be properly set after Drive mount in main)
SIMULATION_ROOT = None
ANALYSIS_OUTPUT_ROOT = None

# Extract config values
TARGET_MODE = MASTER_CTRL["TARGET_MODE"]
TARGET_TIMESTAMP = MASTER_CTRL.get("TARGET_TIMESTAMP", None)
RANKING_MODE = MASTER_CTRL.get("RANKING_MODE", "both")
RANKING_WEIGHTS_STABILITY = MASTER_CTRL["RANKING_WEIGHTS_STABILITY"]
RANKING_WEIGHTS_COMPLEXITY = MASTER_CTRL["RANKING_WEIGHTS_COMPLEXITY"]
RANKING_WEIGHTS_PHYSICAL_LAWS = MASTER_CTRL["RANKING_WEIGHTS_PHYSICAL_LAWS"]
FIGURE_DPI = MASTER_CTRL["FIGURE_DPI"]
FIGURE_FORMAT = MASTER_CTRL["FIGURE_FORMAT"]
PLANCK_TARGET_E = MASTER_CTRL["PLANCK_TARGET_E"]
PLANCK_TARGET_I = MASTER_CTRL["PLANCK_TARGET_I"]

# Visualization settings
try:
    plt.style.use(MASTER_CTRL["PLOT_STYLE"])
except:
    plt.style.use('default')
sns.set_palette(MASTER_CTRL["COLOR_PALETTE"])

# ==========================================================================================
# PHASE 1: DATA COLLECTION & VALIDATION
# ==========================================================================================
# Purpose: Automatically discover and load all simulation results from target mode folder
# Input: TARGET_MODE specifies which batch folder to analyze
# Output: Comprehensive data dictionary with all runs loaded
# ==========================================================================================

def smart_find_file(base_dir: str, filename_patterns: List[str], recursive: bool = True) -> Optional[str]:
    """
    Universal smart file finder - searches for files using multiple patterns.
    
    Args:
        base_dir: Directory to search in
        filename_patterns: List of filename patterns to try (e.g., ["summary*.json", "*.json"])
        recursive: If True, searches subdirectories recursively
    
    Returns:
        Path to first matching file, or None if not found
    """
    for pattern in filename_patterns:
        if recursive:
            # Recursive search
            matches = glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
        else:
            # Direct search only
            matches = glob.glob(os.path.join(base_dir, pattern))
        
        if matches:
            # Return first match, sorted by modification time (newest first)
            matches_sorted = sorted(matches, key=os.path.getmtime, reverse=True)
            return matches_sorted[0]
    
    return None


def find_latest_mode_directory(simulation_root: str, mode: str, timestamp: str = None) -> Optional[str]:
    """
    Intelligently find the latest (or specific) timestamped mode directory.
    Tries multiple naming patterns to maximize compatibility.
    
    Args:
        simulation_root: Base simulation directory
        mode: "batch_ei" or "batch_all"
        timestamp: Specific timestamp or None for latest
    
    Returns:
        Full path to mode directory or None if not found
    """
    # Try multiple search patterns for maximum flexibility
    search_patterns = [
        os.path.join(simulation_root, f"TQE_Universe_Simulation_{mode}_*"),
        os.path.join(simulation_root, f"*{mode}_*"),
        os.path.join(simulation_root, f"*{mode.replace('_', '')}*"),  # batch_all -> batchall
        os.path.join(simulation_root, f"*batch*{mode.split('_')[-1]}*"),  # Extract last part
    ]
    
    all_matching = []
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        # Filter to keep only directories
        matches = [m for m in matches if os.path.isdir(m)]
        all_matching.extend(matches)
    
    # Remove duplicates and sort
    matching_dirs = sorted(list(set(all_matching)))
    
    if not matching_dirs:
        print(f"❌ ERROR: No directories found for mode '{mode}' in {simulation_root}")
        print(f"   Tried patterns: {[os.path.basename(p) for p in search_patterns]}")
        return None
    
    if timestamp:
        # Find specific timestamp (convert to string if needed)
        timestamp_str = str(timestamp)
        for d in matching_dirs:
            if timestamp_str in d:
                print(f"[FOUND] Using specified directory: {os.path.basename(d)}")
                return d
        print(f"❌ ERROR: No directory found with timestamp: {timestamp}")
        print(f"   Available: {[os.path.basename(d) for d in matching_dirs]}")
        return None
    else:
        # Return most recent (sorted alphabetically = chronologically)
        latest = matching_dirs[-1]
        print(f"[AUTO-DETECT] Using latest directory: {os.path.basename(latest)}")
        return latest


def validate_target_mode(target_mode: str) -> bool:
    """Validate that target mode is a batch mode."""
    if "batch_ei" not in target_mode and "batch_all" not in target_mode:
        print(f"❌ ERROR: Target mode must be 'batch_ei' or 'batch_all'")
        print(f"   Received: {target_mode}")
        print(f"   Single modes (single_eonly, single_ei) are not supported for comparative analysis.")
        return False
    return True


def detect_eonly_presence(target_path: str) -> bool:
    """Detect if E-only run is present (batch_all mode)."""
    eonly_dirs = glob.glob(os.path.join(target_path, "Eonly_*"))
    return len(eonly_dirs) > 0


def collect_run_directories(target_path: str) -> Dict[str, List[str]]:
    """
    Collect all run directories from target mode folder.
    
    Returns:
        {
            "eonly": ["Eonly_20251030_223510"],
            "ei": ["EplusI_kl_divergence_...", "EplusI_shannon_...", ...]
        }
    """
    eonly_dirs = sorted(glob.glob(os.path.join(target_path, "Eonly_*")))
    ei_dirs = sorted(glob.glob(os.path.join(target_path, "EplusI_*")))
    
    return {
        "eonly": [os.path.basename(d) for d in eonly_dirs],
        "ei": [os.path.basename(d) for d in ei_dirs]
    }


def load_summary_json(run_dir: str) -> Optional[Dict]:
    """
    Intelligently load summary JSON from a run directory.
    Searches multiple locations and filename patterns automatically.
    """
    # Use smart file finder with multiple patterns
    summary_file = smart_find_file(
        run_dir, 
        filename_patterns=["summary_full.json", "summary.json", "*summary*.json"],
        recursive=True
    )
    
    if summary_file:
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {summary_file}: {e}")
            return None
    
    # Nothing found
    print(f"⚠️  WARNING: No summary JSON found in {os.path.basename(run_dir)}")
    return None


def load_tqe_runs_csv(run_dir: str) -> Optional[pd.DataFrame]:
    """
    Intelligently load TQE runs CSV from a run directory.
    Searches multiple locations and filename patterns automatically.
    """
    # Use smart file finder with multiple patterns
    runs_file = smart_find_file(
        run_dir,
        filename_patterns=["tqe_runs.csv", "runs.csv", "*run*.csv"],
        recursive=True
    )
    
    if runs_file:
        # Filter out bayesian/calibration files
        if "bayesian" in runs_file.lower() or "calibration" in runs_file.lower():
            # Try to find another file
            all_run_files = glob.glob(os.path.join(run_dir, "**", "*run*.csv"), recursive=True)
            filtered = [f for f in all_run_files 
                       if "bayesian" not in f.lower() and "calibration" not in f.lower()]
            if filtered:
                runs_file = sorted(filtered, key=os.path.getmtime, reverse=True)[0]
            else:
                return None  # No valid runs file found
        
        try:
            return pd.read_csv(runs_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {runs_file}: {e}")
    
    # Not found (this is OK - not all runs have trajectory data)
    return None


def load_bayesian_calibration_csv(run_dir: str) -> Optional[pd.DataFrame]:
    """
    Intelligently load Bayesian calibration CSV from a run directory.
    Searches multiple locations and filename patterns automatically.
    """
    # Use smart file finder with multiple patterns
    bayesian_file = smart_find_file(
        run_dir,
        filename_patterns=[
            "bayesian_calibration_*.csv",
            "bayesian*.csv", 
            "calibration*.csv",
            "*bayesian*.csv"
        ],
        recursive=True
    )
    
    if bayesian_file:
        try:
            return pd.read_csv(bayesian_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {bayesian_file}: {e}")
    
    # Not found (this is OK - not all runs have Bayesian data)
    return None


def load_emergent_law_summary(run_dir: str) -> Optional[pd.DataFrame]:
    """Load emergent law summary CSV."""
    law_file = smart_find_file(
        run_dir,
        filename_patterns=["emergent_law_summary.csv", "*emergent_law*.csv"],
        recursive=True
    )
    if law_file:
        try:
            return pd.read_csv(law_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {law_file}: {e}")
    return None


def load_parameter_sensitivity(run_dir: str) -> Optional[pd.DataFrame]:
    """Load parameter sensitivity analysis CSV."""
    sens_file = smart_find_file(
        run_dir,
        filename_patterns=["parameter_sensitivity_analysis.csv", "*sensitivity*.csv"],
        recursive=True
    )
    if sens_file:
        try:
            return pd.read_csv(sens_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {sens_file}: {e}")
    return None


def load_cmb_coldspots(run_dir: str, i_def: str) -> Optional[pd.DataFrame]:
    """Load CMB cold spots summary CSV."""
    coldspot_file = smart_find_file(
        run_dir,
        filename_patterns=[f"cmb_coldspots_summary_{i_def}.csv", "*coldspots*.csv"],
        recursive=True
    )
    if coldspot_file:
        try:
            return pd.read_csv(coldspot_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {coldspot_file}: {e}")
    return None


def load_cmb_aoe(run_dir: str, i_def: str) -> Optional[pd.DataFrame]:
    """Load CMB Axis of Evil summary CSV."""
    aoe_file = smart_find_file(
        run_dir,
        filename_patterns=[f"cmb_aoe_summary_{i_def}.csv", "*aoe*.csv"],
        recursive=True
    )
    if aoe_file:
        try:
            return pd.read_csv(aoe_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {aoe_file}: {e}")
    return None


def load_i_definitions_comparison(run_dir: str) -> Optional[pd.DataFrame]:
    """Load I-definitions comparison CSV."""
    i_comp_file = smart_find_file(
        run_dir,
        filename_patterns=["I_Definitions_Comparison.csv", "*I_Definitions*.csv"],
        recursive=True
    )
    if i_comp_file:
        try:
            return pd.read_csv(i_comp_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {i_comp_file}: {e}")
    return None


def load_life_compatibility_summary(run_dir: str) -> Optional[Dict]:
    """Load life_compatibility_summary.json if present."""
    life_file = smart_find_file(
        run_dir,
        filename_patterns=["life_compatibility_summary.json", "*life_compatibility*.json"],
        recursive=True
    )
    if life_file:
        try:
            with open(life_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {life_file}: {e}")
    return None


def load_planck_validation(run_dir: str) -> Optional[Dict[str, Optional[object]]]:
    """Load Planck validation artifacts (best-fit summary JSON + validation CSV)."""
    summary_file = smart_find_file(
        run_dir,
        filename_patterns=["planck_best_fit_summary.json", "*planck_best_fit*.json"],
        recursive=True
    )
    csv_file = smart_find_file(
        run_dir,
        filename_patterns=["planck_validation*.csv", "*planck*validation*.csv"],
        recursive=True
    )
    
    result: Dict[str, Optional[object]] = {"summary": None, "validation": None}
    if summary_file:
        try:
            with open(summary_file, "r") as f:
                result["summary"] = json.load(f)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {summary_file}: {e}")
    if csv_file:
        try:
            result["validation"] = pd.read_csv(csv_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {csv_file}: {e}")
    
    # Check if any value is not None (handle DataFrame objects properly)
    has_data = False
    for value in result.values():
        if value is not None:
            # For DataFrame, check if it's not empty
            if isinstance(value, pd.DataFrame):
                if not value.empty:
                    has_data = True
                    break
            else:
                has_data = True
                break
    
    return result if has_data else None


def load_entropy_volatility_summary(run_dir: str) -> Optional[pd.DataFrame]:
    """Load entropy volatility CSV (aggregated)."""
    entropy_file = smart_find_file(
        run_dir,
        filename_patterns=["entropy_volatility_summary*.csv", "*entropy_volatility*.csv"],
        recursive=True
    )
    if entropy_file:
        try:
            return pd.read_csv(entropy_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {entropy_file}: {e}")
    return None


def load_stability_sweep(run_dir: str, variant: str) -> Optional[pd.DataFrame]:
    """Load stability sweep CSVs (variant = eps_sweep or zero)."""
    pattern = f"stability_by_I_{variant}*.csv"
    sweep_file = smart_find_file(
        run_dir,
        filename_patterns=[pattern],
        recursive=True
    )
    if sweep_file:
        try:
            return pd.read_csv(sweep_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {sweep_file}: {e}")
    return None


def load_advanced_anomaly_results(run_dir: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Load advanced anomaly detection CSVs (cold spots, Axis of Evil, physical anomalies)."""
    advanced_files = {
        "advanced_anomalies": smart_find_file(
            run_dir,
            filename_patterns=["advanced_anomaly_detection_results*.csv", "*advanced_anomaly*.csv"],
            recursive=True
        ),
        "physical_anomalies": smart_find_file(
            run_dir,
            filename_patterns=["advanced_physics_physical_anomalies*.csv", "*physical_anomalies*.csv"],
            recursive=True
        ),
        "cmb_gaussianity": smart_find_file(
            run_dir,
            filename_patterns=["cmb_gaussianity_check*.csv"],
            recursive=True
        ),
        "cmb_isotropy": smart_find_file(
            run_dir,
            filename_patterns=["cmb_isotropy_check*.csv"],
            recursive=True
        ),
    }
    
    results: Dict[str, Optional[pd.DataFrame]] = {}
    for key, file_path in advanced_files.items():
        if file_path:
            try:
                results[key] = pd.read_csv(file_path)
            except Exception as e:
                print(f"⚠️  WARNING: Could not parse {file_path}: {e}")
                results[key] = None
        else:
            results[key] = None
    return results if any(val is not None for val in results.values()) else None


def load_nested_sampling_samples(run_dir: str) -> Optional[pd.DataFrame]:
    """Load nested_sampling_samples CSV if available."""
    ns_file = smart_find_file(
        run_dir,
        filename_patterns=["nested_sampling_samples*.csv", "*nested_sampling*.csv"],
        recursive=True
    )
    if ns_file:
        try:
            return pd.read_csv(ns_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {ns_file}: {e}")
    return None


def load_pre_fluctuation_pairs(run_dir: str) -> Optional[pd.DataFrame]:
    """Load pre-fluctuation pair CSVs."""
    pre_file = smart_find_file(
        run_dir,
        filename_patterns=["pre_fluctuation_pairs*.csv", "*pre_fluctuation*.csv"],
        recursive=True
    )
    if pre_file:
        try:
            return pd.read_csv(pre_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {pre_file}: {e}")
    return None


def load_universe_seeds(run_dir: str) -> Optional[pd.DataFrame]:
    """Load universe_seeds CSV if present."""
    seeds_file = smart_find_file(
        run_dir,
        filename_patterns=["universe_seeds*.csv", "*seeds*.csv"],
        recursive=True
    )
    if seeds_file:
        try:
            return pd.read_csv(seeds_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {seeds_file}: {e}")
    return None


def extract_i_definition(dirname: str) -> str:
    """Extract I-definition name from directory name."""
    if dirname.startswith("Eonly_"):
        return "energy_only"
    elif dirname.startswith("EplusI_"):
        # EplusI_kl_divergence_20251030_223511 → kl_divergence
        parts = dirname.split("_")
        # Find the timestamp part (8 digits)
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 8:
                # Everything before timestamp is the I-definition
                return "_".join(parts[1:i])
        return "unknown"
    return "unknown"


def collect_simulation_data(target_mode: str) -> Dict:
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
    
    # Find the target directory (latest or specific timestamp)
    target_path = find_latest_mode_directory(SIMULATION_ROOT, target_mode, TARGET_TIMESTAMP)
    
    if target_path is None:
        sys.exit(1)
    
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
            
            #  Load extended analysis data
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
        
        #  Load extended analysis data
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
    
    #  Extended data statistics
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


# ==========================================================================================
# PHASE 2: COMPREHENSIVE METRIC EXTRACTION & DATAFRAME CONSTRUCTION (EXTENDED)
# ==========================================================================================
# Purpose: Extract all metrics from loaded simulation data (basic + extended)
# Input: Collected data dictionary from Phase 1 (summary + extended CSVs)
# Output: Comprehensive metrics DataFrame with 50-80 columns
# ==========================================================================================

def extract_extended_metrics(data: Dict, i_def: str) -> Dict:
    """
    Extract EXTENDED metrics from additional CSV files.
    
    Extracts from:
    - tqe_runs.csv: Friedmann, quantum fields, entanglement, entropy
    - emergent_law_summary.csv: Power-law, phase transitions
    - parameter_sensitivity.csv: E/I/X sensitivity
    - cmb_coldspots.csv: Cold spot metrics
    - cmb_aoe.csv: Axis of Evil metrics
    - i_definitions_comparison.csv: I(E) curves
    
    Returns:
        dict: Extended metrics (30-50 additional fields)
    """
    extended = {}
    
    # EMERGENT LAWS METRICS
    if data.get("emergent_laws") is not None:
        law_df = data["emergent_laws"]
        if len(law_df) > 0:
            extended["power_law_exponent"] = law_df.get("power_law_exponent", pd.Series([np.nan])).mean()
            extended["correlation_strength"] = law_df.get("correlation_strength", pd.Series([np.nan])).mean()
            extended["phase_transition_detected"] = law_df.get("phase_transition_detected", pd.Series([0])).sum() > 0
            extended["n_phase_transitions"] = law_df.get("phase_transition_detected", pd.Series([0])).sum()
        else:
            extended.update({"power_law_exponent": np.nan, "correlation_strength": np.nan, 
                           "phase_transition_detected": False, "n_phase_transitions": 0})
    else:
        extended.update({"power_law_exponent": np.nan, "correlation_strength": np.nan, 
                       "phase_transition_detected": False, "n_phase_transitions": 0})
    
    # FRIEDMANN COSMOLOGY METRICS (from tqe_runs.csv)
    if data.get("tqe_runs") is not None:
        runs_df = data["tqe_runs"]
        if "age_Gyr" in runs_df.columns:
            extended["age_Gyr_mean"] = runs_df["age_Gyr"].mean()
            extended["age_Gyr_std"] = runs_df["age_Gyr"].std()
            extended["age_deviation_from_planck"] = abs(runs_df["age_Gyr"].mean() - 13.8)  # Planck: 13.8 Gyr
        else:
            extended.update({"age_Gyr_mean": np.nan, "age_Gyr_std": np.nan, "age_deviation_from_planck": np.nan})
        
        if "H_today" in runs_df.columns:
            extended["H0_mean"] = runs_df["H_today"].mean()
            extended["H0_std"] = runs_df["H_today"].std()
            extended["H0_deviation_from_planck"] = abs(runs_df["H_today"].mean() - 67.4)  # Planck: 67.4 km/s/Mpc
        else:
            extended.update({"H0_mean": np.nan, "H0_std": np.nan, "H0_deviation_from_planck": np.nan})
        
        # Omega parameters
        for omega in ["Omega_m", "Omega_b", "Omega_k"]:
            if omega in runs_df.columns:
                extended[f"{omega}_mean"] = runs_df[omega].mean()
                extended[f"{omega}_std"] = runs_df[omega].std()
            else:
                extended.update({f"{omega}_mean": np.nan, f"{omega}_std": np.nan})
        
        # LOCK-IN DYNAMICS
        if "lock_epoch" in runs_df.columns and "stable_epoch" in runs_df.columns:
            lockin_data = runs_df[runs_df["lockin"] == 1]
            if len(lockin_data) > 0:
                extended["lock_epoch_mean"] = lockin_data["lock_epoch"].mean()
                extended["lock_epoch_std"] = lockin_data["lock_epoch"].std()
                extended["lockin_efficiency"] = (lockin_data["lock_epoch"] - lockin_data["stable_epoch"]).mean()
                extended["early_lockin_rate"] = (lockin_data["lock_epoch"] < 100).sum() / len(lockin_data) * 100
            else:
                extended.update({"lock_epoch_mean": np.nan, "lock_epoch_std": np.nan, 
                               "lockin_efficiency": np.nan, "early_lockin_rate": 0.0})
        else:
            extended.update({"lock_epoch_mean": np.nan, "lock_epoch_std": np.nan, 
                           "lockin_efficiency": np.nan, "early_lockin_rate": 0.0})
        
        # QUANTUM FIELDS
        if "vacuum_energy_density" in runs_df.columns:
            extended["vacuum_energy_mean"] = runs_df["vacuum_energy_density"].mean()
            extended["vacuum_energy_std"] = runs_df["vacuum_energy_density"].std()
        else:
            extended.update({"vacuum_energy_mean": np.nan, "vacuum_energy_std": np.nan})
        
        if "zero_point_energy" in runs_df.columns:
            extended["zero_point_energy_mean"] = runs_df["zero_point_energy"].mean()
        else:
            extended["zero_point_energy_mean"] = np.nan
        
        if "quantum_fluctuation_amplitude" in runs_df.columns:
            extended["quantum_fluctuation_mean"] = runs_df["quantum_fluctuation_amplitude"].mean()
        else:
            extended["quantum_fluctuation_mean"] = np.nan
        
        # ENTANGLEMENT
        if "entanglement_entropy" in runs_df.columns:
            extended["entanglement_entropy_mean"] = runs_df["entanglement_entropy"].mean()
            extended["entanglement_entropy_std"] = runs_df["entanglement_entropy"].std()
        else:
            extended.update({"entanglement_entropy_mean": np.nan, "entanglement_entropy_std": np.nan})
        
        if "holographic_entropy" in runs_df.columns:
            extended["holographic_entropy_mean"] = runs_df["holographic_entropy"].mean()
        else:
            extended["holographic_entropy_mean"] = np.nan
        
        # ENTROPY & INFORMATION
        if "entropy_volatility" in runs_df.columns:
            extended["entropy_volatility_mean"] = runs_df["entropy_volatility"].mean()
            extended["entropy_volatility_std"] = runs_df["entropy_volatility"].std()
        else:
            extended.update({"entropy_volatility_mean": np.nan, "entropy_volatility_std": np.nan})
        
        # TOPOLOGY
        if "curvature_radius" in runs_df.columns:
            extended["curvature_radius_mean"] = runs_df["curvature_radius"].mean()
        else:
            extended["curvature_radius_mean"] = np.nan
        
        if "topological_defects" in runs_df.columns:
            extended["topological_defect_rate"] = (runs_df["topological_defects"] > 0).sum() / len(runs_df) * 100
        else:
            extended["topological_defect_rate"] = 0.0
    else:
        # No tqe_runs.csv - fill with NaN
        extended.update({
            "age_Gyr_mean": np.nan, "age_Gyr_std": np.nan, "age_deviation_from_planck": np.nan,
            "H0_mean": np.nan, "H0_std": np.nan, "H0_deviation_from_planck": np.nan,
            "Omega_m_mean": np.nan, "Omega_m_std": np.nan,
            "Omega_b_mean": np.nan, "Omega_b_std": np.nan,
            "Omega_k_mean": np.nan, "Omega_k_std": np.nan,
            "lock_epoch_mean": np.nan, "lock_epoch_std": np.nan, "lockin_efficiency": np.nan, "early_lockin_rate": 0.0,
            "vacuum_energy_mean": np.nan, "vacuum_energy_std": np.nan,
            "zero_point_energy_mean": np.nan, "quantum_fluctuation_mean": np.nan,
            "entanglement_entropy_mean": np.nan, "entanglement_entropy_std": np.nan, "holographic_entropy_mean": np.nan,
            "entropy_volatility_mean": np.nan, "entropy_volatility_std": np.nan,
            "curvature_radius_mean": np.nan, "topological_defect_rate": 0.0
        })
    
    # PARAMETER SENSITIVITY
    if data.get("parameter_sensitivity") is not None:
        sens_df = data["parameter_sensitivity"]
        if len(sens_df) > 0:
            extended["E_sensitivity"] = sens_df[sens_df["parameter"] == "E"]["sensitivity"].values[0] if "E" in sens_df["parameter"].values else np.nan
            extended["I_sensitivity"] = sens_df[sens_df["parameter"] == "I"]["sensitivity"].values[0] if "I" in sens_df["parameter"].values else np.nan
            extended["X_sensitivity"] = sens_df[sens_df["parameter"] == "X"]["sensitivity"].values[0] if "X" in sens_df["parameter"].values else np.nan
        else:
            extended.update({"E_sensitivity": np.nan, "I_sensitivity": np.nan, "X_sensitivity": np.nan})
    else:
        extended.update({"E_sensitivity": np.nan, "I_sensitivity": np.nan, "X_sensitivity": np.nan})
    
    # CMB ANOMALIES
    if data.get("cmb_coldspots") is not None:
        coldspots_df = data["cmb_coldspots"]
        if len(coldspots_df) > 0:
            extended["n_coldspots_mean"] = coldspots_df.get("n_coldspots", pd.Series([0])).mean()
            extended["coldspot_depth_mean"] = coldspots_df.get("coldspot_depth_avg", pd.Series([np.nan])).mean()
        else:
            extended.update({"n_coldspots_mean": 0.0, "coldspot_depth_mean": np.nan})
    else:
        extended.update({"n_coldspots_mean": 0.0, "coldspot_depth_mean": np.nan})
    
    if data.get("cmb_aoe") is not None:
        aoe_df = data["cmb_aoe"]
        if len(aoe_df) > 0:
            extended["alignment_angle_mean"] = aoe_df.get("alignment_angle", pd.Series([np.nan])).mean()
            extended["alignment_angle_std"] = aoe_df.get("alignment_angle", pd.Series([np.nan])).std()
        else:
            extended.update({"alignment_angle_mean": np.nan, "alignment_angle_std": np.nan})
    else:
        extended.update({"alignment_angle_mean": np.nan, "alignment_angle_std": np.nan})
    
    # I-DEFINITIONS COMPARISON
    if data.get("i_definitions_comparison") is not None:
        i_comp_df = data["i_definitions_comparison"]
        if i_def in i_comp_df.columns and len(i_comp_df) > 0:
            i_values = i_comp_df[i_def].dropna()
            if len(i_values) > 0:
                extended["I_value_mean"] = i_values.mean()
                extended["I_value_std"] = i_values.std()
                extended["I_value_range"] = i_values.max() - i_values.min()
            else:
                extended.update({"I_value_mean": np.nan, "I_value_std": np.nan, "I_value_range": np.nan})
        else:
            extended.update({"I_value_mean": np.nan, "I_value_std": np.nan, "I_value_range": np.nan})
    else:
        extended.update({"I_value_mean": np.nan, "I_value_std": np.nan, "I_value_range": np.nan})
    
    # LIFE COMPATIBILITY SUMMARY
    life_summary = data.get("life_compatibility")
    if life_summary and isinstance(life_summary, dict):
        metrics_block = life_summary.get("metrics", {})
        extended["life_score_json"] = metrics_block.get("life_compatibility_score", np.nan)
        extended["complexity_score_json"] = metrics_block.get("complexity_score", np.nan)
        extended["information_richness_json"] = metrics_block.get("information_richness", np.nan)
        life_components = life_summary.get("life_components", {})
        extended["life_planck_component"] = life_components.get("planck_fit_quality", np.nan)
        extended["life_stability_component"] = life_components.get("stability_quality", np.nan)
        extended["life_goldilocks_component"] = life_components.get("goldilocks_robustness", np.nan)
    else:
        extended.update({
            "life_score_json": np.nan,
            "complexity_score_json": np.nan,
            "information_richness_json": np.nan,
            "life_planck_component": np.nan,
            "life_stability_component": np.nan,
            "life_goldilocks_component": np.nan
        })

    # PLANCK VALIDATION
    planck_data = data.get("planck_validation")
    if planck_data:
        planck_summary = planck_data.get("summary") or {}
        extended["planck_E"] = planck_summary.get("E", np.nan)
        extended["planck_I"] = planck_summary.get("I", np.nan)
        extended["planck_alpha"] = planck_summary.get("alpha", np.nan)
        extended["planck_chi2_total"] = planck_summary.get("chi2_total", np.nan)
        extended["planck_chi2_reduced"] = planck_summary.get("chi2_reduced", np.nan)
        extended["planck_score"] = planck_summary.get("planck_score", np.nan)
        if planck_data.get("validation") is not None:
            val_df = planck_data["validation"]
            extended["planck_validation_chi2_mean"] = val_df["chi2"].mean() if "chi2" in val_df.columns else np.nan
            if "ell" in val_df.columns:
                extended["planck_validation_ell_span"] = val_df["ell"].max() - val_df["ell"].min()
            else:
                extended["planck_validation_ell_span"] = np.nan
        else:
            extended.update({
                "planck_validation_chi2_mean": np.nan,
                "planck_validation_ell_span": np.nan
            })
    else:
        extended.update({
            "planck_E": np.nan,
            "planck_I": np.nan,
            "planck_alpha": np.nan,
            "planck_chi2_total": np.nan,
            "planck_chi2_reduced": np.nan,
            "planck_score": np.nan,
            "planck_validation_chi2_mean": np.nan,
            "planck_validation_ell_span": np.nan
        })

    # ENTROPY VOLATILITY SUMMARY (aggregated)
    entropy_df = data.get("entropy_volatility")
    if isinstance(entropy_df, pd.DataFrame) and "volatility" in entropy_df.columns:
        extended["entropy_volatility_global_mean"] = entropy_df["volatility"].mean()
        extended["entropy_volatility_global_std"] = entropy_df["volatility"].std()
        extended["entropy_volatility_max"] = entropy_df["volatility"].max()
    else:
        extended.update({
            "entropy_volatility_global_mean": np.nan,
            "entropy_volatility_global_std": np.nan,
            "entropy_volatility_max": np.nan
        })

    # STABILITY SWEEPS
    def _compute_sweep_metrics(df: Optional[pd.DataFrame], column: str) -> Tuple[float, float]:
        if df is None or column not in df.columns:
            return np.nan, np.nan
        try:
            eps_vals = pd.to_numeric(df.get("eps"), errors="coerce")
            ratios = pd.to_numeric(df[column], errors="coerce")
            mask = (~eps_vals.isna()) & (~ratios.isna())
            eps_vals = eps_vals[mask]
            ratios = ratios[mask]
            if len(eps_vals) > 1:
                log_eps = np.log10(eps_vals.replace(0, np.nan))
                valid_mask = ~log_eps.isna()
                log_eps = log_eps[valid_mask]
                ratios = ratios[valid_mask]
                if len(log_eps) > 1:
                    slope, intercept = np.polyfit(log_eps, ratios, 1)
                    return slope, intercept
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    eps_sweep = data.get("stability_sweep_eps")
    zero_sweep = data.get("stability_sweep_zero")
    slope, intercept = _compute_sweep_metrics(eps_sweep, "stable_ratio")
    extended["stability_eps_slope"] = slope
    extended["stability_eps_intercept"] = intercept
    if zero_sweep is not None and "stable_ratio" in zero_sweep.columns:
        try:
            extended["stability_zero_baseline"] = pd.to_numeric(zero_sweep["stable_ratio"], errors="coerce").max()
        except Exception:
            extended["stability_zero_baseline"] = np.nan
    else:
        extended["stability_zero_baseline"] = np.nan

    # ADVANCED ANOMALIES
    adv_anomalies = data.get("advanced_anomalies")
    if adv_anomalies:
        adv_df = adv_anomalies.get("advanced_anomalies")
        if isinstance(adv_df, pd.DataFrame) and "deviation_sigma" in adv_df.columns:
            sigmas = pd.to_numeric(adv_df["deviation_sigma"], errors="coerce")
            extended["advanced_anomaly_sigma_mean"] = sigmas.mean()
            extended["advanced_anomaly_sigma_max"] = sigmas.max()
        else:
            extended["advanced_anomaly_sigma_mean"] = np.nan
            extended["advanced_anomaly_sigma_max"] = np.nan
        phys_df = adv_anomalies.get("physical_anomalies")
        extended["physical_anomaly_count"] = len(phys_df) if isinstance(phys_df, pd.DataFrame) else 0
        gaussian_df = adv_anomalies.get("cmb_gaussianity")
        extended["cmb_gaussianity_p_mean"] = gaussian_df["p_value"].mean() if isinstance(gaussian_df, pd.DataFrame) and "p_value" in gaussian_df.columns else np.nan
        isotropy_df = adv_anomalies.get("cmb_isotropy")
        extended["cmb_anisotropy_index_mean"] = isotropy_df["anisotropy_index"].mean() if isinstance(isotropy_df, pd.DataFrame) and "anisotropy_index" in isotropy_df.columns else np.nan
    else:
        extended.update({
            "advanced_anomaly_sigma_mean": np.nan,
            "advanced_anomaly_sigma_max": np.nan,
            "physical_anomaly_count": 0,
            "cmb_gaussianity_p_mean": np.nan,
            "cmb_anisotropy_index_mean": np.nan
        })

    # NESTED SAMPLING
    nested_df = data.get("nested_sampling")
    if isinstance(nested_df, pd.DataFrame):
        extended["nested_sampling_iterations"] = len(nested_df)
        if "logZ" in nested_df.columns:
            extended["nested_logZ_final"] = nested_df["logZ"].iloc[-1]
            extended["nested_logZ_span"] = nested_df["logZ"].max() - nested_df["logZ"].min()
        else:
            extended["nested_logZ_final"] = np.nan
            extended["nested_logZ_span"] = np.nan
    else:
        extended.update({
            "nested_sampling_iterations": 0,
            "nested_logZ_final": np.nan,
            "nested_logZ_span": np.nan
        })

    # PRE-FLUCTUATION / SEEDS
    pre_pairs = data.get("pre_fluctuation_pairs")
    extended["pre_fluctuation_pairs"] = len(pre_pairs) if isinstance(pre_pairs, pd.DataFrame) else 0
    
    universe_seeds = data.get("universe_seeds")
    if isinstance(universe_seeds, pd.DataFrame):
        if "seed" in universe_seeds.columns:
            extended["unique_seed_count"] = universe_seeds["seed"].nunique()
        else:
            extended["unique_seed_count"] = len(universe_seeds)
    else:
        extended["unique_seed_count"] = np.nan

    # TOP UNIVERSE SNAPSHOT (from summary)
    summary_block = data.get("summary")
    if summary_block and summary_block.get("complexity_analysis", {}).get("top_universes"):
        top_universe = summary_block["complexity_analysis"]["top_universes"][0]
        extended["top_universe_seed"] = top_universe.get("seed")
        extended["top_universe_lock_epoch"] = top_universe.get("lock_epoch")
        extended["top_universe_I"] = top_universe.get("I")
    else:
        extended["top_universe_seed"] = None
        extended["top_universe_lock_epoch"] = np.nan
        extended["top_universe_I"] = np.nan

    return extended


def extract_metrics_from_summary(summary: Dict, i_def: str) -> Dict:
    """
    Extract all relevant metrics from summary JSON.
    
    Includes:
    - Basic stability metrics
    - Goldilocks and Bayesian parameters
    - Advanced complexity indicators
    - Life-compatibility indicators
    """
    stab_sum = summary.get("stability_summary", {})
    gold = summary.get("goldilocks_window_used", {})
    bayes = summary.get("bayesian_model_selection", {})
    
    # Basic metrics
    total_univ = summary.get("N_samples", 0)
    stable_count = stab_sum.get("stable_universes", 0)
    lockin_count = stab_sum.get("lockin_universes", 0)
    
    metrics = {
        "i_definition": i_def,
        
        # Stability metrics
        "total_universes": total_univ,
        "stable_count": stable_count,
        "unstable_count": stab_sum.get("unstable_universes", 0),
        "lockin_count": lockin_count,
        "stable_percent": stab_sum.get("stable_percent", 0.0),
        "unstable_percent": stab_sum.get("unstable_percent", 0.0),
        "lockin_percent": stab_sum.get("lockin_percent", 0.0),
        
        # Goldilocks metrics
        "X_peak": gold.get("X_peak", 0.0),
        "X_peak_uncertainty": gold.get("X_peak_uncertainty", 0.0),
        "X_low": gold.get("X_low_plot_est", 0.0),
        "X_high": gold.get("X_high_plot_est", 0.0),
        "goldilocks_width": gold.get("X_high_plot_est", 0.0) - gold.get("X_low_plot_est", 0.0),
        
        # Bayesian metrics
        "ucb_kappa": gold.get("ucb_kappa", 0.0),
        "gp_noise": gold.get("gp_noise", 0.0),
        "bayesian_samples": gold.get("total_sampled", 0),
        
        # Bayesian model selection
        "BIC": bayes.get("BIC", np.nan),
        "AIC": bayes.get("AIC", np.nan),
        "log_evidence": bayes.get("log_evidence", np.nan),
        "chi_squared_reduced": bayes.get("chi_squared_reduced", np.nan),
    }
    
    # ═══════════════════════════════════════════════════════════════
    # ADVANCED METRICS: COMPLEXITY & LIFE-COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════
    
    # COMPLEXITY SCORE (0-100):
    # Measures structural richness and information content
    complexity_components = []
    
    # 1. Lock-in quality (not quantity!) - Fast, decisive lock-in = complex
    if lockin_count > 0 and total_univ > 0:
        lockin_rate = lockin_count / total_univ
        # Higher lock-in rate among TOTAL (not just stable) = better
        complexity_components.append(min(lockin_rate * 200, 100))  # Scale: 0-50% → 0-100
    else:
        complexity_components.append(0)
    
    # 2. Goldilocks precision (lower uncertainty = sharper, more interesting physics)
    if gold.get("X_peak", 0) > 0:
        rel_uncertainty = gold.get("X_peak_uncertainty", 0) / gold.get("X_peak", 1)
        precision_score = max(0, 100 - rel_uncertainty * 1000)  # Lower uncertainty = higher score
        complexity_components.append(min(precision_score, 100))
    else:
        complexity_components.append(50)
    
    # 3. Information richness (E+I specific) - Is I-coupling effective?
    if i_def != "energy_only":
        # E+I: Effectiveness = stable % relative to E-only (will be computed in comparison)
        # Placeholder: Use lock-in as proxy
        info_richness = min(stab_sum.get("lockin_percent", 0) * 5, 100)  # 20% lock-in = 100 score
        complexity_components.append(info_richness)
    else:
        complexity_components.append(0)  # E-only has no I-coupling
    
    # Average complexity components
    metrics["complexity_score"] = np.mean(complexity_components) if complexity_components else 0
    
    # LIFE-COMPATIBILITY SCORE (0-100):
    # Measures potential for structure formation and life
    life_components = []
    
    # 1. Planck fit quality (observationally compatible universe)
    chi2 = bayes.get("chi_squared_reduced", np.nan)
    if not np.isnan(chi2):
        # Perfect fit = 1.0, good fit < 2.0
        planck_score = max(0, 100 - abs(chi2 - 1.0) * 25)  # |χ²-1| = 0 → 100, |χ²-1| = 4 → 0
        life_components.append(min(planck_score, 100))
    else:
        life_components.append(50)  # Neutral if no data
    
    # 2. Stability quality (stable AND lock-in is best)
    if stable_count > 0:
        # Proportion of stable universes that lock-in (high = good)
        lockin_among_stable = lockin_count / stable_count if stable_count > 0 else 0
        stability_quality = lockin_among_stable * 100
        life_components.append(min(stability_quality, 100))
    else:
        life_components.append(0)
    
    # 3. Goldilocks robustness (wider, more forgiving zone = life-compatible)
    gold_width = gold.get("X_high_plot_est", 0) - gold.get("X_low_plot_est", 0)
    if gold_width > 0:
        # Wider zone = more robust (normalize to ~10-20 typical width)
        robustness = min(gold_width / 20.0 * 100, 100)
        life_components.append(robustness)
    else:
        life_components.append(50)
    
    # Average life-compatibility components
    metrics["life_compatibility_score"] = np.mean(life_components) if life_components else 0
    
    # INFORMATION RICHNESS (E+I specific):
    # How effective is the I-parameter at directing complexity?
    if i_def != "energy_only":
        # Lock-in rate is proxy for I-parameter effectiveness
        info_effectiveness = min(stab_sum.get("lockin_percent", 0) * 5, 100)
        metrics["information_richness"] = info_effectiveness
    else:
        metrics["information_richness"] = 0
    
    return metrics


def build_metrics_dataframe(collected_data: Dict) -> pd.DataFrame:
    """Build comprehensive metrics DataFrame from collected data (EXTENDED)."""
    metrics_list = []
    
    print("\n  Building extended metrics DataFrame...")
    
    # E-only metrics
    for dirname, data in collected_data["eonly"].items():
        metrics = extract_metrics_from_summary(data["summary"], "energy_only")
        metrics["run_type"] = "E-only"
        #  Add extended metrics
        extended = extract_extended_metrics(data, "energy_only")
        metrics.update(extended)
        metrics_list.append(metrics)
        print(f"    ✅ Extracted extended metrics: energy_only")
    
    # E+I metrics
    for i_def, data in collected_data["ei"].items():
        metrics = extract_metrics_from_summary(data["summary"], i_def)
        metrics["run_type"] = "E+I"
        #  Add extended metrics
        extended = extract_extended_metrics(data, i_def)
        metrics.update(extended)
        metrics_list.append(metrics)
        print(f"    ✅ Extracted extended metrics: {i_def}")
    
    df = pd.DataFrame(metrics_list)
    print(f"  ✅ DataFrame built: {len(df)} runs, {len(df.columns)} columns (50-80 extended)")
    return df


# ==========================================================================================
# PHASE 3: COMPARATIVE ANALYSIS
# ==========================================================================================
# Purpose: Compare E+I definitions and E-only vs E+I (if available)
# Input: Metrics DataFrame from Phase 2
# Output: Comparison plots, tables, and improvement metrics
# ==========================================================================================

def compare_ei_definitions(df_metrics: pd.DataFrame, output_dir: str):
    """
    Compare all E+I definitions (10 I-parameters: including Jensen-Shannon).
    
    Generates:
    - Stability rates comparison
    - Goldilocks zones comparison
    - Planck fit comparison
    - Ranking table
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: E+I DEFINITIONS COMPARISON")
    print("="*70)
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data found, skipping E+I comparison")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Stability rates comparison
    print("\n1.1 Stability Rates Comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Stable %
    df_ei_sorted = df_ei.sort_values("stable_percent", ascending=False)
    axes[0].barh(df_ei_sorted["i_definition"], df_ei_sorted["stable_percent"], color='green', alpha=0.7)
    axes[0].set_xlabel("Stable %", fontsize=12, fontweight='bold')
    axes[0].set_title("Stability Rate", fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Unstable %
    df_ei_sorted = df_ei.sort_values("unstable_percent", ascending=True)
    axes[1].barh(df_ei_sorted["i_definition"], df_ei_sorted["unstable_percent"], color='red', alpha=0.7)
    axes[1].set_xlabel("Unstable %", fontsize=12, fontweight='bold')
    axes[1].set_title("Instability Rate", fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Lock-in %
    df_ei_sorted = df_ei.sort_values("lockin_percent", ascending=False)
    axes[2].barh(df_ei_sorted["i_definition"], df_ei_sorted["lockin_percent"], color='blue', alpha=0.7)
    axes[2].set_xlabel("Lock-in %", fontsize=12, fontweight='bold')
    axes[2].set_title("Law Lock-in Rate", fontsize=14, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_rates_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ stability_rates_comparison.png")
    
    # 2. Goldilocks zones comparison
    print("\n1.2 Goldilocks Zones Comparison")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, row in df_ei.iterrows():
        y_pos = i
        x_peak = row["X_peak"]
        x_unc = row["X_peak_uncertainty"]
        x_low = row["X_low"]
        x_high = row["X_high"]
        
        # Goldilocks window
        ax.barh(y_pos, x_high - x_low, left=x_low, height=0.6, 
                alpha=0.3, color='yellow', edgecolor='green', linewidth=2)
        
        # Peak with uncertainty
        ax.errorbar(x_peak, y_pos, xerr=x_unc*1.96, fmt='o', 
                   color='red', markersize=10, capsize=5, capthick=2, linewidth=2)
    
    ax.set_yticks(range(len(df_ei)))
    ax.set_yticklabels(df_ei["i_definition"])
    ax.set_xlabel("X (E·I coupling)", fontsize=12, fontweight='bold')
    ax.set_title("Goldilocks Zones Comparison (Yellow=Zone, Red=Peak±σ)", fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "goldilocks_zones_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ goldilocks_zones_comparison.png")
    
    # 3. Chi-squared comparison
    print("\n1.3 Planck χ² Fit Comparison")
    df_ei_chi = df_ei.dropna(subset=["chi_squared_reduced"])
    
    if len(df_ei_chi) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        df_ei_chi_sorted = df_ei_chi.sort_values("chi_squared_reduced")
        ax.barh(df_ei_chi_sorted["i_definition"], df_ei_chi_sorted["chi_squared_reduced"], 
                color='purple', alpha=0.7)
        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect fit (χ²/dof=1)')
        ax.set_xlabel("χ²/dof", fontsize=12, fontweight='bold')
        ax.set_title("Planck Validation: χ² Fit Quality (lower is better)", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "planck_chi2_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ planck_chi2_comparison.png")
    else:
        print("   ⚠️  No χ² data available")
    
    # 4. Ranking table
    print("\n1.4 Ranking Table")
    ranking_df = df_ei[["i_definition", "stable_percent", "lockin_percent", 
                        "X_peak_uncertainty", "chi_squared_reduced"]].copy()
    ranking_df = ranking_df.sort_values("stable_percent", ascending=False)
    ranking_df.to_csv(os.path.join(output_dir, "ei_ranking_table.csv"), index=False)
    print("   ✅ ei_ranking_table.csv")
    
    print("\n✅ E+I Definitions Comparison Complete!")


def compare_eonly_vs_ei(df_metrics: pd.DataFrame, output_dir: str):
    """
    PHASE 3B: Compare E-only vs E+I (if E-only data available in batch_all mode).
    
    Analyzes improvements from adding I-coupling:
    - Stability rate improvements (stable %, lock-in %)
    - Goldilocks peak shifts (E-only baseline vs E+I peaks)
    - Detailed improvement metrics for each I-definition
    
    Generates:
    - stability_improvement.png (bar charts showing Δ% vs E-only)
    - goldilocks_shift.png (peak position changes with arrows)
    - eonly_vs_ei_metrics.json (detailed improvement data)
    """
    print("\n" + "="*70)
    print("PHASE 3B: E-ONLY vs E+I COMPARISON")
    print("="*70)
    
    df_eonly = df_metrics[df_metrics["run_type"] == "E-only"]
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    if len(df_eonly) == 0:
        print("⚠️  No E-only data found, skipping E-only vs E+I comparison")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get E-only baseline
    eonly_stable = df_eonly["stable_percent"].iloc[0]
    eonly_lockin = df_eonly["lockin_percent"].iloc[0]
    eonly_X_peak = df_eonly["X_peak"].iloc[0]
    
    # 1. Stability improvement
    print("\n2.1 Stability Improvement")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Stable %
    improvements_stable = df_ei["stable_percent"] - eonly_stable
    df_plot = df_ei.copy()
    df_plot["improvement_stable"] = improvements_stable
    df_plot_sorted = df_plot.sort_values("improvement_stable", ascending=False)
    
    colors_stable = ['green' if x > 0 else 'red' for x in df_plot_sorted["improvement_stable"]]
    axes[0].barh(df_plot_sorted["i_definition"], df_plot_sorted["improvement_stable"], color=colors_stable, alpha=0.7)
    axes[0].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[0].set_xlabel("Improvement in Stable % (vs E-only)", fontsize=12, fontweight='bold')
    axes[0].set_title("Stability Improvement", fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Lock-in %
    improvements_lockin = df_ei["lockin_percent"] - eonly_lockin
    df_plot["improvement_lockin"] = improvements_lockin
    df_plot_sorted = df_plot.sort_values("improvement_lockin", ascending=False)
    
    colors_lockin = ['blue' if x > 0 else 'red' for x in df_plot_sorted["improvement_lockin"]]
    axes[1].barh(df_plot_sorted["i_definition"], df_plot_sorted["improvement_lockin"], color=colors_lockin, alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel("Improvement in Lock-in % (vs E-only)", fontsize=12, fontweight='bold')
    axes[1].set_title("Law Lock-in Improvement", fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_improvement.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ stability_improvement.png")
    
    # 2. Goldilocks shift
    print("\n2.2 Goldilocks Peak Shift")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # E-only baseline
    ax.axhline(-1, color='gray', linestyle='--', linewidth=2)
    ax.scatter([eonly_X_peak], [-1], s=200, color='gray', marker='s', 
               label='E-only baseline', zorder=10, edgecolors='black', linewidths=2)
    
    # E+I peaks
    for i, row in df_ei.iterrows():
        y_pos = i
        x_peak = row["X_peak"]
        x_unc = row["X_peak_uncertainty"]
        
        ax.errorbar(x_peak, y_pos, xerr=x_unc*1.96, fmt='o', 
                   color='blue', markersize=10, capsize=5, capthick=2, linewidth=2)
        
        # Arrow from E-only to E+I
        ax.annotate('', xy=(x_peak, y_pos), xytext=(eonly_X_peak, y_pos),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.5))
    
    ax.set_yticks(list(range(-1, len(df_ei))))
    ax.set_yticklabels(['E-only'] + list(df_ei["i_definition"]))
    ax.set_xlabel("X_peak (E·I coupling)", fontsize=12, fontweight='bold')
    ax.set_title("Goldilocks Peak Position Comparison", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "goldilocks_shift.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ goldilocks_shift.png")
    
    # 3. Metrics JSON
    print("\n2.3 Metrics Summary")
    comparison_metrics = {
        "eonly_baseline": {
            "stable_percent": float(eonly_stable),
            "lockin_percent": float(eonly_lockin),
            "X_peak": float(eonly_X_peak)
        },
        "ei_improvements": {
            i_def: {
                "stable_improvement": float(row["stable_percent"] - eonly_stable),
                "lockin_improvement": float(row["lockin_percent"] - eonly_lockin),
                "X_peak_shift": float(row["X_peak"] - eonly_X_peak)
            }
            for i_def, row in df_ei.set_index("i_definition").iterrows()
        }
    }
    
    with open(os.path.join(output_dir, "eonly_vs_ei_metrics.json"), 'w') as f:
        json.dump(comparison_metrics, f, indent=2)
    print("   ✅ eonly_vs_ei_metrics.json")
    
    print("\n✅ E-only vs E+I Comparison Complete!")


# ==========================================================================================
# PHASE 3 - EXTENDED ANALYSIS CATEGORIES
# ==========================================================================================

def analyze_emergent_laws(df_metrics: pd.DataFrame, output_dir: str):
    """
    CATEGORY 1: Emergent Laws Comparison.
    
    Compares:
    - Power-law exponents across I-definitions
    - Phase transition detection rates
    - Correlation strengths
    
    Generates:
    - power_law_exponent_comparison.png
    - phase_transition_detection_rate.png
    - emergent_law_heatmap.png
    - emergent_laws_metrics.csv
    """
    print("\n" + "="*70)
    print("CATEGORY 1: EMERGENT LAWS COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data, skipping emergent laws analysis")
        return
    
    # Check if emergent laws data exists
    if "power_law_exponent" not in df_ei.columns:
        print("⚠️  No emergent laws data available, skipping analysis")
        return
    
    # 1. Power-law exponent comparison
    print("\n1.1 Power-Law Exponent Distribution")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_plot = df_ei.dropna(subset=["power_law_exponent"])
    if len(df_plot) > 0:
        df_sorted = df_plot.sort_values("power_law_exponent", ascending=False)
        bars = ax.barh(df_sorted["i_definition"], df_sorted["power_law_exponent"], 
                      color='purple', alpha=0.7)
        ax.set_xlabel("Power-Law Exponent α", fontsize=12, fontweight='bold')
        ax.set_title("Power-Law Exponent Comparison (X ∝ E^α)", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Linear (α=1.0)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "power_law_exponent_comparison.png"), 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ power_law_exponent_comparison.png")
    else:
        print("   ⚠️  No valid power-law data")
    
    # 2. Phase transition detection rate
    print("\n1.2 Phase Transition Detection Rate")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_plot = df_ei.dropna(subset=["n_phase_transitions"])
    if len(df_plot) > 0:
        df_sorted = df_plot.sort_values("n_phase_transitions", ascending=False)
        bars = ax.barh(df_sorted["i_definition"], df_sorted["n_phase_transitions"], 
                      color='orange', alpha=0.7)
        ax.set_xlabel("Number of Phase Transitions Detected", fontsize=12, fontweight='bold')
        ax.set_title("Phase Transition Detection Rate", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "phase_transition_detection_rate.png"), 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ phase_transition_detection_rate.png")
    else:
        print("   ⚠️  No phase transition data")
    
    # 3. Emergent law quality heatmap
    print("\n1.3 Emergent Law Quality Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create heatmap data
    heatmap_data = df_ei[["i_definition", "power_law_exponent", "correlation_strength", "n_phase_transitions"]].copy()
    heatmap_data = heatmap_data.dropna(subset=["power_law_exponent"])
    
    if len(heatmap_data) > 0:
        heatmap_data = heatmap_data.set_index("i_definition")
        heatmap_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()) * 100 if x.max() > x.min() else x, axis=0)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score (0-100)'}, ax=ax)
        ax.set_title("Emergent Laws Quality Heatmap (Normalized 0-100)", 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel("")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "emergent_law_heatmap.png"), 
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ emergent_law_heatmap.png")
    else:
        print("   ⚠️  Insufficient data for heatmap")
    
    # 4. Export CSV
    print("\n1.4 Emergent Laws Metrics Export")
    metrics_cols = ["i_definition", "power_law_exponent", "correlation_strength", 
                   "phase_transition_detected", "n_phase_transitions"]
    export_df = df_ei[metrics_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "emergent_laws_metrics.csv"), index=False)
    print("   ✅ emergent_laws_metrics.csv")
    
    print("\n✅ Emergent Laws Comparison Complete!")


def analyze_friedmann_cosmology(df_metrics: pd.DataFrame, output_dir: str):
    """
    CATEGORY 2: Friedmann Cosmology Comparison.
    
    Compares:
    - Universe age distribution (vs Planck 13.8 Gyr)
    - Hubble parameter (vs Planck 67.4 km/s/Mpc)
    - Omega parameters (vs Planck 2018)
    - Cosmological epoch timing
    
    Generates:
    - universe_age_distribution.png
    - hubble_parameter_comparison.png
    - omega_parameters_planck_comparison.png
    - cosmological_epoch_timing.png
    - friedmann_consistency_score.png
    - friedmann_metrics.csv
    """
    print("\n" + "="*70)
    print("CATEGORY 2: FRIEDMANN COSMOLOGY COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data, skipping Friedmann analysis")
        return
    
    # Planck 2018 reference values
    PLANCK_AGE = 13.8  # Gyr
    PLANCK_H0 = 67.4   # km/s/Mpc
    PLANCK_OMEGA_M = 0.315
    PLANCK_OMEGA_B = 0.049
    
    # 1. Universe age distribution
    print("\n2.1 Universe Age Distribution")
    if "age_Gyr_mean" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["age_Gyr_mean"])
        if len(df_plot) > 0:
            positions = range(len(df_plot))
            ax.barh(positions, df_plot["age_Gyr_mean"], 
                   xerr=df_plot["age_Gyr_std"] if "age_Gyr_std" in df_plot.columns else None,
                   color='skyblue', alpha=0.7, capsize=5)
            ax.set_yticks(positions)
            ax.set_yticklabels(df_plot["i_definition"])
            ax.axvline(x=PLANCK_AGE, color='red', linestyle='--', linewidth=2.5, 
                      label=f'Planck 2018: {PLANCK_AGE} Gyr')
            ax.set_xlabel("Universe Age (Gyr)", fontsize=12, fontweight='bold')
            ax.set_title("Universe Age Distribution vs Planck 2018", fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "universe_age_distribution.png"), 
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ universe_age_distribution.png")
        else:
            print("   ⚠️  No age data")
    else:
        print("   ⚠️  No age_Gyr_mean column")
    
    # 2. Hubble parameter comparison
    print("\n2.2 Hubble Parameter Comparison")
    if "H0_mean" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["H0_mean"])
        if len(df_plot) > 0:
            positions = range(len(df_plot))
            ax.barh(positions, df_plot["H0_mean"],
                   xerr=df_plot["H0_std"] if "H0_std" in df_plot.columns else None,
                   color='lightcoral', alpha=0.7, capsize=5)
            ax.set_yticks(positions)
            ax.set_yticklabels(df_plot["i_definition"])
            ax.axvline(x=PLANCK_H0, color='red', linestyle='--', linewidth=2.5,
                      label=f'Planck 2018: {PLANCK_H0} km/s/Mpc')
            ax.set_xlabel("Hubble Parameter H₀ (km/s/Mpc)", fontsize=12, fontweight='bold')
            ax.set_title("Hubble Parameter vs Planck 2018", fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "hubble_parameter_comparison.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ hubble_parameter_comparison.png")
        else:
            print("   ⚠️  No H0 data")
    else:
        print("   ⚠️  No H0_mean column")
    
    # 3. Omega parameters comparison
    print("\n2.3 Omega Parameters vs Planck 2018")
    if "Omega_m_mean" in df_ei.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        omega_params = [
            ("Omega_m_mean", PLANCK_OMEGA_M, "Ωₘ (Matter)", axes[0]),
            ("Omega_b_mean", PLANCK_OMEGA_B, "Ωᵦ (Baryon)", axes[1]),
            ("Omega_k_mean", 0.0, "Ωₖ (Curvature)", axes[2])
        ]
        
        for col, planck_val, label, ax in omega_params:
            if col in df_ei.columns:
                df_plot = df_ei.dropna(subset=[col])
                if len(df_plot) > 0:
                    positions = range(len(df_plot))
                    ax.barh(positions, df_plot[col], color='lightgreen', alpha=0.7)
                    ax.set_yticks(positions)
                    ax.set_yticklabels(df_plot["i_definition"])
                    ax.axvline(x=planck_val, color='red', linestyle='--', linewidth=2,
                              label=f'Planck: {planck_val:.3f}')
                    ax.set_xlabel(label, fontsize=11, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle("Omega Parameters vs Planck 2018", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "omega_parameters_planck_comparison.png"),
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ omega_parameters_planck_comparison.png")
    else:
        print("   ⚠️  No Omega data")
    
    # 4. Friedmann consistency score
    print("\n2.4 Friedmann Consistency Score")
    if "age_deviation_from_planck" in df_ei.columns and "H0_deviation_from_planck" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["age_deviation_from_planck", "H0_deviation_from_planck"])
        if len(df_plot) > 0:
            # Compute consistency score (lower deviation = higher score)
            age_score = 100 / (1 + df_plot["age_deviation_from_planck"])
            h0_score = 100 / (1 + df_plot["H0_deviation_from_planck"])
            consistency_score = (age_score + h0_score) / 2
            
            df_plot_sorted = df_plot.copy()
            df_plot_sorted["consistency_score"] = consistency_score.values
            df_plot_sorted = df_plot_sorted.sort_values("consistency_score", ascending=False)
            
            bars = ax.barh(df_plot_sorted["i_definition"], df_plot_sorted["consistency_score"],
                          color='teal', alpha=0.7)
            ax.set_xlabel("Friedmann Consistency Score (0-100)", fontsize=12, fontweight='bold')
            ax.set_title("Friedmann Consistency vs Planck 2018", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "friedmann_consistency_score.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ friedmann_consistency_score.png")
        else:
            print("   ⚠️  No Friedmann deviation data")
    else:
        print("   ⚠️  No Friedmann consistency data")
    
    # 5. Export CSV
    print("\n2.5 Friedmann Metrics Export")
    friedmann_cols = ["i_definition", "age_Gyr_mean", "age_Gyr_std", "age_deviation_from_planck",
                     "H0_mean", "H0_std", "H0_deviation_from_planck",
                     "Omega_m_mean", "Omega_b_mean", "Omega_k_mean"]
    available_cols = [c for c in friedmann_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "friedmann_metrics.csv"), index=False)
    print("   ✅ friedmann_metrics.csv")
    
    print("\n✅ Friedmann Cosmology Comparison Complete!")


def analyze_cmb_anomalies(df_metrics: pd.DataFrame, output_dir: str):
    """
    CATEGORY 3: CMB Anomalies Comparison.
    
    Compares:
    - Cold spot detection rates
    - Axis of Evil alignment angles
    
    Generates:
    - cmb_coldspot_detection_rate.png
    - axis_of_evil_comparison.png
    - cmb_anomaly_metrics.csv
    """
    print("\n" + "="*70)
    print("CATEGORY 3: CMB ANOMALIES COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data, skipping CMB anomalies analysis")
        return
    
    # 1. Cold spot detection rate
    print("\n3.1 CMB Cold Spot Detection Rate")
    if "n_coldspots_mean" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["n_coldspots_mean"])
        if len(df_plot) > 0:
            df_sorted = df_plot.sort_values("n_coldspots_mean", ascending=False)
            bars = ax.barh(df_sorted["i_definition"], df_sorted["n_coldspots_mean"],
                          color='darkblue', alpha=0.7)
            ax.set_xlabel("Average Cold Spots per Universe", fontsize=12, fontweight='bold')
            ax.set_title("CMB Cold Spot Detection Rate", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cmb_coldspot_detection_rate.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ cmb_coldspot_detection_rate.png")
        else:
            print("   ⚠️  No cold spot data")
    else:
        print("   ⚠️  No n_coldspots_mean column")
    
    # 2. Axis of Evil alignment
    print("\n3.2 Axis of Evil Alignment Distribution")
    if "alignment_angle_mean" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["alignment_angle_mean"])
        if len(df_plot) > 0:
            positions = range(len(df_plot))
            ax.barh(positions, df_plot["alignment_angle_mean"],
                   xerr=df_plot["alignment_angle_std"] if "alignment_angle_std" in df_plot.columns else None,
                   color='gold', alpha=0.7, capsize=5)
            ax.set_yticks(positions)
            ax.set_yticklabels(df_plot["i_definition"])
            ax.set_xlabel("Quadrupole-Octupole Alignment Angle (degrees)", fontsize=12, fontweight='bold')
            ax.set_title("Axis of Evil Alignment Comparison", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "axis_of_evil_comparison.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ axis_of_evil_comparison.png")
        else:
            print("   ⚠️  No alignment angle data")
    else:
        print("   ⚠️  No alignment_angle_mean column")
    
    # 3. Export CSV
    print("\n3.3 CMB Anomaly Metrics Export")
    cmb_cols = ["i_definition", "n_coldspots_mean", "coldspot_depth_mean",
               "alignment_angle_mean", "alignment_angle_std"]
    available_cols = [c for c in cmb_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "cmb_anomaly_metrics.csv"), index=False)
    print("   ✅ cmb_anomaly_metrics.csv")
    
    print("\n✅ CMB Anomalies Comparison Complete!")


def analyze_lockin_dynamics(df_metrics: pd.DataFrame, output_dir: str):
    """
    CATEGORY 4: Lock-in Dynamics Comparison.
    
    Compares:
    - Lock-in timing distribution
    - Lock-in efficiency
    - Early vs late lock-in rates
    
    Generates:
    - lockin_timing_comparison.png
    - lockin_efficiency_boxplot.png
    - early_vs_late_lockin.png
    - lockin_dynamics_metrics.csv
    """
    print("\n" + "="*70)
    print("CATEGORY 4: LOCK-IN DYNAMICS COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data, skipping lock-in dynamics analysis")
        return
    
    # 1. Lock-in timing distribution
    print("\n4.1 Lock-in Timing Distribution")
    if "lock_epoch_mean" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["lock_epoch_mean"])
        if len(df_plot) > 0:
            positions = range(len(df_plot))
            ax.barh(positions, df_plot["lock_epoch_mean"],
                   xerr=df_plot["lock_epoch_std"] if "lock_epoch_std" in df_plot.columns else None,
                   color='mediumseagreen', alpha=0.7, capsize=5)
            ax.set_yticks(positions)
            ax.set_yticklabels(df_plot["i_definition"])
            ax.set_xlabel("Lock-in Epoch (average)", fontsize=12, fontweight='bold')
            ax.set_title("Lock-in Timing Comparison", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "lockin_timing_comparison.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ lockin_timing_comparison.png")
        else:
            print("   ⚠️  No lock timing data")
    else:
        print("   ⚠️  No lock_epoch_mean column")
    
    # 2. Lock-in efficiency (Δt = lock - stable)
    print("\n4.2 Lock-in Efficiency")
    if "lockin_efficiency" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["lockin_efficiency"])
        if len(df_plot) > 0:
            df_sorted = df_plot.sort_values("lockin_efficiency")
            positions = range(len(df_sorted))
            bars = ax.barh(positions, df_sorted["lockin_efficiency"],
                          color='mediumpurple', alpha=0.7)
            ax.set_yticks(positions)
            ax.set_yticklabels(df_sorted["i_definition"])
            ax.set_xlabel("Δt = Lock Epoch - Stable Epoch (lower = faster)", 
                         fontsize=12, fontweight='bold')
            ax.set_title("Lock-in Efficiency Comparison", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "lockin_efficiency_boxplot.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ lockin_efficiency_boxplot.png")
        else:
            print("   ⚠️  No efficiency data")
    else:
        print("   ⚠️  No lockin_efficiency column")
    
    # 3. Early vs late lock-in
    print("\n4.3 Early vs Late Lock-in Rate")
    if "early_lockin_rate" in df_ei.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_plot = df_ei.dropna(subset=["early_lockin_rate"])
        if len(df_plot) > 0:
            df_sorted = df_plot.sort_values("early_lockin_rate", ascending=False)
            bars = ax.barh(df_sorted["i_definition"], df_sorted["early_lockin_rate"],
                          color='indianred', alpha=0.7)
            ax.set_xlabel("Early Lock-in Rate (< 100 epochs) %", fontsize=12, fontweight='bold')
            ax.set_title("Early vs Late Lock-in Rate", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "early_vs_late_lockin.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ early_vs_late_lockin.png")
        else:
            print("   ⚠️  No early lock-in data")
    else:
        print("   ⚠️  No early_lockin_rate column")
    
    # 4. Export CSV
    print("\n4.4 Lock-in Dynamics Metrics Export")
    lockin_cols = ["i_definition", "lock_epoch_mean", "lock_epoch_std", 
                  "lockin_efficiency", "early_lockin_rate"]
    available_cols = [c for c in lockin_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "lockin_dynamics_metrics.csv"), index=False)
    print("   ✅ lockin_dynamics_metrics.csv")
    
    print("\n✅ Lock-in Dynamics Comparison Complete!")


def analyze_quantum_fields(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY 5: Quantum Fields Comparison."""
    print("\n" + "="*70)
    print("CATEGORY 5: QUANTUM FIELDS COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0 or "vacuum_energy_mean" not in df_ei.columns:
        print("⚠️  No quantum fields data available")
        return
    
    # 1. Vacuum energy comparison
    print("\n5.1 Vacuum Energy Density Comparison")
    fig, ax = plt.subplots(figsize=(14, 8))
    df_plot = df_ei.dropna(subset=["vacuum_energy_mean"])
    if len(df_plot) > 0:
        positions = range(len(df_plot))
        ax.barh(positions, df_plot["vacuum_energy_mean"],
               xerr=df_plot["vacuum_energy_std"] if "vacuum_energy_std" in df_plot.columns else None,
               color='violet', alpha=0.7, capsize=5)
        ax.set_yticks(positions)
        ax.set_yticklabels(df_plot["i_definition"])
        ax.set_xlabel("Vacuum Energy Density", fontsize=12, fontweight='bold')
        ax.set_title("Vacuum Energy Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vacuum_energy_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ vacuum_energy_comparison.png")
    
    # 2. Export CSV
    print("\n5.2 Quantum Fields Metrics Export")
    quantum_cols = ["i_definition", "vacuum_energy_mean", "vacuum_energy_std",
                   "zero_point_energy_mean", "quantum_fluctuation_mean"]
    available_cols = [c for c in quantum_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "quantum_fields_metrics.csv"), index=False)
    print("   ✅ quantum_fields_metrics.csv")
    print("\n✅ Quantum Fields Comparison Complete!")


def analyze_entanglement(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY 6: Entanglement Comparison."""
    print("\n" + "="*70)
    print("CATEGORY 6: ENTANGLEMENT COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0 or "entanglement_entropy_mean" not in df_ei.columns:
        print("⚠️  No entanglement data available")
        return
    
    # 1. Entanglement entropy
    print("\n6.1 Entanglement Entropy Comparison")
    fig, ax = plt.subplots(figsize=(14, 8))
    df_plot = df_ei.dropna(subset=["entanglement_entropy_mean"])
    if len(df_plot) > 0:
        positions = range(len(df_plot))
        ax.barh(positions, df_plot["entanglement_entropy_mean"],
               xerr=df_plot["entanglement_entropy_std"] if "entanglement_entropy_std" in df_plot.columns else None,
               color='orchid', alpha=0.7, capsize=5)
        ax.set_yticks(positions)
        ax.set_yticklabels(df_plot["i_definition"])
        ax.set_xlabel("Entanglement Entropy", fontsize=12, fontweight='bold')
        ax.set_title("Entanglement Entropy Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "entanglement_entropy_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ entanglement_entropy_comparison.png")
    
    # 2. Export CSV
    print("\n6.2 Entanglement Metrics Export")
    ent_cols = ["i_definition", "entanglement_entropy_mean", "entanglement_entropy_std", "holographic_entropy_mean"]
    available_cols = [c for c in ent_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "entanglement_metrics.csv"), index=False)
    print("   ✅ entanglement_metrics.csv")
    print("\n✅ Entanglement Comparison Complete!")


def analyze_parameter_sensitivity(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY 7: Parameter Sensitivity Comparison."""
    print("\n" + "="*70)
    print("CATEGORY 7: PARAMETER SENSITIVITY COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0 or "E_sensitivity" not in df_ei.columns:
        print("⚠️  No parameter sensitivity data available")
        return
    
    # 1. Sensitivity heatmap
    print("\n7.1 Parameter Sensitivity Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sens_cols = ["E_sensitivity", "I_sensitivity", "X_sensitivity"]
    available_sens = [c for c in sens_cols if c in df_ei.columns]
    if len(available_sens) > 0:
        df_plot = df_ei[["i_definition"] + available_sens].dropna()
        if len(df_plot) > 0:
            heatmap_data = df_plot.set_index("i_definition")
            sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title("Parameter Sensitivity Heatmap", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "parameter_sensitivity_heatmap.png"), dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ parameter_sensitivity_heatmap.png")
    
    # 2. Export CSV
    print("\n7.2 Sensitivity Metrics Export")
    sens_all_cols = ["i_definition"] + sens_cols
    available_cols = [c for c in sens_all_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "sensitivity_metrics.csv"), index=False)
    print("   ✅ sensitivity_metrics.csv")
    print("\n✅ Parameter Sensitivity Comparison Complete!")


def analyze_topology(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY 9: Topology & Physical Anomalies Comparison."""
    print("\n" + "="*70)
    print("CATEGORY 9: TOPOLOGY & PHYSICAL ANOMALIES COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data available")
        return
    
    # 1. Topological defect rate
    if "topological_defect_rate" in df_ei.columns:
        print("\n9.1 Topological Defect Detection Rate")
        fig, ax = plt.subplots(figsize=(14, 8))
        df_plot = df_ei.dropna(subset=["topological_defect_rate"])
        if len(df_plot) > 0:
            df_sorted = df_plot.sort_values("topological_defect_rate", ascending=False)
            ax.barh(df_sorted["i_definition"], df_sorted["topological_defect_rate"], color='crimson', alpha=0.7)
            ax.set_xlabel("Topological Defect Rate %", fontsize=12, fontweight='bold')
            ax.set_title("Topological Defect Detection Rate", fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "topological_defect_rate.png"), dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ topological_defect_rate.png")
    
    # 2. Export CSV
    print("\n9.2 Topology Metrics Export")
    topo_cols = ["i_definition", "curvature_radius_mean", "topological_defect_rate"]
    available_cols = [c for c in topo_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "topology_metrics.csv"), index=False)
    print("   ✅ topology_metrics.csv")
    print("\n✅ Topology & Physical Anomalies Comparison Complete!")


def analyze_i_definitions_direct(df_metrics: pd.DataFrame, collected_data: Dict, output_dir: str):
    """
    CATEGORY 10: I-Definitions Direct Comparison.
    
    Compares I(E) curves and divergence between I-definitions.
    
    Generates:
    - i_definition_E_dependency.png (I(E) curves overlay)
    - i_definition_divergence_matrix.png (pairwise comparison)
    - i_definitions_metrics.csv
    """
    print("\n" + "="*70)
    print("CATEGORY 10: I-DEFINITIONS DIRECT COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find I_Definitions_Comparison.csv from any run
    i_comp_df = None
    for data in list(collected_data['eonly'].values()) + list(collected_data['ei'].values()):
        if data.get("i_definitions_comparison") is not None:
            i_comp_df = data["i_definitions_comparison"]
            break
    
    if i_comp_df is None or len(i_comp_df) == 0:
        print("⚠️  No I_Definitions_Comparison.csv data available")
        return
    
    # 1. I(E) curves comparison
    print("\n10.1 I-Definition E-Dependency Curves")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    if "E" in i_comp_df.columns:
        i_def_cols = [c for c in i_comp_df.columns if c != "E"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(i_def_cols)))
        
        for idx, col in enumerate(i_def_cols):
            if col in i_comp_df.columns:
                ax.plot(i_comp_df["E"], i_comp_df[col], label=col, 
                       linewidth=2.5, alpha=0.8, color=colors[idx])
        
        ax.set_xlabel("E Parameter (Dark Energy Density)", fontsize=12, fontweight='bold')
        ax.set_ylabel("I Parameter Value", fontsize=12, fontweight='bold')
        ax.set_title("I-Parameter Definitions: E-Dependency Comparison", 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "i_definition_E_dependency.png"),
                   dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ i_definition_E_dependency.png")
    else:
        print("   ⚠️  No E column in I_Definitions_Comparison.csv")
    
    # 2. I-definition divergence matrix (pairwise KL-divergence)
    print("\n10.2 I-Definition Divergence Matrix")
    if len(i_comp_df) > 0:
        i_def_cols = [c for c in i_comp_df.columns if c != "E"]
        
        if len(i_def_cols) > 1:
            # Compute pairwise correlation (proxy for divergence)
            divergence_matrix = np.zeros((len(i_def_cols), len(i_def_cols)))
            
            for i, col1 in enumerate(i_def_cols):
                for j, col2 in enumerate(i_def_cols):
                    if i == j:
                        divergence_matrix[i, j] = 0.0
                    else:
                        # Correlation-based divergence: 1 - |correlation|
                        valid_data = i_comp_df[[col1, col2]].dropna()
                        if len(valid_data) > 1:
                            corr = valid_data[col1].corr(valid_data[col2])
                            divergence_matrix[i, j] = 1.0 - abs(corr)
                        else:
                            divergence_matrix[i, j] = np.nan
            
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(divergence_matrix, cmap='YlOrRd', vmin=0, vmax=1)
            ax.set_xticks(range(len(i_def_cols)))
            ax.set_yticks(range(len(i_def_cols)))
            ax.set_xticklabels(i_def_cols, rotation=45, ha='right')
            ax.set_yticklabels(i_def_cols)
            ax.set_title("I-Definition Divergence Matrix\n(1 - |correlation|)", 
                        fontsize=14, fontweight='bold')
            
            # Add values
            for i in range(len(i_def_cols)):
                for j in range(len(i_def_cols)):
                    if not np.isnan(divergence_matrix[i, j]):
                        text = ax.text(j, i, f'{divergence_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black" if divergence_matrix[i, j] < 0.5 else "white")
            
            plt.colorbar(im, ax=ax, label='Divergence (0=identical, 1=opposite)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "i_definition_divergence_matrix.png"),
                       dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ i_definition_divergence_matrix.png")
        else:
            print("   ⚠️  Need at least 2 I-definitions for divergence matrix")
    
    # 3. Export CSV (summary statistics)
    print("\n10.3 I-Definitions Metrics Export")
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    i_def_cols = ["i_definition", "I_value_mean", "I_value_std", "I_value_range"]
    available_cols = [c for c in i_def_cols if c in df_ei.columns]
    export_df = df_ei[available_cols].copy()
    export_df.to_csv(os.path.join(output_dir, "i_definitions_metrics.csv"), index=False)
    print("   ✅ i_definitions_metrics.csv")
    
    print("\n✅ I-Definitions Direct Comparison Complete!")


def analyze_planck_fit(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY: Planck Fit & Proximity."""
    print("\n" + "="*70)
    print("CATEGORY: PLANCK FIT & PROXIMITY")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_planck = df_metrics[df_metrics["planck_E"].notna()].copy()
    
    if len(df_planck) == 0:
        print("⚠️  No Planck validation data available")
        return
    
    # Scatter plot: Planck E vs I proximity
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_planck["planck_E"], df_planck["planck_I"], c=df_planck["planck_chi2_reduced"], cmap="viridis", s=120)
    ax.scatter([PLANCK_TARGET_E], [PLANCK_TARGET_I], color="red", marker="*", s=250, label="Planck Target")
    for _, row in df_planck.iterrows():
        ax.annotate(row["i_definition"], (row["planck_E"], row["planck_I"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel("Planck Best-fit E", fontweight="bold")
    ax.set_ylabel("Planck Best-fit I", fontweight="bold")
    ax.set_title("Planck Best-fit Parameters vs Target", fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("χ² (reduced)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "planck_EI_scatter.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ planck_EI_scatter.png")
    
    # Bar chart: χ² per I-definition
    fig, ax = plt.subplots(figsize=(12, 7))
    df_sorted = df_planck.sort_values("planck_chi2_reduced")
    ax.bar(df_sorted["i_definition"], df_sorted["planck_chi2_reduced"], color="steelblue", alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", label="Perfect Fit (χ²=1)")
    ax.set_ylabel("χ² Reduced", fontweight="bold")
    ax.set_title("Planck χ² per I-definition", fontweight="bold")
    ax.tick_params(axis='x', rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "planck_chi2_bar.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ planck_chi2_bar.png")
    
    # Export CSV
    export_cols = [
        "i_definition", "planck_E", "planck_I", "planck_alpha",
        "planck_chi2_total", "planck_chi2_reduced", "planck_score",
        "planck_validation_chi2_mean", "planck_validation_ell_span"
    ]
    df_planck[export_cols].to_csv(os.path.join(output_dir, "planck_fit_metrics.csv"), index=False)
    print("   ✅ planck_fit_metrics.csv")
    print("\n✅ Planck Fit Analysis Complete!")


def analyze_life_top_universes(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY: Life Compatibility & Top Universes."""
    print("\n" + "="*70)
    print("CATEGORY: LIFE COMPATIBILITY & TOP UNIVERSES")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_life = df_metrics[df_metrics["life_score_json"].notna()].copy()
    
    if len(df_life) == 0:
        print("⚠️  No life compatibility summaries available")
        return
    
    df_life = df_life.sort_values("life_score_json", ascending=False)
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(df_life["i_definition"], df_life["life_score_json"], color="seagreen", alpha=0.8)
    ax.set_ylabel("Life Compatibility Score", fontweight="bold")
    ax.set_title("Life Compatibility per I-definition", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "life_compatibility_bar.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ life_compatibility_bar.png")
    
    # Scatter: life score vs information richness
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df_life["life_score_json"], df_life["information_richness_json"], s=120, color="purple", alpha=0.8)
    for _, row in df_life.iterrows():
        ax.annotate(row["i_definition"], (row["life_score_json"], row["information_richness_json"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel("Life Compatibility Score", fontweight="bold")
    ax.set_ylabel("Information Richness", fontweight="bold")
    ax.set_title("Life vs Information Richness", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "life_vs_information.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ life_vs_information.png")
    
    # Export CSV
    export_cols = [
        "i_definition", "life_score_json", "complexity_score_json",
        "information_richness_json", "life_planck_component",
        "life_stability_component", "life_goldilocks_component",
        "top_universe_seed", "top_universe_lock_epoch", "top_universe_I"
    ]
    df_life[export_cols].to_csv(os.path.join(output_dir, "life_top_universes_metrics.csv"), index=False)
    print("   ✅ life_top_universes_metrics.csv")
    print("\n✅ Life Compatibility Analysis Complete!")


def analyze_entropy_volatility(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY: Entropy Volatility & Stability Sweeps."""
    print("\n" + "="*70)
    print("CATEGORY: ENTROPY VOLATILITY & STABILITY SENSITIVITY")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_entropy = df_metrics[df_metrics["entropy_volatility_global_mean"].notna()].copy()
    
    if len(df_entropy) == 0:
        print("⚠️  No entropy volatility data available")
        return
    
    # Scatter: entropy volatility vs life score
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df_entropy["entropy_volatility_global_mean"], df_entropy["life_score_json"], c=df_entropy["stability_eps_slope"], cmap="coolwarm", s=120)
    ax.set_xlabel("Entropy Volatility (mean)", fontweight="bold")
    ax.set_ylabel("Life Score", fontweight="bold")
    ax.set_title("Entropy Volatility vs Life Compatibility", fontweight="bold")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Stability Sweep Slope")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_vs_life.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ entropy_vs_life.png")
    
    # Line plot: stability slope per I-definition
    fig, ax = plt.subplots(figsize=(12, 6))
    df_slope = df_metrics[df_metrics["stability_eps_slope"].notna()].sort_values("stability_eps_slope", ascending=False)
    x_positions = list(range(len(df_slope)))
    ax.plot(x_positions, df_slope["stability_eps_slope"], marker="o")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_slope["i_definition"], rotation=45, ha="right")
    ax.set_ylabel("Stability Sweep Slope (log ε)", fontweight="bold")
    ax.set_title("Sensitivity of Stability to I (ε-sweep)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stability_sweep_slope.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ stability_sweep_slope.png")
    
    df_entropy[[
        "i_definition", "entropy_volatility_global_mean", "entropy_volatility_global_std",
        "entropy_volatility_max", "stability_eps_slope", "stability_zero_baseline"
    ]].to_csv(os.path.join(output_dir, "entropy_volatility_metrics.csv"), index=False)
    print("   ✅ entropy_volatility_metrics.csv")
    print("\n✅ Entropy Volatility Analysis Complete!")


def analyze_physical_anomalies(df_metrics: pd.DataFrame, output_dir: str):
    """CATEGORY: Advanced Physical Anomalies."""
    print("\n" + "="*70)
    print("CATEGORY: ADVANCED PHYSICAL ANOMALIES")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_anom = df_metrics[df_metrics["advanced_anomaly_sigma_mean"].notna()].copy()
    
    if len(df_anom) == 0:
        print("⚠️  No advanced anomaly data available")
        return
    
    df_anom = df_anom.sort_values("advanced_anomaly_sigma_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(df_anom["i_definition"], df_anom["advanced_anomaly_sigma_mean"], color="darkorange", alpha=0.8)
    ax.set_ylabel("⟨σ deviation⟩", fontweight="bold")
    ax.set_title("Advanced Anomaly Significance", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "advanced_anomaly_bar.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print("   ✅ advanced_anomaly_bar.png")
    
    df_anom[[
        "i_definition", "advanced_anomaly_sigma_mean", "advanced_anomaly_sigma_max",
        "physical_anomaly_count", "cmb_gaussianity_p_mean", "cmb_anisotropy_index_mean"
    ]].to_csv(os.path.join(output_dir, "advanced_anomaly_metrics.csv"), index=False)
    print("   ✅ advanced_anomaly_metrics.csv")
    print("\n✅ Physical Anomalies Analysis Complete!")


def analyze_statistical_finetuning(df_metrics: pd.DataFrame, collected_data: Dict, output_dir: str):
    """
    CATEGORY 8: Statistical Finetuning Comparison.
    
    Analyzes how well each I-definition produces finetuned universes.
    Currently based on Planck fit and parameter precision.
    
    Generates:
    - finetuning_comparison.png
    - finetuning_metrics.csv
    """
    print("\n" + "="*70)
    print("CATEGORY 8: STATISTICAL FINETUNING COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    if len(df_ei) == 0:
        print("⚠️  No E+I data available")
        return
    
    # Compute finetuning score based on available metrics
    # Finetuning = How well parameters match expected/observed values
    
    finetuning_components = []
    
    # 1. Friedmann finetuning (deviation from Planck)
    if "age_deviation_from_planck" in df_ei.columns:
        age_finetuning = 100 / (1 + df_ei["age_deviation_from_planck"])
        finetuning_components.append(age_finetuning)
    
    if "H0_deviation_from_planck" in df_ei.columns:
        h0_finetuning = 100 / (1 + df_ei["H0_deviation_from_planck"])
        finetuning_components.append(h0_finetuning)
    
    # 2. Goldilocks precision (well-tuned parameter space)
    if "X_peak_uncertainty" in df_ei.columns and "X_peak" in df_ei.columns:
        rel_unc = df_ei["X_peak_uncertainty"] / df_ei["X_peak"].replace(0, 1)
        goldilocks_finetuning = 100 / (1 + rel_unc * 100)
        finetuning_components.append(goldilocks_finetuning)
    
    if len(finetuning_components) == 0:
        print("⚠️  No finetuning data available")
        return
    
    # Average finetuning score
    finetuning_score = np.mean(finetuning_components, axis=0)
    df_ei_plot = df_ei.copy()
    df_ei_plot["finetuning_score"] = finetuning_score
    
    # 1. Finetuning comparison plot
    print("\n8.1 Statistical Finetuning Comparison")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_sorted = df_ei_plot.sort_values("finetuning_score", ascending=False)
    bars = ax.barh(df_sorted["i_definition"], df_sorted["finetuning_score"],
                  color='steelblue', alpha=0.7)
    ax.set_xlabel("Finetuning Score (0-100, higher=better)", fontsize=12, fontweight='bold')
    ax.set_title("Statistical Finetuning Comparison\n(Planck consistency + Goldilocks precision)", 
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "finetuning_comparison.png"),
               dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ finetuning_comparison.png")
    
    # 2. Export CSV
    print("\n8.2 Finetuning Metrics Export")
    finetuning_cols = ["i_definition", "finetuning_score", "age_deviation_from_planck", 
                      "H0_deviation_from_planck", "X_peak_uncertainty"]
    available_cols = [c for c in finetuning_cols if c in df_ei_plot.columns or c in df_ei.columns]
    export_df = df_ei_plot[available_cols].copy() if "finetuning_score" in df_ei_plot.columns else df_ei[[c for c in available_cols if c in df_ei.columns]].copy()
    export_df.to_csv(os.path.join(output_dir, "finetuning_metrics.csv"), index=False)
    print("   ✅ finetuning_metrics.csv")
    
    print("\n✅ Statistical Finetuning Comparison Complete!")


def generate_detailed_metrics(df_metrics: pd.DataFrame, output_dir: str):
    """
    PHASE 3C: Generate detailed statistical metrics analysis.
    
    Comprehensive statistical analysis including:
    - Full metrics export (CSV with all runs and all metrics)
    - Correlation matrix (identify interdependencies)
    - Distribution analysis (box plots showing quartiles, outliers)
    - Comparison against E-only baseline (if available)
    
    Generates:
    - all_runs_metrics.csv (comprehensive table)
    - correlation_matrix.png (heatmap of metric correlations)
    - distributions_boxplot.png (stability, lock-in, X_peak, uncertainty distributions)
    """
    print("\n" + "="*70)
    print("PHASE 3C: DETAILED STATISTICAL METRICS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save comprehensive metrics CSV
    print("\n3.1 Comprehensive Metrics Table")
    df_metrics.to_csv(os.path.join(output_dir, "all_runs_metrics.csv"), index=False)
    print("   ✅ all_runs_metrics.csv")
    
    # 2. Correlation matrix (E+I only)
    print("\n3.2 Correlation Matrix")
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    if len(df_ei) > 2:
        numeric_cols = ["stable_percent", "lockin_percent", "X_peak", "X_peak_uncertainty", 
                       "goldilocks_width", "chi_squared_reduced"]
        df_corr = df_ei[numeric_cols].dropna()
        
        if len(df_corr) > 2:
            corr_matrix = df_corr.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Correlation Matrix (E+I runs)", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print("   ✅ correlation_matrix.png")
        else:
            print("   ⚠️  Insufficient data for correlation matrix")
    
    # 3. Box plots
    print("\n3.3 Distribution Box Plots")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    
    # Stable %
    axes[0, 0].boxplot([df_ei["stable_percent"]], labels=['E+I'])
    if len(df_metrics[df_metrics["run_type"] == "E-only"]) > 0:
        eonly_stable = df_metrics[df_metrics["run_type"] == "E-only"]["stable_percent"].iloc[0]
        axes[0, 0].axhline(eonly_stable, color='red', linestyle='--', label='E-only baseline')
        axes[0, 0].legend()
    axes[0, 0].set_ylabel("Stable %", fontweight='bold')
    axes[0, 0].set_title("Stability Rate Distribution", fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Lock-in %
    axes[0, 1].boxplot([df_ei["lockin_percent"]], labels=['E+I'])
    if len(df_metrics[df_metrics["run_type"] == "E-only"]) > 0:
        eonly_lockin = df_metrics[df_metrics["run_type"] == "E-only"]["lockin_percent"].iloc[0]
        axes[0, 1].axhline(eonly_lockin, color='red', linestyle='--', label='E-only baseline')
        axes[0, 1].legend()
    axes[0, 1].set_ylabel("Lock-in %", fontweight='bold')
    axes[0, 1].set_title("Lock-in Rate Distribution", fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # X_peak
    axes[1, 0].boxplot([df_ei["X_peak"]], labels=['E+I'])
    axes[1, 0].set_ylabel("X_peak", fontweight='bold')
    axes[1, 0].set_title("Goldilocks Peak Distribution", fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # X_peak_uncertainty
    axes[1, 1].boxplot([df_ei["X_peak_uncertainty"]], labels=['E+I'])
    axes[1, 1].set_ylabel("X_peak uncertainty (σ)", fontweight='bold')
    axes[1, 1].set_title("Peak Uncertainty Distribution", fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distributions_boxplot.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ distributions_boxplot.png")
    
    print("\n✅ Detailed Metrics Complete!")


def generate_advanced_visualizations(df_metrics: pd.DataFrame, output_dir: str):
    """
    PHASE 4: Generate advanced multi-dimensional visualizations.
    
    Creates publication-quality visualizations for comparative analysis:
    - Radar chart: Multi-metric comparison on normalized 0-100 scale (top 5 models)
    - Performance heatmap: Color-coded strength/weakness matrix
    - Scatter plots: Goldilocks peak vs stability rate relationships
    
    Generates:
    - radar_chart_top5.png (spider plot showing top 5 I-definitions)
    - heatmap_performance.png (green=good, red=poor performance matrix)
    - scatter_X_peak_vs_stability.png (peak position vs stability correlation)
    """
    print("\n" + "="*70)
    print("PHASE 4: ADVANCED VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    # 1. Radar chart
    print("\n4.1 Radar Chart (Spider Plot)")
    
    # Normalize metrics to 0-1 scale for radar chart
    df_radar = df_ei[["i_definition", "stable_percent", "lockin_percent"]].copy()
    df_radar["precision"] = 100 - (df_ei["X_peak_uncertainty"] / df_ei["X_peak"] * 100).clip(0, 100)  # Higher is better
    
    # Add chi-squared if available (invert: lower is better → higher score)
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_min = df_ei["chi_squared_reduced"].min()
        chi_max = df_ei["chi_squared_reduced"].max()
        if chi_max > chi_min:
            df_radar["planck_fit"] = 100 * (1 - (df_ei["chi_squared_reduced"] - chi_min) / (chi_max - chi_min))
        else:
            df_radar["planck_fit"] = 50.0
    else:
        df_radar["planck_fit"] = 50.0  # Neutral if no data
    
    # Select top 5 models for clarity
    df_radar_top5 = df_radar.nlargest(5, "stable_percent")
    
    categories = ['Stability', 'Lock-in', 'Precision', 'Planck Fit']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_radar_top5)))
    
    for i, (idx, row) in enumerate(df_radar_top5.iterrows()):
        values = [row["stable_percent"], row["lockin_percent"], row["precision"], row["planck_fit"]]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row["i_definition"], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title("Top 5 I-Definitions: Multi-Metric Radar Chart", fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart_top5.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ radar_chart_top5.png")
    
    # 2. Performance heatmap
    print("\n4.2 Performance Heatmap")
    
    # Normalize all metrics to 0-100 scale
    df_heatmap = pd.DataFrame()
    df_heatmap['I-Definition'] = df_ei["i_definition"]
    df_heatmap['Stability'] = df_ei["stable_percent"]
    df_heatmap['Lock-in'] = df_ei["lockin_percent"]
    df_heatmap['Precision'] = 100 - (df_ei["X_peak_uncertainty"] / df_ei["X_peak"] * 100).clip(0, 100)
    
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_min = df_ei["chi_squared_reduced"].min()
        chi_max = df_ei["chi_squared_reduced"].max()
        if chi_max > chi_min:
            df_heatmap['Planck Fit'] = 100 * (1 - (df_ei["chi_squared_reduced"] - chi_min) / (chi_max - chi_min))
        else:
            df_heatmap['Planck Fit'] = 50.0
    
    df_heatmap = df_heatmap.set_index('I-Definition')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_heatmap.T, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
               square=False, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Performance Heatmap (0-100 scale, higher is better)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metrics", fontsize=12, fontweight='bold')
    ax.set_xlabel("I-Definitions", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_performance.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ heatmap_performance.png")
    
    # 3. Scatter: X_peak vs Stability
    print("\n4.3 Scatter: X_peak vs Stability Rate")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, row in df_ei.iterrows():
        ax.errorbar(row["X_peak"], row["stable_percent"], 
                   xerr=row["X_peak_uncertainty"]*1.96, 
                   fmt='o', markersize=10, capsize=5, capthick=2, linewidth=2,
                   label=row["i_definition"], alpha=0.7)
    
    ax.set_xlabel("X_peak (Goldilocks Peak)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Stability Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Goldilocks Peak vs Stability Rate", fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_X_peak_vs_stability.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ scatter_X_peak_vs_stability.png")
    
    print("\n✅ Advanced Visualizations Complete!")


def generate_complexity_analysis(df_metrics: pd.DataFrame, output_dir: str):
    """
    PHASE 4B: Generate complexity and life-compatibility analysis.
    
    Critical insight: More stable ≠ Better!
    - E-only may be stable but chaotic (no complexity)
    - E+I may be less stable but more complex (life-compatible)
    
    Generates:
    - complexity_vs_stability.png (scatter: complexity score vs stability %)
    - life_compatibility_comparison.png (bar chart across I-definitions)
    - complexity_heatmap.png (complexity components breakdown)
    - dual_ranking_comparison.png (stability-based vs complexity-based ranking)
    """
    print("\n" + "="*70)
    print("PHASE 4B: COMPLEXITY & LIFE-COMPATIBILITY ANALYSIS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Complexity vs Stability scatter
    print("\n4B.1 Complexity vs Stability Scatter")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, row in df_metrics.iterrows():
        color = 'red' if row['run_type'] == 'E-only' else 'blue'
        marker = 's' if row['run_type'] == 'E-only' else 'o'
        size = 200 if row['run_type'] == 'E-only' else 100
        
        ax.scatter(row['stable_percent'], row['complexity_score'], 
                  s=size, alpha=0.7, color=color, marker=marker,
                  edgecolors='black', linewidths=2,
                  label=row['i_definition'] if idx < 10 else "")
    
    ax.set_xlabel("Stability Rate (%)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Complexity Score (0-100)", fontsize=14, fontweight='bold')
    ax.set_title("CRITICAL INSIGHT: Stability ≠ Complexity\n(E-only = stable chaos, E+I = complex structure)", 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Add quadrants
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(df_metrics['stable_percent'].median(), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complexity_vs_stability.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ complexity_vs_stability.png")
    
    # 2. Life-compatibility comparison
    print("\n4B.2 Life-Compatibility Comparison")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_sorted = df_metrics.sort_values("life_compatibility_score", ascending=False)
    colors = ['red' if rt == 'E-only' else 'green' for rt in df_sorted['run_type']]
    
    ax.barh(df_sorted['i_definition'], df_sorted['life_compatibility_score'], 
           color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(50, color='orange', linestyle='--', linewidth=2, label='Neutral (50)')
    ax.set_xlabel("Life-Compatibility Score (0-100)", fontsize=14, fontweight='bold')
    ax.set_title("Life-Compatibility: Structure Formation Potential\n(Higher = Better for complexity/life)", 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "life_compatibility_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ life_compatibility_comparison.png")
    
    # 3. Complexity components heatmap
    print("\n4B.3 Complexity Components Breakdown")
    
    # Calculate individual components for each run
    complexity_breakdown = []
    for idx, row in df_metrics.iterrows():
        complexity_breakdown.append({
            'I-Definition': row['i_definition'],
            'Lock-in Rate': min(row['lockin_percent'] * 2, 100),  # Normalized
            'Goldilocks Precision': 50,  # Placeholder (would need detailed calc)
            'Info Richness': row['information_richness'],
        })
    
    df_breakdown = pd.DataFrame(complexity_breakdown).set_index('I-Definition')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_breakdown.T, annot=True, fmt='.1f', cmap='YlGnBu', 
               square=False, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Complexity Components Breakdown (0-100 scale)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Components", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complexity_heatmap.png"), dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print("   ✅ complexity_heatmap.png")
    
    # 4. Triple ranking comparison (Stability vs Complexity vs Physical-Laws)
    print("\n4B.4 Triple Ranking Comparison")
    
    df_ei = df_metrics[df_metrics['run_type'] == 'E+I'].copy()
    
    if len(df_ei) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Stability-based ranking
        df_stab_rank = df_ei.sort_values("stable_percent", ascending=False).reset_index(drop=True)
        df_stab_rank['rank'] = range(1, len(df_stab_rank) + 1)
        
        axes[0].barh(df_stab_rank['i_definition'], df_stab_rank['stable_percent'], 
                    color='blue', alpha=0.7)
        axes[0].set_xlabel("Stability Rate (%)", fontsize=12, fontweight='bold')
        axes[0].set_title("RANKING 1: Stability-Focused\n(Traditional Approach)", 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Complexity-based ranking
        df_comp_rank = df_ei.sort_values("complexity_score", ascending=False).reset_index(drop=True)
        df_comp_rank['rank'] = range(1, len(df_comp_rank) + 1)
        
        axes[1].barh(df_comp_rank['i_definition'], df_comp_rank['complexity_score'], 
                    color='green', alpha=0.7)
        axes[1].set_xlabel("Complexity Score (0-100)", fontsize=12, fontweight='bold')
        axes[1].set_title("RANKING 2: Complexity-Focused\n(TQE-Consistent Approach)", 
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dual_ranking_comparison.png"), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("   ✅ dual_ranking_comparison.png")
    
    print("\n✅ Complexity Analysis Complete!")


def select_best_model(df_metrics: pd.DataFrame, output_dir: str):
    """
    PHASE 5: Best model selection using weighted multi-metric ranking.
    
    Applies configurable weights to normalized metrics (0-100 scale) to identify
    the best-performing I-definition for TQE simulations.
    
    Ranking methodology:
    - Stability rate: 30% (higher is better)
    - Lock-in rate: 20% (higher is better)
    - Planck χ² fit: 20% (lower is better, inverted for scoring)
    - Goldilocks precision: 15% (lower uncertainty is better, inverted)
    - CMB anomaly match: 10% (anomaly detection rates)
    - Bayesian efficiency: 5% (GP performance)
    
    Generates:
    - weighted_ranking.csv (all models with component scores)
    - top_3_models.json (top 3 ranked models with full metrics)
    - recommendation_report.md (scientific justification and usage guide)
    """
    print("\n" + "="*70)
    print("PHASE 5: BEST MODEL SELECTION & TRIPLE RANKING")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    # ==================================================================
    # TRIPLE RANKING SYSTEM: STABILITY vs COMPLEXITY vs PHYSICAL-LAWS
    # ==================================================================
    
    # RANKING 1: STABILITY-FOCUSED (Traditional)
    df_stability_rank = pd.DataFrame()
    df_stability_rank['i_definition'] = df_ei["i_definition"]
    
    # Normalize components
    df_stability_rank['stability_score'] = (df_ei["stable_percent"] / df_ei["stable_percent"].max()) * 100 if df_ei["stable_percent"].max() > 0 else 50.0
    
    if df_ei["lockin_percent"].max() > 0:
        df_stability_rank['lockin_score'] = (df_ei["lockin_percent"] / df_ei["lockin_percent"].max()) * 100
    else:
        df_stability_rank['lockin_score'] = 50.0
    
    relative_uncertainty = df_ei["X_peak_uncertainty"] / df_ei["X_peak"]
    df_stability_rank['precision_score'] = (1 - (relative_uncertainty / relative_uncertainty.max()).clip(0, 1)) * 100
    
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_values = df_ei["chi_squared_reduced"].fillna(df_ei["chi_squared_reduced"].max() * 2)
        df_stability_rank['planck_score'] = (1 - (chi_values / chi_values.max()).clip(0, 1)) * 100
    else:
        df_stability_rank['planck_score'] = 50.0
    
    df_stability_rank['anomaly_score'] = 50.0  # Placeholder
    df_stability_rank['bayesian_score'] = 50.0  # Placeholder
    
    # Calculate stability-weighted score
    df_stability_rank['stability_total_score'] = (
        df_stability_rank['stability_score'] * RANKING_WEIGHTS_STABILITY['stability_rate'] +
        df_stability_rank['lockin_score'] * RANKING_WEIGHTS_STABILITY['lockin_rate'] +
        df_stability_rank['planck_score'] * RANKING_WEIGHTS_STABILITY['planck_chi2_fit'] +
        df_stability_rank['precision_score'] * RANKING_WEIGHTS_STABILITY['goldilocks_precision'] +
        df_stability_rank['anomaly_score'] * RANKING_WEIGHTS_STABILITY['cmb_anomaly_match'] +
        df_stability_rank['bayesian_score'] * RANKING_WEIGHTS_STABILITY['bayesian_efficiency']
    )
    
    df_stability_rank = df_stability_rank.sort_values('stability_total_score', ascending=False)
    
    # RANKING 2: COMPLEXITY-FOCUSED (TQE-Consistent)
    df_complexity_rank = pd.DataFrame()
    df_complexity_rank['i_definition'] = df_ei["i_definition"]
    
    # Use already-computed advanced scores
    df_complexity_rank['complexity_score'] = df_ei["complexity_score"]
    df_complexity_rank['life_compatibility_score'] = df_ei["life_compatibility_score"]
    df_complexity_rank['information_richness'] = df_ei["information_richness"]
    
    # Stability quality (not quantity!)
    df_complexity_rank['stability_quality'] = df_ei["lockin_percent"] * 2  # Lock-in among total
    
    # Observational match
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_values = df_ei["chi_squared_reduced"].fillna(df_ei["chi_squared_reduced"].max() * 2)
        df_complexity_rank['observational_score'] = (1 - (chi_values / chi_values.max()).clip(0, 1)) * 100
    else:
        df_complexity_rank['observational_score'] = 50.0
    
    # Calculate complexity-weighted score
    df_complexity_rank['complexity_total_score'] = (
        df_complexity_rank['complexity_score'] * RANKING_WEIGHTS_COMPLEXITY['complexity_score'] +
        df_complexity_rank['life_compatibility_score'] * RANKING_WEIGHTS_COMPLEXITY['life_compatibility'] +
        df_complexity_rank['information_richness'] * RANKING_WEIGHTS_COMPLEXITY['information_richness'] +
        df_complexity_rank['stability_quality'] * RANKING_WEIGHTS_COMPLEXITY['stability_quality'] +
        df_complexity_rank['observational_score'] * RANKING_WEIGHTS_COMPLEXITY['observational_match']
    )
    
    df_complexity_rank = df_complexity_rank.sort_values('complexity_total_score', ascending=False)
    
    # RANKING 3: PHYSICAL-LAWS-FOCUSED
    df_physical_rank = pd.DataFrame()
    df_physical_rank['i_definition'] = df_ei["i_definition"]
    
    # 1. Emergent laws quality
    if "power_law_exponent" in df_ei.columns and "n_phase_transitions" in df_ei.columns:
        # Normalize power-law exponent closeness to 1.0 (linear)
        power_law_quality = 100 / (1 + abs(df_ei["power_law_exponent"].fillna(1.0) - 1.0))
        # Normalize phase transitions
        phase_trans_quality = (df_ei["n_phase_transitions"].fillna(0) / df_ei["n_phase_transitions"].max()) * 100 if df_ei["n_phase_transitions"].max() > 0 else 50.0
        df_physical_rank['emergent_laws_quality'] = (power_law_quality + phase_trans_quality) / 2
    else:
        df_physical_rank['emergent_laws_quality'] = 50.0
    
    # 2. Friedmann consistency
    if "age_deviation_from_planck" in df_ei.columns and "H0_deviation_from_planck" in df_ei.columns:
        age_consistency = 100 / (1 + df_ei["age_deviation_from_planck"].fillna(10))
        h0_consistency = 100 / (1 + df_ei["H0_deviation_from_planck"].fillna(10))
        df_physical_rank['friedmann_consistency'] = (age_consistency + h0_consistency) / 2
    else:
        df_physical_rank['friedmann_consistency'] = 50.0
    
    # 3. CMB anomaly match
    if "n_coldspots_mean" in df_ei.columns:
        # Normalize cold spots (Planck ~1-2 major cold spots)
        coldspot_match = 100 / (1 + abs(df_ei["n_coldspots_mean"].fillna(0) - 1.5))
        df_physical_rank['cmb_anomaly_match'] = coldspot_match
    else:
        df_physical_rank['cmb_anomaly_match'] = 50.0
    
    # 4. Lock-in efficiency (fast, decisive)
    if "lockin_efficiency" in df_ei.columns:
        # Lower efficiency = faster lock-in = better
        eff_values = df_ei["lockin_efficiency"].fillna(df_ei["lockin_efficiency"].max() * 2)
        df_physical_rank['lockin_efficiency'] = (1 - (eff_values / eff_values.max()).clip(0, 1)) * 100
    else:
        df_physical_rank['lockin_efficiency'] = df_ei["lockin_percent"]  # Fallback to lock-in rate
    
    # 5. Quantum field realism
    if "vacuum_energy_mean" in df_ei.columns:
        # Normalize vacuum energy (expect ~E value)
        df_physical_rank['quantum_field_realism'] = 50.0  # Placeholder (complex calculation)
    else:
        df_physical_rank['quantum_field_realism'] = 50.0
    
    # Calculate physical-laws-weighted score
    df_physical_rank['physical_laws_total_score'] = (
        df_physical_rank['emergent_laws_quality'] * RANKING_WEIGHTS_PHYSICAL_LAWS['emergent_laws_quality'] +
        df_physical_rank['friedmann_consistency'] * RANKING_WEIGHTS_PHYSICAL_LAWS['friedmann_consistency'] +
        df_physical_rank['cmb_anomaly_match'] * RANKING_WEIGHTS_PHYSICAL_LAWS['cmb_anomaly_match'] +
        df_physical_rank['lockin_efficiency'] * RANKING_WEIGHTS_PHYSICAL_LAWS['lockin_efficiency'] +
        df_physical_rank['quantum_field_realism'] * RANKING_WEIGHTS_PHYSICAL_LAWS['quantum_field_realism']
    )
    
    df_physical_rank = df_physical_rank.sort_values('physical_laws_total_score', ascending=False)
    
    # Merge all THREE rankings for comprehensive view
    df_scores = df_stability_rank.copy()
    df_scores['complexity_total_score'] = df_complexity_rank.set_index('i_definition').loc[df_scores['i_definition'], 'complexity_total_score'].values
    df_scores['physical_laws_total_score'] = df_physical_rank.set_index('i_definition').loc[df_scores['i_definition'], 'physical_laws_total_score'].values
    
    # Save weighted ranking
    print("\n5.1 Weighted Ranking")
    df_scores.to_csv(os.path.join(output_dir, "weighted_ranking.csv"), index=False)
    print("   ✅ weighted_ranking.csv")
    
    # Save all THREE rankings separately
    print("\n5.2 Triple Ranking CSVs")
    df_stability_rank.to_csv(os.path.join(output_dir, "ranking_stability_focused.csv"), index=False)
    print("   ✅ ranking_stability_focused.csv")
    
    df_complexity_rank.to_csv(os.path.join(output_dir, "ranking_complexity_focused.csv"), index=False)
    print("   ✅ ranking_complexity_focused.csv")
    
    df_physical_rank.to_csv(os.path.join(output_dir, "ranking_physical_laws_focused.csv"), index=False)
    print("   ✅ ranking_physical_laws_focused.csv")
    
    # Top 3 models for each system
    print("\n5.3 Top 3 Models (Triple Rankings)")
    top_3_stability = df_stability_rank.head(3).to_dict('records')
    top_3_complexity = df_complexity_rank.head(3).to_dict('records')
    top_3_physical = df_physical_rank.head(3).to_dict('records')
    
    top_models = {
        "stability_focused": top_3_stability,
        "complexity_focused": top_3_complexity,
        "physical_laws_focused": top_3_physical
    }
    
    with open(os.path.join(output_dir, "top_3_models_triple.json"), 'w') as f:
        json.dump(top_models, f, indent=2)
    print("   ✅ top_3_models_triple.json")
    
    # Triple Recommendation Report
    print("\n5.4 Triple Ranking Recommendation Report")
    report = []
    report.append("# TQE ANALYSIS PIPELINE v4.2.0 PRO - TRIPLE RANKING REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("## CRITICAL INSIGHT: THREE RANKING PERSPECTIVES")
    report.append("")
    report.append("**Different goals require different I-definitions!**")
    report.append("")
    report.append("- **Stability-Focused**: Maximizes stable universe percentage")
    report.append("- **Complexity-Focused**: Maximizes structural complexity and life-compatibility")
    report.append("- **Physical-Laws-Focused**: Maximizes observational realism (Planck, CMB, emergent laws)")
    report.append("")
    report.append("For TQE theory validation, PHYSICAL-LAWS RANKING is most observationally consistent!")
    report.append("")
    report.append("=" * 70)
    report.append("")
    
    # RANKING 1: STABILITY-FOCUSED
    report.append("## RANKING 1: STABILITY-FOCUSED (Traditional Approach)")
    report.append("")
    report.append("### Methodology")
    report.append(f"- Stability Rate: {RANKING_WEIGHTS_STABILITY['stability_rate']*100:.0f}%")
    report.append(f"- Lock-in Rate: {RANKING_WEIGHTS_STABILITY['lockin_rate']*100:.0f}%")
    report.append(f"- Planck χ² Fit: {RANKING_WEIGHTS_STABILITY['planck_chi2_fit']*100:.0f}%")
    report.append(f"- Goldilocks Precision: {RANKING_WEIGHTS_STABILITY['goldilocks_precision']*100:.0f}%")
    report.append(f"- CMB Anomaly: {RANKING_WEIGHTS_STABILITY['cmb_anomaly_match']*100:.0f}%")
    report.append(f"- Bayesian Efficiency: {RANKING_WEIGHTS_STABILITY['bayesian_efficiency']*100:.0f}%")
    report.append("")
    report.append("### Top 3 Models")
    report.append("")
    
    for rank, model in enumerate(top_3_stability, 1):
        i_def = model['i_definition']
        score = model['stability_total_score']
        orig = df_ei[df_ei['i_definition'] == i_def].iloc[0]
        
        report.append(f"**{rank}. {i_def.upper()}** (Score: {score:.2f}/100)")
        report.append(f"   - Stable: {orig['stable_percent']:.2f}%, Lock-in: {orig['lockin_percent']:.2f}%")
        report.append("")
    
    report.append("=" * 70)
    report.append("")
    
    # RANKING 2: COMPLEXITY-FOCUSED
    report.append("## RANKING 2: COMPLEXITY-FOCUSED (TQE-Consistent Approach)")
    report.append("")
    report.append("### Methodology")
    report.append(f"- Complexity Score: {RANKING_WEIGHTS_COMPLEXITY['complexity_score']*100:.0f}%")
    report.append(f"- Life-Compatibility: {RANKING_WEIGHTS_COMPLEXITY['life_compatibility']*100:.0f}%")
    report.append(f"- Information Richness: {RANKING_WEIGHTS_COMPLEXITY['information_richness']*100:.0f}%")
    report.append(f"- Stability Quality: {RANKING_WEIGHTS_COMPLEXITY['stability_quality']*100:.0f}%")
    report.append(f"- Observational Match: {RANKING_WEIGHTS_COMPLEXITY['observational_match']*100:.0f}%")
    report.append("")
    report.append("### Top 3 Models")
    report.append("")
    
    for rank, model in enumerate(top_3_complexity, 1):
        i_def = model['i_definition']
        score = model['complexity_total_score']
        orig = df_ei[df_ei['i_definition'] == i_def].iloc[0]
        
        report.append(f"**{rank}. {i_def.upper()}** (Score: {score:.2f}/100)")
        report.append(f"   - Complexity: {orig['complexity_score']:.2f}, Life: {orig['life_compatibility_score']:.2f}")
        report.append(f"   - Stable: {orig['stable_percent']:.2f}%, Lock-in: {orig['lockin_percent']:.2f}%")
        report.append("")
    
    report.append("=" * 70)
    report.append("")
    
    # RANKING 3: PHYSICAL-LAWS-FOCUSED
    report.append("## RANKING 3: PHYSICAL-LAWS-FOCUSED (Observational Realism)")
    report.append("")
    report.append("### Methodology")
    report.append(f"- Emergent Laws Quality: {RANKING_WEIGHTS_PHYSICAL_LAWS['emergent_laws_quality']*100:.0f}%")
    report.append(f"- Friedmann Consistency: {RANKING_WEIGHTS_PHYSICAL_LAWS['friedmann_consistency']*100:.0f}%")
    report.append(f"- CMB Anomaly Match: {RANKING_WEIGHTS_PHYSICAL_LAWS['cmb_anomaly_match']*100:.0f}%")
    report.append(f"- Lock-in Efficiency: {RANKING_WEIGHTS_PHYSICAL_LAWS['lockin_efficiency']*100:.0f}%")
    report.append(f"- Quantum Field Realism: {RANKING_WEIGHTS_PHYSICAL_LAWS['quantum_field_realism']*100:.0f}%")
    report.append("")
    report.append("### Top 3 Models")
    report.append("")
    
    for rank, model in enumerate(top_3_physical, 1):
        i_def = model['i_definition']
        score = model['physical_laws_total_score']
        orig = df_ei[df_ei['i_definition'] == i_def].iloc[0]
        
        report.append(f"**{rank}. {i_def.upper()}** (Score: {score:.2f}/100)")
        report.append(f"   - Emergent Laws: {model.get('emergent_laws_quality', 50):.1f}, Friedmann: {model.get('friedmann_consistency', 50):.1f}")
        report.append("")
    
    report.append("=" * 70)
    report.append("")
    
    # RECOMMENDATION
    best_stability = top_3_stability[0]['i_definition']
    best_complexity = top_3_complexity[0]['i_definition']
    best_physical = top_3_physical[0]['i_definition']
    
    report.append("## 🏆 FINAL RECOMMENDATION")
    report.append("")
    
    report.append(f"**Stability-Focused Winner: `{best_stability}`**")
    report.append(f"**Complexity-Focused Winner: `{best_complexity}`**")
    report.append(f"**Physical-Laws-Focused Winner: `{best_physical}`**")
    report.append("")
    report.append("### Which to choose?")
    report.append("")
    report.append(f"✅ **For OBSERVATIONAL VALIDATION: USE `{best_physical}`**")
    report.append("   - Best match with Planck 2018 cosmology")
    report.append("   - Realistic emergent laws")
    report.append("   - CMB anomaly reproduction")
    report.append("")
    report.append(f"✅ **For TQE theory validation: USE `{best_complexity}`**")
    report.append("   - More complex, life-compatible universes")
    report.append("   - Information-driven structure formation")
    report.append("")
    report.append(f"⚠️ For maximum stability: USE `{best_stability}`")
    report.append("   - More stable configurations")
    report.append("")
    report.append("### USAGE RECOMMENDATION")
    report.append("")
    report.append(f"For Planck-consistent simulations:")
    report.append("```python")
    report.append(f'I_DEFINITION_MODE = "{best_physical}"')
    report.append("```")
    report.append("")
    report.append(f"For TQE complexity studies:")
    report.append("```python")
    report.append(f'I_DEFINITION_MODE = "{best_complexity}"')
    report.append("```")
    report.append("")
    report.append("=" * 70)
    report.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, "recommendation_report.md"), 'w') as f:
        f.write(report_text)
    print("   ✅ recommendation_report.md")
    
    # Print best models from all THREE rankings
    print("\n" + "="*70)
    print("🏆 TRIPLE RANKING RESULTS")
    print("="*70)
    print(f"Stability-Focused Winner: {best_stability} (Score: {top_3_stability[0]['stability_total_score']:.2f}/100)")
    print(f"Complexity-Focused Winner: {best_complexity} (Score: {top_3_complexity[0]['complexity_total_score']:.2f}/100)")
    print(f"Physical-Laws-Focused Winner: {best_physical} (Score: {top_3_physical[0]['physical_laws_total_score']:.2f}/100)")
    
    print(f"\n✨ RECOMMENDATION:")
    print(f"  • For OBSERVATIONAL VALIDATION: Use {best_physical}")
    print(f"  • For TQE complexity studies: Use {best_complexity}")
    print(f"  • For maximum stability: Use {best_stability}")
    
    print("="*70)
    
    print("\n✅ Triple Ranking System Complete!")


# ==========================================================================================
# PHASE 6-8 HELPERS: EXTENDED REPORTS, SUMMARY EXPORT, VALIDATION
# ==========================================================================================

def generate_extended_reports(df_metrics: pd.DataFrame, collected_data: Dict, output_dir: str):
    """
    PHASE 6: Produce extended markdown report summarizing key findings.
    """
    print("\n" + "="*70)
    print("PHASE 6: EXTENDED ANALYSIS REPORTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append("# TQE Analysis Pipeline v4.2.0 PRO — Extended Report")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Target mode: {collected_data['metadata']['target_mode']}")
    lines.append(f"- Source directory: `{collected_data['metadata']['target_path']}`")
    lines.append("")
    lines.append("## Run Inventory")
    lines.append(f"- Total runs analyzed: **{len(df_metrics)}**")
    lines.append(f"- E-only runs: **{collected_data['metadata']['n_eonly_runs']}**")
    lines.append(f"- E+I runs: **{collected_data['metadata']['n_ei_runs']}**")
    lines.append("")
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    if len(df_ei) > 0:
        best_stability = df_ei.sort_values("stable_percent", ascending=False).iloc[0]
        best_complexity = df_ei.sort_values("complexity_score", ascending=False).iloc[0]
        best_physical = df_ei.sort_values("planck_chi2_reduced", ascending=True).iloc[0]
        
        lines.append("## Highlighted I-Definitions")
        lines.append(f"- **Stability leader**: `{best_stability['i_definition']}` ({best_stability['stable_percent']:.2f}% stable)")
        lines.append(f"- **Complexity leader**: `{best_complexity['i_definition']}` (score {best_complexity['complexity_score']:.2f})")
        lines.append(f"- **Planck proximity leader**: `{best_physical['i_definition']}` (χ²={best_physical['planck_chi2_reduced']:.2f})")
        lines.append("")
    
    lines.append("## Artifact Coverage")
    lines.append("- summary_full.json, tqe_runs.csv, Bayesian calibration")
    lines.append("- Planck validation (scatter + χ² bar + CSV export)")
    lines.append("- Life compatibility, entropy volatility, anomaly diagnostics")
    lines.append("- Nested sampling traces, stability sweeps, seed registries")
    lines.append("")
    
    lines.append("## Suggested Follow-ups")
    lines.append("1. Inspect `04_best_model_selection/recommendation_report.md` for model choices.")
    lines.append("2. Review `02_detailed_metrics/all_runs_metrics.csv` for downstream ML.")
    lines.append("3. Compare E-only vs E+I in `01_comparative_analysis/eonly_vs_ei/` if available.")
    lines.append("")
    
    report_path = os.path.join(output_dir, "extended_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"   ✅ Extended report written to {report_path}")


def export_summary_and_metadata(df_metrics: pd.DataFrame, collected_data: Dict, dirs: Dict[str, str], output_root: str) -> str:
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
    summary_text.append(f"Simulation root: {SIMULATION_ROOT}")
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


def run_validation_checks(dirs: Dict[str, str]):
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


# ==========================================================================================
# MAIN ANALYSIS PIPELINE ORCHESTRATOR
# ==========================================================================================
# Executes all 5 analysis phases sequentially:
#   1. Data Collection & Validation
#   2. Metric Extraction & DataFrame Construction
#   3. Comparative Analysis (E+I comparison + E-only vs E+I)
#   4. Advanced Visualizations
#   5. Best Model Selection & Ranking
# ==========================================================================================

def run_analysis_pipeline():
    """
    Main analysis pipeline orchestrator.
    
    Executes complete comparative analysis workflow:
    1. Validates target mode (batch_ei or batch_all)
    2. Collects simulation data from all runs
    3. Extracts and normalizes metrics
    4. Performs comparative analyses
    5. Generates visualizations
    6. Ranks models and produces recommendation
    
    Returns:
        None (all results saved to disk)
    """
    print("\n" + "="*70)
    print("TQE ANALYSIS PIPELINE v4.2.0 PRO")
    print("="*70)
    print(f"Analysis started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target mode: {TARGET_MODE}")
    print(f"Simulation root: {SIMULATION_ROOT}")
    print("="*70)
    
    # Validate target mode
    if not validate_target_mode(TARGET_MODE):
        sys.exit(1)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_name = TARGET_MODE.replace("TQE_Universe_Simulation_", "").replace("_Pipeline_v4.2.0_PRO", "")
    output_root = os.path.join(ANALYSIS_OUTPUT_ROOT, f"{mode_name}_{timestamp}_analysis")
    os.makedirs(output_root, exist_ok=True)
    
    print(f"\nOutput directory: {output_root}\n")
    
    # Create progress tracker (8 phases total - EXTENDED)
    progress = tqdm(total=8, desc="Analysis Progress", ncols=100)
    
    # PHASE 1: Collect data
    progress.set_description("Phase 1/8: Comprehensive Data Collection")
    collected_data = collect_simulation_data(TARGET_MODE)
    progress.update(1)
    
    # Save collected data
    os.makedirs(os.path.join(output_root, "05_raw_data"), exist_ok=True)
    with open(os.path.join(output_root, "05_raw_data", "collected_data.pkl"), 'wb') as f:
        pickle.dump(collected_data, f)
    
    # PHASE 2: Build metrics DataFrame with advanced scoring
    progress.set_description("Phase 2/8: Extended Metric Extraction")
    print("\n" + "="*70)
    print("PHASE 2: COMPREHENSIVE METRIC EXTRACTION & SCORING (EXTENDED)")
    print("="*70)
    df_metrics = build_metrics_dataframe(collected_data)
    print(f"✅ Comprehensive metrics: complexity, life-compatibility, information richness")
    print(f"✅ Extended metrics: emergent laws, Friedmann, CMB, lock-in, quantum, entanglement, etc.")
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
    
    # PHASE 3: Comprehensive Comparative Analysis (EXTENDED)
    progress.set_description("Phase 3/8: Comparative Analysis")
    
    # 3A: Basic metrics
    compare_ei_definitions(df_metrics, dirs["ei_comparison"])
    
    # 3B-K: Extended category analyses
    analyze_emergent_laws(df_metrics, dirs["emergent_laws"])
    analyze_friedmann_cosmology(df_metrics, dirs["friedmann"])
    analyze_cmb_anomalies(df_metrics, dirs["cmb_anomalies"])
    analyze_lockin_dynamics(df_metrics, dirs["lockin_dynamics"])
    analyze_quantum_fields(df_metrics, dirs["quantum_fields"])
    analyze_entanglement(df_metrics, dirs["entanglement"])
    analyze_parameter_sensitivity(df_metrics, dirs["param_sensitivity"])
    analyze_statistical_finetuning(df_metrics, collected_data, dirs["finetuning"])
    analyze_topology(df_metrics, dirs["topology"])
    analyze_i_definitions_direct(df_metrics, collected_data, dirs["i_definitions_direct"])
    analyze_planck_fit(df_metrics, dirs["planck_fit"])
    analyze_life_top_universes(df_metrics, dirs["life_top"])
    analyze_entropy_volatility(df_metrics, dirs["entropy_volatility"])
    analyze_physical_anomalies(df_metrics, dirs["physical_anomalies"])
    
    # 3L: E-only vs E+I baseline
    if collected_data["metadata"]["has_eonly"]:
        compare_eonly_vs_ei(df_metrics, dirs["eonly_vs_ei"])
    
    generate_detailed_metrics(df_metrics, dirs["detailed_metrics"])
    progress.update(1)
    
    # PHASE 4: Advanced Visualizations
    progress.set_description("Phase 4/8: Advanced Visualizations")
    generate_advanced_visualizations(df_metrics, dirs["visualizations"])
    generate_complexity_analysis(df_metrics, dirs["complexity_analysis"])
    progress.update(1)
    
    # PHASE 5: Triple Ranking System (EXTENDED)
    progress.set_description("Phase 5/8: Triple Model Ranking")
    select_best_model(df_metrics, dirs["best_model"])
    progress.update(1)
    
    # PHASE 6: Extended Analysis Reports
    progress.set_description("Phase 6/8: Extended Reports")
    generate_extended_reports(df_metrics, collected_data, dirs["summary"])
    progress.update(1)
    
    # PHASE 7: Comprehensive Summary Export
    progress.set_description("Phase 7/8: Summary Export")
    summary_text = export_summary_and_metadata(df_metrics, collected_data, dirs, output_root)
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


# ==========================================================================================
# MAIN EXECUTION
# ==========================================================================================
if __name__ == "__main__":
    # Print startup banner
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  TQE ANALYSIS PIPELINE v4.2.0 PRO".center(68) + "║")
    print("║" + "  Comparative Analysis & Model Selection".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Environment detection
    if IN_COLAB:
        print("\n[COLAB] Google Colab environment detected.")
        
        # Try to mount Google Drive
        try:
            from google.colab import drive
            if not os.path.exists("/content/drive/MyDrive"):
                print("[DRIVE] Attempting to mount Google Drive...")
                drive.mount('/content/drive')
                print("[DRIVE] Successfully mounted!")
            else:
                print("[DRIVE] Google Drive already mounted.")
        except Exception as e:
            print(f"[DRIVE] Warning: Could not mount Google Drive: {e}")
    else:
        print("\n[LOCAL] Local environment detected.")
    
    # Setup paths after Drive is mounted
    setup_paths()
    
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

