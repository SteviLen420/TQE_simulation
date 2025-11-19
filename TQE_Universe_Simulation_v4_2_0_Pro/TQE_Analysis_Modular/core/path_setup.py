# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Path setup and file finding utilities

import os
import glob
from typing import List, Optional

def setup_paths(config: dict) -> tuple:
    """
    Setup SIMULATION_ROOT and ANALYSIS_OUTPUT_ROOT based on environment.
    Local execution only - saves to Desktop.
    
    Args:
        config: MASTER_CTRL configuration dictionary
    
    Returns:
        Tuple of (SIMULATION_ROOT, ANALYSIS_OUTPUT_ROOT)
    """
    # Local execution only - Desktop output
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    analysis_results_dir = os.path.join(desktop_path, "TQE_Analysis_Modular_Results")
    
    # Get repo root (2 levels up from Analysis Modular)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up: core -> TQE_Analysis_Modular -> TQE_Universe_Simulation_v4.2.0_Pro -> repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # Simulation root: repo/SIMULATION_RUNS/universe
    simulation_root = os.path.join(repo_root, "SIMULATION_RUNS", "universe")
    
    # Analysis output: Desktop/TQE_Analysis_Modular_Results
    analysis_output_root = analysis_results_dir
    
    # Override with config if specified
    if config.get("SIMULATION_ROOT") is not None:
        simulation_root = config["SIMULATION_ROOT"]
    if config.get("ANALYSIS_OUTPUT_ROOT") is not None:
        analysis_output_root = config["ANALYSIS_OUTPUT_ROOT"]
    
    # Create analysis output directory
    os.makedirs(analysis_output_root, exist_ok=True)
    
    return simulation_root, analysis_output_root


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
    import glob
    eonly_dirs = glob.glob(os.path.join(target_path, "Eonly_*"))
    return len(eonly_dirs) > 0


def collect_run_directories(target_path: str) -> dict:
    """
    Collect all run directories from target mode folder.
    
    Returns:
        {
            "eonly": ["Eonly_20251030_223510"],
            "ei": ["EplusI_kl_divergence_...", "EplusI_shannon_...", ...]
        }
    """
    import glob
    eonly_dirs = sorted(glob.glob(os.path.join(target_path, "Eonly_*")))
    ei_dirs = sorted(glob.glob(os.path.join(target_path, "EplusI_*")))
    
    return {
        "eonly": [os.path.basename(d) for d in eonly_dirs],
        "ei": [os.path.basename(d) for d in ei_dirs]
    }


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

