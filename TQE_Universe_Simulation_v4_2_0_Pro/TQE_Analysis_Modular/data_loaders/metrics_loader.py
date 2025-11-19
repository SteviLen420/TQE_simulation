# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Metrics and specialized data loading functions

import json
from typing import Optional, Dict
import pandas as pd
from ..core.path_setup import smart_find_file

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

