# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_00_EI_UNIVERSE_SIMULATION_config.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script serves as the primary configuration module for the entire simulation
# pipeline. It dynamically constructs the active configuration settings by selecting
# a "profile" (e.g., "demo", "production"). The profile is determined by the
# `TQE_PROFILE` environment variable, defaulting to "demo" if not set.
#
# The core logic involves deep-merging a base configuration (`MASTER_CTRL`) with
# profile-specific overrides. This creates a final, unified configuration object
# named `ACTIVE`. The script also handles critical reproducibility settings, such
# as enforcing thread limits for deterministic execution when required by a profile.
# All subsequent pipeline scripts should import the `ACTIVE` object to access
# their parameters.
#
# ===================================================================================

import os
from typing import Optional


def _deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a

# --- Profile resolution (no cross-imports to other TQE modules) -----------------

# Default profile; can be overridden via env var TQE_PROFILE
DEFAULT_PROFILE = "demo"

# Base (default) configuration for the whole pipeline.
# If your repo already has MASTER_CTRL, you can set BASE = MASTER_CTRL.
BASE = {
    "PIPELINE": {"use_information": True, "run_anomaly_scan": True},
    "ENERGY":   {"num_universes": 1000, "seed": None},
    "ANOMALY": {
        "map": {"resolution_nside": 128, "seed_per_map": True},
        "targets": [
            {"name": "cold_spot", "enabled": True, "patch_deg": 10.0, "zscore_thresh": 3.0},
            {"name": "quad_oct_align", "enabled": True, "l2l3_align_deg": 20.0},
            {"name": "lack_large_angle", "enabled": True, "theta_min_deg": 60.0,
             "lmax": 64, "n_mc": 200, "p_percentile": 0.05},
            {"name": "hemispheric_asymmetry", "enabled": True, "l_max": 40, "n_mc": 200,
             "pval_thresh": 0.05},
        ],
        "save_cutouts": True,
        "save_metrics_csv": True,
    },
    "FINETUNE_DIAG": {"top_k": 5},
    "XAI": {
        "test_size": 0.25,
        "test_random_state": 42,
        "rf_n_estimators": 400,
        "rf_class_weight": None,
        "regression_min": 10,
        "lime_num_features": 5,
        "sklearn_n_jobs": -1,
        "run_lime": True,
    },
    "OUTPUTS": {"local": {"fig_subdir": "figs"}, "mirrors": []},
    "RUNTIME": {"matplotlib_dpi": 180},
}

# Profile-specific overrides. Keep these light; only override what differs from BASE.
PROFILES = {
    "demo": {},
    "paper": {
        # Example only — tune as you wish:
        # "ENERGY": {"num_universes": 5000},
        # "ANOMALY": {"map": {"resolution_nside": 256}},
        # "XAI": {"rf_n_estimators": 800},
    },
    "colab": {
        # Colab-friendly defaults (optional):
        # "ENERGY": {"num_universes": 200},
        # "OUTPUTS": {"local": {"fig_subdir": "figs"}, "mirrors": ["/content/drive/MyDrive/tqe_runs"]},
    },
}

def _copy_tree(d):
    """Shallow copy dict-of-dicts safely for mutation."""
    out = {}
    for k, v in d.items():
        out[k] = dict(v) if isinstance(v, dict) else v
    return out

def build_active(profile_name: Optional[str] = None) -> dict:
    """
    Merge BASE with the selected profile override and return a fresh ACTIVE dict.
    No imports to other TQE modules — avoids circular imports.
    """
    prof = profile_name or os.environ.get("TQE_PROFILE", DEFAULT_PROFILE)
    active = _copy_tree(BASE)
    override = PROFILES.get(prof, {})
    _deep_merge(active, override)
    # Annotate selected profile for downstream logging
    meta = dict(active.get("META", {}))
    meta["selected_profile"] = prof
    active["META"] = meta
    return active

# Backward-compatible globals
SELECTED_PROFILE = os.environ.get("TQE_PROFILE", DEFAULT_PROFILE)
ACTIVE = build_active(SELECTED_PROFILE)

# ===================================================================================
# Profile resolution
# ===================================================================================

# Default profile; override via environment:  export TQE_PROFILE=demo
SELECTED_PROFILE = os.environ.get("TQE_PROFILE", "demo")

def resolve_profile(profile_name: str):
    """
    Build the ACTIVE configuration by deep-merging the base MASTER_CTRL with
    the selected profile overrides under MASTER_CTRL["PROFILES"][profile_name].
    Also applies reproducibility thread caps and optional run-id tagging.
    """
    from copy import deepcopy

    base = deepcopy(MASTER_CTRL)
    profiles = base.pop("PROFILES", {})
    chosen = profiles.get(profile_name, {})

    # Merge (right overrides left)
    active = _deep_merge(base, chosen)

    # Reproducibility: optionally enforce low-thread deterministic settings
    repro = active.get("REPRO", {})
    if repro.get("use_strict_seed"):
        for k, v in repro.get("env_thread_caps", {}).items():
            os.environ[k] = str(v)

    # Optionally tag the run_id with the selected profile
    outputs = active.get("OUTPUTS", {})
    if outputs.get("tag_profile_in_runid", False):
        os.environ["TQE_PROFILE_TAG"] = profile_name

    return active


# ===================================================================================
# Active configuration (import this from all pipeline modules)
# ===================================================================================
ACTIVE = resolve_profile(SELECTED_PROFILE)

# Sanity checks (fail fast if something critical is missing)
assert isinstance(ACTIVE, dict), "ACTIVE must be a dict"
for key in ["META", "PIPELINE", "OUTPUTS"]:
    assert key in ACTIVE, f"ACTIVE missing required section: {key}"
