# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_00_EI_UNIVERSE_SIMULATION_config.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

import os

# Import the master dictionary and deep-merge helper from a clean, importable module.
# Ensure there is a file named `Master_Control.py` in the same folder that contains:
#   - MASTER_CTRL
#   - _deep_merge
from TQE_01_EI_UNIVERSE_SIMULATION_Master_Control import MASTER_CTRL, _deep_merge


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
