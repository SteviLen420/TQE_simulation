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
