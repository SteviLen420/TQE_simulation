# ===================================================================================
# 00_TQE_(E,I)_UNIVERSE_SIMULATION_config.py
# ===================================================================================
# Config wrapper for the TQE simulation.
# - Imports the Master Controller.
# - Resolves the active profile.
# - Exports ACTIVE configuration for all modules.
# Author: Stefan Len
# ===================================================================================

import os
from Master_Control import MASTER_CTRL, _deep_merge

# ===================================================================================
# Profile resolution
# ===================================================================================
SELECTED_PROFILE = os.environ.get("TQE_PROFILE", "paper")  # default: "paper"

def resolve_profile(profile_name: str):
    """
    Resolve profile by merging the MASTER_CTRL with the selected profile overrides.
    """
    from copy import deepcopy

    base = deepcopy(MASTER_CTRL)
    profs = base.pop("PROFILES", {})
    chosen = profs.get(profile_name, {})
    active = _deep_merge(base, chosen)

    # Enforce strict seed reproducibility if requested
    if active["REPRO"]["use_strict_seed"]:
        for k, v in active["REPRO"]["env_thread_caps"].items():
            os.environ[k] = str(v)

    # Optionally tag run_id with profile name
    if base["OUTPUTS"].get("tag_profile_in_runid", False):
        os.environ["TQE_PROFILE_TAG"] = profile_name

    return active

# ===================================================================================
# Active configuration (imported by all pipeline modules)
# ===================================================================================
ACTIVE = resolve_profile(SELECTED_PROFILE)
