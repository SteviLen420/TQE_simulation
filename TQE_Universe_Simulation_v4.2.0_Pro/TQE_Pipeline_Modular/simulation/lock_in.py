# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Lock-in mechanism module
#
import pandas as pd
from ..config.master_ctrl import MASTER_CTRL

def adjust_stability_thresholds(df: pd.DataFrame, config: dict) -> dict:
    """Adjust stability thresholds to achieve target distribution rates."""
    if not config.get("ADJUST_STABILITY_THRESHOLDS", False):
        return config
    
    target_unstable = config.get("TARGET_UNSTABLE_RATE", 0.60)
    target_stable = config.get("TARGET_STABLE_RATE", 0.40)
    target_lockin = config.get("TARGET_LOCKIN_RATE", 0.60)
    adjustment_factor = config.get("STABILITY_ADJUSTMENT_FACTOR", 0.1)
    
    # Calculate current rates
    total_universes = len(df)
    stable_count = df['stable'].sum()
    lockin_count = df['lock_epoch'].ge(0).sum()
    
    current_unstable_rate = (total_universes - stable_count) / total_universes
    current_stable_rate = stable_count / total_universes
    current_lockin_rate = lockin_count / max(stable_count, 1)  # Lock-in rate among stable universes
    
    print(f"[STABILITY ADJUSTMENT] Current rates: Unstable={current_unstable_rate:.3f}, Stable={current_stable_rate:.3f}, Lock-in={current_lockin_rate:.3f}")
    print(f"[STABILITY ADJUSTMENT] Target rates: Unstable={target_unstable:.3f}, Stable={target_stable:.3f}, Lock-in={target_lockin:.3f}")
    
    # Adjust stability threshold
    if current_stable_rate > target_stable + 0.05:  # Too many stable
        config["REL_EPS_STABLE"] *= (1 + adjustment_factor)  # Make stability harder
        print(f"[STABILITY ADJUSTMENT] Increasing REL_EPS_STABLE to {config['REL_EPS_STABLE']:.6f}")
    elif current_stable_rate < target_stable - 0.05:  # Too few stable
        config["REL_EPS_STABLE"] *= (1 - adjustment_factor)  # Make stability easier
        print(f"[STABILITY ADJUSTMENT] Decreasing REL_EPS_STABLE to {config['REL_EPS_STABLE']:.6f}")
    
    # Adjust lock-in threshold
    if current_lockin_rate > target_lockin + 0.05:  # Too many lock-in
        config["REL_EPS_LOCKIN"] *= (1 + adjustment_factor)  # Make lock-in harder
        print(f"[STABILITY ADJUSTMENT] Increasing REL_EPS_LOCKIN to {config['REL_EPS_LOCKIN']:.6f}")
    elif current_lockin_rate < target_lockin - 0.05:  # Too few lock-in
        config["REL_EPS_LOCKIN"] *= (1 - adjustment_factor)  # Make lock-in easier
        print(f"[STABILITY ADJUSTMENT] Decreasing REL_EPS_LOCKIN to {config['REL_EPS_LOCKIN']:.6f}")
    
    return config

