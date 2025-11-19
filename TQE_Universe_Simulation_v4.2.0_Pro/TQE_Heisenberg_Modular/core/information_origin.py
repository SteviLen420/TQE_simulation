# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# information_origin.py - Information Origin Models
# ==========================================================================================
# Three models for I-parameter evolution: emergent, inherent, threshold
# ==========================================================================================

import numpy as np

def compute_I_emergent(I_prev, E_current, E_prev, E_prev2, config):
    """
    Emergent I: Information emerges from energy fluctuation structure.
    
    I_{t+1} = γ·I_t + α·|ΔE_t| + β·corr(ΔE_t, ΔE_{t-1})
    """
    alpha = config.get("I_EMERGENT_ALPHA", 0.3)
    beta = config.get("I_EMERGENT_BETA", 0.2)
    gamma = config.get("I_EMERGENT_GAMMA", 0.95)
    
    dE_t = E_current - E_prev
    dE_t_minus_1 = E_prev - E_prev2
    corr_term = dE_t * dE_t_minus_1 / (1.0 + abs(dE_t) + abs(dE_t_minus_1))
    I_new = gamma * I_prev + alpha * abs(dE_t) + beta * corr_term
    
    return np.clip(I_new, 0.0, 1.0)

def compute_I_inherent(E, config):
    """
    Inherent I: Information is deterministic function of energy.
    
    Options: log, power, linear
    """
    mode = config.get("I_INHERENT_MODE", "log")
    E0 = config.get("I_INHERENT_E0", 10.0)
    gamma_exp = config.get("I_INHERENT_GAMMA", 0.5)
    scale = config.get("I_INHERENT_SCALE", 0.05)
    
    if mode == "log":
        I = scale * np.log(max(E / E0, 0.01))
    elif mode == "power":
        I = scale * (E / E0) ** gamma_exp
    elif mode == "linear":
        I = scale * E
    else:
        I = 0.5
    
    return np.clip(I, 0.0, 1.0)

def compute_I_threshold(E, I_prev, config):
    """
    Threshold I: Information activates above critical energy.
    """
    E_c = config.get("I_THRESHOLD_EC", 15.0)
    slope = config.get("I_THRESHOLD_SLOPE", 0.1)
    I_max = config.get("I_THRESHOLD_MAX", 1.0)
    
    if E < E_c:
        return 0.0
    else:
        I_new = I_prev + slope * (E - E_c)
        return np.clip(I_new, 0.0, I_max)

