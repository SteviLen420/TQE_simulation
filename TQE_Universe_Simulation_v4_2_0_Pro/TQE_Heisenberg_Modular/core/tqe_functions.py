# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# tqe_functions.py - TQE Helper Functions
# ==========================================================================================
# TQE lock-in functions, Heisenberg uncertainty, coherence measures
# ==========================================================================================

import numpy as np
from qutip import expect

def sample_info_beta(n, rng, config):
    """Sample information parameter I from Beta(a, b) distribution."""
    a = config.get("BETA_A", 2.0)
    b = config.get("BETA_B", 2.0)
    return rng.beta(a, b, size=n)

def f_lockin(E, I, config):
    """
    TQE lock-in function: f(E,I) = exp(-(E-Ec)²/(2σ²)) · (1 + α·I)
    
    Analytical Form:
    ---------------
    f(E,I) represents the coupling strength between vacuum energy fluctuations
    and the intrinsic information content of the quantum state.
    
    Components:
    1. Gaussian envelope: exp(-(E-Ec)²/(2σ²))
       - Peaks at Ec (Goldilocks energy center)
       - Width σ defines stability window
       - Suppresses fluctuations outside optimal energy range
    
    2. Information modulation: (1 + α·I)
       - Linear bias proportional to I ∈ [0,1]
       - Strength α controls information coupling
       - Enhances stability for higher information content
    
    Physical Interpretation:
    - f(E,I) → 0: Strong suppression (outside Goldilocks zone)
    - f(E,I) ≈ 1: Neutral (near Ec with I≈0)
    - f(E,I) > 1: Enhancement (near Ec with I>0)
    
    This function modulates:
    - Dissipation rates: γ_eff = γ₀ · √f(E,I)
    - Potential landscape: V_eff = V₀ · f(E,I)
    """
    Ec = config.get("EC", 25.0)
    sigma = config.get("SIGMA", 8.0)
    alpha = config.get("ALPHA", 0.8)
    
    gaussian = np.exp(-(E - Ec)**2 / (2.0 * sigma**2))
    return gaussian * (1.0 + alpha * I)

def lockin_rate_scale(E_proxy, I_val, config):
    """Return multiplicative scale for dissipators given E and I."""
    return float(np.clip(f_lockin(E_proxy, I_val, config), 0.25, 2.0))

def lockin_potential_scale(E_proxy, I_val, config):
    """Scale anharmonic strength depending on f(E,I)."""
    return float(np.clip(0.6 + 0.6*f_lockin(E_proxy, I_val, config), 0.2, 1.8))

def compute_heisenberg_uncertainty(rho, x_op, p_op, config):
    """
    Compute Heisenberg uncertainty product: Δx·Δp
    
    Returns:
    --------
    delta_x : float
        Position uncertainty
    delta_p : float
        Momentum uncertainty
    uncertainty_product : float
        Δx·Δp (should satisfy Δx·Δp ≥ ℏ/2)
    heisenberg_violation : bool
        True if uncertainty product < theoretical minimum
    """
    # Expectation values
    x_mean = float(expect(x_op, rho))
    p_mean = float(expect(p_op, rho))
    
    # Second moments
    x2_mean = float(expect(x_op * x_op, rho))
    p2_mean = float(expect(p_op * p_op, rho))
    
    # Variances
    delta_x = np.sqrt(max(0.0, x2_mean - x_mean**2))
    delta_p = np.sqrt(max(0.0, p2_mean - p_mean**2))
    
    # Uncertainty product
    uncertainty_product = delta_x * delta_p
    
    # Check Heisenberg limit (ℏ/2 = 0.5 in natural units)
    hbar = config.get("HBAR", 1.0)
    hbar_half = hbar / 2.0
    heisenberg_violation = uncertainty_product < (hbar_half * 0.99)  # 1% tolerance
    
    return delta_x, delta_p, uncertainty_product, heisenberg_violation

def normalize_coherence(rho, mode=None):
    """
    Normalized coherence measure: C ∈ [0, 1]
    
    C = (Σ|ρ_ij| - Σ|ρ_ii|) / (N² - N)
    
    where:
    - Numerator: sum of off-diagonal elements
    - Denominator: maximum possible off-diagonal contribution
    - N: Hilbert space dimension
    
    Returns 0 for completely mixed state, 1 for maximally coherent state.
    """
    if mode is not None:
        # Partial trace for two-mode system
        rho = rho.ptrace(mode)
    
    rho_matrix = rho.full()
    N = rho_matrix.shape[0]
    
    # Sum of absolute values of all elements
    total_abs = float(np.sum(np.abs(rho_matrix)))
    
    # Sum of absolute values of diagonal elements
    diag_abs = float(np.sum(np.abs(np.diag(rho_matrix))))
    
    # Off-diagonal contribution
    off_diag_abs = total_abs - diag_abs
    
    # Maximum possible off-diagonal contribution (N² - N elements, each ≤ 1)
    max_off_diag = N * N - N
    
    # Normalized coherence
    coherence_normalized = off_diag_abs / max(max_off_diag, 1.0)
    
    return float(np.clip(coherence_normalized, 0.0, 1.0))

