# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# quantum_system.py - Quantum System Setup
# ==========================================================================================
# Build quantum system operators, Hamiltonians, collapse operators
# ==========================================================================================

import numpy as np
from qutip import destroy, num, tensor, qeye

def build_quantum_system(config):
    """
    Build quantum system operators and Hamiltonians.
    
    Returns:
    --------
    dict
        Dictionary containing all quantum system components:
        - H: Hamiltonian
        - c_ops: List of collapse operators
        - Nop1_full, Nop2_full: Number operators
        - x1, p1, x2, p2: Position and momentum operators
        - x_op_mode1_cached, p_op_mode1_cached: Cached tensor operators
        - two_mode: Boolean indicating if two-mode system
    """
    N_HILB = config["N_HILB"]
    omega1 = config["OMEGA_1"]
    omega2 = config["OMEGA_2"]
    lam_x4 = config["LAM_X4"]
    dw_c2 = config["DW_C2"]
    dw_c4 = config["DW_C4"]
    g_coup = config["G_COUP"]
    drive_amp = config["DRIVE_AMP"]
    drive_omega = config["DRIVE_OMEGA"]
    
    # Single-mode operators
    a1 = destroy(N_HILB)
    x1 = (a1 + a1.dag()) / np.sqrt(2.0)
    p1 = (a1 - a1.dag()) / (1j * np.sqrt(2.0))
    H1 = omega1 * (a1.dag() * a1)
    
    # Add potentials on mode-1
    if config["ANHARMONIC_X4"]:
        H1 = H1 + lam_x4 * (x1**4)
    
    if config["DOUBLE_WELL"]:
        H1 = omega1 * (a1.dag()*a1) + dw_c2*(x1**2) + dw_c4*(x1**4)
    
    # Handle one vs two-mode system
    two_mode = config["TWO_MODE_COUPLING"]
    ident = qeye(N_HILB)
    
    if two_mode:
        a2 = destroy(N_HILB)
        x2 = (a2 + a2.dag()) / np.sqrt(2.0)
        H2_local = omega2 * (a2.dag() * a2)
        
        H1_full = tensor(H1, ident)
        H2_full = tensor(ident, H2_local)
        Hc_full = g_coup * tensor(a1 + a1.dag(), a2 + a2.dag())
        
        H = H1_full + H2_full + Hc_full
    else:
        H = H1
        x2 = None
        p2 = None
    
    # Time-dependent drive on mode-1
    if config["TIME_DEP_DRIVE"]:
        # Create closure for drive coefficient function
        drive_amp_local = drive_amp
        drive_omega_local = drive_omega
        def H_drive_coeff(t, args):
            return drive_amp_local * np.cos(drive_omega_local * t)
        
        drive_op = tensor(x1, ident) if two_mode else x1
        H = [H, [drive_op, H_drive_coeff]]
    
    # Number operators
    Nop1 = num(N_HILB)
    if two_mode:
        Nop2 = num(N_HILB)
        Nop1_full = tensor(Nop1, ident)
        Nop2_full = tensor(ident, Nop2)
    else:
        Nop2 = None
        Nop1_full = Nop1
        Nop2_full = None
    
    # Build collapse operators
    c_ops = []
    
    # Mode-1 operators
    a1_op = tensor(a1, ident) if two_mode else a1
    n1_op = tensor(a1.dag() * a1, ident) if two_mode else a1.dag() * a1
    
    gamma_phi1 = config["GAMMA_PHI_1"]
    kappa1 = config["KAPPA_1"]
    nth1 = config["NTH_1"]
    
    c_ops.append(np.sqrt(gamma_phi1) * n1_op)
    if config["THERMAL_BATH"] and nth1 > 0:
        c_ops.append(np.sqrt(kappa1 * (nth1 + 1.0)) * a1_op)
        c_ops.append(np.sqrt(kappa1 * nth1) * a1_op.dag())
    else:
        c_ops.append(np.sqrt(kappa1) * a1_op)
    
    # Mode-2 operators
    if two_mode:
        a2_op = tensor(ident, a2)
        n2_op = tensor(ident, a2.dag() * a2)
        
        gamma_phi2 = config["GAMMA_PHI_2"]
        kappa2 = config["KAPPA_2"]
        nth2 = config["NTH_2"]
        
        c_ops.append(np.sqrt(gamma_phi2) * n2_op)
        if config["THERMAL_BATH"] and nth2 > 0:
            c_ops.append(np.sqrt(kappa2 * (nth2 + 1.0)) * a2_op)
            c_ops.append(np.sqrt(kappa2 * nth2) * a2_op.dag())
        else:
            c_ops.append(np.sqrt(kappa2) * a2_op)
    
    # Pre-compute tensor operators for performance
    x_op_mode1_cached = None
    p_op_mode1_cached = None
    
    if config.get("CACHE_TENSOR_OPS", True) and two_mode:
        x_op_mode1_cached = tensor(x1, ident)
        p_op_mode1_cached = tensor(p1, ident)
    
    return {
        'H': H,
        'c_ops': c_ops,
        'Nop1_full': Nop1_full,
        'Nop2_full': Nop2_full,
        'x1': x1,
        'p1': p1,
        'x2': x2,
        'p2': p2,
        'a1': a1,
        'a2': a2 if two_mode else None,
        'x_op_mode1_cached': x_op_mode1_cached,
        'p_op_mode1_cached': p_op_mode1_cached,
        'two_mode': two_mode,
        'ident': ident,
        'N_HILB': N_HILB,
        'omega1': omega1,
        'omega2': omega2,
        'lam_x4': lam_x4,
        'dw_c2': dw_c2,
        'dw_c4': dw_c4,
        'g_coup': g_coup,
    }

