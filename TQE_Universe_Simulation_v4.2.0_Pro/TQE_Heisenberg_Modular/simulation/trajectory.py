# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# trajectory.py - Single Trajectory Evolution
# ==========================================================================================
# run_single function for simulating one member of ensemble
# ==========================================================================================

import numpy as np
from scipy.stats import entropy as shannon_entropy
from qutip import coherent, tensor, mesolve, expect, entropy_vn
from qutip.solver import Options

from ..core.tqe_functions import (
    compute_heisenberg_uncertainty, normalize_coherence,
    lockin_rate_scale, lockin_potential_scale
)
from ..core.information_origin import (
    compute_I_emergent, compute_I_inherent, compute_I_threshold
)

def run_single(alpha1, alpha2, I_val, enable_lockin, config, quantum_system, tlist):
    """
    Simulate one member of ensemble.
    
    Parameters
    ----------
    alpha1 : complex
        Coherent state amplitude for mode 1
    alpha2 : complex or None
        Coherent state amplitude for mode 2 (if two-mode)
    I_val : float
        Information parameter value (0 ≤ I ≤ 1)
    enable_lockin : bool
        If True, apply TQE lock-in mechanism (WITH-LAW)
        If False, pure fluctuations (NO-LAW)
    config : dict
        MASTER_CTRL configuration dictionary
    quantum_system : dict
        Dictionary from build_quantum_system() containing all operators
    tlist : array
        Time points for evolution
    
    Returns
    -------
    dict or None
        Dictionary with time series and final state, or None if simulation failed
    """
    # Extract quantum system components
    H = quantum_system['H']
    c_ops = quantum_system['c_ops']
    Nop1_full = quantum_system['Nop1_full']
    Nop2_full = quantum_system['Nop2_full']
    two_mode = quantum_system['two_mode']
    N_HILB = quantum_system['N_HILB']
    x1 = quantum_system['x1']
    p1 = quantum_system['p1']
    x_op_mode1_cached = quantum_system.get('x_op_mode1_cached')
    p_op_mode1_cached = quantum_system.get('p_op_mode1_cached')
    ident = quantum_system['ident']
    a1 = quantum_system['a1']
    a2 = quantum_system.get('a2')
    omega1 = quantum_system['omega1']
    omega2 = quantum_system['omega2']
    lam_x4 = quantum_system['lam_x4']
    dw_c2 = quantum_system['dw_c2']
    dw_c4 = quantum_system['dw_c4']
    g_coup = quantum_system['g_coup']
    
    # Initial state
    psi1 = coherent(N_HILB, alpha1)
    if two_mode and alpha2 is not None:
        psi2 = coherent(N_HILB, alpha2)
        psi0 = tensor(psi1, psi2)
    else:
        psi0 = psi1

    # Optimized Options: store_states can be disabled for memory efficiency
    store_states_flag = not config.get("MEMORY_EFFICIENT", False)
    opts = Options(store_states=store_states_flag, nsteps=10000, atol=1e-8, rtol=1e-6)

    # ===== NO LOCK-IN: Single-shot evolution =====
    if not enable_lockin:
        e_ops_list = [Nop1_full]
        if two_mode and Nop2_full is not None:
            e_ops_list.append(Nop2_full)

        res = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops_list, options=opts)

        if not res.states or len(res.states) < 2:
            return None

        e_series, Svon_series, coh_series, info_drift = [], [], [], []
        initial_rho = res.states[0]
        if two_mode:
            rho_prev_pop = np.real(np.diag(initial_rho.ptrace(0).full()).flatten())
        else:
            rho_prev_pop = np.real(np.diag(initial_rho.full()).flatten())

        uncertainty_series, delta_x_series, delta_p_series = [], [], []
        
        for rho in res.states[1:]:
            if two_mode:
                e_series.append(expect(Nop1_full, rho) + (expect(Nop2_full, rho) if Nop2_full is not None else 0))
                rho1 = rho.ptrace(0)
                coh_series.append(normalize_coherence(rho, mode=0))
                rho_pop = np.real(np.diag(rho1.full()).flatten())
                
                # Heisenberg uncertainty for mode 1 (use cached operators)
                if x_op_mode1_cached is not None and p_op_mode1_cached is not None:
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1_cached, p_op_mode1_cached, config)
                else:
                    x_op_mode1 = tensor(x1, ident)
                    p_op_mode1 = tensor(p1, ident)
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1, p_op_mode1, config)
            else:
                e_series.append(expect(Nop1_full, rho))
                rho_pop = np.real(np.diag(rho.full()).flatten())
                coh_series.append(normalize_coherence(rho))
                dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x1, p1, config)
            
            delta_x_series.append(dx)
            delta_p_series.append(dp)
            uncertainty_series.append(unc_prod)
            Svon_series.append(float(entropy_vn(rho)))

            p = (rho_prev_pop + 1e-12); p /= (p.sum() + 1e-15)
            q = (rho_pop + 1e-12); q /= (q.sum() + 1e-15)
            m = 0.5 * (p + q)
            js = 0.5 * shannon_entropy(p, m, base=2) + 0.5 * shannon_entropy(q, m, base=2)
            info_drift.append(float(js))
            rho_prev_pop = rho_pop
        
        if not e_series:
            return None
        
        return dict(
            energies=np.array(e_series),
            entropy=np.array(Svon_series),
            coherence=np.array(coh_series),
            info_drift=np.array(info_drift),
            delta_x=np.array(delta_x_series),
            delta_p=np.array(delta_p_series),
            uncertainty_product=np.array(uncertainty_series),
            I_evolution=None,
            I_mode="none",
            final_state=res.states[-1],
            final_energy=e_series[-1]
        )
    
    # ===== WITH LOCK-IN: Segmented evolution =====
    else:
        T_FINAL = config["T_FINAL"]
        segs = config["N_SEGMENTS"]
        seg_edges = np.linspace(0.0, T_FINAL, segs + 1)

        all_states_collected = []
        H_static_base = H if not isinstance(H, list) else H[0]
        state_in = psi0
        
        # Dynamic I tracking (initialize based on I_ORIGIN_MODE)
        I_mode = config.get("I_ORIGIN_MODE", "fixed")
        I_current = I_val
        I_history = [I_current]
        
        # Initialize E_history with initial energy estimate
        E_proxy_init = expect(Nop1_full, psi0) + (expect(Nop2_full, psi0) if two_mode and Nop2_full is not None else 0)
        E_history = [E_proxy_init, E_proxy_init, E_proxy_init]

        for s in range(segs):
            t0, t1 = seg_edges[s], seg_edges[s+1]
            t_points_in_seg = tlist[(tlist >= t0) & (tlist <= t1)]

            t_solve_seg = np.union1d([t0], t_points_in_seg)
            if len(t_solve_seg) < 2:
                continue

            E_proxy = E_proxy_init
            if all_states_collected:
                last_rho = all_states_collected[-1]
                E_proxy = expect(Nop1_full, last_rho) + (expect(Nop2_full, last_rho) if two_mode and Nop2_full is not None else 0)
                E_history.append(E_proxy)
            
            # Update I dynamically based on I_ORIGIN_MODE
            if I_mode == "emergent":
                I_current = compute_I_emergent(
                    I_current, 
                    E_history[-1], 
                    E_history[-2], 
                    E_history[-3], 
                    config
                )
            elif I_mode == "inherent":
                I_current = compute_I_inherent(E_proxy, config)
            elif I_mode == "threshold":
                I_current = compute_I_threshold(E_proxy, I_current, config)
            
            I_history.append(I_current)

            H_current_static = H_static_base
            cseg = list(c_ops)
            
            if s > 0:
                rscale = lockin_rate_scale(E_proxy, I_current, config)
                cseg = [np.sqrt(rscale) * cc for cc in cseg]
                
                if config["ANHARMONIC_X4"] or config["DOUBLE_WELL"]:
                    pscale = lockin_potential_scale(E_proxy, I_current, config)
                    if config["DOUBLE_WELL"]:
                        H1_dyn_local = omega1*(a1.dag()*a1) + (pscale * dw_c2)*(x1**2) + (pscale * dw_c4)*(x1**4)
                    else:
                        H1_dyn_local = omega1*(a1.dag()*a1) + (pscale * lam_x4)*(x1**4)

                    if two_mode and a2 is not None:
                        H2_local = omega2*(a2.dag() * a2)
                        H_current_static = tensor(H1_dyn_local, ident) + tensor(ident, H2_local) + g_coup*tensor(a1+a1.dag(), a2+a2.dag())
                    else:
                        H_current_static = H1_dyn_local

            # Handle time-dependent drive
            if config["TIME_DEP_DRIVE"] and isinstance(H, list):
                Hseg = [H_current_static, H[1]]
            else:
                Hseg = H_current_static
            res = mesolve(Hseg, state_in, t_solve_seg, cseg, [], options=opts)

            if not res.states or len(res.states) == 0:
                break
            
            state_in = res.states[-1]
            start_index = 1 if len(all_states_collected) > 0 else 0
            all_states_collected.extend(res.states[start_index:])

        if len(all_states_collected) < 2:
            return None

        e_series, Svon_series, coh_series, info_drift = [], [], [], []
        uncertainty_series, delta_x_series, delta_p_series = [], [], []
        initial_rho = all_states_collected[0]
        if two_mode:
            rho_prev_pop = np.real(np.diag(initial_rho.ptrace(0).full()).flatten())
        else:
            rho_prev_pop = np.real(np.diag(initial_rho.full()).flatten())

        for rho in all_states_collected[1:]:
            if two_mode:
                e_series.append(expect(Nop1_full, rho) + (expect(Nop2_full, rho) if Nop2_full is not None else 0))
                rho1 = rho.ptrace(0)
                coh_series.append(normalize_coherence(rho, mode=0))
                rho_pop = np.real(np.diag(rho1.full()).flatten())
                
                if x_op_mode1_cached is not None and p_op_mode1_cached is not None:
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1_cached, p_op_mode1_cached, config)
                else:
                    x_op_mode1 = tensor(x1, ident)
                    p_op_mode1 = tensor(p1, ident)
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1, p_op_mode1, config)
            else:
                e_series.append(expect(Nop1_full, rho))
                rho_pop = np.real(np.diag(rho.full()).flatten())
                coh_series.append(normalize_coherence(rho))
                dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x1, p1, config)
            
            delta_x_series.append(dx)
            delta_p_series.append(dp)
            uncertainty_series.append(unc_prod)
            Svon_series.append(float(entropy_vn(rho)))

            p = (rho_prev_pop + 1e-12); p /= (p.sum() + 1e-15)
            q = (rho_pop + 1e-12); q /= (q.sum() + 1e-15)
            m = 0.5 * (p + q)
            js = 0.5 * shannon_entropy(p, m, base=2) + 0.5 * shannon_entropy(q, m, base=2)
            info_drift.append(float(js))
            rho_prev_pop = rho_pop
        
        if not e_series:
            return None
        
        return dict(
            energies=np.array(e_series),
            entropy=np.array(Svon_series),
            coherence=np.array(coh_series),
            info_drift=np.array(info_drift),
            delta_x=np.array(delta_x_series),
            delta_p=np.array(delta_p_series),
            uncertainty_product=np.array(uncertainty_series),
            I_evolution=np.array(I_history) if I_mode != "fixed" else None,
            I_mode=I_mode,
            final_state=all_states_collected[-1],
            final_energy=e_series[-1]
        )

