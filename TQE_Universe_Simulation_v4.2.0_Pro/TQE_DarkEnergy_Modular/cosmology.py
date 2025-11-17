# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# cosmology.py - Cosmology Module
# ==========================================================================================
# TQE–ΛSim: Friedmann evolution and cosmological calculations
# ==========================================================================================

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from .config import MASTER_CTRL, FIDUCIAL_PARAMS, c_light
from .tqe_core import CouplingModel

class FriedmannEvolution:
    """
    TQE-Modified Friedmann Evolution
    
    Implements the standard Friedmann equations with TQE dark energy coupling:
    
    FRIEDMANN EQUATION:
        H²(a)/H₀² = E²(a) = Ω_m(a) + Ω_r(a) + Ω_DE(a)
        
        where Ω_DE(a) is modulated by TQE coupling: ρ_DE = ρ_Λ · f(E,I)
    
    TQE CONNECTION:
        The I-parameter (information orientation) affects dark energy density through
        the coupling models, which translates the TQE fine-tuning function f(E,I)
        into observable cosmological quantities.
    
    PHYSICS:
        - Matter: Ω_m(a) = Ω_m,0 · a⁻³ (dust, standard dilution)
        - Radiation: Ω_r(a) = Ω_r,0 · a⁻⁴ (photons + neutrinos)
        - Dark Energy: Ω_DE(a) from TQE coupling (model-dependent)
    
    OUTPUTS:
        - Hubble parameter: H(a) [km/s/Mpc]
        - Distances: D_C(z), D_A(z), D_L(z), D_M(z) [Mpc]
        - Growth factor: D(a) and growth rate f(a) = d ln D/d ln a
    """
    
    def __init__(self, coupling_model, fiducial_params=None):
        """
        Initialize TQE-modified Friedmann evolution
        
        Args:
            coupling_model: CouplingModel instance (implements ρ_DE(a) with I-coupling)
            fiducial_params: Cosmological parameters (H₀, Ω_m, Ω_Λ, etc.)
        """
        
        self.coupling = coupling_model
        self.params = fiducial_params if fiducial_params is not None else FIDUCIAL_PARAMS.copy()
        
        # Extract parameters
        self.H0 = self.params['H0']
        self.Omega_m = self.params['Omega_m']
        self.Omega_Lambda = self.params['Omega_Lambda']
        self.Omega_b = self.params['Omega_b']
        
        # Compute Omega_r from N_eff and T_CMB (ΛCDM standard)
        # Ω_r h² = Ω_γ h² · (1 + 0.2271·N_eff)
        # Ω_γ h² = 2.469e-5 · (T_CMB/2.7255)⁴
        N_eff = MASTER_CTRL.get('N_EFF', 3.046)
        T_CMB = MASTER_CTRL.get('T_CMB', 2.7255)
        h = self.H0 / 100.0
        
        Omega_gamma_h2 = 2.469e-5 * (T_CMB / 2.7255)**4
        Omega_r_h2 = Omega_gamma_h2 * (1.0 + 0.2271 * N_eff)
        self.Omega_r = Omega_r_h2 / h**2
        
        # Store h for later use
        self.h = h
        
        # Compute rho_Lambda today (base cosmological constant)
        self.rho_Lambda_today = self.Omega_Lambda
        
        # Flatness check: Ω_total(a=1) should be ~1.0
        Omega_total_today = self.Omega_m + self.Omega_r + self.Omega_Lambda
        flatness_error = abs(Omega_total_today - 1.0)
        
        if flatness_error > 1e-3:
            msg = f"Universe not flat! Ω_total = {Omega_total_today:.6f} (should be 1.0), " \
                  f"Ω_m = {self.Omega_m:.6f}, Ω_r = {self.Omega_r:.6e}, Ω_Λ = {self.Omega_Lambda:.6f}"
            
            if MASTER_CTRL.get('STRICT_FLATNESS', False):
                # FAIL-FAST mode: raise exception
                raise ValueError(f"❌ STRICT FLATNESS VIOLATION: {msg}")
            else:
                # WARNING mode: just print
                print(f"⚠ WARNING: {msg}")
        
        print(f"✓ Friedmann evolution initialized")
        print(f"  H0 = {self.H0:.2f} km/s/Mpc, h = {self.h:.4f}")
        print(f"  Ω_m = {self.Omega_m:.3f}")
        print(f"  Ω_Λ = {self.Omega_Lambda:.3f}")
        print(f"  Ω_r = {self.Omega_r:.6e} (N_eff = {N_eff:.3f}, T_CMB = {T_CMB:.4f} K)")
        print(f"  Ω_total = {Omega_total_today:.6f} (flatness check)")
    
    def Omega_components(self, a):
        """
        Compute density components at scale factor a
        
        STANDARD COSMOLOGY:
            - Matter: Ω_m(a) = Ω_m,0 · a⁻³ (dust dilution)
            - Radiation: Ω_r(a) = Ω_r,0 · a⁻⁴ (photon + redshift dilution)
        
        TQE MODIFICATION:
            - Dark Energy: Ω_DE(a) = ρ_DE(a)/ρ_crit,0
              where ρ_DE(a) is modulated by I-parameter coupling
              
              Coupling models implement:
              • E-only:  ρ_DE = ρ_Λ · exp(-α·E)
              • E+I:     ρ_DE = ρ_Λ · exp(-α·E·(1-I))
              
              This translates TQE's f(E,I) into observable dark energy evolution
        
        Returns:
            (Omega_m, Omega_r, Omega_DE): All dimensionless (fraction of critical density)
        """
        
        # Standard matter and radiation evolution
        Omega_m_a = self.Omega_m * a**(-3)
        Omega_r_a = self.Omega_r * a**(-4)
        
        # TQE-modified dark energy (passes friedmann=self for dynamic I-parameter access)
        Omega_DE_a = self.coupling.rho_DE(a, self.rho_Lambda_today, friedmann=self)
        
        return Omega_m_a, Omega_r_a, Omega_DE_a
    
    def E_squared(self, a):
        """
        TQE-Modified Friedmann Equation (dimensionless form)
        
        EQUATION:
            E²(a) = H²(a)/H₀² = Ω_m(a) + Ω_r(a) + Ω_DE(a)
        
        TQE IMPACT:
            Ω_DE(a) is modulated by I-parameter coupling, causing deviations from
            standard ΛCDM. This affects:
            - Expansion history: H(z) evolution
            - Distance-redshift relations: D_L(z), D_A(z)
            - Structure growth: D(a) via modified expansion rate
        
        PHYSICAL CONSTRAINTS:
            E²(a) > 0 enforced (expansion rate must be real and positive)
        """
        
        Omega_m_a, Omega_r_a, Omega_DE_a = self.Omega_components(a)
        
        # Friedmann equation: sum of all energy density components
        E2 = Omega_m_a + Omega_r_a + Omega_DE_a
        
        # Physical guards
        if np.any(E2 <= 0):
            bad_indices = np.where(E2 <= 0)[0] if not np.isscalar(E2) else None
            if bad_indices is not None and len(bad_indices) > 0:
                a_bad = a if np.isscalar(a) else a[bad_indices[0]]
                E2_bad = E2 if np.isscalar(E2) else E2[bad_indices[0]]
                raise ValueError(f"Non-physical E²(a) <= 0 at a={a_bad:.4f}: E²={E2_bad:.6e}")
            elif np.isscalar(E2):
                raise ValueError(f"Non-physical E²(a) <= 0 at a={a:.4f}: E²={E2:.6e}")
        
        if np.any(~np.isfinite(E2)):
            raise ValueError(f"Non-finite E²(a) detected")
        
        return E2
    
    def E(self, a):
        # Dimensionless Hubble parameter: E(a) = H(a)/H₀
        # This is the standard cosmology notation
        E2 = self.E_squared(a)
        return np.sqrt(E2)
    
    def H(self, a):
        # Hubble parameter with units: H(a) [km/s/Mpc]
        # H(a) = H₀ · E(a)
        
        E_val = self.E(a)
        H_val = self.H0 * E_val
        
        # Physical guards
        if np.any(H_val <= 0):
            raise ValueError(f"Non-physical H(a) <= 0 detected")
        
        if np.any(~np.isfinite(H_val)):
            raise ValueError(f"Non-finite H(a) detected")
        
        return H_val
    
    def comoving_distance(self, z):
        # Compute comoving distance to redshift z
        # D_C(z) = c ∫_0^z dz'/H(z')
        # Using scale factor integration: D_C(z) = c ∫_a^1 da/(a²·H(a))
        
        # PRODUCTION: Use log-spaced grid for better early-universe resolution
        a_start = 1.0 / (1.0 + z)
        a_end = 1.0
        
        if MASTER_CTRL.get('USE_LOG_A_GRID', False):
            # Log-spaced grid (better for small a)
            n_grid = MASTER_CTRL.get('A_GRID_N_LOG', 4096)
            a_grid = np.exp(np.linspace(np.log(max(a_start, 1e-4)), np.log(a_end), n_grid))
        else:
            # Linear grid (default)
            a_grid = np.linspace(a_start, a_end, 2000)
        
        # Integrand: c/(a²·H(a))
        integrand = c_light / (a_grid**2 * self.H(a_grid))
        
        # Integrate using trapezoidal rule
        D_C = np.trapz(integrand, a_grid)
        
        return D_C
    
    def angular_diameter_distance(self, z):
        # Compute angular diameter distance: D_A(z) = D_C(z)/(1+z)
        if z <= -1.0:
            raise ValueError(f"Invalid redshift z={z:.4f}, must be > -1 for physical distances")
        D_C = self.comoving_distance(z)
        return D_C / (1.0 + z)
    
    def comoving_transverse_distance(self, z):
        # Comoving transverse distance D_M(z) = D_C(z) for flat (k=0) cosmology
        # This is the correct BAO distance measure (NOT D_A!)
        return self.comoving_distance(z)
    
    def luminosity_distance(self, z):
        # Compute luminosity distance: D_L(z) = D_C(z)·(1+z)
        D_C = self.comoving_distance(z)
        return D_C * (1.0 + z)
    
    def distance_modulus(self, z):
        # Compute distance modulus for SNe Ia: μ(z) = 5·log10(D_L/10pc)
        D_L_Mpc = self.luminosity_distance(z)
        mu = 5.0 * np.log10(D_L_Mpc) + 25.0  # 25 = 5·log10(10pc/Mpc)
        return mu
    
    def dlnH_dlna(self, a):
        # Compute d ln H / d ln a at scale factor a
        # Standard cosmology: d ln H / d ln a = -1/2 · [3Ω_m(a) + 4Ω_r(a) + 3(1+w(a))Ω_DE(a)] / E²(a)
        # This is needed for the growth factor ODE and equation of state analysis
        
        # PRODUCTION HARDENING: Can use w_eff_CPL(a) if CPL is active
        # This allows dynamic w(a) from CPL parameterization to affect growth
        
        Omega_m_a, Omega_r_a, Omega_DE_a = self.Omega_components(a)
        E2_a = Omega_m_a + Omega_r_a + Omega_DE_a
        
        # Get effective w(a) for dark energy
        # Priority: 1) CPL (if enabled), 2) Coupling model w_DE, 3) ΛCDM default
        w_eff = -1.0  # Default: cosmological constant
        
        # PRODUCTION: Check CPL first (highest priority)
        w_cpl = w_eff_CPL(a)
        if w_cpl is not None:
            # CPL is active - use it and skip coupling model w_DE
            w_eff = w_cpl
        elif hasattr(self.coupling, 'w_DE'):
            # No CPL - try coupling model w_DE
            try:
                w_eff = self.coupling.w_DE(a)
                # Clamp to physical range: w >= -1.3 (energy conditions)
                w_eff = np.maximum(w_eff, -1.3)
            except (AttributeError, ValueError, TypeError) as e:
                # Fallback to ΛCDM if coupling model fails
                w_eff = -1.0
        # else: w_eff = -1.0 (ΛCDM default)
        
        # d ln H / d ln a formula
        dlnH_dlna = -0.5 * (3*Omega_m_a + 4*Omega_r_a + 3*(1+w_eff)*Omega_DE_a) / E2_a
        
        return dlnH_dlna
    
    def growth_factor(self, z):
        # Compute linear growth factor D(z) using proper ODE
        # ODE: d²D/d(ln a)² + [2 + d ln H/d ln a] dD/d(ln a) - (3/2)Ω_m(a) D = 0
        # Initial conditions: D(a→0) → a (matter domination), dD/d(ln a) → a
        # Normalized to D(z=0) = 1
        
        from scipy.integrate import solve_ivp
        
        a_target = 1.0 / (1.0 + z)
        
        # Check if ODE growth is enabled
        use_ode_growth = MASTER_CTRL.get('USE_ODE_GROWTH', True)
        
        if use_ode_growth:
            # PROPER ODE SOLUTION for growth factor
            # d²D/d(ln a)² + [2 + d ln H/d ln a] dD/d(ln a) = (3/2)Ω_m(a) D
            
            def growth_ode(lna, y):
                # ODE system: y = [D, dD/d(ln a)]
                a_val = np.exp(lna)
                D, dD_dlna = y
                
                # Compute Ω_m(a) = Ω_m,0 · a^(-3) / E²(a)
                Omega_m_a, Omega_r_a, Omega_DE_a = self.Omega_components(a_val)
                E2_a = Omega_m_a + Omega_r_a + Omega_DE_a
                Omega_m_frac = Omega_m_a / E2_a
                
                # Compute d ln H / d ln a
                dlnH_dlna_val = self.dlnH_dlna(a_val)
                
                # ODE: d²D/d(ln a)² = -(2 + d ln H/d ln a) dD/d(ln a) + (3/2)Ω_m(a) D
                d2D_dlna2 = -(2.0 + dlnH_dlna_val) * dD_dlna + 1.5 * Omega_m_frac * D
                
                return [dD_dlna, d2D_dlna2]
            
            # Initial conditions at early time (deep matter domination)
            a_init = 0.001  # z ~ 999
            lna_init = np.log(a_init)
            lna_target = np.log(a_target)
            
            # At early times in matter domination: D ≈ a, dD/d(ln a) ≈ a
            D_init = a_init
            dD_dlna_init = a_init
            y0 = [D_init, dD_dlna_init]
            
            # Solve ODE with high precision
            try:
                # PRODUCTION HARDENING: Higher precision ODE solver
                rtol = MASTER_CTRL.get('ODE_GROWTH_RTOL', 1e-8)  # Relative tolerance
                atol = MASTER_CTRL.get('ODE_GROWTH_ATOL', 1e-10)  # Absolute tolerance
                max_step = MASTER_CTRL.get('ODE_GROWTH_MAX_STEP', 0.1)  # Max step in ln(a)
                
                sol = solve_ivp(growth_ode, (lna_init, lna_target), y0, 
                               method='RK45', rtol=rtol, atol=atol, 
                               dense_output=True, max_step=max_step)
                
                # Evaluate at target a
                D_at_target = sol.sol(lna_target)[0]
                
                # Normalize to D(z=0) = 1
                lna_z0 = np.log(1.0)
                D_at_z0 = sol.sol(lna_z0)[0]
                D_normalized = D_at_target / D_at_z0
                
                # Sanity check
                if not np.isfinite(D_normalized) or D_normalized < 0:
                    print(f"⚠ ODE growth unstable at z={z:.2f}, falling back to integral")
                    use_ode_growth = False  # Fall back to integral
                else:
                    return D_normalized
                    
            except Exception as e:
                print(f"⚠ ODE growth failed at z={z:.2f}: {e}, falling back to integral")
                use_ode_growth = False
        
        # FALLBACK: Integral approximation (if ODE disabled or failed)
        if not use_ode_growth:
            a_grid = np.linspace(0.01, a_target, 500)
            H_grid = np.array([self.H(a_val) for a_val in a_grid])
            integrand = 1.0 / (a_grid * H_grid**3)
            integral = np.trapz(integrand, a_grid)
            
            D_unnorm = self.H(a_target) * integral
            D_at_z0 = self.H(1.0) * np.trapz(1.0 / (np.linspace(0.01, 1.0, 500) * 
                                              np.array([self.H(a_val) for a_val in np.linspace(0.01, 1.0, 500)])**3),
                                              np.linspace(0.01, 1.0, 500))
            
            D_normalized = D_unnorm / D_at_z0
            
            # Safety checks
            if not np.isfinite(D_normalized):
                print(f"⚠ Growth factor NaN at z={z:.2f}, returning a (matter-dom fallback)")
                return a_target
            
            if D_normalized < 0:
                print(f"⚠ Negative growth factor D={D_normalized:.4f} at z={z:.2f}, clamping to a")
                return max(a_target, 0.0)
            
            return D_normalized

# ==========================================================================================
# DATA LOADERS (Real Observational Data)
# ==========================================================================================

