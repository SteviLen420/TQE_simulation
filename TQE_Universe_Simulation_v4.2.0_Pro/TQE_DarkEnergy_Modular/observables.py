# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# observables.py - Observable Predictions Module
# ==========================================================================================
# TQE–ΛSim: Observable predictions for cosmological observations
# ==========================================================================================

import numpy as np
from .config import MASTER_CTRL, c_light

class ObservablePredictions:
    # Compute observable quantities for model comparison
    # SNe Ia Hubble diagram, BAO, CMB C_ℓ, LSS power spectrum
    
    def __init__(self, friedmann_evolution):
        # Initialize observable predictions
        
        self.friedmann = friedmann_evolution
        
        print(f"✓ Observable predictions module initialized")
    
    def sne_hubble_diagram(self, z_array):
        # Predict SNe Ia Hubble diagram: μ(z)
        # Returns array of distance moduli
        
        mu_array = np.array([self.friedmann.distance_modulus(z) for z in z_array])
        
        return mu_array
    
    def bao_observables(self, z_array):
        # Predict BAO observables: D_M(z), H(z)
        # D_M: comoving transverse distance (NOT angular diameter distance!)
        # For flat (k=0) cosmology: D_M(z) = D_C(z)
        # H: Hubble parameter
        
        D_M_array = np.array([self.friedmann.comoving_transverse_distance(z) for z in z_array])
        H_array = np.array([self.friedmann.H(1.0 / (1.0 + z)) for z in z_array])
        
        return D_M_array, H_array
    
    def cmb_power_spectrum(self, use_camb=True):
        # Predict CMB power spectrum C_ℓ
        # Uses CAMB if available, otherwise simplified calculation
        # WARNING: CMB predictions use baseline ΛCDM parameters
        # I-parameter coupling effects are NOT included in CMB calculation
        
        if use_camb and CAMB_AVAILABLE:
            # Use CAMB for accurate CMB prediction
            # NOTE: This uses standard ΛCDM parameters - I-parameter effects not included
            print("⚠ CMB calculation: Using baseline ΛCDM (I-parameter effects not included)")
            pars = camb.CAMBparams()
            pars.set_cosmology(
                H0=self.friedmann.H0,
                ombh2=self.friedmann.Omega_b * (self.friedmann.H0/100)**2,
                omch2=(self.friedmann.Omega_m - self.friedmann.Omega_b) * (self.friedmann.H0/100)**2
            )
            pars.InitPower.set_params(ns=self.friedmann.params['n_s'])
            pars.set_for_lmax(MASTER_CTRL['CMB_LMAX'], lens_potential_accuracy=0)
            
            # Calculate results
            results = camb.get_results(pars)
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
            totCL = powers['total']
            
            ell = np.arange(totCL.shape[0])
            
            return ell, totCL[:, 0]  # TT spectrum
        
        else:
            # Simplified CMB prediction (placeholder)
            print("⚠ CAMB not available - using simplified CMB prediction")
            print("⚠ CMB calculation: Baseline approximation (I-parameter effects not included)")
            ell = np.arange(MASTER_CTRL['CMB_LMIN'], MASTER_CTRL['CMB_LMAX'])
            # Simplified Sachs-Wolfe plateau
            C_ell = 5000.0 / (ell * (ell + 1)) * 2.0 * np.pi
            return ell, C_ell
    
    def matter_power_spectrum(self, k_array, z=0):
        # Predict matter power spectrum P(k,z)
        # NOTE: This is a VISUAL/DIAGNOSTIC approximation only
        # For accurate LSS predictions, use CAMB/CLASS with modified background
        # 
        # Simplified power spectrum: P(k) ∝ k^n_s · T²(k) · D²(z)
        # T(k): transfer function (simplified)
        # D(z): growth factor
        
        n_s = self.friedmann.params['n_s']
        sigma_8 = self.friedmann.params['sigma_8']
        
        # Simplified transfer function
        k_eq = 0.073 * self.friedmann.Omega_m * (self.friedmann.H0/100)**2  # Mpc^-1
        T_k = np.log(1.0 + 2.34 * k_array / k_eq) / (2.34 * k_array / k_eq)
        
        # Power spectrum normalization
        P_k = k_array**n_s * T_k**2 * sigma_8**2
        
        return P_k
    
    def sigma8_from_pk(self, z=0.0):
        """
        PRODUCTION HARDENING: Compute σ₈ from P(k) integral (not from parameter)
        
        σ₈² = (1/2π²) ∫ P(k) W²(k·R_8) k² dk
        
        where W(x) = 3(sin(x) - x·cos(x))/x³ is the top-hat filter
        and R_8 = 8 Mpc/h is the filtering scale
        
        Args:
            z: redshift
        
        Returns:
            sigma8: RMS fluctuation amplitude at R=8 Mpc/h
        """
        # PRODUCTION: Check computation method
        # S8_FROM_PARAM=False → compute from P(k), True → use parameter
        if MASTER_CTRL.get('S8_FROM_PARAM', False):
            # OLD METHOD: Return fixed parameter value (testing/legacy)
            return MASTER_CTRL.get('SIGMA_8', 0.811)
        
        # NEW METHOD: Compute from P(k) integral
        
        # Define k-space grid (log-spaced for better integration)
        k_min = 1e-4  # h/Mpc
        k_max = 10.0  # h/Mpc
        n_k = 500     # Integration points
        k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
        
        # Get matter power spectrum at redshift z
        # If CAMB/CLASS available, use it; otherwise use simplified P(k)
        try:
            if CAMB_AVAILABLE:
                # Use CAMB for accurate P(k)
                import camb
                pars = camb.CAMBparams()
                pars.set_cosmology(H0=self.friedmann.H0, ombh2=self.friedmann.Omega_b*(self.friedmann.H0/100)**2,
                                   omch2=(self.friedmann.Omega_m-self.friedmann.Omega_b)*(self.friedmann.H0/100)**2)
                pars.InitPower.set_params(ns=self.friedmann.params.get('n_s', 0.965))
                pars.set_matter_power(redshifts=[z], kmax=k_max)
                results = camb.get_results(pars)
                kh, z_arr, pk = results.get_matter_power_spectrum(minkh=k_min, maxkh=k_max, npoints=n_k)
                Pk_grid = pk[0, :]  # z=0 index
            else:
                # Use simplified Eisenstein-Hu approximation
                Pk_grid = self.matter_power_spectrum(k_grid, z=z)
        except (ImportError, AttributeError, IndexError) as e:
            # Fallback to simplified P(k) if CAMB unavailable
            print(f"⚠ CAMB P(k) failed, using simplified Eisenstein-Hu: {e}")
            Pk_grid = self.matter_power_spectrum(k_grid, z=z)
        
        # Top-hat window function W(x) = 3(sin(x) - x·cos(x))/x³
        R_8 = 8.0  # Mpc/h
        
        def window_tophat(x):
            """Top-hat filter in Fourier space"""
            # Handle x→0 limit: W(0) = 1
            x = np.atleast_1d(x)
            W = np.zeros_like(x)
            
            # Small x: Taylor expansion W(x) ≈ 1 - x²/10 + ...
            small_mask = np.abs(x) < 1e-3
            W[small_mask] = 1.0 - x[small_mask]**2 / 10.0
            
            # Normal x
            large_mask = ~small_mask
            x_large = x[large_mask]
            W[large_mask] = 3.0 * (np.sin(x_large) - x_large * np.cos(x_large)) / x_large**3
            
            return W
        
        # Compute σ₈² = (1/2π²) ∫ P(k) W²(kR₈) k² dk
        kR = k_grid * R_8
        W_kR = window_tophat(kR)
        
        # Integrand: P(k) · W²(kR) · k²
        integrand = Pk_grid * W_kR**2 * k_grid**2
        
        # Integrate using trapezoidal rule in log-space (more accurate)
        sigma8_squared = np.trapz(integrand, k_grid) / (2.0 * np.pi**2)
        
        # Safety: ensure positive
        sigma8_squared = max(sigma8_squared, 1e-20)
        sigma8 = np.sqrt(sigma8_squared)
        
        return sigma8
    
    def S8_parameter(self, z=0.0):
        """
        Compute S₈ parameter: S₈ = σ₈ · √(Ω_m/0.3)
        
        PRODUCTION HARDENING: If S8_FROM_PARAM=False, compute σ₈ from P(k) integral
        
        Args:
            z: redshift
        
        Returns:
            S8: Structure formation parameter
        """
        # Compute σ₈ (either from parameter or P(k) integral)
        sigma8 = self.sigma8_from_pk(z)
        
        # Compute Ω_m(z)
        a = 1.0 / (1.0 + z)
        Omega_m_z, _, _ = self.friedmann.Omega_components(a)
        E_z = self.friedmann.E(a)
        Omega_m_normalized = Omega_m_z / E_z**2
        
        # S₈ = σ₈ · √(Ω_m/0.3)
        S8 = sigma8 * np.sqrt(Omega_m_normalized / 0.3)
        
        return S8
    
    def compute_likelihood(self):
        # Compute likelihood from SNe, BAO, H0 prior
        # Returns: chi2_total, components dict
        # NOW SUPPORTS REAL DATA! (Pantheon+, BOSS)
        
        chi2_components = {}
        
        # 1. SNe Ia likelihood
        if MASTER_CTRL.get('USE_REAL_SNE_DATA', False):
            # Load real Pantheon+ data
            z_sne, mu_obs, sigma_mu, cov_sne = load_pantheon_plus_data(
                MASTER_CTRL.get('PANTHEON_PLUS_DATA_PATH', None),
                MASTER_CTRL.get('PANTHEON_PLUS_COV_PATH', None)
            )
        else:
            # Use enhanced mock data (50 points)
            z_sne, mu_obs, sigma_mu, cov_sne = load_pantheon_plus_data(None, None)
        
        mu_model = np.array([self.friedmann.distance_modulus(z) for z in z_sne])
        chi2_sne = np.sum(((mu_obs - mu_model) / sigma_mu)**2)
        chi2_components['SNe'] = chi2_sne
        
        # 2. BAO likelihood (D_M/r_d + H(z))
        if MASTER_CTRL.get('USE_REAL_BAO_DATA', False):
            # Load real BOSS data
            z_bao, DM_over_rd_obs, sigma_DM, H_obs, sigma_H, cov_bao = load_boss_bao_data(
                MASTER_CTRL.get('BOSS_BAO_DATA_PATH', None),
                MASTER_CTRL.get('BOSS_BAO_COV_PATH', None)
            )
        else:
            # Use enhanced mock data (10 points)
            z_bao, DM_over_rd_obs, sigma_DM, H_obs, sigma_H, cov_bao = load_boss_bao_data(None, None)
        
        r_d = 147.0  # Fiducial sound horizon (Mpc)
        DM_model = np.array([self.friedmann.comoving_transverse_distance(z) for z in z_bao])
        DM_over_rd_model = DM_model / r_d
        
        chi2_bao = np.sum(((DM_over_rd_obs - DM_over_rd_model) / sigma_DM)**2)
        chi2_components['BAO_DM'] = chi2_bao
        
        # BAO H(z) likelihood (if available)
        # Convert H_obs to array and handle None values
        H_obs_array = np.array(H_obs, dtype=float)  # Convert, None → NaN
        if not np.all(np.isnan(H_obs_array)):
            H_model = np.array([self.friedmann.H(1.0 / (1.0 + z)) for z in z_bao])
            valid_mask = ~np.isnan(H_obs_array)
            if np.any(valid_mask):
                sigma_H_array = np.array(sigma_H, dtype=float)
                chi2_H = np.sum(((H_obs_array[valid_mask] - H_model[valid_mask]) / sigma_H_array[valid_mask])**2)
                chi2_components['BAO_H'] = chi2_H
            else:
                chi2_components['BAO_H'] = 0.0
        else:
            chi2_components['BAO_H'] = 0.0
        
        # 3. H0 prior (Gaussian)
        H0_obs = 67.4  # Planck 2018
        sigma_H0 = 0.5
        chi2_H0 = ((self.friedmann.H0 - H0_obs) / sigma_H0)**2
        chi2_components['H0_prior'] = chi2_H0
        
        # 4. CMB (if enabled in MASTER_CTRL)
        # PRODUCTION HARDENING: CMB_REFERENCE_ONLY flag disables CMB contribution
        if MASTER_CTRL.get('CMB_REFERENCE_ONLY', True):
            # CMB is baseline ΛCDM reference only (no I-parameter effects)
            # Do NOT include in χ² until I-aware Boltzmann solver integrated
            chi2_components['CMB'] = 0.0
        elif MASTER_CTRL.get('INCLUDE_CMB_IN_LIKE', False):
            # Simplified CMB constraint: Ω_m h^2
            Omega_m_h2_obs = 0.1430
            sigma_Omega_m_h2 = 0.0011
            Omega_m_h2_model = self.friedmann.Omega_m * (self.friedmann.H0 / 100.0)**2
            chi2_cmb = ((Omega_m_h2_model - Omega_m_h2_obs) / sigma_Omega_m_h2)**2
            chi2_components['CMB'] = chi2_cmb
        else:
            chi2_components['CMB'] = 0.0
        
        # Total chi2
        chi2_total = sum(chi2_components.values())
        
        # Compute AIC, BIC (need coupling_type from external context)
        n_data = len(z_sne) + len(z_bao) + 1  # SNe + BAO + H0
        if MASTER_CTRL.get('INCLUDE_CMB_IN_LIKE', False):
            n_data += 1
        
        # Store for return
        likelihood_results = {
            'chi2_total': chi2_total,
            'chi2_components': chi2_components,
            'n_data': n_data
        }
        
        return likelihood_results
    
# ==========================================================================================
# VALIDATION AND SANITY CHECKS
# ==========================================================================================

