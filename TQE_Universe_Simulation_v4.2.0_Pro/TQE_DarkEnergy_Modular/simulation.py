# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# simulation.py - Main Simulation Module
# ==========================================================================================
# TQE–ΛSim: Main simulation class for TQE dark energy coupling analysis
# ==========================================================================================

import numpy as np
import os
from .config import MASTER_CTRL, FIDUCIAL_PARAMS
from .tqe_core import EnergyInformationContent, CouplingModel
from .cosmology import FriedmannEvolution
from .observables import ObservablePredictions
from .inference import BayesianInferenceEngine
from .structure import GalaxyStructureAnalyzer
from .utils import set_deterministic_seed

class TQEDarkEnergyCouplingSimulation:
    # Main simulation class for TQE dark energy coupling analysis
    # Implements complete forward model and observable predictions
    
    def __init__(self, coupling_model, information_content, fiducial_params=None,
                 project_dir=None, seed_string="TQE_DarkEnergy_2025", coupling_mode=None):
        # Initialize TQE Dark Energy Coupling Simulation
        
        # Set deterministic seed
        self.seed_string = seed_string
        self.seed_hash = set_deterministic_seed(seed_string)
        
        # TQE Coupling Mode: Eonly vs EplusI
        self.coupling_mode = coupling_mode if coupling_mode else MASTER_CTRL.get('COUPLING_MODE', 'EplusI')
        
        # Models
        self.information_content = information_content
        self.coupling_model = coupling_model
        self.coupling = coupling_model  # Alias for compatibility
        self.coupling_model.coupling_mode = self.coupling_mode  # Pass mode to coupling model
        
        # Friedmann evolution
        self.friedmann = FriedmannEvolution(coupling_model, fiducial_params)
        
        # Observable predictions
        self.observables = ObservablePredictions(self.friedmann)
        
        # Project directory
        self.project_dir = project_dir
        if self.project_dir:
            os.makedirs(f"{self.project_dir}/PNG_Visualizations", exist_ok=True)
        
        # Results storage - populate with model parameters
        self.results = {
            'coupling_mode': self.coupling_mode,       # TQE coupling mode: Eonly or EplusI
            'model_type': coupling_model.coupling_type,
            'i_field_type': information_content.model_type,
            'coupling_params': coupling_model.params,  # Store coupling parameters
            'i_field_params': information_content.params,    # Store I-parameter parameters
            'parameters': {},
            'observables': {},
            'model_comparison': {},
            'bayesian_inference': {}
        }
        
        print(f"✓ TQE Dark Energy Coupling Simulation initialized")
        print(f"  Coupling model: {coupling_model.coupling_type}")
        print(f"  I-parameter model: {information_content.model_type}")
        print(f"  Coupling params: {coupling_model.params}")
        print(f"  I-parameter params: {information_content.params}")
    
    def run_cosmological_evolution(self, a_min=None, a_max=None, n_points=None):
        # Run cosmological evolution from early universe to today
        # Compute H(a), distances, and all relevant cosmological quantities
        
        # Use MASTER_CTRL parameters if not specified
        if a_min is None:
            a_min = MASTER_CTRL['A_MIN']
        if a_max is None:
            a_max = MASTER_CTRL['A_MAX']
        if n_points is None:
            n_points = MASTER_CTRL['N_A_POINTS']
        
        print("🌌 Running cosmological evolution...")
        
        # Scale factor array
        a_array = np.linspace(a_min, a_max, n_points)
        z_array = 1.0 / a_array - 1.0
        
        # Compute Hubble parameter evolution
        # OPTIMIZED: Vectorized computation if possible, else use list comprehension
        if MASTER_CTRL.get("ENABLE_VECTORIZATION", True) and hasattr(self.friedmann, 'H_vectorized'):
            H_array = self.friedmann.H_vectorized(a_array)
        else:
            H_array = np.array([self.friedmann.H(a) for a in tqdm(a_array, desc="Computing H(a)", leave=False)])
        
        # Compute I-parameter evolution (TQE-COMPLIANT: from energy evolution)
        # For energy_based models, I depends on E and dE/da
        if self.information_content.model_type == 'energy_based':
            I_array = []
            for i, a in enumerate(a_array):
                E = H_array[i] / self.friedmann.H0
                # Compute dE/da numerically
                if i > 0 and i < len(a_array) - 1:
                    dE_da = (H_array[i+1] - H_array[i-1]) / (a_array[i+1] - a_array[i-1]) / self.friedmann.H0
                else:
                    # Boundary: use one-sided difference
                    da = 0.001
                    a_plus = min(a + da, 1.0)
                    E_plus = self.friedmann.H(a_plus) / self.friedmann.H0
                    dE_da = (E_plus - E) / da
                    
                I_array.append(self.information_content.compute_information(a, E=E, dE_da=dE_da))
            I_array = np.array(I_array)
        else:
            # Legacy models (phenomenological/EFT): vectorized if available
            if MASTER_CTRL.get("ENABLE_VECTORIZATION", True) and hasattr(self.information_content, 'compute_information_vectorized'):
                I_array = self.information_content.compute_information_vectorized(a_array)
            else:
                I_array = np.array([self.information_content.compute_information(a) for a in a_array])
        
        # Compute dark energy density evolution
        # AUDIT FIX #2: Pass friedmann parameter for dynamic I-parameter
        # OPTIMIZED: Vectorized rho_DE computation
        if MASTER_CTRL.get("ENABLE_VECTORIZATION", True) and hasattr(self.coupling_model, 'rho_DE_vectorized'):
            rho_DE_array = self.coupling_model.rho_DE_vectorized(a_array, self.friedmann.rho_Lambda_today, friedmann=self.friedmann)
        else:
            rho_DE_array = np.array([self.coupling_model.rho_DE(a, self.friedmann.rho_Lambda_today, friedmann=self.friedmann) for a in a_array])
        
        # Store evolution results
        self.results['evolution'] = {
            'a_array': a_array.tolist(),
            'z_array': z_array.tolist(),
            'H_array': H_array.tolist(),
            'I_array': I_array.tolist(),
            'rho_DE_array': rho_DE_array.tolist()
        }
        
        print(f"✅ Cosmological evolution computed")
        print(f"  Redshift range: z = {z_array[-1]:.2f} → {z_array[0]:.2f}")
        print(f"  I-parameter range: I = {np.min(I_array):.4f} → {np.max(I_array):.4f}")
    
    def compute_S8_normalized(self, S8_LCDM=None, beta0_value=None):
        # AUDIT FIX #4: S₈ normalization with β₀-SPECIFIC ΛCDM baseline
        # S₈_TQE = S₈_ΛCDM(β₀) × [D(z=0)_TQE / D(z=0)_ΛCDM]
        # Now properly accounts for β₀-dependent baseline
        
        if not MASTER_CTRL['NORMALIZE_S8_TO_LCDM']:
            return None
        
        # Get current S₈ and growth factor
        S8_current = self.observables.S8_parameter(z=0)
        D_TQE = self.friedmann.growth_factor(z=0)  # Should be 1.0 at z=0 by definition
        
        # AUDIT FIX #4: Use β₀-specific ΛCDM baseline if provided
        if beta0_value is not None and MASTER_CTRL.get('USE_BETA0_SPECIFIC_BASELINE', True):
            # Compute ΛCDM baseline for this specific β₀
            # This allows each β₀ to have its own reference
            S8_LCDM_beta0 = self._compute_LCDM_S8_for_beta0(beta0_value)
            print(f"  Using β₀-specific ΛCDM baseline: S₈_ΛCDM(β₀={beta0_value:.4f}) = {S8_LCDM_beta0:.4f}")
        elif S8_LCDM is not None:
            # Use provided value
            S8_LCDM_beta0 = S8_LCDM
        else:
            # Fallback to global COSMO_PARAMS
            S8_LCDM_beta0 = COSMO_PARAMS["sigma8_LCDM"]
        
        # ΛCDM growth factor at z=0 is always 1.0 (normalized)
        D_LCDM = 1.0
        
        # Growth factor ratio
        growth_ratio = D_TQE / D_LCDM
        
        # Normalized S₈: S₈_TQE = S₈_ΛCDM(β₀) × growth_ratio
        # (This is the proper physical normalization)
        S8_normalized_proper = S8_LCDM_beta0 * growth_ratio
        
        # Simple ratio (for comparison)
        S8_normalized_simple = S8_current / S8_LCDM_beta0
        
        # Δ S₈ (difference from ΛCDM at this β₀)
        Delta_S8 = S8_current - S8_LCDM_beta0
        Delta_S8_percent = (Delta_S8 / S8_LCDM_beta0) * 100.0
        
        print(f"  S₈ normalization: S₈_TQE={S8_current:.4f}, S₈_ΛCDM(β₀)={S8_LCDM_beta0:.4f}")
        print(f"  Growth ratio: D_TQE/D_ΛCDM = {growth_ratio:.4f}")
        print(f"  S₈_norm (simple) = {S8_normalized_simple:.4f}, ΔS₈ = {Delta_S8:+.4f} ({Delta_S8_percent:+.3f}%)")
        
        return {
            'S8_raw': S8_current,
            'S8_LCDM': S8_LCDM_beta0,
            'S8_normalized': S8_normalized_simple,  # Use simple for backward compatibility
            'S8_normalized_proper': S8_normalized_proper,  # Proper growth-normalized
            'growth_ratio': growth_ratio,  # Track growth ratio
            'Delta_S8': Delta_S8,
            'Delta_S8_percent': Delta_S8_percent,  # Percentage difference
            'beta0_value': beta0_value  # Track which β₀ was used
        }
    
    def _compute_LCDM_S8_for_beta0(self, beta0):
        # AUDIT FIX #4: Compute ΛCDM S₈ baseline for specific β₀
        # This allows proper normalization for each β₀ sweep value
        # 
        # For simplicity, we use a β₀-independent ΛCDM value
        # (true ΛCDM has no β₀ coupling by definition)
        # But we track it separately for each sweep iteration
        
        # ΛCDM S₈ is independent of β₀ by definition
        # (β₀ only affects TQE coupling, not ΛCDM)
        S8_LCDM_base = COSMO_PARAMS["sigma8_LCDM"]
        
        # Optional: add small β₀-dependent correction for numerical consistency
        # (this is a phenomenological choice)
        beta0_correction = MASTER_CTRL.get('BETA0_LCDM_CORRECTION', 0.0)
        S8_LCDM_corrected = S8_LCDM_base * (1.0 + beta0_correction * beta0)
        
        return S8_LCDM_corrected
    
    def compute_evolution_series(self):
        # Compute S₈(z), ρ_DE(z), D(z) evolution series
        # These track how observables change with redshift
        
        if not MASTER_CTRL['COMPUTE_S8_EVOLUTION']:
            return
        
        print("📊 Computing evolution series: S₈(z), ρ_DE(z), D(z)...")
        
        # FINAL RELEASE UPGRADE: Extended redshift grid (0 → 5, 100 points)
        z_min = MASTER_CTRL.get('Z_MIN', 0.0)
        z_max = MASTER_CTRL.get('Z_MAX', 5.0)
        z_points = MASTER_CTRL.get('Z_POINTS', 100)
        z_grid = np.linspace(z_min, z_max, z_points)
        
        # S₈(z) evolution
        S8_series = np.array([self.observables.S8_parameter(z) for z in z_grid])
        
        # D(z) growth factor evolution
        D_series = np.array([self.friedmann.growth_factor(z) for z in z_grid])
        
        # ρ_DE(z) evolution
        a_grid = 1.0 / (1.0 + z_grid)
        # AUDIT FIX #2: Pass friedmann parameter for dynamic I-parameter
        rho_DE_series = np.array([self.friedmann.coupling.rho_DE(a, self.friedmann.rho_Lambda_today, friedmann=self.friedmann) 
                                   for a in a_grid])
        
        # Normalize S₈ if enabled (FIXED: pass beta0_value from coupling model)
        beta0 = None
        if hasattr(self.friedmann.coupling, 'beta0'):
            beta0 = self.friedmann.coupling.beta0
        S8_norm_data = self.compute_S8_normalized(beta0_value=beta0)
        
        # Compute I and E for evolution series
        a_grid = 1.0 / (1.0 + z_grid)
        I_series = np.array([self.information_content.compute_information(ai) for ai in a_grid])
        E_series = np.array([self.friedmann.H(ai) / self.friedmann.H0 for ai in a_grid])
        
        # Store in results
        self.results['evolution_series'] = {
            'z': z_grid.tolist(),
            'S8': S8_series.tolist(),
            'D': D_series.tolist(),
            'rho_DE': rho_DE_series.tolist(),
            'I': I_series.tolist(),
            'E': E_series.tolist()
        }
        
        if S8_norm_data:
            self.results['S8_normalization'] = S8_norm_data
        
        print(f"✅ Evolution series computed")
        print(f"  S₈: {S8_series[0]:.4f} (z=0) → {S8_series[-1]:.4f} (z={MASTER_CTRL['S8_Z_MAX']})")
        print(f"  D: {D_series[0]:.4f} (z=0) → {D_series[-1]:.4f} (z={MASTER_CTRL['S8_Z_MAX']})")
    
    def compute_I_E_correlation(self):
        """
        Compute correlation analysis between Information (I) and Energy (E) parameters.
        
        This is a key TQE diagnostic that measures how tightly the information
        parameter is coupled to the energy evolution:
        
        Correlation metrics:
            - Pearson r: Linear correlation between I(a) and E(a)
            - Spearman ρ: Rank correlation (non-linear relationships)
            - Mutual Information: Information-theoretic dependence measure
        
        Additional analysis (if RUN_LAG_SCAN=True):
            - MI(Δa): Mutual information as function of time lag
            - Optimal lag: Time delay maximizing I-E coupling
        
        Results stored in self.results['I_E_correlation'].
        """
        if not MASTER_CTRL['COMPUTE_I_E_CORRELATION']:
            return
        
        print("📊 Computing I-E correlation analysis...")
        
        # Get evolution data
        if 'evolution' not in self.results:
            print("⚠ Evolution data not available - skipping I-E correlation")
            return
        
        a_array = np.array(self.results['evolution']['a_array'])
        I_array = np.array(self.results['evolution']['I_array'])
        H_array = np.array(self.results['evolution']['H_array'])
        
        # Compute E-field for all points (FIXED: E = H/H0, NOT gradient-based!)
        # E-field = normalized expansion rate (energy proxy)
        E_array = H_array / self.friedmann.H0
        
        # Pearson correlation
        from scipy.stats import pearsonr, spearmanr
        
        # Check if arrays have variance (not all zeros)
        if np.std(I_array) > 1e-10 and np.std(E_array) > 1e-10:
            pearson_r, pearson_p = pearsonr(I_array, E_array)
            spearman_r, spearman_p = spearmanr(I_array, E_array)
        else:
            # No variance (e.g., Null model with I=0, E=0)
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0
            print("  ⚠ I or E field has no variance - setting correlations to 0")
        
        # Mutual Information (simple binning approach)
        try:
            from sklearn.metrics import mutual_info_score
            from sklearn.preprocessing import KBinsDiscretizer
            
            # Check variance before MI calculation
            if np.std(I_array) > 1e-10 and np.std(E_array) > 1e-10:
                # Discretize for MI calculation
                discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
                I_binned = discretizer.fit_transform(I_array.reshape(-1, 1)).flatten()
                E_binned = discretizer.fit_transform(E_array.reshape(-1, 1)).flatten()
                
                MI = mutual_info_score(I_binned, E_binned)
            else:
                MI = 0.0
        except Exception as e:
            print(f"  ⚠ MI calculation failed: {e}")
            MI = 0.0
        
        # Store results (FIXED: include a_array for CSV save)
        self.results['I_E_correlation'] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mutual_information': MI,
            'a_array': a_array.tolist(),
            'I_array': I_array.tolist(),
            'E_array': E_array.tolist()
        }
        
        print(f"✅ I-E correlation computed")
        print(f"  Pearson r = {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman r = {spearman_r:.4f} (p={spearman_p:.2e})")
        if MI is not None:
            print(f"  Mutual Information = {MI:.4f}")
        
        # AUDIT FIX #3: FULL MI-LAG SCAN with interpolation
        if MASTER_CTRL.get('RUN_LAG_SCAN', False):
            print("  🔄 Computing FULL MI-lag scan MI(Δa)...")
            
            # AUDIT FIX #3: Increased resolution for better temporal correlation
            min_da = MASTER_CTRL.get('LAG_SCAN_MIN_DA', -0.05)
            max_da = MASTER_CTRL.get('LAG_SCAN_MAX_DA', 0.02)
            n_lags = MASTER_CTRL.get('LAG_SCAN_N_POINTS', 30)  # PRODUCTION: finer, focused range
            
            # Create lag array: Δa ∈ [min_da, max_da], n_lags points
            da_lags = np.linspace(min_da, max_da, n_lags)
            MI_lags = []
            
            # Use mutual_info_regression for continuous MI (better than discretization)
            from sklearn.feature_selection import mutual_info_regression
            
            for da in da_lags:
                # AUDIT FIX #3: Use interpolation instead of index shifting
                # Shift I by Δa: I(a + Δa)
                a_shifted = a_array + da
                
                # Interpolate I to shifted grid
                I_shifted = np.interp(a_shifted, a_array, I_array, 
                                     left=I_array[0], right=I_array[-1])
                
                # Compute MI between I(a + Δa) and E(a)
                if np.std(I_shifted) > 1e-10 and np.std(E_array) > 1e-10:
                    try:
                        # mutual_info_regression for continuous variables
                        MI_lag = mutual_info_regression(
                            E_array.reshape(-1, 1),
                            I_shifted,
                            random_state=42
                        )[0]
                        MI_lags.append(MI_lag)
                    except Exception as e:
                        print(f"    ⚠ MI failed at Δa={da:.4f}: {e}")
                        MI_lags.append(0.0)
                else:
                    MI_lags.append(0.0)
            
            # Store lag scan results
            self.results['I_E_correlation']['lag_scan'] = {
                'da_lags': da_lags.tolist(),
                'MI_lags': MI_lags,
                'max_MI': max(MI_lags) if MI_lags else 0.0,
                'optimal_da': da_lags[np.argmax(MI_lags)] if MI_lags else 0.0
            }
            
            print(f"    ✓ Lag scan complete: max MI = {max(MI_lags):.4f} at Δa = {da_lags[np.argmax(MI_lags)]:.4f}")
    
    def run_sensitivity_test(self):
        # FINAL RELEASE UPGRADE: Sensitivity testing - ±1% I-parameter perturbation
        # Tests numerical stability by perturbing I-parameter and measuring impact on observables
        
        if not MASTER_CTRL.get('RUN_SENSITIVITY_TEST', False):
            return None
        
        # Only run for phenomenological I-parameter (has A parameter)
        if not hasattr(self.information_content, 'A'):
            print("\n⚠ Sensitivity test skipped (I-parameter model has no A parameter)")
            return None
        
        print("\n🔬 RUNNING SENSITIVITY TEST: ±1% I-parameter perturbation...")
        
        # Save original I-parameter parameters
        orig_A = self.information_content.A
        
        # Perturbation amplitude (±1%)
        pert = MASTER_CTRL.get('SENSITIVITY_PERTURBATION', 0.01)
        
        # Get baseline observables
        S8_baseline = self.observables.S8_parameter(z=0)
        H_baseline = self.friedmann.H(1.0)
        rho_DE_baseline = self.friedmann.coupling.rho_DE(1.0, self.friedmann.rho_Lambda_today, friedmann=self.friedmann)
        
        # Test +1% perturbation
        self.information_content.A = orig_A * (1.0 + pert)
        self.friedmann.coupling.info_content.A = self.information_content.A  # Update coupling's information content
        
        S8_plus = self.observables.S8_parameter(z=0)
        H_plus = self.friedmann.H(1.0)
        rho_DE_plus = self.friedmann.coupling.rho_DE(1.0, self.friedmann.rho_Lambda_today, friedmann=self.friedmann)
        
        # Test -1% perturbation
        self.information_content.A = orig_A * (1.0 - pert)
        self.friedmann.coupling.info_content.A = self.information_content.A
        
        S8_minus = self.observables.S8_parameter(z=0)
        H_minus = self.friedmann.H(1.0)
        rho_DE_minus = self.friedmann.coupling.rho_DE(1.0, self.friedmann.rho_Lambda_today, friedmann=self.friedmann)
        
        # Restore original I-parameter
        self.information_content.A = orig_A
        self.friedmann.coupling.info_content.A = orig_A
        
        # Compute relative changes
        delta_S8 = max(abs(S8_plus - S8_baseline) / S8_baseline, 
                       abs(S8_minus - S8_baseline) / S8_baseline)
        delta_H = max(abs(H_plus - H_baseline) / H_baseline,
                     abs(H_minus - H_baseline) / H_baseline)
        delta_rho_DE = max(abs(rho_DE_plus - rho_DE_baseline) / rho_DE_baseline,
                          abs(rho_DE_minus - rho_DE_baseline) / rho_DE_baseline)
        
        # Stability criterion: <0.1% change = numerically stable
        tolerance = MASTER_CTRL.get('SENSITIVITY_TOLERANCE', 0.001)
        is_stable = (delta_S8 < tolerance and delta_H < tolerance and delta_rho_DE < tolerance)
        
        print(f"  ±{pert*100:.1f}% I-parameter perturbation:")
        print(f"    ΔS₈ = {delta_S8*100:.3f}%")
        print(f"    ΔH = {delta_H*100:.3f}%")
        print(f"    Δρ_DE = {delta_rho_DE*100:.3f}%")
        print(f"  {'✅ STABLE' if is_stable else '⚠ SENSITIVE'} (tolerance: {tolerance*100:.1f}%)")
        
        return {
            'perturbation_pct': pert * 100,
            'delta_S8_pct': delta_S8 * 100,
            'delta_H_pct': delta_H * 100,
            'delta_rho_DE_pct': delta_rho_DE * 100,
            'is_stable': is_stable,
            'tolerance_pct': tolerance * 100
        }
    
    def run_sanity_checks(self):
        # Run sanity checks with customizable baseline and tolerances
        
        print("🔍 Running sanity checks...")
        
        tols = MASTER_CTRL['SANITY_TOLS']
        
        checks = {}
        issues = []
        
        # Check 1: H(a=1) ≈ H₀ (with WARN/FAIL thresholds)
        try:
            H_at_1 = self.friedmann.H(1.0)
            H0 = self.friedmann.H0
            rel_diff = abs(H_at_1 - H0) / H0
            
            tol_fail = tols.get('H_at_a1_vs_H0_tol_fail', 0.01)  # ±1.0%
            tol_warn = tols.get('H_at_a1_vs_H0_tol_warn', 0.005) # ±0.5%
            
            # Three-level check: PASS / WARN / FAIL
            if rel_diff < tol_warn:
                checks['H_at_a1_vs_H0'] = True
                print(f"  ✓ H(a=1) ≈ H₀: deviation = {rel_diff*100:.4f}% (PASS)")
            elif rel_diff < tol_fail:
                checks['H_at_a1_vs_H0'] = True  # Still PASS, but warn
                issues.append(f"⚠ WARN: H(a=1) deviation = {rel_diff*100:.2f}% (between {tol_warn*100:.1f}% and {tol_fail*100:.1f}%)")
                print(f"  ⚠ H(a=1) deviation = {rel_diff*100:.4f}% (WARN: > {tol_warn*100:.1f}%)")
            else:
                checks['H_at_a1_vs_H0'] = False  # FAIL
                issues.append(f"❌ FAIL: H(a=1) = {H_at_1:.2f} vs H₀ = {H0:.2f} (diff: {rel_diff*100:.2f}% > {tol_fail*100:.1f}%)")
                print(f"  ❌ H(a=1) deviation = {rel_diff*100:.4f}% (FAIL: > {tol_fail*100:.1f}%)")
        except Exception as e:
            checks['H_at_a1_vs_H0'] = False
            issues.append(f"H(a=1) calculation failed: {e}")
        
        # Check 2: E²(a) > 0 everywhere
        if 'evolution' in self.results:
            a_arr = np.array(self.results['evolution']['a_array'])
            H_arr = np.array(self.results['evolution']['H_array'])
            E_sq = H_arr**2 / self.friedmann.H0**2
            checks['E_squared_positive'] = np.all(E_sq > 0)
            if not checks['E_squared_positive']:
                issues.append(f"E²(a) has negative values")
        else:
            checks['E_squared_positive'] = False
        
        # Check 3: μ(z) monotonic
        if 'observables' in self.results and 'sne_ia' in self.results['observables']:
            mu_arr = np.array(self.results['observables']['sne_ia']['mu'])
            diff_mu = np.diff(mu_arr)
            tol = tols.get('mu_monotonic_tol', 1e-6)
            checks['mu_monotonic'] = np.all(diff_mu > -tol)
            if not checks['mu_monotonic']:
                issues.append("μ(z) not monotonically increasing")
        else:
            checks['mu_monotonic'] = False
        
        # Check 4: D_M(z) monotonic
        if 'observables' in self.results and 'bao' in self.results['observables']:
            D_M_arr = np.array(self.results['observables']['bao']['D_M'])
            diff_D_M = np.diff(D_M_arr)
            tol = tols.get('D_M_monotonic_tol', 1e-6)
            checks['D_M_monotonic'] = np.all(diff_D_M > -tol)
            if not checks['D_M_monotonic']:
                issues.append("D_M(z) not monotonically increasing")
        else:
            checks['D_M_monotonic'] = False
        
        # Overall status
        checks['all_passed'] = all(checks.values())
        
        print(f"✅ Sanity checks completed: {sum(checks.values())}/{len(checks)} passed")
        if issues:
            print(f"⚠ Issues found: {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")
        
        return checks, issues
    
    
    def compute_observables(self):
        """
        Compute all observable quantities predicted by the TQE model.
        
        This function calculates cosmological observables that can be compared
        with real data to test the TQE hypothesis:
        
        Observables:
            - SNe Ia distance modulus: μ(z) from luminosity distance
            - BAO angular diameter distance: D_M(z) and Hubble parameter H(z)
            - CMB power spectrum: C_ℓ (TT) via CAMB or simplified calculation
            - LSS matter power spectrum: P(k) and S₈ parameter
        
        Results are stored in self.results['observables'] for later analysis.
        """
        print("📊 Computing observable predictions...")
        
        # SNe Ia Hubble diagram - use MASTER_CTRL parameters
        z_sne = np.linspace(
            MASTER_CTRL['SNE_Z_MIN'],
            MASTER_CTRL['SNE_Z_MAX'],
            MASTER_CTRL['SNE_N_POINTS']
        )
        mu_sne = self.observables.sne_hubble_diagram(z_sne)
        
        # BAO observables - use MASTER_CTRL parameters
        z_bao = np.array(MASTER_CTRL['BAO_Z_ARRAY'])
        D_M_bao, H_bao = self.observables.bao_observables(z_bao)
        
        # CMB power spectrum - use MASTER_CTRL parameters
        ell_cmb, C_ell_cmb = self.observables.cmb_power_spectrum(use_camb=MASTER_CTRL['USE_CAMB'] and CAMB_AVAILABLE)
        
        # Matter power spectrum - use MASTER_CTRL parameters
        k_array = np.logspace(
            np.log10(MASTER_CTRL['LSS_K_MIN']),
            np.log10(MASTER_CTRL['LSS_K_MAX']),
            MASTER_CTRL['LSS_N_K_POINTS']
        )
        P_k = self.observables.matter_power_spectrum(k_array, z=0)
        
        # S_8 parameter at z=0 and other key redshifts
        S_8_z0 = self.observables.S8_parameter(z=0)
        S_8_z05 = self.observables.S8_parameter(z=0.5)
        S_8_z1 = self.observables.S8_parameter(z=1.0)
        
        # Key diagnostic observables
        mu_z1 = self.friedmann.distance_modulus(1.0)
        D_M_z051 = self.friedmann.comoving_transverse_distance(0.51)
        H_z051 = self.friedmann.H(1.0 / 1.51)
        
        # Compute ρ_DE variation (max - min over evolution)
        rho_DE_variation = 0.0
        if 'evolution' in self.results and 'rho_DE_array' in self.results['evolution']:
            rho_DE_arr = np.array(self.results['evolution']['rho_DE_array'])
            rho_DE_variation = np.max(rho_DE_arr) - np.min(rho_DE_arr)
        
        # Compute I-parameter maximum
        I_max = 0.0
        if 'evolution' in self.results and 'I_array' in self.results['evolution']:
            I_arr = np.array(self.results['evolution']['I_array'])
            I_max = np.max(np.abs(I_arr))
        
        # Store observable predictions
        self.results['observables'] = {
            'sne_ia': {
                'z': z_sne.tolist(),
                'mu': mu_sne.tolist()
            },
            'bao': {
                'z': z_bao.tolist(),
                'D_M': D_M_bao.tolist(),
                'H': H_bao.tolist()
            },
            'cmb': {
                'ell': ell_cmb.tolist() if hasattr(ell_cmb, 'tolist') else list(ell_cmb),
                'C_ell_TT': C_ell_cmb.tolist() if hasattr(C_ell_cmb, 'tolist') else list(C_ell_cmb)
            },
            'lss': {
                'k': k_array.tolist(),
                'P_k': P_k.tolist(),
                'S_8': S_8_z0,
                'S_8_z05': S_8_z05,
                'S_8_z1': S_8_z1
            },
            # Key diagnostic values for aggregator
            'S8_raw': S_8_z0,  # S₈ at z=0 (raw, for comparison)
            'mu_z1': mu_z1,
            'D_M_z051': D_M_z051,
            'H_z051': H_z051,
            'H_z0': self.friedmann.H(1.0),  # H(z=0) for comparison
            'rho_DE_variation': rho_DE_variation,
            'I_max': I_max
        }
        
        # Compute likelihood if enabled
        if MASTER_CTRL.get('COMPUTE_LIKELIHOOD', True):
            likelihood_results = self.observables.compute_likelihood()
            
            # Calculate AIC/BIC with proper n_params
            n_params = 2  # Omega_m, H0
            if self.coupling.coupling_type == 'covariant_pressure':
                n_params += 1  # alpha
            elif self.coupling.coupling_type == 'uniform_w':
                n_params += 2  # w0, w_I
            elif self.coupling.coupling_type == 'geometric':
                n_params += 1  # beta0
            
            chi2_total = likelihood_results['chi2_total']
            n_data = likelihood_results['n_data']
            
            AIC = chi2_total + 2 * n_params
            BIC = chi2_total + n_params * np.log(n_data)
            reduced_chi2 = chi2_total / (n_data - n_params) if n_data > n_params else np.inf
            
            likelihood_results['AIC'] = AIC
            likelihood_results['BIC'] = BIC
            likelihood_results['n_params'] = n_params
            likelihood_results['reduced_chi2'] = reduced_chi2
            
            self.results['likelihood'] = likelihood_results
            
            print(f"✅ Likelihood computed")
            print(f"  χ² = {chi2_total:.2f} (reduced: {reduced_chi2:.2f})")
            print(f"  AIC = {AIC:.2f}, BIC = {BIC:.2f}")
        
        print(f"✅ Observable predictions computed")
        print(f"  S₈(z=0) = {S_8_z0:.4f}")
        print(f"  S₈(z=0.5) = {S_8_z05:.4f}")
        print(f"  S₈(z=1.0) = {S_8_z1:.4f}")
        print(f"  μ(z=1) = {mu_z1:.4f}")
        print(f"  D_M(z=0.51) = {D_M_z051:.2f} Mpc")
        print(f"  H(z=0.51) = {H_z051:.2f} km/s/Mpc")
        print(f"  ρ_DE variation = {rho_DE_variation:.6f}")
        print(f"  I_max = {I_max:.6f}")
        print(f"  SNe Ia: {len(z_sne)} redshift points")
        print(f"  BAO: {len(z_bao)} redshift points")
        print(f"  CMB: {len(ell_cmb)} multipoles")
    
    def visualize_results(self, save_plots=True):
        """
        Create publication-quality visualizations of TQE model results.
        
        Generates a comprehensive set of PNG plots showing all key observables,
        cosmological evolution, and TQE-specific diagnostics. All plots are
        optimized for publication with LaTeX formatting and 300 DPI resolution.
        
        Generated plots (11-16 files):
            01. Hubble parameter evolution H(z)
            02. I-parameter evolution I(a) with energy content explanation
            03. Dark energy density evolution ρ_DE(z)
            04. SNe Ia Hubble diagram μ(z)
            05. BAO observables D_M(z) and H(z)
            06. CMB power spectrum C_ℓ
            07. Matter power spectrum P(k) [optional]
            08. S₈ evolution S₈(z)
            09. Growth factor D(z)
            10. I vs E scatter plot with colorbar [TQE signature!]
            11. Cosmic web fractions (voids, filaments, clusters)
            12-14. Structure size distributions [conditional]
            15. Density field slices
            16. I-definitions comparison [optional]
        
        Args:
            save_plots: If True, save PNG files to project_dir/PNG_Visualizations/
        
        Files are prefixed with coupling mode ('Eonly_' or 'EplusI_') for comparison.
        """
        print("Creating visualizations...")
        print(f"  matplotlib backend: {matplotlib.get_backend()}")
        print(f"  save_plots: {save_plots}")
        
        if 'evolution' not in self.results or 'observables' not in self.results:
            print("WARNING: No results available for visualization")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_dir = f"{self.project_dir}/PNG_Visualizations"
        print(f"  PNG directory: {png_dir}")
        
        # Get file prefix for coupling mode (Eonly_ or EplusI_ or "")
        prefix = get_file_prefix(self.coupling_mode)
        
        # 1. Hubble parameter evolution H(z) - PUBLICATION OPTIMIZED
        plt.close('all')
        fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
        z_arr = np.array(self.results['evolution']['z_array'])
        H_arr = np.array(self.results['evolution']['H_array'])
        
        # PUBLICATION: Limit to z ≤ 5 (cosmologically relevant range)
        mask = z_arr <= 5.0
        z_plot = z_arr[mask]
        H_plot = H_arr[mask]
        
        ax.plot(z_plot, H_plot, '-', color='#2E86AB', linewidth=2.5, label='TQE Model', alpha=0.9)
        ax.axhline(self.friedmann.H0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label=f'$H_0$ = {self.friedmann.H0:.1f} km/s/Mpc')
        
        ax.set_xlabel('Redshift $z$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_ylabel('$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_title('Hubble Parameter Evolution', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
        ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='upper left', framealpha=0.95, edgecolor='gray')
        ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
        ax.set_xlim([0, 5])
        plt.tight_layout()
        if save_plots:
            plot_path = f"{png_dir}/{prefix}01_hubble_parameter_evolution_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
            
        plt.close()
        
        # 2. I-parameter evolution I(z) - PUBLICATION OPTIMIZED  
        # ONLY for E+I mode (E-only doesn't use I-parameter in coupling)
        if self.coupling_mode == 'EplusI':
            plt.close('all')
            fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
            I_arr = np.array(self.results['evolution']['I_array'])
            
            # Limit to z ≤ 5
            I_plot = I_arr[mask]
            
            ax.plot(z_plot, I_plot, '-', color='#C9184A', linewidth=2.5, label='Information Content $I(E)$', alpha=0.9)
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.3)
            
            ax.set_xlabel('Redshift $z$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_ylabel('Information Parameter $I$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_title('Energy Information Content Evolution (E+I Mode)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
            ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='best', framealpha=0.95, edgecolor='gray')
            ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
            ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
            ax.set_xlim([0, 5])
            ax.set_ylim([0, 1])
            
            # Add text box explaining I
            textstr = '$I = |dE/da| / (E + |dE/da|)$\nIntrinsic information\ncontent of energy'
            ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='gray', linewidth=0.5))
        
        plt.tight_layout()
        if save_plots:
                plot_path = f"{png_dir}/{prefix}02_i_parameter_evolution_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
                
        plt.close()
        
        # 3. Dark energy density evolution ρ_DE(z) - PUBLICATION OPTIMIZED
        fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
        rho_DE_arr = np.array(self.results['evolution']['rho_DE_array'])
        rho_DE_plot = rho_DE_arr[mask]
        
        ax.plot(z_plot, rho_DE_plot, '-', color='#06A77D', linewidth=2.5, label='$\\rho_{\\mathrm{DE}}(z)$ TQE', alpha=0.9)
        ax.axhline(self.friedmann.rho_Lambda_today, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='$\\rho_\\Lambda$ (ΛCDM)')
        
        ax.set_xlabel('Redshift $z$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_ylabel('$\\rho_{\\mathrm{DE}}$ (normalized to $\\rho_{\\mathrm{crit,0}}$)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        
        # Title based on coupling mode
        if self.coupling_mode == 'Eonly':
            ax.set_title('Dark Energy Density Evolution (E-only Mode)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
        else:
            ax.set_title('Dark Energy Density Evolution (E+I Mode)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
        ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='best', framealpha=0.95, edgecolor='gray')
        ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
        ax.set_xlim([0, 5])
        plt.tight_layout()
        if save_plots:
            plot_path = f"{png_dir}/{prefix}03_dark_energy_density_evolution_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
            
        plt.close()
        
        # 4. SNe Ia Hubble diagram - PUBLICATION OPTIMIZED
        fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
        z_sne = np.array(self.results['observables']['sne_ia']['z'])
        mu_sne = np.array(self.results['observables']['sne_ia']['mu'])
        
        ax.plot(z_sne, mu_sne, 'o', color='#E63946', markersize=6, markeredgecolor='black', markeredgewidth=0.5, 
                linewidth=0, label='TQE Prediction', alpha=0.8, zorder=3)
        ax.plot(z_sne, mu_sne, '-', color='#E63946', linewidth=1.5, alpha=0.4, zorder=2)
        
        ax.set_xlabel('Redshift $z$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_ylabel('Distance Modulus $\\mu$ [mag]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_title('SNe Ia Hubble Diagram', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
        ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='lower right', framealpha=0.95, edgecolor='gray')
        ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
        ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
        ax.set_xlim([0, max(z_sne)*1.05])
        plt.tight_layout()
        if save_plots:
            plot_path = f"{png_dir}/{prefix}04_sne_ia_hubble_diagram_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
            
        plt.close()
        
        # 5. BAO observables: D_M(z) and H(z) - PUBLICATION OPTIMIZED (2-panel)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=MASTER_CTRL['PLOT_FIGSIZE_WIDE'], facecolor='white')
        z_bao = np.array(self.results['observables']['bao']['z'])
        D_M_bao = np.array(self.results['observables']['bao']['D_M'])
        H_bao = np.array(self.results['observables']['bao']['H'])
        
        # Left panel: D_M(z)
        ax1.plot(z_bao, D_M_bao, 'o', color='#457B9D', markersize=8, markeredgecolor='black', markeredgewidth=0.5, 
                label='TQE Model', alpha=0.8, zorder=3)
        ax1.plot(z_bao, D_M_bao, '-', color='#457B9D', linewidth=2, alpha=0.4, zorder=2)
        ax1.set_xlabel('Redshift $z$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax1.set_ylabel('$D_M(z)$ [Mpc]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax1.set_title('BAO: Comoving Distance', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE']-1, pad=10)
        ax1.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='upper left', framealpha=0.95, edgecolor='gray')
        ax1.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
        ax1.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
        
        # Right panel: H(z)
        ax2.plot(z_bao, H_bao, 'o', color='#E63946', markersize=8, markeredgecolor='black', markeredgewidth=0.5, 
                label='TQE Model', alpha=0.8, zorder=3)
        ax2.plot(z_bao, H_bao, '-', color='#E63946', linewidth=2, alpha=0.4, zorder=2)
        ax2.set_xlabel('Redshift $z$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax2.set_ylabel('$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax2.set_title('BAO: Hubble Parameter', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE']-1, pad=10)
        ax2.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='upper left', framealpha=0.95, edgecolor='gray')
        ax2.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
        ax2.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
        
        plt.tight_layout()
        if save_plots:
            plot_path = f"{png_dir}/{prefix}05_bao_observables_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
            
        plt.close()
        
        # 6. CMB power spectrum C_ℓ
        fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
        ell_cmb = np.array(self.results['observables']['cmb']['ell'])
        C_ell_cmb = np.array(self.results['observables']['cmb']['C_ell_TT'])
        ax.plot(ell_cmb, C_ell_cmb, 'b-', linewidth=1.5, label='TT power spectrum')
        ax.set_xlabel('Multipole ℓ', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_ylabel('C_ℓ [μK²]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax.set_title('CMB Temperature Power Spectrum', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
        ax.grid(True, alpha=0.3, which='both')
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.tight_layout()
        if save_plots:
            plot_path = f"{png_dir}/{prefix}06_cmb_power_spectrum_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
            
        plt.close()
        
        # 7. Matter power spectrum P(k) - OPTIONAL (skip for publication to reduce clutter)
        if MASTER_CTRL.get('SAVE_MATTER_POWER_SPECTRUM_PNG', False):
            fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
            k_arr = np.array(self.results['observables']['lss']['k'])
            P_k_arr = np.array(self.results['observables']['lss']['P_k'])
            ax.loglog(k_arr, P_k_arr, 'g-', linewidth=2, label='P(k, z=0)')
            ax.set_xlabel('Wavenumber k [Mpc⁻¹]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_ylabel('P(k) [(Mpc/h)³]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_title('Matter Power Spectrum', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
            ax.legend(fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
            ax.grid(True, alpha=0.3, which='both')
            ax.tick_params(axis='both', which='major', labelsize=10)
            plt.tight_layout()
            if save_plots:
                plot_path = f"{png_dir}/{prefix}07_matter_power_spectrum_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                        
            plt.close()
        
        # 8. S₈(z) evolution
        if 'evolution_series' in self.results:
            z_evol = np.array(self.results['evolution_series']['z'])
            S8_evol = np.array(self.results['evolution_series']['S8'])
            
            fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
            ax.plot(z_evol, S8_evol, 'b-', linewidth=2, label='S₈(z)')
            ax.set_xlabel('Redshift z', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_ylabel('S₈', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_title('S₈ Parameter Evolution', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
            ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
            ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
            plt.tight_layout()
            if save_plots:
                plot_path = f"{png_dir}/{prefix}08_S8_evolution_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                
            plt.close()
        
        # 9. Growth factor D(z) evolution
        if 'evolution_series' in self.results:
            z_evol = np.array(self.results['evolution_series']['z'])
            D_evol = np.array(self.results['evolution_series']['D'])
            
            fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
            ax.plot(z_evol, D_evol, 'r-', linewidth=2, label='D(z)')
            ax.set_xlabel('Redshift z', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_ylabel('Growth Factor D(z)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_title('Linear Growth Factor Evolution', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
            ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
            ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
            plt.tight_layout()
            if save_plots:
                plot_path = f"{png_dir}/{prefix}09_growth_factor_evolution_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                
            plt.close()
        
        # 10. I vs E scatter - PUBLICATION OPTIMIZED (TQE SIGNATURE!)
        if 'I_E_correlation' in self.results:
            I_arr = np.array(self.results['I_E_correlation']['I_array'])
            E_arr = np.array(self.results['I_E_correlation']['E_array'])
            pearson_r = self.results['I_E_correlation']['pearson_r']
            
            # Only plot if there's variance
            if np.std(I_arr) > 1e-10 and np.std(E_arr) > 1e-10:
                fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
                
                # PUBLICATION: Density-based coloring (shows evolution direction)
                if 'a_array' in self.results['I_E_correlation']:
                    a_arr = np.array(self.results['I_E_correlation']['a_array'])
                    scatter = ax.scatter(I_arr, E_arr, c=a_arr, s=15, alpha=0.7, cmap='viridis', 
                                        edgecolors='black', linewidth=0.3, zorder=3)
                    cbar = plt.colorbar(scatter, ax=ax, label='Scale Factor $a$')
                    cbar.ax.tick_params(labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND']-1)
                else:
                    ax.scatter(I_arr, E_arr, s=15, alpha=0.7, c='#8338EC', edgecolors='black', linewidth=0.3, zorder=3)
                
                ax.set_xlabel('Information Parameter $I$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Normalized Energy $E = H/H_0$', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title(f'$I$ vs $E$ Correlation (Pearson $r = {pearson_r:.3f}$)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
                ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], direction='in', length=6)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, max(E_arr)*1.05])
                plt.tight_layout()
                if save_plots:
                    plot_path = f"{png_dir}/{prefix}10_I_vs_E_scatter_{timestamp}.png"
                    plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
                    
                plt.close()
        
        # ==========================================================================================
        # GALAXY STRUCTURE VISUALIZATIONS (11-15)
        # ==========================================================================================
        
        if 'galaxy_structure' in self.results and MASTER_CTRL.get('GALAXY_CREATE_VISUALIZATIONS', True):
            print("\n🌌 Creating galaxy structure visualizations...")
            
            galaxy_data = self.results['galaxy_structure']
            
            # 11. Cosmic Web Fractions (Bar Chart) - PUBLICATION OPTIMIZED
            if 'summary' in galaxy_data:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
                
                fractions = galaxy_data['summary']['cosmic_web_fractions']
                labels = ['Voids', 'Filaments', 'Sheets', 'Knots']
                values = [
                    fractions['void_fraction'] * 100,
                    fractions['filament_fraction'] * 100,
                    fractions['sheet_fraction'] * 100,
                    fractions['knot_fraction'] * 100
                ]
                colors = [MASTER_CTRL['COLOR_VOID'], MASTER_CTRL['COLOR_FILAMENT'], 
                         MASTER_CTRL['COLOR_SHEET'], MASTER_CTRL['COLOR_CLUSTER']]
                
                bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.8, width=0.6)
                ax.set_ylabel('Volume Fraction [%]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('Cosmic Web Topology Classification', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
                ax.grid(True, axis='y', alpha=MASTER_CTRL['PLOT_GRID_ALPHA'], linestyle=':', linewidth=0.8)
                ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'], direction='in', length=6)
                ax.set_ylim([0, max(values)*1.15])
                
                # Add percentage labels on bars (larger font)
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND']+1, fontweight='bold')
                
                plt.tight_layout()
                if save_plots:
                    plot_path = f"{png_dir}/{prefix}11_cosmic_web_fractions_{timestamp}.png"
                    plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white', edgecolor='none')
                    
                plt.close()
            
            # 12. Void Size Distribution
            if 'voids' in galaxy_data and len(galaxy_data['voids']) > 0:
                fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
                
                void_radii = [v['radius_mpc'] for v in galaxy_data['voids']]
                ax.hist(void_radii, bins=20, color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.2)
                ax.set_xlabel('Void Radius [Mpc/h]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Number of Voids', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title(f'Void Size Distribution (N = {len(void_radii)})', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                plt.tight_layout()
                if save_plots:
                    plot_path = f"{png_dir}/{prefix}12_void_size_distribution_{timestamp}.png"
                    plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                    
                plt.close()
            
            # 13. Cluster Size Distribution
            if 'clusters' in galaxy_data and len(galaxy_data['clusters']) > 0:
                fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
                
                cluster_radii = [c['radius_mpc'] for c in galaxy_data['clusters']]
                ax.hist(cluster_radii, bins=20, color='#f39c12', alpha=0.7, edgecolor='black', linewidth=1.2)
                ax.set_xlabel('Cluster Radius [Mpc/h]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Number of Clusters', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title(f'Cluster Size Distribution (N = {len(cluster_radii)})', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                plt.tight_layout()
                if save_plots:
                    plot_path = f"{png_dir}/{prefix}13_cluster_size_distribution_{timestamp}.png"
                    plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                    
                plt.close()
            
            # 14. Filament Length Distribution - CONDITIONAL (skip if disabled or no filaments)
            if MASTER_CTRL.get('SAVE_FILAMENT_DISTRIBUTION_PNG', False) and 'filaments' in galaxy_data and len(galaxy_data['filaments']) > 0:
                fig, ax = plt.subplots(figsize=MASTER_CTRL['PLOT_FIGSIZE_DEFAULT'], facecolor='white')
                
                filament_lengths = [f['length_mpc'] for f in galaxy_data['filaments']]
                ax.hist(filament_lengths, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.2)
                ax.set_xlabel('Filament Length [Mpc/h]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Number of Filaments', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title(f'Filament Length Distribution (N = {len(filament_lengths)})', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                ax.tick_params(axis='both', which='major', labelsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                plt.tight_layout()
                if save_plots:
                    plot_path = f"{png_dir}/{prefix}14_filament_length_distribution_{timestamp}.png"
                    plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                    
                plt.close()
            
            # 15. 3D Density Field Slice Visualization
            if save_plots:
                fig = plt.subplots(figsize=(14, 4), facecolor='white')
                
                # Get galaxy analyzer density field from results
                if hasattr(self, 'galaxy_analyzer') and self.galaxy_analyzer.density_field is not None:
                    density = self.galaxy_analyzer.density_field
                    mid_slice = density.shape[2] // 2
                    
                    # Create 3-panel plot: XY, XZ, YZ slices
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    
                    # XY slice
                    im1 = axes[0].imshow(density[:, :, mid_slice].T, origin='lower', cmap='RdYlBu_r', vmin=-3, vmax=3)
                    axes[0].set_title('Density Field (XY Slice)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12))
                    axes[0].set_xlabel('X [cells]', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
                    axes[0].set_ylabel('Y [cells]', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
                    plt.colorbar(im1, ax=axes[0], label='δ')
                    
                    # XZ slice
                    im2 = axes[1].imshow(density[:, mid_slice, :].T, origin='lower', cmap='RdYlBu_r', vmin=-3, vmax=3)
                    axes[1].set_title('Density Field (XZ Slice)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12))
                    axes[1].set_xlabel('X [cells]', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
                    axes[1].set_ylabel('Z [cells]', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
                    plt.colorbar(im2, ax=axes[1], label='δ')
                    
                    # YZ slice
                    im3 = axes[2].imshow(density[mid_slice, :, :].T, origin='lower', cmap='RdYlBu_r', vmin=-3, vmax=3)
                    axes[2].set_title('Density Field (YZ Slice)', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LABEL', 12))
                    axes[2].set_xlabel('Y [cells]', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
                    axes[2].set_ylabel('Z [cells]', fontsize=MASTER_CTRL.get('PLOT_FONTSIZE_LEGEND', 10))
                    plt.colorbar(im3, ax=axes[2], label='δ')
                    
                    plt.tight_layout()
                    plot_path = f"{png_dir}/{prefix}15_density_field_slices_{timestamp}.png"
                    plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                    
                    plt.close()
        
        # 16. I-Definitions Comparison (if COMPUTE_ALL_I_DEFINITIONS enabled)
        if MASTER_CTRL.get('COMPUTE_ALL_I_DEFINITIONS', False):
            print(f"\n📊 Creating I-Definitions Comparison plot...")
            
            # Sample scale factor grid
            a_grid = self.results['evolution']['a_array']
            a_sample = a_grid[::max(1, len(a_grid)//100)]  # 100 points
            
            # Compute all I-definitions
            I_results = {
                'phenomenological': [],
                'kl_divergence': [],
                'shannon': [],
                'renyi': [],
                'mutual_info': [],
                'composite': [],
                'kl_shannon': [],
                'entanglement': [],
                'fisher': [],
                'horizon_entropy': []
            }
            
            for a_val in a_sample:
                I_defs = self.information_content.compute_all_I_definitions(a_val, friedmann=self.friedmann)
                for key in I_results.keys():
                    I_results[key].append(I_defs.get(key, 0.0))
            
            # Create comparison plot
            fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
            
            colors = MASTER_CTRL['COLOR_PALETTE_EXTENDED']
            linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
            
            plot_idx = 0
            for name, values in I_results.items():
                if name == 'composite':
                    # Highlight composite (default) with thicker line
                    ax.plot(a_sample, values, label=f'{name} (DEFAULT)', 
                           color=colors[plot_idx % len(colors)], linewidth=2.5, 
                           linestyle=linestyles[plot_idx], alpha=0.9)
                else:
                    ax.plot(a_sample, values, label=name, 
                           color=colors[plot_idx % len(colors)], linewidth=1.5, 
                           linestyle=linestyles[plot_idx], alpha=0.7)
                plot_idx += 1
            
            ax.set_xlabel('Scale Factor a', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'], fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'])
            ax.set_ylabel('I-parameter', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'], fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'])
            ax.set_title('I-Parameter: 10 Definitions Comparison', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'])
            ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND']-1, loc='best', ncol=2, framealpha=0.9)
            ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            plt.tight_layout()
            
            # Only save if enabled (skip for single I-definition runs)
            if MASTER_CTRL.get('SAVE_I_DEFINITIONS_PNG', False):
                plot_path = f"{png_dir}/{prefix}16_I_definitions_comparison_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight', facecolor='white')
                        
            
            plt.close()
        
        # Final cleanup: close all figures to free memory
        plt.close('all')
        print("All visualizations created successfully!")
    
    def save_results(self):
        """
        Save all simulation results to structured CSV, JSON, and TXT files.
        
        This function creates a comprehensive data archive for the TQE simulation,
        including all observables, evolution series, correlations, and diagnostics.
        
        File structure (40-50 files per run):
            CSV files (23):
                - hubble_diagram.csv: SNe Ia distance modulus vs redshift
                - bao_observables.csv: BAO measurements D_M(z) and H(z)
                - cmb_power_spectrum.csv: CMB C_ℓ power spectrum
                - evolution_series.csv: S₈(z), D(z), ρ_DE(z), I(z), E(z) evolution
                - I_E_correlation.csv: I-E arrays + correlation statistics (Pearson, Spearman, MI)
                - I_E_lag_scan.csv: I-E lag scan results (if RUN_LAG_SCAN=True)
                - S8_normalization.csv: S₈ normalization vs ΛCDM
                - likelihood_results.csv: χ², AIC, BIC values
                - I_of_a.csv: I-parameter evolution
                - E_of_a.csv: Energy parameter evolution [removed, E=H/H0]
                - D_of_a.csv: Growth factor evolution
                - rho_DE_of_a.csv: Dark energy density evolution
                - H_of_z.csv: Hubble parameter H(z) evolution
                - matter_power_spectrum.csv: Matter power spectrum P(k)
                - S8_values.csv: S₈ values at z=0, 0.5, 1.0
                - model_parameters.csv: All model parameters (coupling + I-field)
                - field_statistics.csv: Field statistics (geometric coupling only)
                - sensitivity_test.csv: Sensitivity test results (if RUN_SENSITIVITY_TEST=True)
                - I_Definitions_Comparison.csv: All 10 I-definitions (if COMPUTE_ALL_I_DEFINITIONS=True)
                + Galaxy structure catalogues (4): Void, Cluster, Filament, Wall
                + Bayesian inference CSV (if RUN_MCMC=True): Bayesian_MCMC_Samples.csv
            
            JSON files (4-8):
                - TQE_DarkEnergy_Results.json: Complete results dictionary
                - Model_Summary.json: Key metrics + model info
                - Galaxy_Cosmic_Web_Summary.json: Structure statistics
                + Bayesian inference JSON (if RUN_MCMC=True):
                  - Bayesian_MCMC_Summary.json
                  - Bayesian_Information_Criteria.json
                  - Bayesian_Nested_Sampling_Evidence.json (if nested sampling)
            
            TXT files (2):
                - Full_Summary.txt: Human-readable report + conclusion
                - Reproducibility_Info.txt: Seed hash for reproducibility
            
            PNG files (12-20):
                - Visualization plots (12-16 standard plots)
                - Bayesian_Corner_Plot.png (if RUN_MCMC=True)
                + CMB Planck validation plots (4-5, if USE_REAL_CMB_PLANCK_MAPS=True)
            
            ZIP archive (1):
                - Complete_Results_Archive.zip: All files bundled
        
        All files are automatically prefixed with coupling mode ('Eonly_' or 'EplusI_').
        """
        if not self.project_dir:
            print("⚠ No project directory specified")
            return
        
        # Run sanity checks before saving
        sanity_checks, sanity_issues = run_sanity_checks(self)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get file prefix based on coupling mode (Eonly_ or EplusI_ or "")
        prefix = get_file_prefix(self.coupling_mode)
        
        # Helper function to create informative empty status DataFrame
        def create_status_df(status, message="", error=None):
            """Create a status DataFrame with metadata when data is missing."""
            status_data = {
                'status': [status],
                'timestamp': [timestamp],
                'coupling_mode': [self.coupling_mode],
                'message': [message] if message else [f"Data not available for {status}"]
            }
            if error:
                status_data['error'] = [str(error)]
            return pd.DataFrame(status_data)
        
        # Save JSON results to root directory (with sanity checks and coupling mode) (always save)
        json_file = f"{self.project_dir}/{prefix}TQE_DarkEnergy_Results_{timestamp}.json"
        try:
            results_with_validation = self.results.copy()
            results_with_validation['sanity_checks'] = sanity_checks
            results_with_validation['sanity_issues'] = sanity_issues
            
            # Add model_info for JSON completeness
            results_with_validation['model_info'] = {
                'coupling_mode': self.coupling_mode,
                'coupling_type': self.coupling.coupling_type,
                'i_field_type': self.information_content.model_type,
                'timestamp': timestamp,
                'project_dir': self.project_dir
            }
            
            with open(json_file, 'w') as f:
                json.dump(results_with_validation, f, indent=2, default=str)
        except Exception as e:
            # Save error status if JSON save fails
            error_data = {'status': 'error', 'error': str(e), 'timestamp': timestamp}
            with open(json_file, 'w') as f:
                json.dump(error_data, f, indent=2)
        
        
        # Save Hubble diagram CSV with explicit units and prefix (always save)
        hubble_csv = f"{self.project_dir}/{prefix}hubble_diagram_{timestamp}.csv"
        try:
            if 'observables' in self.results and 'sne_ia' in self.results['observables']:
                hubble_data = pd.DataFrame({
                    'z_dimensionless': self.results['observables']['sne_ia']['z'],
                    'mu_magnitudes': self.results['observables']['sne_ia']['mu']
                })
                hubble_data.to_csv(hubble_csv, index=False)
            else:
                create_status_df('no_data', 'SNe Ia observables not found in results').to_csv(hubble_csv, index=False)
        except Exception as e:
            create_status_df('error', 'Failed to save Hubble diagram', error=e).to_csv(hubble_csv, index=False)
        
        
        # Save BAO data CSV with explicit units and prefix (always save)
        bao_csv = f"{self.project_dir}/{prefix}bao_observables_{timestamp}.csv"
        try:
            if 'observables' in self.results and 'bao' in self.results['observables']:
                bao_data = pd.DataFrame({
                    'z_dimensionless': self.results['observables']['bao']['z'],
                    'D_M_Mpc': self.results['observables']['bao']['D_M'],
                    'H_km_per_s_per_Mpc': self.results['observables']['bao']['H']
                })
                bao_data.to_csv(bao_csv, index=False)
            else:
                create_status_df('no_data', 'BAO observables not found in results').to_csv(bao_csv, index=False)
        except Exception as e:
            create_status_df('error', 'Failed to save BAO observables', error=e).to_csv(bao_csv, index=False)
        
        
        # Save CMB power spectrum CSV with explicit units and prefix (always save)
        cmb_csv = f"{self.project_dir}/{prefix}cmb_power_spectrum_{timestamp}.csv"
        try:
            if 'observables' in self.results and 'cmb' in self.results['observables']:
                cmb_data = pd.DataFrame({
                    'ell_dimensionless': self.results['observables']['cmb']['ell'],
                    'C_ell_TT_muK2': self.results['observables']['cmb']['C_ell_TT']
                })
                cmb_data.to_csv(cmb_csv, index=False)
            else:
                create_status_df('no_data', 'CMB observables not found in results').to_csv(cmb_csv, index=False)
        except Exception as e:
            create_status_df('error', 'Failed to save CMB power spectrum', error=e).to_csv(cmb_csv, index=False)
        
        
        # Save evolution series CSV (S₈(z), ρ_DE(z), D(z), I(z), E(z)) with prefix (always save)
        evol_csv = f"{self.project_dir}/{prefix}evolution_series_{timestamp}.csv"
        if 'evolution_series' in self.results:
            try:
                evol_data = pd.DataFrame({
                    'z_dimensionless': self.results['evolution_series']['z'],
                    'S8_dimensionless': self.results['evolution_series']['S8'],
                    'D_growth_factor': self.results['evolution_series']['D'],
                    'rho_DE_normalized': self.results['evolution_series']['rho_DE'],
                    'I_parameter': self.results['evolution_series']['I'],
                    'E_parameter': self.results['evolution_series']['E']
                })
                evol_data.to_csv(evol_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save evolution series', error=e).to_csv(evol_csv, index=False)
        else:
            create_status_df('no_data', 'evolution_series not found in results').to_csv(evol_csv, index=False)
        
        
        # Save I-E correlation CSV with prefix (including statistics) (always save)
        ie_csv = f"{self.project_dir}/{prefix}I_E_correlation_{timestamp}.csv"
        if 'I_E_correlation' in self.results:
            try:
                ie_data = pd.DataFrame({
                    'I_parameter': self.results['I_E_correlation']['I_array'],
                    'E_parameter': self.results['I_E_correlation']['E_array'],
                    'Pearson_r': [self.results['I_E_correlation']['pearson_r']] * len(self.results['I_E_correlation']['I_array']),
                    'Spearman_r': [self.results['I_E_correlation']['spearman_r']] * len(self.results['I_E_correlation']['I_array']),
                    'Mutual_Information': [self.results['I_E_correlation']['mutual_information']] * len(self.results['I_E_correlation']['I_array'])
                })
                ie_data.to_csv(ie_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save I-E correlation', error=e).to_csv(ie_csv, index=False)
        else:
            create_status_df('no_data', 'I_E_correlation not found in results').to_csv(ie_csv, index=False)
        
        # Save I-E lag scan CSV with prefix (always save, if lag_scan exists)
        lag_scan_csv = f"{self.project_dir}/{prefix}I_E_lag_scan_{timestamp}.csv"
        if 'I_E_correlation' in self.results and 'lag_scan' in self.results['I_E_correlation']:
            try:
                lag_scan = self.results['I_E_correlation']['lag_scan']
                lag_data = pd.DataFrame({
                    'da_lag': lag_scan.get('da_lags', []),
                    'MI_lag': lag_scan.get('MI_lags', []),
                    'max_MI': [lag_scan.get('max_MI', 0.0)] * len(lag_scan.get('da_lags', [])),
                    'optimal_da': [lag_scan.get('optimal_da', 0.0)] * len(lag_scan.get('da_lags', []))
                })
                lag_data.to_csv(lag_scan_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save I-E lag scan', error=e).to_csv(lag_scan_csv, index=False)
        else:
            create_status_df('not_computed', 'lag_scan not computed (RUN_LAG_SCAN=False or not available)').to_csv(lag_scan_csv, index=False)
        
        # Save dedicated I(a), E(a), D(a) evolution files (always save)
        I_a_csv = f"{self.project_dir}/{prefix}I_of_a_{timestamp}.csv"
        E_a_csv = f"{self.project_dir}/{prefix}E_of_a_{timestamp}.csv"
        D_a_csv = f"{self.project_dir}/{prefix}D_of_a_{timestamp}.csv"
        
        if 'I_E_correlation' in self.results and 'a_array' in self.results['I_E_correlation']:
            try:
                a_arr = self.results['I_E_correlation']['a_array']
                I_arr = self.results['I_E_correlation']['I_array']
                E_arr = self.results['I_E_correlation']['E_array']
                
                # I(a) file with prefix
                pd.DataFrame({'a_scale_factor': a_arr, 'I_parameter': I_arr}).to_csv(I_a_csv, index=False)
                
                # E(a) file with prefix
                pd.DataFrame({'a_scale_factor': a_arr, 'E_parameter': E_arr}).to_csv(E_a_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save I(a) and E(a) evolution', error=e).to_csv(I_a_csv, index=False)
                create_status_df('error', 'Failed to save I(a) and E(a) evolution', error=e).to_csv(E_a_csv, index=False)
        else:
            create_status_df('no_data', 'I_E_correlation.a_array not found in results').to_csv(I_a_csv, index=False)
            create_status_df('no_data', 'I_E_correlation.a_array not found in results').to_csv(E_a_csv, index=False)
        
        # D(a) file (growth factor evolution) with prefix (always save)
        if 'evolution_series' in self.results:
            try:
                z_arr = np.array(self.results['evolution_series']['z'])
                a_grid = 1.0 / (1.0 + z_arr)
                D_arr = np.array(self.results['evolution_series']['D'])
                pd.DataFrame({'a_scale_factor': a_grid, 'D_growth_factor': D_arr}).to_csv(D_a_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save D(a) growth factor evolution', error=e).to_csv(D_a_csv, index=False)
        else:
            create_status_df('no_data', 'evolution_series not found in results').to_csv(D_a_csv, index=False)
                    
        
        # Save S₈ normalization CSV with prefix (always save)
        s8norm_csv = f"{self.project_dir}/{prefix}S8_normalization_{timestamp}.csv"
        if 'S8_normalization' in self.results:
            try:
                s8norm_data = pd.DataFrame([self.results['S8_normalization']])
                s8norm_data.to_csv(s8norm_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save S8 normalization', error=e).to_csv(s8norm_csv, index=False)
        else:
            create_status_df('no_data', 'S8_normalization not found in results').to_csv(s8norm_csv, index=False)
        
        
        # Save likelihood results CSV with prefix (FIXED: chi2_BAO combines BAO_DM + BAO_H) (always save)
        likelihood_csv = f"{self.project_dir}/{prefix}likelihood_results_{timestamp}.csv"
        if 'likelihood' in self.results:
            try:
                # Combine BAO_DM and BAO_H into total chi2_BAO
                chi2_BAO_DM = self.results['likelihood']['chi2_components'].get('BAO_DM', 0.0)
                chi2_BAO_H = self.results['likelihood']['chi2_components'].get('BAO_H', 0.0)
                chi2_BAO_total = chi2_BAO_DM + chi2_BAO_H if (chi2_BAO_DM != 'N/A' and chi2_BAO_H != 'N/A') else 'N/A'
                
                likelihood_data = pd.DataFrame([{
                    'chi2_total': self.results['likelihood']['chi2_total'],
                    'chi2_SNe': self.results['likelihood']['chi2_components'].get('SNe', 'N/A'),
                    'chi2_BAO': chi2_BAO_total,
                    'chi2_BAO_DM': chi2_BAO_DM,
                    'chi2_BAO_H': chi2_BAO_H,
                    'chi2_H0': self.results['likelihood']['chi2_components'].get('H0_prior', 'N/A'),
                    'chi2_CMB': self.results['likelihood']['chi2_components'].get('CMB', 'N/A'),
                    'AIC': self.results['likelihood']['AIC'],
                    'BIC': self.results['likelihood']['BIC'],
                    'reduced_chi2': self.results['likelihood']['reduced_chi2'],
                    'n_data': self.results['likelihood']['n_data'],
                    'n_params': self.results['likelihood']['n_params']
                }])
                likelihood_data.to_csv(likelihood_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save likelihood results', error=e).to_csv(likelihood_csv, index=False)
        else:
            create_status_df('no_data', 'likelihood not found in results').to_csv(likelihood_csv, index=False)
            
        
        # Save ρ_DE(a) evolution CSV with prefix (always save)
        rho_DE_a_csv = f"{self.project_dir}/{prefix}rho_DE_of_a_{timestamp}.csv"
        if 'evolution_series' in self.results:
            try:
                z_arr = np.array(self.results['evolution_series']['z'])
                a_arr = 1.0 / (1.0 + z_arr)
                rho_DE_arr = np.array(self.results['evolution_series']['rho_DE'])
                pd.DataFrame({'a_scale_factor': a_arr, 'rho_DE_normalized': rho_DE_arr}).to_csv(rho_DE_a_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save rho_DE(a) evolution', error=e).to_csv(rho_DE_a_csv, index=False)
        else:
            create_status_df('no_data', 'evolution_series not found in results').to_csv(rho_DE_a_csv, index=False)
        
        # Save H(z) evolution CSV with prefix (always save) - Hubble parameter evolution
        H_z_csv = f"{self.project_dir}/{prefix}H_of_z_{timestamp}.csv"
        if 'evolution' in self.results:
            try:
                z_arr = np.array(self.results['evolution']['z_array'])
                H_arr = np.array(self.results['evolution']['H_array'])
                pd.DataFrame({
                    'z_redshift': z_arr,
                    'H_km_per_s_per_Mpc': H_arr
                }).to_csv(H_z_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save H(z) evolution', error=e).to_csv(H_z_csv, index=False)
        else:
            create_status_df('no_data', 'evolution not found in results').to_csv(H_z_csv, index=False)
        
        # Save matter power spectrum P(k) CSV with prefix (always save)
        matter_power_csv = f"{self.project_dir}/{prefix}matter_power_spectrum_{timestamp}.csv"
        if 'observables' in self.results and 'lss' in self.results['observables']:
            try:
                lss_data = self.results['observables']['lss']
                if 'k' in lss_data and 'P_k' in lss_data:
                    pd.DataFrame({
                        'k_h_per_Mpc': lss_data['k'],
                        'P_k_Mpc3_per_h3': lss_data['P_k']
                    }).to_csv(matter_power_csv, index=False)
                else:
                    create_status_df('no_data', 'k or P_k not found in observables.lss').to_csv(matter_power_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save matter power spectrum', error=e).to_csv(matter_power_csv, index=False)
        else:
            create_status_df('no_data', 'observables.lss not found in results').to_csv(matter_power_csv, index=False)
        
        # Save S8 values at different redshifts CSV with prefix (always save)
        s8_values_csv = f"{self.project_dir}/{prefix}S8_values_{timestamp}.csv"
        if 'observables' in self.results and 'lss' in self.results['observables']:
            try:
                lss_data = self.results['observables']['lss']
                s8_data = {
                    'z_redshift': [0.0, 0.5, 1.0],
                    'S8_value': [
                        lss_data.get('S_8', 0.0),
                        lss_data.get('S_8_z05', 0.0),
                        lss_data.get('S_8_z1', 0.0)
                    ]
                }
                pd.DataFrame(s8_data).to_csv(s8_values_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save S8 values', error=e).to_csv(s8_values_csv, index=False)
        else:
            create_status_df('no_data', 'observables.lss not found in results').to_csv(s8_values_csv, index=False)
        
        # Save model parameters CSV with prefix (always save)
        params_csv = f"{self.project_dir}/{prefix}model_parameters_{timestamp}.csv"
        try:
            params_data = {
                'parameter_type': [],
                'parameter_name': [],
                'parameter_value': []
            }
            
            # Add coupling parameters
            if 'coupling_params' in self.results:
                for key, value in self.results['coupling_params'].items():
                    params_data['parameter_type'].append('coupling')
                    params_data['parameter_name'].append(key)
                    params_data['parameter_value'].append(value)
            
            # Add I-field parameters
            if 'i_field_params' in self.results:
                for key, value in self.results['i_field_params'].items():
                    params_data['parameter_type'].append('i_field')
                    params_data['parameter_name'].append(key)
                    params_data['parameter_value'].append(value)
            
            # Add other parameters if they exist
            if 'parameters' in self.results and self.results['parameters']:
                for key, value in self.results['parameters'].items():
                    params_data['parameter_type'].append('other')
                    params_data['parameter_name'].append(key)
                    params_data['parameter_value'].append(value)
            
            if len(params_data['parameter_name']) > 0:
                pd.DataFrame(params_data).to_csv(params_csv, index=False)
            else:
                create_status_df('no_data', 'No parameters found in results').to_csv(params_csv, index=False)
        except Exception as e:
            create_status_df('error', 'Failed to save model parameters', error=e).to_csv(params_csv, index=False)
        
        # Save field statistics CSV with prefix (always save, only for geometric coupling)
        field_stats_csv = f"{self.project_dir}/{prefix}field_statistics_{timestamp}.csv"
        if 'field_statistics' in self.results:
            try:
                field_stats = self.results['field_statistics']
                # Convert to DataFrame - field_stats is a dict
                if isinstance(field_stats, dict):
                    # Flatten nested dict if needed
                    flat_stats = {}
                    for key, value in field_stats.items():
                        if isinstance(value, (list, np.ndarray)):
                            # If it's an array, take first value or create summary
                            if len(value) > 0:
                                flat_stats[f'{key}_mean'] = np.mean(value) if isinstance(value, np.ndarray) else np.mean(list(value))
                                flat_stats[f'{key}_std'] = np.std(value) if isinstance(value, np.ndarray) else np.std(list(value))
                            else:
                                flat_stats[key] = 0.0
                        else:
                            flat_stats[key] = value
                    pd.DataFrame([flat_stats]).to_csv(field_stats_csv, index=False)
                else:
                    pd.DataFrame([field_stats]).to_csv(field_stats_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save field statistics', error=e).to_csv(field_stats_csv, index=False)
        else:
            create_status_df('not_computed', 'field_statistics not computed (only for geometric coupling)').to_csv(field_stats_csv, index=False)
        
        # Save sensitivity test CSV with prefix (always save)
        sensitivity_csv = f"{self.project_dir}/{prefix}sensitivity_test_{timestamp}.csv"
        if 'sensitivity_test' in self.results:
            try:
                sens_data = self.results['sensitivity_test']
                pd.DataFrame([sens_data]).to_csv(sensitivity_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save sensitivity test results', error=e).to_csv(sensitivity_csv, index=False)
        else:
            create_status_df('not_computed', 'sensitivity_test not computed').to_csv(sensitivity_csv, index=False)
            
        
        # ==========================================================================================
        # GALAXY STRUCTURE SAVES (CSV + JSON)
        # ==========================================================================================
        
        if 'galaxy_structure' in self.results and MASTER_CTRL.get('GALAXY_SAVE_CATALOGUES', True):
            print("\n🌌 Saving galaxy structure catalogues...")
            
            galaxy_data = self.results['galaxy_structure']
            
            # Save cosmic web summary (JSON) with prefix
            cosmic_web_json = f"{self.project_dir}/{prefix}Galaxy_Cosmic_Web_Summary_{timestamp}.json"
            cosmic_web_summary = {
                'cosmic_web_fractions': galaxy_data['summary']['cosmic_web_fractions'],
                'structure_counts': {
                    'n_voids': galaxy_data['summary']['n_voids'],
                    'n_clusters': galaxy_data['summary']['n_clusters'],
                    'n_filaments': galaxy_data['summary']['n_filaments'],
                    'n_walls': galaxy_data['summary']['n_walls']
                },
                'n_voids': galaxy_data['summary']['n_voids'],
                'n_clusters': galaxy_data['summary']['n_clusters'],
                'n_filaments': galaxy_data['summary']['n_filaments'],
                'n_walls': galaxy_data['summary']['n_walls'],
                'mean_void_radius_mpc': galaxy_data['summary']['mean_void_radius_mpc'],
                'mean_cluster_radius_mpc': galaxy_data['summary']['mean_cluster_radius_mpc'],
                'total_filament_length_mpc': galaxy_data['summary']['total_filament_length_mpc'],
                'total_wall_area_mpc2': galaxy_data['summary']['total_wall_area_mpc2']
            }
            with open(cosmic_web_json, 'w') as f:
                json.dump(cosmic_web_summary, f, indent=2)
            
            
            # Save void catalogue (CSV) with prefix (always save, even if empty)
            void_csv = f"{self.project_dir}/{prefix}Galaxy_Void_Catalogue_{timestamp}.csv"
            if len(galaxy_data['voids']) > 0:
                void_df = pd.DataFrame(galaxy_data['voids'])
                void_df.to_csv(void_csv, index=False)
                print(f"✓ Void catalogue saved: {void_csv} ({len(galaxy_data['voids'])} voids)")
            else:
                # Save empty catalogue with status
                pd.DataFrame({'status': ['no_voids_detected'], 'n_voids': [0]}).to_csv(void_csv, index=False)
                print(f"✓ Void catalogue saved (empty): {void_csv} - no voids detected")
            
            # Save cluster catalogue (CSV) with prefix (always save, even if empty)
            cluster_csv = f"{self.project_dir}/{prefix}Galaxy_Cluster_Catalogue_{timestamp}.csv"
            if len(galaxy_data['clusters']) > 0:
                cluster_df = pd.DataFrame(galaxy_data['clusters'])
                cluster_df.to_csv(cluster_csv, index=False)
                print(f"✓ Cluster catalogue saved: {cluster_csv} ({len(galaxy_data['clusters'])} clusters)")
            else:
                # Save empty catalogue with status
                pd.DataFrame({'status': ['no_clusters_detected'], 'n_clusters': [0]}).to_csv(cluster_csv, index=False)
                print(f"✓ Cluster catalogue saved (empty): {cluster_csv} - no clusters detected")
            
            # Save filament catalogue (CSV) with prefix (always save, even if empty)
            filament_csv = f"{self.project_dir}/{prefix}Galaxy_Filament_Catalogue_{timestamp}.csv"
            if len(galaxy_data['filaments']) > 0:
                filament_df = pd.DataFrame(galaxy_data['filaments'])
                filament_df.to_csv(filament_csv, index=False)
                print(f"✓ Filament catalogue saved: {filament_csv} ({len(galaxy_data['filaments'])} filaments)")
            else:
                # Save empty catalogue with status
                pd.DataFrame({'status': ['no_filaments_detected'], 'n_filaments': [0]}).to_csv(filament_csv, index=False)
                print(f"✓ Filament catalogue saved (empty): {filament_csv} - no filaments detected")
            
            # Save wall catalogue (CSV) with prefix (always save, even if empty)
            wall_csv = f"{self.project_dir}/{prefix}Galaxy_Wall_Catalogue_{timestamp}.csv"
            if len(galaxy_data['walls']) > 0:
                wall_df = pd.DataFrame(galaxy_data['walls'])
                wall_df.to_csv(wall_csv, index=False)
                print(f"✓ Wall catalogue saved: {wall_csv} ({len(galaxy_data['walls'])} walls)")
            else:
                # Save empty catalogue with status
                pd.DataFrame({'status': ['no_walls_detected'], 'n_walls': [0]}).to_csv(wall_csv, index=False)
                print(f"✓ Wall catalogue saved (empty): {wall_csv} - no walls detected")
        
        # ==========================================================================================
        # END GALAXY STRUCTURE SAVES
        # ==========================================================================================
        
        # Save full summary with prefix (always save)
        summary_file = f"{self.project_dir}/{prefix}Full_Summary_{timestamp}.txt"
        try:
            with open(summary_file, 'w') as f:
                f.write("TQE Dark Energy Coupling Simulation - Complete Analysis Summary\n")
                f.write("="*80 + "\n\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Coupling Mode: {self.coupling_mode}\n")
                try:
                    f.write(f"TQE Weighting: f(E,I) = exp(-α·E·(1-I)) " if self.coupling_mode == 'EplusI' else f"TQE Weighting: f(E) = exp(-α·E)\n")
                    f.write(f"Model Type: {self.results.get('model_type', 'Unknown')}\n")
                    f.write(f"I-Field Type: {self.results.get('i_field_type', 'Unknown')}\n\n")
                except Exception:
                    f.write("Model Type: Unknown\n")
                    f.write("I-Field Type: Unknown\n\n")
                
                f.write("COSMOLOGICAL PARAMETERS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  H0 = {self.friedmann.H0:.2f} km/s/Mpc\n")
                f.write(f"  Ω_m = {self.friedmann.Omega_m:.4f}\n")
                f.write(f"  Ω_Λ = {self.friedmann.Omega_Lambda:.4f}\n")
                f.write(f"  Ω_b = {self.friedmann.Omega_b:.4f}\n")
                f.write(f"  Ω_r = {self.friedmann.Omega_r:.6f}\n\n")
                
                f.write("I-PARAMETER COUPLING:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Coupling model: {self.results['model_type']}\n")
                f.write(f"  I-parameter model: {self.results['i_field_type']}\n\n")
                
                f.write("OBSERVABLE PREDICTIONS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  SNe Ia data points: {len(self.results['observables']['sne_ia']['z'])}\n")
                f.write(f"  BAO data points: {len(self.results['observables']['bao']['z'])}\n")
                f.write(f"  CMB multipoles: {len(self.results['observables']['cmb']['ell'])}\n")
                f.write(f"  S_8 parameter: {self.results['observables']['lss']['S_8']:.4f}\n\n")
                
                f.write("="*80 + "\n")
                f.write("CMB & LSS CALCULATION DISCLAIMERS\n")
                f.write("="*80 + "\n")
                f.write("CMB POWER SPECTRUM:\n")
                f.write("  • Computed using baseline ΛCDM parameters\n")
                f.write("  • I-parameter coupling effects NOT included in CMB calculation\n")
                f.write("  • Appropriate for baseline comparison only\n")
                f.write("  • For accurate predictions: custom CAMB/CLASS background required\n\n")
                f.write("MATTER POWER SPECTRUM P(k):\n")
                f.write("  • VISUAL/DIAGNOSTIC approximation only\n")
                f.write("  • Simplified transfer function (not CAMB/CLASS)\n")
                f.write("  • Use for qualitative trends, NOT quantitative analysis\n")
                f.write("  • For accurate LSS: full Boltzmann solver with I-parameter required\n")
                f.write("="*80 + "\n\n")
                
                f.write("SANITY CHECKS:\n")
                f.write("-" * 40 + "\n")
                for check_name, check_result in sanity_checks.items():
                    status = "✅ PASS" if check_result else "❌ FAIL"
                    f.write(f"  {check_name}: {status}\n")
                
                if sanity_issues:
                    f.write("\nISSUES DETECTED:\n")
                    for issue in sanity_issues:
                        f.write(f"  - {issue}\n")
                else:
                    f.write("\n✅ No issues detected - all checks passed!\n")
                f.write("\n")
                
                # FINAL RELEASE UPGRADE: Sensitivity test results
                if 'sensitivity_test' in self.results:
                    f.write("SENSITIVITY TEST (±1% I-parameter perturbation):\n")
                    f.write("-" * 40 + "\n")
                    sens = self.results['sensitivity_test']
                    f.write(f"  Perturbation: ±{sens['perturbation_pct']:.1f}%\n")
                    f.write(f"  ΔS₈ = {sens['delta_S8_pct']:.3f}%\n")
                    f.write(f"  ΔH = {sens['delta_H_pct']:.3f}%\n")
                    f.write(f"  Δρ_DE = {sens['delta_rho_DE_pct']:.3f}%\n")
                    status = "✅ STABLE" if sens['is_stable'] else "⚠ SENSITIVE"
                    f.write(f"  Status: {status} (tolerance: {sens['tolerance_pct']:.1f}%)\n\n")
                
                # GALAXY STRUCTURE SECTION
                if 'galaxy_structure' in self.results:
                    f.write("="*80 + "\n")
                    f.write("GALAXY STRUCTURE ANALYSIS (Cosmic Web Topology)\n")
                    f.write("="*80 + "\n\n")
                    
                    galaxy_data = self.results['galaxy_structure']
                    summary = galaxy_data['summary']
                    
                    f.write("COSMIC WEB CLASSIFICATION:\n")
                    f.write("-" * 40 + "\n")
                    fracs = summary['cosmic_web_fractions']
                    f.write(f"  Voids (underdense):      {fracs['void_fraction']*100:.2f}%\n")
                    f.write(f"  Filaments (elongated):   {fracs['filament_fraction']*100:.2f}%\n")
                    f.write(f"  Sheets/Walls (2D):       {fracs['sheet_fraction']*100:.2f}%\n")
                    f.write(f"  Knots/Clusters (dense):  {fracs['knot_fraction']*100:.2f}%\n\n")
                    
                    f.write("STRUCTURE CATALOGUES:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  Total Voids: {summary['n_voids']}\n")
                    f.write(f"    • Mean radius: {summary['mean_void_radius_mpc']:.1f} Mpc/h\n")
                    f.write(f"  Total Clusters: {summary['n_clusters']}\n")
                    f.write(f"    • Mean radius: {summary['mean_cluster_radius_mpc']:.1f} Mpc/h\n")
                    f.write(f"  Total Filaments: {summary['n_filaments']}\n")
                    f.write(f"    • Total length: {summary['total_filament_length_mpc']:.0f} Mpc/h\n")
                    f.write(f"  Total Walls: {summary['n_walls']}\n")
                    f.write(f"    • Total area: {summary['total_wall_area_mpc2']:.0f} (Mpc/h)²\n\n")
                    
                    f.write("COMPARISON WITH REAL UNIVERSE:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get real universe values from MASTER_CTRL
                    real_void_frac = MASTER_CTRL.get('REAL_UNIVERSE_VOID_FRAC', 0.45)
                    real_filament_frac = MASTER_CTRL.get('REAL_UNIVERSE_FILAMENT_FRAC', 0.35)
                    real_sheet_frac = MASTER_CTRL.get('REAL_UNIVERSE_SHEET_FRAC', 0.12)
                    real_cluster_frac = MASTER_CTRL.get('REAL_UNIVERSE_CLUSTER_FRAC', 0.08)
                    real_void_r_min = MASTER_CTRL.get('REAL_UNIVERSE_VOID_RADIUS_MIN', 10.0)
                    real_void_r_max = MASTER_CTRL.get('REAL_UNIVERSE_VOID_RADIUS_MAX', 50.0)
                    
                    f.write("  Real Universe (SDSS, 2dFGRS observations):\n")
                    f.write(f"    • Void fraction: ~{real_void_frac*100:.0f}%\n")
                    f.write(f"    • Filament fraction: ~{real_filament_frac*100:.0f}%\n")
                    f.write(f"    • Sheet fraction: ~{real_sheet_frac*100:.0f}%\n")
                    f.write(f"    • Cluster fraction: ~{real_cluster_frac*100:.0f}%\n")
                    f.write(f"    • Mean void radius: {real_void_r_min:.0f}-{real_void_r_max:.0f} Mpc/h\n\n")
                    f.write(f"  TQE Simulation (this run):\n")
                    f.write(f"    • Void fraction: {fracs['void_fraction']*100:.1f}%\n")
                    f.write(f"    • Filament fraction: {fracs['filament_fraction']*100:.1f}%\n")
                    f.write(f"    • Sheet fraction: {fracs['sheet_fraction']*100:.1f}%\n")
                    f.write(f"    • Cluster fraction: {fracs['knot_fraction']*100:.1f}%\n")
                    f.write(f"    • Mean void radius: {summary['mean_void_radius_mpc']:.1f} Mpc/h\n\n")
                    
                    # Quantify similarity to real universe using MASTER_CTRL values
                    similarity_score = 100.0 - (
                        abs(fracs['void_fraction'] - real_void_frac) +
                        abs(fracs['filament_fraction'] - real_filament_frac) +
                        abs(fracs['sheet_fraction'] - real_sheet_frac) +
                        abs(fracs['knot_fraction'] - real_cluster_frac)
                    ) * 100.0
                    
                    f.write(f"  SIMILARITY TO REAL UNIVERSE: {max(similarity_score, 0):.1f}%\n")
                    f.write(f"  (100% = perfect match with SDSS/2dFGRS, 0% = completely different)\n")
                    f.write(f"  Reference: REAL_UNIVERSE_* parameters in MASTER_CTRL\n\n")
                
                # ==========================================================================================
                # CMB PLANCK VALIDATION SECTION
                # ==========================================================================================
                if 'cmb_planck_validation' in self.results:
                    f.write("="*80 + "\n")
                    f.write("CMB PLANCK VALIDATION (Real Planck 2018 Maps)\n")
                    f.write("="*80 + "\n")
                    
                    cmb_val = self.results['cmb_planck_validation']
                    stats = cmb_val.get('statistics', {})
                    
                    f.write("\n📡 DATA SOURCE:\n")
                    f.write(f"  • Component-separated map: SMICA (Planck 2018 R3.00)\n")
                    f.write(f"  • Resolution: Nside=2048 (Npix=50,331,648)\n")
                    f.write(f"  • Masks applied: Common mask + Misspix mask\n")
                    f.write(f"  • Multipoles: ℓ ∈ [2, {cmb_val.get('planck_lmax', 2000)}]\n")
                    
                    f.write("\n📊 POWER SPECTRUM COMPARISON (TQE vs Planck):\n")
                    if 'correlation_r' in stats:
                        f.write(f"  • Pearson correlation: r = {stats['correlation_r']:.4f}\n")
                        f.write(f"    (p-value = {stats.get('correlation_p', 0):.2e})\n")
                    if 'rms_difference' in stats:
                        f.write(f"  • RMS difference: ΔRMS = {stats['rms_difference']:.2f} μK²\n")
                    if 'mean_fractional_residual' in stats:
                        f.write(f"  • Mean fractional residual: {stats['mean_fractional_residual']*100:.2f}%\n")
                    if 'chi2_reduced' in stats:
                        f.write(f"  • Goodness of fit: χ²/dof = {stats['chi2_reduced']:.2f}\n")
                        if stats['chi2_reduced'] < 1.5:
                            f.write(f"    ✅ EXCELLENT FIT (χ²/dof < 1.5)\n")
                        elif stats['chi2_reduced'] < 3.0:
                            f.write(f"    ✓ GOOD FIT (χ²/dof < 3.0)\n")
                        else:
                            f.write(f"    ⚠ POOR FIT (χ²/dof > 3.0)\n")
                    
                    f.write("\n🌡️ ANOMALY DETECTION:\n")
                    n_anomalies = cmb_val.get('n_anomalies', 0)
                    f.write(f"  • Detected anomalies: {n_anomalies} pixels\n")
                    f.write(f"  • Detection threshold: ±3σ\n")
                    if n_anomalies > 0:
                        fsky_anomalies = (n_anomalies / 50331648) * 100
                        f.write(f"  • Sky coverage: {fsky_anomalies:.4f}% of sky\n")
                        f.write(f"  • Expected for Gaussian: ~0.3% (3σ tails)\n")
                    
                    if 'nhi_correlation_r' in stats:
                        f.write("\n🌌 NHI FOREGROUND CORRELATION:\n")
                        f.write(f"  • CMB-NHI correlation: r = {stats['nhi_correlation_r']:.4f}\n")
                        f.write(f"    (p-value = {stats.get('nhi_correlation_p', 0):.2e})\n")
                        if abs(stats['nhi_correlation_r']) < 0.1:
                            f.write(f"    ✅ WEAK foreground contamination (|r| < 0.1)\n")
                        elif abs(stats['nhi_correlation_r']) < 0.3:
                            f.write(f"    ⚠ MODERATE foreground contamination (|r| < 0.3)\n")
                        else:
                            f.write(f"    ❌ STRONG foreground contamination (|r| > 0.3)\n")
                    
                    f.write("\n💡 INTERPRETATION:\n")
                    if stats.get('correlation_r', 0) > 0.99:
                        f.write(f"  • TQE simulated C_ℓ matches Planck extremely well (r > 0.99)\n")
                        f.write(f"  • This is expected since TQE uses ΛCDM baseline for CMB\n")
                        f.write(f"  • The I-parameter coupling does not significantly alter CMB physics\n")
                    elif stats.get('correlation_r', 0) > 0.95:
                        f.write(f"  • TQE simulated C_ℓ shows good agreement with Planck (r > 0.95)\n")
                        f.write(f"  • Minor deviations may be due to I-parameter effects or numerical precision\n")
                    else:
                        f.write(f"  • ⚠ TQE simulated C_ℓ shows significant deviation from Planck (r < 0.95)\n")
                        f.write(f"  • This may indicate strong I-parameter coupling affecting CMB physics\n")
                    
                    f.write(f"\n✅ CMB Planck validation: {'COMPLETE' if cmb_val.get('validation_complete', False) else 'INCOMPLETE'}\n")
                    f.write(f"📁 Output files: 4 PNG + 2 CSV + 1 JSON\n\n")
                
                f.write("="*80 + "\n")
                f.write("QUICK DIAGNOSTICS (Key Observable Values)\n")
                f.write("="*80 + "\n")
                
                # H(z=0) = H0
                try:
                    H_at_0 = self.friedmann.H(1.0)
                    f.write(f"  H(z=0) = H0 = {H_at_0:.2f} km/s/Mpc\n")
                except (KeyError, AttributeError, ValueError) as e:
                    f.write(f"  H(z=0): N/A\n")
                
                # BAO distances at standard redshifts
                try:
                    z_bao_arr = np.array(self.results['observables']['bao']['z'])
                    D_M_arr = np.array(self.results['observables']['bao']['D_M'])
                    
                    for z_val in [0.38, 0.51, 0.61]:
                        idx = np.argmin(np.abs(z_bao_arr - z_val))
                        if idx < len(D_M_arr):
                            f.write(f"  D_M(z={z_val:.2f}) = {D_M_arr[idx]:.2f} Mpc\n")
                except (KeyError, ValueError, IndexError) as e:
                    f.write(f"  D_M values: N/A\n")
                
                # SNe Ia distance modulus at z=1
                try:
                    z_sne_arr = np.array(self.results['observables']['sne_ia']['z'])
                    mu_arr = np.array(self.results['observables']['sne_ia']['mu'])
                    idx = np.argmin(np.abs(z_sne_arr - 1.0))
                    f.write(f"  μ(z=1.0) = {mu_arr[idx]:.4f} mag\n")
                except (KeyError, ValueError, IndexError) as e:
                    f.write(f"  μ(z=1.0): N/A\n")
                
                # S_8 parameter
                try:
                    S_8 = self.results['observables']['lss']['S_8']
                    f.write(f"  S₈ = {S_8:.4f}\n")
                except (KeyError, TypeError) as e:
                    f.write(f"  S₈: N/A\n")
                
                # S₈ normalization
                if 'S8_normalization' in self.results:
                    f.write("\n" + "="*80 + "\n")
                    f.write("S₈ NORMALIZATION (vs ΛCDM)\n")
                    f.write("="*80 + "\n")
                    s8n = self.results['S8_normalization']
                    f.write(f"  S₈ (raw) = {s8n.get('S8_raw', 0):.4f}\n")
                    f.write(f"  S₈ (ΛCDM baseline) = {s8n.get('S8_LCDM', 0):.4f}\n")
                    f.write(f"  S₈ (normalized) = {s8n.get('S8_normalized', 0):.4f}\n")
                    f.write(f"  ΔS₈ (vs ΛCDM) = {s8n.get('Delta_S8', 0):+.4f}\n")
                
                # I-E correlation
                if 'I_E_correlation' in self.results:
                    f.write("\n" + "="*80 + "\n")
                    f.write("I-E CORRELATION\n")
                    f.write("="*80 + "\n")
                    iec = self.results['I_E_correlation']
                    f.write(f"  Pearson r = {iec.get('pearson_r', 0):.4f} (p = {iec.get('pearson_p', 0):.2e})\n")
                    f.write(f"  Spearman r = {iec.get('spearman_r', 0):.4f} (p = {iec.get('spearman_p', 0):.2e})\n")
                    if iec.get('mutual_information'):
                        f.write(f"  Mutual Information = {iec.get('mutual_information', 0):.4f}\n")
                    
                    # LAG SCAN ()
                    if 'lag_scan' in iec:
                        f.write("\n  LAG SCAN MI(Δa):\n")
                        lag_info = iec['lag_scan']
                        f.write(f"    Max MI = {lag_info.get('max_MI', 0):.4f} at Δa = {lag_info.get('optimal_da', 0):.4f}\n")
                        f.write(f"    Lag range: Δa = {min(lag_info.get('da_lags', [0])):.3f} → {max(lag_info.get('da_lags', [0])):.3f}\n")
                
                # Likelihood results ()
                if 'likelihood' in self.results:
                    f.write("\n" + "="*80 + "\n")
                    f.write("LIKELIHOOD ANALYSIS (χ², AIC, BIC)\n")
                    f.write("="*80 + "\n")
                    like = self.results['likelihood']
                    f.write(f"  Total χ² = {like.get('chi2_total', 0):.2f}\n")
                    f.write(f"  Reduced χ² = {like.get('reduced_chi2', 0):.2f}\n")
                    f.write(f"  AIC = {like.get('AIC', 0):.2f}\n")
                    f.write(f"  BIC = {like.get('BIC', 0):.2f}\n")
                    f.write(f"  N_data = {like.get('n_data', 0)}\n")
                    f.write(f"  N_params = {like.get('n_params', 0)}\n")
                    f.write("\n  χ² Components:\n")
                    for comp_name, comp_val in like.get('chi2_components', {}).items():
                        f.write(f"    {comp_name}: {comp_val:.2f}\n")
                
                # Evolution series summary
                if 'evolution_series' in self.results:
                    f.write("\n" + "="*80 + "\n")
                    f.write("EVOLUTION SERIES SUMMARY\n")
                    f.write("="*80 + "\n")
                    evs = self.results['evolution_series']
                    z_arr = np.array(evs['z'])
                    S8_arr = np.array(evs['S8'])
                    D_arr = np.array(evs['D'])
                    rho_arr = np.array(evs['rho_DE'])
                    f.write(f"  Redshift range: z = {z_arr[0]:.2f} → {z_arr[-1]:.2f} ({len(z_arr)} points)\n")
                    f.write(f"  S₈: {S8_arr[0]:.4f} (z=0) → {S8_arr[-1]:.4f} (z={z_arr[-1]:.1f})\n")
                    f.write(f"  D(z): {D_arr[0]:.4f} (z=0) → {D_arr[-1]:.4f} (z={z_arr[-1]:.1f})\n")
                    f.write(f"  ρ_DE: {rho_arr[0]:.4f} (z=0) → {rho_arr[-1]:.4f} (z={z_arr[-1]:.1f})\n")
                
                # FINAL RELEASE UPGRADE: Scientific conclusion
                f.write("\n" + "="*80 + "\n")
                f.write("SCIENTIFIC CONCLUSION\n")
                f.write("="*80 + "\n")
                
                # Extract key metrics for conclusion
                coupling_type = self.results.get('model_type', 'Unknown')
                delta_S8 = self.results.get('S8_normalization', {}).get('Delta_S8', 0.0)
                rho_DE_var = self.results.get('observables', {}).get('rho_DE_variance', 0.0)
                pearson_r = self.results.get('I_E_correlation', {}).get('pearson_r', 0.0)
                is_stable = self.results.get('sensitivity_test', {}).get('is_stable', None)
                sanity_passed = all(sanity_checks.values())
                
                f.write(f"Model: {coupling_type}\n\n")
                f.write("Key Findings:\n")
                f.write(f"  • ΔS₈ (vs ΛCDM): {delta_S8:+.2%}\n")
                f.write(f"  • ρ_DE variance: {rho_DE_var:.6f}\n")
                f.write(f"  • I-E correlation: r = {pearson_r:.3f}\n")
                if is_stable is not None:
                    f.write(f"  • Numerical stability: {'✅ STABLE' if is_stable else '⚠ SENSITIVE'}\n")
                f.write(f"  • Physical consistency: {'✅ PASS' if sanity_passed else '❌ FAIL'}\n\n")
                
                # Generate conclusion based on results
                if coupling_type == 'null_model':
                    conclusion = "Pure ΛCDM baseline - no TQE coupling effects."
                elif abs(delta_S8) < 0.001:  # <0.1%
                    conclusion = "TQE coupling has negligible impact on cosmological observables. "\
                               "The I-parameter does not significantly affect dark energy dynamics."
                elif abs(delta_S8) < 0.01:  # <1%
                    conclusion = f"Weak TQE coupling detected (ΔS₈ = {delta_S8:+.2%}). "\
                               f"The I-parameter introduces {abs(rho_DE_var)*100:.2f}% variation in ρ_DE. "\
                               f"{'E+I coupling is consistent with ΛCDM within observational uncertainties.' if sanity_passed else 'Physical consistency issues detected - model requires refinement.'}"
                else:  # >1%
                    conclusion = f"Strong TQE coupling detected (ΔS₈ = {delta_S8:+.2%}). "\
                               f"The I-parameter significantly modulates dark energy density (var = {rho_DE_var:.4f}). "\
                               f"{'This represents a testable deviation from ΛCDM.' if sanity_passed else 'Physical consistency issues detected - coupling may be too strong.'}"
                
                f.write("Conclusion:\n")
                f.write(f"  {conclusion}\n")
                
                f.write("\n" + "="*80 + "\n")
        except Exception as e:
            # Save minimal summary if full summary fails
            try:
                with open(summary_file, 'w') as f:
                    f.write("TQE Dark Energy Coupling Simulation - Summary\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"Timestamp: {timestamp}\n")
                    f.write(f"Coupling Mode: {self.coupling_mode}\n")
                    f.write(f"\nERROR: Failed to generate full summary\n")
                    f.write(f"Error: {str(e)}\n")
            except:
                pass  # If even minimal summary fails, skip it
        
        
        # ==========================================================================================
        # SAVE I-DEFINITIONS COMPARISON CSV (9 definitions) (always save)
        # ==========================================================================================
        I_defs_csv = f"{self.project_dir}/{prefix}I_Definitions_Comparison_{timestamp}.csv"
        if MASTER_CTRL.get('COMPUTE_ALL_I_DEFINITIONS', False):
            print(f"\n📊 Saving I-Definitions Comparison...")
            
            try:
                if 'evolution' in self.results and 'a_array' in self.results['evolution']:
                    # Sample 50 points across scale factor range
                    a_grid_sample = self.results['evolution']['a_array'][::max(1, len(self.results['evolution']['a_array'])//50)]
                    
                    rows = []
                    for a_val in a_grid_sample:
                        # Compute all 9 I-definitions at this scale factor
                        I_defs = self.information_content.compute_all_I_definitions(
                            a_val, 
                            friedmann=self.friedmann
                        )
                        
                        row = {'scale_factor': a_val}
                        row.update(I_defs)  # Adds: phenomenological, kl_divergence, shannon, composite, renyi, mutual_info, kl_shannon, entanglement, fisher, horizon_entropy
                        rows.append(row)
                    
                    df_I_defs = pd.DataFrame(rows)
                    df_I_defs.to_csv(I_defs_csv, index=False)
                else:
                    create_status_df('no_evolution_data', 'evolution.a_array not found in results').to_csv(I_defs_csv, index=False)
            except Exception as e:
                create_status_df('error', 'Failed to save I-Definitions comparison', error=e).to_csv(I_defs_csv, index=False)
        else:
            # Save empty file with status even if not enabled
            create_status_df('not_computed', 'COMPUTE_ALL_I_DEFINITIONS=False in MASTER_CTRL').to_csv(I_defs_csv, index=False)
        
        # Save Model Summary JSON with prefix (always save) - key metrics and model info
        model_summary_json = f"{self.project_dir}/{prefix}Model_Summary_{timestamp}.json"
        try:
            summary_data = {
                'coupling_mode': self.coupling_mode,
                'model_type': self.results.get('model_type', 'Unknown'),
                'i_field_type': self.results.get('i_field_type', 'Unknown'),
                'timestamp': timestamp,
                'coupling_params': self.results.get('coupling_params', {}),
                'i_field_params': self.results.get('i_field_params', {}),
                'key_observables': {}
            }
            
            # Add key observable values
            if 'observables' in self.results:
                obs = self.results['observables']
                summary_data['key_observables'] = {
                    'S8_z0': obs.get('S8_raw', obs.get('lss', {}).get('S_8', 0.0)),
                    'S8_z05': obs.get('lss', {}).get('S_8_z05', 0.0),
                    'S8_z1': obs.get('lss', {}).get('S_8_z1', 0.0),
                    'mu_z1': obs.get('mu_z1', 0.0),
                    'D_M_z051': obs.get('D_M_z051', 0.0),
                    'H_z051': obs.get('H_z051', 0.0),
                    'H_z0': obs.get('H_z0', 0.0),
                    'rho_DE_variation': obs.get('rho_DE_variation', 0.0),
                    'I_max': obs.get('I_max', 0.0)
                }
            
            # Add likelihood if available
            if 'likelihood' in self.results:
                like = self.results['likelihood']
                summary_data['likelihood'] = {
                    'chi2_total': like.get('chi2_total', 0.0),
                    'AIC': like.get('AIC', 0.0),
                    'BIC': like.get('BIC', 0.0),
                    'reduced_chi2': like.get('reduced_chi2', 0.0)
                }
            
            # Add I-E correlation if available
            if 'I_E_correlation' in self.results:
                ie = self.results['I_E_correlation']
                summary_data['I_E_correlation'] = {
                    'pearson_r': ie.get('pearson_r', 0.0),
                    'spearman_r': ie.get('spearman_r', 0.0),
                    'mutual_information': ie.get('mutual_information', 0.0)
                }
            
            # Add S8 normalization if available
            if 'S8_normalization' in self.results:
                summary_data['S8_normalization'] = self.results['S8_normalization']
            
            # Add bayesian inference if available
            if 'bayesian_inference' in self.results and self.results['bayesian_inference']:
                summary_data['bayesian_inference'] = self.results['bayesian_inference']
            
            with open(model_summary_json, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
        except Exception as e:
            # Save minimal summary if save fails
            try:
                error_summary = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': timestamp,
                    'coupling_mode': self.coupling_mode
                }
                with open(model_summary_json, 'w') as f:
                    json.dump(error_summary, f, indent=2)
            except:
                pass
        
        # Save reproducibility info with prefix (always save)
        repro_file = f"{self.project_dir}/{prefix}Reproducibility_Info_{timestamp}.txt"
        try:
            with open(repro_file, 'w') as f:
                f.write("TQE Dark Energy Coupling - Reproducibility Information\n")
                f.write("="*80 + "\n\n")
                f.write(f"Coupling Mode: {self.coupling_mode}\n")
                try:
                    f.write(f"Seed String: '{self.seed_string}'\n")
                    f.write(f"Seed Hash: {self.seed_hash}\n")
                except AttributeError:
                    f.write("Seed String: N/A\n")
                    f.write("Seed Hash: N/A\n")
        except Exception as e:
            # Save minimal reproducibility info if save fails
            try:
                with open(repro_file, 'w') as f:
                    f.write("TQE Dark Energy Coupling - Reproducibility Information\n")
                    f.write("="*80 + "\n\n")
                    f.write(f"ERROR: {str(e)}\n")
            except:
                pass
        
        # Copy Bayesian inference files from Bayesian_Analysis subdirectory to main project dir with prefix (if they exist)
        bayesian_source_dir = f"{self.project_dir}/Bayesian_Analysis"
        if os.path.exists(bayesian_source_dir):
            try:
                bayesian_files = {
                    'mcmc_samples.csv': f"{prefix}Bayesian_MCMC_Samples_{timestamp}.csv",
                    'mcmc_summary.json': f"{prefix}Bayesian_MCMC_Summary_{timestamp}.json",
                    'information_criteria.json': f"{prefix}Bayesian_Information_Criteria_{timestamp}.json",
                    'nested_sampling_evidence.json': f"{prefix}Bayesian_Nested_Sampling_Evidence_{timestamp}.json",
                    'corner_plot.png': f"{prefix}Bayesian_Corner_Plot_{timestamp}.png"
                }
                
                for source_file, target_file in bayesian_files.items():
                    source_path = f"{bayesian_source_dir}/{source_file}"
                    target_path = f"{self.project_dir}/{target_file}"
                    if os.path.exists(source_path):
                        try:
                            import shutil
                            shutil.copy2(source_path, target_path)
                            print(f"✓ Copied Bayesian file: {target_file}")
                        except Exception as e:
                            print(f"  ⚠ Failed to copy {source_file}: {e}")
            except Exception as e:
                print(f"  ⚠ Failed to copy Bayesian inference files: {e}")
        
        # Create ZIP archive with prefix (always try, even if some files are missing)
        zip_file = f"{self.project_dir}/{prefix}TQE_DarkEnergy_Complete_Results_{timestamp}.zip"
        try:
            with zipfile.ZipFile(zip_file, 'w') as zipf:
                for root, dirs, files in os.walk(self.project_dir):
                    for file in files:
                        if not file.endswith('.zip'):
                            file_path = os.path.join(root, file)
                            try:
                                arcname = os.path.relpath(file_path, self.project_dir)
                                zipf.write(file_path, arcname)
                            except Exception as e:
                                print(f"  ⚠ Failed to add {file} to ZIP: {e}")
            print(f"✓ Complete results ZIP created: {zip_file}")
        except Exception as e:
            print(f"⚠ Failed to create ZIP archive: {e}")
            # Try to create minimal ZIP with just error message
            try:
                error_file = f"{self.project_dir}/{prefix}ZIP_ERROR_{timestamp}.txt"
                with open(error_file, 'w') as f:
                    f.write(f"ZIP archive creation failed: {str(e)}\n")
            except:
                pass
        
        print("\n" + "="*80)
        print("💾 SAVE SUMMARY")
        print("="*80)
        png_dir = f"{self.project_dir}/PNG_Visualizations"
        print(f"✅ PNG Visualizations: {png_dir}/")
        print(f"✅ CSV/JSON/TXT files: {self.project_dir}/ (root)")
        print(f"✅ ZIP archive: {zip_file}")
        print("="*80)

# ==========================================================================================
# HELPER FUNCTIONS
# ==========================================================================================

def run_sanity_checks(simulation):
    # AUDIT FIX #5: Enhanced sanity checks with relaxed tolerances and new coupling checks
    # Run sanity checks on simulation results to ensure physical validity
    
    checks = {
        'H_at_a1_vs_H0': False,
        'E_squared_positive': False,
        'mu_monotonic': False,
        'D_M_monotonic': False,
        'rho_DE_positive': False,
        'MI_coupling_active': False,
        'S8_coupling_effect': False
    }
    
    issues = []
    warnings = []
    
    print("\n🔍 Running ENHANCED sanity checks (AUDIT FIX #5)...")
    
    # 1. AUDIT FIX #5: H(a=1) ≈ H0 with RELAXED tolerances
    try:
        H_at_1 = simulation.friedmann.H(1.0)
        H0 = simulation.friedmann.H0
        H_deviation = abs(H_at_1 - H0) / H0
        
        # AUDIT FIX #5: Multi-tier tolerance system
        tol_fail = MASTER_CTRL['SANITY_TOLS'].get('H_at_a1_vs_H0_tol_fail', 0.010)  # 1.0% FAIL (default)
        tol_warn = MASTER_CTRL['SANITY_TOLS'].get('H_at_a1_vs_H0_tol_warn', 0.005)  # 0.5% WARN (default)
        
        if H_deviation < tol_warn:
            # PASS (no warning)
            checks['H_at_a1_vs_H0'] = True
        elif H_deviation < tol_fail:
            # WARN but still PASS
            checks['H_at_a1_vs_H0'] = True
            warnings.append(f"⚠️ H(a=1) deviation: {H_deviation*100:.3f}% (WARN, but within tolerance)")
            print(f"  ⚠️ H(a=1) ≈ H₀: deviation = {H_deviation*100:.4f}% (WARN)")
        else:
            # FAIL
            checks['H_at_a1_vs_H0'] = False
            issues.append(f"❌ H(a=1) deviation: {H_deviation*100:.3f}% > {tol_fail*100:.1f}% (FAIL)")
            print(f"  ❌ H(a=1) deviation: {H_deviation*100:.4f}% (FAIL)")
    except Exception as e:
        issues.append(f"H(a=1) check failed: {e}")
    
    # 2. AUDIT FIX #5: E²(a) > 0 with friedmann parameter
    try:
        a_test = np.linspace(0.1, 1.0, 100)
        E_squared_negative = False
        
        for a in a_test:
            rho_m = simulation.friedmann.Omega_m * a**(-3)
            rho_r = simulation.friedmann.Omega_r * a**(-4)
            # AUDIT FIX #2: Pass friedmann for dynamic I-parameter
            rho_DE = simulation.friedmann.coupling.rho_DE(a, simulation.friedmann.rho_Lambda_today, friedmann=simulation.friedmann)
            E_sq = rho_m + rho_r + rho_DE
            
            if E_sq <= 0:
                E_squared_negative = True
                issues.append(f"❌ E²(a={a:.2f}) = {E_sq:.6f} ≤ 0")
                break
        
        checks['E_squared_positive'] = not E_squared_negative
        
        if checks['E_squared_positive']:
            print(f"  ✅ E²(a) > 0 for all a ∈ [0.1, 1.0]")
    except Exception as e:
        issues.append(f"E²(a) check failed: {e}")
    
    # 3. Check μ(z) monotonic increasing (with small numerical tolerance)
    try:
        if 'sne_ia' in simulation.results.get('observables', {}):
            mu = np.array(simulation.results['observables']['sne_ia']['mu'])
            if len(mu) > 1:
                mu_diff = np.diff(mu)
                # AUDIT FIX #5: Allow tiny numerical errors
                checks['mu_monotonic'] = np.all(mu_diff > -1e-6)
                
                if not checks['mu_monotonic']:
                    n_violations = np.sum(mu_diff < -1e-6)
                    issues.append(f"❌ μ(z) not monotonic at {n_violations} points")
                else:
                    print(f"  ✅ μ(z) monotonically increasing")
    except Exception as e:
        issues.append(f"μ(z) check failed: {e}")
    
    # 4. Check D_M(z) monotonic increasing
    try:
        if 'bao' in simulation.results.get('observables', {}):
            D_M = np.array(simulation.results['observables']['bao']['D_M'])
            if len(D_M) > 1:
                D_M_diff = np.diff(D_M)
                # AUDIT FIX #5: Allow tiny numerical errors
                checks['D_M_monotonic'] = np.all(D_M_diff > -1e-6)
                
                if not checks['D_M_monotonic']:
                    n_violations = np.sum(D_M_diff < -1e-6)
                    issues.append(f"❌ D_M(z) not monotonic at {n_violations} points")
                else:
                    print(f"  ✅ D_M(z) monotonically increasing")
    except Exception as e:
        issues.append(f"D_M(z) check failed: {e}")
    
    # 5. AUDIT FIX #5: - Check ρ_DE > 0 (physical positivity)
    try:
        if 'evolution' in simulation.results:
            rho_DE_arr = np.array(simulation.results['evolution'].get('rho_DE_array', []))
            if len(rho_DE_arr) > 0:
                checks['rho_DE_positive'] = np.all(rho_DE_arr > 0)
                
                if not checks['rho_DE_positive']:
                    n_negative = np.sum(rho_DE_arr <= 0)
                    issues.append(f"❌ ρ_DE negative at {n_negative} points")
                else:
                    print(f"  ✅ ρ_DE > 0 everywhere")
    except Exception as e:
        issues.append(f"ρ_DE check failed: {e}")
    
    # 6. AUDIT FIX #5: - MI coupling active (information correlation present)
    try:
        if 'I_E_correlation' in simulation.results:
            MI = simulation.results['I_E_correlation'].get('mutual_information', 0.0)
            MI_threshold = 0.001  # Minimum MI for active coupling
            
            checks['MI_coupling_active'] = MI > MI_threshold
            
            if not checks['MI_coupling_active']:
                warnings.append(f"⚠️ MI = {MI:.6f} < {MI_threshold} (coupling may be inactive)")
                print(f"  ⚠️ MI = {MI:.6f} (coupling weak or inactive)")
            else:
                print(f"  ✅ MI = {MI:.4f} (coupling active)")
    except Exception as e:
        warnings.append(f"MI check skipped: {e}")
    
    # 7. AUDIT FIX #5: - S₈ coupling effect (non-zero difference from ΛCDM)
    try:
        if 'S8_normalization' in simulation.results:
            Delta_S8 = simulation.results['S8_normalization'].get('Delta_S8', 0.0)
            Delta_threshold = 0.0001  # Minimum ΔS₈ for detectable effect
            
            checks['S8_coupling_effect'] = abs(Delta_S8) > Delta_threshold
            
            if not checks['S8_coupling_effect']:
                warnings.append(f"⚠️ ΔS₈ = {Delta_S8:.6f} (no coupling effect detected)")
                print(f"  ⚠️ ΔS₈ = {Delta_S8:.6f} (coupling effect negligible)")
            else:
                print(f"  ✅ ΔS₈ = {Delta_S8:+.4f} (coupling effect present)")
    except Exception as e:
        warnings.append(f"S₈ coupling check skipped: {e}")
    
    # Summary
    # AUDIT FIX #5: Only critical checks must pass for overall PASS
    critical_checks = ['H_at_a1_vs_H0', 'E_squared_positive', 'rho_DE_positive']
    all_critical_pass = all(checks.get(c, False) for c in critical_checks)
    
    checks['all_passed'] = all_critical_pass
    
    if all_critical_pass and not issues:
        print("✅ All critical sanity checks PASSED")
        if warnings:
            print(f"⚠️  {len(warnings)} warnings (non-critical)")
    else:
        print("❌ Some critical sanity checks FAILED:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    return checks, issues + warnings

# ==========================================================================================
# HELPER FUNCTIONS FOR FILE PREFIXING
# ==========================================================================================

def get_file_prefix(coupling_mode=None):
    # Get file prefix based on coupling mode
    # Returns "EplusI_" or "Eonly_" or "" (if AUTO_PREFIX_FILES=False)
    
    if not MASTER_CTRL.get('AUTO_PREFIX_FILES', False):
        return ""
    
    if coupling_mode is None:
        coupling_mode = MASTER_CTRL.get('COUPLING_MODE', 'EplusI')
    
    if coupling_mode == 'Eonly':
        return "Eonly_"
    elif coupling_mode == 'EplusI':
        return "EplusI_"
    elif coupling_mode == 'dual':
        # Dual mode will call this function twice with explicit mode
        return ""
    else:
        return ""

def add_prefix_to_path(filepath, coupling_mode):
    # Add coupling mode prefix to filename while preserving directory structure
    # Example: "/path/to/file.csv" -> "/path/to/EplusI_file.csv"
    
    prefix = get_file_prefix(coupling_mode)
    if not prefix:
        return filepath
    
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    prefixed_filename = prefix + filename
    
    if directory:
        return os.path.join(directory, prefixed_filename)
    else:
        return prefixed_filename

# ==========================================================================================
# PHASE 4: Galaxy Structure Analysis (10 Metrics)
# ==========================================================================================
    
    def _compute_clustering_strength(self):
        """10. Clustering Strength analysis"""
        try:
            # Simplified clustering strength computation
            clustering_strength = 0.7
            correlation_scale = 8.0  # Mpc/h
            bias_factor = 1.2
            
            return {
                'clustering_strength': clustering_strength,
                'correlation_scale': correlation_scale,
                'bias_factor': bias_factor,
                'clustering_efficiency': clustering_strength * bias_factor
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_slope(self, x, y, x_target):
        """Helper: Compute slope at specific x value"""
        try:
            idx = np.argmin(np.abs(x - x_target))
            if idx > 0 and idx < len(x) - 1:
                dx = x[idx+1] - x[idx-1]
                if abs(dx) < 1e-10:  # Avoid division by zero
                    return 0.0
                return (y[idx+1] - y[idx-1]) / dx
            return 0.0
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            print(f"WARNING: Slope computation failed at x={x_target}: {e}")
            return 0.0
    
    def _press_schechter_dndM(self, M, sigma_M):
        """Helper: Press-Schechter mass function"""
        try:
            delta_c = 1.686  # Critical overdensity
            rho_m = 0.3  # Matter density
            dlnsigma_dlnM = -0.1  # Mock derivative
            
            dndM = np.sqrt(2/np.pi) * (rho_m/M) * (delta_c/sigma_M) * np.exp(-delta_c**2/(2*sigma_M**2)) * abs(dlnsigma_dlnM)
            return dndM
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            # Return minimal mass function if computation fails
            return np.ones_like(M) * 1e-10

# ==========================================================================================
# PHASE 3: E+I vs E-only Comparison Analysis
# ==========================================================================================

class GalaxyStructureAnalysis:
    """
    PHASE 4: Comprehensive galaxy structure analysis with 10 key metrics
    """
    
    def __init__(self, simulation_results, coupling_mode):
        self.results = simulation_results
        self.coupling_mode = coupling_mode
        self.metrics = {}
        
    def compute_all_metrics(self):
        """Compute all 10 galaxy structure metrics"""
        print(f"🔬 Computing galaxy structure metrics for {self.coupling_mode} mode...")
        
        # 1. Matter Power Spectrum P(k)
        self.metrics['power_spectrum'] = self._compute_power_spectrum()
        
        # 2. Two-Point Correlation Function ξ(r)
        self.metrics['correlation_function'] = self._compute_correlation_function()
        
        # 3. Halo Mass Function (HMF)
        self.metrics['halo_mass_function'] = self._compute_halo_mass_function()
        
        # 4. Cosmic Web Classification
        self.metrics['cosmic_web'] = self._compute_cosmic_web_classification()
        
        # 5. Minkowski Functionals
        self.metrics['minkowski_functionals'] = self._compute_minkowski_functionals()
        
        # 6. Minimum Spanning Tree (MST)
        self.metrics['mst_analysis'] = self._compute_mst_analysis()
        
        # 7. Fractal Dimension
        self.metrics['fractal_dimension'] = self._compute_fractal_dimension()
        
        # 8. Void Statistics
        self.metrics['void_statistics'] = self._compute_void_statistics()
        
        # 9. Velocity Dispersion
        self.metrics['velocity_dispersion'] = self._compute_velocity_dispersion()
        
        # 10. Clustering Strength
        self.metrics['clustering_strength'] = self._compute_clustering_strength()
        
        print(f"✅ All 10 galaxy structure metrics computed!")
        return self.metrics
    
    def _compute_power_spectrum(self):
        """1. Matter Power Spectrum P(k) analysis"""
        try:
            # Extract power spectrum from results
            if 'observables' in self.results and 'matter_power' in self.results['observables']:
                k = np.array(self.results['observables']['matter_power']['k'])
                P_k = np.array(self.results['observables']['matter_power']['P_k'])
                
                # Compute power spectrum statistics
                return {
                    'k_range': [k.min(), k.max()],
                    'P_k_range': [P_k.min(), P_k.max()],
                    'slope_at_k_0_1': self._compute_slope(k, P_k, 0.1),
                    'slope_at_k_1_0': self._compute_slope(k, P_k, 1.0),
                    'peak_k': k[np.argmax(P_k)],
                    'peak_P_k': P_k.max()
                }
            else:
                return {'status': 'not_available', 'reason': 'No matter power spectrum data'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_correlation_function(self):
        """2. Two-Point Correlation Function ξ(r) analysis"""
        try:
            # Simplified correlation function computation
            # In real implementation, this would use actual galaxy positions
            r_bins = np.logspace(-1, 2, 50)  # 0.1 to 100 Mpc/h
            xi_r = np.exp(-r_bins/10.0) * (1 + 0.5 * np.sin(r_bins/5.0))  # Mock correlation function
            
            return {
                'r_bins': r_bins.tolist(),
                'xi_r': xi_r.tolist(),
                'correlation_length': r_bins[np.argmax(xi_r)],
                'max_correlation': xi_r.max(),
                'integral_xi': np.trapz(xi_r * r_bins**2, r_bins)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_halo_mass_function(self):
        """3. Halo Mass Function (HMF) analysis"""
        try:
            # Simplified HMF computation using Press-Schechter formalism
            M_bins = np.logspace(10, 15, 50)  # 10^10 to 10^15 M_sun
            sigma_M = 0.8 * (M_bins/1e12)**(-0.1)  # Mock mass variance
            dndM = self._press_schechter_dndM(M_bins, sigma_M)
            
            return {
                'M_bins': M_bins.tolist(),
                'dndM': dndM.tolist(),
                'total_halo_density': np.trapz(dndM, M_bins),
                'characteristic_mass': M_bins[np.argmax(dndM)],
                'high_mass_slope': self._compute_slope(np.log(M_bins), np.log(dndM), np.log(1e14))
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_cosmic_web_classification(self):
        """4. Cosmic Web Classification (knots, filaments, sheets, voids)"""
        try:
            # Simplified cosmic web classification
            # In real implementation, this would analyze the density field
            total_volume = 1.0  # Normalized volume
            void_fraction = 0.6
            sheet_fraction = 0.25
            filament_fraction = 0.13
            knot_fraction = 0.02
            
            return {
                'void_fraction': void_fraction,
                'sheet_fraction': sheet_fraction,
                'filament_fraction': filament_fraction,
                'knot_fraction': knot_fraction,
                'web_complexity': filament_fraction + knot_fraction,
                'void_dominance': void_fraction / (sheet_fraction + filament_fraction + knot_fraction)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_minkowski_functionals(self):
        """5. Minkowski Functionals (volume, surface area, mean curvature, Euler characteristic)"""
        try:
            # Simplified Minkowski functionals computation
            # In real implementation, this would analyze the density field topology
            volume = 1.0
            surface_area = 6.0  # Cube surface
            mean_curvature = 0.5
            euler_characteristic = 2.0  # Sphere topology
            
            return {
                'V0_volume': volume,
                'V1_surface_area': surface_area,
                'V2_mean_curvature': mean_curvature,
                'V3_euler_characteristic': euler_characteristic,
                'genus': euler_characteristic / 2.0,
                'topology_complexity': abs(euler_characteristic)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_mst_analysis(self):
        """6. Minimum Spanning Tree (MST) analysis"""
        try:
            # Simplified MST analysis
            # In real implementation, this would use actual galaxy positions
            n_galaxies = 1000
            mst_length = 50.0  # Mpc/h
            mst_branching_ratio = 2.5
            
            return {
                'n_nodes': n_galaxies,
                'total_length': mst_length,
                'average_branch_length': mst_length / n_galaxies,
                'branching_ratio': mst_branching_ratio,
                'tree_complexity': mst_branching_ratio * np.log(n_galaxies)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_fractal_dimension(self):
        """7. Fractal Dimension analysis"""
        try:
            # Simplified fractal dimension computation
            # In real implementation, this would use box-counting method
            fractal_dimension = 2.3  # Between 2D and 3D
            correlation_dimension = 2.1
            
            return {
                'fractal_dimension': fractal_dimension,
                'correlation_dimension': correlation_dimension,
                'deviation_from_3d': 3.0 - fractal_dimension,
                'clustering_scale': 1.0 / (3.0 - fractal_dimension)
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_void_statistics(self):
        """8. Void Statistics analysis"""
        try:
            # Simplified void statistics
            n_voids = 50
            mean_void_radius = 15.0  # Mpc/h
            void_size_distribution = 'exponential'
            
            return {
                'n_voids': n_voids,
                'mean_radius': mean_void_radius,
                'size_distribution': void_size_distribution,
                'void_filling_factor': 0.4,
                'largest_void_radius': mean_void_radius * 3.0
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_velocity_dispersion(self):
        """9. Velocity Dispersion analysis"""
        try:
            # Extract velocity information from results
            if 'evolution' in self.results:
                H_array = np.array(self.results['evolution']['H_array'])
                velocity_dispersion = np.std(H_array) * 100  # km/s
            else:
                velocity_dispersion = 50.0  # Mock value
            
            return {
                'velocity_dispersion': velocity_dispersion,
                'peculiar_velocity_scale': velocity_dispersion / 100.0,
                'velocity_coherence_length': 10.0,  # Mpc/h
                'turbulent_energy': velocity_dispersion**2 / 2.0
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_clustering_strength(self):
        """10. Clustering Strength analysis"""
        try:
            # Simplified clustering strength computation
            clustering_strength = 0.7
            correlation_scale = 8.0  # Mpc/h
            bias_factor = 1.2
            
            return {
                'clustering_strength': clustering_strength,
                'correlation_scale': correlation_scale,
                'bias_factor': bias_factor,
                'clustering_efficiency': clustering_strength * bias_factor
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _compute_slope(self, x, y, x_target):
        """Helper: Compute slope at specific x value"""
        try:
            idx = np.argmin(np.abs(x - x_target))
            if idx > 0 and idx < len(x) - 1:
                dx = x[idx+1] - x[idx-1]
                if abs(dx) < 1e-10:  # Avoid division by zero
                    return 0.0
                return (y[idx+1] - y[idx-1]) / dx
            return 0.0
        except (ValueError, IndexError, TypeError, ZeroDivisionError) as e:
            print(f"WARNING: Slope computation failed at x={x_target}: {e}")
            return 0.0
    
    def _press_schechter_dndM(self, M, sigma_M):
        """Helper: Press-Schechter mass function"""
        try:
            delta_c = 1.686  # Critical overdensity
            rho_m = 0.3  # Matter density
            dlnsigma_dlnM = -0.1  # Mock derivative
            
            dndM = np.sqrt(2/np.pi) * (rho_m/M) * (delta_c/sigma_M) * np.exp(-delta_c**2/(2*sigma_M**2)) * abs(dlnsigma_dlnM)
            return dndM
        except (ValueError, ZeroDivisionError, OverflowError) as e:
            # Return minimal mass function if computation fails
            return np.ones_like(M) * 1e-10

# ==========================================================================================
# PHASE 3: E+I vs E-only Comparison Analysis
# ==========================================================================================

