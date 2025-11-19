# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# tqe_core.py - TQE Core Components Module
# ==========================================================================================
# TQE–ΛSim: Core TQE components - EnergyInformationContent and CouplingModel
# Based on the Theory of the Question of Existence (TQE)
# ==========================================================================================

import numpy as np
from .config import MASTER_CTRL, FIDUCIAL_PARAMS

# ==========================================================================================
# I-PARAMETER MODELS
# ==========================================================================================

class EnergyInformationContent:
    # Energy Information Content Calculator (TQE-compliant)
    # Computes I parameter as INTRINSIC property of energy evolution
    # I = |dE/da| / (E + |dE/da|) - normalized asymmetry in energy system
    
    def __init__(self, model_type='phenomenological', params=None):
        # Initialize I-parameter model
        # model_type: 'phenomenological', 'eft_lagrangian', or 'energy_based' (TQE-compliant)
        # params: dictionary of model parameters
        
        self.model_type = model_type
        self.params = params if params is not None else {}
        
        # Default parameters for phenomenological model (LEGACY)
        if model_type == 'phenomenological':
            self.A = self.params.get('A', 0.1)      # Amplitude
            self.gamma = self.params.get('gamma', 0.5)  # Power law index
        
        # Default parameters for EFT Lagrangian model
        elif model_type == 'eft_lagrangian':
            self.c1 = self.params.get('c1', 1.0)    # I² coefficient
            self.c2 = self.params.get('c2', 0.1)    # (∂I)² coefficient
            self.gamma = self.params.get('gamma', 0.5)  # Power law index for evolution
        
        # TQE-COMPLIANT: energy_based (I derived from E evolution)
        elif model_type == 'energy_based':
            self.epsilon = self.params.get('epsilon', 1e-6)  # Regularization
            self.normalization = self.params.get('normalization', 'tanh')  # 'tanh' or 'rational'
        
        # Quiet mode: only print if verbose
        if MASTER_CTRL.get("VERBOSE", True):
            print(f"✓ I-parameter model initialized: {model_type}")
    
    def compute_information(self, a, E=None, dE_da=None):
        """
        Compute I-parameter value at scale factor a
        
        TQE-COMPLIANT DEFINITION:
        I = information content of the ENERGY system (not independent field!)
        I measures the asymmetry/change rate in energy evolution
        
        Args:
            a: scale factor (a=1 today, a→0 early universe)
            E: normalized energy E = H(a)/H0 (optional, for energy_based mode)
            dE_da: energy derivative dE/da (optional, for energy_based mode)
        """
        
        if self.model_type == 'phenomenological':
            # LEGACY: Phenomenological curve I(a) = A·a^γ
            return self.A * a**self.gamma
        
        elif self.model_type == 'eft_lagrangian':
            # EFT-based evolution (simplified)
            return self.c1 * a**self.gamma
        
        elif self.model_type == 'energy_based':
            # TQE-COMPLIANT: I derived from energy evolution
            # I = information content of E (normalized asymmetry/change rate)
            
            # If E not provided, approximate from a
            if E is None:
                # Approximate: E ≈ sqrt(Ω_m/a³ + Ω_Λ) (ΛCDM-like)
                Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                E = np.sqrt(Omega_m / a**3 + Omega_Lambda)
            
            if dE_da is None:
                # Numerical derivative
                da = 0.001
                a_plus = np.minimum(a + da, 1.0)
                a_minus = np.maximum(a - da, MASTER_CTRL['A_MIN'])
                
                # Approximate dE/da
                Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                E_plus = np.sqrt(Omega_m / a_plus**3 + Omega_Lambda)
                E_minus = np.sqrt(Omega_m / a_minus**3 + Omega_Lambda)
                dE_da = (E_plus - E_minus) / (a_plus - a_minus)
            
            # I = normalized change rate (TQE definition)
            # Measures how fast the energy system is changing (asymmetry)
            abs_dE_da = np.abs(dE_da)
            
            if self.normalization == 'tanh':
                # Hyperbolic tangent normalization (smooth, bounded [0,1])
                I = np.tanh(abs_dE_da / (E + self.epsilon))
            else:
                # Rational normalization (TQE original)
                I = abs_dE_da / (E + abs_dE_da + self.epsilon)
            
            # Ensure [0, 1] bounds
            I = np.clip(I, 0.0, 1.0)
            
            return I
        
        else:
            return 0.0
    
    def compute_information_derivative(self, a, E=None, dE_da=None, d2E_da2=None):
        # Compute time derivative of I-parameter: dI/da
        
        if self.model_type == 'phenomenological':
            # d/da [A·a^γ] = A·γ·a^(γ-1)
            return self.A * self.gamma * a**(self.gamma - 1)
        
        elif self.model_type == 'eft_lagrangian':
            return self.c1 * self.gamma * a**(self.gamma - 1)
        
        elif self.model_type == 'energy_based':
            # Numerical derivative of I(a, E, dE/da)
            da = 0.001
            a_plus = min(a + da, 1.0)
            a_minus = max(a - da, MASTER_CTRL['A_MIN'])
            
            I_plus = self.compute_information(a_plus, E, dE_da)
            I_minus = self.compute_information(a_minus, E, dE_da)
            
            dI_da = (I_plus - I_minus) / (a_plus - a_minus)
            return dI_da
        
        else:
            return 0.0
    
    def compute_information_gradient_squared(self, a):
        # Compute effective smoothness penalty: |∇I|²
        # NOTE: This is NOT a true spatial gradient |∇I|²
        # This is an effective smoothness penalty based on temporal variation
        # Used as a proxy for field smoothness in the coupling models
        # 
        # Simplified model: assume slow temporal variation with safe boundary handling
        
        # Ensure a is within valid range
        a_safe = np.clip(a, MASTER_CTRL['A_MIN'], 1.0)
        da = 0.001
        
        # Safe derivative calculation with boundary protection
        a_plus = np.minimum(a_safe + da, 1.0)
        a_minus = np.maximum(a_safe - da, MASTER_CTRL['A_MIN'])
        
        dI_da = (self.compute_information(a_plus) - self.compute_information(a_minus)) / (a_plus - a_minus)
        return dI_da**2
    
    def compute_information_time_derivative_squared(self, a, H):
        # Compute time derivative squared: (∂_t I)²
        # ∂_t I = (dI/da) · (da/dt) = (dI/da) · a · H(a)
        dI_da = self.compute_information_derivative(a)
        dt_I = dI_da * a * H
        return dt_I**2
    
    def compute_information_dynamic(self, a, E_field=None, dE_da=None):
        """
        TQE Information Parameter with Dynamic E-Field Feedback
        
        TQE THEORETICAL FOUNDATION:
        ─────────────────────────────
        The I-parameter represents energy's intrinsic information orientation—its
        internal tendency toward complexity and structure formation. This is NOT
        an external field, but an intrinsic property of the energy state itself.
        
        IDEAL DEFINITION (from TQE theory):
            I = D_KL(P_t || P_{t+1}) / (1 + D_KL(P_t || P_{t+1}))
            
            where:
            - P_t: Quantum probability distribution at epoch t
            - D_KL: Kullback-Leibler divergence (measures directional bias)
            - 0 ≤ I ≤ 1 (normalized information content)
        
        COSMOLOGICAL APPROXIMATION (implemented here):
        ───────────────────────────────────────────────
        Since we don't have access to the full quantum probability distribution
        P(ψ) in a cosmological simulation, we compute I from energy evolution:
        
            E(a) = H(a) / H0  (normalized expansion rate)
            I(a) = |dE/da| / (E + |dE/da|)  (normalized change rate)
        
        This is TQE-COMPLIANT: I is the INTRINSIC information content of the
        energy system, measured by its temporal asymmetry (change rate).
        
        For energy_based model: I_field(a, E, dE_da) directly computes this.
        For legacy models: we add E-feedback correction to phenomenological base.
        
        PHYSICAL INTERPRETATION:
        ────────────────────────
        I measures how rapidly the energy system is evolving. High |dE/da| → high I
        (system is far from equilibrium, high information asymmetry).
        Low |dE/da| → low I (near equilibrium, low asymmetry).
        
        Args:
            a: scale factor (a=1 today, a→0 early universe)
            E_field: normalized expansion rate E(a) = H(a)/H₀ (energy proxy)
            dE_da: derivative of E with respect to a (evolutionary rate)
        
        Returns:
            I_dynamic: information field value with E-feedback (0 ≤ I ≤ 1)
        """
        
        # For energy_based model, delegate to I_field (already E-dependent)
        if self.model_type == 'energy_based':
            return self.compute_information(a, E=E_field, dE_da=dE_da)
        
        # For phenomenological/EFT models: base + E-feedback correction
        I_base = self.compute_information(a)
        
        # If no E-field provided, return static I-parameter
        if E_field is None:
            return I_base
        
        # Coupling parameters (adjustable via MASTER_CTRL)
        coupling_strength = MASTER_CTRL.get('I_E_COUPLING_STRENGTH', 0.1)
        damping_tau = MASTER_CTRL.get('I_DAMPING_TAU', 0.5)
        E_direct_coupling = MASTER_CTRL.get('I_E_DIRECT_COUPLING', 0.05)
        
        # Initialize dynamic corrections
        E_feedback = 0.0
        E_coupling_term = 0.0
        
        # E-gradient feedback: I reacts to local E-field changes
        if dE_da is not None:
            # Convert to array if scalar
            a_arr = np.atleast_1d(a)
            dE_da_arr = np.atleast_1d(dE_da)
            
            # Local response to E-gradient (dimensionless)
            E_feedback = coupling_strength * dE_da_arr * a_arr
        
        # E-field direct coupling with exponential damping
        # Suppressed in early universe (a << 1), active late-time (a → 1)
        a_arr = np.atleast_1d(a)
        E_arr = np.atleast_1d(E_field)
        damping = np.exp(-a_arr / damping_tau)
        E_coupling_term = E_direct_coupling * E_arr * damping
        
        # Total dynamic I-parameter
        I_dynamic = I_base + E_feedback + E_coupling_term
        
        # Physical constraint: I ∈ [0, 1]
        I_dynamic = np.clip(I_dynamic, 0.0, 1.0)
        
        # Return scalar if input was scalar
        if np.isscalar(a):
            return float(I_dynamic)
        else:
            return I_dynamic
    
    def E_field_DEPRECATED(self, a, H, weights=None):
        # DEPRECATED: This was gradient-based E-field (WRONG definition!)
        # E = w_g·|∇I|² + w_t·(∂_t I)²  ❌ This is NOT energy!
        # 
        # CORRECT TQE DEFINITION:
        # E = H(a) / H0  (normalized expansion rate)
        # 
        # This function is kept for backward compatibility but should NOT be used.
        # Use: E = H / H0 directly instead!
        
        if weights is None:
            weights = MASTER_CTRL['E_WEIGHTS']
        
        w_g = weights.get('w_g', 1.0)
        w_t = weights.get('w_t', 1.0)
        
        # Spatial component
        grad_I_sq = self.compute_information_gradient_squared(a)
        
        # Temporal component
        time_deriv_sq = self.compute_information_time_derivative_squared(a, H)
        
        # Total E-field (DEPRECATED)
        E = w_g * grad_I_sq + w_t * time_deriv_sq
        
        return E
    
    def compute_I_mean(self, a_grid):
        # Compute mean I-parameter value: ⟨I⟩
        # Used for demeaned coupling
        
        I_values = np.array([self.compute_information(a) for a in a_grid])
        I_mean = np.mean(I_values)
        
        return I_mean
    
    # ==========================================================================================
    # MULTI-DEFINITION I-PARAMETER METHODS
    # ==========================================================================================
    
    def I_from_KL_divergence(self, P_t, P_t_plus_1):
        """
        Compute I from KL-divergence (Primary definition from theory)
        
        I_KL = D_KL(P_t || P_{t+1}) / (1 + D_KL(P_t || P_{t+1}))
        
        Args:
            P_t: Probability distribution at epoch t
            P_t_plus_1: Probability distribution at epoch t+1
        
        Returns:
            I_KL: Information parameter from KL-divergence (0 ≤ I_KL ≤ 1)
        """
        # Ensure probabilities are normalized
        P_t = np.asarray(P_t)
        P_t_plus_1 = np.asarray(P_t_plus_1)
        
        # Avoid log(0) and division by zero - clip BEFORE normalization
        epsilon = 1e-12
        P_t = np.clip(P_t, epsilon, None)
        P_t_plus_1 = np.clip(P_t_plus_1, epsilon, None)
        
        # Normalize (now safe, sums are guaranteed > epsilon)
        P_t = P_t / np.sum(P_t)
        P_t_plus_1 = P_t_plus_1 / np.sum(P_t_plus_1)
        
        # Compute KL-divergence: D_KL(P||Q) = Σ P_i log(P_i/Q_i)
        D_KL = np.sum(P_t * np.log(P_t / P_t_plus_1))
        
        # Normalize to [0, 1]
        I_KL = D_KL / (1.0 + D_KL)
        
        return np.clip(I_KL, 0.0, 1.0)
    
    def I_from_Shannon_entropy(self, P_t):
        """
        Compute I from Shannon entropy
        
        I_Shannon = H(P_t) / H_max
        where H(P) = -Σ P_i log(P_i)
        
        Args:
            P_t: Probability distribution at epoch t
        
        Returns:
            I_Shannon: Information parameter from entropy (0 ≤ I_Shannon ≤ 1)
        """
        P_t = np.asarray(P_t)
        P_t = P_t / np.sum(P_t)  # Normalize
        
        epsilon = 1e-10
        P_t = np.clip(P_t, epsilon, 1.0)
        
        # Shannon entropy
        H = -np.sum(P_t * np.log(P_t))
        
        # Maximum entropy (uniform distribution)
        N = len(P_t)
        H_max = np.log(N)
        
        # Normalized information
        I_Shannon = H / H_max if H_max > 0 else 0.0
        
        return np.clip(I_Shannon, 0.0, 1.0)
    
    def I_from_Renyi_entropy(self, P_t, alpha=2.0):
        """
        Compute I from Rényi entropy (generalized entropy family)
        
        H_α(P) = (1/(1-α)) log(Σ P_i^α)
        I_Renyi = H_α / H_α_max
        
        Special cases:
        - α → 1: Shannon entropy
        - α = 2: Collision entropy
        - α → ∞: Min-entropy
        
        Args:
            P_t: Probability distribution at epoch t
            alpha: Rényi parameter (default: 2.0 for collision entropy)
        
        Returns:
            I_Renyi: Information parameter from Rényi entropy (0 ≤ I_Renyi ≤ 1)
        """
        P_t = np.asarray(P_t)
        P_t = P_t / np.sum(P_t)  # Normalize
        
        epsilon = 1e-10
        P_t = np.clip(P_t, epsilon, 1.0)
        
        N = len(P_t)
        
        if np.abs(alpha - 1.0) < 1e-6:
            # Limit case: Shannon entropy
            return self.I_from_Shannon_entropy(P_t)
        
        # Rényi entropy
        H_alpha = (1.0 / (1.0 - alpha)) * np.log(np.sum(P_t**alpha))
        
        # Maximum Rényi entropy (uniform distribution)
        P_uniform = np.ones(N) / N
        H_alpha_max = (1.0 / (1.0 - alpha)) * np.log(np.sum(P_uniform**alpha))
        
        # Normalized information
        I_Renyi = H_alpha / H_alpha_max if H_alpha_max != 0 else 0.0
        
        return np.clip(I_Renyi, 0.0, 1.0)
    
    def I_from_mutual_information(self, P_t, P_t_plus_1):
        """
        Compute I from mutual information between successive epochs
        
        MI(X,Y) = H(X) + H(Y) - H(X,Y)
        I_MI = MI / MI_max
        
        Args:
            P_t: Probability distribution at epoch t
            P_t_plus_1: Probability distribution at epoch t+1
        
        Returns:
            I_MI: Information parameter from mutual information (0 ≤ I_MI ≤ 1)
        """
        P_t = np.asarray(P_t)
        P_t_plus_1 = np.asarray(P_t_plus_1)
        
        P_t = P_t / np.sum(P_t)
        P_t_plus_1 = P_t_plus_1 / np.sum(P_t_plus_1)
        
        epsilon = 1e-10
        P_t = np.clip(P_t, epsilon, 1.0)
        P_t_plus_1 = np.clip(P_t_plus_1, epsilon, 1.0)
        
        # Individual entropies
        H_t = -np.sum(P_t * np.log(P_t))
        H_t_plus_1 = -np.sum(P_t_plus_1 * np.log(P_t_plus_1))
        
        # Joint entropy (approximation: assume independence)
        # For correlated distributions, would need full joint P(X,Y)
        # Here we use outer product as approximation
        P_joint = np.outer(P_t, P_t_plus_1).flatten()
        P_joint = P_joint / np.sum(P_joint)
        P_joint = np.clip(P_joint, epsilon, 1.0)
        H_joint = -np.sum(P_joint * np.log(P_joint))
        
        # Mutual information
        MI = H_t + H_t_plus_1 - H_joint
        
        # Maximum mutual information (perfect correlation)
        MI_max = min(H_t, H_t_plus_1)
        
        # Normalized information
        I_MI = MI / MI_max if MI_max > 0 else 0.0
        
        return np.clip(I_MI, 0.0, 1.0)
    
    def I_composite_fusion(self, I_KL, I_Shannon, method='product'):
        """
        Combine multiple I-definitions using fusion methods
        
        Args:
            I_KL: I from KL-divergence
            I_Shannon: I from Shannon entropy
            method: 'product', 'average', 'max', 'min'
        
        Returns:
            I_composite: Fused information parameter (0 ≤ I_composite ≤ 1)
        """
        if method == 'product':
            # Product fusion (amplifies agreement, suppresses disagreement)
            return I_KL * I_Shannon
        
        elif method == 'average':
            # Arithmetic mean
            return (I_KL + I_Shannon) / 2.0
        
        elif method == 'max':
            # Maximum (optimistic)
            return max(I_KL, I_Shannon)
        
        elif method == 'min':
            # Minimum (conservative)
            return min(I_KL, I_Shannon)
        
        else:
            # Default: product
            return I_KL * I_Shannon
    
    def compute_all_I_definitions(self, a, friedmann=None):
        """
        Compute I using all 5 definitions for comparison
        
        Args:
            a: scale factor
            friedmann: FriedmannEvolution instance (for E-field)
        
        Returns:
            dict: All I-definitions
                {
                    'phenomenological': I(a) = A·a^γ,
                    'kl_divergence': I from KL-divergence,
                    'shannon': I from Shannon entropy,
                    'renyi': I from Rényi entropy,
                    'mutual_info': I from mutual information,
                    'composite': I_KL × I_Shannon
                }
        """
        results = {}
        
        # 1. Phenomenological (baseline)
        results['phenomenological'] = self.compute_information(a)
        
        # For information-theoretic measures, we need probability distributions
        # Approximate using density field evolution (simplified for cosmology)
        # P_t ∝ exp(-E(a)²/2σ²) with E(a) = H(a)/H₀
        
        if friedmann is not None:
            # Create synthetic probability distribution from energy landscape
            a_grid = np.linspace(max(0.01, a-0.1), min(1.0, a+0.1), 50)
            E_vals = np.array([friedmann.E(a_i) for a_i in a_grid])
            
            # P ∝ exp(-E²) (Boltzmann-like)
            sigma = 1.0
            P_t = np.exp(-E_vals**2 / (2*sigma**2))
            P_t = P_t / np.sum(P_t)  # Normalize
            
            # Next epoch (shifted)
            a_grid_next = a_grid + 0.01
            a_grid_next = np.clip(a_grid_next, 0.01, 1.0)
            E_vals_next = np.array([friedmann.E(a_i) for a_i in a_grid_next])
            P_t_plus_1 = np.exp(-E_vals_next**2 / (2*sigma**2))
            P_t_plus_1 = P_t_plus_1 / np.sum(P_t_plus_1)
            
            # 2. KL-divergence based
            results['kl_divergence'] = self.I_from_KL_divergence(P_t, P_t_plus_1)
            
            # 3. Shannon entropy based
            results['shannon'] = self.I_from_Shannon_entropy(P_t)
            
            # 4. Rényi entropy based
            alpha_renyi = MASTER_CTRL.get('RENYI_ALPHA', 2.0)
            results['renyi'] = self.I_from_Renyi_entropy(P_t, alpha=alpha_renyi)
            
            # 5. Mutual information based
            results['mutual_info'] = self.I_from_mutual_information(P_t, P_t_plus_1)
            
            # 6. Composite fusion
            fusion_method = MASTER_CTRL.get('I_FUSION_METHOD', 'product')
            results['composite'] = self.I_composite_fusion(
                results['kl_divergence'], 
                results['shannon'], 
                method=fusion_method
            )
            
            # 7. KL-Shannon refined (harmonic mean)
            results['kl_shannon'] = self.I_from_KL_Shannon_refined(
                results['kl_divergence'],
                results['shannon']
            )
            
            # 8. Entanglement entropy
            results['entanglement'] = self.I_from_entanglement_entropy(P_t)
            
            # 9. Fisher information
            results['fisher'] = self.I_from_Fisher_information(P_t)
            
            # 10. Horizon entropy (cosmological)
            results['horizon_entropy'] = self.I_from_horizon_entropy(a, friedmann)
        else:
            # Fallback: use phenomenological for all
            results['kl_divergence'] = results['phenomenological']
            results['shannon'] = results['phenomenological']
            results['renyi'] = results['phenomenological']
            results['mutual_info'] = results['phenomenological']
            results['composite'] = results['phenomenological']
            results['kl_shannon'] = results['phenomenological']
            results['entanglement'] = results['phenomenological']
            results['fisher'] = results['phenomenological']
            results['horizon_entropy'] = results['phenomenological']
        
        return results
    
    def I_from_KL_Shannon_refined(self, I_KL, I_Shannon):
        """
        Compute I from KL-Shannon refined fusion (harmonic mean)
        
        I_KLS = 2·I_KL·I_Shannon / (I_KL + I_Shannon)
        
        This is the HARMONIC MEAN of I_KL and I_Shannon, which:
        - Avoids zero-product issues (if either is 0, result is 0)
        - Gives more weight to the smaller value
        - More conservative than arithmetic or geometric mean
        
        Args:
            I_KL: I from KL-divergence
            I_Shannon: I from Shannon entropy
        
        Returns:
            I_KLS: Harmonic mean of I_KL and I_Shannon (0 ≤ I_KLS ≤ 1)
        """
        if I_KL + I_Shannon == 0:
            return 0.0
        
        I_KLS = 2.0 * I_KL * I_Shannon / (I_KL + I_Shannon)
        return np.clip(I_KLS, 0.0, 1.0)
    
    def I_from_entanglement_entropy(self, P_t):
        """
        Compute I from entanglement entropy (quantum correlations)
        
        I_ent = S_ent / S_ent_max
        where S_ent = -Tr(ρ_A log ρ_A) is von Neumann entropy
        
        PHYSICAL MEANING:
        - Measures quantum correlations between subsystems
        - S_ent = 0: Pure state (no entanglement)
        - S_ent = max: Maximally entangled state
        
        IMPLEMENTATION:
        - Approximate entanglement via Schmidt decomposition
        - Partition system into two subsystems A and B
        - Compute reduced density matrix ρ_A = Tr_B(ρ)
        - S_ent = -Tr(ρ_A log ρ_A)
        
        Args:
            P_t: Probability distribution (treated as quantum state amplitudes)
        
        Returns:
            I_ent: Entanglement entropy based information (0 ≤ I_ent ≤ 1)
        """
        P_t = np.asarray(P_t)
        P_t = P_t / np.sum(P_t)  # Normalize
        
        epsilon = 1e-10
        P_t = np.clip(P_t, epsilon, 1.0)
        
        N = len(P_t)
        
        # Schmidt decomposition approximation
        # Partition into two subsystems of equal size
        schmidt_rank = min(MASTER_CTRL.get('ENTANGLEMENT_SCHMIDT_RANK', 10), N // 2)
        
        if schmidt_rank == 0:
            return 0.0
        
        # Reshape into matrix (bipartite system)
        # For simplicity, use SVD to approximate entanglement
        try:
            # Treat P_t as amplitudes, compute density matrix
            psi = np.sqrt(P_t)
            
            # Reshape to matrix (if not square, pad)
            dim = int(np.ceil(np.sqrt(N)))
            psi_padded = np.zeros(dim * dim)
            psi_padded[:N] = psi
            psi_matrix = psi_padded.reshape(dim, dim)
            
            # SVD: singular values are Schmidt coefficients
            U, s, Vh = np.linalg.svd(psi_matrix, full_matrices=False)
            
            # Schmidt coefficients (squared for probabilities)
            lambda_i = s[:schmidt_rank]**2
            lambda_i = lambda_i / np.sum(lambda_i)  # Normalize
            lambda_i = np.clip(lambda_i, epsilon, 1.0)
            
            # von Neumann entropy: S_ent = -Σ λ_i log(λ_i)
            S_ent = -np.sum(lambda_i * np.log(lambda_i))
            
            # Maximum entanglement entropy (uniform distribution over schmidt_rank)
            S_ent_max = np.log(schmidt_rank)
            
            # Normalized information
            I_ent = S_ent / S_ent_max if S_ent_max > 0 else 0.0
            
        except (ValueError, np.linalg.LinAlgError, AttributeError) as e:
            # Fallback: use Shannon entropy as approximation
            print(f"⚠ Entanglement entropy failed, using Shannon fallback: {e}")
            H = -np.sum(P_t * np.log(P_t))
            H_max = np.log(N)
            I_ent = H / H_max if H_max > 0 else 0.0
        
        return np.clip(I_ent, 0.0, 1.0)
    
    def I_from_Fisher_information(self, P_t, theta=None):
        """
        Compute I from Fisher information (information geometry)
        
        I_Fisher = F / F_max
        where F = ∫ P(x|θ) [∂log P/∂θ]² dx
        
        PHYSICAL MEANING:
        - Measures precision of parameter estimation
        - High F → sharp distribution, easy to estimate θ
        - Low F → broad distribution, hard to estimate θ
        
        Fisher information is the "curvature" of the information manifold:
        - Related to Cramér-Rao bound (minimum variance of estimators)
        - Fundamental in quantum metrology and parameter estimation
        
        IMPLEMENTATION:
        - Finite difference approximation: ∂P/∂θ ≈ [P(θ+ε) - P(θ-ε)]/(2ε)
        - Assume θ parameterizes the distribution evolution
        
        Args:
            P_t: Probability distribution
            theta: Parameter value (if None, use distribution index as proxy)
        
        Returns:
            I_Fisher: Fisher information based information (0 ≤ I_Fisher ≤ 1)
        """
        P_t = np.asarray(P_t)
        P_t = P_t / np.sum(P_t)  # Normalize
        
        epsilon_fisher = MASTER_CTRL.get('FISHER_EPSILON', 1e-4)
        eps = 1e-10
        P_t = np.clip(P_t, eps, 1.0)
        
        # Approximate ∂log P/∂θ via finite differences
        # For cosmological context, θ could be scale factor or time
        # Here we use spatial variation as proxy
        
        N = len(P_t)
        if N < 3:
            return 0.0
        
        # Finite difference: ∂P/∂x at each point
        grad_log_P = np.zeros(N)
        
        # Central differences (interior points)
        for i in range(1, N-1):
            dP = (P_t[i+1] - P_t[i-1]) / (2.0 * epsilon_fisher)
            grad_log_P[i] = dP / P_t[i] if P_t[i] > eps else 0.0
        
        # Forward/backward at boundaries
        grad_log_P[0] = (P_t[1] - P_t[0]) / epsilon_fisher / P_t[0] if P_t[0] > eps else 0.0
        grad_log_P[-1] = (P_t[-1] - P_t[-2]) / epsilon_fisher / P_t[-1] if P_t[-1] > eps else 0.0
        
        # Fisher information: F = ∫ P(x) [∂log P/∂θ]² dx
        # Discrete: F = Σ P_i (∂log P_i/∂θ)²
        F = np.sum(P_t * grad_log_P**2)
        
        # Normalize by theoretical maximum
        # For Gaussian: F_max ~ 1/σ² (inversely proportional to variance)
        # For uniform: F_max ~ 0 (no information)
        # Use variance-based normalization
        variance = np.sum(P_t * (np.arange(N) - np.sum(P_t * np.arange(N)))**2)
        F_max = 1.0 / max(variance, eps)
        
        # Normalized information
        I_Fisher = F / F_max if F_max > 0 else 0.0
        
        return np.clip(I_Fisher, 0.0, 1.0)
    
    def I_from_horizon_entropy(self, a, friedmann=None):
        """
        Compute I from cosmological horizon entropy (Bekenstein-Hawking)
        
        I_horizon = S_H / S_H_max
        where S_H = A_H / (4G) with A_H = 4π r_H²
        
        PHYSICAL MEANING:
        - Bekenstein-Hawking entropy of cosmological horizon
        - S_H ∝ horizon area (holographic principle)
        - Measures information content accessible to an observer
        
        COSMOLOGICAL CONTEXT:
        - Hubble horizon: r_H = c/H(a)
        - Horizon area: A_H = 4π (c/H)²
        - S_H = π c² / (G H²) in natural units
        
        HOLOGRAPHIC INTERPRETATION:
        - All information in a volume is encoded on its boundary
        - Horizon entropy = maximum information accessible
        - Related to dark energy and vacuum entropy
        
        Args:
            a: scale factor
            friedmann: FriedmannEvolution instance (for H(a))
        
        Returns:
            I_horizon: Horizon entropy based information (0 ≤ I_horizon ≤ 1)
        """
        if friedmann is None:
            # Fallback: use phenomenological I-parameter
            return self.compute_information(a)
        
        # Physical constants (in natural units c=ℏ=1)
        # G_newton in SI: 6.674e-11 m³/(kg·s²)
        # In cosmological units: use dimensionless ratios
        
        # Use ΛCDM approximation for E (avoid recursion in rho_DE)
        Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
        Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
        E_a = np.sqrt(Omega_m / a**3 + Omega_Lambda)
        E_1 = np.sqrt(Omega_m + Omega_Lambda)  # E at a=1
        
        # Normalized entropy (relative to today)
        # S_H(a) / S_H(a=1) = [H(1)/H(a)]² = [E(1)/E(a)]²
        S_H_ratio = (E_1 / E_a)**2 if E_a > 1e-10 else 0.0
        
        # Information parameter: normalize to [0,1]
        # Early universe (a→0): E→∞ → S_H_ratio→0 → I→0
        # Today (a=1): S_H_ratio=1 → I=1
        # Future (a>1): E<1 → S_H_ratio>1 → I clamped to 1
        
        I_horizon = S_H_ratio
        
        return np.clip(I_horizon, 0.0, 1.0)

# ==========================================================================================
# COUPLING MODELS
# ==========================================================================================

class CouplingModel:
    """
    TQE Coupling Models: Implementing P'(ψ) = P(ψ) · f(E,I)
    
    This class implements three rival coupling models that translate the TQE
    fine-tuning function f(E,I) into cosmological observables.
    
    TQE FUNDAMENTAL EQUATION:
        P'(ψ) = P(ψ) · f(E,I)
        
        where:
        - P(ψ): Baseline quantum probability distribution
        - E(a) = H(a)/H₀: Normalized expansion rate (vacuum energy proxy)
        - I(a): Information parameter (KL-divergence based, 0 ≤ I ≤ 1)
        - f(E,I): Fine-tuning function biasing towards stable universes
    
    FINE-TUNING FUNCTION (theoretical):
        f(E,I) = exp(-(E-E_c)²/(2σ²)) · (1 + α·I)
        
        Components:
        1. Goldilocks zone: exp(-(E-E_c)²/(2σ²)) → stability around E_c
        2. Information bias: (1 + α·I) → complexity preference
    
    COSMOLOGICAL IMPLEMENTATIONS:
        1. Covariant E-pressure: ρ_DE = ρ_Λ · exp(-α·E·(1-I))  [E+I]
                                 ρ_DE = ρ_Λ · exp(-α·E)        [E-only]
        
        2. Uniform w(I): w_DE(a) = w₀ + w_I·I(a), ρ_DE from integral evolution
        
        3. Geometric: ρ_DE = ρ_Λ · exp(β₀·F[I,∇I,∂I])
    
    These implementations capture the same underlying TQE mechanism but with
    different mathematical formulations for observational testing.
    """
    
    def __init__(self, coupling_type, information_content, coupling_params=None, coupling_mode=None):
        """
        Initialize coupling model
        
        Args:
            coupling_type: 'covariant_pressure', 'uniform_w', 'geometric', or 'null'
            information_content: EnergyInformationContent instance (provides I(a) evolution)
            coupling_params: dictionary of coupling parameters (α, w_I, β₀, etc.)
            coupling_mode: 'Eonly' or 'EplusI' (CRITICAL: determines if I-parameter affects ρ_DE)
        """
        
        self.coupling_type = coupling_type
        self.info_content = information_content
        self.params = coupling_params if coupling_params is not None else {}
        
        # TQE Coupling Mode (FIXED: now properly initialized from parameter)
        self.coupling_mode = coupling_mode  # 'Eonly' or 'EplusI'
        
        # Coupling parameters
        if coupling_type == 'covariant_pressure':
            self.alpha = self.params.get('alpha', 0.1)  # Coupling strength
            
        elif coupling_type == 'uniform_w':
            self.w0 = self.params.get('w0', -1.0)       # Base equation of state
            self.w_I = self.params.get('w_I', 0.1)      # I-coupling to w
            
            # Pre-compute integration grid for 10-50× performance boost
            print("  🔧 Building w(a) integration grid for optimization...")
            self._build_w_integration_grid()
            print("  ✓ Integration grid built - fast interpolation enabled")
            
        elif coupling_type == 'geometric':
            self.beta0 = self.params.get('beta0', 0.1)  # Geometric coupling strength
        
        # Advanced coupling parameters (from MASTER_CTRL)
        self.rho_DE_form = MASTER_CTRL.get('RHO_DE_FORM', 'linear')
        self.use_I_mean = MASTER_CTRL.get('USE_I_MEAN', False)
        self.use_exp_form = MASTER_CTRL.get('USE_EXP_FORM', False)
        
        # Store I_mean (will be computed during evolution)
        self.I_mean = 0.0
        
        # Initialize statistics (will be computed during evolution)
        self.I_mean = 0.0
        self.I_std = 1.0
        self.F_I_mean = 0.0
        self.statistics_computed = False
        
        print(f"✓ Coupling model initialized: {coupling_type}")
        print(f"  ρ_DE form: {self.rho_DE_form}, Use I_mean: {self.use_I_mean}, Use exp: {self.use_exp_form}")
    
    def compute_field_statistics(self, a_grid, friedmann):
        # Compute I-parameter and F_I statistics over evolution grid
        # This enables mean-centered, robust coupling
        
        print("📊 Computing I-parameter and F_I statistics...")
        
        # Compute I(a) on grid (TQE-COMPLIANT: from energy evolution)
        I_values = []
        for a in a_grid:
            if self.info_content.model_type == 'energy_based':
                # Use ΛCDM approximation for E (avoid recursion if called during initialization)
                Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                E = np.sqrt(Omega_m / a**3 + Omega_Lambda)
                dE_da = -1.5 * Omega_m / (a**4 * E)
                I_values.append(self.info_content.compute_information(a, E=E, dE_da=dE_da))
            else:
                I_values.append(self.info_content.compute_information(a))
        
        I_values = np.array(I_values)
        self.I_mean = np.mean(I_values)
        self.I_std = np.std(I_values)
        
        if self.I_std < 1e-10:
            self.I_std = 1.0  # Prevent division by zero
        
        # Compute F_I(a) on grid for mean calculation
        F_I_values = []
        for a in a_grid:
            if self.info_content.model_type == 'energy_based':
                # Use ΛCDM approximation for E (avoid recursion)
                Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                E = np.sqrt(Omega_m / a**3 + Omega_Lambda)
                dE_da = -1.5 * Omega_m / (a**4 * E)
                I_val = self.info_content.compute_information(a, E=E, dE_da=dE_da)
            else:
                I_val = self.info_content.compute_information(a)
            
            # Standardize
            I_std_val = (I_val - self.I_mean) / self.I_std
            I_sigmoid = 1.0 / (1.0 + np.exp(-I_std_val))
            
            # Gradient component
            I_grad2 = self.info_content.compute_information_gradient_squared(a)
            grad_sigmoid = 1.0 / (1.0 + np.exp(-np.sqrt(I_grad2) * 10))
            
            # F_I
            F_I = I_sigmoid**2 + grad_sigmoid**2
            F_I_norm = F_I / (1.0 + F_I)
            F_I_values.append(F_I_norm)
        
        self.F_I_mean = np.mean(F_I_values)
        self.statistics_computed = True
        
        print(f"  ✓ I statistics: ⟨I⟩ = {self.I_mean:.6f}, σ_I = {self.I_std:.6f}")
        print(f"  ✓ F_I statistics: ⟨F_I⟩ = {self.F_I_mean:.6f}")
        
        return {
            'I_mean': self.I_mean,
            'I_std': self.I_std,
            'F_I_mean': self.F_I_mean
        }
    
    def compute_G_field(self, a, H, H0):
        # Compute combined E-I-parameter: G(a) = w_E·(E-1) + w_I·(I-⟨I⟩)
        # E = H(a)/H0 (normalized Hubble), I = I-parameter
        # Both zero-mean for stability
        
        if not MASTER_CTRL.get('FI_USE_EI_COMBO', False):
            return None
        
        # E component (normalized Hubble, zero-mean)
        E = H / H0
        E_zero_mean = E - 1.0  # ⟨E⟩ = 1 by definition
        
        # I component (zero-mean)
        I_val = self.info_content.compute_information(a)
        if self.statistics_computed and self.I_std > 1e-10:  # Epsilon guard against division by zero
            I_zero_mean = (I_val - self.I_mean) / self.I_std  # Standardized
        else:
            I_zero_mean = I_val
        
        # Combined G-field
        w_E = MASTER_CTRL.get('W_E_WEIGHT', 1.0)
        w_I = MASTER_CTRL.get('W_I_WEIGHT', 1.0)
        
        G = w_E * E_zero_mean + w_I * I_zero_mean
        
        return G
    
    def rho_DE_CPL(self, a, rho_Lambda, w0=-1.0, wa=0.0):
        # CPL (Chevallier-Polarski-Linder) parameterization fallback
        # w(a) = w0 + wa·(1-a)
        # ρ_DE(a) = ρ_DE,0 · exp[-3∫(1+w(a'))d ln a']
        # Exact solution: ρ_DE(a) = ρ_Λ · a^(-3(1+w0+wa)) · exp[-3wa(1-a)]
        
        # Safety: prevent overflow
        exponent = -3.0*((1.0 + w0 + wa)*np.log(np.maximum(a, 1e-12)) + wa*(1.0 - a))
        exponent = np.clip(exponent, -50, 50)  # Prevent overflow
        
        rho_DE = rho_Lambda * np.exp(exponent)
        
        # Ensure positive density
        rho_DE = np.maximum(1e-12, rho_DE)
        
        return rho_DE
    
    def _build_w_integration_grid(self):
        # Build pre-computed integration grid for uniform_w model
        # This gives 10-50× performance boost for large grids
        
        # Build fine grid from a_min to 1
        self.a_grid_w = np.linspace(MASTER_CTRL['A_MIN'], 1.0, 2000)
        
        # Compute w(a) for all grid points
        self.w_grid = np.array([self.w_DE(a) for a in self.a_grid_w])
        
        # Cumulative integral: ∫ (1+w(a'))/a' da' from a_min to a
        integrand = (1.0 + self.w_grid) / self.a_grid_w
        self.cumulative_integral = cumulative_trapezoid(
            integrand, self.a_grid_w, initial=0
        )
        
        # Build interpolator for fast lookup
        self.integral_interpolator = interp1d(
            self.a_grid_w, 
            self.cumulative_integral,
            kind='cubic', 
            fill_value='extrapolate',
            bounds_error=False
        )
    
    def rho_DE(self, a, rho_Lambda, friedmann=None):
        """
        TQE Dark Energy Density with I-Field Coupling
        
        Translates the TQE fine-tuning function f(E,I) into cosmological dark energy:
        
        TQE THEORETICAL FOUNDATION:
            P'(ψ) = P(ψ) · f(E,I)
            where f(E,I) = exp(-(E-E_c)²/(2σ²)) · (1 + α·I)
        
        COSMOLOGICAL IMPLEMENTATIONS:
            
            1. COVARIANT E-PRESSURE (Primary Model):
               • E-only:  ρ_DE = ρ_Λ · exp(-α·E)
                          Pure energy magnitude effect
               
               • E+I:     ρ_DE = ρ_Λ · exp(-α·E·(1-I))
                          Information modulates coupling strength
                          When I→1 (high information), coupling weakens → ρ_DE closer to ρ_Λ
                          When I→0 (low information), coupling strengthens → ρ_DE deviates more
               
               This captures TQE's mechanism: information orientation affects how
               energy couples to the vacuum, modulating dark energy density.
            
            2. UNIFORM W (Alternative Model):
               w_DE(a) = w₀ + w_I·I(a)
               ρ_DE via integral: exp[-3∫(1+w(a'))d ln a']
               
               Information directly modulates equation of state.
            
            3. GEOMETRIC (Gradient Model):
               ρ_DE = ρ_Λ · exp(β₀·F[I,∇I,∂I])
               
               Tests whether spatial/temporal I-parameter gradients matter.
        
        Args:
            a: Scale factor (0 < a ≤ 1)
            rho_Lambda: Base cosmological constant density (Ω_Λ)
            friedmann: FriedmannEvolution instance (for dynamic E-field)
        
        Returns:
            rho_DE: TQE-modified dark energy density (dimensionless Omega units)
        """
        
        if self.coupling_type == 'covariant_pressure':
            # TQE-COMPLIANT: Exponential coupling with energy-derived I-parameter
            # ρ_DE = ρ_Λ·exp(-α·E·(1-I))  where I = I(E, dE/da)
            
            # TQE ENERGY DEFINITION: E = H(a) / H0 (normalized expansion rate)
            # CRITICAL: Use ΛCDM approximation to avoid infinite recursion!
            # (friedmann.H() calls E_squared() which calls rho_DE() → circular!)
            Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
            Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
            E_field = np.sqrt(Omega_m / a**3 + Omega_Lambda)
            
            # Analytical dE/da for ΛCDM approximation
            dE_da = -1.5 * Omega_m / (a**4 * E_field)
            
            # Compute I-parameter from energy evolution (TQE-COMPLIANT)
            use_dynamic_I = MASTER_CTRL.get('USE_DYNAMIC_I_FIELD', True)
            
            if use_dynamic_I:
                # Dynamic I-parameter with E-feedback (uses I_field_dynamic wrapper)
                I_val = self.info_content.compute_information_dynamic(a, E_field, dE_da)
            else:
                # Direct I-parameter (for energy_based, still needs E and dE_da)
                if self.info_content.model_type == 'energy_based':
                    I_val = self.info_content.compute_information(a, E=E_field, dE_da=dE_da)
                else:
                    # Legacy models (phenomenological/EFT): static I(a)
                    I_val = self.info_content.compute_information(a)
            
            # Get USE_EXP_COUPLING flag from global MASTER_CTRL
            use_exp = MASTER_CTRL.get('USE_EXP_COUPLING', False)
            
            if use_exp:
                # TQE WEIGHTING FUNCTION: P'(ψ) = P(ψ)·f(E,I)
                # f(E,I) = exp(-α·E·(1-I_dynamic))    E+I coupling WITH DYNAMIC I
                # f(E)   = exp(-α·E)                  E-only
                
                alpha_damp = MASTER_CTRL.get('ALPHA_DAMPING', 0.001)
                
                # Use instance coupling_mode if set, otherwise global
                coupling_mode = self.coupling_mode if self.coupling_mode else MASTER_CTRL.get('COUPLING_MODE', 'EplusI')
                
                if coupling_mode == 'Eonly':
                    # E-ONLY: f(E) = exp(-α·E)
                    exponent = -alpha_damp * E_field
                elif coupling_mode == 'EplusI' or coupling_mode == 'dual':
                    # E+I: f(E,I_dynamic) = exp(-α·E·(1-I_dynamic))
                    # NOW WITH DYNAMIC I!
                    exponent = -alpha_damp * E_field * (1.0 - I_val)
                else:
                    # Fallback: old exponential form
                    beta0 = MASTER_CTRL.get('BETA0_OPTIMAL', 0.015)
                    exponent = beta0 * I_val - alpha_damp * E_field
                
                # PHYSICAL CONSTRAINT: Limit |Δ ln ρ_DE| to prevent early-time instability
                # For z≲1 (a≳0.5), I-parameter should only cause ~10% variation
                # This prevents w(a) → unphysical values and ensures energy conditions
                max_delta_ln_rho = MASTER_CTRL.get('I_FIELD_MAX_DELTA_LN_RHO', 0.1)
                exponent = np.clip(exponent, -max_delta_ln_rho, max_delta_ln_rho)
                
                # Apply TQE weighting
                f_TQE = np.exp(exponent)
                
                rho_DE = rho_Lambda * f_TQE
            else:
                # Linear form (original): ρ_DE = ρ_Λ·[1 + α·I]
                rho_DE = rho_Lambda * (1.0 + self.alpha * I_val)
            
            # Ensure positive density (numpy-compatible)
            rho_DE = np.maximum(1e-12, rho_DE)
            
            # CPL FALLBACK: If USE_CPL_FALLBACK=True or rho_DE unstable, use CPL
            if MASTER_CTRL.get('USE_CPL_FALLBACK', False):
                w0 = MASTER_CTRL.get('CPL_W0', -1.0)
                wa = MASTER_CTRL.get('CPL_WA', 0.0)
                rho_DE_cpl = self.rho_DE_CPL(a, rho_Lambda, w0, wa)
                return rho_DE_cpl
            
            # Stability check: warn if extreme deviation (but don't auto-switch)
            if np.any(rho_DE > 10.0 * rho_Lambda):
                print(f"  ⚠ rho_DE > 10×ρ_Λ detected (max factor: {np.max(rho_DE)/rho_Lambda:.2f})")
            if np.any(rho_DE < 0.01 * rho_Lambda):
                print(f"  ⚠ rho_DE < 0.01×ρ_Λ detected (min factor: {np.min(rho_DE)/rho_Lambda:.2f})")
            
            return rho_DE
        
        elif self.coupling_type == 'uniform_w':
            # Variable w(a) evolution: ρ_DE(a) = ρ_DE0 * exp[-3 ∫ (1+w(a')) d ln a']
            # OPTIMIZED: Uses pre-computed grid + interpolation (10-50× faster)
            
            # Handle both scalar and array inputs
            a_input = np.atleast_1d(a)
            
            # Fast interpolation from pre-computed cumulative integral
            # Get cumulative integral at a=1 (reference point)
            integral_at_1 = self.cumulative_integral[-1]
            
            # Get cumulative integral at input points (fast interpolation)
            integral_at_a = self.integral_interpolator(a_input)
            
            # ρ_DE(a) = ρ_Λ * exp[-3 * ∫_a^1 (1+w(a'))/a' da']
            # = ρ_Λ * exp[-3 * (integral_at_1 - integral_at_a)]
            rho_DE = rho_Lambda * np.exp(-3.0 * (integral_at_1 - integral_at_a))
            
            # Ensure positive density (numpy-compatible)
            rho_DE = np.maximum(1e-12, rho_DE)
            
            # Return scalar if input was scalar
            if np.isscalar(a):
                return float(rho_DE[0])
            return rho_DE
        
        elif self.coupling_type == 'geometric':
            # Geometric coupling: ρ_DE = ρ_Λ · exp(β₀ · (F_I - ⟨F_I⟩))
            # ADVANCED: Sigmoid-normalized, mean-centered for stability
            # FIXED: E-only vs E+I distinction
            
            coupling_mode = self.coupling_mode if self.coupling_mode else MASTER_CTRL.get('COUPLING_MODE', 'EplusI')
            
            if coupling_mode == 'Eonly':
                # E-only: ρ_DE = ρ_Λ (constant, no I-parameter coupling)
                rho_DE = np.maximum(1e-12, rho_Lambda)
                return rho_DE
            
            # E+I mode (or 'dual' fallback): full I-parameter geometric coupling
            if coupling_mode not in ['EplusI', 'dual']:
                # Safety: unknown mode defaults to E+I
                coupling_mode = 'EplusI'
            
            # Compute I-parameter (TQE-COMPLIANT: from energy evolution)
            if self.info_content.model_type == 'energy_based':
                # CRITICAL: Use ΛCDM approximation to avoid infinite recursion!
                # (friedmann.H() calls E_squared() which calls rho_DE() → circular!)
                Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                E_field = np.sqrt(Omega_m / a**3 + Omega_Lambda)
                dE_da = -1.5 * Omega_m / (a**4 * E_field)
                
                I_val = self.info_content.compute_information(a, E=E_field, dE_da=dE_da)
            else:
                # Legacy models
                I_val = self.info_content.compute_information(a)
            
            # Compute F_I using sigmoid normalization if enabled
            if MASTER_CTRL.get('FI_USE_SIGMOID', True):
                # Sigmoid-based F_I (robust, bounded)
                # F_I = sigmoid((I-⟨I⟩)/σ_I)² + sigmoid(dI/da·aH/κ)²
                
                # I component (standardized)
                if hasattr(self, 'I_mean') and hasattr(self, 'I_std') and self.I_std > 0:
                    I_standardized = (I_val - self.I_mean) / self.I_std
                else:
                    I_standardized = I_val
                
                # Sigmoid function: 1/(1 + exp(-x))
                I_sigmoid = 1.0 / (1.0 + np.exp(-I_standardized))
                
                # Gradient component (if available)
                I_grad2 = self.info_content.compute_information_gradient_squared(a)
                grad_sigmoid = 1.0 / (1.0 + np.exp(-np.sqrt(I_grad2) * 10))  # Scale by 10 for sensitivity
                
                # Combined F_I (bounded 0→1)
                F_I = I_sigmoid**2 + grad_sigmoid**2
                F_I_normalized = F_I / (1.0 + F_I)  # Extra normalization
            else:
                # Simple normalized F_I (fallback)
                I_grad2 = self.info_content.compute_information_gradient_squared(a)
                F_I_raw = I_val**2 + I_grad2
                F_I_normalized = F_I_raw / (1.0 + F_I_raw)
            
            # Mean-centered F_I
            if hasattr(self, 'F_I_mean'):
                F_I_centered = F_I_normalized - self.F_I_mean
            else:
                F_I_centered = F_I_normalized
            
            # Exponential coupling (always positive, bounded growth)
            rho_DE = rho_Lambda * np.exp(self.beta0 * F_I_centered)
            
            # Ensure positive density (numpy-compatible)
            rho_DE = np.maximum(1e-12, rho_DE)
            
            return rho_DE
        
        else:
            # Null model: pure ΛCDM
            # PRODUCTION: Floor guard (even for Null model)
            rho_DE = np.maximum(1e-12, rho_Lambda)
            return rho_DE
    
    def rho_DE_advanced(self, a, rho_Lambda, H):
        # Advanced ρ_DE calculation with new coupling forms
        # Supports: exponential, demeaned, E-field coupling
        
        I_val = self.info_content.compute_information(a)
        
        # Apply I_mean correction if enabled
        if self.use_I_mean and self.I_mean != 0.0:
            I_effective = I_val - self.I_mean
        else:
            I_effective = I_val
        
        # Compute E (TQE definition: E = H/H0)
        if self.use_exp_form:
            E_val = H / MASTER_CTRL['H0']  # TQE-compliant: normalized expansion rate
        else:
            E_val = 0.0
        
        # Apply coupling form
        if self.rho_DE_form == 'exp' and self.use_exp_form:
            # Exponential form: ρ_DE = ρ_Λ · exp(β₀·I - α·E)
            # SAFETY: Clamp exponent to prevent overflow
            MAX_EXP = 5.0
            
            if self.coupling_type == 'covariant_pressure':
                exponent = self.alpha * I_effective - self.alpha * E_val
                if abs(exponent) > MAX_EXP:
                    print(f"⚠ Exponential clamp: |β₀·I - α·E| = {abs(exponent):.2f} > {MAX_EXP}, clamping")
                    exponent = np.clip(exponent, -MAX_EXP, MAX_EXP)
                rho_DE = rho_Lambda * np.exp(exponent)
            elif self.coupling_type == 'geometric':
                exponent = self.beta0 * I_effective
                if abs(exponent) > MAX_EXP:
                    print(f"⚠ Exponential clamp: |β₀·I| = {abs(exponent):.2f} > {MAX_EXP}, clamping")
                    exponent = np.clip(exponent, -MAX_EXP, MAX_EXP)
                rho_DE = rho_Lambda * np.exp(exponent)
            else:
                rho_DE = rho_Lambda
        
        elif self.rho_DE_form == 'demeaned':
            # Demeaned form: ρ_DE = ρ_Λ · [1 + β₀·(I - ⟨I⟩)]
            if self.coupling_type == 'covariant_pressure':
                rho_DE = rho_Lambda * (1.0 + self.alpha * I_effective)
            elif self.coupling_type == 'geometric':
                rho_DE = rho_Lambda * (1.0 + self.beta0 * I_effective)
            else:
                rho_DE = rho_Lambda
        
        else:
            # Default: use standard rho_DE
            rho_DE = self.rho_DE(a, rho_Lambda)
        
        # Ensure positive density
        rho_DE = np.maximum(1e-12, rho_DE)
        
        return rho_DE
    
    def w_DE(self, a):
        # Compute dark energy equation of state at scale factor a
        
        if self.coupling_type == 'uniform_w':
            # Model 2: w_DE(a) = w_0 + w_I·I(a)
            # FIXED: E-only vs E+I distinction
            
            coupling_mode = self.coupling_mode if self.coupling_mode else MASTER_CTRL.get('COUPLING_MODE', 'EplusI')
            
            if coupling_mode == 'Eonly':
                # E-only: w(a) = w_0 (constant, no I-parameter coupling)
                return self.w0
            elif coupling_mode == 'EplusI' or coupling_mode == 'dual':
                # E+I: w(a) = w_0 + w_I·I(a) (I-parameter modulates equation of state)
                
                # For energy_based I-parameter, approximate E and dE/da
                if self.info_content.model_type == 'energy_based':
                    # Approximate E from ΛCDM (no friedmann available here)
                    Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                    Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                    E_approx = np.sqrt(Omega_m / a**3 + Omega_Lambda)
                    dE_da_approx = -1.5 * Omega_m / (a**4 * E_approx)
                    I_val = self.info_content.compute_information(a, E=E_approx, dE_da=dE_da_approx)
                else:
                    # Legacy models
                    I_val = self.info_content.compute_information(a)
                
                return self.w0 + self.w_I * I_val
            else:
                # Fallback to E+I
                if self.info_content.model_type == 'energy_based':
                    Omega_m = MASTER_CTRL.get('OMEGA_M', 0.315)
                    Omega_Lambda = MASTER_CTRL.get('OMEGA_LAMBDA', 0.685)
                    E_approx = np.sqrt(Omega_m / a**3 + Omega_Lambda)
                    dE_da_approx = -1.5 * Omega_m / (a**4 * E_approx)
                    I_val = self.info_content.compute_information(a, E=E_approx, dE_da=dE_da_approx)
                else:
                    I_val = self.info_content.compute_information(a)
            return self.w0 + self.w_I * I_val
        
        else:
            # For other models, assume w = -1 (cosmological constant)
            return -1.0

# ==========================================================================================
# FRIEDMANN EVOLUTION
# ==========================================================================================

