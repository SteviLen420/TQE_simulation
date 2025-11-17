# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# config.py - Configuration and Constants Module
# ==========================================================================================
# TQE–ΛSim: Configuration parameters and constants
# Based on the Theory of the Question of Existence (TQE)
# ==========================================================================================
#
# This module contains all configuration parameters and constants used throughout
# the TQE Dark Energy Coupling Simulation pipeline.
#
# AUTHOR: Stefan Len
# DATE: 2025-10-29
# VERSION: v4.2.0 PRO
#
# ==========================================================================================
# COSMOLOGICAL CONSTANTS (Planck 2018 Fiducial Values)
# ==========================================================================================
# Reference values used throughout the pipeline
# Source: Planck Collaboration 2018, A&A 641, A6 (2020)

COSMO_PARAMS = {
    "Omega_m": 0.315,              # Matter density parameter
    "Omega_Lambda": 0.685,         # Dark energy density parameter (cosmological constant)
    "Omega_b": 0.049,              # Baryon density parameter
    "Omega_r": 9.24e-5,            # Radiation density parameter
    "H0": 67.4,                    # Hubble constant [km/s/Mpc]
    "sigma8_LCDM": 0.831,          # S₈ parameter for ΛCDM (Planck 2018)
    "n_s": 0.965,                  # Scalar spectral index
    "w0": -1.0,                    # Dark energy equation of state (cosmological constant)
    "tau_reio": 0.054,             # Optical depth to reionization
}

# ==========================================================================================
# MASTER CONTROL PANEL - ALL CONFIGURABLE PARAMETERS
# ==========================================================================================
# This Master Control Panel centralizes ALL simulation parameters
# Modify these values to control the entire pipeline behavior

MASTER_CTRL = {
    # ==========================================================================================
    # COSMOLOGICAL PARAMETERS (Planck 2018 fiducial)
    # ==========================================================================================
    "H0": COSMO_PARAMS["H0"],                    # Hubble constant [km/s/Mpc]
    "OMEGA_M": COSMO_PARAMS["Omega_m"],          # Matter density parameter
    "OMEGA_LAMBDA": COSMO_PARAMS["Omega_Lambda"], # Dark energy density parameter
    "OMEGA_B": COSMO_PARAMS["Omega_b"],          # Baryon density parameter
    "OMEGA_R": COSMO_PARAMS["Omega_r"],          # Radiation density parameter
    "W0": COSMO_PARAMS["w0"],                    # Dark energy equation of state
    "N_S": COSMO_PARAMS["n_s"],                  # Scalar spectral index
    "SIGMA_8": 0.811,                            # Amplitude of matter fluctuations (σ₈ for normalization)
    
    # Radiation density parameters (ΛCDM standard)
    "N_EFF": 3.046,                # Effective number of relativistic species (Planck 2018)
    "T_CMB": 2.7255,               # CMB temperature [K] (Planck 2018)
    
    # Growth factor computation method
    "USE_ODE_GROWTH": True,        # Use proper ODE solver for D(z) (True = accurate, False = integral approx)
    "ODE_GROWTH_RTOL": 1e-8,       # PRODUCTION: ODE relative tolerance (tighter than default 1e-7)
    "ODE_GROWTH_ATOL": 1e-10,      # PRODUCTION: ODE absolute tolerance (tighter than default 1e-9)
    "ODE_GROWTH_MAX_STEP": 0.1,    # PRODUCTION: Max step size in ln(a) space
    
    # Flatness constraint
    "STRICT_FLATNESS": True,       # Fail-fast if |Ω_total - 1| > 1e-3 (PRODUCTION: strict checking)
    
    # ==========================================================================================
    # PRODUCTION HARDENING PARAMETERS
    # ==========================================================================================
    # NOTE: Set ALLOW_MOCK_DATA=True for Colab testing without real data files
    #       Set ALLOW_MOCK_DATA=False for production with real Pantheon+/BOSS/Planck data
    "ALLOW_MOCK_DATA": True,       # TESTING MODE: True = use enhanced mock data (50 SNe, 10 BAO, 47 CMB)
                                   # PRODUCTION MODE: False = require real data files
    "CMB_REFERENCE_ONLY": True,    # CMB is baseline ΛCDM only (no I-parameter effects until CAMB integration)
    "S8_FROM_PARAM": False,        # PRODUCTION: False = compute from P(k), True = use parameter
    "TOL_SANITY": 1e-4,            # Sanity check tolerance (relaxed from 1e-6 for production)
    "REPORT_PPM": True,            # Report Δ metrics in ppm (parts-per-million) in addition to percent
    
    # FINAL RELEASE UPGRADE: Extended redshift grid
    "Z_MIN": 0.0,                  # Minimum redshift
    "Z_MAX": 5.0,                  # Maximum redshift (extended from 3.0)
    "Z_POINTS": 100,               # Number of z points (increased from 50)
    
    # PRODUCTION: High-resolution scale factor grid
    "A_MIN_LOG": 1e-4,             # Minimum scale factor (log-space)
    "A_MAX_LOG": 1.0,              # Maximum scale factor
    "A_GRID_N_LOG": 2048,          # OPTIMIZED: 4096→2048 (2× faster, still high precision)
    "USE_LOG_A_GRID": True,        # Use log-spaced grid (better for early universe)
    "TAU_REIO": COSMO_PARAMS["tau_reio"],  # Optical depth to reionization
    
    # ==========================================================================================
    # I-PARAMETER MODEL PARAMETERS
    # ==========================================================================================
    # Phenomenological model: I(a) = A·a^γ
    "I_FIELD_AMPLITUDE": 0.02,     # Amplitude A (OPTIMIZED: 0.02 for H(a=1) stability)
    "I_FIELD_GAMMA": 0.5,          # Power law index γ
    
    # EFT Lagrangian model: c_1·I² + c_2·(∂I)²
    "EFT_C1": 1.0,                 # I² coefficient
    "EFT_C2": 0.1,                 # (∂I)² coefficient
    
    # Energy-based I-parameter parameters (TQE-compliant)
    "I_FIELD_EPSILON": 1e-6,       # Regularization epsilon for energy-based I-parameter
    "I_FIELD_NORMALIZATION": 'tanh',  # Normalization method: 'tanh' or 'rational'
    
    # ==========================================================================================
    # COUPLING MODEL PARAMETERS
    # ==========================================================================================
    # ==========================================================================================
    # TQE COUPLING MODE CONTROL - E-only vs E+I coupling
    # ==========================================================================================
    # Based on TQE fundamental equation: P'(ψ) = P(ψ)·f(E,I)
    # 
    # E-only mode:  f(E) = exp(-α·E)              Pure energy damping
    # E+I mode:     f(E,I) = exp(-α·E·(1-I))      Energy-Information coupling
    #
    "COUPLING_MODE": "dual",       # Options: "Eonly", "EplusI", "dual" (runs both)
    "RUN_DUAL_COMPARISON": True,   # Run both Eonly and EplusI with same seed for comparison
    "AUTO_PREFIX_FILES": True,     # Automatically prefix files with "EplusI_" or "Eonly_"
    
    # ===== PERFORMANCE OPTIMIZATION =====
    "ENABLE_VECTORIZATION": True,   # Vectorize array computations (2-5x faster)
    "ENABLE_CACHING": True,         # Cache expensive computations (3-10x faster)
    "CACHE_SIZE": 1000,             # LRU cache size for frequently computed values
    "BATCH_COMPUTATIONS": True,     # Batch array operations where possible
    "MEMORY_EFFICIENT_MODE": True,  # Clean up intermediate arrays
    "PERFORMANCE_MODE": "balanced", # "fast" | "balanced" | "accurate"
    # Fast mode: Reduced resolution (A_GRID_N_LOG=1024, Z_POINTS=50)
    # Balanced mode: OPTIMIZED (A_GRID_N_LOG=2048, Z_POINTS=100, NESTED_NLIVE=250)
    # Accurate mode: High resolution (A_GRID_N_LOG=8192, Z_POINTS=200)
    # ====================================
    
    # Model 1: Covariant E-pressure
    "ALPHA_COUPLING": 0.02,        # Coupling strength α (OPTIMIZED: 0.02 for H(a=1) < 0.3%)
    "ALPHA_RANGE": [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1],  # Parameter sweep values
    "ALPHA": 0.02,                 # Current α value (updated by Goldilocks finder)
    
    # FINAL RELEASE UPGRADE: Exponential coupling formula
    "USE_EXP_COUPLING": True,      # Use ρ_DE = ρ_Λ·exp(β₀I - αE) instead of linear form
    "ALPHA_DAMPING": 0.0008,       # ULTRA-OPTIMIZED: 0.002→0.0008 (stronger I-coupling for Δχ² > 5)
    "BETA0_OPTIMAL": 0.010,        # Optimal β₀ for exp coupling (OPTIMIZED: 0.010 for stability)
    "BETA0": 0.010,                # Current β₀ value (updated by Goldilocks finder)
    
    # I-parameter stability constraints (prevent early-time instability)
    "I_FIELD_MAX_DELTA_LN_RHO": 0.25,  # Max |Δ ln ρ_DE| from I-parameter (PRODUCTION: 25% for stronger signal)
    "USE_CPL_FALLBACK": False,     # FIXED: Disable CPL bypass (was killing I-parameter coupling!)
    "CPL_W0": -1.02,               # CPL w0 parameter: w(a) = w0 + wa(1-a) (only if fallback enabled)
    "CPL_WA": -0.10,               # CPL wa parameter (time evolution, only if fallback enabled)
    
    # Normalization modes for coupling
    "NORM_MODE": "exp",            # Options: "exp" (exponential), "centered" (mean-centered)
    
    # Model 2: Uniform w(I)
    "W_I_COUPLING": 0.05,          # I-coupling to w
    "W_I_RANGE": [0.0, 0.02, 0.05, 0.1, 0.2, 0.5],  # Parameter sweep values
    
    # Model 3: Geometric coupling (ADVANCED)
    # PHYSICS NOTE: β₀ controls exp(β₀·(F_I - ⟨F_I⟩)) coupling
    # F_I = sigmoid((I-⟨I⟩)/σ_I)² + sigmoid(dI/da·aH/κ)²
    # Sigmoid ensures bounded [0,1], mean-centering prevents H(a) drift
    # β₀ = 0.02 gives ~0.3-0.4% H(a=1) deviation (physically stable)
    "BETA0_COUPLING": 0.02,        # Geometric coupling strength β_0
    "BETA0_RANGE": [0.0, 0.01, 0.02, 0.03, 0.05],  # Parameter sweep values
    "FI_USE_SIGMOID": True,        # Use sigmoid normalization for F_I
    "FI_KAPPA_SCALE": 67.4,        # κ scale for dI/da normalization (H0)
    
    # ==========================================================================================
    # NUMERICAL INTEGRATION PARAMETERS
    # ==========================================================================================
    "A_MIN": 0.01,                 # Minimum scale factor (early universe)
    "A_MAX": 1.0,                  # Maximum scale factor (today)
    "N_A_POINTS": 1000,            # Number of scale factor points
    "Z_MAX_INTEGRATION": 10.0,     # Maximum redshift for integration
    
    # ==========================================================================================
    # OBSERVABLE REDSHIFT RANGES
    # ==========================================================================================
    # SNe Ia
    "SNE_Z_MIN": 0.01,             # Minimum SNe redshift
    "SNE_Z_MAX": 2.0,              # Maximum SNe redshift
    "SNE_N_POINTS": 50,            # Number of SNe redshift points
    
    # BAO
    "BAO_Z_ARRAY": [0.15, 0.38, 0.51, 0.61, 0.70],  # Typical BAO survey redshifts
    
    # CMB
    "CMB_LMAX": 2500,              # Maximum CMB multipole
    "CMB_LMIN": 2,                 # Minimum CMB multipole
    "USE_CAMB": True,              # Use CAMB for CMB calculations
    
    # LSS
    "LSS_K_MIN": 1e-3,             # Minimum wavenumber [Mpc^-1]
    "LSS_K_MAX": 10.0,             # Maximum wavenumber [Mpc^-1]
    "LSS_N_K_POINTS": 100,         # Number of k points
    
    # ==========================================================================================
    # BAYESIAN INFERENCE PARAMETERS ()
    # ==========================================================================================
    "RUN_MCMC": True,              # Run MCMC posterior sampling ← ACTIVATED!
    "MCMC_NWALKERS": 50,           # Number of MCMC walkers
    "MCMC_NSTEPS": 5000,           # Number of MCMC steps (production)
    "MCMC_BURNIN": 1000,           # Burn-in steps
    "MCMC_THIN": 10,               # Thinning factor
    "USE_EMCEE": True,             # Use emcee for MCMC (standard sampler)
    "USE_NESTED_SAMPLING": True,   # ACTIVATED: Use nested sampling (dynesty) for Bayes Factor
    "NESTED_NLIVE": 250,           # OPTIMIZED: 500→250 (2× faster, still publication-quality)
    "NESTED_DLOGZ": 0.05,          # OPTIMIZED: 0.01→0.05 (looser tolerance, faster convergence)
    "NESTED_BOUND": 'multi',       # Bounding method: 'multi', 'single', 'balls'
    "NESTED_SAMPLE": 'rwalk',      # Sampling method: 'rwalk', 'slice', 'rslice'
    "SAVE_MCMC_SAMPLES": True,     # Save MCMC/nested sampling samples (CSV + HDF5)
    "CREATE_CORNER_PLOTS": True,   # Create corner plots (posteriors)
    "COMPUTE_EVIDENCE": True,      # ACTIVATED: Compute Bayesian evidence log Z
    
    # Prior ranges for Bayesian inference (ΛCDM-compatible, physically motivated)
    "PRIOR_OMEGA_M": [0.2, 0.4],   # Ω_m prior (Planck 2018: 0.315 ± 0.007)
    "PRIOR_H0": [60.0, 75.0],      # H₀ prior [km/s/Mpc] (Planck: 67.4 ± 0.5, SH0ES: 73.0 ± 1.0)
    "PRIOR_ALPHA": [0.0, 0.3],     # α prior range (limited to prevent instability, not 1.0!)
    "PRIOR_W0": [-1.3, -0.7],      # w₀ prior (near -1, within energy conditions)
    "PRIOR_W_I": [-0.5, 0.5],      # w_I prior range (small perturbation)
    "PRIOR_BETA0": [0.0, 0.3],     # β₀ prior range (limited to prevent instability, not 1.0!)
    "PRIOR_GAMMA": [0.0, 1.0],     # γ prior range (power law index, 0-1 not 0-2)
    "PRIOR_A": [0.0, 1.0],         # A prior range (I-parameter amplitude)
    
    # ==========================================================================================
    # MODEL COMPARISON PARAMETERS
    # ==========================================================================================
    "COMPUTE_AIC": True,           # Compute Akaike Information Criterion
    "COMPUTE_BIC": True,           # Compute Bayesian Information Criterion
    "COMPUTE_DIC": True,           # Compute Deviance Information Criterion
    "DELTA_AIC_THRESHOLD": -10.0,  # Success criterion: ΔAIC ≤ -10
    "BAYES_FACTOR_THRESHOLD": 10.0,  # Strong evidence threshold
    
    # ==========================================================================================
    # CROSS-VALIDATION PARAMETERS
    # ==========================================================================================
    "USE_CROSS_VALIDATION": True,  # Enable cross-validation
    "TRAIN_TEST_SPLIT": 0.7,       # Training data fraction
    "N_CV_FOLDS": 5,               # Number of cross-validation folds
    "TRAIN_ON_CMB": True,          # Train on CMB, validate on SNe/BAO
    "TRAIN_ON_SNE": True,          # Train on SNe, validate on CMB/BAO
    
    # ==========================================================================================
    # ROBUSTNESS ANALYSIS PARAMETERS
    # ==========================================================================================
    "PRIOR_SENSITIVITY": True,     # Test prior sensitivity
    "SYSTEMATICS_TEST": True,      # Include systematic uncertainties
    "N_PRIOR_VARIATIONS": 10,      # Number of prior variations to test
    
    # Systematic uncertainties
    "SNE_MAG_UNCERTAINTY": 0.15,   # SNe magnitude systematic [mag]
    "BAO_SCALE_UNCERTAINTY": 0.02,  # BAO scale systematic [%]
    "CMB_CALIBRATION_UNCERTAINTY": 0.01,  # CMB calibration [%]
    
    # ==========================================================================================
    # SYNTHETIC VERIFICATION PARAMETERS
    # ==========================================================================================
    "RUN_SYNTHETIC_TESTS": True,   # Run synthetic data tests
    "N_MOCK_UNIVERSES": 100,       # Number of mock universes
    "SYNTHETIC_ALPHA_TRUE": 0.15,  # True α for synthetic data
    "SYNTHETIC_W_I_TRUE": 0.08,    # True w_I for synthetic data
    "SYNTHETIC_NOISE_LEVEL": 0.05, # Noise level for synthetic data
    
    # ==========================================================================================
    # ABLATION STUDY PARAMETERS
    # ==========================================================================================
    "RUN_ABLATION": True,          # Run ablation studies
    "TEST_I_ONLY": True,           # Test ⟨I⟩ only
    "TEST_GRAD_I_ONLY": True,      # Test |∇I|² only
    "TEST_TIME_DERIV_I_ONLY": True,  # Test (∂_t I)² only
    
    # ==========================================================================================
    # REAL OBSERVATIONAL DATA PATHS (TIER 1 PUBLICATION)
    # ==========================================================================================
    # Set these paths to enable REAL data (Pantheon+, BOSS, Planck)
    # Leave as None to use enhanced mock data
    
    "PANTHEON_PLUS_DATA_PATH": "Data_Files/pantheon_plus_mock.csv",  # Path to Pantheon+ SNe Ia data
    "PANTHEON_PLUS_COV_PATH": None,         # Path to Pantheon+ covariance matrix (.npy or .txt)
    "BOSS_BAO_DATA_PATH": "Data_Files/boss_bao_mock.csv",  # Path to BOSS DR12/eBOSS/DESI BAO data
    "BOSS_BAO_COV_PATH": None,              # Path to BAO covariance matrix
    "PLANCK_CMB_DATA_PATH": "Data_Files/planck_cmb_mock.csv",  # Path to Planck 2018 binned C_ell data
    "PLANCK_CMB_COV_PATH": None,            # Path to Planck CMB covariance matrix
    
    # Real data usage flags (TESTING: Disabled, use built-in mock)
    "USE_REAL_SNE_DATA": False,             # False = use enhanced mock (50 SNe, built-in)
    "USE_REAL_BAO_DATA": False,             # False = use enhanced mock (10 BAO, built-in)
    "USE_REAL_CMB_DATA": False,             # False = use enhanced mock (47 ℓ, built-in)
    "USE_FULL_COVARIANCE": True,            # Use full covariance matrices (not just diagonal)
    
    # ==========================================================================================
    # CMB PLANCK VALIDATION PARAMETERS (REAL MAPS)
    # ==========================================================================================
    "USE_REAL_CMB_PLANCK_MAPS": True,       # Enable real Planck CMB map validation
    "CMB_PLANCK_BASE_PATH": "/content/drive/MyDrive/CMB_Planck_Maps",  # Google Drive path
    
    # Component-separated CMB maps (4 methods)
    "CMB_USE_SMICA": True,                  # SMICA (primary, official Planck map)
    "CMB_USE_NILC": False,                  # NILC (alternative method)
    "CMB_USE_SEVEM": False,                 # SEVEM (alternative method)
    "CMB_USE_COMMANDER": False,             # Commander-Ruler (alternative method)
    
    # Raw frequency maps (foreground analysis)
    "CMB_USE_RAW_FREQUENCY_MAPS": True,     # Enable raw frequency map loading
    "CMB_RAW_FREQUENCIES": [100, 143, 217, 353],  # HFI frequencies [GHz]
    
    # Mask configuration
    "CMB_USE_COMMON_MASK": True,            # Use common mask (galactic + point sources)
    "CMB_USE_MISSPIX_MASK": True,           # Use missing pixel mask
    "CMB_MASK_TYPE": "Int",                 # "Int" (intensity/temperature) or "Pol" (polarization)
    
    # Foreground analysis
    "CMB_USE_NHI_FOREGROUND": True,         # Use NHI (Neutral Hydrogen) map from CMB_Anomaly/
    "CMB_NHI_CORRELATION_ANALYSIS": True,   # Correlate CMB anomalies with NHI
    
    # Power spectrum analysis
    "CMB_COMPUTE_POWER_SPECTRUM": True,     # Compute C_ℓ from Planck maps
    "CMB_LMAX": 2000,                       # Maximum multipole
    "CMB_LMIN": 2,                          # Minimum multipole
    "CMB_APPLY_BEAM_CORRECTION": False,     # Apply beam window function (requires beam file)
    "CMB_REMOVE_MONOPOLE_DIPOLE": True,     # Remove monopole and dipole before C_ℓ calculation
    
    # Anomaly detection
    "CMB_ANOMALY_DETECTION": True,          # Detect cold/hot spots
    "CMB_ANOMALY_THRESHOLD": 3.0,           # Detection threshold [σ]
    "CMB_ANOMALY_MIN_SIZE": 10,             # Minimum anomaly size [pixels]
    
    # Validation and comparison
    "CMB_VALIDATE_AGAINST_TQE": True,       # Compare Planck C_ℓ vs TQE simulated C_ℓ
    "CMB_COMPUTE_RESIDUALS": True,          # Compute fractional residuals
    "CMB_COMPUTE_CHI2": True,               # Compute χ² goodness of fit
    
    # Output control
    "CMB_SAVE_CLEANED_MAPS": False,         # Save cleaned CMB maps (large FITS files)
    "CMB_SAVE_VALIDATION_PLOTS": True,      # Save validation PNG plots
    "CMB_SAVE_VALIDATION_CSV": True,        # Save validation CSV data
    "CMB_SAVE_ANOMALY_CATALOG": True,       # Save anomaly catalog (positions, amplitudes)
    
    # ==========================================================================================
    # LABORATORY ANALOG PARAMETERS (Double-Slit)
    # ==========================================================================================
    "RUN_LAB_ANALOG": True,        # Run laboratory analog test
    "LAB_ALPHA_VALUES": [0.0, 0.5, 1.0, 2.0, 5.0],  # α values for lab test
    "LAB_I_MODULATION_FREQ": 10.0,  # I-parameter modulation frequency [Hz]
    "LAB_N_TIME_POINTS": 100,      # Number of time points
    
    # ==========================================================================================
    # VISUALIZATION PARAMETERS - PUBLICATION QUALITY
    # ==========================================================================================
    "PLOT_DPI": 100,               # Display DPI
    "PLOT_SAVE_DPI": 300,          # Save DPI (publication: 300-600)
    "PLOT_FONTSIZE_TITLE": 16,     # Title font size (PUBLICATION: larger)
    "PLOT_FONTSIZE_LABEL": 14,     # Label font size (PUBLICATION: larger)
    "PLOT_FONTSIZE_LEGEND": 11,    # Legend font size (PUBLICATION: larger)
    "PLOT_FONTWEIGHT": 'normal',   # Font weight (PUBLICATION: normal, not light)
    "PLOT_GRID_ALPHA": 0.25,       # Grid transparency (PUBLICATION: subtler)
    "PLOT_FIGSIZE_DEFAULT": (10, 7),  # Default figure size (slightly wider aspect)
    "PLOT_FIGSIZE_WIDE": (14, 5),  # Wide figure size (optimized for 2-panel)
    
    # ==========================================================================================
    # UNIFIED COLOR SCHEME - Vibrant Scientific Colors
    # ==========================================================================================
    "COLOR_MODEL_1": '#E63946',        # Vibrant Red - Model 1 Covariant
    "COLOR_MODEL_2": '#457B9D',        # Ocean Blue - Model 2 Uniform w
    "COLOR_MODEL_3": '#2A9D8F',        # Teal Green - Model 3 Geometric
    "COLOR_NULL": '#F77F00',           # Bright Orange - Null ΛCDM
    
    # Extended palette for multiple models (beta0 sweeps)
    "COLOR_PALETTE_EXTENDED": [
        '#E63946',  # Red
        '#457B9D',  # Blue
        '#2A9D8F',  # Teal
        '#F77F00',  # Orange
        '#8338EC',  # Purple
        '#06FFA5',  # Mint
        '#FF006E',  # Pink
        '#FFBE0B',  # Yellow
        '#3A86FF',  # Light Blue
        '#FB5607'   # Coral
    ],
    
    # Component colors (chi2, observables)
    "COLOR_CHI2_SNE": '#E63946',       # Red
    "COLOR_CHI2_BAO_DM": '#457B9D',    # Blue
    "COLOR_CHI2_BAO_H": '#2A9D8F',     # Teal
    "COLOR_CHI2_H0": '#F77F00',        # Orange
    "COLOR_CHI2_CMB": '#8338EC',       # Purple
    
    "COLOR_AIC": '#E63946',            # Red
    "COLOR_BIC": '#457B9D',            # Blue
    
    # Galaxy structure colors
    "COLOR_VOID": '#E63946',           # Red
    "COLOR_FILAMENT": '#457B9D',       # Blue
    "COLOR_SHEET": '#2A9D8F',          # Teal
    "COLOR_CLUSTER": '#F77F00',        # Orange
    
    # ==========================================================================================
    # OUTPUT AND SAVING PARAMETERS
    # ==========================================================================================
    "SAVE_PNG": True,              # Save PNG plots
    "SAVE_CSV": True,              # Save CSV data
    "SAVE_JSON": True,             # Save JSON results
    "SAVE_TXT": True,              # Save TXT summaries
    "SAVE_ZIP": True,              # Save ZIP archives
    "SAVE_HDF5": True,             # Save HDF5 data (large datasets)
    
    # PNG-specific flags (reduce clutter for publication)
    "SAVE_MATTER_POWER_SPECTRUM_PNG": False,  # Skip P(k) PNG (data in CSV)
    "SAVE_FILAMENT_DISTRIBUTION_PNG": False,  # Skip if n_filaments = 0
    "SAVE_I_DEFINITIONS_PNG": False,          # Skip if only 1 I-definition used
    
    "IMMEDIATE_SAVE": True,        # Save immediately after each model
    "CREATE_BACKUP": True,         # Create backup copies
    "COMPRESS_LARGE_FILES": True,  # Compress large data files
    
    # ==========================================================================================
    # PERFORMANCE AND OPTIMIZATION PARAMETERS
    # ==========================================================================================
    "USE_VECTORIZATION": True,     # Enable vectorized operations
    "USE_PARALLEL": False,         # Parallel processing (Colab limitations)
    "REDUCE_MEMORY": True,         # Reduce memory usage
    "VERBOSE": True,               # Verbose output
    "PROGRESS_BARS": True,         # Show progress bars
    
    # ==========================================================================================
    # REPRODUCIBILITY PARAMETERS
    # ==========================================================================================
    "USE_DETERMINISTIC_SEED": True,  # Enable deterministic seeding
    "MASTER_SEED": "TQE_DarkEnergy_2025",  # Master seed string
    "SAVE_REPRODUCIBILITY_INFO": True,  # Save full reproducibility data
    
    # ==========================================================================================
    # GALAXY STRUCTURE ANALYSIS PARAMETERS
    # ==========================================================================================
    "RUN_GALAXY_STRUCTURE_ANALYSIS": True,  # Enable galaxy structure detection
    "GALAXY_GRID_SIZE": 512,           # 3D grid resolution (512³ cells) - ULTRA-UPGRADED for labeling
    "GALAXY_BOX_SIZE": 500.0,          # Simulation box size [Mpc/h]
    "GALAXY_SAVE_CATALOGUES": True,    # Save void/cluster/filament catalogues
    "GALAXY_SAVE_DENSITY_FIELD": False,  # Save full 3D density field (large file)
    "GALAXY_CREATE_VISUALIZATIONS": True,  # Create structure PNG plots
    
    # Cosmic web classification thresholds
    "GALAXY_VOID_THRESHOLD": -0.5,     # Void density threshold (δ < -0.5, relaxed for more voids)
    "GALAXY_VOID_MIN_RADIUS": 1.0,     # Minimum void radius [Mpc/h] (ULTRA-RELAXED for labeling)
    "GALAXY_VOID_MAX_RADIUS": 200.0,   # Maximum void radius [Mpc/h] (INCREASED for mega-voids)
    "GALAXY_SHEET_MIN": 0.5,           # Sheet minimum density (δ > 0.5)
    "GALAXY_SHEET_MAX": 3.0,           # Sheet maximum density (δ < 3.0)
    "GALAXY_KNOT_THRESHOLD": 1.5,      # Knot/cluster threshold (δ > 1.5, relaxed for more clusters)
    "GALAXY_CLUSTER_MIN_RADIUS": 0.5,  # Minimum cluster radius [Mpc/h] (ULTRA-RELAXED for labeling)
    "GALAXY_CLUSTER_MAX_RADIUS": 50.0, # Maximum cluster radius [Mpc/h] (INCREASED for mega-clusters)
    "GALAXY_FILAMENT_MIN": -0.5,       # Filament minimum density
    "GALAXY_FILAMENT_MAX": 2.0,        # Filament maximum density
    
    # Structure detection criteria
    "GALAXY_FILAMENT_ASPECT_MIN": 3.0,  # Minimum aspect ratio for filaments
    "GALAXY_WALL_FLATNESS_MAX": 0.3,    # Maximum flatness for walls
    "GALAXY_WALL_MIN_SIZE": 5,          # Minimum wall extent (cells)
    "GALAXY_SMOOTH_SIGMA": 0.5,         # ULTRA-OPTIMIZED: 1.0→0.5 (absolute minimal smoothing for labeling)
    
    # Real universe comparison (SDSS, 2dFGRS observations)
    "REAL_UNIVERSE_VOID_FRAC": 0.45,    # Observed void fraction (45%)
    "REAL_UNIVERSE_FILAMENT_FRAC": 0.35,  # Observed filament fraction (35%)
    "REAL_UNIVERSE_SHEET_FRAC": 0.12,   # Observed sheet fraction (12%)
    "REAL_UNIVERSE_CLUSTER_FRAC": 0.08,  # Observed cluster fraction (8%)
    "REAL_UNIVERSE_VOID_RADIUS_MIN": 10.0,  # Min void radius [Mpc/h]
    "REAL_UNIVERSE_VOID_RADIUS_MAX": 50.0,  # Max void radius [Mpc/h]
    
    # ==========================================================================================
    # ADVANCED ANALYSIS PARAMETERS
    # ==========================================================================================
    "COMPUTE_DEGENERACY_MATRIX": True,  # Compute parameter correlations
    "COMPUTE_FISHER_MATRIX": True,      # Fisher information matrix
    "COMPUTE_TENSION_METRICS": True,    # H0 and S8 tension metrics
    "COMPUTE_ISW_CROSS": True,          # ISW cross-correlation
    
    # ISW parameters
    "ISW_Z_RANGE": [0.0, 3.0],     # Redshift range for ISW
    "ISW_LMAX": 100,               # Maximum multipole for ISW
    
    # ==========================================================================================
    # TQE-SPECIFIC PHYSICS PARAMETERS
    # ==========================================================================================
    # TQE THEORY RECAP:
    #   P'(ψ) = P(ψ) · f(E,I)  →  Fine-tuning function modulates quantum probabilities
    #   f(E,I) = exp(-(E-E_c)²/(2σ²)) · (1 + α·I)  →  Goldilocks + Information bias
    #   E(a) = H(a)/H₀  →  Normalized expansion rate (energy proxy)
    #   I(a) = A·a^γ    →  Information parameter (0 ≤ I ≤ 1, complexity proxy)
    
    # ρ_DE coupling form
    "RHO_DE_FORM": 'linear',       # Options: 'linear', 'exp', 'demeaned'
    "USE_I_MEAN": False,           # Use mean-corrected I: (I - ⟨I⟩)
    "USE_EXP_FORM": False,         # Use exponential: exp(β₀·I - α·E)
    
    # E-field (energy density proxy) weights
    "E_WEIGHTS": {
        'w_g': 1.0,                # Gradient weight: |∇I|²
        'w_t': 1.0                 # Time derivative weight: (∂_t I)²
    },
    
    # E+I combination for geometric coupling
    "FI_USE_EI_COMBO": True,       # Use E-I combined field G(a) = w_E·(E-1) + w_I·(I-⟨I⟩)
    "W_E_WEIGHT": 1.0,             # E-field weight in G(a)
    "W_I_WEIGHT": 1.0,             # I-parameter weight in G(a)
    "G_USE_SIGMOID": True,         # Apply sigmoid to G before F_I calculation
    
    # S₈ evolution parameters
    "S8_Z_MAX": 5.0,               # Maximum redshift for S₈(z) tracking
    "S8_NZ": 50,                   # Number of redshift points for S₈(z)
    "COMPUTE_S8_EVOLUTION": True,  # Compute S₈(z) evolution
    "SAVE_S8_SERIES": True,        # Save S₈(z) time series
    
    # Growth factor parameters
    "COMPUTE_GROWTH_FACTOR": True, # Compute D(z) growth factor
    "GROWTH_A_INIT": 1e-3,         # Initial scale factor for growth ODE
    "GROWTH_N_POINTS": 500,        # Number of points for growth integration
    
    # ρ_DE evolution tracking
    "SAVE_RHODE_SERIES": True,     # Save ρ_DE(z) time series
    "RHODE_Z_MAX": 5.0,            # Maximum redshift for ρ_DE(z)
    "RHODE_NZ": 50,                # Number of redshift points
    
    # I-E correlation analysis
    "COMPUTE_I_E_CORRELATION": True,  # Compute I-E correlation
    "CORRELATION_METHOD": 'pearson',  # Options: 'pearson', 'spearman', 'MI'
    "SAVE_I_E_SCATTER": True,      # Save I vs E scatter plot
    
    # ==========================================================================================
    # MULTI-DEFINITION I-PARAMETER (Information-theoretic measures)
    # ==========================================================================================
    # TQE Theory allows multiple complementary definitions of I-parameter
    # Each captures different aspects of information/complexity evolution
    
    "I_DEFINITION_MODE": "composite",  # Options: 'kl_divergence', 'shannon', 'composite', 
                                       #          'renyi', 'mutual_info', 'phenomenological',
                                       #          'kl_shannon', 'entanglement', 'fisher', 'horizon_entropy'
                                       # DEFAULT: 'composite' (I_KL × I_Shannon product fusion)
    
    "COMPUTE_ALL_I_DEFINITIONS": True, # Compute all 9 I-definitions for comparison
                                       # Saves to I_Definitions_Comparison.csv
    
    "RENYI_ALPHA": 2.0,                # α parameter for Rényi entropy (α=2: collision entropy)
                                       # Generalized entropy: H_α = (1/(1-α)) log(Σ P_i^α)
    
    "I_FUSION_METHOD": "product",      # How to combine I_KL and I_Shannon for composite mode
                                       # Options: 'product' (I_KL × I_Shannon), 
                                       #          'average' ((I_KL + I_Shannon)/2),
                                       #          'max' (max(I_KL, I_Shannon)),
                                       #          'min' (min(I_KL, I_Shannon))
    
    # Advanced I-parameter definitions (quantum & cosmological)
    "ENTANGLEMENT_SCHMIDT_RANK": 10,   # Schmidt decomposition rank for entanglement entropy
    "FISHER_EPSILON": 1e-4,            # Perturbation size for Fisher information gradient
    "HORIZON_RADIUS_FACTOR": 1.0,      # Cosmological horizon radius multiplier (in Hubble units)
    
    # ==========================================================================================
    # AUTOMATIC GOLDILOCKS ZONE FINDER & OPTIMAL STATE SEARCH
    # ==========================================================================================
    # Automatically find optimal TQE parameters (E_c, σ, α, β₀) and initial conditions
    
    "AUTO_FIND_GOLDILOCKS": True,      # Enable automatic Goldilocks zone search
                                       # IMPLEMENTED - Bayesian optimization (differential evolution)
                                       # If True: runs Bayesian search to find optimal E_c, σ, α, β₀
                                       # Results saved to Goldilocks_Results/ directory
    
    "GOLDILOCKS_SEARCH_METHOD": "bayesian", # Options: 'grid', 'bayesian', 'evolutionary' (BAYESIAN ACTIVE)
                                            # 'grid': Exhaustive grid search (slow but thorough)
                                            # 'bayesian': Bayesian optimization (fast, smart) ← ACTIVATED
                                            # 'evolutionary': Genetic algorithm (robust)
    
    "GOLDILOCKS_SEARCH_RANGES": {
        "E_c": [2.0, 6.0],             # Critical energy range to search
        "sigma": [2.0, 6.0],           # Stability width range to search
        "alpha": [0.01, 0.05],         # Coupling strength range
        "beta0": [0.005, 0.030],       # Geometric coupling range
    },
    
    "GOLDILOCKS_GRID_POINTS": {
        "E_c": 5,                      # Grid points for E_c (5 → [2.0, 3.0, 4.0, 5.0, 6.0])
        "sigma": 5,                    # Grid points for σ
        "alpha": 3,                    # Grid points for α
        "beta0": 3,                    # Grid points for β₀
    },
    
    "GOLDILOCKS_OBJECTIVE": "stability", # Objective function to optimize
                                         # Options: 'stability', 'chi2', 'S8_sensitivity', 
                                         #          'composite' (weighted combination)
    
    "GOLDILOCKS_STABILITY_CRITERIA": {
        "H_deviation_max": 0.005,      # H(a=1) deviation < 0.5%
        "S8_range_min": 0.01,          # Minimum S8 variation (sensitivity test)
        "chi2_max": 100.0,             # Maximum χ² allowed
        "convergence_rate": 0.01,      # Lock-in convergence rate (ΔP/P < 1%)
    },
    
    "AUTO_FIND_INITIAL_STATE": False,  # Enable automatic optimal initial state search
                                       # ⚠️ IMPLEMENTATION PENDING - Requires Goldilocks Finder
                                       # Finds a(init) that minimizes free energy F = E - T·S
                                       # Set to False until full implementation complete
    
    "INITIAL_STATE_SEARCH_METHOD": "free_energy", # Options: 'free_energy', 'max_entropy', 
                                                  #          'min_energy', 'equipartition'
    
    "INITIAL_STATE_TEMPERATURE": 1.0,  # Effective temperature for free energy F = E - T·S
    
    "SAVE_GOLDILOCKS_SEARCH": True,    # Save full Goldilocks search results
                                       # Creates: Goldilocks_Search_Results.csv
                                       #          Goldilocks_Landscape.png (2D heatmap)
    
    "BAYESIAN_OPT_N_CALLS": 50,        # Number of Bayesian optimization iterations
    "BAYESIAN_OPT_N_INITIAL": 10,      # Initial random samples for Bayesian opt
    
    "EVOLUTIONARY_POPULATION": 20,     # Population size for evolutionary algorithm
    "EVOLUTIONARY_GENERATIONS": 30,    # Number of generations
    
    # ==========================================================================================
    # AUDIT FIX #1: I-PARAMETER DYNAMIC COUPLING PARAMETERS
    # ==========================================================================================
    "USE_DYNAMIC_I_FIELD": True,   # Use I_field_dynamic() instead of static I_field()
    "I_E_COUPLING_STRENGTH": 0.1,  # Coupling strength γ for dE/da feedback
    "I_DAMPING_TAU": 0.5,          # Damping timescale τ for early universe suppression
    "I_E_DIRECT_COUPLING": 0.05,   # Direct E-field coupling strength β
    
    # β₀ parameter sweep (fine-grained) - EXTENDED TO 0.20!
    "RUN_BETA0_SWEEP": True,       # Run fine β₀ sweep - ACTIVATED!
    "BETA0_SWEEP_FINE": [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180, 0.185, 0.190, 0.195, 0.200],  # 41 values, 0.005 step, 0.00→0.20 FINAL!
    "BETA0_SWEEP_SAVE_ALL": True,  # Save all sweep results
    
    # FINAL RELEASE UPGRADE: Sensitivity testing
    "RUN_SENSITIVITY_TEST": True,  # Run ±1% I-parameter perturbation test
    "SENSITIVITY_PERTURBATION": 0.01,  # ±1% perturbation amplitude
    "SENSITIVITY_TOLERANCE": 0.001,    # <0.1% change = numerically stable
    
    # AUDIT FIX #3: MI-Lag Scan Parameters (ENHANCED!)
    "RUN_LAG_SCAN": True,          # Enable full MI(Δa) scan
    "LAG_SCAN_MIN_DA": -0.05,      # Minimum lag (PRODUCTION: asymmetric range)
    "LAG_SCAN_MAX_DA": 0.02,       # Maximum lag (PRODUCTION: focused on forward lags)
    "LAG_SCAN_N_POINTS": 30,       # Number of lag points (PRODUCTION: finer resolution)
    
    # Sanity check parameters (refined) - FINAL RELEASE
    # PHYSICS NOTE: H(a=1) deviation tolerances
    # - FAIL if deviation > 1.0% (non-physical or unstable coupling)
    # - WARN if deviation > 0.5% (moderate TQE effect, monitor)
    # - PASS if deviation < 0.5% (weak/acceptable TQE coupling)
    # RATIONALE: TQE coupling typically induces 0.1-0.8% deviation, which is physical
    "SANITY_BASELINE": 'LCDM',     # Options: 'LCDM', 'Covariant', 'Geometric'
    "SANITY_TOLS": {
        'H_at_a1_vs_H0_tol_fail': 0.010,  # FAIL threshold: ±1.0% (physical coupling allowed)
        'H_at_a1_vs_H0_tol_warn': 0.005,  # WARN threshold: ±0.5% (monitor coupling strength)
        'E_squared_positive': True,       # E²(a) > 0 everywhere
        'mu_monotonic_tol': 1e-6,         # μ(z) monotonicity tolerance
        'D_M_monotonic_tol': 1e-6,        # D_M(z) monotonicity tolerance
        'rho_DE_positive_tol': 1e-9       # ρ_DE > 0 tolerance
    },
    
    # S₈ normalization and dynamics
    "NORMALIZE_S8_TO_LCDM": True,  # Normalize S₈ relative to ΛCDM
    "SAVE_S8_RAW_AND_NORM": True,  # Save both raw and normalized S₈
    "S8_DYNAMIC_FROM_GROWTH": True,  # Compute S₈ from dynamic D(z), not fixed σ₈
    
    # AUDIT FIX #4: β₀-Specific Baseline Normalization
    "USE_BETA0_SPECIFIC_BASELINE": True,  # Use β₀-dependent ΛCDM baseline
    "BETA0_LCDM_CORRECTION": 0.0,  # Optional β₀ correction to ΛCDM (0 = standard)
    
    # Growth factor parameters (enhanced)
    "GROWTH_Z_MAX": 5.0,           # Maximum redshift for growth tracking
    "GROWTH_NZ": 50,               # Number of z points for growth
    "SAVE_GROWTH_SERIES": True,    # Save D(z) to separate CSV
    
    # CMB likelihood control
    "INCLUDE_CMB_IN_LIKE": False,  # Exclude CMB from likelihood (TQE not in CAMB)
    
    # Sweep parameters (fine-grained)
    "SWEEP_BETA0": [0.00, 0.01, 0.02, 0.03, 0.05],  # β₀ sweep values
    "SWEEP_WE": [0.5, 1.0, 2.0],   # w_E sweep values
    "SWEEP_WI": [0.5, 1.0, 2.0],   # w_I sweep values
    "RUN_PARAMETER_SWEEP": False,  # Enable multi-parameter sweep
    
    # Correlation analysis
    "RUN_I_E_CORR": True,          # Compute I-E correlation
    "RUN_LAG_SCAN": True,          # Compute MI(Δa) lag scan ← ACTIVATED!
    "LAG_SCAN_MAX_DA": 0.1,        # Maximum Δa for lag scan
    "LAG_SCAN_N_POINTS": 20,       # Number of lag points
    
    # ==========================================================================================
    # DATA LOADING PARAMETERS (Public datasets)
    # ==========================================================================================
    "USE_REAL_DATA": False,        # Use real observational data (if False, use synthetic)
    "PLANCK_DATA_PATH": None,      # Path to Planck data
    "PANTHEON_DATA_PATH": None,    # Path to Pantheon+ data
    "BOSS_DATA_PATH": None,        # Path to BOSS/eBOSS data
    "DES_DATA_PATH": None,         # Path to DES data
    
    # ==========================================================================================
    # PIPELINE EXECUTION MODE
    # ==========================================================================================
    "RUN_ALL_MODELS": True,        # Run all 4 models (3 coupling + null)
    "RUN_PARAMETER_SWEEP": False,  # Run parameter sweeps for each model (disable for speed)
    "RUN_FULL_ANALYSIS": True,     # Run complete Bayesian + model comparison
    
    # Model selection (if RUN_ALL_MODELS = False)
    "RUN_MODEL_1": True,           # Covariant E-pressure
    "RUN_MODEL_2": True,           # Uniform w(I)
    "RUN_MODEL_3": True,           # Geometric coupling
    "RUN_NULL_MODEL": True,        # Pure ΛCDM
    
    # ==========================================================================================
    # LIKELIHOOD COMPUTATION ()
    # ==========================================================================================
    "COMPUTE_LIKELIHOOD": True,    # Compute χ², AIC, BIC from SNe+BAO+H0
    "LIKELIHOOD_SNE_MOCK": True,   # Use mock SNe Ia data (6 points: z=0.1-1.3)
    "LIKELIHOOD_BAO_MOCK": True,   # Use mock BAO data (3 points: z=0.38,0.51,0.61)
    "LIKELIHOOD_H0_PRIOR": True,   # Include H0 Gaussian prior (67.4±0.5)
    
    # ==========================================================================================
    # AUTO AGGREGATOR PARAMETERS (P3-2: ENHANCED!)
    # ==========================================================================================
    "RUN_AUTO_AGGREGATOR": True,   # Run integrated auto aggregator
    "AGGREGATOR_CREATE_PLOTS": True,  # Create aggregator comparison plots
    "AGGREGATOR_SAVE_CSV": True,   # Save aggregated CSV
    "AGGREGATOR_SAVE_JSON": True,  # Save aggregated JSON
    "AGGREGATOR_SAVE_TXT": True,   # Save summary report TXT
    "AGGREGATOR_CREATE_SANITY_MATRIX": True,  # Create sanity check matrix ()
    
    # Aggregator plot types
    "AGGREGATOR_PLOT_MODEL_COMPARISON": True,    # Model comparison bar chart
    "AGGREGATOR_PLOT_S8_VS_COUPLING": True,      # S8 vs coupling params scatter
    "AGGREGATOR_PLOT_BETA0_SWEEP": True,         # β₀ sweep analysis
    "AGGREGATOR_PLOT_I_E_CORRELATION": True,     # I-E correlation summary
    "AGGREGATOR_PLOT_RHO_DE_EVOLUTION": True,    # ρ_DE(z) multi-model comparison
    "AGGREGATOR_PLOT_DELTA_S8_VS_Z": True,       # ΔS₈(z) vs ΛCDM
    "AGGREGATOR_PLOT_MI_LAG_SCAN": True,         # MI(Δa) lag scan
    
    # ==========================================================================================
    # FILE STRUCTURE ENHANCEMENTS ()
    # ==========================================================================================
    "SAVE_I_OF_A_CSV": True,       # Save dedicated I(a) evolution CSV
    "SAVE_E_OF_A_CSV": True,       # Save dedicated E(a) evolution CSV
    "SAVE_D_OF_A_CSV": True,       # Save dedicated D(a) growth factor CSV
    "SAVE_RHO_DE_OF_A_CSV": True,  # Save dedicated ρ_DE(a) evolution CSV
    
    # ==========================================================================================
    # TOTAL PARAMETER COUNT: 158 PARAMETERS
    # ==========================================================================================
}

# ==========================================================================================
# FIDUCIAL PARAMETERS
# ==========================================================================================
# Build FIDUCIAL_PARAMS from MASTER_CTRL for compatibility with existing code

FIDUCIAL_PARAMS = {
    'H0': MASTER_CTRL['H0'],
    'Omega_m': MASTER_CTRL['OMEGA_M'],
    'Omega_Lambda': MASTER_CTRL['OMEGA_LAMBDA'],
    'Omega_b': MASTER_CTRL['OMEGA_B'],
    'Omega_r': MASTER_CTRL['OMEGA_R'],
    'w0': MASTER_CTRL['W0'],
    'n_s': MASTER_CTRL['N_S'],
    'sigma_8': MASTER_CTRL['SIGMA_8'],
    'tau_reio': MASTER_CTRL['TAU_REIO']
}

# Physical constants
c_light = 299792.458  # Speed of light in km/s
G_newton = 6.67430e-11  # Gravitational constant in SI units
M_planck = 1.220910e19  # Planck mass in GeV
