# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# MASTER_CTRL Configuration Dictionary
# Extracted from TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py
# Includes Planck-aware validation controls, gradient-based fine-tuning knobs, and
# reproducible Goldilocks targeting identical to the monolithic pipeline.
#
import os
import numpy as np

MASTER_CTRL = {
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    CORE PIPELINE CONTROLS                      ║
    # ║          (Main settings - adjust these for basic runs)         ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    # === EXECUTION MODE (DEPRECATED - use RUN_MODE instead) ===
    "PIPELINE_VARIANT":      "full",        # "full" (E+I) | "energy_only" (E-only) - auto-set by RUN_MODE
    
    # === RUN MODE SELECTION ===
    "RUN_MODE":              "single_ei",   # "single_eonly" | "single_ei" | "batch_ei" | "batch_all"
    
    # === I-PARAMETER ===
    "I_DEFINITION_MODE":     "jensen_shannon",  # Active I-definition (used if RUN_MODE = "single_ei")
    # Available I-definitions: kl_divergence, shannon, renyi, mutual_info, composite, 
    #                          kl_shannon, entanglement, fisher, fisher_kl_fusion, jensen_shannon (10 total)
    
    # === COMPLEXITY & LIFE-COMPATIBILITY ANALYSIS ===
    "ENABLE_COMPLEXITY_ANALYSIS": True,     # Compute complexity & life-compatibility metrics
    "SAVE_COMPLEXITY_PLOTS": True,          # Persist complexity/life visualizations
    "COMPLEXITY_TOP_N": 10,                 # Number of top universes exported in ranking
    "COMPLEXITY_THRESHOLD": 60.0,           # Minimum complexity score considered “high”
    "LIFE_COMPATIBILITY_THRESHOLD": 60.0,   # Minimum life-compatibility score considered “favorable”
    
    # === GOLDILOCKS - BAYESIAN ADAPTIVE OPTIMIZATION (ONLY METHOD) ===
    # Bayesian Adaptive Sampling uses Gaussian Process Regression to intelligently
    # find the optimal Goldilocks zone with minimal universe samples.
    # 
    # Budget Allocation Strategy:
    #   - 30% of NUM_UNIVERSES used for Bayesian Goldilocks discovery (exploration)
    #   - 70% of NUM_UNIVERSES used for full simulation in discovered zone (exploitation)
    #   Example: NUM_UNIVERSES=1000 → 300 (Bayesian) + 700 (full sim) = 1000 total
    # 
    # Bayesian Optimization Control Parameters:
    "CALIBRATION_EPOCHS": 500,                     # Epochs per stability check during Bayesian sampling
    "BAYESIAN_UCB_KAPPA": 2.0,                     # Exploration-exploitation trade-off (higher = more exploration)
                                                   # Recommended: 1.5 (aggressive), 2.0 (balanced), 3.0 (conservative)
    "BAYESIAN_GP_NOISE": 0.01,                     # GP noise level (robustness to noisy stability measurements)
                                                   # Recommended: 0.005 (low noise), 0.01 (balanced), 0.05 (high noise)
    
    # === UNIVERSE SAMPLING ===
    "NUM_UNIVERSES":         300,       # PHASE 1-28 universes (main simulation) - kompromisszum: gyors + elég statisztika
                                        # 30% Bayesian (90) + 70% full sim (210) - jobb esély a Planck értékekhez
    "SEED":                  None,     # Master seed (None = auto)
    
    # === EPOCH SETTINGS ===
    "TIME_STEPS":            1000,     # Stability run epochs
    "LOCKIN_EPOCHS":         500,      # Lock-in dynamics epochs
    "EXPANSION_EPOCHS":      1000,     # Expansion dynamics epochs
    "FL_EXP_EPOCHS":         2000,     # Fluctuation expansion panel epochs
    
    # === COUPLING PARAMETERS ===
    "X_SCALE":               20.0,     # E-I coupling scale
    "ALPHA_I":               0.9,      # I coupling strength
    "X_MODE":                "E_plus_I",  # "E_plus_I" | "product" | "E_times_I_pow"
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                   DETAILED CONFIGURATION                       ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    # === PERFORMANCE OPTIMIZATION ===
    "USE_MULTIPROCESSING":        True,      # Enable parallel universe simulation
    "MAX_WORKERS":                None,      # Worker count (None = auto)
    "ENABLE_CMB_CACHE":           True,      # Cache CMB maps
    "CMB_CACHE_SIZE":             1000,      # LRU cache size
    "STABILITY_EARLY_STOP":       True,      # Stop when converged
    "STABILITY_TOLERANCE":        0.02,      # Convergence tolerance
    "PERFORMANCE_MODE":           "balanced", # Performance mode
    # ====================================

    # --- Physical model parameters ---
    "USE_PHYSICAL_MODEL":         True,    # Enable physics-based E, I, CMB
    "USE_ENHANCED_PHYSICS":       True,    # Enable enhanced physics
    "CAMB_INTEGRATION":           True,    # Use CAMB for realistic CMB power spectra
    # NOTE: Set to False to let anomalies emerge naturally from CMB generation (no artificial addition)
    # If True, physical anomalies (cosmic strings, domain walls, etc.) are artificially added
    "ENABLE_PHYSICAL_ANOMALIES":  False,   # DISABLED: Anomalies should emerge naturally, not be added artificially
    "RUN_PLANCK_VALIDATION":      True,    # Run Planck validation
    "PLANCK_DATA_PATH": os.path.join("planck_data", "COM_PowerSpect_CMB-TT-full_R3.01.txt"),  # Relative path to Planck C_ell
    "PLANCK_AUTO_DOWNLOAD":       True,    # Attempt to download Planck TT spectrum if missing
    "PLANCK_DATA_URL":            "https://raw.githubusercontent.com/cmbant/CAMB/master/planck/COM_PowerSpect_CMB-TT-full_R3.01.txt",
    "PLANCK_GENERATE_IF_MISSING": True,    # Generate surrogate Planck spectrum if download fails
    "PLANCK_ALLOW_CAMB_SYNTHESIS": True,   # Use CAMB (if available) for surrogate generation
    "PLANCK_SYNTHETIC_LMAX":      2500,    # Maximum ell when synthesizing surrogate spectrum
    "PLANCK_TARGET_SIGMA":        0.05,    # Target lognormal sigma for E around Planck value
    "PLANCK_TUNING_DELTA":        0.08,    # +/- window for truncating E samples around Planck value
    "PLANCK_PRIOR_SIGMA":         0.03,    # Gaussian prior width on E during Planck validation
    "PLANCK_PRIOR_WEIGHT":        1.0,     # Weight multiplier for the E prior term in χ²
    "PLANCK_AMPLITUDE_CALIBRATION": True,  # Fit a global amplitude for each CMB map before χ²
    # --- Planck-aligned emergent fine-tuning controls ---
    "ENABLE_PLANCK_FINE_TUNING":  True,    # Bias sampling toward Planck attractor without hard constraints
    "PLANCK_TARGET_E":            0.7619,  # Desired emergent Omega_Lambda
    "PLANCK_TARGET_I":            0.1309,  # Desired emergent information/horizon entropy
    "PLANCK_TARGET_ALPHA":        9.146,   # Desired emergent amplitude calibration factor
    "PLANCK_TARGET_CHI2_PER_DOF": 59.742,  # Desired emergent χ²/dof
    "PLANCK_FINE_TUNE_WIDTH_E":   0.026115,   # Gaussian width for E-attractor bias (refined)
    "PLANCK_FINE_TUNE_WIDTH_I":   0.035598,   # Coupled width mapping E deviations into I bias (refined)
    "PLANCK_FINE_TUNE_STRENGTH_E": 0.414364,  # Mixing weight toward target E (gentle gradient pull)
    "PLANCK_FINE_TUNE_STRENGTH_I": 0.517883,  # Mixing weight toward target I (gentle gradient pull)
    "PLANCK_FINE_TUNE_STRENGTH_ALPHA": 0.0,   # Amplitude fine-tune handled via validation stage
    "PLANCK_FINE_TUNE_JITTER_E":  0.100664,   # Residual jitter to keep stochasticity in E sampling
    "PLANCK_FINE_TUNE_JITTER_I":  0.014951,   # Residual jitter on I around attractor
    "PLANCK_FINE_TUNE_JITTER_ALPHA": 0.03,    # Fractional jitter on map amplitude scaling
    "PLANCK_AMPLITUDE_TARGET_SCALE": 0.1093,  # ≈1/α_target, applied when near attractor

    # --- Cosmological parameters ---
    "OMEGA_M":                    0.3153,   # Matter density fraction
    "OMEGA_B":                    0.0493,   # Baryon density fraction
    "OMEGA_LAMBDA":               0.6847,   # Dark energy density fraction
    "H0":                         67.36,    # Hubble constant (km/s/Mpc)
    "T_CMB":                      2.7255,   # CMB temperature (K)
    "N_EFF":                      3.046,    # Effective neutrino species
    "Y_HE":                       0.2453,   # Helium mass fraction
    
    # --- Real-world physics constants ---
    "C_LIGHT": 299792458.0,  # Speed of light (m/s)
    "G_NEWTON": 6.67430e-11,  # Gravitational constant (m³/kg/s²)
    "K_BOLTZMANN": 1.380649e-23,  # Boltzmann constant (J/K)
    "H_PLANCK": 6.62607015e-34,  # Planck constant (J⋅s)
    "H_BAR": 1.054571817e-34,  # Reduced Planck constant (J⋅s)
    "E_CHARGE": 1.602176634e-19,  # Elementary charge (C)
    "M_PROTON": 1.67262192369e-27,  # Proton mass (kg)
    "M_ELECTRON": 9.1093837015e-31,  # Electron mass (kg)
    "ALPHA_EM": 1/137.035999084,  # Fine structure constant
    "ALPHA_S": 0.1181,  # Strong coupling constant at MZ
    
    # --- Real-world cosmological observations ---
    "PLANCK_2018_OMEGA_M": 0.3153,  # Planck 2018 matter density
    "PLANCK_2018_OMEGA_B": 0.0493,  # Planck 2018 baryon density
    "PLANCK_2018_OMEGA_LAMBDA": 0.6847,  # Planck 2018 dark energy density
    "PLANCK_2018_H0": 67.36,  # Planck 2018 Hubble constant
    "PLANCK_2018_SIGMA8": 0.8102,  # Planck 2018 σ₈
    "PLANCK_2018_NS": 0.9649,  # Planck 2018 scalar spectral index
    "PLANCK_2018_AS": 2.100e-9,  # Planck 2018 scalar amplitude
    "PLANCK_2018_TAU": 0.0544,  # Planck 2018 optical depth
    
    # --- Real-world particle physics parameters ---
    "NEUTRINO_MASS_SUM": 0.12,  # Sum of neutrino masses (eV) (Planck 2018)
    "DARK_MATTER_MASS": 100.0,  # Typical WIMP mass (GeV)
    "INFLATION_SCALE": 1e16,  # Inflation energy scale (GeV)
    "REHEATING_TEMPERATURE": 1e15,  # Reheating temperature (GeV)
    "BARYOGENESIS_CP_VIOLATION": 1e-10,  # CP violation parameter

    # --- Energy distribution ---
    "E_DISTR":              "lognormal", # energy sampling mode (future-proof)
    "E_LOG_MU":             np.log(0.7619),  # lognormal mean centered on Planck target (≈-0.272)
    "E_LOG_SIGMA":          0.15,      # lognormal sigma (csökkentve: szűkebb eloszlás, mert E túl magas volt)
    "E_TRUNC_LOW":          0.65,      # optional post-sample clamp (low) (emelve: ne legyen túl alacsony)
    "E_TRUNC_HIGH":         0.80,      # optional post-sample clamp (high) (csökkentve: E túl magas volt, max 0.80)
    
    # --- Physical E parameter interpretation ---
    "E_COSMOLOGICAL_PARAM": "Omega_Lambda",  # Physical interpretation
    "E_OBS_VALUE": 0.7619,  # Planck-aligned reference
    "E_EXPLORATION_SIGMA": 0.10,  # Tighter exploration around observed value (csökkentve: szűkebb mintavétel, E túl magas volt)

    # --- Information parameter I controls ---
    # ENHANCEMENT B: Add I_DEFINITION_MODE to select the physical basis for I.
    "I_DIM":                8,         # Hilbert space dimension for random kets
    "I_ENTANGLEMENT_SUBSYS_DIM": 4, # Subsystem dimension for entanglement entropy (total dim = d*d)
    "KL_EPS":               1e-12,     # numerical epsilon for KL/entropy
    "INFO_FUSION_MODE":      "weighted", # "product" | "weighted"
    "INFO_WEIGHT_KL":        0.4,      # used if INFO_FUSION_MODE == "weighted"
    "INFO_WEIGHT_SHANNON":   0.6,      # used if INFO_FUSION_MODE == "weighted"
    "I_EXPONENT":            1.0,      # optional nonlinearity: I <- I**I_EXPONENT
    "I_MIN_EPS":             0.0,      # clamp floor for I (avoid exact zeros)

    # --- E–I coupling (X definition) ---
    "X_I_POWER":             1.0,             # if "E_times_I_pow": X = E * (I ** X_I_POWER)

    # --- Fluctuation / superposition module toggles & params ---
    "RUN_FLUCTUATION_BLOCK": True,  # Show the t<0 superposition, t=0 collapse, and t>0 expansion panels.
    "RUN_QUANTUM_FLUCT":     True,  # Generate the standalone quantum-fluctuation time-series panel.
    "FL_SUPER_T":            10.0,    # duration for t<0 superposition plot (arb. units)
    "FL_SUPER_DT":           0.05,    # time step for superposition time series
    "FL_SUPER_DIM":          4,       # small Hilbert dim for toy density evolution
    "FL_SUPER_NOISE":        0.03,    # Noise amplitude (reduced for smoother dynamics)
    "FL_SUPER_KICK":         0.07,    # Random unitary kick strength (reduced for cleaner signal)
    "FL_FLUCT_OBS":          "Z",     # Observable for quantum fluctuation panel ("Z", "X", or "rand").
    "FL_FLUCT_T":            6.0,     # Duration for quantum fluctuation panel
    "FL_FLUCT_DT":           0.02,    # Time step for quantum fluctuation panel
    "FL_SUPER_OBS_JITTER":   0.03,    # Observable jitter

    "FL_COLLAPSE_T_PRE":     0.22,    # window before t=0 (collapse)
    "FL_COLLAPSE_T_POST":    0.22,    # window after t=0
    "FL_COLLAPSE_DT":        0.002,   # time step
    "FL_COLLAPSE_PRE_SIGMA": 0.55,    # volatility before t=0
    "FL_COLLAPSE_POST_SIGMA":0.015,   # small jitter after t=0
    "FL_COLLAPSE_REVERT":    0.35,    # mean-reversion towards X_lock after t=0 (OU factor)

    "FL_EXP_DRIFT":          0.45,    # upward drift for A
    "FL_EXP_JITTER":         0.9,     # noise for A random walk
    "FL_EXP_I_JITTER":       0.04,    # small jitter for I track

    # --- Stability thresholds ---
    "REL_EPS_STABLE":             0.035,   # Stability threshold
    "REL_EPS_LOCKIN":             0.040,   # Lock-in threshold
    "CALM_STEPS_STABLE":          4,       # Consecutive calm steps for stability
    "CALM_STEPS_LOCKIN":          3,       # Consecutive calm steps for lock-in
    "MIN_LOCKIN_EPOCH":           120,     # Minimum epoch for lock-in
    "LOCKIN_WINDOW":              6,       # Rolling window size
    "LOCKIN_ROLL_METRIC":         "mean",  # Aggregator over window
    "LOCKIN_REQUIRES_STABLE":     True,    # Require stable_at before lock-in
    "LOCKIN_MIN_STABLE_EPOCH":    10,      # Minimum epochs after stable_at
    
    # --- I-definition specific lock-in thresholds ---
    "I_DEFINITION_LOCKIN_THRESHOLDS": {  # Custom thresholds for specific I-definitions (if needed)
        # All definitions use default REL_EPS_LOCKIN (0.030) - balanced for ~30 lock-ins
    },

    # --- Target Universe Distribution Control ---
    "TARGET_UNSTABLE_RATE":  0.50,
    "TARGET_STABLE_RATE":    0.50,
    "TARGET_LOCKIN_RATE":    0.50,
    "ADJUST_STABILITY_THRESHOLDS": False,  # Physics-driven: let TQE dynamics determine distribution
    "STABILITY_ADJUSTMENT_FACTOR": 0.1,  # How much to adjust thresholds per iteration (0.1 = 10%)
    "MAX_STABILITY_ADJUSTMENTS": 10,     # Maximum number of threshold adjustment iterations

    # --- Goldilocks zone controls (integrated into PHASE 1) ---
    "STAB_BINS":             40,         # number of bins in stability curve
    "SPLINE_K":              3,          # spline order for smoothing (3=cubic)
    "GOLDILOCKS_THRESHOLD":  0.15,       # stability threshold for peak detection (balanced - 15%)
    "GOLDILOCKS_MARGIN":     0.12,       # margin for window boundaries (balanced)
    "SIGMA_ALPHA":           1.5,        # curvature inside Goldilocks (sigma shaping)
    "OUTSIDE_PENALTY":       5.0,        # sigma multiplier outside Goldilocks zone

    # --- I-parameter definition comparison (new controls) ---
    "COMPUTE_ALL_I_DEFINITIONS": True, # Enable CSV export for all 10 I-definitions
    "I_DEFINITIONS_SAMPLE_POINTS": 50,  # Sample points for a-grid if needed
    
    # --- Single Run E+I Multi-Definition Mode ---
    "SINGLE_RUN_SELECTED_I_DEFINITION": "kl_shannon",  # If SINGLE_RUN_ALL_I_DEFINITIONS=False, use this I-definition only

    # --- BAYESIAN MODEL SELECTION ---
    "ENABLE_BAYESIAN_ANALYSIS": True,      # Enable Bayesian Model Selection (BIC, AIC, Bayes Factor)
    "ENABLE_NESTED_SAMPLING": True,        # Enable Nested Sampling for Bayesian Evidence
    "NESTED_SAMPLING_NLIVE": 1000,
    "NESTED_SAMPLING_DLOGZ": 0.1,
    "NESTED_SAMPLING_MAX_ITER": 10000,     # Maximum iterations for nested sampling
    "ENABLE_CORNER_PLOTS": True,           # Enable corner plots for parameter posteriors
    "BAYESIAN_PRIOR_X_SCALE": (10.0, 50.0),  # Uniform prior range for X_SCALE
    "BAYESIAN_PRIOR_ALPHA_I": (0.1, 2.0),    # Uniform prior range for ALPHA_I
    "BAYESIAN_PARALLEL": False,            # Parallel nested sampling (requires more memory)

    # --- CMB-Calibrated Goldilocks controls (Enhancement A) ---
    "CMB_CALIB_NUM_UNIVERSES": 2000, # Number of universes for the pre-run (can be smaller for speed)
    "CMB_CALIB_QUALITY_WEIGHTS": {"r2": 0.5, "gaussianity": 0.25, "isotropy": 0.25},

    # --- Noise shaping (lock-in loop) ---
    "EXP_NOISE_BASE":        0.12,    # baseline noise for updates (sigma0)
    "LL_BASE_NOISE":         8e-4,    # absolute noise floor (never go below this)
    "NOISE_DECAY_TAU":       500,     # e-folding time for noise decay (epochs)
    "NOISE_FLOOR_FRAC":      0.25,    # fraction of initial sigma preserved by decay
    "NOISE_COEFF_A":         1.0,     # per-variable noise multiplier (A)
    "NOISE_COEFF_NS":        0.10,    # per-variable noise multiplier (ns)
    "NOISE_COEFF_H":         0.20,    # per-variable noise multiplier (H)

    # --- Expansion dynamics (if/when used) ---
    "EXP_GROWTH_BASE":       1.005,   # baseline exponential growth rate
    # (EXP_NOISE_BASE above is reused as expansion amplitude baseline)
    # TWEAK: Centralize magic number for expansion starting amplitude.
    "FL_EXP_START_AMPLITUDE": 20.0,

    # --- Best-universe visualization (by category) ---
    "BEST_UNIVERSE_FIGS_LOCKIN": 3,         # Number of figures for top lock-in universes
    "BEST_UNIVERSE_FIGS_STABLE": 3,         # Number of figures for top stable-only universes (stable but no lock-in)
    "BEST_UNIVERSE_FIGS_UNSTABLE": 3,       # Number of figures for top unstable universes
    "BEST_N_REGIONS": 8,            # number of region-level entropy traces
    "BEST_STAB_THRESHOLD": 3.5,             # horizontal reference line on plots
    "BEST_SAVE_CSV": True,          # also export per-universe time series as CSV
    "BEST_SEED_OFFSET": 777,        # reproducible offset for the synthetic entropy generator
    "BEST_MAX_FIGS": 50,       # safety clamp

    # --- Best-universe: phase-change knobs ---
    "BEST_SIGMA_PRE":   0.065,   # Stronger noise before lock-in
    "BEST_SIGMA_POST":  0.012,   # Weaker noise after lock-in
    "BEST_SMOOTH_PRE":  8,       # Smaller smoothing window before lock-in
    "BEST_SMOOTH_POST": 36,      # Larger smoothing window after lock-in
    "BEST_SIGMA_DECAY_TAU": 250, # Time constant (steps) for how fast the noise decays/cleans up

    # --- Noise / smoothing knobs for entropy evolution ---
    "BEST_REGION_MU": 5.1,          # Target mean for region entropy traces
    "BEST_REGION_SIGMA": 0.01,          # Noise amplitude for region traces (lower = smoother)
    "BEST_GLOBAL_JITTER": 0.005,        # Small jitter added to the global entropy curve
    "BEST_SMOOTH_WINDOW": 30,           # Rolling average window size (>=1, 1 = disabled)
    "BEST_SHOW_REGIONS": True,          # If False, only plot the global entropy curve
    "BEST_ANNOTATE_LOCKIN": True,       # Draw vertical lock-in marker and annotation text
    "BEST_ANNOTATION_OFFSET": 3,            # Horizontal offset for annotation text placement

    # TWEAK: Centralize magic numbers for entropy evolution.
    # --- Entropy evolution knobs ---
    "BEST_ENTROPY_BASE": 5.6,
    "BEST_ENTROPY_SCALE": 0.45,
    "BEST_ENTROPY_DECAY_DIV": 6,

    # --- Extra robustness / docs ---
    "STAB_MIN_COUNT":        10,      # Minimum samples required in a stability bin; bins with fewer are ignored.
    "REGRESSION_MIN":        10,      # Minimum number of lock-in cases required to train/evaluate the regression.
    "MAX_SHAP_SAMPLES":      1000,    # Upper limit on samples used for SHAP plotting to keep it fast and stable.
    "SHAP_BACKGROUND_SIZE": 200,     # Size of the SHAP background (reference) dataset for model-agnostic explainers.

    # --- CMB best-universe map generation ---
    "CMB_BEST_ENABLE":            True,      # Enable CMB generation
    "CMB_BEST_SEED_OFFSET":       909,       # Seed offset
    "CMB_BEST_MODE":              "healpix", # Backend mode

    # --- CMB map parameters ---
    "CMB_NSIDE":                  128,       # HEALPix resolution parameter
    "CMB_NPIX": 64,            # Reduced pixel count for Colab (was 128)
    "CMB_PIXSIZE_ARCMIN": 3.0,         # Larger pixel size for efficiency
    "CMB_POWER_SLOPE": 2.5,        # Power spectrum slope (Pk ~ k^-slope)
    "CMB_SMOOTH_FWHM_DEG": 0.5,       # Slightly larger smoothing for efficiency
    "CMB_AMPLITUDE_SCALE": 5.47e-11,   # Overall amplitude of CMB fluctuations
    
    # --- CMB Cold Spot Physics Parameters (EMERGENT!) ---
    # NOTE: Set to False to let cold spots emerge naturally from CMB generation (no artificial addition)
    # If True, cold spots are artificially added to the map (NOT recommended for genuine TQE validation)
    "CMB_COLDSPOT_PHYSICS_ENABLE": False,  # DISABLED: Cold spots should emerge naturally, not be added artificially
    "CMB_COLDSPOT_PROBABILITY": 0.15,  # Base probability of cold spot per universe
    "CMB_COLDSPOT_DEPTH_CENTER": -70.0,  # Target depth center (µK) - peak should be near this (Planck reference)
    "CMB_COLDSPOT_DEPTH_SPREAD": 35.0,  # Spread/width of depth distribution (µK) - larger = more spread
    "CMB_COLDSPOT_DEPTH_MIN": -120.0,  # Minimum depth (µK) - deepest possible cold spot
    "CMB_COLDSPOT_DEPTH_MAX": -30.0,  # Maximum depth (µK) - shallowest cold spot
    "CMB_COLDSPOT_AMPLITUDE_FACTOR": 1.0,  # Multiplier for final depth (1.0 = use depth directly, >1.0 = deeper)
    "CMB_COLDSPOT_SCALE_FACTOR": 1.8,  # Angular size multiplier (larger = bigger cold spots)
    # Legacy parameter (kept for backward compatibility, but not used if new params are set)
    "CMB_COLDSPOT_DEPTH_RANGE": (-400, -60),  # DEPRECATED: use DEPTH_CENTER/SPREAD/MIN/MAX instead

    # --- CMB cold-spot detector ---
    "CMB_COLD_ENABLE":            True,            # Enable cold-spot detector
    "CMB_COLD_TOPK":              5,               # Top-K cold spots per universe
    "CMB_COLD_SIGMA_ARCMIN":      [180, 360],      # Gaussian smoothing scales (arcmin)
    "CMB_COLD_MIN_SEP_ARCMIN":    30,              # Minimum separation (arcmin)
    "CMB_COLD_Z_THRESH":          -0.5,            # Z-score threshold
    "CMB_COLD_SAVE_PATCHES":      False,           # Save cutout patches
    "CMB_COLD_PATCH_SIZE_ARCMIN": 200,             # Patch size (arcmin)
    "CMB_COLD_MODE":              "healpix",       # Backend mode
    "CMB_COLD_OVERLAY":           True,            # Draw markers on maps
    "CMB_COLD_MAX_OVERLAYS":      3,               # ENABLED: Generate top 3 overlay PNGs + aggregate density map
    "CMB_COLD_REF_Z":             -70.0,           # Planck cold spot reference
    "CMB_COLD_UK_THRESH":         -70.0,           # Temperature threshold (µK)
    "PLANCK_COLDSPOT_REFERENCE":   -70.0,                  # Planck cold spot reference for calibration

    # --- CMB Axis-of-Evil detector ---
    "CMB_AOE_ENABLE":             True,      # Enable AOE detector
    "CMB_AOE_LMAX":               3,         # Maximum multipole ℓ
    "CMB_AOE_NREALIZ":            20,        # Monte Carlo realizations
    "CMB_AOE_OVERLAY":            True,      # Overlay axes on maps
    "CMB_AOE_MODE":               "healpix", # Backend mode
    "CMB_AOE_SEED_OFFSET":        909,       # Seed offset
    "CMB_AOE_MAX_OVERLAYS":       3,         # ENABLED: Generate top 3 overlay PNGs + aggregate density map
    "CMB_AOE_PHASE_LOCK":         False,     # Force ℓ=2,3 boost
    "CMB_AOE_LMAX_BEST":          128,       # Alm lmax for phase lock
    "CMB_AOE_L23_BOOST":          1.0,       # Boost factor
    "AOE_REF_ANGLE_DEG":          20.0,      # Reference alignment angle
    "AOE_P_THRESHOLD":            0.10,      # P-value threshold
    "AOE_ALIGN_THRESHOLD":   0.5,     # fallback if only angle is present (score = 1 - angle/180)

    # --- Statistical Finetuning Detector ---
    "RUN_STATISTICAL_FINETUNING_DETECTOR": True,   # Enable finetuning detector
    "FT_EPS_EQ":                          0.5,     # Threshold for E≈I slice

    # --- Feature Importance Detector ---
    "RUN_FEATURE_IMPORTANCE_DETECTOR":    True,    # Enable feature importance
    "FI_RF_N_ESTIMATORS":                 100,     # Random forest tree count
    "FI_TEST_SIZE":                       0.3,     # Test split size

    # --- Emergent Law Detectors ---
    "RUN_EMERGENT_LAW_DETECTORS": True,  # Master switch for all new law-proxy detectors
    
    # --- Advanced Anomaly Detection Parameters ---
    "ENABLE_QUANTUM_ANOMALY_DETECTION": True,     # Quantum field anomalies
    "ENABLE_ENTROPY_ANOMALY_DETECTION": True,     # Entropy fluctuation anomalies
    "ENABLE_TOPOLOGICAL_ANOMALY_DETECTION": True, # Topological defect anomalies
    "ENABLE_ENERGY_ANOMALY_DETECTION": True,      # Energy conservation anomalies
    "ENABLE_INFORMATION_ANOMALY_DETECTION": True, # Information theory anomalies
    "ENABLE_CMB_ANOMALY_DETECTION": True,         # CMB statistical anomalies
    "ENABLE_GRAVITATIONAL_ANOMALY_DETECTION": True, # Gravitational wave anomalies
    "ENABLE_DARK_MATTER_ANOMALY_DETECTION": True, # Dark matter distribution anomalies
    "ENABLE_INFLATION_ANOMALY_DETECTION": True,   # Inflation field anomalies
    "ENABLE_REIONIZATION_ANOMALY_DETECTION": True, # Reionization anomalies
    
    # --- Advanced Law Detection Parameters ---
    "ENABLE_CONSERVATION_LAW_DETECTION": True,    # Conservation laws (energy, momentum, charge)
    "ENABLE_SYMMETRY_LAW_DETECTION": True,        # Symmetry breaking laws
    "ENABLE_SCALING_LAW_DETECTION": True,         # Scaling laws and power laws
    "ENABLE_EMERGENT_LAW_DETECTION": True,        # Emergent behavior laws
    "ENABLE_QUANTUM_LAW_DETECTION": True,         # Quantum mechanical laws
    "ENABLE_THERMODYNAMIC_LAW_DETECTION": True,   # Thermodynamic laws
    "ENABLE_STATISTICAL_LAW_DETECTION": True,     # Statistical mechanics laws
    "ENABLE_FIELD_LAW_DETECTION": True,           # Field theory laws
    "ENABLE_GEOMETRIC_LAW_DETECTION": True,       # Geometric and topological laws
    "ENABLE_INFORMATION_LAW_DETECTION": True,     # Information theory laws

    # --- ENHANCEMENT C: Comparative analysis of I definitions ---
    "I_COMPARATIVE_ANALYSIS": False, # DEPRECATED: Use RUN_MODE = "batch_ei" or "batch_all" instead
    "RUN_MULTI_MODE_GOLDILOCKS": True, # Generate Goldilocks diagrams for all I parameter modes

    # --- Enhanced Physics Parameters ---
    "FRIEDMANN_AGE_CALCULATION": True,     # Enable universe age calculation
    "QUANTUM_FIELD_FLUCTUATIONS": True,    # Enable quantum field analysis
    "COSMIC_ENTANGLEMENT_NETWORK": True,   # Enable entanglement network analysis
    "PHYSICAL_ANOMALY_GENERATION": True,   # Enable physical anomaly generation
    
    # --- Analysis and Visualization Controls ---
    "ENABLE_EI_IMPORTANCE_COMPARISON": True,  # Generate E+I importance comparison plots
    "ENABLE_CMB_ANALYSIS_PLOTS": True,        # Generate CMB analysis plots
    "ENABLE_CMB_ANOMALY_ANALYSIS": True,      # Generate CMB anomaly analysis plots
    "ENABLE_COMPREHENSIVE_CORRELATION": True, # Generate comprehensive correlation analysis
    "ENABLE_ADVANCED_STATISTICAL": True,      # Generate advanced statistical analysis
    "ENABLE_ENHANCED_PHYSICS_ANALYSIS": True, # Generate enhanced physics analysis
    "ENABLE_COMPREHENSIVE_DATA_EXTRACTION": False, # Generate comprehensive data extraction (DISABLED: SeedSequence entropy bug)
    
    # --- Batch Mode Configuration ---
    "MULTI_I_ANALYSIS_MODE": False,  # Set to True when running in batch mode (auto-set by RUN_MODE)
    "MULTI_I_SAVE_DIR": None,        # Parent directory for batch runs (auto-set by RUN_MODE)

    # --- Plotting and Visualization Parameters ---
    "PLOT_FIGSIZE_DEFAULT": (12, 8),
    "PLOT_DPI": 300,
    "PLOT_FONTSIZE_TITLE": 18,
    "PLOT_FONTSIZE_LABEL": 16,
    "PLOT_FONTSIZE_LEGEND": 14,
    "PLOT_ALPHA_DEFAULT": 0.7,              # Default alpha for plots
    "PLOT_LINEWIDTH_DEFAULT": 2.0,          # Default line width
    "PLOT_MARKERSIZE_DEFAULT": 8,           # Default marker size
    "PLOT_GRID_ALPHA": 0.3,                 # Grid transparency
    "PLOT_GRID_LINEWIDTH": 0.5,             # Grid line width
    "PLOT_EDGE_LINEWIDTH": 0.8,             # Axis edge line width
    "PLOT_SAVE_DPI": 180,                   # DPI for saved figures
    "PLOT_COLOR_CYCLE": ['#87CEEB', '#FA8072', '#98FB98', '#DDA0DD', '#F0E68C', '#FFB6C1', '#20B2AA'],
    
    # --- Statistical Analysis Parameters ---
    "STAT_ALPHA_LEVEL": 0.05,               # Significance level for statistical tests
    "STAT_BOOTSTRAP_SAMPLES": 1000,         # Bootstrap samples for confidence intervals
    "STAT_CORRELATION_METHOD": "pearson",   # Correlation method (pearson, spearman, kendall)
    "STAT_REGRESSION_MIN_SAMPLES": 10,      # Minimum samples for regression analysis
    "STAT_OUTLIER_THRESHOLD": 3.0,          # Z-score threshold for outlier detection
    
    # --- Machine Learning Parameters ---
    "ML_RANDOM_STATE": 42,                  # Random state for ML algorithms
    "ML_TEST_SIZE": 0.2,                    # Test set size for train/test split
    "ML_CROSS_VALIDATION_FOLDS": 5,         # Number of CV folds
    "ML_FEATURE_IMPORTANCE_THRESHOLD": 0.01, # Minimum feature importance to display
    "ML_SHAP_SAMPLES": 100,                 # SHAP explanation samples
    
    # --- Data Processing Parameters ---
    "DATA_NUMERICAL_PRECISION": 1e-12,      # Numerical precision for calculations
    "DATA_MISSING_VALUE_THRESHOLD": 0.1,    # Maximum fraction of missing values allowed
    "DATA_OUTLIER_DETECTION_METHOD": "iqr", # Outlier detection method (iqr, zscore)
    "DATA_SMOOTHING_WINDOW": 5,             # Rolling window for data smoothing
    "DATA_INTERPOLATION_METHOD": "linear",  # Interpolation method for missing data
    
    # --- Memory and Performance Parameters ---
    "MEMORY_CLEANUP_FREQUENCY": 10,         # Cleanup memory every N operations
    "MEMORY_WARNING_THRESHOLD": 0.8,        # Memory usage warning threshold
    "PARALLEL_PROCESSING_WORKERS": 4,       # Number of parallel workers
    "CHUNK_SIZE_LARGE_DATASETS": 1000,      # Chunk size for large dataset processing
    "CACHE_SIZE_LIMIT": 1000,               # Maximum cache size in MB
    
    # --- File I/O Parameters ---
    "FILE_COMPRESSION_LEVEL": 6,            # Compression level for saved files (1-9)
    "FILE_BUFFER_SIZE": 8192,               # Buffer size for file operations
    "FILE_TIMEOUT_SECONDS": 30,             # File operation timeout
    "FILE_RETRY_ATTEMPTS": 3,               # Number of retry attempts for failed operations
    "FILE_BACKUP_ENABLED": True,            # Enable automatic backup of important files
    
    # --- Validation and Quality Control ---
    "VALIDATION_ENABLE_CHECKS": True,       # Enable data validation checks
    "VALIDATION_TOLERANCE": 1e-6,           # Numerical tolerance for validation
    "VALIDATION_MAX_ITERATIONS": 1000,      # Maximum iterations for convergence
    "VALIDATION_CONVERGENCE_THRESHOLD": 1e-8, # Convergence threshold
    "QUALITY_CONTROL_ENABLED": True,        # Enable quality control checks
    
    # --- Debugging and Logging Parameters ---
    "DEBUG_LEVEL": "INFO",                  # Debug level (DEBUG, INFO, WARNING, ERROR)
    "LOG_TO_FILE": True,                    # Enable logging to file
    "LOG_FILE_MAX_SIZE": 10,                # Maximum log file size in MB
    "LOG_BACKUP_COUNT": 5,                  # Number of backup log files
    "PROGRESS_BAR_ENABLED": True,           # Enable progress bars
    "PROGRESS_BAR_UPDATE_INTERVAL": 1,      # Progress bar update interval in seconds
    
    # --- Physics Engine Parameters ---
    "PHYSICS_ENGINE_SEED_OFFSET": 1000,     # Seed offset for physics engine
    "FRIEDMANN_INTEGRATION_STEPS": 1000,    # Integration steps for Friedmann equations
    "QUANTUM_FIELD_RESOLUTION": 64,         # Resolution for quantum field calculations
    "ENTANGLEMENT_NETWORK_NODES": 100,      # Number of nodes in entanglement network
    "PHYSICAL_ANOMALY_SCALE": 1.0,          # Scale factor for physical anomalies
    
    # --- CMB Generation Parameters ---
    "CMB_LMAX": 2000,                       # Maximum multipole for CMB generation
    "CMB_NSIDE_PHYSICS": 128,               # NSIDE for physics-based CMB generation
    "CMB_PHYSICS_SEED_OFFSET": 2000,        # Seed offset for CMB physics
    "CMB_SILK_DAMPING_ENABLE": True,        # Enable Silk damping corrections
    "CMB_BAO_ENABLE": True,                 # Enable Baryon Acoustic Oscillations
    "CMB_LENSING_ENABLE": True,             # Enable gravitational lensing
    "CMB_SZ_EFFECT_ENABLE": True,           # Enable Sunyaev-Zel'dovich effect
    
    # --- Analysis Thresholds ---
    "STABILITY_THRESHOLD": 3.5,             # Stability threshold for universe classification
    "LOCKIN_CONFIDENCE_THRESHOLD": 0.60,    # Confidence threshold for lock-in detection (balanced)
    "CORRELATION_SIGNIFICANCE_THRESHOLD": 0.05, # Significance threshold for correlations
    "ANOMALY_DETECTION_THRESHOLD": 3.0,     # Z-score threshold for anomaly detection
    
    # --- Performance Optimization ---
    "ENABLE_VECTORIZATION": True,           # Enable vectorized operations where possible
    "ENABLE_PARALLEL_PROCESSING": True,     # Enable parallel processing for large datasets
    "CHUNK_SIZE_ANALYSIS": 100,             # Chunk size for analysis operations
    "CACHE_INTERMEDIATE_RESULTS": True,     # Cache intermediate results for efficiency
    "OPTIMIZE_MEMORY_USAGE": True,          # Optimize memory usage during processing

    # --- Outputs / IO (optimized for Colab) ---
    "SAVE_FIGS":             True,    # save plots to disk
    "SAVE_JSON":             True,    # save summary JSON
    "DRIVE_BASE_DIR":        "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO",
    "COLAB_OPTIMIZED":       True,    # Enable Colab-specific optimizations
    "REDUCE_MEMORY_USAGE":   True,    # Reduce memory usage for Colab
    "RUN_ID_PREFIX":         "TQE_Universe_Simulation_Full_Pipeline_",      # prefix for run_id
    "RUN_ID_FORMAT":         "%Y%m%d_%H%M%S",        # time format for run_id
    "ALLOW_FILE_EXTS":       [".png", ".fits", ".csv", ".json", ".txt", ".npy"],
    "MAX_FILES_TO_SAVE":     None,    # global cap across all allowed extensions
    "VERBOSE":               False,   # extra prints/logs (OPTIMIZED: True → False, cleaner console!)

    # --- Plot toggles ---
    "PLOT_AVG_LOCKIN":       True,    # plot average lock-in curve
    "PLOT_LOCKIN_HIST":      True,    # plot histogram of lock-in epochs
    "PLOT_STABILITY_BASIC": False,     # simple stability diagnostic plot

    # --- Reproducibility knobs ---
    "USE_STRICT_SEED":       True,    # optionally seed other libs/system for strict reproducibility
    "PER_UNIVERSE_SEED_MODE": "rng" # "rng" | "np_random" — how per-universe seeds are derived
}

