# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# config.py - Master Control Panel
# ==========================================================================================
# TQE Heisenberg Modular: All configurable parameters
# ==========================================================================================

MASTER_CTRL = {
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    CORE SIMULATION CONTROLS                    ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    # === REPRODUCIBILITY ===
    "SEED": None,                       # Master random seed (None = auto-generate random seed each run)
    
    # === ENSEMBLE SIZE ===
    "N_ENSEMBLE": 100,                  # Number of initial quantum states (tune for speed/quality)
    
    # === QUANTUM SYSTEM ===
    "N_HILB": 20,                       # Fock space truncation per mode (balanced for speed/accuracy)
    "OMEGA_1": 1.0,                     # Mode-1 frequency
    "OMEGA_2": 0.8,                     # Mode-2 frequency (if two-mode)
    "HBAR": 1.0,                        # Reduced Planck constant (natural units)
    
    # === TIME EVOLUTION ===
    "T_FINAL": 12.0,                    # Total evolution time
    "N_T": 300,                         # Number of time points (balanced resolution)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    QUANTUM SYSTEM FEATURES                     ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "ANHARMONIC_X4": True,              # Add λx⁴ term
    "DOUBLE_WELL": False,               # Replace x⁴ by symmetric double-well
    "TWO_MODE_COUPLING": True,          # Second oscillator + coupling
    "TIME_DEP_DRIVE": True,             # H(t) drive term
    "THERMAL_BATH": True,               # Thermal Lindblad at nth > 0
    "TRAJECTORIES": False,              # Use mcsolve instead of mesolve
    "DYNAMIC_LOCKIN": True,             # Adapt rates/potential online using f(E,I)
    
    # === NONLINEARITIES / POTENTIALS ===
    "LAM_X4": 0.02,                     # Strength of x⁴
    "DW_C2": -0.5,                      # Double-well quadratic coeff (negative)
    "DW_C4": 0.02,                      # Double-well quartic coeff (positive)
    "G_COUP": 0.05,                     # Two-mode coupling strength
    
    # === TIME-DEPENDENT DRIVE ===
    "DRIVE_AMP": 0.08,                  # Drive amplitude
    "DRIVE_OMEGA": 0.7,                 # Drive frequency
    
    # === OPEN-SYSTEM RATES (PRE-LAW BASELINE) ===
    "GAMMA_PHI_1": 0.08,                # Dephasing for mode-1
    "KAPPA_1": 0.06,                    # Amplitude damping for mode-1
    "NTH_1": 0.5,                       # Thermal photons mode-1
    
    "GAMMA_PHI_2": 0.06,                # Dephasing for mode-2
    "KAPPA_2": 0.05,                    # Amplitude damping for mode-2
    "NTH_2": 0.3,                       # Thermal photons mode-2
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    TQE LOCK-IN PARAMETERS                      ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "BETA_A": 2.0,                      # I ~ Beta(a,b) shape parameter a
    "BETA_B": 2.0,                      # I ~ Beta(a,b) shape parameter b
    "EC": 25.0,                         # Goldilocks energy center
    "SIGMA": 8.0,                       # Stability window width
    "ALPHA": 0.8,                       # Information bias strength
    
    # === DYNAMIC LOCK-IN (SCENARIO 2) ===
    "N_SEGMENTS": 12,                   # Number of segmented evolution steps (increased for smoother adaptation)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    HEISENBERG UNCERTAINTY LIMIT                ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "HEISENBERG_LIMIT_ACTIVE": True,    # Enforce Heisenberg uncertainty principle explicitly
    "DELTA_X_MIN": 0.5,                 # Minimum position uncertainty (ℏ=1 units)
    "DELTA_P_MIN": 0.5,                 # Minimum momentum uncertainty (ℏ=1 units)
    "UNCERTAINTY_PRODUCT_MIN": 0.5,     # Minimum Δx·Δp (theoretical minimum: ℏ/2 = 0.5)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    INFORMATION ORIGIN MODELS                   ║
    # ╚════════════════════════════════════════════════════════════════╝
    # NOTE: ALL 3 models (emergent, inherent, threshold) run automatically!
    #       This parameter is for reference/override only (emergent used for main stats)
    
    "I_ORIGIN_MODE": "emergent",        # Reference mode (all 3 tested automatically)
    
    # === EMERGENT I (spontaneous from fluctuations) ===
    "I_EMERGENT_ALPHA": 0.3,            # Weight for |ΔE_t| contribution
    "I_EMERGENT_BETA": 0.2,             # Weight for autocorrelation contribution
    "I_EMERGENT_GAMMA": 0.95,           # Decay factor (I persistence)
    
    # === INHERENT I (deterministic function of E) ===
    "I_INHERENT_MODE": "log",           # "log" | "power" | "linear"
    "I_INHERENT_E0": 10.0,              # Reference energy for log mode
    "I_INHERENT_GAMMA": 0.5,            # Exponent for power mode
    "I_INHERENT_SCALE": 0.05,           # Scale factor
    
    # === THRESHOLD I (activated above critical energy) ===
    "I_THRESHOLD_EC": 15.0,             # Critical energy threshold
    "I_THRESHOLD_SLOPE": 0.1,           # Growth rate above threshold
    "I_THRESHOLD_MAX": 1.0,             # Maximum I value
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    PARAMETER SWEEP CONTROL                     ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "ENABLE_PARAMETER_SWEEP": False,    # Run parameter sweep analysis
    "SWEEP_VARIABLE": "EC",             # "EC" | "SIGMA" | "ALPHA"
    "SWEEP_VALUES": [15.0, 20.0, 25.0, 30.0, 35.0],  # Values to sweep
    "SWEEP_N_ENSEMBLE": 200,            # Reduced ensemble for sweep (speed)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    CONTROL / BENCHMARK MODELS                  ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "ENABLE_CONTROL_DECOHERENCE": True, # Run pure decoherence control (no lock-in)
    "ENABLE_PLANCK_BENCHMARK": False,   # Compare fluctuation scales to Planck/BBN (requires external data)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    INITIAL STATE SAMPLING                      ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "COHERENT_LOG_MEAN": 1.3,           # Lognormal mean for coherent state amplitude
    "COHERENT_LOG_SIGMA": 0.55,         # Lognormal sigma (controls heavy tail)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    PERFORMANCE OPTIMIZATION                    ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "USE_MULTIPROCESSING": False,       # DISABLED: run_single uses 273 global variables (not MP-safe)
    "MAX_WORKERS": None,                # (Not used - serial execution only)
    "CACHE_TENSOR_OPS": True,           # Pre-compute tensor operators (2-3x speedup in serial mode)
    "MEMORY_EFFICIENT": False,          # If True, don't store all states (less memory, slower)
    
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    OUTPUT & VISUALIZATION                      ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    "BASE_FOLDER_NAME": "TQE_Heisenberg_Fluctuation",
    "PLOT_DPI": 300,                    # Figure DPI for high-quality output
    "PLOT_FONTSIZE_TITLE": 14,
    "PLOT_FONTSIZE_LABEL": 12,
    "PLOT_FONTSIZE_LEGEND": 10,
}

