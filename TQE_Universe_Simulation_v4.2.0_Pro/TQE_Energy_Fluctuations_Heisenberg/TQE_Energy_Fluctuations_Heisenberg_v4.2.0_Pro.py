# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# TQE_Energy_Fluctuations_Heisenberg.py
# ==========================================================================================
# TQE Heisenberg Fluctuation Simulation: Law Suppression of Vacuum Fluctuations
# Based on the Theory of the Question of Existence (TQE)
# ==========================================================================================
#
# AUTHOR: Stefan Len
# DATE: 2025-10-31
# VERSION: v4.2.0 PRO
#
# ==========================================================================================
# THEORETICAL FOUNDATION: THEORY OF THE QUESTION OF EXISTENCE (TQE)
# ==========================================================================================
#
# WHY DO STABLE PHYSICAL LAWS PREVENT UNIVERSE-SPAWNING FLUCTUATIONS?
#
# The TQE framework proposes that in a pre-law state (no stable physical constants),
# the Heisenberg Uncertainty Principle permits arbitrarily large vacuum fluctuations.
# Once stable laws emerge and lock in, these same laws SUPPRESS large-scale fluctuations,
# preventing new universes from forming within an existing universe.
#
# CORE TQE HYPOTHESIS:
#   In a "pre-law" state: Large vacuum fluctuations are NOT suppressed
#                        → Universe-spawning magnitudes are possible
#
#   In a "with-law" state: Stable physical constants effectively suppress fluctuations
#                         → Universe-spawning events become impossible
#
# CRITICAL PREDICTION:
#   Quantum systems with NO stabilizing mechanism (no lock-in) will exhibit:
#     • LARGER energy variance over time
#     • HIGHER entropy growth rates
#     • GREATER quantum coherence decay
#     • MORE extreme fluctuation magnitudes
#
#   Systems WITH lock-in (f(E,I) coupling) will exhibit:
#     • SUPPRESSED energy variance
#     • CONTROLLED entropy evolution
#     • STABILIZED quantum states
#     • BOUNDED fluctuation magnitudes
#
# SCIENTIFIC QUESTIONS:
#   1. PRIMARY: Can we demonstrate quantitatively that stable physical laws 
#      (represented by the TQE lock-in mechanism f(E,I)) act as a SUPPRESSION 
#      mechanism for large vacuum fluctuations, thereby explaining why new 
#      universes do not form within our existing cosmos?
#
#   2. INFORMATION ORIGIN: Where does the information parameter I come from?
#      Does I emerge spontaneously from energy fluctuation structure (emergent),
#      is it an inherent property of energy states (inherent I = f(E)), or does
#      it activate only above a critical threshold (threshold I at E > E_c)?
#      Each model tests a different hypothesis about the fundamental origin of
#      information in physical systems.
#
# ==========================================================================================
# PIPELINE OVERVIEW
# ==========================================================================================
#
# This pipeline implements a COMPARATIVE ANALYSIS of quantum vacuum fluctuations
# in two distinct scenarios:
#
# SCENARIO 1: PRE-LAW (NO LOCK-IN)
#   • Pure quantum fluctuations without stabilizing mechanism
#   • Open quantum system with thermal bath, dephasing, damping
#   • No f(E,I) coupling applied
#   • Represents "pre-universe" state where laws have NOT yet formed
#   • Expected outcome: LARGE, UNBOUNDED fluctuations
#
# SCENARIO 2: WITH-LAW (WITH LOCK-IN)
#   • Same quantum system + dynamics
#   • f(E,I) lock-in mechanism ACTIVE
#   • TQE coupling modulates dissipation rates and potential landscape
#   • Represents mature universe with stable physical laws
#   • Expected outcome: SUPPRESSED, BOUNDED fluctuations
#
# QUANTUM SYSTEM FEATURES:
#   • One or two-mode harmonic oscillators (QuTiP implementation)
#   • Anharmonic potential: λx⁴ term (optional double-well)
#   • Two-mode coupling: g(a₁+a₁†)(a₂+a₂†)
#   • Time-dependent drive: H(t) = H₀ + A·cos(ωt)·x
#   • Open-system dynamics: Lindblad master equation
#     - Amplitude damping (energy dissipation)
#     - Dephasing (quantum coherence loss)
#     - Thermal bath (nth > 0)
#   • Optional quantum trajectories (stochastic unraveling)
#
# TQE LOCK-IN MECHANISM (SCENARIO 2):
#   The f(E,I) function modulates system parameters online:
#     f(E,I) = exp(-(E-E_c)²/(2σ²)) · (1 + α·I)
#
#   Dynamic adaptation:
#     • Dissipation rates scaled by √f(E,I)
#     • Anharmonic potential strength scaled by f(E,I)
#     • Energy E = ⟨n₁⟩ + ⟨n₂⟩ (photon number proxy)
#     • Information I ~ Beta(a,b) (intrinsic orientation parameter)
#
# COMPARATIVE METRICS (Multi-Dimensional Tracking):
#   For each scenario, we track:
#     • Energy: mean ⟨E(t)⟩, variance σ²(t), max fluctuation
#     • Entropy: von Neumann entropy S_vN(ρ)
#     • Coherence: normalized off-diagonal density matrix elements C ∈ [0,1]
#     • Information drift: Jensen-Shannon divergence between ρ(t), ρ(t+dt)
#     • Heisenberg uncertainty: Δx, Δp, Δx·Δp (compliance check with ℏ/2 limit)
#     • I evolution: I(t) trajectory for dynamic I-modes
#
#   Suppression ratios quantify law effectiveness:
#     Variance ratio = σ²(WITH-LAW) / σ²(NO-LAW)
#     Coherence ratio = C(WITH-LAW) / C(NO-LAW)
#     Uncertainty ratio = Δx·Δp(WITH-LAW) / Δx·Δp(NO-LAW)
#     Ratio < 1 → Laws suppress fluctuations (TQE prediction confirmed)
#
#   I-origin diagnostics:
#     • EMERGENT: Monitor ⟨I(t)⟩, Var(I), Corr(E,I) trends
#       → If I ≠ 0 at t→∞, information emerges spontaneously
#     • INHERENT: Track I = f(E) deterministic relationship
#       → Test if all energy states have intrinsic information content
#     • THRESHOLD: Detect I activation at E > E_c
#       → Test quantum criticality hypothesis for information birth
#
# ANALYSIS OUTPUTS:
#   • Time-series comparison plots (NO-LAW vs WITH-LAW)
#   • Statistical comparison tables (mean, std, max, growth rates)
#   • Quantitative suppression ratios: variance, uncertainty, coherence
#   • Heisenberg uncertainty principle compliance check
#   • Phase space analysis (E vs S trajectories)
#   • Multi-dimensional tracking (E, S, C, Δx·Δp)
#   • I evolution tracking (emergent/inherent/threshold models)
#   • Parameter sweep analysis (EC, SIGMA, ALPHA)
#   • Publication-quality visualizations (9-10 plots)
#
# OUTPUT STRUCTURE:
#   TQE_Heisenberg_Fluctuation_YYYYMMDD_HHMMSS/
#     ├── summary.json                    # Complete run metadata + results + Heisenberg compliance
#     ├── comparative_analysis.json       # NO-LAW vs WITH-LAW statistics + suppression ratios
#     ├── data/
#     │   ├── no_law_timeseries.csv      # Scenario 1 time-series data
#     │   ├── with_law_timeseries.csv    # Scenario 2 time-series data
#     │   ├── ensemble_final_energies.csv # Final states for all realizations
#     │   └── parameter_sweep_{var}.csv  # Parameter sweep results (if enabled)
#     └── figs/
#         ├── 01_energy_comparison.png    # Energy evolution (both scenarios)
#         ├── 02_variance_comparison.png  # Variance evolution
#         ├── 03_entropy_comparison.png   # von Neumann entropy
#         ├── 04_coherence_comparison.png # Quantum coherence (normalized)
#         ├── 05_final_energy_dist.png    # Final energy distributions
#         ├── 06_suppression_summary.png  # Summary bar chart
#         ├── 07_heisenberg_uncertainty.png # Δx·Δp evolution + ℏ/2 limit
#         ├── 08_phase_space_E_vs_S.png   # Phase space trajectories
#         ├── 09_multidimensional_tracking.png # (E, S, C, Δx·Δp) 4-panel
#         ├── 10_parameter_sweep_{var}.png # Parameter sweep (if enabled)
#         ├── 11_I_evolution_emergent.png  # I(t) emergent model
#         ├── 12_I_evolution_inherent.png  # I(t) inherent model
#         ├── 13_I_evolution_threshold.png # I(t) threshold model
#         └── 14_I_mode_comparison.png     # All 3 I-modes compared
#
# INFORMATION ORIGIN MODES (I-MODES):
#   ALL 3 MODELS RUN AUTOMATICALLY in Phase 2:
#   • "emergent": I_{t+1} = γ·I_t + α·|ΔE_t| + β·corr(ΔE_t, ΔE_{t-1})
#                 Tests spontaneous information emergence from fluctuation structure
#   • "inherent": I = scale · f(E)  where f = log(E/E0) or (E/E0)^γ or E
#                 Tests hypothesis that all energy states have intrinsic I
#   • "threshold": I = 0 if E < E_c, else I += slope·(E - E_c)
#                 Tests quantum criticality model for information birth
#   Each model generates a separate PNG (11, 12, 13) + comparison plot (14)
#
# For detailed parameter descriptions, see MASTER_CTRL below.
#
# ==========================================================================================

# ======== CRITICAL: PACKAGE INSTALLATION FIRST ========
import sys
import subprocess

def _ensure(pkg):
    """Ensure a package is installed before importing."""
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ⚡ INSTALL PACKAGES BEFORE ANY IMPORTS ⚡
print("[SETUP] Checking and installing dependencies...")
essential_packages = ["numpy", "matplotlib", "scipy", "tqdm"]
quantum_packages = ["qutip"]

for pkg in essential_packages:
    print(f"[SETUP] Checking {pkg}...")
    _ensure(pkg)
    print(f"[SETUP] ✓ {pkg} ready")

# Try quantum packages
for pkg in quantum_packages:
    try:
        print(f"[SETUP] Checking {pkg}... (this may take 30-60 sec on first install)")
        _ensure(pkg)
        print(f"[SETUP] ✓ {pkg} ready")
    except Exception as e:
        print(f"[SETUP] Warning: Could not install {pkg}: {e}")

print("[SETUP] ✓ All dependencies ready")

# ======== NOW SAFE TO IMPORT ========
import os
import time
import json
import warnings
from datetime import datetime, timezone
import numpy as np
from scipy.stats import entropy as shannon_entropy
# multiprocessing not used (run_single has 273+ global dependencies - not MP-safe)
import gc

# Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Configure matplotlib for proper PNG generation
plt.ioff()  # Turn off interactive mode
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['figure.max_open_warning'] = 0

from tqdm.auto import tqdm

# QuTiP imports
try:
    from qutip import (basis, destroy, num, coherent, thermal_dm, Qobj, mesolve,
                       mcsolve, expect, enr_thermal_dm, tensor, qeye, entropy_vn)
    from qutip.solver import Options
    QUTIP_AVAILABLE = True
except ImportError:
    print("[WARNING] qutip not available - simulation cannot run")
    QUTIP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ======== COLAB DETECTION + DRIVE MOUNT ========
IN_COLAB = ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_BACKEND_VERSION" in os.environ)

if IN_COLAB:
    print("[COLAB] Google Colab environment detected.")
    try:
        from google.colab import drive
        print("[DRIVE] Attempting to mount Google Drive...")
        drive.mount('/content/drive', force_remount=True)
        print("[DRIVE] Successfully mounted!")
    except Exception as e:
        print(f"[DRIVE][WARN] Mount failed: {e}")
        print("[DRIVE] Falling back to local storage (/content/runs)")
else:
    print("[SETUP] Running in local environment.")

# ==========================================================================================
# MASTER CONTROL PANEL - ALL CONFIGURABLE PARAMETERS
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

# ==========================================================================================
# DIRECTORY SETUP
# ==========================================================================================

BASE_FOLDER_NAME = MASTER_CTRL["BASE_FOLDER_NAME"]

if IN_COLAB:
    ROOT_DIR = f"/content/drive/MyDrive/{BASE_FOLDER_NAME}"
else:
    # Use SIMULATION_RUNS/heisenberg directory in repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ROOT_DIR = os.path.join(repo_root, "SIMULATION_RUNS", "heisenberg")

os.makedirs(ROOT_DIR, exist_ok=True)

# Create timestamped run directory
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"{BASE_FOLDER_NAME}_{RUN_STAMP}"
OUTDIR = os.path.join(ROOT_DIR, RUN_NAME)
FIGDIR = os.path.join(OUTDIR, "figs")
DATADIR = os.path.join(OUTDIR, "data")

os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

print("\n" + "="*80)
print("DIRECTORY SETUP")
print("="*80)
print(f"Output Directory: {OUTDIR}")
print(f"  ├── data/       (CSV timeseries)")
print(f"  └── figs/       (PNG visualizations)")
print("="*80)

# ==========================================================================================
# PLOTTING STYLE SETUP
# ==========================================================================================

def setup_scientific_plotting_style():
    """Setup clean, scientific plotting style."""
    plt.style.use('default')
    
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.color': 'lightgray',
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.weight': 'light',
        'font.size': 11,
        'axes.titlesize': MASTER_CTRL['PLOT_FONTSIZE_TITLE'],
        'axes.labelsize': MASTER_CTRL['PLOT_FONTSIZE_LABEL'],
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': MASTER_CTRL['PLOT_FONTSIZE_LEGEND'],
        'figure.dpi': MASTER_CTRL['PLOT_DPI'],
        'savefig.dpi': MASTER_CTRL['PLOT_DPI'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

setup_scientific_plotting_style()

# ==========================================================================================
# REPRODUCIBILITY SETUP
# ==========================================================================================

SEED = MASTER_CTRL["SEED"]
if SEED is None:
    # Generate truly random seed from multiple entropy sources
    import hashlib
    entropy_sources = [
        str(time.time()),
        str(os.urandom(16)),
        str(os.getpid()),
    ]
    entropy_string = "".join(entropy_sources)
    hash_digest = hashlib.sha256(entropy_string.encode()).hexdigest()
    SEED = int(hash_digest[:8], 16) % (2**31)  # Use first 8 hex chars as seed
    MASTER_CTRL["SEED"] = SEED
    print(f"[SEED] Generated random master seed: {SEED}")
else:
    print(f"[SEED] Using specified master seed: {SEED}")

# Create RNG instances
rng = np.random.default_rng(SEED)
np.random.seed(SEED)  # Also seed legacy numpy RNG for compatibility

# Set environment variables for strict reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)

print(f"[SEED] ✓ RNG initialized (reproducible)")
print("="*80)

# ==========================================================================================
# TQE HELPER FUNCTIONS
# ==========================================================================================

def sample_info_beta(n, a=None, b=None):
    """Sample information parameter I from Beta(a, b) distribution."""
    if a is None:
        a = MASTER_CTRL["BETA_A"]
    if b is None:
        b = MASTER_CTRL["BETA_B"]
    return rng.beta(a, b, size=n)

def f_lockin(E, I, Ec=None, sigma=None, alpha=None):
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
    if Ec is None:
        Ec = MASTER_CTRL["EC"]
    if sigma is None:
        sigma = MASTER_CTRL["SIGMA"]
    if alpha is None:
        alpha = MASTER_CTRL["ALPHA"]
    
    gaussian = np.exp(-(E - Ec)**2 / (2.0 * sigma**2))
    return gaussian * (1.0 + alpha * I)

def lockin_rate_scale(E_proxy, I_val):
    """Return multiplicative scale for dissipators given E and I."""
    return float(np.clip(f_lockin(E_proxy, I_val), 0.25, 2.0))

def lockin_potential_scale(E_proxy, I_val):
    """Scale anharmonic strength depending on f(E,I)."""
    return float(np.clip(0.6 + 0.6*f_lockin(E_proxy, I_val), 0.2, 1.8))

def compute_heisenberg_uncertainty(rho, x_op, p_op):
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
    hbar_half = MASTER_CTRL["HBAR"] / 2.0
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

# ==========================================================================================
# QUANTUM SYSTEM SETUP
# ==========================================================================================

if not QUTIP_AVAILABLE:
    raise ImportError("QuTiP is required but not available. Please install it.")

print("\n" + "="*80)
print("QUANTUM SYSTEM INITIALIZATION")
print("="*80)
print("[QUANTUM] Building quantum system operators...")
print(f"[QUANTUM] Hilbert space dimension: {MASTER_CTRL['N_HILB']} per mode")

N_HILB = MASTER_CTRL["N_HILB"]
omega1 = MASTER_CTRL["OMEGA_1"]
omega2 = MASTER_CTRL["OMEGA_2"]
lam_x4 = MASTER_CTRL["LAM_X4"]
dw_c2 = MASTER_CTRL["DW_C2"]
dw_c4 = MASTER_CTRL["DW_C4"]
g_coup = MASTER_CTRL["G_COUP"]
drive_amp = MASTER_CTRL["DRIVE_AMP"]
drive_omega = MASTER_CTRL["DRIVE_OMEGA"]

# Single-mode operators
print("[QUANTUM] Creating mode-1 operators...")
a1 = destroy(N_HILB)
x1 = (a1 + a1.dag()) / np.sqrt(2.0)
p1 = (a1 - a1.dag()) / (1j * np.sqrt(2.0))
H1 = omega1 * (a1.dag() * a1)
print("[QUANTUM] ✓ Mode-1 ready")

# Add potentials on mode-1
if MASTER_CTRL["ANHARMONIC_X4"]:
    H1 = H1 + lam_x4 * (x1**4)

if MASTER_CTRL["DOUBLE_WELL"]:
    H1 = omega1 * (a1.dag()*a1) + dw_c2*(x1**2) + dw_c4*(x1**4)

# Handle one vs two-mode system
two_mode = MASTER_CTRL["TWO_MODE_COUPLING"]
ident = qeye(N_HILB)

if two_mode:
    print("[QUANTUM] Creating mode-2 operators and tensor products...")
    print(f"[QUANTUM] (This may take 30-60 sec for N_HILB={N_HILB})")
    a2 = destroy(N_HILB)
    x2 = (a2 + a2.dag()) / np.sqrt(2.0)
    H2_local = omega2 * (a2.dag() * a2)

    H1_full = tensor(H1, ident)
    H2_full = tensor(ident, H2_local)
    Hc_full = g_coup * tensor(a1 + a1.dag(), a2 + a2.dag())

    H = H1_full + H2_full + Hc_full
    print("[QUANTUM] ✓ Two-mode system ready")
else:
    H = H1
    print("[QUANTUM] ✓ Single-mode system ready")

# Time-dependent drive on mode-1
if MASTER_CTRL["TIME_DEP_DRIVE"]:
    def H_drive_coeff(t, args):
        return drive_amp * np.cos(drive_omega * t)

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

print("[QUANTUM] ✓ Quantum system operators ready")

# ==========================================================================================
# PRE-COMPUTE TENSOR OPERATORS FOR PERFORMANCE
# ==========================================================================================

if MASTER_CTRL.get("CACHE_TENSOR_OPS", True) and two_mode:
    print("[OPTIMIZATION] Pre-computing tensor operators...")
    x_op_mode1_cached = tensor(x1, ident)
    p_op_mode1_cached = tensor(p1, ident)
    print("[OPTIMIZATION] ✓ Tensor operators cached (2-3x speedup)")
else:
    x_op_mode1_cached = None
    p_op_mode1_cached = None
    
print("="*80)

# ==========================================================================================
# COLLAPSE OPERATORS (PRE-LAW BASELINE)
# ==========================================================================================

gamma_phi1 = MASTER_CTRL["GAMMA_PHI_1"]
kappa1 = MASTER_CTRL["KAPPA_1"]
nth1 = MASTER_CTRL["NTH_1"]
gamma_phi2 = MASTER_CTRL["GAMMA_PHI_2"]
kappa2 = MASTER_CTRL["KAPPA_2"]
nth2 = MASTER_CTRL["NTH_2"]

c_ops = []

# Mode-1 operators
a1_op = tensor(a1, ident) if two_mode else a1
n1_op = tensor(a1.dag() * a1, ident) if two_mode else a1.dag() * a1

c_ops.append(np.sqrt(gamma_phi1) * n1_op)
if MASTER_CTRL["THERMAL_BATH"] and nth1 > 0:
    c_ops.append(np.sqrt(kappa1 * (nth1 + 1.0)) * a1_op)
    c_ops.append(np.sqrt(kappa1 * nth1) * a1_op.dag())
else:
    c_ops.append(np.sqrt(kappa1) * a1_op)

# Mode-2 operators
if two_mode:
    a2_op = tensor(ident, a2)
    n2_op = tensor(ident, a2.dag() * a2)

    c_ops.append(np.sqrt(gamma_phi2) * n2_op)
    if MASTER_CTRL["THERMAL_BATH"] and nth2 > 0:
        c_ops.append(np.sqrt(kappa2 * (nth2 + 1.0)) * a2_op)
        c_ops.append(np.sqrt(kappa2 * nth2) * a2_op.dag())
    else:
        c_ops.append(np.sqrt(kappa2) * a2_op)

# ==========================================================================================
# INITIAL STATE SAMPLING
# ==========================================================================================

def sample_coherent_states(n):
    """Sample coherent state amplitudes with heavy-tailed distribution."""
    mags = np.sqrt(rng.lognormal(
        mean=MASTER_CTRL["COHERENT_LOG_MEAN"],
        sigma=MASTER_CTRL["COHERENT_LOG_SIGMA"],
        size=n
    ))
    phases = rng.uniform(0, 2*np.pi, size=n)
    return mags * np.exp(1j * phases)

N_ENSEMBLE = MASTER_CTRL["N_ENSEMBLE"]
T_FINAL = MASTER_CTRL["T_FINAL"]
N_T = MASTER_CTRL["N_T"]
tlist = np.linspace(0.0, T_FINAL, N_T)

print("\n" + "="*80)
print("INITIAL STATE SAMPLING")
print("="*80)
print(f"[ENSEMBLE] Sampling {N_ENSEMBLE} initial coherent states...")
alphas1 = sample_coherent_states(N_ENSEMBLE)
if two_mode:
    alphas2 = sample_coherent_states(N_ENSEMBLE)
else:
    alphas2 = None

I_samples = sample_info_beta(N_ENSEMBLE)
print(f"[ENSEMBLE] ✓ Initial states ready")
print(f"[ENSEMBLE] I-parameter range: [{I_samples.min():.3f}, {I_samples.max():.3f}]")
print("="*80)

# ==========================================================================================
# INFORMATION ORIGIN MODELS
# ==========================================================================================

def compute_I_emergent(I_prev, E_current, E_prev, E_prev2, config):
    """
    Emergent I: Information emerges from energy fluctuation structure.
    
    I_{t+1} = γ·I_t + α·|ΔE_t| + β·corr(ΔE_t, ΔE_{t-1})
    """
    alpha = config.get("I_EMERGENT_ALPHA", 0.3)
    beta = config.get("I_EMERGENT_BETA", 0.2)
    gamma = config.get("I_EMERGENT_GAMMA", 0.95)
    
    dE_t = E_current - E_prev
    dE_t_minus_1 = E_prev - E_prev2
    corr_term = dE_t * dE_t_minus_1 / (1.0 + abs(dE_t) + abs(dE_t_minus_1))
    I_new = gamma * I_prev + alpha * abs(dE_t) + beta * corr_term
    
    return np.clip(I_new, 0.0, 1.0)

def compute_I_inherent(E, config):
    """
    Inherent I: Information is deterministic function of energy.
    
    Options: log, power, linear
    """
    mode = config.get("I_INHERENT_MODE", "log")
    E0 = config.get("I_INHERENT_E0", 10.0)
    gamma_exp = config.get("I_INHERENT_GAMMA", 0.5)
    scale = config.get("I_INHERENT_SCALE", 0.05)
    
    if mode == "log":
        I = scale * np.log(max(E / E0, 0.01))
    elif mode == "power":
        I = scale * (E / E0) ** gamma_exp
    elif mode == "linear":
        I = scale * E
    else:
        I = 0.5
    
    return np.clip(I, 0.0, 1.0)

def compute_I_threshold(E, I_prev, config):
    """
    Threshold I: Information activates above critical energy.
    """
    E_c = config.get("I_THRESHOLD_EC", 15.0)
    slope = config.get("I_THRESHOLD_SLOPE", 0.1)
    I_max = config.get("I_THRESHOLD_MAX", 1.0)
    
    if E < E_c:
        return 0.0
    else:
        I_new = I_prev + slope * (E - E_c)
        return np.clip(I_new, 0.0, I_max)

# ==========================================================================================
# SIMULATION CORE: SINGLE TRAJECTORY EVOLUTION
# ==========================================================================================

def run_single(alpha1, alpha2=None, I_val=0.5, enable_lockin=False):
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
    
    Returns
    -------
    dict or None
        Dictionary with time series and final state, or None if simulation failed
    """
    # Initial state
    psi1 = coherent(N_HILB, alpha1)
    if two_mode:
        psi2 = coherent(N_HILB, alpha2)
        psi0 = tensor(psi1, psi2)
    else:
        psi0 = psi1

    # Optimized Options: store_states can be disabled for memory efficiency
    store_states_flag = not MASTER_CTRL.get("MEMORY_EFFICIENT", False)
    opts = Options(store_states=store_states_flag, nsteps=10000, atol=1e-8, rtol=1e-6)

    # ===== NO LOCK-IN: Single-shot evolution =====
    if not enable_lockin:
        e_ops_list = [Nop1_full]
        if two_mode:
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
                e_series.append(expect(Nop1_full, rho) + expect(Nop2_full, rho))
                rho1 = rho.ptrace(0)
                # Use normalized coherence
                coh_series.append(normalize_coherence(rho, mode=0))
                rho_pop = np.real(np.diag(rho1.full()).flatten())
                
                # Heisenberg uncertainty for mode 1 (use cached operators)
                if x_op_mode1_cached is not None:
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1_cached, p_op_mode1_cached)
                else:
                    x_op_mode1 = tensor(x1, ident)
                    p_op_mode1 = tensor(p1, ident)
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1, p_op_mode1)
            else:
                e_series.append(expect(Nop1_full, rho))
                rho_pop = np.real(np.diag(rho.full()).flatten())
                # Use normalized coherence
                coh_series.append(normalize_coherence(rho))
                
                # Heisenberg uncertainty
                dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x1, p1)
            
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
            I_evolution=None,  # NO-LAW: no I evolution (no lock-in)
            I_mode="none",     # NO-LAW: no I-mode active
            final_state=res.states[-1],
            final_energy=e_series[-1]
        )
    
    # ===== WITH LOCK-IN: Segmented evolution =====
    else:
        segs = MASTER_CTRL["N_SEGMENTS"]
        seg_edges = np.linspace(0.0, T_FINAL, segs + 1)

        all_states_collected = []
        H_static_base = H if not isinstance(H, list) else H[0]
        state_in = psi0
        
        # Dynamic I tracking (initialize based on I_ORIGIN_MODE)
        I_mode = MASTER_CTRL.get("I_ORIGIN_MODE", "fixed")
        I_current = I_val  # Start with initial I
        I_history = [I_current]
        
        # Initialize E_history with initial energy estimate
        E_proxy_init = expect(Nop1_full, psi0) + (expect(Nop2_full, psi0) if two_mode else 0)
        E_history = [E_proxy_init, E_proxy_init, E_proxy_init]  # Pre-fill for emergent model

        for s in range(segs):
            t0, t1 = seg_edges[s], seg_edges[s+1]
            t_points_in_seg = tlist[(tlist >= t0) & (tlist <= t1)]

            t_solve_seg = np.union1d([t0], t_points_in_seg)
            if len(t_solve_seg) < 2:
                continue

            E_proxy = E_proxy_init  # Default to initial
            if all_states_collected:
                last_rho = all_states_collected[-1]
                E_proxy = expect(Nop1_full, last_rho) + (expect(Nop2_full, last_rho) if two_mode else 0)
                E_history.append(E_proxy)
            
            # Update I dynamically based on I_ORIGIN_MODE (every segment, not just s > 0)
            if I_mode == "emergent":
                # Emergent I: requires at least 3 energy measurements (always satisfied now)
                I_current = compute_I_emergent(
                    I_current, 
                    E_history[-1], 
                    E_history[-2], 
                    E_history[-3], 
                    MASTER_CTRL
                )
            elif I_mode == "inherent":
                # Inherent I: deterministic function of E
                I_current = compute_I_inherent(E_proxy, MASTER_CTRL)
            elif I_mode == "threshold":
                # Threshold I: activated above critical energy
                I_current = compute_I_threshold(E_proxy, I_current, MASTER_CTRL)
            # else: "fixed" mode - I_current stays constant
            
            I_history.append(I_current)

            H_current_static = H_static_base
            cseg = list(c_ops)
            
            if s > 0:
                # Use dynamic I_current instead of fixed I_val
                rscale = lockin_rate_scale(E_proxy, I_current)
                cseg = [np.sqrt(rscale) * cc for cc in cseg]
                
                if MASTER_CTRL["ANHARMONIC_X4"] or MASTER_CTRL["DOUBLE_WELL"]:
                    pscale = lockin_potential_scale(E_proxy, I_current)
                    if MASTER_CTRL["DOUBLE_WELL"]:
                        H1_dyn_local = omega1*(a1.dag()*a1) + (pscale * dw_c2)*(x1**2) + (pscale * dw_c4)*(x1**4)
                    else:
                        H1_dyn_local = omega1*(a1.dag()*a1) + (pscale * lam_x4)*(x1**4)

                    if two_mode:
                        H2_local = omega2*(a2.dag()*a2)
                        H_current_static = tensor(H1_dyn_local, ident) + tensor(ident, H2_local) + g_coup*tensor(a1+a1.dag(), a2+a2.dag())
                    else:
                        H_current_static = H1_dyn_local

            Hseg = [H_current_static, H[1]] if MASTER_CTRL["TIME_DEP_DRIVE"] else H_current_static
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
                e_series.append(expect(Nop1_full, rho) + expect(Nop2_full, rho))
                rho1 = rho.ptrace(0)
                # Use normalized coherence
                coh_series.append(normalize_coherence(rho, mode=0))
                rho_pop = np.real(np.diag(rho1.full()).flatten())
                
                # Heisenberg uncertainty for mode 1 (use cached operators)
                if x_op_mode1_cached is not None:
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1_cached, p_op_mode1_cached)
                else:
                    x_op_mode1 = tensor(x1, ident)
                    p_op_mode1 = tensor(p1, ident)
                    dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x_op_mode1, p_op_mode1)
            else:
                e_series.append(expect(Nop1_full, rho))
                rho_pop = np.real(np.diag(rho.full()).flatten())
                # Use normalized coherence
                coh_series.append(normalize_coherence(rho))
                
                # Heisenberg uncertainty
                dx, dp, unc_prod, _ = compute_heisenberg_uncertainty(rho, x1, p1)
            
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
            I_evolution=np.array(I_history) if I_mode != "fixed" else None,  # Track I evolution
            I_mode=I_mode,  # Record which model was used
            final_state=all_states_collected[-1],
            final_energy=e_series[-1]
        )

# ==========================================================================================
# MAIN SIMULATION: COMPARATIVE ANALYSIS
# ==========================================================================================

print("\n" + "="*80)
print("TQE HEISENBERG FLUCTUATION PIPELINE v4.2.0 PRO")
print("="*80)
print(f"Ensemble Size:    {N_ENSEMBLE} trajectories")
print(f"Time Points:      {N_T} (dt = {T_FINAL/N_T:.4f})")
print(f"Hilbert Dim:      {N_HILB} per mode")
print(f"Multiprocessing:  {'Enabled' if MASTER_CTRL.get('USE_MULTIPROCESSING', True) else 'Disabled'}")
print(f"Master Seed:      {SEED}")
print("="*80 + "\n")

# Create master progress bar
progress = tqdm(total=7, desc="Pipeline Progress", unit="phase", leave=True)

# ===== SCENARIO 1: NO-LAW (PRE-LAW STATE) =====
progress.set_description("1/7: NO-LAW Simulation")
print("\n" + "="*80)
print("[PHASE 1] NO-LAW SIMULATION")
print("="*80)
print("  Scenario: Pre-law quantum fluctuations (no lock-in mechanism)")
print(f"  Ensemble: {N_ENSEMBLE} trajectories")
print(f"  Time Points: {N_T} (dt = {T_FINAL/N_T:.4f})")

results_no_law = []
I_kept_no_law = []

# Serial execution (optimized with tensor cache - 2-3x speedup)
for i in tqdm(range(N_ENSEMBLE), desc="  Simulating", leave=False):
    a2_sample = alphas2[i] if two_mode else None
    r = run_single(alphas1[i], a2_sample, I_samples[i], enable_lockin=False)
    
    if r is not None:
        results_no_law.append(r)
        I_kept_no_law.append(I_samples[i])

print(f"[PHASE 1] ✓ Complete: {len(results_no_law)}/{N_ENSEMBLE} valid trajectories")
progress.update(1)

# Memory cleanup
gc.collect()

# ===== SCENARIO 2: WITH-LAW (STABLE LAWS) - ALL 3 I-MODES =====
progress.set_description("2/7: WITH-LAW Simulation (3 I-modes)")
print("\n" + "="*80)
print("[PHASE 2] WITH-LAW SIMULATION - ALL 3 I-ORIGIN MODES")
print("="*80)
print("  Scenario: TQE lock-in mechanism active (f(E,I) coupling)")
print(f"  Ensemble: {N_ENSEMBLE} trajectories × 3 I-modes")
print(f"  I-Modes: emergent, inherent, threshold")
print(f"  Segments: {MASTER_CTRL['N_SEGMENTS']} (dynamic adaptation)")

# Store original I_ORIGIN_MODE
original_I_mode = MASTER_CTRL.get("I_ORIGIN_MODE", "emergent")

# Run all 3 I-modes
results_with_law_emergent = []
results_with_law_inherent = []
results_with_law_threshold = []
I_kept_with_law = []

I_modes_to_test = ["emergent", "inherent", "threshold"]

for mode_idx, i_mode in enumerate(I_modes_to_test):
    print(f"\n  [{mode_idx+1}/3] Running I-mode: {i_mode}")
    MASTER_CTRL["I_ORIGIN_MODE"] = i_mode
    
    results_mode = []
    for i in tqdm(range(N_ENSEMBLE), desc=f"    Simulating ({i_mode})", leave=False):
        a2_sample = alphas2[i] if two_mode else None
        r = run_single(alphas1[i], a2_sample, I_samples[i], enable_lockin=True)
        
        if r is not None:
            results_mode.append(r)
    
    # Store results for each mode
    if i_mode == "emergent":
        results_with_law_emergent = results_mode
    elif i_mode == "inherent":
        results_with_law_inherent = results_mode
    elif i_mode == "threshold":
        results_with_law_threshold = results_mode
    
    print(f"    ✓ {i_mode}: {len(results_mode)}/{N_ENSEMBLE} valid")

# Restore original mode and use emergent as default for stats
MASTER_CTRL["I_ORIGIN_MODE"] = original_I_mode
results_with_law = results_with_law_emergent  # Use emergent for main comparison
I_kept_with_law = [I_samples[i] for i in range(min(len(results_with_law), len(I_samples)))]

print(f"\n[PHASE 2] ✓ Complete: 3 I-modes tested ({N_ENSEMBLE} traj each)")
progress.update(1)

# Memory cleanup
gc.collect()

# ===== VALIDATION =====
if not results_no_law or not results_with_law:
    raise RuntimeError("Simulation produced no valid results. Cannot continue.")

# ==========================================================================================
# DATA AGGREGATION & STATISTICAL ANALYSIS
# ==========================================================================================

progress.set_description("3/7: Data Aggregation")
print("\n" + "="*80)
print("[PHASE 3] DATA AGGREGATION")
print("="*80)
print("  Aggregating time-series data from all trajectories...")

# Extract final energies
final_energies_no_law = np.array([r["final_energy"] for r in results_no_law])
final_energies_with_law = np.array([r["final_energy"] for r in results_with_law])

# Time-series matrices (ensure same length)
T_len_no_law = min(len(r["energies"]) for r in results_no_law)
T_len_with_law = min(len(r["energies"]) for r in results_with_law)
T_len = min(T_len_no_law, T_len_with_law)

E_mat_no_law = np.vstack([r["energies"][:T_len] for r in results_no_law])
S_mat_no_law = np.vstack([r["entropy"][:T_len] for r in results_no_law])
C_mat_no_law = np.vstack([r["coherence"][:T_len] for r in results_no_law])
U_mat_no_law = np.vstack([r["uncertainty_product"][:T_len] for r in results_no_law])
DX_mat_no_law = np.vstack([r["delta_x"][:T_len] for r in results_no_law])
DP_mat_no_law = np.vstack([r["delta_p"][:T_len] for r in results_no_law])

E_mat_with_law = np.vstack([r["energies"][:T_len] for r in results_with_law])
S_mat_with_law = np.vstack([r["entropy"][:T_len] for r in results_with_law])
C_mat_with_law = np.vstack([r["coherence"][:T_len] for r in results_with_law])
U_mat_with_law = np.vstack([r["uncertainty_product"][:T_len] for r in results_with_law])
DX_mat_with_law = np.vstack([r["delta_x"][:T_len] for r in results_with_law])
DP_mat_with_law = np.vstack([r["delta_p"][:T_len] for r in results_with_law])

# Compute statistics
mean_E_no_law = np.mean(E_mat_no_law, axis=0)
std_E_no_law = np.std(E_mat_no_law, axis=0)
mean_E_with_law = np.mean(E_mat_with_law, axis=0)
std_E_with_law = np.std(E_mat_with_law, axis=0)

mean_S_no_law = np.mean(S_mat_no_law, axis=0)
mean_S_with_law = np.mean(S_mat_with_law, axis=0)

mean_C_no_law = np.mean(C_mat_no_law, axis=0)
mean_C_with_law = np.mean(C_mat_with_law, axis=0)

mean_U_no_law = np.mean(U_mat_no_law, axis=0)
std_U_no_law = np.std(U_mat_no_law, axis=0)
mean_U_with_law = np.mean(U_mat_with_law, axis=0)
std_U_with_law = np.std(U_mat_with_law, axis=0)

mean_DX_no_law = np.mean(DX_mat_no_law, axis=0)
mean_DX_with_law = np.mean(DX_mat_with_law, axis=0)

tlist_agg = tlist[:T_len]

print(f"[PHASE 3] ✓ Complete: {len(results_no_law) + len(results_with_law)} trajectories aggregated")
progress.update(1)

# ==========================================================================================
# COMPARATIVE STATISTICS
# ==========================================================================================

progress.set_description("4/7: Statistical Analysis")
print("\n" + "="*80)
print("[PHASE 4] STATISTICAL ANALYSIS")
print("="*80)
print("  Computing comparative statistics (NO-LAW vs WITH-LAW)...")

stats_comparison = {
    "run_metadata": {
        "run_name": RUN_NAME,
        "timestamp_utc": RUN_STAMP,
        "master_seed": SEED,
        "reproducibility_note": "Use 'master_seed' value in MASTER_CTRL['SEED'] to reproduce this exact run",
    },
    "NO_LAW": {
        "scenario_description": "Pre-law state (pure quantum fluctuations, no lock-in)",
        "n_valid_trajectories": len(results_no_law),
        "final_energy_mean": float(np.mean(final_energies_no_law)),
        "final_energy_std": float(np.std(final_energies_no_law)),
        "final_energy_max": float(np.max(final_energies_no_law)),
        "final_energy_min": float(np.min(final_energies_no_law)),
        "variance_mean": float(np.mean(std_E_no_law**2)),
        "variance_max": float(np.max(std_E_no_law**2)),
        "entropy_final_mean": float(mean_S_no_law[-1]),
        "coherence_final_mean": float(mean_C_no_law[-1]),
        "heisenberg_uncertainty_final_mean": float(mean_U_no_law[-1]),
        "heisenberg_uncertainty_final_std": float(std_U_no_law[-1]),
        "delta_x_final_mean": float(mean_DX_no_law[-1]),
    },
    "WITH_LAW": {
        "scenario_description": "Stable laws active (TQE lock-in mechanism enabled)",
        "n_valid_trajectories": len(results_with_law),
        "final_energy_mean": float(np.mean(final_energies_with_law)),
        "final_energy_std": float(np.std(final_energies_with_law)),
        "final_energy_max": float(np.max(final_energies_with_law)),
        "final_energy_min": float(np.min(final_energies_with_law)),
        "variance_mean": float(np.mean(std_E_with_law**2)),
        "variance_max": float(np.max(std_E_with_law**2)),
        "entropy_final_mean": float(mean_S_with_law[-1]),
        "coherence_final_mean": float(mean_C_with_law[-1]),
        "heisenberg_uncertainty_final_mean": float(mean_U_with_law[-1]),
        "heisenberg_uncertainty_final_std": float(std_U_with_law[-1]),
        "delta_x_final_mean": float(mean_DX_with_law[-1]),
    },
    "SUPPRESSION_RATIOS": {
        "description": "Ratios < 1.0 indicate suppression by stable laws",
        "variance_ratio": float(np.mean(std_E_with_law**2) / (np.mean(std_E_no_law**2) + 1e-15)),
        "std_ratio": float(np.std(final_energies_with_law) / (np.std(final_energies_no_law) + 1e-15)),
        "max_energy_ratio": float(np.max(final_energies_with_law) / (np.max(final_energies_no_law) + 1e-15)),
        "uncertainty_ratio": float(mean_U_with_law[-1] / (mean_U_no_law[-1] + 1e-15)),
        "coherence_ratio": float(mean_C_with_law[-1] / (mean_C_no_law[-1] + 1e-15)),
    },
    "HEISENBERG_COMPLIANCE": {
        "description": "Check if Heisenberg uncertainty principle is satisfied",
        "min_uncertainty_no_law": float(np.min(mean_U_no_law)),
        "min_uncertainty_with_law": float(np.min(mean_U_with_law)),
        "theoretical_minimum": float(MASTER_CTRL["HBAR"] / 2.0),
        "no_law_compliant": bool(np.min(mean_U_no_law) >= MASTER_CTRL["HBAR"] / 2.0 * 0.99),
        "with_law_compliant": bool(np.min(mean_U_with_law) >= MASTER_CTRL["HBAR"] / 2.0 * 0.99),
    },
    "INFORMATION_ORIGIN": {
        "description": "All 3 I-origin models tested: emergent, inherent, threshold",
        "I_initial_mean": float(np.mean(I_samples)),
        "I_initial_std": float(np.std(I_samples)),
        "emergent": {
            "n_trajectories": len(results_with_law_emergent),
            "model": "I_{t+1} = γ·I_t + α·|ΔE_t| + β·corr(ΔE_t, ΔE_{t-1})",
        },
        "inherent": {
            "n_trajectories": len(results_with_law_inherent),
            "model": "I = scale · log(E/E0) or (E/E0)^γ",
        },
        "threshold": {
            "n_trajectories": len(results_with_law_threshold),
            "model": "I = 0 if E < E_c, else I += slope·(E-E_c)",
        },
    }
}

print(f"[PHASE 4] ✓ Complete: Statistics computed")
progress.update(1)

print("\n" + "="*80)
print("COMPARATIVE STATISTICS SUMMARY")
print("="*80)
print(f"\nNO-LAW (Pre-law state):")
print(f"  Final Energy: {stats_comparison['NO_LAW']['final_energy_mean']:.3f} ± {stats_comparison['NO_LAW']['final_energy_std']:.3f}")
print(f"  Max Energy: {stats_comparison['NO_LAW']['final_energy_max']:.3f}")
print(f"  Mean Variance: {stats_comparison['NO_LAW']['variance_mean']:.3f}")

print(f"\nWITH-LAW (Stable laws active):")
print(f"  Final Energy: {stats_comparison['WITH_LAW']['final_energy_mean']:.3f} ± {stats_comparison['WITH_LAW']['final_energy_std']:.3f}")
print(f"  Max Energy: {stats_comparison['WITH_LAW']['final_energy_max']:.3f}")
print(f"  Mean Variance: {stats_comparison['WITH_LAW']['variance_mean']:.3f}")

print(f"\nSUPPRESSION RATIOS (WITH-LAW / NO-LAW):")
print(f"  Variance Ratio: {stats_comparison['SUPPRESSION_RATIOS']['variance_ratio']:.4f}")
print(f"  Std Dev Ratio: {stats_comparison['SUPPRESSION_RATIOS']['std_ratio']:.4f}")
print(f"  Max Energy Ratio: {stats_comparison['SUPPRESSION_RATIOS']['max_energy_ratio']:.4f}")
print(f"  Uncertainty Product Ratio: {stats_comparison['SUPPRESSION_RATIOS']['uncertainty_ratio']:.4f}")
print(f"  Coherence Ratio: {stats_comparison['SUPPRESSION_RATIOS']['coherence_ratio']:.4f}")

print(f"\nHEISENBERG UNCERTAINTY COMPLIANCE:")
print(f"  Theoretical Minimum (ℏ/2): {stats_comparison['HEISENBERG_COMPLIANCE']['theoretical_minimum']:.3f}")
print(f"  NO-LAW Min: {stats_comparison['HEISENBERG_COMPLIANCE']['min_uncertainty_no_law']:.3f} " + 
      f"({'✓' if stats_comparison['HEISENBERG_COMPLIANCE']['no_law_compliant'] else '✗'})")
print(f"  WITH-LAW Min: {stats_comparison['HEISENBERG_COMPLIANCE']['min_uncertainty_with_law']:.3f} " +
      f"({'✓' if stats_comparison['HEISENBERG_COMPLIANCE']['with_law_compliant'] else '✗'})")
print("="*80 + "\n")

# ==========================================================================================
# SAVE DATA
# ==========================================================================================

progress.set_description("5/7: Saving Data Files")
print("\n" + "="*80)
print("[PHASE 5] SAVING DATA FILES")
print("="*80)
print("  Writing CSV and JSON files to disk...")

# Save comparative analysis JSON
comparative_filepath = os.path.join(OUTDIR, "comparative_analysis.json")
with open(comparative_filepath, 'w') as f:
    json.dump(stats_comparison, f, indent=4)
print(f"[SAVED] comparative_analysis.json")

# Save summary JSON
summary_data = {
    "run_info": {
        "run_name": RUN_NAME,
        "timestamp_utc": RUN_STAMP,
        "seed": SEED,
    },
    "parameters": MASTER_CTRL,
    "results": stats_comparison,
}

summary_filepath = os.path.join(OUTDIR, "summary.json")
with open(summary_filepath, 'w') as f:
    json.dump(summary_data, f, indent=4)
print(f"[SAVED] summary.json")

# Save time-series data with seed metadata
if tlist_agg.size > 0:
    # NO-LAW time-series (with Heisenberg uncertainty)
    csv_no_law = os.path.join(DATADIR, "no_law_timeseries.csv")
    data_no_law = np.vstack([tlist_agg, mean_E_no_law, std_E_no_law, mean_S_no_law, mean_C_no_law, 
                              mean_U_no_law, std_U_no_law, mean_DX_no_law]).T
    header_no_law = (f"# TQE Heisenberg Fluctuation - NO-LAW Scenario\n"
                     f"# Run: {RUN_NAME}\n"
                     f"# Timestamp: {RUN_STAMP}\n"
                     f"# Master Seed: {SEED}\n"
                     f"# N_Ensemble: {len(results_no_law)}\n"
                     f"# Reproducibility: Set MASTER_CTRL['SEED']={SEED} to reproduce this run\n"
                     f"time,mean_energy,std_energy,mean_entropy,mean_coherence,mean_uncertainty,std_uncertainty,mean_delta_x")
    np.savetxt(csv_no_law, data_no_law, delimiter=",", header=header_no_law, comments="")
    print(f"[SAVED] no_law_timeseries.csv")
    
    # WITH-LAW time-series (with Heisenberg uncertainty)
    csv_with_law = os.path.join(DATADIR, "with_law_timeseries.csv")
    data_with_law = np.vstack([tlist_agg, mean_E_with_law, std_E_with_law, mean_S_with_law, mean_C_with_law,
                                mean_U_with_law, std_U_with_law, mean_DX_with_law]).T
    header_with_law = (f"# TQE Heisenberg Fluctuation - WITH-LAW Scenario\n"
                       f"# Run: {RUN_NAME}\n"
                       f"# Timestamp: {RUN_STAMP}\n"
                       f"# Master Seed: {SEED}\n"
                       f"# N_Ensemble: {len(results_with_law)}\n"
                       f"# Reproducibility: Set MASTER_CTRL['SEED']={SEED} to reproduce this run\n"
                       f"time,mean_energy,std_energy,mean_entropy,mean_coherence,mean_uncertainty,std_uncertainty,mean_delta_x")
    np.savetxt(csv_with_law, data_with_law, delimiter=",", header=header_with_law, comments="")
    print(f"[SAVED] with_law_timeseries.csv")

# Save final energies with seed metadata
csv_final = os.path.join(DATADIR, "ensemble_final_energies.csv")
data_final = np.vstack([final_energies_no_law, final_energies_with_law]).T
header_final = (f"# TQE Heisenberg Fluctuation - Final Energies\n"
                f"# Run: {RUN_NAME}\n"
                f"# Timestamp: {RUN_STAMP}\n"
                f"# Master Seed: {SEED}\n"
                f"# N_Ensemble (NO-LAW): {len(results_no_law)}\n"
                f"# N_Ensemble (WITH-LAW): {len(results_with_law)}\n"
                f"# Reproducibility: Set MASTER_CTRL['SEED']={SEED} to reproduce this run\n"
                f"no_law_final_energy,with_law_final_energy")
np.savetxt(csv_final, data_final, delimiter=",", header=header_final, comments="")
print(f"[SAVED] ensemble_final_energies.csv")

print(f"\n[PHASE 5] Summary:")
print(f"  ✓ comparative_analysis.json")
print(f"  ✓ summary.json")
print(f"  ✓ no_law_timeseries.csv")
print(f"  ✓ with_law_timeseries.csv")
print(f"  ✓ ensemble_final_energies.csv")
print(f"[PHASE 5] ✓ Complete: 5 data files saved (2 JSON + 3 CSV)")
progress.update(1)

# ==========================================================================================
# VISUALIZATION
# ==========================================================================================

progress.set_description("6/7: Generating Visualizations")
print("\n" + "="*80)
print("[PHASE 6] GENERATING VISUALIZATIONS")
print("="*80)
print("  Creating publication-quality plots...")

# ===== FIGURE 1: Energy Evolution Comparison =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tlist_agg, mean_E_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
ax.fill_between(tlist_agg, mean_E_no_law - std_E_no_law, mean_E_no_law + std_E_no_law,
                alpha=0.3, color='red')
ax.plot(tlist_agg, mean_E_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
ax.fill_between(tlist_agg, mean_E_with_law - std_E_with_law, mean_E_with_law + std_E_with_law,
                alpha=0.3, color='blue')
ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('Mean Energy ⟨E(t)⟩', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Energy Evolution: NO-LAW vs WITH-LAW', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_energy_comparison.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 01_energy_comparison.png")

# ===== FIGURE 2: Variance Comparison =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tlist_agg, std_E_no_law**2, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
ax.plot(tlist_agg, std_E_with_law**2, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('Energy Variance σ²(t)', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Variance Evolution: Fluctuation Suppression', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "02_variance_comparison.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 02_variance_comparison.png")

# ===== FIGURE 3: Entropy Comparison =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tlist_agg, mean_S_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
ax.plot(tlist_agg, mean_S_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('Mean von Neumann Entropy', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Entropy Evolution: NO-LAW vs WITH-LAW', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_entropy_comparison.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 03_entropy_comparison.png")

# ===== FIGURE 4: Coherence Comparison =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tlist_agg, mean_C_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
ax.plot(tlist_agg, mean_C_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('Mean Quantum Coherence', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Coherence Evolution: NO-LAW vs WITH-LAW', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "04_coherence_comparison.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 04_coherence_comparison.png")

# ===== FIGURE 5: Final Energy Distributions =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(final_energies_no_law, bins=30, alpha=0.6, label='NO-LAW (Pre-law)', color='red', density=True)
ax.hist(final_energies_with_law, bins=30, alpha=0.6, label='WITH-LAW (Stable laws)', color='blue', density=True)
ax.set_xlabel('Final Energy', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('Probability Density', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Final Energy Distribution Comparison', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "05_final_energy_dist.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 05_final_energy_dist.png")

# ===== FIGURE 6: Suppression Summary Bar Chart =====
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Variance\nRatio', 'Std Dev\nRatio', 'Max Energy\nRatio']
values = [
    stats_comparison['SUPPRESSION_RATIOS']['variance_ratio'],
    stats_comparison['SUPPRESSION_RATIOS']['std_ratio'],
    stats_comparison['SUPPRESSION_RATIOS']['max_energy_ratio'],
]
colors = ['skyblue', 'lightcoral', 'lightgreen']
bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add horizontal line at 1.0
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='No suppression (ratio = 1)')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Ratio (WITH-LAW / NO-LAW)', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Fluctuation Suppression Summary\n(Values < 1.0 indicate suppression)', 
             fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_suppression_summary.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 06_suppression_summary.png")

# ===== FIGURE 7: Heisenberg Uncertainty Evolution =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(tlist_agg, mean_U_no_law, 'r-', linewidth=2, label='NO-LAW (Pre-law)', alpha=0.8)
ax.fill_between(tlist_agg, mean_U_no_law - std_U_no_law, mean_U_no_law + std_U_no_law,
                alpha=0.3, color='red')
ax.plot(tlist_agg, mean_U_with_law, 'b-', linewidth=2, label='WITH-LAW (Stable laws)', alpha=0.8)
ax.fill_between(tlist_agg, mean_U_with_law - std_U_with_law, mean_U_with_law + std_U_with_law,
                alpha=0.3, color='blue')
# Add Heisenberg limit
hbar_half = MASTER_CTRL["HBAR"] / 2.0
ax.axhline(y=hbar_half, color='black', linestyle='--', linewidth=2, 
           label=f'Heisenberg Limit (ℏ/2 = {hbar_half:.2f})', alpha=0.7)
ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('Uncertainty Product Δx·Δp', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Heisenberg Uncertainty Evolution', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "07_heisenberg_uncertainty.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 07_heisenberg_uncertainty.png")

# ===== FIGURE 8: Phase Space (Energy vs Entropy) =====
fig, ax = plt.subplots(figsize=(10, 8))
# Sample subset for clarity
n_sample = min(100, len(results_no_law), len(results_with_law))
for i in range(n_sample):
    if i < len(results_no_law):
        ax.plot(results_no_law[i]["energies"], results_no_law[i]["entropy"], 
               'r-', alpha=0.05, linewidth=0.5)
    if i < len(results_with_law):
        ax.plot(results_with_law[i]["energies"], results_with_law[i]["entropy"], 
               'b-', alpha=0.05, linewidth=0.5)

# Add mean trajectories
ax.plot(mean_E_no_law, mean_S_no_law, 'r-', linewidth=3, label='NO-LAW mean', alpha=0.9)
ax.plot(mean_E_with_law, mean_S_with_law, 'b-', linewidth=3, label='WITH-LAW mean', alpha=0.9)
ax.set_xlabel('Energy ⟨E⟩', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_ylabel('von Neumann Entropy S', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
ax.set_title('Phase Space: Energy vs Entropy', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "08_phase_space_E_vs_S.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 08_phase_space_E_vs_S.png")

# ===== FIGURE 9: Multi-Dimensional Tracking (E, I, S, C) =====
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Energy
ax1.plot(tlist_agg, mean_E_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
ax1.plot(tlist_agg, mean_E_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
ax1.set_ylabel('Energy ⟨E⟩', fontsize=11)
ax1.set_title('(A) Energy Evolution', fontsize=12, pad=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Entropy
ax2.plot(tlist_agg, mean_S_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
ax2.plot(tlist_agg, mean_S_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
ax2.set_ylabel('Entropy S', fontsize=11)
ax2.set_title('(B) von Neumann Entropy', fontsize=12, pad=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Coherence (normalized)
ax3.plot(tlist_agg, mean_C_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
ax3.plot(tlist_agg, mean_C_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
ax3.set_xlabel('Time', fontsize=11)
ax3.set_ylabel('Coherence C (normalized)', fontsize=11)
ax3.set_title('(C) Quantum Coherence', fontsize=12, pad=10)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Uncertainty product
ax4.plot(tlist_agg, mean_U_no_law, 'r-', linewidth=2, label='NO-LAW', alpha=0.8)
ax4.plot(tlist_agg, mean_U_with_law, 'b-', linewidth=2, label='WITH-LAW', alpha=0.8)
ax4.axhline(y=hbar_half, color='black', linestyle='--', linewidth=2, alpha=0.7, label='ℏ/2')
ax4.set_xlabel('Time', fontsize=11)
ax4.set_ylabel('Δx·Δp', fontsize=11)
ax4.set_title('(D) Heisenberg Uncertainty', fontsize=12, pad=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.suptitle('Multi-Dimensional Quantum State Tracking', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "09_multidimensional_tracking.png"), dpi=MASTER_CTRL['PLOT_DPI'])
plt.close()
print("[SAVED] 09_multidimensional_tracking.png")

# ===== FIGURE 11-13: I Evolution for ALL 3 I-Modes =====
I_mode_results = {
    "emergent": results_with_law_emergent,
    "inherent": results_with_law_inherent,
    "threshold": results_with_law_threshold
}

I_mode_colors = {
    "emergent": "dodgerblue",
    "inherent": "forestgreen",
    "threshold": "darkorange"
}

n_I_plots = 0
I_mean_all_modes = {}  # Store mean I(t) for comparison plot

for mode_name, results_mode in I_mode_results.items():
    if results_mode and len(results_mode) > 0:
        I_evolutions = [r.get('I_evolution') for r in results_mode if r.get('I_evolution') is not None]
        
        if I_evolutions:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all I trajectories
            color = I_mode_colors[mode_name]
            for I_evo in I_evolutions:
                if len(I_evo) > 0:
                    t_I = np.linspace(0, T_FINAL, len(I_evo))
                    ax.plot(t_I, I_evo, alpha=0.1, color=color, linewidth=0.8)
            
            # Plot mean trajectory
            max_len = max([len(I_evo) for I_evo in I_evolutions])
            I_matrix = np.full((len(I_evolutions), max_len), np.nan)
            for i, I_evo in enumerate(I_evolutions):
                I_matrix[i, :len(I_evo)] = I_evo
            I_mean = np.nanmean(I_matrix, axis=0)
            t_I_mean = np.linspace(0, T_FINAL, max_len)
            ax.plot(t_I_mean, I_mean, color='red', linewidth=2.5, label='Mean I(t)', zorder=10)
            
            # Store for comparison plot
            I_mean_all_modes[mode_name] = (t_I_mean, I_mean)
            
            # Formatting
            ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_ylabel('Information I', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax.set_title(f'Information Evolution: {mode_name.capitalize()} Model', 
                        fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
            ax.legend(loc='best', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save with mode-specific number
            if mode_name == "emergent":
                filename = "11_I_evolution_emergent.png"
            elif mode_name == "inherent":
                filename = "12_I_evolution_inherent.png"
            elif mode_name == "threshold":
                filename = "13_I_evolution_threshold.png"
            
            plt.savefig(os.path.join(FIGDIR, filename), dpi=MASTER_CTRL['PLOT_DPI'])
            plt.close()
            print(f"[SAVED] {filename}")
            n_I_plots += 1

# ===== FIGURE 14: I-Mode Comparison (all 3 on one plot) =====
if len(I_mean_all_modes) > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for mode_name, (t_I, I_mean) in I_mean_all_modes.items():
        color = I_mode_colors[mode_name]
        ax.plot(t_I, I_mean, color=color, linewidth=2.5, label=f'{mode_name.capitalize()}', alpha=0.85)
    
    ax.set_xlabel('Time', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
    ax.set_ylabel('Mean Information I(t)', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
    ax.set_title('Information Origin Models Comparison', 
                fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=15)
    ax.legend(loc='best', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "14_I_mode_comparison.png"), dpi=MASTER_CTRL['PLOT_DPI'])
    plt.close()
    print("[SAVED] 14_I_mode_comparison.png")
    n_I_plots += 1

n_plots = 9 + n_I_plots  # 9 core plots + 4 I-plots (11, 12, 13, 14)
print(f"\n[PHASE 6] Summary:")
print(f"  ✓ 01_energy_comparison.png")
print(f"  ✓ 02_variance_comparison.png")
print(f"  ✓ 03_entropy_comparison.png")
print(f"  ✓ 04_coherence_comparison.png")
print(f"  ✓ 05_final_energy_dist.png")
print(f"  ✓ 06_suppression_summary.png")
print(f"  ✓ 07_heisenberg_uncertainty.png")
print(f"  ✓ 08_phase_space_E_vs_S.png")
print(f"  ✓ 09_multidimensional_tracking.png")
print(f"  ✓ 11_I_evolution_emergent.png")
print(f"  ✓ 12_I_evolution_inherent.png")
print(f"  ✓ 13_I_evolution_threshold.png")
print(f"  ✓ 14_I_mode_comparison.png")
print(f"[PHASE 6] ✓ Complete: {n_plots} visualization plots generated (9 core + 4 I-origin)")
progress.update(1)

# ==========================================================================================
# PARAMETER SWEEP ANALYSIS (Optional)
# ==========================================================================================

progress.set_description("7/7: Parameter Sweep (optional)")

if MASTER_CTRL.get("ENABLE_PARAMETER_SWEEP", False):
    print("\n" + "="*80)
    print("[PHASE 7] PARAMETER SWEEP ANALYSIS")
    print("="*80)
    
    sweep_var = MASTER_CTRL["SWEEP_VARIABLE"]
    sweep_values = MASTER_CTRL["SWEEP_VALUES"]
    sweep_n_ensemble = MASTER_CTRL["SWEEP_N_ENSEMBLE"]
    
    print(f"  Sweep Variable: {sweep_var}")
    print(f"  Sweep Range:    {len(sweep_values)} points")
    print(f"  Ensemble/point: {sweep_n_ensemble} trajectories")
    
    sweep_results = []
    
    for val in tqdm(sweep_values, desc=f"  Sweeping {sweep_var}", leave=False):
        # Override parameter
        orig_val = MASTER_CTRL[sweep_var]
        MASTER_CTRL[sweep_var] = val
        
        # Sample smaller ensemble
        alphas1_sweep = sample_coherent_states(sweep_n_ensemble)
        alphas2_sweep = sample_coherent_states(sweep_n_ensemble) if two_mode else None
        I_sweep = sample_info_beta(sweep_n_ensemble)
        
        # Run WITH-LAW only
        results_sweep = []
        for i in range(sweep_n_ensemble):
            a2_s = alphas2_sweep[i] if two_mode else None
            r = run_single(alphas1_sweep[i], a2_s, I_sweep[i], enable_lockin=True)
            if r is not None:
                results_sweep.append(r)
        
        # Compute statistics
        if results_sweep:
            final_E_sweep = np.array([r["final_energy"] for r in results_sweep])
            variance_sweep = np.var(final_E_sweep)
            mean_E_sweep = np.mean(final_E_sweep)
            
            sweep_results.append({
                sweep_var: val,
                "mean_energy": float(mean_E_sweep),
                "variance": float(variance_sweep),
                "n_trajectories": len(results_sweep)
            })
        
        # Restore original value
        MASTER_CTRL[sweep_var] = orig_val
    
    if sweep_results:
        # Save sweep results
        import pandas as pd
        df_sweep = pd.DataFrame(sweep_results)
        sweep_csv = os.path.join(DATADIR, f"parameter_sweep_{sweep_var}.csv")
        df_sweep.to_csv(sweep_csv, index=False)
        print(f"[SAVED] parameter_sweep_{sweep_var}.csv")
        
        # Plot sweep
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(df_sweep[sweep_var], df_sweep['mean_energy'], 'o-', linewidth=2, markersize=8, color='purple')
        ax1.set_xlabel(f'{sweep_var}', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax1.set_ylabel('Mean Final Energy', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax1.set_title(f'Parameter Sweep: Mean Energy vs {sweep_var}', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df_sweep[sweep_var], df_sweep['variance'], 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel(f'{sweep_var}', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax2.set_ylabel('Energy Variance', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
        ax2.set_title(f'Parameter Sweep: Variance vs {sweep_var}', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'], pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, f"10_parameter_sweep_{sweep_var}.png"), dpi=MASTER_CTRL['PLOT_DPI'])
        plt.close()
        print(f"[SAVED] 10_parameter_sweep_{sweep_var}.png")
        print(f"\n[PHASE 7] Summary:")
        print(f"  ✓ parameter_sweep_{sweep_var}.csv")
        print(f"  ✓ 10_parameter_sweep_{sweep_var}.png")
        print(f"[PHASE 7] ✓ Complete: Parameter sweep finished (1 CSV + 1 PNG)")
else:
    print("\n[PHASE 7] Skipped (ENABLE_PARAMETER_SWEEP = False)")

progress.update(1)
progress.close()

# ==========================================================================================
# FINAL REPORT
# ==========================================================================================

print("\n" + "="*80)
print("PIPELINE COMPLETED: TQE Heisenberg Fluctuation Analysis")
print("="*80)
print(f"Ensemble:   {len(results_no_law)}/{N_ENSEMBLE} NO-LAW | {len(results_with_law)}/{N_ENSEMBLE} WITH-LAW")
print(f"I-Modes:    emergent ({len(results_with_law_emergent)}) | inherent ({len(results_with_law_inherent)}) | threshold ({len(results_with_law_threshold)})")
print(f"\nSuppression (emergent model):")
print(f"  Variance: {stats_comparison['SUPPRESSION_RATIOS']['variance_ratio']:.4f} | Δx·Δp: {stats_comparison['SUPPRESSION_RATIOS']['uncertainty_ratio']:.4f} | Coherence: {stats_comparison['SUPPRESSION_RATIOS']['coherence_ratio']:.4f}")
print(f"  Heisenberg: {'✓ PASSED' if stats_comparison['HEISENBERG_COMPLIANCE']['with_law_compliant'] else '✗ FAILED'}")
n_plots_final = 9 + 4 + (1 if MASTER_CTRL.get("ENABLE_PARAMETER_SWEEP", False) else 0)  # 9 core + 4 I-origin
print(f"\nSaved: 2 JSON + 3 CSV + {n_plots_final} PNG (including 4 I-origin plots)")
print(f"Seed:  {SEED} | Dir: {OUTDIR}")
print("="*80)
