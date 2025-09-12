# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_Universe_Simulation_Full_Pipeline_colab.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

import os
# Set before importing heavy numeric libs would be ideal,
# but applying here is still helpful for thread pools.
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time, json, warnings, sys, subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt

# --- Colab detection + optional Drive mount ---
IN_COLAB = ("COLAB_RELEASE_TAG" in os.environ) or ("COLAB_BACKEND_VERSION" in os.environ)
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

# --- Core deps: ensure (no heavy extras) ---
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["qutip", "pandas", "scipy", "scikit-learn", "healpy"]:
    _ensure(pkg)

import qutip as qt
import pandas as pd
from scipy.interpolate import make_interp_spline
warnings.filterwarnings("ignore")

# --- XAI stack: SHAP + LIME only ---
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "shap==0.45.0", "lime==0.2.0.1", "scikit-learn==1.5.2", "-q"])
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    
# ======================================================
# 1) MASTER CONTROLLER
# ======================================================
MASTER_CTRL = {
    # --- Core simulation ---
    "NUM_UNIVERSES":        5000,   # number of universes in Monte Carlo run
    "TIME_STEPS":           1000,    # epochs per stability run (if used elsewhere)
    "LOCKIN_EPOCHS":        700,    # epochs for law lock-in dynamics
    "EXPANSION_EPOCHS":     1000,    # epochs for expansion dynamics
    "FL_EXP_EPOCHS":        800,    # length of t>0 expansion panel
    "SEED":                 None,   # master RNG seed (auto-generated if None)
    "PIPELINE_VARIANT": "full",     # "full" = E+I pipeline, "energy_only" = E only (I disabled)

    # --- Energy distribution ---
    "E_DISTR":              "lognormal",  # energy sampling mode (future-proof)
    "E_LOG_MU":             2.5,    # lognormal mean for initial energy
    "E_LOG_SIGMA":          0.8,    # lognormal sigma for initial energy
    "E_TRUNC_LOW":          None,   # optional post-sample clamp (low)
    "E_TRUNC_HIGH":         None,   # optional post-sample clamp (high)

    # --- Information parameter I controls ---
    "I_DIM":                8,         # Hilbert space dimension for random kets
    "KL_EPS":               1e-12,     # numerical epsilon for KL/entropy
    "INFO_FUSION_MODE":     "product", # "product" | "weighted"
    "INFO_WEIGHT_KL":       0.5,       # used if INFO_FUSION_MODE == "weighted"
    "INFO_WEIGHT_SHANNON":  0.5,       # used if INFO_FUSION_MODE == "weighted"
    "I_EXPONENT":           1.0,       # optional nonlinearity: I <- I**I_EXPONENT
    "I_MIN_EPS":            0.0,       # clamp floor for I (avoid exact zeros)

    # --- E–I coupling (X definition) ---
    "X_MODE":               "product",  # "product" | "E_plus_I" | "E_times_I_pow"
    "X_I_POWER":            1.0,        # if "E_times_I_pow": X = E * (I ** X_I_POWER)
    "X_SCALE":              1.0,        # global X scaling prior to Goldilocks
    "ALPHA_I":              0.8,        # coupling factor: strength of I in E·I (heuristics)

    # --- Fluctuation / superposition module toggles & params ---
    "RUN_FLUCTUATION_BLOCK": True,  # Show the t<0 superposition, t=0 collapse, and t>0 expansion panels.
    "RUN_QUANTUM_FLUCT":     True,  # Generate the standalone quantum-fluctuation time-series panel.
    "FL_SUPER_T":            10.0,     # duration for t<0 superposition plot (arb. units)
    "FL_SUPER_DT":           0.05,     # time step for superposition time series
    "FL_SUPER_DIM":          4,        # small Hilbert dim for toy density evolution
    "FL_SUPER_NOISE":        0.06,     # depolarizing-like noise amplitude
    "FL_SUPER_KICK":         0.18,     # Strength of random unitary kicks in superposition (t<0).  
    "FL_FLUCT_OBS":           "Z",     # Observable for quantum fluctuation panel ("Z", "X", or "rand").  

    "FL_COLLAPSE_T_PRE":     0.22,     # window before t=0 (collapse)
    "FL_COLLAPSE_T_POST":    0.22,     # window after t=0
    "FL_COLLAPSE_DT":        0.002,    # time step
    "FL_COLLAPSE_PRE_SIGMA": 0.55,     # volatility before t=0
    "FL_COLLAPSE_POST_SIGMA":0.015,    # small jitter after t=0
    "FL_COLLAPSE_REVERT":    0.35,     # mean-reversion towards X_lock after t=0 (OU factor)

    "FL_EXP_DRIFT":          0.45,     # upward drift for A
    "FL_EXP_JITTER":         0.9,      # noise for A random walk
    "FL_EXP_I_JITTER":       0.04,     # small jitter for I track

    # --- Stability thresholds ---
    "REL_EPS_STABLE":       0.010,    # relative calmness threshold for stability
    "REL_EPS_LOCKIN":       5e-3,     # relative calmness threshold for lock-in (~0.5%)
    "CALM_STEPS_STABLE":    8,        # consecutive calm steps required (stable)
    "CALM_STEPS_LOCKIN":    6,        # consecutive calm steps required (lock-in)
    "MIN_LOCKIN_EPOCH":     300,      # lock-in can only occur after this epoch
    "LOCKIN_WINDOW":        8,        # rolling window size for averaging delta_rel
    "LOCKIN_ROLL_METRIC":   "mean",   # "mean" | "median" | "max" — aggregator over window
    "LOCKIN_REQUIRES_STABLE": True,   # require stable_at before checking lock-in
    "LOCKIN_MIN_STABLE_EPOCH": 0,     # require n - stable_at >= this many epochs

    # --- Goldilocks zone controls ---
    "GOLDILOCKS_MODE":      "dynamic",  # "heuristic" | "dynamic"
    "E_CENTER":             4.0,    # heuristic: energy sweet-spot center (used for X window)
    "E_WIDTH":              4.0,    # heuristic: energy sweet-spot width (used for X window)
    "GOLDILOCKS_THRESHOLD": 0.50,   # dynamic: fraction of max stability to define zone
    "GOLDILOCKS_MARGIN":    0.12,   # dynamic fallback margin around peak (±10%)
    "SIGMA_ALPHA":          1.5,    # curvature inside Goldilocks (sigma shaping)
    "OUTSIDE_PENALTY":      5.0,     # sigma multiplier outside Goldilocks zone
    "STAB_BINS":            40,     # number of bins in stability curve
    "SPLINE_K":             3,      # spline order for smoothing (3=cubic)

    # --- Noise shaping (lock-in loop) ---
    "EXP_NOISE_BASE":       0.12,   # baseline noise for updates (sigma0)
    "LL_BASE_NOISE":        8e-4,   # absolute noise floor (never go below this)
    "NOISE_DECAY_TAU":      500,    # e-folding time for noise decay (epochs)
    "NOISE_FLOOR_FRAC":     0.25,   # fraction of initial sigma preserved by decay
    "NOISE_COEFF_A":        1.0,    # per-variable noise multiplier (A)
    "NOISE_COEFF_NS":       0.10,   # per-variable noise multiplier (ns)
    "NOISE_COEFF_H":        0.20,   # per-variable noise multiplier (H)

    # --- Expansion dynamics (if/when used) ---
    "EXP_GROWTH_BASE":      1.005,  # baseline exponential growth rate
    # (EXP_NOISE_BASE above is reused as expansion amplitude baseline)

    # --- Finetune / ablation detector ---
    "RUN_FINETUNE_DETECTOR": True,  # turn on/off the comparator block
    "FT_EPS_EQ":             1e-3,  # threshold for E≈I slice (|E - I| <= eps)
    "FT_TEST_SIZE":          0.25,  # test split for the detector
    "FT_RANDOM_STATE":       42,    # reproducibility for splits
    "FT_ONLY_LOCKIN": False,        # If True, fine-tune detector uses only lock-in universes (lock_epoch >= 0)
    "FT_METRIC": "lockin",          # Use lock-in probability (P(lock-in)) instead of stability as the main metric

    # --- Best-universe visualization (lock-in only) ---
    "BEST_UNIVERSE_FIGS": 3,      # how many figures to export (typical: 1 or 5)
    "BEST_N_REGIONS": 10,         # number of region-level entropy traces
    "BEST_STAB_THRESHOLD": 3.5,   # horizontal reference line on plots
    "BEST_SAVE_CSV": True,        # also export per-universe time series as CSV
    "BEST_SEED_OFFSET": 777,      # reproducible offset for the synthetic entropy generator
    "BEST_MAX_FIGS": 50,          # safety clamp

    # --- Noise / smoothing knobs for entropy evolution ---
    "BEST_REGION_MU": 5.1,          # Target mean for region entropy traces
    "BEST_REGION_SIGMA": 0.06,      # Noise amplitude for region traces (lower = smoother)
    "BEST_GLOBAL_JITTER": 0.008,    # Small jitter added to the global entropy curve
    "BEST_SMOOTH_WINDOW": 9,        # Rolling average window size (>=1, 1 = disabled)
    "BEST_SHOW_REGIONS": True,      # If False, only plot the global entropy curve
    "BEST_ANNOTATE_LOCKIN": True,   # Draw vertical lock-in marker and annotation text
    "BEST_ANNOTATION_OFFSET": 3,    # Horizontal offset for annotation text placement

    # --- Extra robustness / docs ---
    "STAB_MIN_COUNT":       10,    # Minimum samples required in a stability bin; bins with fewer are ignored.
    "REGRESSION_MIN":       10,    # Minimum number of lock-in cases required to train/evaluate the regression.
    "MAX_SHAP_SAMPLES":     1000,  # Upper limit on samples used for SHAP plotting to keep it fast and stable.
    "SHAP_BACKGROUND_SIZE": 200,   # Size of the SHAP background (reference) dataset for model-agnostic explainers.

    # --- CMB best-universe map generation ---
    "CMB_BEST_ENABLE": True,          # Enable best-CMB PNG export
    "CMB_BEST_FIGS": 3,               # How many best CMB PNGs to export (1..5)
    "CMB_BEST_SEED_OFFSET": 909,      # Per-universe seed offset for reproducibility
    "CMB_BEST_MODE": "healpix",       # "auto" | "healpix" | "flat"

    # --- CMB map parameters ---
    "CMB_NSIDE": 256,                  # Resolution for healpy maps
    "CMB_NPIX": 512,                   # Pixel count for flat-sky maps
    "CMB_PIXSIZE_ARCMIN": 3.0,         # Pixel size in arcmin for flat-sky
    "CMB_POWER_SLOPE": 1.0,            # Power spectrum slope (Pk ~ k^-slope)
    "CMB_SMOOTH_FWHM_DEG": 0.1,        # Gaussian beam smoothing in degrees (FWHM); higher = blurrier map

    # --- CMB cold-spot detector ---
    "CMB_COLD_ENABLE":            True,                 # Enable/disable the cold-spot detector
    "CMB_COLD_TOPK":              1,                    # Top-K cold spots to keep per universe
    "CMB_COLD_SIGMA_ARCMIN":      [60, 120, 240, 480],  # Gaussian smoothing scales (arcmin)
    "CMB_COLD_MIN_SEP_ARCMIN":    30,                   # Minimal separation between spots (arcmin)
    "CMB_COLD_Z_THRESH":          -2.0,                 # Keep spots with z <= threshold (more negative = colder)
    "CMB_COLD_SAVE_PATCHES":      False,                # Flat-sky: also save small cutout PNGs around spots
    "CMB_COLD_PATCH_SIZE_ARCMIN": 200,                  # Flat-sky: patch size (arcmin) for thumbnails
    "CMB_COLD_MODE":              "healpix",            # Backend selection: "auto" | "healpix" | "flat"
    "CMB_COLD_OVERLAY":           True,                 # Draw markers on the full-sky/flat map overlays
    "CMB_COLD_MAX_OVERLAYS":      3,                    # max. cold-spot overlay PNG
    "CMB_COLD_REF_Z":             -70.0,                # Planck cold spot reference depth (µK or z-score)
    "CMB_COLD_UK_THRESH":         -70.0,                # Use µK-based flag threshold (for unit-aware cold_flag logic)

    # --- CMB Axis-of-Evil detector ---
    "CMB_AOE_ENABLE":      True,        # Enable/disable the Axis-of-Evil detector
    "CMB_AOE_LMAX":        3,           # Maximum multipole ℓ to check (ℓ=3 is standard for AoE)
    "CMB_AOE_NREALIZ":     1000,        # Number of Monte Carlo randomizations for significance (p-value)
    "CMB_AOE_OVERLAY":     True,        # Overlay principal axes on the CMB map PNG
    "CMB_AOE_MODE":        "healpix",   # Backend selection: "auto" | "healpix" | "flat"
    "CMB_AOE_SEED_OFFSET": 909,         # Per-universe seed offset to keep AoE maps reproducible
    "CMB_AOE_MAX_OVERLAYS": 3,          # maximum number of AoE overlay PNGs to generate
    "CMB_AOE_PHASE_LOCK":  True,        # do the quadrupole-axis rotation & boost
    "CMB_AOE_LMAX_BEST":   64,          # alm lmax during phase lock step
    "CMB_AOE_L23_BOOST":   1.0,         # 1.5–3.0: strength of ℓ=2,3 boost
    "AOE_REF_ANGLE_DEG":   10.0,        # reference alignment angle (Planck/WMAP ~20°)
    "AOE_P_THRESHOLD":      0.10,       # if you have p-values in cmb_aoe_summary.csv
    "AOE_ALIGN_THRESHOLD":  0.8,       # fallback if only angle is present (score = 1 - angle/180)

    # --- XAI: enable targets and outputs ---
    "XAI_ENABLE_STABILITY": True,   # run stability targets
    "XAI_ENABLE_COLD": True,        # run cold-spot targets
    "XAI_ENABLE_AOE": True,         # run AoE targets
    "XAI_SAVE_SHAP": True,          # save SHAP plots
    "XAI_SAVE_LIME": True,          # save LIME plots
    "XAI_ALLOW_CONST_FINETUNE": True,
    "XAI_LIME_K": 50,               # samples for averaged LIME
    "XAI_RUN_BOTH_FEATSETS": False, # only matching feature-set per variant
    "REGRESSION_MIN": 3,            # minimum finite rows for regression targets
    

    # --- Machine Learning / XAI ---
    "RUN_XAI":              True,   # master switch for XAI section
    "RUN_SHAP":             True,   # SHAP on/off
    "RUN_LIME":             True,   # LIME on/off
    "LIME_NUM_FEATURES":    5,      # number of features in LIME plot
    "TEST_SIZE":            0.25,   # test split ratio
    "TEST_RANDOM_STATE":    42,     # split reproducibility
    "RF_N_ESTIMATORS":      400,    # number of trees in random forest
    "RF_CLASS_WEIGHT": "balanced",  # e.g., "balanced" for skewed classes
    "SKLEARN_N_JOBS":       -1,     # parallelism for RF
    "FT_MIN_PER_SLICE":     30,     # min elems inside/outside the |E-I|<=eps slice for CI plots

        # --- XAI (Explainable AI) controls ---
    "XAI_ENABLE_STABILITY":   True,   # Enable SHAP/LIME analysis for stability classification
    "XAI_ENABLE_COLD":        True,   # Enable SHAP/LIME analysis for cold-spot detection
    "XAI_ENABLE_AOE":         True,   # Enable SHAP/LIME analysis for Axis-of-Evil
    "XAI_ENABLE_FINETUNE":    True,   # Enable SHAP/LIME analysis for fine-tuning detector

    # Feature sets for model training
    "XAI_FEATURES_E_ONLY":    ["E", "logE", "E_rank"],  
    "XAI_FEATURES_EIX":       ["E", "I", "X", "abs_E_minus_I", "logX", "dist_to_goldilocks"],

    # SHAP / LIME options
    "XAI_SAVE_SHAP":          True,   # Save SHAP outputs
    "XAI_SAVE_LIME":          True,   # Save LIME outputs
    "XAI_LIME_K":             50,     # Number of LIME samples averaged

    # Data split options
    "XAI_TEST_SIZE":          0.25,   # Test split size
    "XAI_RANDOM_STATE":       42,     # Reproducibility for train/test split

    # Targets for supervised XAI analysis
    "XAI_TARGETS": [
        "stability_cls",   # binary classification: stable vs unstable
        "lock_epoch_reg",  # regression: law lock-in epoch
        "cold_flag_cls",   # classification: cold-spot presence
        "cold_min_z_reg",  # regression: cold-spot minimum depth
        "aoe_flag_cls",    # classification: Axis-of-Evil presence
        "aoe_align_reg"    # regression: Axis-of-Evil alignment strength
    ],
    
    # --- Outputs / IO ---
    "SAVE_FIGS":            True,   # save plots to disk
    "SAVE_JSON":            True,   # save summary JSON
    "SAVE_DRIVE_COPY":      True,   # copy results to Google Drive
    "DRIVE_BASE_DIR":       "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline",
    "RUN_ID_PREFIX":        "TQE_Universe_Simulation_Full_Pipeline_",   # prefix for run_id
    "RUN_ID_FORMAT":        "%Y%m%d_%H%M%S",          # time format for run_id
    "ALLOW_FILE_EXTS":      [".png", ".fits", ".csv", ".json", ".txt", ".npy"],
    "MAX_FILES_TO_SAVE":    None,   # global cap across all allowed extensions
    "VERBOSE":              True,   # extra prints/logs

    # --- Plot toggles ---
    "PLOT_AVG_LOCKIN":      True,   # plot average lock-in curve
    "PLOT_LOCKIN_HIST":     True,   # plot histogram of lock-in epochs
    "PLOT_STABILITY_BASIC": False,  # simple stability diagnostic plot

    # --- Reproducibility knobs ---
    "USE_STRICT_SEED":      True,   # optionally seed other libs/system for strict reproducibility
    "PER_UNIVERSE_SEED_MODE": "rng" # "rng" | "np_random" — how per-universe seeds are derived
}

# --- Strict determinism knobs (optional but recommended) ---
if MASTER_CTRL.get("USE_STRICT_SEED", True):
    # Set before importing heavy numeric libs would be ideal,
    # but applying here is still helpful for thread pools.
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ======================================================
# 2) Master seed initialization (reproducibility)
# ======================================================
if MASTER_CTRL["SEED"] is None:
    MASTER_CTRL["SEED"] = int(np.random.SeedSequence().generate_state(1)[0])

master_seed = MASTER_CTRL["SEED"]

# Create both modern (rng) and legacy (np.random) RNG streams
rng = np.random.default_rng(master_seed)
np.random.seed(master_seed)  # sync legacy RNG for QuTiP calls

print(f"🎲 Using master seed: {master_seed}")

# --- Variant tag + filename helper ---
VARIANT = MASTER_CTRL.get("PIPELINE_VARIANT", "full")

def with_variant(path: str) -> str:
    """
    Insert _{VARIANT} before file extension.
    Example: figs/stability_curve.png -> figs/stability_curve_E+I.png
    """
    root, ext = os.path.splitext(path)
    if VARIANT == "energy_only":
        tag = "E-Only"
    elif VARIANT == "full":
        tag = "E+I"
    else:
        tag = VARIANT
    return f"{root}_{tag}{ext}"

# Output dirs
if VARIANT == "energy_only":
    variant_tag = "E-Only"
elif VARIANT == "full":
    variant_tag = "E+I"
else:
    variant_tag = VARIANT

run_id = MASTER_CTRL["RUN_ID_PREFIX"] + variant_tag + "_" + time.strftime(MASTER_CTRL["RUN_ID_FORMAT"])
SAVE_DIR = os.path.join(os.getcwd(), run_id)
FIG_DIR  = os.path.join(SAVE_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(path):
    """Save a figure only if SAVE_FIGS is True."""
    if not MASTER_CTRL.get("SAVE_FIGS", True):
        plt.close()
        return
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    
def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

print(f"💾 Results saved in: {SAVE_DIR}")
print(f"⚙️  Pipeline variant: {MASTER_CTRL.get('PIPELINE_VARIANT','full')}")

# ======================================================
# 3) Information parameter I = g(KL, Shannon) (fusion)
# ======================================================
def sample_information_param(dim=None):
    """
    Sample the composite information parameter I by fusing:
      - KL divergence (between two random quantum states)
      - Normalized Shannon entropy
    The fusion can be multiplicative ("product") or weighted linear.
    """
    dim = dim or MASTER_CTRL["I_DIM"]
    eps = MASTER_CTRL["KL_EPS"]

    # --- Generate two random quantum states (kets) ---
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)

    # --- Convert them to probability distributions ---
    p1 = np.abs(psi1.full().flatten())**2
    p1 /= p1.sum()  # normalize
    p2 = np.abs(psi2.full().flatten())**2
    p2 /= p2.sum()  # normalize

    # --- KL divergence (asymmetry in distributions) ---
    KL = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)))
    I_kl = KL / (1.0 + KL)  # squashing to [0,1]

    # --- Shannon entropy (normalized) ---
    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))  # normalized by log(dim)

    # --- Fusion of KL and Shannon into a single I ---
    mode = MASTER_CTRL["INFO_FUSION_MODE"]
    if mode == "weighted":
        # Weighted linear fusion (normalized weights)
        w_kl = max(0.0, MASTER_CTRL["INFO_WEIGHT_KL"])
        w_sh = max(0.0, MASTER_CTRL["INFO_WEIGHT_SHANNON"])
        s = w_kl + w_sh
        if s == 0.0:
            # Safe fallback: equal weights if both are zero
            w_kl = w_sh = 0.5
            s = 1.0
        w_kl /= s
        w_sh /= s
        I_raw = w_kl * I_kl + w_sh * I_shannon
    else:  # "product" mode
        # Multiplicative fusion (naturally bounded in [0,1])
        I_raw = I_kl * I_shannon

    # --- Post-processing: exponent & floor ---
    I = I_raw ** MASTER_CTRL["I_EXPONENT"]    # optional nonlinearity
    I = max(I, MASTER_CTRL["I_MIN_EPS"])      # enforce minimum > 0 if configured

    return float(I)

# ======================================================
# 4) Energy sampling
# ======================================================
def sample_energy(rng_local=None):
    """Sample energy using the provided RNG (per-universe) for reproducibility."""
    r = rng_local or rng
    if MASTER_CTRL["E_DISTR"] == "lognormal":
        E = float(r.lognormal(MASTER_CTRL["E_LOG_MU"], MASTER_CTRL["E_LOG_SIGMA"]))
    else:
        E = float(r.lognormal(MASTER_CTRL["E_LOG_MU"], MASTER_CTRL["E_LOG_SIGMA"]))  # fallback

    lo, hi = MASTER_CTRL["E_TRUNC_LOW"], MASTER_CTRL["E_TRUNC_HIGH"]
    if lo is not None:
        E = max(E, lo)
    if hi is not None:
        E = min(E, hi)
    return E

# ======================================================
# 5) Fluctuation / Superposition diagnostics (standalone)
# ======================================================

def _save_df_safe_local(df_in, path):
    try:
        df_in.to_csv(path, index=False)
    except Exception as e:
        print(f"[ERR] CSV save failed: {path} -> {e}")

def _pauli_like(dim: int, axis: str = "Z"):
    """
    Build a simple Pauli-like observable in higher dim:
    - "Z": diag(+1,...,+1, -1,...,-1)
    - "X": flip-like (off-diagonal ones on super- and sub-diagonal)
    - "rand": random Hermitian normalized to ||A|| ~ 1
    """
    if axis == "Z":
        half = dim // 2
        vals = np.array([1.0]*half + [-1.0]*(dim-half), dtype=float)
        return qt.Qobj(np.diag(vals))
    if axis == "X":
        M = np.zeros((dim, dim), dtype=complex)
        for i in range(dim-1):
            M[i, i+1] = 1.0
            M[i+1, i] = 1.0
        return qt.Qobj(M)
    # random Hermitian
    H = qt.rand_herm(dim)
    # scale to unit spectral norm (roughly)
    eigs = np.linalg.eigvalsh(H.full())
    scale = max(1.0, float(np.max(np.abs(eigs))))
    return (1.0/scale) * H

def simulate_superposition_series(
    T=10.0, dt=0.05, dim=4,
    noise=0.03,              # depolarizing-like noise
    kick=0.15,               # random unitary kick strength
    obs_jitter=0.02,         # tiny measurement jitter
    seed=None
):
    rgen = np.random.default_rng(seed)
    n = int(np.ceil(T/dt)) + 1
    times = np.linspace(0, T, n)

    psi = qt.rand_ket(dim)
    rho = psi.proj()

    ent_list, pur_list = [], []
    for _ in times:
        H = qt.rand_herm(dim)
        U = (1j * kick * H).expm()
        rho = U * rho * U.dag()

        z = np.clip(noise + rgen.normal(0, noise/3), 0.0, 0.25)
        mix = qt.qeye(dim) / dim
        rho = (1 - z) * rho + z * mix
        rho = rho.unit()

        S = qt.entropy_vn(rho, base=np.e)
        P = float((rho*rho).tr().real)
        S_norm = float(S / np.log(dim)) + rgen.normal(0, obs_jitter)
        P_noisy = P + rgen.normal(0, obs_jitter)

        ent_list.append(np.clip(S_norm, 0.0, 1.2))
        pur_list.append(np.clip(P_noisy, 0.0, 1.0))

    return times, np.array(ent_list), np.array(pur_list)

def simulate_quantum_fluctuation_series(
    T=6.0, dt=0.02, dim=4,
    kick=0.12, noise=0.05,
    obs_kind="Z",
    obs_jitter=0.0,
    seed=None
):
    """
    Standalone 'quantum fluctuation' panel:
    Evolve a pure state under small random unitary kicks + weak depolarizing noise.
    Track <A> and Var(A) for a chosen observable A.
    Returns: times, exp_values, variances
    """
    rgen = np.random.default_rng(seed)
    n = int(np.ceil(T/dt)) + 1
    times = np.linspace(0, T, n)

    # initial random pure state and observable
    psi = qt.rand_ket(dim)
    rho = psi.proj()
    A = _pauli_like(dim, obs_kind)

    exp_vals, variances = [], []

    for _ in times:
        # small random unitary kick
        H = qt.rand_herm(dim)
        U = (1j * kick * H).expm()
        rho = U * rho * U.dag()

        # weak depolarizing noise
        z = np.clip(noise + rgen.normal(0, noise/3), 0.0, 0.25)
        mix = qt.qeye(dim) / dim
        rho = (1 - z) * rho + z * mix
        rho = rho.unit()

        # expectation and variance for observable A
        expA = float((rho * A).tr().real)
        expA2 = float((rho * (A*A)).tr().real)
        varA = max(0.0, expA2 - expA**2)

        # optional tiny observation jitter
        if obs_jitter:
            expA += rgen.normal(0, obs_jitter)
            varA += rgen.normal(0, obs_jitter/2)

        exp_vals.append(expA)
        variances.append(max(0.0, varA))

    return times, np.array(exp_vals), np.array(variances)

def simulate_collapse_series(X_lock, t_pre=0.2, t_post=0.2, dt=0.002,
                             pre_sigma=0.5, post_sigma=0.02, revert=0.35, seed=None):
    """
    t = 0 panel: pre-collapse high-volatility OU process that snaps to X_lock at t>=0.
    """
    rgen = np.random.default_rng(seed)
    t_before = np.arange(-t_pre, 0.0, dt)
    t_after  = np.arange(0.0,  t_post+1e-12, dt)
    # pre: zero-mean noisy fluctuations around X_lock with big sigma
    x_pre = X_lock + rgen.normal(0, pre_sigma, size=len(t_before)) * (1 + 0.5*rgen.standard_normal(len(t_before)))
    # post: mean-reverting OU towards X_lock with small noise
    x = X_lock
    xs_post = []
    for _ in t_after:
        x += revert*(X_lock - x)*dt + rgen.normal(0, post_sigma)
        xs_post.append(x)
    x_after = np.array(xs_post)
    t = np.concatenate([t_before, t_after])
    x = np.concatenate([x_pre, x_after])
    return t, x

def simulate_expansion_panel(epochs=500, drift=0.4, jitter=0.9, i_jitter=0.04, seed=None):
    """
    t > 0 panel: simple stochastic growth for A and a near-flat I track.
    """
    rgen = np.random.default_rng(seed)
    A = np.empty(epochs); Itrk = np.empty(epochs)
    a = 20.0  # start amplitude
    i0 = 0.0
    for k in range(epochs):
        a = max(0.0, a + drift + rgen.normal(0, jitter))
        i0 += rgen.normal(0, i_jitter)
        Itrk[k] = i0
        A[k] = a
    return np.arange(epochs), A, Itrk
    
# ======================================================
# 6) Goldilocks noise function
# ======================================================
def sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high):
    """Goldilocks-shaped noise: outside penalty + quadratic curvature inside."""
    if E_c_low is None or E_c_high is None:
        return sigma0
    if X < E_c_low or X > E_c_high:
        return sigma0 * MASTER_CTRL["OUTSIDE_PENALTY"]
    mid = 0.5 * (E_c_low + E_c_high)
    width = max(0.5 * (E_c_high - E_c_low), 1e-12)
    dist = abs(X - mid) / width  # 0 center, 1 edges
    return sigma0 * (1 + alpha * dist**2)  # <-- use the passed-in alpha

# ======================================================
# 7) Lock-in simulation (drop-in: MASTER_CTRL-driven)
# ======================================================
def simulate_lock_in(
    X, N_epoch,
    rel_eps_stable=MASTER_CTRL["REL_EPS_STABLE"],
    rel_eps_lockin=MASTER_CTRL["REL_EPS_LOCKIN"],
    sigma0=0.2, alpha=1.0,
    E_c_low=None, E_c_high=None, rng=None
):
    """
    Simulate law stabilization and lock-in under MASTER_CTRL-driven noise model.
    - Goldilocks-shaped noise via sigma_goldilocks(...)
    - Time-decaying noise with non-zero floor
    - Per-variable noise multipliers (A, ns, H)
    - Rolling-window lock-in condition with optional prior-stable requirement
    """
    from collections import deque

    if rng is None:
        rng = np.random.default_rng()

    # State variables (arbitrary but consistent scales)
    A  = rng.normal(50, 5)
    ns = rng.normal(0.8, 0.05)
    H  = rng.normal(0.7, 0.08)

    # Tracking states
    stable_at, lockin_at = None, None
    consec_stable, consec_lockin = 0, 0

    # Rolling window for delta_rel aggregation
    window = deque(maxlen=MASTER_CTRL["LOCKIN_WINDOW"])

    # Small epsilon to avoid division-by-zero in relative deltas
    _eps = 1e-12

    # Helper for window aggregation
    def _agg(vals):
        m = MASTER_CTRL["LOCKIN_ROLL_METRIC"]
        if m == "median":
            return float(np.median(vals))
        if m == "max":
            return float(np.max(vals))
        return float(np.mean(vals))

    for n in range(1, N_epoch + 1):
        # Base noise shaped by Goldilocks window (outside penalty handled inside)
        sigma = sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high)

        # Time decay of noise (never goes to zero; clamped by LL_BASE_NOISE)
        decay = (
            MASTER_CTRL["NOISE_FLOOR_FRAC"]
            + (1 - MASTER_CTRL["NOISE_FLOOR_FRAC"])
              * np.exp(-n / MASTER_CTRL["NOISE_DECAY_TAU"])
        )
        sigma = max(MASTER_CTRL["LL_BASE_NOISE"], sigma * decay)

        # Save previous state
        A_prev, ns_prev, H_prev = A, ns, H

        # Per-variable stochastic updates
        A  += rng.normal(0, sigma * MASTER_CTRL["NOISE_COEFF_A"])
        ns += rng.normal(0, sigma * MASTER_CTRL["NOISE_COEFF_NS"])
        H  += rng.normal(0, sigma * MASTER_CTRL["NOISE_COEFF_H"])

        # Relative change with epsilon guards
        delta_rel = (
            abs(A  - A_prev) / max(abs(A_prev),  _eps) +
            abs(ns - ns_prev) / max(abs(ns_prev), _eps) +
            abs(H  - H_prev) / max(abs(H_prev),  _eps)
        ) / 3.0

        # Push into rolling window
        window.append(delta_rel)

        # --- stable check ---
        if delta_rel < rel_eps_stable:
            consec_stable += 1
            if consec_stable >= MASTER_CTRL["CALM_STEPS_STABLE"] and stable_at is None:
                stable_at = n
        else:
            consec_stable = 0

        # --- lock-in check (rolling avg + min epoch + prior stable if required) ---
        can_check_lock = (len(window) == window.maxlen) and (n >= MASTER_CTRL["MIN_LOCKIN_EPOCH"])

        if MASTER_CTRL["LOCKIN_REQUIRES_STABLE"]:
            can_check_lock = can_check_lock and (stable_at is not None)

        if MASTER_CTRL["LOCKIN_MIN_STABLE_EPOCH"] > 0 and stable_at is not None:
            can_check_lock = can_check_lock and (n - stable_at >= MASTER_CTRL["LOCKIN_MIN_STABLE_EPOCH"])

        if can_check_lock and (_agg(window) < rel_eps_lockin):
            consec_lockin += 1
            if consec_lockin >= MASTER_CTRL["CALM_STEPS_LOCKIN"] and lockin_at is None:
                lockin_at = n
        else:
            consec_lockin = 0

    # Outcomes
    is_stable = 1 if stable_at is not None else 0
    is_lockin = 1 if lockin_at is not None else 0

    return is_stable, is_lockin, (stable_at if stable_at else -1), (lockin_at if lockin_at else -1)

# ======================================================
# 8) Helpers for MC runs and dynamic Goldilocks estimation
# ======================================================
def run_mc(E_c_low=None, E_c_high=None):
    """
    Single-pass Monte Carlo run. If E_c_low/high are provided, they are used
    to shape noise via sigma_goldilocks; otherwise, no Goldilocks shaping.
    Returns a DataFrame with per-universe results.
    """
    prev_state = np.random.get_state()   # save the legacy RNG state
    try:
        rows = []
        universe_seeds = []
        pre_pairs = []

        for i in range(MASTER_CTRL["NUM_UNIVERSES"]):
            # derive per-universe seed from master rng (Generator)
            uni_seed = int(rng.integers(0, 2**32 - 1))
            universe_seeds.append(uni_seed)

            # per-universe RNG-k
            rng_uni = np.random.default_rng(uni_seed)
            np.random.seed(uni_seed)  # libs, e.g. QuTiP, use the legacy RNG

            # --- Sample energy & information parameter depending on pipeline variant ---
            E = sample_energy(rng_local=rng_uni)
            variant = MASTER_CTRL.get("PIPELINE_VARIANT", "full")

            if variant == "energy_only":
                # I disabled: force to 0.0, X depends only on E
                I = 0.0
                X = E * MASTER_CTRL["X_SCALE"]
            else:
                # Normal E+I pipeline
                I = sample_information_param()
                mode = MASTER_CTRL["X_MODE"]
                aI = MASTER_CTRL["ALPHA_I"]
                if mode == "E_plus_I":
                    X = (E + aI * I) * MASTER_CTRL["X_SCALE"]
                elif mode == "E_times_I_pow":
                    X = E * ((aI * I) ** MASTER_CTRL["X_I_POWER"]) * MASTER_CTRL["X_SCALE"]
                else:  # "product"
                    X = (E * (aI * I)) * MASTER_CTRL["X_SCALE"]

            # Simulation
            stable, lockin, stable_epoch, lock_epoch = simulate_lock_in(
                X,
                MASTER_CTRL["LOCKIN_EPOCHS"],
                sigma0=MASTER_CTRL["EXP_NOISE_BASE"],
                alpha=MASTER_CTRL.get("SIGMA_ALPHA", 1.0),
                E_c_low=E_c_low,
                E_c_high=E_c_high,
                rng=rng_uni
            )

            rec = {
                "universe_id": i,
                "seed": uni_seed,
                "E": E,
                "I": I,
                "X": X,
                "stable": stable,
                "lockin": lockin,
                "stable_epoch": stable_epoch,
                "lock_epoch": lock_epoch
            }
            rows.append(rec)

            pre_pairs.append({
                "universe_id": i,
                "E": E,
                "I": I,
                "X": X
             })

        df_out = pd.DataFrame(rows)
        # persist per-universe seeds
        pd.DataFrame({"universe_id": np.arange(len(df_out)), "seed": universe_seeds}).to_csv(
            with_variant(os.path.join(SAVE_DIR, "universe_seeds.csv")), index=False
        )

        # --- SAVE THE PRE-FLUCTUATION DATA ---
        pd.DataFrame(pre_pairs).to_csv(
            with_variant(os.path.join(SAVE_DIR, "pre_fluctuation_pairs.csv")),
            index=False
        )
              
        return df_out
    finally:
        np.random.set_state(prev_state) 

def compute_dynamic_goldilocks(df_in):
    """
    Estimate Goldilocks window dynamically from the stability curve P(stable | X).

    Returns:
        (E_c_low, E_c_high, xs, ys, xx, yy, df_tmp)
        - E_c_low, E_c_high: estimated X-window bounds
        - xs, ys: smoothed curve samples for plotting
        - xx, yy: bin means (X) and empirical stability rates
        - df_tmp: input df with a 'bin' column (filtered to valid bins)
    """
    # ---------- Guards & config ----------
    if df_in is None or len(df_in) == 0:
        # Degenerate fallback
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    # Ensure numeric X
    Xvals = pd.to_numeric(df_in["X"], errors="coerce").values
    if np.all(~np.isfinite(Xvals)):
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    # Binning parameters
    nbins = int(max(5, MASTER_CTRL.get("STAB_BINS", 40)))
    min_per_bin = int(max(1, MASTER_CTRL.get("STAB_MIN_COUNT", 10)))

    # ---------- Binning (left-closed, right-open; np.digitize(..., right=False)) ----------
    x_min = np.nanmin(Xvals)
    x_max = np.nanmax(Xvals)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        # Degenerate X range -> cannot bin
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    # Create nbins+1 equally spaced edges
    bins = np.linspace(x_min, x_max, nbins + 1)
    df_tmp = df_in.copy()

    # Digitize values: interval is (bins[i-1], bins[i]]
    idx = np.digitize(df_tmp["X"].values, bins, right=False)

    # Special fix: include x == x_min in the first bin
    idx[idx == 0] = 1

    # Assign bins back to DataFrame
    df_tmp["bin"] = idx

    # Drop out-of-range bins (shouldn't happen except for NaN)
    df_tmp = df_tmp[(df_tmp["bin"] > 0) & (df_tmp["bin"] <= nbins)]

    # ---------- Aggregate per bin ----------
    bin_stats = df_tmp.groupby("bin").agg(
        mean_X=("X", "mean"),
        stable_rate=("stable", "mean"),
        count=("stable", "size")
    ).dropna()

    # Keep only bins with enough data
    bin_stats = bin_stats[bin_stats["count"] >= min_per_bin]
    if bin_stats.empty:
        # Not enough data -> fallback
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_tmp

    # ---------- Prepare data for smoothing ----------
    # Sort by mean_X, remove duplicate X by averaging stability
    bin_stats = bin_stats.sort_values("mean_X")
    xx = bin_stats["mean_X"].values
    yy = bin_stats["stable_rate"].values

    if len(xx) > 1:
        df_u = pd.DataFrame({"x": xx, "y": yy}).groupby("x", as_index=False)["y"].mean()
        xx, yy = df_u["x"].values, df_u["y"].values

    # Clip probabilities to [0,1]
    yy = np.clip(yy, 0.0, 1.0)

    # ---------- Smoothing (spline if possible, else linear) ----------
    if len(xx) >= 2:
        xs = np.linspace(xx.min(), xx.max(), 300)
        # Spline order must be < number of unique points
        k_max = max(1, len(xx) - 1)
        k_cfg = int(MASTER_CTRL.get("SPLINE_K", 3))
        k_use = min(k_cfg, k_max)
        try:
            if k_use >= 2:
                spline = make_interp_spline(xx, yy, k=k_use)
                ys = spline(xs)
            else:
                # Too few points for cubic/quadratic: linear interpolation
                ys = np.interp(xs, xx, yy)
        except Exception:
            # Any spline failure -> linear fallback
            ys = np.interp(xs, xx, yy)
    else:
        # Only one point -> no curve
        xs, ys = xx.copy(), yy.copy()

    # Final clip to [0,1]
    ys = np.clip(ys, 0.0, 1.0)

    # ---------- Window extraction ----------
    if len(xs) == 0 or len(ys) == 0:
        return None, None, xs, ys, xx, yy, df_tmp

    peak_idx = int(np.argmax(ys))
    peak_val = float(ys[peak_idx]) if len(ys) else 0.0

    threshold = float(MASTER_CTRL.get("GOLDILOCKS_THRESHOLD", 0.5))
    half_max = threshold * peak_val

    # Handle flat/near-zero curves: use full range (with margin) as conservative fallback
    if not np.isfinite(peak_val) or peak_val <= 1e-12:
        margin = float(MASTER_CTRL.get("GOLDILOCKS_MARGIN", 0.10))
        x_mid = float(np.median(xx)) if len(xx) else float(np.median(Xvals))
        E_c_low = x_mid * (1 - margin)
        E_c_high = x_mid * (1 + margin)
        return E_c_low, E_c_high, xs, ys, xx, yy, df_tmp

    valid_mask = ys >= half_max
    if np.any(valid_mask):
        valid_region = xs[valid_mask]
        E_c_low = float(valid_region.min())
        E_c_high = float(valid_region.max())
    else:
        # Fallback: ± margin around peak x
        peak_x = float(xs[peak_idx])
        margin = float(MASTER_CTRL.get("GOLDILOCKS_MARGIN", 0.10))
        E_c_low = peak_x * (1 - margin)
        E_c_high = peak_x * (1 + margin)
        print("⚠️ No wide peak region found, using ±margin around peak.")

    return E_c_low, E_c_high, xs, ys, xx, yy, df_tmp

# ======================================================
# 9) Monte Carlo universes — single or two-phase run
# ======================================================
# Phase selection based on GOLDILOCKS_MODE:
# - "heuristic": build window from E_CENTER/E_WIDTH, run once with shaping
# - "dynamic":   run once w/o shaping to estimate window, then run again with shaping

E_c_low, E_c_high = (None, None)
df_pre = None

if MASTER_CTRL["GOLDILOCKS_MODE"] == "heuristic":
    X_center    = MASTER_CTRL["E_CENTER"] * MASTER_CTRL["ALPHA_I"]
    X_halfwidth = 0.35 * MASTER_CTRL["E_WIDTH"] * max(1e-12, MASTER_CTRL["ALPHA_I"])
    E_c_low     = max(1e-12, X_center - X_halfwidth)
    E_c_high    = X_center + X_halfwidth
    df = run_mc(E_c_low=E_c_low, E_c_high=E_c_high)

elif MASTER_CTRL["GOLDILOCKS_MODE"] == "dynamic":
    # Pass 1: estimate window from unshaped run
    print("[MC] Dynamic mode: estimating Goldilocks window...")
    df_pre = run_mc(E_c_low=None, E_c_high=None)
    E_c_low, E_c_high, _, _, _, _, _ = compute_dynamic_goldilocks(df_pre)

    def _fmt(x):
        return f"{float(x):.4f}" if (x is not None and np.isfinite(x)) else "N/A"

    print(f"[MC] Estimated Goldilocks (X) window: {_fmt(E_c_low)} .. {_fmt(E_c_high)}")

    # Pass 2: final run with window shaping (if no valid window, run without shaping)
    if (E_c_low is not None) and (E_c_high is not None):
        df = run_mc(E_c_low=E_c_low, E_c_high=E_c_high)
    else:
        print("[MC][WARN] No valid Goldilocks window estimated; running without shaping.")
        df = run_mc(E_c_low=None, E_c_high=None)

else:
    print(f"[MC][WARN] Unknown GOLDILOCKS_MODE={MASTER_CTRL['GOLDILOCKS_MODE']!r}; running without shaping.")
    df = run_mc(E_c_low=None, E_c_high=None)

# Save main run
df.to_csv(with_variant(os.path.join(SAVE_DIR, "tqe_runs.csv")), index=False)

# ======================================================
# 10) Stability curve (binned) + Goldilocks window plot
# ======================================================
E_c_low_plot, E_c_high_plot, xs, ys, xx, yy, df_binned = compute_dynamic_goldilocks(df)

plt.figure(figsize=(8,5))
plt.scatter(xx, yy, s=30, alpha=0.7, label="bin means")
if len(xs):
    plt.plot(xs, ys, "r-", lw=2, label="spline fit")
def _lbl(v, name):
    try:
        fv = float(v)
        return f"{name} = {fv:.2f}" if np.isfinite(fv) else f"{name} = N/A"
    except Exception:
        return f"{name} = N/A"

if len(xs) > 0:
    peak_idx = np.argmax(ys)
    peak_x = xs[peak_idx]
    peak_y = ys[peak_idx]
    plt.plot(peak_x, peak_y, "ro", label=f"Peak = {peak_x:.2f}")
    plt.axvline(peak_x, color="red", linestyle="--", linewidth=1.8, alpha=0.85)

    # --- Goldilocks zone boundaries (90% of peak) ---
    thr = 0.9 * peak_y
    left_idx = np.where(ys[:peak_idx] <= thr)[0]
    right_idx = np.where(ys[peak_idx:] <= thr)[0]

    if len(left_idx) > 0:
        left_x = xs[left_idx[-1]]
        plt.axvline(left_x, color="green", linestyle="--", linewidth=1.5,
                    label=f"Goldi left = {left_x:.2f}")
    if len(right_idx) > 0:
        right_x = xs[peak_idx + right_idx[0]]
        plt.axvline(right_x, color="purple", linestyle="--", linewidth=1.5,
                    label=f"Goldi right = {right_x:.2f}")


if VARIANT == "energy_only":
    plt.xlabel("X = E")
    plt.title("Goldilocks zone: stability vs E")
else:
    plt.xlabel("X = E·I")
    plt.title("Goldilocks zone: stability vs E·I")
    
plt.legend()
savefig(with_variant(os.path.join(FIG_DIR, "stability_curve.png")))

# ======================================================
# 11) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
sc = plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)")
if VARIANT == "energy_only":
    plt.ylabel("Information parameter I (disabled = 0)")
    plt.title("Universe outcomes in E (I disabled)")
else:
    plt.ylabel("Information parameter (I: KL×Shannon)")
    plt.title("Universe outcomes in (E, I) space")
cb = plt.colorbar(sc, ticks=[0, 1])
cb.set_label("Stable (0/1)")
savefig(with_variant(os.path.join(FIG_DIR, "scatter_EI.png")))

# ======================================================
# 12) Fluctuation panels (t<0, t=0, t>0) + CSV exports
# ======================================================

# ---- (0) Quantum fluctuation (standalone) ----
if MASTER_CTRL.get("RUN_QUANTUM_FLUCT", True):
    tF, expA, varA = simulate_quantum_fluctuation_series(
        T=MASTER_CTRL.get("FL_FLUCT_T", 6.0),
        dt=MASTER_CTRL.get("FL_FLUCT_DT", 0.02),
        dim=MASTER_CTRL.get("FL_FLUCT_DIM", 4),
        kick=MASTER_CTRL.get("FL_FLUCT_KICK", 0.12),
        noise=MASTER_CTRL.get("FL_FLUCT_NOISE", 0.05),
        obs_kind=MASTER_CTRL.get("FL_FLUCT_OBS", "Z"),
        obs_jitter=MASTER_CTRL.get("FL_SUPER_OBS_JITTER", 0.0),
        seed=master_seed + 10
    )

    # save CSV
    fluc_df = pd.DataFrame({"time": tF, "exp_A": expA, "var_A": varA})
    fluc_csv = with_variant(os.path.join(SAVE_DIR, "fl_fluctuation_timeseries.csv"))
    _save_df_safe_local(fluc_df, fluc_csv)

    # plot
    plt.figure(figsize=(8,5))
    plt.title("Quantum fluctuation: ⟨A⟩ and Var(A)")
    plt.plot(tF, expA, label="⟨A⟩", ls="--", alpha=0.95)
    plt.plot(tF, varA, label="Var(A)", alpha=0.95)
    plt.xlabel("time")
    plt.legend()
    savefig(with_variant(os.path.join(FIG_DIR, "fl_fluctuation.png")))

if MASTER_CTRL.get("RUN_FLUCTUATION_BLOCK", True):
    print("[FL] Generating superposition / collapse / expansion panels...")

    # Choose an X_lock reference. Prefer median from the current run; otherwise heuristic.
    if "X" in df.columns and len(df) > 0 and np.isfinite(df["X"]).any():
        X_lock = float(np.median(df["X"]))
    else:
        X_lock = MASTER_CTRL.get("E_CENTER", 4.0) * MASTER_CTRL.get("ALPHA_I", 0.8)

    # ---- (1) t<0 : superposition entropy & purity ----
    tS, ent, pur = simulate_superposition_series(
        T=MASTER_CTRL["FL_SUPER_T"],
        dt=MASTER_CTRL["FL_SUPER_DT"],
        dim=MASTER_CTRL["FL_SUPER_DIM"],
        noise=MASTER_CTRL["FL_SUPER_NOISE"],
        kick=MASTER_CTRL.get("FL_SUPER_KICK", 0.15),
        obs_jitter=MASTER_CTRL.get("FL_SUPER_OBS_JITTER", 0.02),
        seed=master_seed + 11
    )
    # save CSV
    sup_df = pd.DataFrame({"time": tS, "entropy": ent, "purity": pur})
    sup_csv = with_variant(os.path.join(SAVE_DIR, "fl_superposition_timeseries.csv"))
    _save_df_safe_local(sup_df, sup_csv)

    # plot
    plt.figure(figsize=(8,5))
    plt.title("t < 0 : Quantum superposition")
    plt.plot(tS, ent, label="Entropy", ls="--", alpha=0.9)
    plt.plot(tS, pur, label="Purity",  ls="--", alpha=0.9)
    plt.xlabel("time"); plt.legend()
    savefig(with_variant(os.path.join(FIG_DIR, "fl_superposition.png")))

    # ---- (2) t = 0 : collapse (fluctuation -> lock-in) ----
    tC, xC = simulate_collapse_series(
        X_lock,
        t_pre=MASTER_CTRL["FL_COLLAPSE_T_PRE"],
        t_post=MASTER_CTRL["FL_COLLAPSE_T_POST"],
        dt=MASTER_CTRL["FL_COLLAPSE_DT"],
        pre_sigma=MASTER_CTRL["FL_COLLAPSE_PRE_SIGMA"],
        post_sigma=MASTER_CTRL["FL_COLLAPSE_POST_SIGMA"],
        revert=MASTER_CTRL["FL_COLLAPSE_REVERT"],
        seed=master_seed + 22
    )
    col_df = pd.DataFrame({"time": tC, "X": xC, "X_lock": X_lock})
    col_csv = with_variant(os.path.join(SAVE_DIR, "fl_collapse_timeseries.csv"))
    _save_df_safe_local(col_df, col_csv)

    plt.figure(figsize=(8,5))
    plt.title("t = 0 : Collapse (lock-in of X)")
    plt.plot(tC, xC, color="gray", label="fluctuation → lock-in")
    plt.axvline(0.0, color="red")
    plt.axhline(X_lock, color="red", ls="--", label=f"Lock-in X={X_lock:.2f}")
    plt.xlabel("time")
    plt.ylabel("X = E" if VARIANT == "energy_only" else "X = E·I")
    plt.legend()
    savefig(with_variant(os.path.join(FIG_DIR, "fl_collapse.png")))

    # --- (3) t > 0 : expansion dynamics ----
    te, Atrack, Itrack = simulate_expansion_panel(
        epochs=MASTER_CTRL["FL_EXP_EPOCHS"],
        drift=MASTER_CTRL["FL_EXP_DRIFT"],
        jitter=MASTER_CTRL["FL_EXP_JITTER"],
        i_jitter=MASTER_CTRL["FL_EXP_I_JITTER"],
        seed=master_seed + 33
    )
    exp_df = pd.DataFrame({"epoch": te, "A": Atrack, "I_track": Itrack})
    exp_csv = with_variant(os.path.join(SAVE_DIR, "fl_expansion_timeseries.csv"))
    _save_df_safe_local(exp_df, exp_csv)

    plt.figure(figsize=(9,5))
    plt.title("t > 0 : Expansion dynamics")
    plt.plot(te, Atrack, label="Amplitude A")
    plt.plot(te, Itrack, label="Orientation I")

    # --- real lock-in marker, if data is available ---
    if "lock_epoch" in df.columns and (df["lock_epoch"] >= 0).any():
        lock_ep = int(np.median(df.loc[df["lock_epoch"] >= 0, "lock_epoch"]))
        plt.axvline(lock_ep, color="red", ls="--", label=f"Law lock-in ≈ {lock_ep}")

    eqA = np.percentile(Atrack, 50)
    plt.axhline(eqA, color="gray", ls="--", alpha=0.7, label="Equilibrium A")

    plt.xlabel("epoch"); plt.ylabel("Parameters"); plt.legend()
    savefig(with_variant(os.path.join(FIG_DIR, "fl_expansion.png")))

# ======================================================
# 13) Stability by I (exact zero vs eps sweep) — extended
# ======================================================
def _stability_stats(mask: pd.Series, label: str):
    total = int(mask.sum())
    stables = int(df.loc[mask, "stable"].sum())
    lockins = int((df.loc[mask, "lock_epoch"] >= 0).sum())
    return {
        "group": label,
        "n": total,
        "stable_n": stables,
        "stable_ratio": (stables / total) if total > 0 else float("nan"),
        "lockin_n": lockins,
        "lockin_ratio": (lockins / total) if total > 0 else float("nan")
    }

# Exact split
mask_I_eq0 = (df["I"] == 0.0)
mask_I_gt0 = (df["I"]  > 0.0)
zero_split_rows = [
    _stability_stats(mask_I_eq0, "I == 0"),
    _stability_stats(mask_I_gt0, "I > 0"),
]
zero_split_df = pd.DataFrame(zero_split_rows)
zero_split_path = with_variant(os.path.join(SAVE_DIR, "stability_by_I_zero.csv"))
zero_split_df.to_csv(zero_split_path, index=False)
print("\n📈 Stability by I (exact zero vs positive):")
print(zero_split_df.to_string(index=False))
if zero_split_df.loc[zero_split_df["group"] == "I == 0", "n"].iloc[0] == 0:
    print("⚠️ No exact I = 0 values in this sample; see epsilon sweep below.")

# Epsilon sweep
eps_list = [1e-12, 1e-9, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1]
eps_rows = []
for eps in eps_list:
    eps_rows.append({**_stability_stats(df["I"] <= eps, f"I <= {eps}"), "eps": eps})
    eps_rows.append({**_stability_stats(df["I"]  > eps, f"I > {eps}"),  "eps": eps})
eps_df = pd.DataFrame(eps_rows)
eps_path = with_variant(os.path.join(SAVE_DIR, "stability_by_I_eps_sweep.csv"))
eps_df.to_csv(eps_path, index=False)
print("\n📈 Epsilon sweep (near-zero thresholds, preview):")
print(eps_df.head(12).to_string(index=False))
print(f"\n📝 Saved breakdowns to:\n - {zero_split_path}\n - {eps_path}")

# ======================================================
# 14) Finetune Detector (E vs E+I(+X))
# ======================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
import itertools

# --- Finetune output dir ---
FINETUNE_DIR = FIG_DIR
os.makedirs(FINETUNE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
METRIC = MASTER_CTRL.get("FT_METRIC", "stability")  # "stability" or "lockin"

def _safe_auc(y_true, y_proba):
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return float("nan")

def _cm_counts(y_true, y_pred):
    # tp, fp, tn, fn simple counts
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true==1)&(y_pred==1)))
    tn = int(np.sum((y_true==0)&(y_pred==0)))
    fp = int(np.sum((y_true==0)&(y_pred==1)))
    fn = int(np.sum((y_true==1)&(y_pred==0)))
    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn}

def _wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - half), p, min(1.0, center + half))

def _plot_two_bar_with_ci(labels, counts, totals, title, out_png):
    # Draw bars with Wilson CI and SAVE the PNG
    lows, ps, highs = [], [], []
    for k, n in zip(counts, totals):
        lo, p, hi = _wilson_interval(k, n)
        lows.append(p - lo); highs.append(hi - p); ps.append(p)
    x = np.arange(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar(x, ps, edgecolor="black")
    plt.errorbar(x, ps, yerr=[lows, highs], fmt='none', capsize=6)
    plt.xticks(x, labels, rotation=0)
    ylabel = "P(lock-in)" if METRIC == "lockin" else "P(stable)"  # choose label by metric
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()

def _select_eps_by_share(gaps, target_share=0.20, min_n=30):
    """
    Choose eps so that about `target_share` of samples satisfy |E-I| <= eps,
    but at least `min_n` samples fall inside. Returns float eps or None.
    """
    g = np.asarray(gaps, dtype=float)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return None
    g = np.abs(g)
    g.sort()
    target_n = max(int(np.ceil(target_share * len(g))), int(min_n))
    target_n = min(target_n, len(g) - 1) if len(g) > 1 else 0
    eps = float(g[target_n])
    # ha minden 0, adjunk nagyon kicsi értéket
    if not np.isfinite(eps) or eps == 0.0:
        eps = float(g[-1]) if np.isfinite(g[-1]) else 0.0
    return eps

def _stability_vs_gap_quantiles(df_in, qbins=10, out_csv=None, out_dir=None, bar_png=None):
    dx = np.abs(df_in["E"] - df_in["I"]).values
    mask = np.isfinite(dx)
    dx = dx[mask]
    stable_arr = df_in["stable"].astype(int).values[mask]
    lockin_arr = (df_in["lock_epoch"].values[mask] >= 0).astype(int) if "lock_epoch" in df_in.columns else None
    if len(dx) == 0:
        return pd.DataFrame()

    qs = np.linspace(0, 1, qbins+1)
    edges = np.unique(np.quantile(dx, qs))
    rows = []
    for i in range(len(edges)-1):
        lo, hi = edges[i], edges[i+1] if i+1 < len(edges) else edges[-1]
        m = (dx >= lo) & (dx <= hi if i+1 == len(edges)-1 else dx < hi)
        n = int(m.sum())
        k = int(lockin_arr[m].sum()) if (METRIC == "lockin" and lockin_arr is not None) else int(stable_arr[m].sum())
        lo_ci, p_hat, hi_ci = _wilson_interval(k, n)
        rows.append({
            "q_lo": qs[i], "q_hi": qs[i+1] if i+1 < len(qs) else 1.0,
            "gap_lo": float(lo), "gap_hi": float(hi),
            "n": n, "k": k, "p": p_hat,
            "ci_lo": lo_ci, "ci_hi": hi_ci
        })

    dfq = pd.DataFrame(rows)
    if out_csv: dfq.to_csv(out_csv, index=False)

    if out_dir is None:
        out_dir = FINETUNE_DIR

    mid = 0.5*(dfq["gap_lo"] + dfq["gap_hi"])
    y = dfq["p"].values
    yerr = np.vstack([y - dfq["ci_lo"].values, dfq["ci_hi"].values - y])

    # (A) main curve
    plt.figure(figsize=(7,5))
    plt.errorbar(mid, y, yerr=yerr, fmt='-o')
    plt.title("Fine-tuning — Lock-in probability vs |E − I|")
    plt.xlabel("|E − I| (bin mid)")
    plt.ylabel("P(lock-in)" if METRIC=="lockin" else "P(stable)")
    plt.tight_layout()
    out_png1 = with_variant(os.path.join(out_dir, "finetune_gap_curve.png"))
    plt.savefig(out_png1, dpi=220, bbox_inches="tight")
    plt.close()

    # (B) adaptive split
    plt.figure(figsize=(7,5))
    plt.errorbar(mid, y, yerr=yerr, fmt='-o')
    plt.title("Fine-tuning — Lock-in probability by adaptive |E − I| split")
    plt.xlabel("|E − I| (bin mid)")
    plt.ylabel("P(lock-in)" if METRIC=="lockin" else "P(stable)")
    plt.tight_layout()
    out_png2 = with_variant(os.path.join(out_dir, "finetune_gap_adaptive.png"))
    plt.savefig(out_png2, dpi=220, bbox_inches="tight")
    plt.close()

    # quick sanity print
    print("[FT][CHECK] PNG exists:", os.path.exists(out_png1), "->", out_png1)
    print("[FT][CHECK] PNG exists:", os.path.exists(out_png2), "->", out_png2)

    # optional: push to Drive
    try:
        if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
            DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
            GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id, "figs", "Finetune")
            os.makedirs(GOOGLE_DIR, exist_ok=True)
            for pth in [bar_png, out_png1, out_png2]:
                if pth and os.path.exists(pth):
                    shutil.copy2(pth, os.path.join(GOOGLE_DIR, os.path.basename(pth)))
                    print("[FT][PUSH] ->", os.path.join(GOOGLE_DIR, os.path.basename(pth)))
    except Exception as e:
        print("[FT][PUSH][WARN]", e)

    return dfq

# ---------- Fine-tune explainability helpers (E≈I) ----------

def run_finetune_detector(df_in: pd.DataFrame):
    """
    Train/test comparator:
      - Model_E:   features = ['E']
      - Model_EIX: features = subset available from ['E','I','X'] in df
    Saves: CSV metrics, ROC (if applicable), E≈I slice CSV + barplot.
    Returns: dict (metrics + filepaths) for summary.
    """
    out = {"metrics": {}, "files": {}}
    if df_in is None or len(df_in) == 0:
        return out

    # --- Only use lock-in universes if requested (single place; not duplicated) ---
    if MASTER_CTRL.get("FT_ONLY_LOCKIN", False):
        df_in = df_in[df_in.get("lock_epoch", -1) >= 0].copy()
        if len(df_in) == 0:
            print("[FT] No lock-in universes under FT_ONLY_LOCKIN; returning empty result.")
            return out
        else:
            print(f"[FT] Using only lock-in universes: n={len(df_in)}")

    # ------------------ Classification setup ------------------
    y = df_in["stable"].astype(int).values
    cols_e   = [c for c in ["E"] if c in df_in.columns]
    cols_eix = [c for c in ["E","I","X"] if c in df_in.columns]
    X_E   = df_in[cols_e].copy()
    X_EIX = df_in[cols_eix].copy()

    # Common split indices (for fair comparison)
    idx = np.arange(len(df_in))
    Xtr_idx, Xte_idx = train_test_split(
        idx,
        test_size=MASTER_CTRL.get("FT_TEST_SIZE", 0.25),
        random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42),
        stratify=y if (len(np.unique(y))==2 and (y==0).sum()>=2 and (y==1).sum()>=2) else None
    )

    def _fit_cls(Xdf, label):
        """Fit RF classifier on train split and evaluate on test split."""
        if len(np.unique(y[Xtr_idx])) < 2:
            return {"label": label, "acc": float("nan"), "auc": float("nan"),
                    "cm": {"tp":0,"fp":0,"tn":0,"fn":0}}
        clf = RandomForestClassifier(
            n_estimators=MASTER_CTRL.get("RF_N_ESTIMATORS", 400),
            random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42),
            n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS", -1),
            class_weight=MASTER_CTRL.get("RF_CLASS_WEIGHT", None)
        )
        clf.fit(Xdf.iloc[Xtr_idx], y[Xtr_idx])
        yp = clf.predict(Xdf.iloc[Xte_idx])
        acc = float(accuracy_score(y[Xte_idx], yp))
        try:
            proba = clf.predict_proba(Xdf.iloc[Xte_idx])[:,1]
        except Exception:
            proba = yp.astype(float)
        auc  = _safe_auc(y[Xte_idx], proba)
        cm   = _cm_counts(y[Xte_idx], yp)

        # Save feature importances
        fi_df = pd.DataFrame({
            "feature": Xdf.columns,
            "importance": getattr(clf, "feature_importances_", np.zeros(len(Xdf.columns)))
        }).sort_values("importance", ascending=False)
        fi_csv = with_variant(os.path.join(FIG_DIR, f"ft_feat_importance_{label}.csv"))
        fi_df.to_csv(fi_csv, index=False)
        out["files"][f"feat_importance_{label}"] = fi_csv

        return {"label":label, "acc":acc, "auc":auc, "cm":cm}

    # ------------------ Fit classifiers ------------------
    mE   = _fit_cls(X_E,   "E")
    mEIX = _fit_cls(X_EIX, "EIX")
    met_df = pd.DataFrame([mE, mEIX])
    met_csv   = with_variant(os.path.join(SAVE_DIR, "ft_metrics_cls.csv"))
    met_df.to_json(with_variant(os.path.join(SAVE_DIR, "ft_metrics_cls.json")), indent=2)
    met_df.to_csv(met_csv, index=False)
    print("[FT] metrics_cls ->", met_csv)  
    out["files"]["metrics_cls_csv"] = met_csv
    out["metrics"]["cls"] = {"E": mE, "EIX": mEIX}

    # ------------------ Optional regression on lock_epoch ------------------
    reg_mask = df_in["lock_epoch"] >= 0
    out["metrics"]["reg"] = {}
    if reg_mask.sum() >= MASTER_CTRL.get("REGRESSION_MIN", 10):
        yr = df_in.loc[reg_mask, "lock_epoch"].values
        XR_E   = df_in.loc[reg_mask, cols_e].copy()
        XR_EIX = df_in.loc[reg_mask, cols_eix].copy()

        ridx = np.arange(len(XR_E))
        Rtr, Rte = train_test_split(
            ridx, test_size=MASTER_CTRL.get("FT_TEST_SIZE", 0.25),
            random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42)
        )

        def _fit_reg(Xdf, label):
            reg = RandomForestRegressor(
                n_estimators=MASTER_CTRL.get("RF_N_ESTIMATORS", 400),
                random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42),
                n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS", -1)
            )
            reg.fit(Xdf.iloc[Rtr], yr[Rtr])
            r2 = float(r2_score(yr[Rte], reg.predict(Xdf.iloc[Rte])))
            return {"label": label, "r2": r2}

        rE   = _fit_reg(XR_E,   "E")
        rEIX = _fit_reg(XR_EIX, "EIX")
        reg_df  = pd.DataFrame([rE, rEIX])
        reg_csv   = with_variant(os.path.join(SAVE_DIR, "ft_metrics_reg.csv"))
        reg_df.to_json(with_variant(os.path.join(SAVE_DIR, "ft_metrics_reg.json")), indent=2)
        reg_df.to_csv(reg_csv, index=False)
        print("[FT] metrics_reg ->", reg_csv)  
        out["files"]["metrics_reg_csv"] = reg_csv
        out["metrics"]["reg"] = {"E": rE, "EIX": rEIX}

    # ------------------ E≈I slice analysis ------------------
    if all(c in df_in.columns for c in ["E", "I"]):
        gaps = np.abs(df_in["E"] - df_in["I"]).values
        eps_auto = _select_eps_by_share(
            gaps, target_share=0.20,
            min_n=MASTER_CTRL.get("FT_MIN_PER_SLICE", 30)
        )
        eps = float(MASTER_CTRL.get("FT_EPS_EQ", eps_auto if eps_auto is not None else 1e-3))

        m_eq_try = (np.abs(df_in["E"] - df_in["I"]) <= eps)
        if m_eq_try.sum() == 0 or (~m_eq_try).sum() == 0:
            if eps_auto is not None:
                eps = float(eps_auto)

        m_eq  = (np.abs(df_in["E"] - df_in["I"]) <= eps)
        m_neq = ~m_eq

        def _slice(mask, name):
            n = int(mask.sum())
            # choose numerator by metric
            if METRIC == "lockin" and "lock_epoch" in df_in.columns:
                k = int((df_in.loc[mask, "lock_epoch"] >= 0).sum())   # successes
            else:
                k = int(df_in.loc[mask, "stable"].sum())
            lo_ci, p_hat, hi_ci = _wilson_interval(k, n)
            # keep both counts for CSV clarity
            st = int(df_in.loc[mask, "stable"].sum())
            lk = int((df_in.loc[mask, "lock_epoch"] >= 0).sum()) if "lock_epoch" in df_in.columns else 0
            return {
                "slice": name, "eps": eps,
                "n": n, "stable_n": st, "lockin_n": lk, "k": k, "p": p_hat,
                "ci_lo": lo_ci, "ci_hi": hi_ci
            }

        if VARIANT == "energy_only":
            s_eq  = _slice(m_eq,  f"E ≤ {eps:.3g}")
            s_neq = _slice(m_neq, f"E > {eps:.3g}")
        else:
            s_eq  = _slice(m_eq,  f"|E-I| ≤ {eps:.3g}")
            s_neq = _slice(m_neq, f"|E-I| > {eps:.3g}")

        sl_df = pd.DataFrame([s_eq, s_neq]).sort_values("slice")
        sl_csv    = with_variant(os.path.join(SAVE_DIR, "ft_slice_adaptive.csv"))
        sl_df.to_csv(sl_csv, index=False)
        print("[FT] slice ->", sl_csv)  
        out["files"]["slice_csv"] = sl_csv

        bar_png = with_variant(os.path.join(FIG_DIR, "lockin_by_eqI_bar.png"))
        print("[FT] barplot ->", bar_png) 
        title = ("Lock-in" if METRIC=="lockin" else "Stability") + \
                (" by Energy (Only E)" if VARIANT == "energy_only" else " by E≈I (adaptive epsilon)")
        # use the chosen numerator column "k"
        _plot_two_bar_with_ci(
            labels=sl_df["slice"].tolist(),
            counts=sl_df["k"].tolist(),
            totals=sl_df["n"].tolist(),
            title=title,
            out_png=bar_png
        )
        out["files"]["slice_png"] = bar_png

        q_csv     = with_variant(os.path.join(SAVE_DIR, "finetune_stability_vs_gap_quantiles.csv"))
        _stability_vs_gap_quantiles(
            df_in,
            qbins=MASTER_CTRL.get("FT_GAP_QBINS", 10),
            out_csv=q_csv,
            out_dir=FIG_DIR,
            bar_png=bar_png
        )
        out["files"]["gap_quantiles_csv"] = q_csv
        out["files"]["gap_quantiles_png_curve"]    = with_variant(os.path.join(FIG_DIR, "finetune_gap_curve.png"))
        out["files"]["gap_quantiles_png_adaptive"] = with_variant(os.path.join(FIG_DIR, "finetune_gap_adaptive.png"))
    else:
        print("[FT] Skipping E≈I slice analysis (missing E or I column).")

    # ------------------ Delta summary ------------------
    delta = {
        "acc_delta": (mEIX["acc"] - mE["acc"]) if np.isfinite(mEIX["acc"]) and np.isfinite(mE["acc"]) else float("nan"),
        "auc_delta": (mEIX["auc"] - mE["auc"]) if np.isfinite(mEIX["auc"]) and np.isfinite(mE["auc"]) else float("nan"),
    }
    if out["metrics"].get("reg"):
        rE   = out["metrics"]["reg"].get("E",   {"r2": float("nan")})
        rEIX = out["metrics"]["reg"].get("EIX", {"r2": float("nan")})
        delta["r2_delta"] = (rEIX["r2"] - rE["r2"]) if np.isfinite(rEIX["r2"]) and np.isfinite(rE["r2"]) else float("nan")

    out["metrics"]["delta"] = delta
    pd.DataFrame([delta]).to_csv(with_variant(os.path.join(SAVE_DIR, "ft_delta_summary.csv")), index=False)
    out["files"]["delta_csv"] = with_variant(os.path.join(SAVE_DIR, "ft_delta_summary.csv"))
    return out

# --- Finetune detector (E vs E+I(+X)) ---
ft_result = {}
if MASTER_CTRL.get("RUN_FINETUNE_DETECTOR", True):
    print("[FT] Running finetune/ablation detector (E vs E+I(+X)) ...")
    try:
        ft_result = run_finetune_detector(df)
        print("[FT] Done.")
    except Exception as e:
        print(f"[FT][ERR] Detector failed: {e}")

# ======================================================
# 15) Best CMB thumbnails (per top lock-in universes)
# ======================================================

global MAP_REG
if "MAP_REG" not in globals():
    MAP_REG = []

if MASTER_CTRL.get("CMB_BEST_ENABLE", True):
    print("[CMB][BEST] Generating best-CMB PNGs...")

    # Pick rendering backend: prefer healpy when available (unless forced flat)
    use_healpy = False
    mode_req = MASTER_CTRL.get("CMB_BEST_MODE", MASTER_CTRL.get("CMB_MODE", "auto"))
    if mode_req in ("auto", "healpix"):
        try:
            import healpy as hp
            use_healpy = True
        except Exception:
            use_healpy = False
            if mode_req == "healpix":
                print("[CMB][BEST][WARN] healpy not available; falling back to flat-sky.")

    # Ensure selection dataframe exists (early lock-in first, break ties by |E-I|)
    if "df_lock" in globals():
        sel_df = df_lock.copy()
    else:
        sel_df = df[df["lock_epoch"] >= 0].copy()
        if "I" in sel_df.columns:
            sel_df["_gap"] = np.abs(sel_df["E"] - sel_df["I"])
        else:
            sel_df["_gap"] = 0.0
        sel_df = sel_df.sort_values(["lock_epoch", "_gap"]).reset_index(drop=True)

    if len(sel_df) == 0:
        print("[CMB][BEST] No lock-in universes; skipping best-CMB export.")
    else:
        # Clamp how many figures to 1..5
        n_best = int(np.clip(MASTER_CTRL.get("CMB_BEST_FIGS", 3), 1, 5))

        out_dir = os.path.join(FIG_DIR, "cmb_best")
        os.makedirs(out_dir, exist_ok=True)

        # --- also save raw maps + registry for later detectors ---
        MAPS_DIR = os.path.join(FIG_DIR, "cmb_best", "maps")
        os.makedirs(MAPS_DIR, exist_ok=True)

        # Global registry of saved maps (uid, E, I, lock_epoch, mode, path)
        if "MAP_REG" not in globals():
            MAP_REG = []

        # Helper: consistent file naming with your variant tag (E+I / E-only)
        def _out_path(rank, uid):
            base = os.path.join(out_dir, f"best_cmb_rank{rank:02d}_uid{uid:05d}.png")
            return with_variant(base)

        # Rendering helpers
        def _save_close(path):
            if MASTER_CTRL.get("SAVE_FIGS", True):
                plt.savefig(path, dpi=200, bbox_inches="tight")
            plt.close()

        # Generate up to n_best CMB thumbnails
        made = []
        for r in range(n_best):
            row = sel_df.iloc[r]
            uid = int(row["universe_id"])
            u_seed = int(row["seed"]) if "seed" in row else (master_seed + uid)
            cmb_seed = u_seed + int(MASTER_CTRL.get("CMB_BEST_SEED_OFFSET", 909))
            rng_best = np.random.default_rng(cmb_seed)

            title_variant = "E-only" if VARIANT == "energy_only" else "E+I"
            E_val = float(row["E"])
            I_val = float(row["I"]) if "I" in row else 0.0
            lock_ep = int(row["lock_epoch"])

            # File path (will include _E-Only or _E+I via with_variant)
            png_path = _out_path(r+1, uid)

            if use_healpy:
                # ---- HEALPix full-sky, Planck-like Mollweide ----
                nside = int(MASTER_CTRL.get("CMB_NSIDE", 128))
                lmax  = 3 * nside - 1

                # Simple power-law C_ell ~ ell^{-slope} (start from ell>=2)
                slope = float(MASTER_CTRL.get("CMB_POWER_SLOPE", 2.0))
                ells  = np.arange(lmax + 1, dtype=float)
                Cl    = np.zeros_like(ells, dtype=float)
                Cl[2:] = 1.0 / np.maximum(ells[2:], 1.0) ** slope

                # Small per-universe amplitude jitter (keeps diversity)
                amp = float(np.exp(rng_best.normal(0.0, 0.2)))
                Cl *= amp * 1e-10    # arbitrary μK^2 scale to keep values reasonable

                # Simulate full-sky map and smooth a bit (beam)
                m_uK = hp.synfast(Cl, nside=nside, lmax=lmax, new=True, verbose=False) * 1e6
                fwhm_deg = float(MASTER_CTRL.get("CMB_SMOOTH_FWHM_DEG", 1.0))
                if fwhm_deg > 0:
                    m_uK = hp.smoothing(m_uK, fwhm=np.deg2rad(fwhm_deg), verbose=False)

                # --- AoE linking: optional phase-lock & boost for ℓ=2,3 BEFORE saving FITS ---
                if MASTER_CTRL.get("CMB_AOE_PHASE_LOCK", False):
                    LMAX_AOE = int(MASTER_CTRL.get("CMB_AOE_LMAX_BEST", 64))
                    LMAX_AOE = min(LMAX_AOE, 3*nside-1)

                    # 1) spherical harmonics of the current map
                    alm_full = hp.map2alm(m_uK, lmax=LMAX_AOE, iter=0)

                    # 2) find quadrupole & octupole principal axes (same helpers as AOE block)
                    def _axis_from_lmap(alm, nside, ell, lmax):
                        fl = np.zeros(lmax+1); fl[ell] = 1.0
                        alm_l = hp.almxfl(alm, fl)
                        m_l   = hp.alm2map(alm_l, nside=nside, verbose=False)
                        ip    = int(np.argmax(np.abs(m_l)))
                        th, ph = hp.pix2ang(nside, ip)
                        return (float(np.degrees(ph) % 360.0), float(90.0 - np.degrees(th)))

                    q_lon, q_lat = _axis_from_lmap(alm_full, nside, 2, LMAX_AOE)
                    o_lon, o_lat = _axis_from_lmap(alm_full, nside, 3, LMAX_AOE)

                    # 3) rotate so that the common axis = quadrupole axis (use Euler angles in radians)
                    alpha = np.deg2rad(q_lon)          # Z-rotation
                    beta  = np.deg2rad(90.0 - q_lat)   # Y-rotation (to pole)
                    gamma = 0.0                        # final Z-rotation
                    
                    hp.rotate_alm(alm_full, alpha, beta, gamma)

                    # 4) gently boost ℓ=2 and ℓ=3 to emphasize AoE alignment
                    l_arr, m_arr = hp.Alm.getlm(LMAX_AOE)
                    mask23 = (l_arr == 2) | (l_arr == 3)
                    alm_full[mask23] *= float(MASTER_CTRL.get("CMB_AOE_L23_BOOST", 1.5))

                    # 5) back to a map (still in μK) for saving/plot
                    m_uK = hp.alm2map(alm_full, nside=nside, verbose=False)

                # --- save HEALPix map to FITS + register ---
                map_fits = with_variant(os.path.join(MAPS_DIR, f"cmb_uid{uid:05d}.fits"))
                hp.write_map(map_fits, m_uK, overwrite=True)
                MAP_REG.append({
                    "uid": uid,
                    "E": E_val, "I": I_val, "lock_epoch": lock_ep,
                    "mode": "healpix",
                    "path": map_fits
                })

                # Robust color stretch + Planck colormap
                vmin, vmax = np.percentile(m_uK, 1), np.percentile(m_uK, 99)
                plt.figure(figsize=(9, 5.3))
                try:
                    cmap = hp.visufunc.planck_cmap()
                except Exception:
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap("coolwarm")

                hp.mollview(
                    m_uK,
                    title=f"Best CMB [{title_variant}] — uid {uid}, lock-in {lock_ep}\nE={E_val:.3g}, I={I_val:.3g}",
                    unit="μK",
                    min=vmin, max=vmax,
                    cmap=cmap  
                )
                hp.graticule(ls=":", alpha=0.5)
                _save_close(png_path)
                
            else:
                # ---- Flat-sky GRF with power-law spectrum ----
                N   = int(MASTER_CTRL.get("CMB_NPIX", 512))
                dx  = float(MASTER_CTRL.get("CMB_PIXSIZE_ARCMIN", 5.0))
                slope = float(MASTER_CTRL.get("CMB_POWER_SLOPE", 2.0))

                kx = np.fft.fftfreq(N, d=1.0) * 2*np.pi
                ky = np.fft.fftfreq(N, d=1.0) * 2*np.pi
                kx, ky = np.meshgrid(kx, ky, indexing="xy")
                kk = np.sqrt(kx**2 + ky**2); kk[0,0] = 1.0

                Pk = 1.0 / (kk ** slope)
                noise_real = rng_best.normal(size=(N, N))
                noise_imag = rng_best.normal(size=(N, N))
                F = (noise_real + 1j*noise_imag) * np.sqrt(Pk / 2.0)
                # Enforce Hermitian symmetry
                F = (F + np.conj(np.flipud(np.fliplr(F)))) / 2.0

                m = np.fft.ifft2(F).real
                m = (m - np.mean(m)) / (np.std(m) + 1e-12)

                # --- save flat map to .npy + register ---
                map_npy = with_variant(os.path.join(MAPS_DIR, f"cmb_uid{uid:05d}.npy"))
                np.save(map_npy, m)
                MAP_REG.append({
                    "uid": uid,
                    "E": E_val, "I": I_val, "lock_epoch": lock_ep,
                    "mode": "flat",
                    "path": map_npy
                })

                extent_deg = (N * dx) / 60.0
                plt.figure(figsize=(7.8, 6.4))
                plt.imshow(m, origin="lower", extent=[0, extent_deg, 0, extent_deg])
                plt.colorbar(label="μK (z-score)")
                plt.xlabel("deg"); plt.ylabel("deg")
                plt.title(f"Best CMB [{title_variant}] — uid {uid}, lock-in {lock_ep}\nE={E_val:.3g}, I={I_val:.3g}")
                _save_close(png_path)

            made.append(png_path)

        print(f"[CMB][BEST] Wrote {len(made)} PNG(s):")
        for p in made:
            print("   -", p)

        # Optional: copy to Drive like other artifacts
        try:
            if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
                DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
                GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id, os.path.relpath(os.path.join(FIG_DIR, "cmb_best"), SAVE_DIR))
                os.makedirs(GOOGLE_DIR, exist_ok=True)
                cnt = 0
                for fn in sorted(os.listdir(out_dir)):
                    if fn.endswith(".png"):
                        shutil.copy2(os.path.join(out_dir, fn), os.path.join(GOOGLE_DIR, fn))
                        cnt += 1
                print(f"[CMB][BEST] Copied {cnt} PNG(s) to Drive: {GOOGLE_DIR}")
        except Exception as e:
            print("[CMB][BEST][WARN] Drive copy failed:", e)
            print(f"[CMB][BEST] MAP_REG entries now: {len(MAP_REG)}")

   

# ======================================================
# 16) CMB Cold-spot detector (use SAVED maps from MAP_REG)
# ======================================================

if MASTER_CTRL.get("CMB_COLD_ENABLE", True):
    print("[CMB][COLD] MAP_REG length:", len(MAP_REG))

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    # --- try HEALPix; fall back to flat branches when map mode=="flat"
    try:
        import healpy as hp
        HAVE_HP = True
    except Exception:
        HAVE_HP = False

    # --- params
    COLD_DIR = os.path.join(FIG_DIR, "cmb_coldspots")
    os.makedirs(COLD_DIR, exist_ok=True)
    title_variant = "E-only" if VARIANT == "energy_only" else "E+I"
    ARC_MIN_TO_RAD = np.pi / (180.0 * 60.0)
    pix_arcmin     = float(MASTER_CTRL.get("CMB_PIXSIZE_ARCMIN", 5.0))  
    sigma_list_am  = MASTER_CTRL.get("CMB_COLD_SIGMA_ARCMIN", [30, 60, 120])
    min_sep_am     = float(MASTER_CTRL.get("CMB_COLD_MIN_SEP_ARCMIN", 45))
    z_thresh       = float(MASTER_CTRL.get("CMB_COLD_DETECT_THRESH", -60.0))
    topk           = int(MASTER_CTRL.get("CMB_COLD_TOPK", 5)) 

    # --- helper: greedy min-picking with minimum distance
    def _greedy_pick(coords, values, min_sep):
        order = np.argsort(values)  # most negative first
        picked = []
        for idx in order:
            c = coords[idx]
            if not picked:
                picked.append(idx); continue
            ok = True
            for j in picked:
                if np.linalg.norm(coords[j] - c) < min_sep:
                    ok = False; break
            if ok:
                picked.append(idx)
        return np.array(picked, dtype=int)

    # --- check
    if "MAP_REG" not in globals() or len(MAP_REG) == 0:
        print("[CMB][COLD] No MAP_REG found (no saved Best CMB maps). Skipping.")
    else:
        all_rows = []
        max_ol = int(MASTER_CTRL.get("CMB_COLD_MAX_OVERLAYS", 3))
        ol_cnt = 0

        for rec in MAP_REG:
            uid = int(rec["uid"])
            E_val = float(rec["E"]); I_val = float(rec["I"]); lock_ep = int(rec["lock_epoch"])
            mode = rec["mode"]; path = rec["path"]

            # --- only include lock-in universes in histograms ---
            if lock_ep < 0:
                continue

            if mode == "healpix":
                if not HAVE_HP:
                    print(f"[CMB][COLD][WARN] healpy missing; skip uid={uid}")
                    continue
                # --- load HEALPix map
                m = hp.read_map(path, verbose=False)
                nside = hp.get_nside(m); npix = m.size
                theta, phi = hp.pix2ang(nside, np.arange(npix))
                # unit sphere coordinates (for chord distance)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                xyz = np.vstack([x, y, z]).T
                sep_rad   = min_sep_am * ARC_MIN_TO_RAD
                min_chord = 2.0 * np.sin(sep_rad / 2.0)

                # --- multi-scale smoothing + local minima
                cand_idx, cand_z, cand_scale = [], [], []
                for sig_am in sigma_list_am:
                    sigma_rad = sig_am * ARC_MIN_TO_RAD
                    fwhm = sigma_rad * np.sqrt(8.0 * np.log(2.0))
                    ms = hp.smoothing(m, fwhm=fwhm, verbose=False)

                    # neighbor check
                    is_min = np.ones(npix, dtype=bool)
                    for p in range(npix):
                        neigh = hp.get_all_neighbours(nside, p); neigh = neigh[neigh >= 0]
                        if neigh.size and not np.all(ms[p] <= ms[neigh]):
                            is_min[p] = False

                    idx = np.where(is_min & (ms <= z_thresh))[0]
                    if idx.size:
                        cand_idx.append(idx)
                        cand_z.append(ms[idx])
                        cand_scale.extend([sig_am]*len(idx))

                if len(cand_idx):
                    idx_all   = np.concatenate(cand_idx)
                    z_all     = np.concatenate(cand_z)
                    scale_all = np.array(cand_scale, dtype=float)
                    picked = _greedy_pick(xyz[idx_all], z_all, min_chord)
                    if topk > 0: picked = picked[:topk]

                    for i, pidx in enumerate(picked, start=1):
                        pix = int(idx_all[pidx]); th, ph = float(theta[pix]), float(phi[pix])
                        lat_deg = 90.0 - np.degrees(th); lon_deg = (np.degrees(ph) % 360.0)
                        all_rows.append({
                            "universe_id": uid, "rank": i, "z_value": float(z_all[pidx]),
                            "sigma_arcmin": float(scale_all[pidx]),
                            "lon_deg": lon_deg, "lat_deg": lat_deg,
                            "E": E_val, "I": I_val, "lock_epoch": lock_ep, "variant": title_variant
                        })

                    if MASTER_CTRL.get("CMB_COLD_OVERLAY", True) and picked.size and ol_cnt < max_ol:
                        fig = plt.figure(figsize=(9.2, 5.5))
                        title = (f"Cold spots [{title_variant}] — uid {uid}, lock-in {lock_ep}\n"
                                 f"E={E_val:.3g}, I={I_val:.3g}  (top {len(picked)})")
                        hp.mollview(m, title=title, unit="μK (z-score)", norm=None, fig=fig.number)
                        hp.graticule()
                        for pidx in picked:
                            pix = int(idx_all[pidx]); th, ph = float(theta[pix]), float(phi[pix])
                            hp.projplot(th, ph, 'o', ms=6)
                        out_png = with_variant(os.path.join(COLD_DIR, f"coldspots_overlay_uid{uid:05d}.png"))
                        plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)
                        ol_cnt += 1

            else:
                # --- load FLAT map (.npy)
                m = np.load(path)  # (N,N), z-score or μK scale depending on your saving convention
                H, W = m.shape
                extent_deg = (H * pix_arcmin) / 60.0

                cand_xy, cand_z, cand_scale = [], [], []
                for sig_am in sigma_list_am:
                    sigma_px = max(0.5, sig_am / pix_arcmin)
                    ms = ndi.gaussian_filter(m, sigma=sigma_px, mode="reflect")
                    neigh = ndi.minimum_filter(ms, size=3, mode="reflect")
                    mask_min = (ms <= neigh) & (ms <= z_thresh)
                    yy, xx = np.where(mask_min)
                    if len(xx):
                        cand_xy.append(np.stack([xx, yy], axis=1))
                        cand_z.append(ms[yy, xx])
                        cand_scale.extend([sig_am]*len(xx))

                if len(cand_xy):
                    xy_all   = np.concatenate(cand_xy, axis=0)
                    z_all    = np.concatenate(cand_z)
                    scale_all = np.array(cand_scale, dtype=float)
                    min_sep_px = float(min_sep_am / pix_arcmin)
                    picked = _greedy_pick(xy_all.astype(float), z_all, min_sep_px)
                    if topk > 0: picked = picked[:topk]

                    for i, pidx in enumerate(picked, start=1):
                        x_px, y_px = xy_all[pidx]
                        lon_deg = (x_px / (H - 1)) * extent_deg
                        lat_deg = (y_px / (W - 1)) * extent_deg
                        all_rows.append({
                            "universe_id": uid, "rank": i, "z_value": float(z_all[pidx]),
                            "sigma_arcmin": float(scale_all[pidx]),
                            "x_px": int(x_px), "y_px": int(y_px),
                            "lon_deg": float(lon_deg), "lat_deg": float(lat_deg),
                            "E": E_val, "I": I_val, "lock_epoch": lock_ep, "variant": title_variant
                        })

                    if MASTER_CTRL.get("CMB_COLD_OVERLAY", True) and picked.size and ol_cnt < max_ol:
                        plt.figure(figsize=(7.8, 6.4))
                        plt.imshow(m, origin="lower", extent=[0, extent_deg, 0, extent_deg])
                        for pidx in picked:
                            x_px, y_px = xy_all[pidx]
                            x_deg = (x_px / (H - 1)) * extent_deg
                            y_deg = (y_px / (W - 1)) * extent_deg
                            plt.plot(x_deg, y_deg, 'o', ms=5)
                        plt.colorbar(label="μK (z-score)")
                        plt.xlabel("deg"); plt.ylabel("deg")
                        plt.title(f"Cold spots [{title_variant}] — uid {uid}, lock-in {lock_ep} (top {len(picked)})")
                        out_png = with_variant(os.path.join(COLD_DIR, f"coldspots_overlay_uid{uid:05d}.png"))
                        plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()
                        ol_cnt += 1

        # --- outputs
        import pandas as pd
        if len(all_rows):
            cold_df = pd.DataFrame(all_rows).sort_values(["universe_id", "rank"])
            out_csv = with_variant(os.path.join(SAVE_DIR, "cmb_coldspots_summary.csv"))
            cold_df.to_csv(out_csv, index=False)
            print("[CMB][COLD] CSV:", out_csv)

            # Depth histogram
            plt.figure(figsize=(7.6, 4.2))
            plt.hist(cold_df["z_value"].values, bins=75, edgecolor="black")
            plt.xlabel("Cold-spot z (μK vagy z-score)"); plt.ylabel("Count")
            plt.title(f"Cold-spot depth distribution [{title_variant}]")

            # --- add red dashed line for Planck reference cold spot ---
            planck_ref = MASTER_CTRL.get("CMB_COLD_REF_Z", None)  # e.g. -129.0 (μK vagy z-score)
            if planck_ref is not None:
                plt.axvline(float(planck_ref), color="red", linestyle="--", linewidth=2,
                            label=f"Planck cold spot ≈ {float(planck_ref):.2f}")
            else:
                # fallback: our dataset min
                z_star = float(np.nanmin(cold_df["z_value"].values))
                plt.axvline(z_star, color="red", linestyle="--", linewidth=2,
                            label=f"min z = {z_star:.2f}")
            plt.legend()
            
            out_hist = with_variant(os.path.join(COLD_DIR, "coldspots_z_hist.png"))
            plt.savefig(out_hist, dpi=200, bbox_inches="tight"); plt.close()
            print("[CMB][COLD] FIG:", out_hist)

            # Position heatmap (lon-lat)
            try:
                lons = cold_df["lon_deg"].values
                lats = cold_df["lat_deg"].values
                plt.figure(figsize=(7.8, 6.2))
                H2, xedges, yedges = np.histogram2d(lons, lats,
                                                    bins=(72, 36),  # ~5°x5° felbontás
                                                    range=[[0, 360], [-90, 90]])
                plt.imshow(H2.T, origin="lower", extent=[0, 360, -90, 90], aspect="auto")
                plt.colorbar(label="Előfordulás")
                plt.xlabel("Longitude (°)"); plt.ylabel("Latitude (°)")
                plt.title("Cold Spot position distribution (all selected maps)")
                out_pos = with_variant(os.path.join(COLD_DIR, "coldspots_pos_heatmap.png"))
                plt.savefig(out_pos, dpi=200, bbox_inches="tight"); plt.close()
                print("[CMB][COLD] FIG:", out_pos)
            except Exception as e:
                print("[CMB][COLD][WARN] Position heatmap failed:", e)

            # Optional Drive copy
            try:
                if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
                    DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
                    GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id, "figs", "cmb_coldspots")
                    os.makedirs(GOOGLE_DIR, exist_ok=True)
                    copied = 0
                    for fn in sorted(os.listdir(COLD_DIR)):
                        if fn.endswith(".png"):
                            shutil.copy2(os.path.join(COLD_DIR, fn), os.path.join(GOOGLE_DIR, fn))
                            copied += 1
                    shutil.copy2(out_csv, os.path.join(DRIVE_BASE, run_id, os.path.relpath(out_csv, SAVE_DIR)))
                    print(f"[CMB][COLD] Copied {copied} PNG(s) + CSV to Drive.")
            except Exception as e:
                print("[CMB][COLD][WARN] Drive copy failed:", e)
        else:
            print("[CMB][COLD] No cold spots recorded; no CSV produced.")

# ======================================================
# 17) CMB Axis-of-Evil detector (uses pre-made maps in MAP_REG)
# ======================================================

if MASTER_CTRL.get("CMB_AOE_ENABLE", True):
    print("[CMB][AOE] Running Axis-of-Evil detector (using saved maps)...")

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # healpy is required here
    try:
        import healpy as hp
    except Exception as e:
        print("[CMB][AOE][ERR] healpy is required for AoE:", e)
        hp = None

    AOE_DIR = os.path.join(FIG_DIR, "cmb_axisofevil")
    os.makedirs(AOE_DIR, exist_ok=True)

    if (hp is None) or ("MAP_REG" not in globals()) or (len(MAP_REG) == 0):
        print("[CMB][AOE] No MAP_REG / healpy missing. Skipping.")
    else:
        rows = []
        max_n = int(MASTER_CTRL.get("CMB_AOE_MAX_OVERLAYS", 3))
        ol_cnt = 0

        # --- helper: extract a single-ℓ map via almxfl (safe & lmax-agnostic) ---
        def _axis_from_lmap(alm_full, nside, ell_pick, lmax_used):
            """
            Keep only the requested multipole ell_pick from alm_full and build a map.
            Then return the longitude/latitude (deg) of the max |T| pixel and its value.
            """
            # Build an ℓ-filter: 1 at ell_pick, 0 elsewhere
            fl = np.zeros(lmax_used + 1, dtype=float)
            fl[ell_pick] = 1.0
            alm_l = hp.almxfl(alm_full, fl)                 # keep only ℓ = ell_pick
            m_l   = hp.alm2map(alm_l, nside=nside, verbose=False)
            ip    = int(np.argmax(np.abs(m_l)))
            th, ph = hp.pix2ang(nside, ip)
            return (float(np.degrees(ph) % 360.0),          # lon (deg)
                    float(90.0 - np.degrees(th)),           # lat (deg)
                    float(m_l[ip]))                         # peak value

        # Compute great-circle angle between two lon/lat points in degrees
        def _ang(lon1, lat1, lon2, lat2):
            th1 = np.radians(90.0 - lat1); ph1 = np.radians(lon1)
            th2 = np.radians(90.0 - lat2); ph2 = np.radians(lon2)
            x1 = np.sin(th1)*np.cos(ph1); y1 = np.sin(th1)*np.sin(ph1); z1 = np.cos(th1)
            x2 = np.sin(th2)*np.cos(ph2); y2 = np.sin(th2)*np.sin(ph2); z2 = np.cos(th2)
            c = np.clip(x1*x2 + y1*y2 + z1*z2, -1.0, 1.0)
            return float(np.degrees(np.arccos(c)))

        for rec in MAP_REG:    
            if rec.get("mode") != "healpix":
                continue  # AoE requires HEALPix

            uid = int(rec["uid"])
            path = rec["path"]
            E_val = float(rec["E"]); I_val = float(rec["I"]); lock_ep = int(rec["lock_epoch"])

            # --- only include lock-in universes in histograms ---
            if lock_ep < 0:
                continue

            # Load map and compute alms up to ℓ=3
            m = hp.read_map(path, verbose=False)
            nside = hp.get_nside(m)
            lmax_used = 3
            alm = hp.map2alm(m, lmax=lmax_used, iter=0)

            # Quadrupole and octupole axes (using the safe helper)
            q_lon, q_lat, q_T = _axis_from_lmap(alm, nside, 2, lmax_used)
            o_lon, o_lat, o_T = _axis_from_lmap(alm, nside, 3, lmax_used)
            angle_deg = _ang(q_lon, q_lat, o_lon, o_lat)

            rows.append({
                "universe_id": uid,
                "E": E_val, "I": I_val, "lock_epoch": lock_ep,
                "q_lon_deg": q_lon, "q_lat_deg": q_lat, "q_T_peak": q_T,
                "o_lon_deg": o_lon, "o_lat_deg": o_lat, "o_T_peak": o_T,
                "angle_deg": angle_deg
            })

            # Optional overlay
            if MASTER_CTRL.get("CMB_AOE_OVERLAY", True) and ol_cnt < max_n:
                fig = plt.figure(figsize=(9.2, 5.5))
                title = (f"Axis of Evil — uid {uid}  (angle={angle_deg:.1f}°)\n"
                         f"E={E_val:.3g}, I={I_val:.3g}, lock-in {lock_ep}")
                hp.mollview(m, title=title, unit="μK", norm=None, fig=fig.number)
                hp.graticule(ls=":", alpha=0.5)
                # Mark quadrupole and octupole axes + antipodes
                for (lon, lat, mk) in [(q_lon, q_lat, 'o'), (o_lon, o_lat, 's')]:
                    hp.projplot(lon, lat, mk, lonlat=True, ms=6)
                    hp.projplot((lon+180.0) % 360.0, -lat, mk, lonlat=True, ms=6)
                out_png = with_variant(os.path.join(AOE_DIR, f"aoe_overlay_uid{uid:05d}.png"))
                plt.savefig(out_png, dpi=200, bbox_inches="tight")
                plt.close(fig)
                ol_cnt += 1

        # Save summary
        if rows:
            df_aoe = pd.DataFrame(rows).sort_values("universe_id")
            csv_path = with_variant(os.path.join(SAVE_DIR, "cmb_aoe_summary.csv"))
            df_aoe.to_csv(csv_path, index=False)
            print("[CMB][AOE] CSV:", csv_path)

            # Angle histogram
            plt.figure(figsize=(7, 4.2))
            plt.hist(df_aoe["angle_deg"].values, bins=75, edgecolor="black")

            # --- ADD: reference line at expected Planck/WMAP alignment angle ---
            AOE_REF = float(MASTER_CTRL.get("AOE_REF_ANGLE_DEG", 10.0))  # degrees
            plt.axvline(AOE_REF, color="red", linestyle="--", linewidth=2,
                        label=f"Reference alignment ≈ {AOE_REF:.0f}°")
            plt.legend()  # show legend for the red dashed line
            
            plt.xlabel("Quadrupole–Octupole angle (deg)")
            plt.ylabel("Count")
            plt.title("Axis-of-Evil alignment angle distribution")
            hist_path = with_variant(os.path.join(AOE_DIR, "aoe_angle_hist.png"))
            plt.savefig(hist_path, dpi=200, bbox_inches="tight")
            plt.close()
            print("[CMB][AOE] FIG:", hist_path)

            # Optional Drive copy
            try:
                if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
                    DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
                    GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id, "figs", "cmb_axisofevil")
                    os.makedirs(GOOGLE_DIR, exist_ok=True)
                    copied = 0
                    for fn in sorted(os.listdir(AOE_DIR)):
                        if fn.endswith(".png"):
                            shutil.copy2(os.path.join(AOE_DIR, fn), os.path.join(GOOGLE_DIR, fn))
                            copied += 1
                    shutil.copy2(csv_path, os.path.join(DRIVE_BASE, run_id, os.path.relpath(csv_path, SAVE_DIR)))
                    print(f"[CMB][AOE] Copied {copied} PNG(s) + CSV to Drive.")
            except Exception as e:
                print("[CMB][AOE][WARN] Drive copy failed:", e)
        else:
            print("[CMB][AOE] No AoE rows collected; nothing to save.")

# ======================================================
# 18) Multi-target XAI (SHAP+LIME)
# ======================================================

import os, json, shutil, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

# ---------- tiny helpers ----------
def _safe_read_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception as e:
        print(f"[JOIN][WARN] Failed to read {path}: {e}")
    return None

def _wilson_interval(k, n, z=1.96):
    if n <= 0: return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (max(0.0, center - half), p, min(1.0, center + half))

def _plot_two_bar_with_ci(labels, counts, totals, title, out_png, ylabel="Probability"):
    lows, ps, highs = [], [], []
    for k, n in zip(counts, totals):
        lo, p, hi = _wilson_interval(k, n)
        lows.append(p - lo); highs.append(hi - p); ps.append(p)
    x = np.arange(len(labels))
    plt.figure(figsize=(6,5))
    plt.bar(x, ps, edgecolor="black")
    plt.errorbar(x, ps, yerr=[lows, highs], fmt='none', capsize=6)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title); plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight"); plt.close()

# ======================================================
# (A) JOIN: attach Cold / AoE / Finetune 
# ======================================================
cold_csv = with_variant(os.path.join(SAVE_DIR, "cmb_coldspots_summary.csv"))
aoe_csv = with_variant(os.path.join(SAVE_DIR, "cmb_aoe_summary.csv"))
if not os.path.exists(aoe_csv):
    aoe_csv = with_variant(os.path.join(SAVE_DIR, "cmb_axis_of_evil_summary.csv"))

ft_csv   = with_variant(os.path.join(SAVE_DIR, "ft_delta_summary.csv"))

cold_df = _safe_read_csv(cold_csv)
aoe_df  = _safe_read_csv(aoe_csv)
ft_df   = _safe_read_csv(ft_csv)

df_join = df.copy()

# Cold aggregation
if cold_df is not None and not cold_df.empty:
    agg = (cold_df.groupby("universe_id")
                  .agg(cold_min_z=("z_value","min"),
                       cold_mean_z_topk=("z_value","mean"),
                       cold_count=("z_value",lambda s: int(np.sum(np.isfinite(s)))),
                       cold_sigma_best=("sigma_arcmin","median"))
                  .reset_index())
    df_join = df_join.merge(agg, on="universe_id", how="left")
    vals = df_join["cold_min_z"].values
    medabs = np.nanmedian(np.abs(vals))
    # unit-aware threshold
    if np.isfinite(medabs) and medabs > 20:
        z_thr = float(MASTER_CTRL.get("CMB_COLD_UK_THRESH", -70.0))
    else:
        z_thr = float(MASTER_CTRL.get("CMB_COLD_Z_THRESH", -2.5))
    df_join["cold_flag"] = (df_join["cold_min_z"] <= z_thr).astype(int)
else:
    print("[JOIN] No cold summary; skipping cold metrics.")

# AoE attach (score from angle if needed)
if aoe_df is not None and not aoe_df.empty:
    if "aoe_align_score" not in aoe_df.columns and "angle_deg" in aoe_df.columns:
        aoe_df["aoe_align_score"] = 1.0 - (aoe_df["angle_deg"].astype(float) / 180.0)
    keep = [c for c in ["universe_id","angle_deg","aoe_align_score","aoe_pvalue"] if c in aoe_df.columns]
    aoe_compact = aoe_df[keep].drop_duplicates("universe_id")
    df_join = df_join.merge(aoe_compact, on="universe_id", how="left")
    p_thr = float(MASTER_CTRL.get("AOE_P_THRESHOLD", 0.05))
    if "aoe_pvalue" in df_join.columns:
        df_join["aoe_flag"] = (df_join["aoe_pvalue"] < p_thr).astype(int)
    elif "aoe_align_score" in df_join.columns:
        thr = float(MASTER_CTRL.get("AOE_ALIGN_THRESHOLD", 0.9))
        df_join["aoe_flag"] = (df_join["aoe_align_score"] >= thr).astype(int)
else:
    print("[JOIN] No AoE summary; skipping AoE metrics.")

# Finetune deltas broadcast (if exists)
if ft_df is not None and not ft_df.empty:
    for col in ["acc_delta","auc_delta","r2_delta"]:
        if col in ft_df.columns:
            df_join[f"ft_{col}"] = float(ft_df[col].iloc[0])
else:
    for col in ["ft_acc_delta","ft_auc_delta","ft_r2_delta"]:
        if col not in df_join.columns:
            df_join[col] = np.nan

# Engineered features (stay compatible)
df_join["abs_E_minus_I"] = np.abs(df_join.get("E", np.nan) - df_join.get("I", 0.0))
df_join["logE"] = np.log(df_join.get("E", np.nan) + 1e-12)
df_join["logX"] = np.log(df_join.get("X", np.nan) + 1e-12)
df_join["E_rank"] = df_join["E"].rank(pct=True)
df_join["X_rank"] = df_join["X"].rank(pct=True)

# Distance to Goldilocks (if window captured in 'summary')
try:
    x_low  = summary.get("goldilocks_window_used",{}).get("X_low", None)
    x_high = summary.get("goldilocks_window_used",{}).get("X_high", None)
except Exception:
    x_low = x_high = None
if x_low is not None and x_high is not None and np.isfinite(x_low) and np.isfinite(x_high):
    mid = 0.5*(float(x_low)+float(x_high))
    width = max(1e-12, 0.5*(float(x_high)-float(x_low)))
    df_join["dist_to_goldilocks"] = np.abs(df_join["X"] - mid) / width
else:
    df_join["dist_to_goldilocks"] = np.nan

# Fallback 'stable' if missing
if "stable" not in df_join.columns and "lock_epoch" in df_join.columns:
    df_join["stable"] = (df_join["lock_epoch"] >= 0).astype(int)

# Persist joined view (debug)
joined_csv = with_variant(os.path.join(SAVE_DIR, "metrics_joined.csv"))
df_join.to_csv(joined_csv, index=False)
print("[JOIN] Wrote:", joined_csv)

# Working table for XAI:
df_xai = df_join

# ======================================================
# (B) FINETUNE detector (E vs E+I(+X))  
# ======================================================
def _safe_auc(y_true, y_proba):
    try:
        if len(np.unique(y_true)) < 2: return float("nan")
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return float("nan")

def _cm_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true==1)&(y_pred==1)))
    tn = int(np.sum((y_true==0)&(y_pred==0)))
    fp = int(np.sum((y_true==0)&(y_pred==1)))
    fn = int(np.sum((y_true==1)&(y_pred==0)))
    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn}

def _select_eps_by_share(gaps, target_share=0.20, min_n=30):
    g = np.asarray(gaps, dtype=float); g = g[np.isfinite(g)]
    if g.size == 0: return None
    g = np.abs(g); g.sort()
    target_n = max(int(np.ceil(target_share * len(g))), int(min_n))
    target_n = min(target_n, len(g)-1) if len(g) > 1 else 0
    eps = float(g[target_n]) if len(g) else None
    if eps is None or not np.isfinite(eps) or eps == 0.0:
        eps = float(g[-1]) if len(g) else 1e-3
    return eps

def run_finetune_detector(df_in: pd.DataFrame):
    out = {"metrics": {}, "files": {}}
    if df_in is None or len(df_in) == 0: return out

    METRIC = MASTER_CTRL.get("FT_METRIC", "stability")  # "stability"|"lockin"
    y = df_in["stable"].astype(int).values
    cols_e   = [c for c in ["E"] if c in df_in.columns]
    cols_eix = [c for c in ["E","I","X"] if c in df_in.columns]
    X_E, X_EIX = df_in[cols_e].copy(), df_in[cols_eix].copy()

    idx = np.arange(len(df_in))
    Xtr_idx, Xte_idx = train_test_split(
        idx,
        test_size=MASTER_CTRL.get("FT_TEST_SIZE", 0.25),
        random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42),
        stratify=y if (len(np.unique(y))==2 and (y==0).sum()>=2 and (y==1).sum()>=2) else None
    )

    def _fit_cls(Xdf, label):
        clf = RandomForestClassifier(
            n_estimators=MASTER_CTRL.get("RF_N_ESTIMATORS", 400),
            random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42),
            n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS", -1),
            class_weight=MASTER_CTRL.get("RF_CLASS_WEIGHT", None)
        )
        clf.fit(Xdf.iloc[Xtr_idx], y[Xtr_idx])
        yp = clf.predict(Xdf.iloc[Xte_idx])
        acc = float(accuracy_score(y[Xte_idx], yp))
        try:
            proba = clf.predict_proba(Xdf.iloc[Xte_idx])[:, 1]
        except Exception:
            proba = yp.astype(float)
        auc = _safe_auc(y[Xte_idx], proba)
        cm  = _cm_counts(y[Xte_idx], yp)
        return {"label": label, "acc": acc, "auc": auc, "cm": cm}, clf
        # save FI
        fi_df = pd.DataFrame({"feature": Xdf.columns,
                              "importance": getattr(clf, "feature_importances_", np.zeros(len(Xdf.columns)))}
                             ).sort_values("importance", ascending=False)
        fi_csv = with_variant(os.path.join(FIG_DIR, f"ft_feat_importance_{label}.csv"))
        fi_df.to_csv(fi_csv, index=False)
        out["files"][f"feat_importance_{label}"] = fi_csv
        return {"label":label, "acc":acc, "auc":auc, "cm":cm}

    mE,   clf_E   = _fit_cls(X_E,   "E")
    mEIX, clf_EIX = _fit_cls(X_EIX, "EIX")
    met_df = pd.DataFrame([mE, mEIX])
    met_df.to_csv(with_variant(os.path.join(SAVE_DIR, "ft_metrics_cls.csv")), index=False)
    met_df.to_json(with_variant(os.path.join(SAVE_DIR, "ft_metrics_cls.json")), indent=2)

    # --- Row-level finetune targets for XAI (classification) ---
    proba_E   = clf_E.predict_proba(X_E.iloc[Xte_idx])[:,1]     
    proba_EIX = clf_EIX.predict_proba(X_EIX.iloc[Xte_idx])[:,1]

    true = y[Xte_idx].astype(int)
    pred_E   = (proba_E   >= 0.5).astype(int)
    pred_EIX = (proba_EIX >= 0.5).astype(int)

    ft_proba_gain = np.where(true==1, proba_EIX - proba_E, (1-proba_EIX) - (1-proba_E))
    ft_cls_gain   = (pred_EIX==true).astype(int) - (pred_E==true).astype(int)  # +1 / 0 / -1

    df_xai.loc[df_xai.index[Xte_idx], "ft_proba_gain"] = ft_proba_gain
    df_xai.loc[df_xai.index[Xte_idx], "ft_cls_gain"]   = ft_cls_gain

    # Optional regression on lock_epoch
    out["metrics"]["reg"] = {}
    reg_mask = "lock_epoch" in df_in.columns and (df_in["lock_epoch"] >= 0)
    if isinstance(reg_mask, pd.Series) and reg_mask.sum() >= MASTER_CTRL.get("REGRESSION_MIN", 10):
        yr = df_in.loc[reg_mask, "lock_epoch"].values
        XR_E, XR_EIX = df_in.loc[reg_mask, cols_e].copy(), df_in.loc[reg_mask, cols_eix].copy()
        ridx = np.arange(len(XR_E))
        Rtr, Rte = train_test_split(ridx,
                                    test_size=MASTER_CTRL.get("FT_TEST_SIZE", 0.25),
                                    random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42))
        def _fit_reg(Xdf, label):
            reg = RandomForestRegressor(
                n_estimators=MASTER_CTRL.get("RF_N_ESTIMATORS", 400),
                random_state=MASTER_CTRL.get("FT_RANDOM_STATE", 42),
                n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS", -1)
            )
            reg.fit(Xdf.iloc[Rtr], yr[Rtr]); r2 = float(r2_score(yr[Rte], reg.predict(Xdf.iloc[Rte])))
            return {"label":label, "r2":r2}
        rE, rEIX = _fit_reg(XR_E, "E"), _fit_reg(XR_EIX, "EIX")
        pd.DataFrame([rE, rEIX]).to_csv(with_variant(os.path.join(SAVE_DIR, "ft_metrics_reg.csv")), index=False)
        out["metrics"]["reg"] = {"E":rE,"EIX":rEIX}

    # E≈I slice + adaptive eps
    if all(c in df_in.columns for c in ["E","I"]):
        gaps = np.abs(df_in["E"] - df_in["I"]).values
        eps_auto = _select_eps_by_share(gaps, target_share=0.20,
                                        min_n=MASTER_CTRL.get("FT_MIN_PER_SLICE", 30))
        eps = float(MASTER_CTRL.get("FT_EPS_EQ", eps_auto if eps_auto is not None else 1e-3))
        m_eq = (np.abs(df_in["E"] - df_in["I"]) <= eps); m_neq = ~m_eq
        def _slice(mask, name):
            n = int(mask.sum())
            st = int(df_in.loc[mask, "stable"].sum()) if "stable" in df_in.columns else 0
            lk = int((df_in.loc[mask, "lock_epoch"] >= 0).sum()) if "lock_epoch" in df_in.columns else 0
            k = lk if MASTER_CTRL.get("FT_METRIC","stability")=="lockin" else st
            lo,p,hi = _wilson_interval(k,n)
            return {"slice":name,"eps":eps,"n":n,"k":k,"p":p,"ci_lo":lo,"ci_hi":hi}
        lab_eq  = ("E ≤ " if VARIANT=="energy_only" else "|E−I| ≤ ") + f"{eps:.3g}"
        lab_neq = ("E > " if VARIANT=="energy_only" else "|E−I| > ") + f"{eps:.3g}"
        sl_df = pd.DataFrame([_slice(m_eq,lab_eq), _slice(m_neq,lab_neq)]).sort_values("slice")
        sl_csv = with_variant(os.path.join(SAVE_DIR, "ft_slice_adaptive.csv")); sl_df.to_csv(sl_csv, index=False)
        bar_png = with_variant(os.path.join(FIG_DIR, "lockin_by_eqI_bar.png"))
        title = ("Lock-in" if MASTER_CTRL.get("FT_METRIC","stability")=="lockin" else "Stability") + \
                (" by Energy (Only E)" if VARIANT=="energy_only" else " by E≈I (adaptive epsilon)")
        _plot_two_bar_with_ci(sl_df["slice"].tolist(), sl_df["k"].tolist(), sl_df["n"].tolist(),
                              title=title, out_png=bar_png,
                              ylabel="P(lock-in)" if MASTER_CTRL.get("FT_METRIC","stability")=="lockin" else "P(stable)")
        # quantile curve pngs
        qs = np.linspace(0,1,MASTER_CTRL.get("FT_GAP_QBINS",10)+1)
        edges = np.unique(np.quantile(np.abs(gaps[np.isfinite(gaps)]), qs))
        mids, p, lo, hi, n = [], [], [], [], []
        for i in range(len(edges)-1):
            lo_e, hi_e = edges[i], edges[i+1]
            m = (gaps >= lo_e) & (gaps < hi_e if i+1<len(edges)-1 else gaps <= hi_e)
            nn = int(m.sum()); kk = int(df_in.loc[m, "stable"].sum()) if MASTER_CTRL.get("FT_METRIC","stability")=="stability" else int((df_in.loc[m, "lock_epoch"]>=0).sum())
            lo_ci, p_hat, hi_ci = _wilson_interval(kk, nn)
            mids.append(0.5*(lo_e+hi_e)); p.append(p_hat); lo.append(lo_ci); hi.append(hi_ci); n.append(nn)
        y = np.array(p); yerr = np.vstack([y - np.array(lo), np.array(hi) - y])
        plt.figure(figsize=(7,5)); plt.errorbar(mids, y, yerr=yerr, fmt='-o')
        plt.title("Fine-tune — probability vs |E−I|"); plt.xlabel("|E−I| (bin mid)")
        plt.ylabel("P(lock-in)" if MASTER_CTRL.get("FT_METRIC","stability")=="lockin" else "P(stable)")
        plt.tight_layout(); plt.savefig(with_variant(os.path.join(FIG_DIR, "finetune_gap_curve.png")), dpi=220, bbox_inches="tight"); plt.close()
        plt.figure(figsize=(7,5)); plt.errorbar(mids, y, yerr=yerr, fmt='-o')
        plt.title("Fine-tune — probability by adaptive |E−I| split"); plt.xlabel("|E−I| (bin mid)")
        plt.ylabel("P(lock-in)" if MASTER_CTRL.get("FT_METRIC","stability")=="lockin" else "P(stable)")
        plt.tight_layout(); plt.savefig(with_variant(os.path.join(FIG_DIR, "finetune_gap_adaptive.png")), dpi=220, bbox_inches="tight"); plt.close()

    # deltas JSON/CSV
    delta = {
        "acc_delta": (mEIX["acc"] - mE["acc"]) if np.isfinite(mEIX.get("acc",np.nan)) and np.isfinite(mE.get("acc",np.nan)) else float("nan"),
        "auc_delta": (mEIX["auc"] - mE["auc"]) if np.isfinite(mEIX.get("auc",np.nan)) and np.isfinite(mE.get("auc",np.nan)) else float("nan"),
    }
    if out["metrics"].get("reg"):
        rE   = out["metrics"]["reg"].get("E",   {"r2": np.nan})
        rEIX = out["metrics"]["reg"].get("EIX", {"r2": np.nan})
        delta["r2_delta"] = (rEIX["r2"] - rE["r2"]) if np.isfinite(rEIX["r2"]) and np.isfinite(rE["r2"]) else float("nan")
    pd.DataFrame([delta]).to_csv(with_variant(os.path.join(SAVE_DIR, "ft_delta_summary.csv")), index=False)
    return delta

if MASTER_CTRL.get("RUN_FINETUNE_DETECTOR", True) and MASTER_CTRL.get("XAI_ENABLE_FINETUNE", True):
    try:
        _ = run_finetune_detector(df_join)
        print("[FT] Finetune detector finished.")
        # Refresh df_xai with (possibly) new ft_delta_summary
        ft_df = _safe_read_csv(with_variant(os.path.join(SAVE_DIR, "ft_delta_summary.csv")))
        if ft_df is not None and not ft_df.empty:
            for col in ["acc_delta","auc_delta","r2_delta"]:
                if col in ft_df.columns:
                    df_xai[f"ft_{col}"] = float(ft_df[col].iloc[0])
    except Exception as e:
        print("[FT][WARN] detector failed:", e)

# ======================================================
# (C) XAI — targets + SHAP/LIME (E-only vs E+I(+X)), foldered outputs
# ======================================================

# Safe SHAP/LIME availability (Colab may have numba/np mismatch)
try:
    import shap; SHAP_OK = True
except Exception as e:
    print("[XAI][WARN] SHAP disabled:", e); SHAP_OK = False
try:
    from lime.lime_tabular import LimeTabularExplainer; LIME_OK = True
except Exception as e:
    print("[XAI][WARN] LIME disabled:", e); LIME_OK = False

# Controls
XAI_ENABLE_STAB = bool(MASTER_CTRL.get("XAI_ENABLE_STABILITY", True))
XAI_ENABLE_COLD = bool(MASTER_CTRL.get("XAI_ENABLE_COLD", True))
XAI_ENABLE_AOE  = bool(MASTER_CTRL.get("XAI_ENABLE_AOE", True))
XAI_ALLOW_CONST_FINETUNE = bool(MASTER_CTRL.get("XAI_ALLOW_CONST_FINETUNE", True))
SAVE_SHAP = bool(MASTER_CTRL.get("XAI_SAVE_SHAP", True)) and SHAP_OK
SAVE_LIME = bool(MASTER_CTRL.get("XAI_SAVE_LIME", True)) and LIME_OK
TEST_SIZE = float(MASTER_CTRL.get("XAI_TEST_SIZE", MASTER_CTRL.get("TEST_SIZE", 0.25)))
RSTATE    = int(MASTER_CTRL.get("XAI_RANDOM_STATE", MASTER_CTRL.get("TEST_RANDOM_STATE", 42)))
LIME_K    = int(MASTER_CTRL.get("XAI_LIME_K", 50))
REGRESSION_MIN = int(MASTER_CTRL.get("REGRESSION_MIN", MASTER_CTRL.get("XAI_REGRESSION_MIN", 10)))
FEATS_E_ONLY = MASTER_CTRL.get("XAI_FEATURES_E_ONLY", ["E","logE","E_rank"])
FEATS_EIX    = MASTER_CTRL.get("XAI_FEATURES_EIX",
                               ["E","I","X","abs_E_minus_I","logX","dist_to_goldilocks","E_rank","X_rank"])
RUN_BOTH = bool(MASTER_CTRL.get("XAI_RUN_BOTH_FEATSETS", False))
variant_title = "E-only" if VARIANT=="energy_only" else "E+I(+X)"

# Directories per target
XAI_FIG_DIR  = os.path.join(FIG_DIR, "xai")
XAI_SAVE_DIR = os.path.join(SAVE_DIR, "xai")
os.makedirs(XAI_FIG_DIR,  exist_ok=True)
os.makedirs(XAI_SAVE_DIR, exist_ok=True)
SUBDIRS = {
    "stability_cls": ("stability","XAI — Stability (classification)"),
    "lock_epoch_reg":("lockin",   "XAI — Lock-in epoch (regression)"),
    "cold_flag_cls": ("cold",     "XAI — Cold-spot anomaly (classification)"),
    "cold_min_z_reg":("cold",     "XAI — Cold-spot depth (regression)"),
    "aoe_flag_cls":  ("aoe",      "XAI — AoE anomaly (classification)"),
    "aoe_align_reg": ("aoe",      "XAI — AoE alignment score (regression)"),
    "finetune_acc_delta": ("finetune","XAI — Fine-tune ΔACC (regression)"),
    "finetune_auc_delta": ("finetune","XAI — Fine-tune ΔAUC (regression)"),
    "finetune_r2_delta":  ("finetune","XAI — Fine-tune ΔR2  (regression)"),
}
def _mk_dirs_for_target(tname):
    sub = SUBDIRS.get(tname, ("misc", tname))[0]
    fig_dir  = os.path.join(XAI_FIG_DIR,  sub)
    save_dir = os.path.join(XAI_SAVE_DIR, sub)
    os.makedirs(fig_dir,  exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    return fig_dir, save_dir
def _file_prefix(fig_dir, save_dir, target_name, featset):
    tag = "Eonly" if featset=="E_ONLY" else "EIX"
    return (with_variant(os.path.join(fig_dir,  f"{target_name}__{tag}")),
            with_variant(os.path.join(save_dir, f"{target_name}__{tag}")))
def _title_with_feat(base_title, featset):
    return f"{base_title} [{'E-only' if featset=='E_ONLY' else 'E+I(+X)'}]"

def _ensure_cols(df_in, cols): return [c for c in cols if c in df_in.columns]

def _shap_summary(model, X_plot, feat_names, out_png, fig_title=None):
    try:
        expl = shap.TreeExplainer(model, feature_perturbation="interventional", model_output="raw")
        sv = expl.shap_values(X_plot, check_additivity=False)
    except Exception:
        expl = shap.Explainer(model, X_plot)
        sv = expl(X_plot).values

    if isinstance(sv, list):  # for binary classification, shap returns [neg_class, pos_class]
        sv = sv[-1]

    # --- CLEAN TYPOGRAPHY FIX ---
    plt.close('all')  # reset figure state to avoid overlap
    fig = plt.figure(figsize=(9, 6))  # bigger canvas for clarity
    shap.summary_plot(sv, X_plot.values, feature_names=feat_names, show=False)

    if fig_title:
        # use suptitle with padding to avoid overlap with the plot
        fig.suptitle(fig_title, y=0.99, fontsize=13)

    # leave room for the title
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

# Target list
targets = []
if XAI_ENABLE_STAB and "stable" in df_xai.columns:
    targets.append(("stability_cls","cls","stable",None))
if XAI_ENABLE_STAB and "lock_epoch" in df_xai.columns:
    m = (df_xai["lock_epoch"] >= 0)
    if int(m.sum()) >= REGRESSION_MIN:
        targets.append(("lock_epoch_reg","reg","lock_epoch",m))
if XAI_ENABLE_COLD and "cold_flag" in df_xai.columns:
    targets.append(("cold_flag_cls","cls","cold_flag",None))
if XAI_ENABLE_COLD and "cold_min_z" in df_xai.columns:
    m = np.isfinite(df_xai["cold_min_z"])
    if int(m.sum()) >= REGRESSION_MIN:
        targets.append(("cold_min_z_reg","reg","cold_min_z",m))
if XAI_ENABLE_AOE and "aoe_flag" in df_xai.columns:
    targets.append(("aoe_flag_cls","cls","aoe_flag",None))
if XAI_ENABLE_AOE and "aoe_align_score" in df_xai.columns:
    m = np.isfinite(df_xai["aoe_align_score"])
    if int(m.sum()) >= REGRESSION_MIN:
        targets.append(("aoe_align_reg","reg","aoe_align_score",m))

# Finetune deltas as regression targets (added even if nearly-constant; main loop guards)
for nm, col in [("finetune_acc_delta","ft_acc_delta"),
                ("finetune_auc_delta","ft_auc_delta"),
                ("finetune_r2_delta", "ft_r2_delta")]:
    if col in df_xai.columns:
        targets.append((nm, "reg", col, None))

if not targets:
    print("[XAI][INFO] No XAI targets — nothing to do.")

# Main loop
for target_name, kind, y_col, mask in targets:
    fig_dir, save_dir = _mk_dirs_for_target(target_name)
    data = df_xai[mask].copy() if (mask is not None) else df_xai.copy()

    featsets = [("E_ONLY", FEATS_E_ONLY), ("EIX", FEATS_EIX)] if RUN_BOTH \
               else ([("E_ONLY", FEATS_E_ONLY)] if VARIANT=="energy_only" else [("EIX", FEATS_EIX)])

    for featset, feat_cols in featsets:
        cols = _ensure_cols(data, feat_cols)
        if not cols:
            print(f"[XAI] {target_name} — {featset}: no usable features; skip."); continue

        # Clean on EXACT used columns + target to keep rows aligned
        need = [y_col] + cols
        d = data.replace([np.inf,-np.inf], np.nan).dropna(subset=need, how="any")
        if d.empty:
            print(f"[XAI] {target_name} — {featset}: empty after clean; skip."); continue

        X = d[cols].copy(); y = d[y_col].values
        if len(X) != len(y):
            print(f"[XAI][WARN] length mismatch for {target_name} {featset}: X={len(X)} y={len(y)}; skip."); continue

        y_series = d[y_col]
        if kind == "cls":
            if y_series.nunique() < 2:
                print(f"[XAI] {target_name}: single class; skip."); continue
        else:
            if (y_series.nunique() < 3) or (y_series.std() < 1e-8):
                if target_name.startswith("finetune_") and XAI_ALLOW_CONST_FINETUNE:
                    print(f"[XAI] {target_name}: nearly constant, kept due to XAI_ALLOW_CONST_FINETUNE.")
                else:
                    print(f"[XAI] {target_name}: nearly constant; skip."); continue

        # Stratify only if binary with enough each
        strat = None
        if kind == "cls":
            uniq = np.unique(y)
            if len(uniq)==2 and min((y==uniq[0]).sum(), (y==uniq[1]).sum()) >= 2:
                strat = y

        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                                  random_state=RSTATE, stratify=strat)
        except ValueError as e:
            print(f"[XAI][WARN] split failed for {target_name} ({featset}): {e}"); continue

        base_png, base_csv = _file_prefix(fig_dir, save_dir, target_name, featset)

        if kind == "cls":
            model = RandomForestClassifier(
                n_estimators=MASTER_CTRL.get("RF_N_ESTIMATORS",400),
                random_state=RSTATE,
                n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS",-1),
                class_weight=MASTER_CTRL.get("RF_CLASS_WEIGHT","balanced")
            )
            model.fit(Xtr, ytr)
            yp = model.predict(Xte)
            acc = accuracy_score(yte, yp)
            try:
                proba = model.predict_proba(Xte)[:,1]
                auc = roc_auc_score(yte, proba) if len(np.unique(yte))==2 else np.nan
            except Exception:
                auc = np.nan
            print(f"[XAI] {target_name} [{variant_title}] {featset}: ACC={acc:.3f}, AUC={auc if np.isnan(auc) else round(auc,3)}")

            if SAVE_SHAP:
                _shap_summary(model, Xte, X.columns.tolist(),
                              out_png=base_png.replace(target_name, f"shap_summary__{target_name}") + ".png",
                              fig_title=_title_with_feat(SUBDIRS[target_name][1], featset))
            if SAVE_LIME and len(np.unique(ytr))==2 and len(Xte) >= 5:
                # LIME on reduced data to avoid zero-variance traps
                X_np = Xtr.values; stds = X_np.std(axis=0); keep = stds > 1e-12
                X_red = X_np[:, keep]; feat_red = [f for f,k in zip(Xtr.columns, keep) if k]
                if X_red.shape[1] >= 1:
                    from lime.lime_tabular import LimeTabularExplainer
                    full_means = Xtr.mean(axis=0).values; keep_idx = np.where(keep)[0]
                    def _predict_proba_on_reduced(Z_red):
                        Z_red = np.asarray(Z_red); Z_red = Z_red.reshape(1,-1) if Z_red.ndim==1 else Z_red
                        Z_full = np.tile(full_means, (Z_red.shape[0], 1)); Z_full[:, keep_idx] = Z_red
                        return model.predict_proba(Z_full)
                    expl = LimeTabularExplainer(training_data=X_red, feature_names=feat_red,
                                                discretize_continuous=True, mode="classification")
                    rng = np.random.default_rng(RSTATE)
                    idxs = rng.choice(X_red.shape[0], size=min(LIME_K, X_red.shape[0]), replace=False)
                    rows = []
                    pos = int(np.argmax(getattr(model,"classes_", [0,1])))
                    for i in idxs:
                        exp = expl.explain_instance(X_red[i], _predict_proba_on_reduced,
                                                    num_features=min(8, X_red.shape[1]))
                        for name, w in exp.as_list(label=pos):
                            base = name.split()[0]; rows.append((base, float(w)))
                    if rows:
                        import re

                        # Build dataframe and average weights per feature
                        dfw = (pd.DataFrame(rows, columns=["feature", "weight"])
                                 .groupby("feature", as_index=False)["weight"].mean()
                                 .sort_values("weight"))

                        # --- Pretty labels for the y-axis ---
                        def _pretty_label(s: str) -> str:
                            base = str(s).strip()
                            # Drop bin boundaries or numeric parts accidentally included by LIME
                            m = re.match(r"^([A-Za-z_]+)", base)
                            if m:
                                base = m.group(1)

                            # Human-friendly renames
                            base = (base
                                    .replace("abs_E_minus_I", "|E − I|")
                                    .replace("logX", "log X")
                                    .replace("dist_to_goldilocks", "Goldilocks X"))
                            return base

                        dfw["feature_pretty"] = dfw["feature"].map(_pretty_label)

                        # Plot with pretty labels
                        plt.figure(figsize=(7, 4))
                        plt.barh(dfw["feature_pretty"], dfw["weight"], edgecolor="black")
                        plt.xlabel("Avg LIME weight")
                        plt.title("LIME avg — " + _title_with_feat(SUBDIRS[target_name][1], featset))
                        plt.gcf().tight_layout(rect=[0, 0, 1, 0.95])
                        plt.tight_layout()
                        plt.savefig(
                            base_png.replace(target_name, f"lime_avg__{target_name}") + ".png",
                            dpi=220, bbox_inches="tight"
                        )
                        plt.close()

            pd.DataFrame([{
                "target": target_name, "variant": variant_title, "featset": featset,
                "acc": acc, "auc": float(auc) if np.isfinite(auc) else np.nan,
                "n_train": len(Xtr), "n_test": len(Xte)
            }]).to_csv(base_csv.replace(target_name, f"metrics__{target_name}") + ".csv", index=False)

        else:  # regression
            model = RandomForestRegressor(
                n_estimators=MASTER_CTRL.get("RF_N_ESTIMATORS",400),
                random_state=RSTATE, n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS",-1)
            )
            model.fit(Xtr, ytr); r2 = r2_score(yte, model.predict(Xte))
            print(f"[XAI] {target_name} [{variant_title}] {featset}: R2={r2:.3f}")
            if SAVE_SHAP:
                _shap_summary(model, Xte, X.columns.tolist(),
                              out_png=base_png.replace(target_name, f"shap_summary__{target_name}") + ".png",
                              fig_title=_title_with_feat(SUBDIRS[target_name][1], featset))
            pd.DataFrame([{
                "target": target_name, "variant": variant_title, "featset": featset,
                "r2": r2, "n_train": len(Xtr), "n_test": len(Xte)
            }]).to_csv(base_csv.replace(target_name, f"metrics__{target_name}") + ".csv", index=False)

print("[XAI] Completed.")       
            
# ======================================================
# 19) PATCH: Robust copy to Google Drive (MASTER_CTRL-driven)
# ======================================================
if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
    try:
        # Config from MASTER_CTRL
        DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
        ALLOWED_EXTS = set(MASTER_CTRL.get("ALLOW_FILE_EXTS", [".png", ".fits", ".csv", ".json", ".txt", ".npy"]))
        MAX_FILES = MASTER_CTRL.get("MAX_FILES_TO_SAVE",
                    MASTER_CTRL.get("MAX_FIGS_TO_SAVE", None))  
        VERBOSE = MASTER_CTRL.get("VERBOSE", True)

        # Ensure base directory exists
        os.makedirs(DRIVE_BASE, exist_ok=True)

        # Destination run folder (deterministic naming from run_id)
        GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id)
        os.makedirs(GOOGLE_DIR, exist_ok=True)

        # Optional listing before copy
        if VERBOSE and os.path.isdir(FIG_DIR):
            print("\n[INFO] Files in FIG_DIR before Drive copy:")
            for fn in sorted(os.listdir(FIG_DIR)):
                print("   -", fn)

        # Walk source tree and copy files matching allowed extensions
        copied, skipped, errored = [], [], []
        to_copy = []

        for root, dirs, files in os.walk(SAVE_DIR):
            # Collect eligible files first (we will sort globally for determinism)
            for file in files:
                if any(file.endswith(ext) for ext in ALLOWED_EXTS):
                    src = os.path.join(root, file)
                    rel = os.path.relpath(src, SAVE_DIR)
                    to_copy.append((rel, src))

        for root, dirs, files in os.walk(FIG_DIR):
            for file in files:
                if any(file.endswith(ext) for ext in ALLOWED_EXTS):
                    src = os.path.join(root, file)
                    rel_under_figs = os.path.relpath(src, FIG_DIR)           # pl. "cmb_best/xxx.png"
                    rel = os.path.join("figs", rel_under_figs)                # run_id/figs/...
                    to_copy.append((rel, src))

        # PRIORITIZE Fine-tune PNGs (root of FIG_DIR)
        prio_local = [
            with_variant(os.path.join(FIG_DIR, "lockin_by_eqI_bar.png")),
            with_variant(os.path.join(FIG_DIR, "finetune_gap_curve.png")),
            with_variant(os.path.join(FIG_DIR, "finetune_gap_adaptive.png")),
            with_variant(os.path.join(FIG_DIR, "finetune_panel.png")),  # merged panel
        ]

        prio_pairs = []
        for src in prio_local:
            if os.path.exists(src):
                rel_under_figs = os.path.relpath(src, FIG_DIR)   # e.g. "Finetune/xxx.png"
                rel = os.path.join("figs", rel_under_figs)        # run_id/figs/Finetune/xxx.png
                prio_pairs.append((rel, src))

        # Put priorities in front and deduplicate by rel
        rel_seen = set()
        to_copy = prio_pairs + to_copy
        tmp = []
        for rel, src in to_copy:
            if rel not in rel_seen:
                tmp.append((rel, src))
                rel_seen.add(rel)
        to_copy = tmp

        # Sort all files by their relative path to copy deterministically
        to_copy.sort(key=lambda t: t[0])

        # DEBUG
        print("[COPY][DEBUG] total to_copy:", len(to_copy))
        print("[COPY][DEBUG] first 15:", [r for r,_ in to_copy[:15]])

        # Apply optional MAX_FILES cap
        if isinstance(MAX_FILES, int) and MAX_FILES > 0:
            to_copy = to_copy[:MAX_FILES]
            
        # Perform the copy
        for rel, src in to_copy:
            dst_dir = os.path.join(GOOGLE_DIR, os.path.dirname(rel))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(rel))
            try:
                # If already same file (same inode), skip
                if os.path.exists(dst):
                    try:
                        if os.path.samefile(src, dst):
                            skipped.append(rel)
                            continue
                    except Exception:
                        # On some FS, samefile may fail; fall back to size+mtime heuristic
                        src_stat, dst_stat = os.stat(src), os.stat(dst)
                        if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
                            skipped.append(rel)
                            continue
                shutil.copy2(src, dst)
                copied.append(rel)
                if VERBOSE and len(copied) % 25 == 0:
                    print(f"[COPY] {len(copied)} files copied...")
            except Exception as e:
                errored.append((rel, str(e)))
                if VERBOSE:
                    print(f"[ERR] Copy failed for {rel}: {e}")

        # Summary
        print("☁️ Copy finished.")
        print(f"Copied: {len(copied)} files")
        print(f"Skipped: {len(skipped)} files")
        if errored:
            print(f"Errors: {len(errored)} files")
        print("Google Drive folder:", GOOGLE_DIR)

        # Optional verbose lists
        if VERBOSE:
            if skipped:
                print("[INFO] Skipped (pre-existing or identical):")
                for rel in skipped[:20]:
                    print("   -", rel)
                if len(skipped) > 20:
                    print(f"   ... and {len(skipped)-20} more")
            if errored:
                print("[INFO] Errors (first 10):")
                for rel, msg in errored[:10]:
                    print(f"   - {rel}: {msg}")

    except Exception as e:
        print(f"[ERR] Drive copy block failed: {e}")

# ======================================================
# 20) Best-universe entropy evolution (lock-in only)
# ======================================================

# Build a local config dict from MASTER_CTRL (type-safe + clamps)
BEST_CFG = {
    "N_TOP_BEST": int(np.clip(MASTER_CTRL.get("BEST_UNIVERSE_FIGS", 1),
                              1, MASTER_CTRL.get("BEST_MAX_FIGS", 50))),
    "TIME_STEPS": int(MASTER_CTRL.get("TIME_STEPS", 1000)),
    "N_REGIONS":  int(MASTER_CTRL.get("BEST_N_REGIONS", 10)),
    "STAB_THRESH": float(MASTER_CTRL.get("BEST_STAB_THRESHOLD", 3.5)),
    "SAVE_CSV": bool(MASTER_CTRL.get("BEST_SAVE_CSV", True)),
    "SEED_OFFSET": int(MASTER_CTRL.get("BEST_SEED_OFFSET", 777)),

    # --- FIX: region noise parameters ---
    "REGION_MU": float(MASTER_CTRL.get("BEST_REGION_MU", 5.1)),
    "REGION_SIGMA": float(MASTER_CTRL.get("BEST_REGION_SIGMA", 0.06)),
    "GLOBAL_JITTER": float(MASTER_CTRL.get("BEST_GLOBAL_JITTER", 0.008)),
    "SMOOTH_WINDOW": int(MASTER_CTRL.get("BEST_SMOOTH_WINDOW", 9)),
    "SHOW_REGIONS": bool(MASTER_CTRL.get("BEST_SHOW_REGIONS", True)),
    "ANNOTATE_LOCKIN": bool(MASTER_CTRL.get("BEST_ANNOTATE_LOCKIN", True)),
    "ANNOTATION_OFFSET": int(MASTER_CTRL.get("BEST_ANNOTATION_OFFSET", 5)),
}

def _entropy_evolution(seed: int, steps: int, n_regions: int):
    """
    Synthetic entropy generator for best-universe plots.
    All noise/smoothing parameters are taken from BEST_CFG.
    """
    r = np.random.default_rng(seed)
    t = np.arange(steps)

    base_mu  = BEST_CFG["REGION_MU"]
    base_sig = max(1e-6, BEST_CFG["REGION_SIGMA"])  # safety guard

    regions = []
    for _ in range(n_regions):
        x = np.empty(steps, dtype=float)
        x[0] = base_mu + r.normal(0, base_sig)
        for k in range(1, steps):
            # Mean-reverting random walk with noise
            x[k] = x[k-1] + 0.04*(base_mu - x[k-1]) + r.normal(0, base_sig*0.6)
        # Optional rolling-average smoothing
        w = max(1, int(BEST_CFG["SMOOTH_WINDOW"]))
        if w > 1:
            c = np.convolve(x, np.ones(w)/w, mode="same")
            x = c
        regions.append(x)
    regions = np.vstack(regions) if n_regions > 0 else np.empty((0, steps))

    # Global entropy: smooth growth + independent jitter
    jitter = BEST_CFG["GLOBAL_JITTER"]
    g = 5.6 + 0.45 * (1 - np.exp(-t / (steps/6))) + r.normal(0, jitter, size=steps)

    return t, regions, g

def _plot_best_universe(unirec: dict, steps: int, n_regions: int,
                        save_png: str, save_csv_dir: str):
    """
    Render one figure for a selected (lock-in) universe.
    Controlled entirely by BEST_CFG parameters.
    """
    uid = int(unirec["universe_id"])
    seed = int(unirec["seed"])
    lock_ep = int(unirec["lock_epoch"])

    t, regions, g = _entropy_evolution(seed + BEST_CFG["SEED_OFFSET"], steps, n_regions)

    # Export per-universe time series if enabled
    if BEST_CFG["SAVE_CSV"]:
        os.makedirs(save_csv_dir, exist_ok=True)
        df_reg = pd.DataFrame(regions.T, columns=[f"region_{i}_entropy" for i in range(n_regions)]) if n_regions>0 else pd.DataFrame()
        df_reg.insert(0, "time_step", t)
        df_reg["global_entropy"] = g
        df_reg["lock_epoch"] = lock_ep
        df_reg.to_csv(os.path.join(save_csv_dir, f"best_uni_{uid:05d}_entropy_timeseries.csv"), index=False)

    plt.figure(figsize=(10, 6.2))
    title_suffix = "(E)" if VARIANT == "energy_only" else "(E,I)"
    plt.title(f"Best-universe entropy evolution {title_suffix}")

    # Plot region curves if enabled
    if BEST_CFG["SHOW_REGIONS"] and n_regions > 0:
        for i in range(n_regions):
            plt.plot(t, regions[i], lw=1.0, alpha=0.55, label=f"Region {i} entropy" if i < 10 else None)

    # Plot global curve
    plt.plot(t, g, color="black", lw=3.0, label="Global entropy")

    # Stability threshold
    plt.axhline(BEST_CFG["STAB_THRESH"], color="red", ls="--", lw=1.5, label="Stability threshold")

    # Lock-in marker + annotation
    if BEST_CFG["ANNOTATE_LOCKIN"] and (0 <= lock_ep < steps):
        plt.axvline(lock_ep, color="purple", ls=(0, (5, 5)), lw=2.0)
        y_text = float(np.nanmin(g)) + 0.15
        plt.text(lock_ep + BEST_CFG["ANNOTATION_OFFSET"], y_text,
                 f"Lock-in step ≈ {lock_ep}", color="purple")

    plt.xlabel("Time step"); plt.ylabel("Entropy")

    # Compact legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) > 13:
        handles = handles[:12] + handles[-2:]
        labels  = labels[:12]  + labels[-2:]
    if handles:
        plt.legend(handles, labels, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    savefig(save_png)

# ---------- Selection + rendering (lock-in only) ----------
df_lock = df[df["lock_epoch"] >= 0].copy()
if len(df_lock) == 0:
    print("[BEST] No lock-in universes found; skipping best-universe plots.")
else:
    # Ranking: earlier lock-in is better; ties broken by smaller |E - I| if available.
    if "I" in df_lock.columns:
        df_lock["_gap"] = np.abs(df_lock["E"] - df_lock["I"])
    else:
        df_lock["_gap"] = 0.0
    df_lock = df_lock.sort_values(["lock_epoch", "_gap"]).reset_index(drop=True)

    n_take = int(np.clip(BEST_CFG["N_TOP_BEST"], 1, 50))
    picked = df_lock.head(n_take)

    # Output folders
    BEST_DIR = os.path.join(FIG_DIR, "best_universes")
    BEST_CSV_DIR = os.path.join(SAVE_DIR, "best_universes_csv")
    os.makedirs(BEST_DIR, exist_ok=True)

    made = []
    for rank, row in picked.iterrows():
        uid = int(row["universe_id"])
        png_path = with_variant(os.path.join(BEST_DIR, f"best_uni_rank{rank+1:02d}_uid{uid:05d}.png"))
        _plot_best_universe(
            unirec=row.to_dict(),
            steps=BEST_CFG["TIME_STEPS"],
            n_regions=BEST_CFG["N_REGIONS"],
            save_png=png_path,
            save_csv_dir=BEST_CSV_DIR
        )
        made.append(png_path)

    print(f"[BEST] Generated {len(made)} figure(s) for lock-in universes.")

# ======================================================
# 21) Save consolidated summary (single write)
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = int(len(df) - stable_count)
lockin_count = int((df["lock_epoch"] >= 0).sum())

summary = {
    "params": MASTER_CTRL,
    "master_seed": master_seed,
    "run_id": run_id,
    "N_samples": int(len(df)),
    "stability_summary": {
        "total_universes": len(df),
        "stable_universes": stable_count,
        "unstable_universes": unstable_count,
        "lockin_universes": lockin_count,
        "stable_percent": float(stable_count/len(df)*100),
        "unstable_percent": float(unstable_count/len(df)*100),
        "lockin_percent": float(lockin_count/len(df)*100)
    },
    "goldilocks_window_used": {
        "mode": MASTER_CTRL["GOLDILOCKS_MODE"],
        "X_low": E_c_low if E_c_low is not None else E_c_low_plot,
        "X_high": E_c_high if E_c_high is not None else E_c_high_plot
    },
    "figures": {
        "stability_curve": with_variant(os.path.join(FIG_DIR, "stability_curve.png")),
        "scatter_EI": with_variant(os.path.join(FIG_DIR, "scatter_EI.png")),
        "stability_distribution": with_variant(os.path.join(FIG_DIR, "stability_distribution.png")),  
        "fl_fluctuation": with_variant(os.path.join(FIG_DIR, "fl_fluctuation.png")),
        "fl_superposition": with_variant(os.path.join(FIG_DIR, "fl_superposition.png")),
        "fl_collapse":      with_variant(os.path.join(FIG_DIR, "fl_collapse.png")),
        "fl_expansion":     with_variant(os.path.join(FIG_DIR, "fl_expansion.png")),
        "ft_slice_EeqI": ft_result.get("files", {}).get("slice_png"),
        "best_universes_dir": os.path.join(FIG_DIR, "best_universes"),
        "stability_distribution_three": with_variant(os.path.join(FIG_DIR, "stability_distribution_three.png")),
    },
    "artifacts": {
        "tqe_runs_csv": with_variant(os.path.join(SAVE_DIR, "tqe_runs.csv")),
        "universe_seeds_csv": with_variant(os.path.join(SAVE_DIR, "universe_seeds.csv")),
        "pre_fluctuation_pairs_csv": with_variant(os.path.join(SAVE_DIR, "pre_fluctuation_pairs.csv")),
        "stability_by_I_zero_csv": with_variant(os.path.join(SAVE_DIR, "stability_by_I_zero.csv")),
        "stability_by_I_eps_sweep_csv": with_variant(os.path.join(SAVE_DIR, "stability_by_I_eps_sweep.csv")),  
        "fl_fluctuation_csv": with_variant(os.path.join(SAVE_DIR, "fl_fluctuation_timeseries.csv")),
        "fl_superposition_csv": with_variant(os.path.join(SAVE_DIR, "fl_superposition_timeseries.csv")),
        "fl_collapse_csv":      with_variant(os.path.join(SAVE_DIR, "fl_collapse_timeseries.csv")),
        "fl_expansion_csv":     with_variant(os.path.join(SAVE_DIR, "fl_expansion_timeseries.csv")),
        "ft_metrics_cls_csv": ft_result.get("files", {}).get("metrics_cls_csv"),
        "ft_metrics_reg_csv": ft_result.get("files", {}).get("metrics_reg_csv"),
        "ft_slice_EeqI_csv":  ft_result.get("files", {}).get("slice_csv"),
        "ft_delta_summary_csv": ft_result.get("files", {}).get("delta_csv"),
        "best_universes_csv_dir": os.path.join(SAVE_DIR, "best_universes_csv"),
    },
        "finetune_detector": {
        "enabled": bool(MASTER_CTRL.get("RUN_FINETUNE_DETECTOR", True)),
        "metrics": ft_result.get("metrics", {}),
        "artifacts": ft_result.get("files", {})
    },
    "meta": {
        "code_version": "2025-09-03a",
        "platform": sys.platform,
        "python": sys.version.split()[0]
    }
}
if MASTER_CTRL.get("SAVE_JSON", True):
    save_json(with_variant(os.path.join(SAVE_DIR, "summary_full.json")), summary)

print("\n🌌 Universe Stability Summary (final run)")
print(f"Total universes: {len(df)}")
print(f"Stable:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")
print(f"Lock-in:  {lockin_count} ({lockin_count/len(df)*100:.2f}%)")

# ======================================================
# 22) Universe Stability Distribution — clean & robust
# ======================================================

def _variant_label():
    """Short label for the plot title based on the pipeline variant."""
    return "E-only" if VARIANT == "energy_only" else "E+I"

# ---------- Recompute counts locally (no external state) ----------
total_n        = int(len(df))
stable_total   = int(df["stable"].sum())
unstable_count = int(total_n - stable_total)
lockin_count   = int((df["lock_epoch"] >= 0).sum())

# --- Disjoint categories: lock-in, stable-but-NOT-lockin, unstable ---
stable_only = max(0, stable_total - lockin_count)
values_disjoint = np.array([lockin_count, stable_only, unstable_count], dtype=float)
perc_disjoint   = (values_disjoint / max(1, total_n)) * 100.0

labels_disjoint = [
    f"Lock-in\n({lockin_count}, {perc_disjoint[0]:.1f}%)",
    f"Stable (no lock-in)\n({stable_only}, {perc_disjoint[1]:.1f}%)",
    f"Unstable\n({unstable_count}, {perc_disjoint[2]:.1f}%)",
]

plt.figure(figsize=(8.5, 7))
bars_d = plt.bar([0, 1, 2], values_disjoint,
                 edgecolor="black",
                 color=["#6aaed6", "#2ca02c", "#d62728"])

# counts above bars
for i, b in enumerate(bars_d):
    y = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2.0,
             y + (0.01 * max(1, total_n)),
             f"{int(values_disjoint[i])}",
             ha="center", va="bottom", fontsize=10)

plt.xticks([0, 1, 2], labels_disjoint, fontsize=12)
plt.ylabel("Number of Universes")
plt.title(f"Universe Stability Distribution ({_variant_label()}) — three categories")
plt.ylim(0, max(values_disjoint) * 1.12 + 1)
plt.tight_layout()
savefig(with_variant(os.path.join(FIG_DIR, "stability_distribution_three.png")))
print("[FIG] Wrote:", with_variant(os.path.join(FIG_DIR, "stability_distribution_three.png")))

# ---------- Overlapping categories: Stable total / Unstable / Lock-in ----------
values_overlap = np.array([stable_total, unstable_count, lockin_count], dtype=float)
labels_overlap = [
    f"Stable (total)\n({stable_total}, {stable_total/total_n*100:.2f}%)",
    f"Unstable\n({unstable_count}, {unstable_count/total_n*100:.2f}%)",
    f"Lock-in\n({lockin_count}, {lockin_count/total_n*100:.2f}%)",
]

plt.figure(figsize=(8.5, 7))
bars_o = plt.bar([0, 1, 2], values_overlap, edgecolor="black")

for i, b in enumerate(bars_o):
    y = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2.0,
             y + (0.01 * max(1, total_n)),
             f"{int(values_overlap[i])}",
             ha="center", va="bottom", fontsize=10)

plt.xticks([0, 1, 2], labels_overlap, fontsize=12)
plt.ylabel("Number of Universes")
plt.title("Universe Stability — overlapping categories (Stable / Unstable / Lock-in)")
plt.ylim(0, max(values_overlap) * 1.12 + 1)
plt.tight_layout()
savefig(with_variant(os.path.join(FIG_DIR, "stability_distribution_three_overlap.png")))
print("[FIG] Wrote:", with_variant(os.path.join(FIG_DIR, "stability_distribution_three_overlap.png")))

# ------------------------------------------------------
# 23) Compact version (backward compatibility, fixed)
# ------------------------------------------------------
values = values_disjoint
perc   = perc_disjoint
labels_compact = [
    f"Lock-in ({lockin_count}, {perc[0]:.1f}%)",
    f"Stable (no lock-in) ({stable_only}, {perc[1]:.1f}%)",
    f"Unstable ({unstable_count}, {perc[2]:.1f}%)",
]
plt.figure(figsize=(7,6))
plt.bar(labels_compact, values, color=["steelblue", "green", "red"], edgecolor="black")
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution (compact)")
plt.tight_layout()
savefig(with_variant(os.path.join(FIG_DIR, "stability_distribution.png")))
print("[FIG] Wrote:", with_variant(os.path.join(FIG_DIR, "stability_distribution.png")))

# --- FINAL COPY: Ensure 3-column chart goes to Google Drive ---
try:
    if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
        DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
        GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id)
        os.makedirs(GOOGLE_DIR, exist_ok=True)

        three_local = with_variant(os.path.join(FIG_DIR, "stability_distribution_three.png"))
        three_dst   = os.path.join(GOOGLE_DIR, os.path.relpath(three_local, SAVE_DIR))
        os.makedirs(os.path.dirname(three_dst), exist_ok=True)
        if os.path.exists(three_local):
            shutil.copy2(three_local, three_dst)
            print("[FINAL COPY] 3-column chart ->", three_dst)
        else:
            print("[FINAL COPY][WARN] Local 3-column chart not found:", three_local)
except Exception as e:
    print("[FINAL COPY][ERR][three]", e)

# --- FINAL COPY: Ensure best_universes PNGs are copied to Google Drive ---
try:
    if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
        DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_Universe_Simulation_Full_Pipeline")
        GOOGLE_DIR = os.path.join(DRIVE_BASE, run_id)
        src_dir = os.path.join(FIG_DIR, "best_universes")
        dst_dir = os.path.join(GOOGLE_DIR, os.path.relpath(src_dir, SAVE_DIR))
        if os.path.isdir(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            copied_cnt = 0
            for fn in sorted(os.listdir(src_dir)):
                if fn.endswith(".png"):
                    shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
                    copied_cnt += 1
            print(f"[FINAL COPY] best_universes PNGs copied: {copied_cnt} -> {dst_dir}")
        else:
            print("[FINAL COPY][WARN] best_universes directory missing:", src_dir)
except Exception as e:
    print("[FINAL COPY][ERR][best]", e)

# --- CHECK: Verify existence of 3-column chart ---
pth_three = with_variant(os.path.join(FIG_DIR, "stability_distribution_three.png"))
print("[CHECK] 3-column chart exists:", os.path.exists(pth_three), "->", pth_three)

# --- CHECK: Verify best_universes PNGs are generated ---
best_dir = os.path.join(FIG_DIR, "best_universes")
if os.path.isdir(best_dir):
    pngs = [f for f in os.listdir(best_dir) if f.endswith(".png")]
    print(f"[CHECK] best_universes PNG count: {len(pngs)} in {best_dir}")
    for f in pngs[:5]:
        print("   -", f)
else:
    print("[CHECK][WARN] best_universes directory missing:", best_dir)
