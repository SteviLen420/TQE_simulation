# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_EI_Universe_Simulation_Full_Pipeline.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

import os, time, json, warnings, sys, subprocess, shutil
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

for pkg in ["qutip", "pandas", "scipy", "scikit-learn"]:
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
    "TIME_STEPS":           800,    # epochs per stability run (if used elsewhere)
    "LOCKIN_EPOCHS":        500,    # epochs for law lock-in dynamics
    "EXPANSION_EPOCHS":     800,    # epochs for expansion dynamics
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

    # --- Stability thresholds ---
    "REL_EPS_STABLE":       0.010,    # relative calmness threshold for stability
    "REL_EPS_LOCKIN":       5e-3,     # relative calmness threshold for lock-in (0.2%)
    "CALM_STEPS_STABLE":    10,       # consecutive calm steps required (stable)
    "CALM_STEPS_LOCKIN":    12,       # consecutive calm steps required (lock-in)
    "MIN_LOCKIN_EPOCH":     200,      # lock-in can only occur after this epoch
    "LOCKIN_WINDOW":        10,       # rolling window size for averaging delta_rel
    "LOCKIN_ROLL_METRIC":   "median", # "mean" | "median" | "max" — aggregator over window
    "LOCKIN_REQUIRES_STABLE": True,   # require stable_at before checking lock-in
    "LOCKIN_MIN_STABLE_EPOCH": 0,     # require n - stable_at >= this many epochs

    # --- Goldilocks zone controls ---
    "GOLDILOCKS_MODE":      "dynamic",  # "heuristic" | "dynamic"
    "E_CENTER":             4.0,    # heuristic: energy sweet-spot center (used for X window)
    "E_WIDTH":              4.0,    # heuristic: energy sweet-spot width (used for X window)
    "GOLDILOCKS_THRESHOLD": 0.85,   # dynamic: fraction of max stability to define zone
    "GOLDILOCKS_MARGIN":    0.10,   # dynamic fallback margin around peak (±10%)
    "SIGMA_ALPHA":          1.5,    # curvature inside Goldilocks (sigma shaping)
    "OUTSIDE_PENALTY":      5,      # sigma multiplier outside Goldilocks zone
    "STAB_BINS":            40,     # number of bins in stability curve
    "SPLINE_K":             3,      # spline order for smoothing (3=cubic)

    # --- Noise shaping (lock-in loop) ---
    "EXP_NOISE_BASE":       0.12,   # baseline noise for updates (sigma0)
    "LL_BASE_NOISE":        8e-4,   # absolute noise floor (never go below this)
    "NOISE_DECAY_TAU":      500,    # e-folding time for noise decay (epochs)
    "NOISE_FLOOR_FRAC":     0.25,    # fraction of initial sigma preserved by decay
    "NOISE_COEFF_A":        1.0,    # per-variable noise multiplier (A)
    "NOISE_COEFF_NS":       0.10,   # per-variable noise multiplier (ns)
    "NOISE_COEFF_H":        0.20,   # per-variable noise multiplier (H)

    # --- Expansion dynamics (if/when used) ---
    "EXP_GROWTH_BASE":      1.005,  # baseline exponential growth rate
    # (EXP_NOISE_BASE above is reused as expansion amplitude baseline)

    # --- Machine Learning / XAI ---
    "RUN_XAI":              True,   # master switch for XAI section
    "RUN_SHAP":             True,   # SHAP on/off
    "RUN_LIME":             True,   # LIME on/off
    "LIME_NUM_FEATURES":    5,      # number of features in LIME plot
    "TEST_SIZE":            0.25,   # test split ratio
    "TEST_RANDOM_STATE":    42,     # split reproducibility
    "RF_N_ESTIMATORS":      400,    # number of trees in random forest
    "RF_CLASS_WEIGHT":      None,   # e.g., "balanced" for skewed classes
    "SKLEARN_N_JOBS":       -1,     # parallelism for RF

    # --- Outputs / IO ---
    "SAVE_FIGS":            True,   # save plots to disk
    "SAVE_JSON":            True,   # save summary JSON
    "SAVE_DRIVE_COPY":      True,   # copy results to Google Drive
    "DRIVE_BASE_DIR":       "/content/drive/MyDrive/TQE_EI_Universe_Simulation_Full_Pipeline",
    "RUN_ID_PREFIX":        "TQE_EI_Universe_Simulation_Full_Pipeline_",   # prefix for run_id
    "RUN_ID_FORMAT":        "%Y%m%d_%H%M%S",          # time format for run_id
    "ALLOW_FILE_EXTS":      [".png", ".fits", ".csv", ".json", ".txt", ".npy"],
    "MAX_FIGS_TO_SAVE":     None,   # limit number of figs (None = no limit)
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
    Example: figs/stability_curve.png -> figs/stability_curve_full.png
    """
    root, ext = os.path.splitext(path)
    return f"{root}_{VARIANT}{ext}"

# Output dirs
run_id = MASTER_CTRL["RUN_ID_PREFIX"] + VARIANT + "_" + time.strftime(MASTER_CTRL["RUN_ID_FORMAT"])
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
# 5) Goldilocks noise function
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
# 6) Lock-in simulation (drop-in: MASTER_CTRL-driven)
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
# 7) Helpers for MC runs and dynamic Goldilocks estimation
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

    # ---------- Binning (inclusive last bin, right=True) ----------
    x_min = np.nanmin(Xvals)
    x_max = np.nanmax(Xvals)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        # Degenerate X range -> cannot bin
        return None, None, np.array([]), np.array([]), np.array([]), np.array([]), df_in

    eps_max = 1e-12
    bins = np.linspace(x_min, x_max + eps_max, nbins)
    df_tmp = df_in.copy()
    df_tmp["bin"] = np.digitize(df_tmp["X"].values, bins, right=True)

    # Drop out-of-range / zero-bin
    df_tmp = df_tmp[(df_tmp["bin"] > 0) & np.isfinite(df_tmp["bin"])]

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
# 8) Monte Carlo universes — single or two-phase run
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

    # Pass 2: final run with window shaping (ha nincs érvényes ablak, fusson shaping nélkül)
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
# 9) Stability curve (binned) + Goldilocks window plot
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

if E_c_low is not None and E_c_high is not None:
    plt.axvline(E_c_low,  color='g', ls='--', label=_lbl(E_c_low,  "E_c_low"))
    plt.axvline(E_c_high, color='m', ls='--', label=_lbl(E_c_high, "E_c_high"))
elif E_c_low_plot is not None and E_c_high_plot is not None:
    plt.axvline(E_c_low_plot,  color='g', ls='--', label=_lbl(E_c_low_plot,  "E_c_low(curve)"))
    plt.axvline(E_c_high_plot, color='m', ls='--', label=_lbl(E_c_high_plot, "E_c_high(curve)"))

plt.xlabel("X = E·I (or configured)")
plt.ylabel("P(stable)")
plt.title("Goldilocks zone: stability curve")
plt.legend()
savefig(with_variant(os.path.join(FIG_DIR, "stability_curve.png")))

# ======================================================
# 10) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
sc = plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information parameter (I: KL×Shannon)")
plt.title("Universe outcomes in (E, I) space")
cb = plt.colorbar(sc, ticks=[0, 1])
cb.set_label("Stable (0/1)")
savefig(with_variant(os.path.join(FIG_DIR, "scatter_EI.png")))

# ======================================================
# 11) Save consolidated summary (single write)
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
        "stability_distribution": with_variant(os.path.join(FIG_DIR, "stability_distribution.png"))
    },
    "artifacts": {
        "tqe_runs_csv": with_variant(os.path.join(SAVE_DIR, "tqe_runs.csv")),
        "universe_seeds_csv": with_variant(os.path.join(SAVE_DIR, "universe_seeds.csv")),
        "pre_fluctuation_pairs_csv": with_variant(os.path.join(SAVE_DIR, "pre_fluctuation_pairs.csv")),
        "stability_by_I_zero_csv": with_variant(os.path.join(SAVE_DIR, "stability_by_I_zero.csv")),
        "stability_by_I_eps_sweep_csv": with_variant(os.path.join(SAVE_DIR, "stability_by_I_eps_sweep.csv"))
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
# 12) Universe Stability Distribution (bar chart)
# ======================================================
labels = [
    f"Lock-in ({lockin_count}, {lockin_count/len(df)*100:.1f}%)",
    f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)"
]
values = [lockin_count, stable_count, unstable_count]
colors = ["blue", "green", "red"]  # fixed colors for categories

plt.figure(figsize=(7,6))
plt.bar(labels, values, color=colors, edgecolor="black")
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution")
plt.tight_layout()
savefig(with_variant(os.path.join(FIG_DIR, "stability_distribution.png")))

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
# 14) XAI (SHAP + LIME) — robust, MASTER_CTRL-driven
# ======================================================

# --- Ensure classifier var exists even if RUN_XAI=False (for LIME guard) ---
rf_cls = None

def _savefig_safe(path):
    """Helper: safe figure saving with tight bounding box."""
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def _save_df_safe(df_in, path):
    """Helper: safe DataFrame saving with error handling."""
    try:
        df_in.to_csv(path, index=False)
        print(f"[SAVE] CSV: {path}")
    except Exception as e:
        print(f"[ERR] CSV save failed: {path} -> {e}")

if MASTER_CTRL.get("RUN_XAI", True):
    # -------------------- Features & targets --------------------
    X_feat = df[["E", "I", "X"]].copy()
    y_cls  = df["stable"].astype(int).values
    reg_mask = df["lock_epoch"] >= 0
    X_reg = X_feat[reg_mask]
    y_reg = df.loc[reg_mask, "lock_epoch"].values

    # --- Sanity checks ---
    assert not np.isnan(X_feat.values).any(), "NaN in X_feat!"
    if len(X_reg) > 0:
        assert not np.isnan(X_reg.values).any(), "NaN in X_reg!"

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, accuracy_score

    # --------- Stratify guard (only if both classes present) ----------
    vals, cnts = np.unique(y_cls, return_counts=True)
    can_stratify = (len(vals) == 2) and (cnts.min() >= 2)
    stratify_arg = y_cls if can_stratify else None
    if not can_stratify:
        print(f"[XAI][WARN] Skipping stratify: class counts = {dict(zip(vals, cnts))}")

    # -------------------- Classification split --------------------
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        X_feat, y_cls,
        test_size=MASTER_CTRL["TEST_SIZE"],
        random_state=MASTER_CTRL.get("TEST_RANDOM_STATE", 42),
        stratify=stratify_arg
    )

    # -------------------- Regression split --------------------
    have_reg = len(X_reg) >= MASTER_CTRL.get("REGRESSION_MIN", 10)
    if have_reg:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_reg, y_reg,
            test_size=MASTER_CTRL["TEST_SIZE"],
            random_state=MASTER_CTRL.get("TEST_RANDOM_STATE", 42)
        )

    # -------------------- Train models --------------------
    rf_cls = None
    if len(np.unique(ytr_c)) == 2:
        rf_cls = RandomForestClassifier(
            n_estimators=MASTER_CTRL["RF_N_ESTIMATORS"],
            random_state=MASTER_CTRL.get("TEST_RANDOM_STATE", 42),
            n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS", -1),
            class_weight=MASTER_CTRL.get("RF_CLASS_WEIGHT", None)
        )
        rf_cls.fit(Xtr_c, ytr_c)
        cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
        print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")
    else:
        print("[XAI][WARN] Classification skipped: training set has a single class.")

    rf_reg, reg_r2 = None, None
    if have_reg:
        rf_reg = RandomForestRegressor(
            n_estimators=MASTER_CTRL["RF_N_ESTIMATORS"],
            random_state=MASTER_CTRL.get("TEST_RANDOM_STATE", 42),
            n_jobs=MASTER_CTRL.get("SKLEARN_N_JOBS", -1)
        )
        rf_reg.fit(Xtr_r, ytr_r)
        reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
        print(f"[XAI] Regression R^2 (lock_epoch): {reg_r2:.3f}")
    else:
        print("[XAI] Not enough locked samples for regression (need ~10+).")

    # -------------------- SHAP: classification --------------------
    if MASTER_CTRL.get("RUN_SHAP", True) and rf_cls is not None:
        try:
            X_plot = Xte_c.copy()
            # Prefer TreeExplainer; fallback to model-agnostic Explainer
            try:
                expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional", model_output="raw")
                sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
            except Exception:
                expl_cls = shap.Explainer(rf_cls, Xtr_c)
                shap_res = expl_cls(X_plot)
                sv_cls = getattr(shap_res, "values", shap_res)

            # Normalize shape
            if isinstance(sv_cls, list) and len(sv_cls) > 1:
                sv_cls = sv_cls[1]  # positive class
            sv_cls = np.asarray(sv_cls)
            if sv_cls.ndim == 3:
                if sv_cls.shape[0] == X_plot.shape[0]:
                    sv_cls = sv_cls[:, :, 1 if sv_cls.shape[2] > 1 else 0]
                elif sv_cls.shape[-1] == X_plot.shape[1]:
                    sv_cls = sv_cls[1 if sv_cls.shape[0] > 1 else 0, :, :]

            assert sv_cls.shape[1] == X_plot.shape[1], f"SHAP shape mismatch: {sv_cls.shape} vs {X_plot.shape}"

            plt.figure()
            shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
            _savefig_safe(with_variant(os.path.join(FIG_DIR, "shap_summary_cls_stable.png")))

            _save_df_safe(pd.DataFrame(sv_cls, columns=X_plot.columns),
                          with_variant(os.path.join(FIG_DIR, "shap_values_classification.csv")))
            np.save(with_variant(os.path.join(FIG_DIR, "shap_values_cls.npy")), sv_cls)
        except Exception as e:
            print(f"[XAI][ERR] SHAP classification failed: {e}")

    # -------------------- SHAP: regression (robust) --------------------
    if MASTER_CTRL.get("RUN_SHAP", True) and rf_reg is not None and have_reg:
        try:
            MAX_SHAP_SAMPLES = int(MASTER_CTRL.get("MAX_SHAP_SAMPLES", 1000))
            SHAP_BG_SIZE     = int(MASTER_CTRL.get("SHAP_BACKGROUND_SIZE", 200))
            RNG_STATE        = MASTER_CTRL.get("TEST_RANDOM_STATE", 42)

            X_plot_r = Xte_r.copy()
            if len(X_plot_r) > MAX_SHAP_SAMPLES:
                X_plot_r = X_plot_r.sample(MAX_SHAP_SAMPLES, random_state=RNG_STATE)

            sv_reg = None
            try:
                expl_reg = shap.TreeExplainer(rf_reg, feature_perturbation="interventional", model_output="raw")
                sv_reg = expl_reg.shap_values(X_plot_r, check_additivity=False)
            except Exception:
                background = Xtr_r.sample(SHAP_BG_SIZE, random_state=RNG_STATE) if len(Xtr_r) > SHAP_BG_SIZE else Xtr_r
                expl_reg = shap.Explainer(rf_reg, background)
                shap_res_r = expl_reg(X_plot_r)
                sv_reg = getattr(shap_res_r, "values", shap_res_r)

            sv_reg = np.asarray(sv_reg)
            if sv_reg.ndim == 3:
                if sv_reg.shape[2] == 1 and sv_reg.shape[0] == len(X_plot_r):
                    sv_reg = sv_reg[:, :, 0]
                elif sv_reg.shape[0] == 1 and sv_reg.shape[1] == len(X_plot_r):
                    sv_reg = sv_reg[0, :, :]
                elif sv_reg.shape[-1] == X_plot_r.shape[1]:
                    sv_reg = sv_reg[0, :, :]

            assert sv_reg.ndim == 2 and sv_reg.shape[1] == X_plot_r.shape[1], \
                f"SHAP shape mismatch: {sv_reg.shape} vs {X_plot_r.shape}"

            if MASTER_CTRL.get("SAVE_FIGS", True):
                plt.figure()
                shap.summary_plot(sv_reg, X_plot_r.values,
                                  feature_names=X_plot_r.columns.tolist(),
                                  show=False)
                _savefig_safe(with_variant(os.path.join(FIG_DIR, "shap_summary_reg_lock_epoch.png")))

            _save_df_safe(pd.DataFrame(sv_reg, columns=X_plot_r.columns),
                          with_variant(os.path.join(FIG_DIR, "shap_values_regression.csv")))
            np.save(with_variant(os.path.join(FIG_DIR, "shap_values_reg.npy")), sv_reg)

        except Exception as e:
            print(f"[XAI][ERR] SHAP regression failed: {e}")

    if MASTER_CTRL.get("RUN_SHAP", True) and rf_reg is None:
        print("[XAI] Regression SHAP skipped (no regressor).")

# -------------------- LIME: lock-in only + averaged importances --------------------
if (MASTER_CTRL.get("RUN_LIME", True)
    and rf_cls is not None
    and "lockin" in df.columns):

    # Keep only universes where law-lockin happened
    df_lock = df[df["lockin"] == 1].copy()

    if len(df_lock) < 10:
        print(f"[LIME] Not enough lock-in samples for LIME (have {len(df_lock)}, need ≥10).")
    else:
        # Features and labels restricted to lock-in universes
        X_lock = df_lock[["E", "I", "X"]].copy()
        y_lock = df_lock["stable"].astype(int).values

        # Build LIME explainer on the lock-in distribution
        lime_explainer = LimeTabularExplainer(
            training_data=X_lock.values,
            feature_names=X_lock.columns.tolist(),
            discretize_continuous=True,
            mode="classification"
        )

        # Choose which class to explain (positive=1 if available)
        target_label = 1 if 1 in getattr(rf_cls, "classes_", [0, 1]) else 0

        # --- (A) Averaged LIME over multiple lock-in instances ---
        rng_local = np.random.default_rng(MASTER_CTRL.get("TEST_RANDOM_STATE", 42))
        K = int(min(50, len(X_lock)))  # up to 50 instances for averaging
        idxs = rng_local.choice(len(X_lock), size=K, replace=False)

        rows = []
        for idx in idxs:
            exp = lime_explainer.explain_instance(
                X_lock.iloc[idx].values,
                rf_cls.predict_proba,
                num_features=min(MASTER_CTRL.get("LIME_NUM_FEATURES", 5), X_lock.shape[1])
            )
            # Collect (feature, weight) pairs for the chosen class
            for feat, weight in exp.as_list(label=target_label):
                # Normalize feature name to the raw column when possible
                # (LIME may produce conditions like "X > 5.76"; we keep the prefix)
                base = feat.split()[0]
                if base not in X_lock.columns:
                    base = feat  # fallback: keep original label
                rows.append({"feature": base, "weight": float(weight)})

        lime_avg = (pd.DataFrame(rows)
                      .groupby("feature", as_index=False)["weight"].mean()
                      .sort_values("weight"))

        # Save CSV of averaged weights
        _save_df_safe(lime_avg, with_variant(os.path.join(FIG_DIR, "lime_lockin_avg.csv")))

        # Plot averaged horizontal bar chart (saved as PNG)
        plt.figure(figsize=(6, 4))
        plt.barh(lime_avg["feature"], lime_avg["weight"], edgecolor="black")
        plt.xlabel("Avg LIME weight (lock-in only)")
        plt.ylabel("Feature")
        plt.title("LIME (average over lock-in universes)")
        plt.tight_layout()
        _savefig_safe(with_variant(os.path.join(FIG_DIR, "lime_lockin_avg.png")))

        # --- (B) Single-instance classic LIME figure (also saved as PNG) ---
        # Take one representative instance (e.g., the first of the sampled indices)
        eg_idx = int(idxs[0])
        exp_one = lime_explainer.explain_instance(
            X_lock.iloc[eg_idx].values,
            rf_cls.predict_proba,
            num_features=min(MASTER_CTRL.get("LIME_NUM_FEATURES", 5), X_lock.shape[1])
        )
        fig = exp_one.as_pyplot_figure()
        fig.suptitle("LIME explanation (lock-in, single instance)", y=1.02)
        fig.savefig(with_variant(os.path.join(FIG_DIR, "lime_lockin_example.png")), dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"[LIME] Saved PNGs: "
              f"{with_variant(os.path.join(FIG_DIR, 'lime_lockin_avg.png'))} and "
              f"{with_variant(os.path.join(FIG_DIR, 'lime_lockin_example.png'))}")
        
            
# ======================================================
# 15) PATCH: Robust copy to Google Drive (MASTER_CTRL-driven)
# ======================================================
if MASTER_CTRL.get("SAVE_DRIVE_COPY", True):
    try:
        # Config from MASTER_CTRL
        DRIVE_BASE = MASTER_CTRL.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/TQE_(E,I)_KL_Shannon")
        ALLOWED_EXTS = set(MASTER_CTRL.get("ALLOW_FILE_EXTS", [".png", ".fits", ".csv", ".json", ".txt", ".npy"]))
        MAX_FILES = MASTER_CTRL.get("MAX_FIGS_TO_SAVE", None)  # None = no limit
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

        # Sort all files by their relative path to copy deterministically
        to_copy.sort(key=lambda t: t[0])

        # Apply optional MAX_FILES cap
        if isinstance(MAX_FILES, int) and MAX_FILES >= 0:
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
