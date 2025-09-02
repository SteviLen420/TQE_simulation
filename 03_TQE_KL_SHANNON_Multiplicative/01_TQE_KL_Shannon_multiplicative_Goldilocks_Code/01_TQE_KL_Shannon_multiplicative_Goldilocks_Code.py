# =============================================================================
# Theory of the Question of Existence (TQE)
# Energy–Information Simulation — KL × Shannon multiplicatively
# =============================================================================
# Author: Stefan Len
#
# SUMMARY
# Monte Carlo simulation of universes using a composite information parameter 
# I = f(KL, Shannon). KL divergence (between random quantum states) and 
# normalized Shannon entropy are multiplicatively fused, bounded in [0,1]. 
# Energy E is sampled from a log-normal distribution, with X = E·I determining 
# stabilization in a Goldilocks zone. Stability is defined as consecutive calm 
# steps (Δ_rel < ε). Results include stability curves, E–I scatter plots, 
# epsilon-sweep analysis for I ≈ 0, and XAI outputs (SHAP/LIME). All runs are 
# reproducible with master + per-universe seeds, exported to CSV/JSON/PNG.

# KEYWORDS: Goldilocks window; KL divergence; Shannon entropy; energy–
# information coupling; Monte Carlo; stability detection; SHAP; LIME;
# reproducibility; epsilon sweep
# =============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, warnings, sys, subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt

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
# MASTER CONTROLLER — unified parameters (KL × Shannon)
# ======================================================
MASTER_CTRL = {
    # --- Core simulation ---
    "NUM_UNIVERSES":        5000,   # number of universes in Monte Carlo run
    "TIME_STEPS":           800,    # epochs per stability run
    "LOCKIN_EPOCHS":        500,    # epochs for law lock-in dynamics
    "EXPANSION_EPOCHS":     800,    # epochs for expansion dynamics
    "SEED":                 None,   # master RNG seed (auto-generated if None)

    # --- Energy distribution (lognormal + Goldilocks) ---
    "E_LOG_MU":             2.5,    # lognormal mean for initial energy
    "E_LOG_SIGMA":          0.8,    # lognormal sigma for initial energy
    "E_CENTER":             6.0,    # Goldilocks center (energy sweet spot)
    "E_WIDTH":              6.0,    # Goldilocks width (spread of stable energy)
    "ALPHA_I":              0.8,    # coupling factor: strength of I in E·I

    # --- Stability thresholds ---
    "REL_EPS_STABLE":       0.04,   # relative calmness threshold for stability
    "REL_EPS_LOCKIN":       5e-3,   # relative calmness threshold for lock-in
    "CALM_STEPS_STABLE":    5,      # consecutive calm steps required (stable)
    "CALM_STEPS_LOCKIN":    20,     # consecutive calm steps required (lock-in)

    # --- Goldilocks tuning ---
    "GOLDILOCKS_THRESHOLD": 0.8,   # fraction of max stability used for zone width

    # --- Law lock-in shaping ---
    "LL_BASE_NOISE":        1e6,    # baseline noise level for law lock-in

    # --- Expansion dynamics ---
    "EXP_GROWTH_BASE":      1.005,  # baseline exponential growth rate
    "EXP_NOISE_BASE":       1.0,    # baseline noise for expansion amplitude

    # --- Machine Learning / XAI ---
    "TEST_SIZE":            0.25,   # test split ratio
    "RF_N_ESTIMATORS":      400,    # number of trees in random forest
    "RUN_XAI":              True,   # run SHAP + LIME explainability
    "REGRESSION_MIN":       30,     # min lock-in samples for regression

    # --- Outputs ---
    "SAVE_FIGS":            True,   # save plots to disk
    "SAVE_JSON":            True,   # save summary JSON
    "SAVE_DRIVE_COPY":      True,   # copy results to Google Drive

    # --- Plot toggles ---
    "PLOT_AVG_LOCKIN":      True,   # plot average lock-in curve
    "PLOT_LOCKIN_HIST":     True,   # plot histogram of lock-in epochs
    "PLOT_STABILITY_BASIC": False   # simple stability diagnostic plot
}

# ======================================================
# Master seed initialization (reproducibility)
# ======================================================
if MASTER_CTRL["SEED"] is None:
    MASTER_CTRL["SEED"] = int(np.random.SeedSequence().generate_state(1)[0])

master_seed = MASTER_CTRL["SEED"]

# Create both modern (rng) and legacy (np.random) RNG streams
rng = np.random.default_rng(master_seed)
np.random.seed(master_seed)  # sync legacy RNG for QuTiP calls

print(f"🎲 Using master seed: {master_seed}")

# Output dirs
run_id  = time.strftime("TQE_(E,I)_KL_SHANNON_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(os.getcwd(), run_id)
FIG_DIR  = os.path.join(SAVE_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(path):
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

print(f"💾 Results saved in: {SAVE_DIR}")

# ======================================================
# 1) Information parameter I = g(KL, Shannon) (multiplicative fusion)
# ======================================================
def sample_information_param(dim=8):
    # Two random kets for KL, one reused for Shannon
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1 = np.abs(psi1.full().flatten())**2
    p2 = np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12

    # KL-divergence-based info in [0,1]
    KL = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)))
    I_kl = KL / (1.0 + KL)

    # Normalized Shannon entropy of psi1 in [0,1]
    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))

    # Multiplicative coupling, squashed to [0,1]
    I_raw = I_kl * I_shannon
    I = I_raw / (1.0 + I_raw)
    return float(I)

# ======================================================
# 2) Energy sampling
# ======================================================
def sample_energy_lognormal(mu=2.5, sigma=0.9):
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ======================================================
# 3) Goldilocks noise function
# ======================================================
def sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high):
    if E_c_low is None or E_c_high is None:
        return sigma0
    if X < E_c_low or X > E_c_high:
        return sigma0 * 1.5
    mid = 0.5 * (E_c_low + E_c_high)
    width = max(0.5 * (E_c_high - E_c_low), 1e-12)
    dist = abs(X - mid) / width  # 0 center, 1 edges
    return sigma0 * (1 + alpha * dist**2)

# ======================================================
# 4) Lock-in simulation (patched: use rng passed in)
# ======================================================
def simulate_lock_in(X, N_epoch,
                     rel_eps_stable=MASTER_CTRL["REL_EPS_STABLE"],
                     rel_eps_lockin=MASTER_CTRL["REL_EPS_LOCKIN"],
                     sigma0=0.2, alpha=1.0,
                     E_c_low=None, E_c_high=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    A, ns, H = rng.normal(50, 5), rng.normal(0.8, 0.05), rng.normal(0.7, 0.08)

    # tracking states
    stable_at, lockin_at = None, None
    consec_stable, consec_lockin = 0, 0

    for n in range(1, N_epoch + 1):
        sigma = sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high)
        A_prev, ns_prev, H_prev = A, ns, H
        A  += rng.normal(0, sigma)
        ns += rng.normal(0, sigma/10)
        H  += rng.normal(0, sigma/5)

        delta_rel = (abs(A - A_prev)/abs(A_prev) +
                     abs(ns - ns_prev)/abs(ns_prev) +
                     abs(H - H_prev)/abs(H_prev)) / 3.0

        # --- stable check ---
        if delta_rel < rel_eps_stable:
            consec_stable += 1
            if consec_stable >= MASTER_CTRL["CALM_STEPS_STABLE"] and stable_at is None:
                stable_at = n
        else:
            consec_stable = 0

        # --- lock-in check ---
        if delta_rel < rel_eps_lockin:
            consec_lockin += 1
            if consec_lockin >= MASTER_CTRL["CALM_STEPS_LOCKIN"] and lockin_at is None:
                lockin_at = n
        else:
            consec_lockin = 0

    # outcomes
    is_stable = 1 if stable_at is not None else 0
    is_lockin = 1 if lockin_at is not None else 0

    return is_stable, is_lockin, (stable_at if stable_at else -1), (lockin_at if lockin_at else -1)

# ======================================================
# 5) Monte Carlo universes — KL × Shannon version
# ======================================================
rows = []
universe_seeds = []

for i in range(MASTER_CTRL["NUM_UNIVERSES"]):
    # --- derive per-universe seed from master rng ---
    uni_seed = int(rng.integers(0, 2**32 - 1))
    universe_seeds.append(uni_seed)

    # --- build per-universe RNG ---
    rng_uni = np.random.default_rng(uni_seed)
    np.random.seed(uni_seed)  # for libraries using np.random (QuTiP, etc.)

    # --- sample energy and information ---
    E = float(rng_uni.lognormal(MASTER_CTRL["E_LOG_MU"], MASTER_CTRL["E_LOG_SIGMA"]))
    I = sample_information_param(dim=8)
    X = E * I

    # --- Goldilocks X-window (heuristic based on E_CENTER / E_WIDTH / ALPHA_I) ---
    X_center    = MASTER_CTRL["E_CENTER"] * MASTER_CTRL["ALPHA_I"]
    X_halfwidth = 0.5 * MASTER_CTRL["E_WIDTH"] * max(1e-12, MASTER_CTRL["ALPHA_I"])
    E_c_low     = max(1e-12, X_center - X_halfwidth)
    E_c_high    = X_center + X_halfwidth

    # --- simulate stability / lock-in ---
    stable, lockin, stable_epoch, lock_epoch = simulate_lock_in(
        X,
        MASTER_CTRL["LOCKIN_EPOCHS"],
        sigma0=MASTER_CTRL["EXP_NOISE_BASE"],
        alpha=1.0,
        E_c_low=E_c_low,
        E_c_high=E_c_high,
        rng=rng_uni
    )

    rows.append({
        "universe_id": i,
        "seed": uni_seed,
        "E": E,
        "I": I,
        "X": X,
        "stable": stable,
        "lockin": lockin,
        "stable_epoch": stable_epoch,
        "lock_epoch": lock_epoch
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_DIR, "tqe_runs.csv"), index=False)

# Save per-universe seeds
pd.DataFrame({"universe_id": np.arange(len(df)), "seed": universe_seeds}).to_csv(
    os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False
)

# ======================================================
# 6) Stability curve (binned) + dynamic Goldilocks window
# ======================================================

# Bin X values into intervals for averaging
bins = np.linspace(df["X"].min(), df["X"].max(), 40)
df["bin"] = np.digitize(df["X"], bins)

# Aggregate mean X and stability rate per bin
bin_stats = df.groupby("bin").agg(
    mean_X=("X", "mean"),
    stable_rate=("stable", "mean"),
    count=("stable", "size")
).dropna()

xx = bin_stats["mean_X"].values
yy = bin_stats["stable_rate"].values

# Smooth the stability curve using spline interpolation
if len(xx) > 3:
    spline = make_interp_spline(xx, yy, k=3)
    xs = np.linspace(xx.min(), xx.max(), 300)
    ys = spline(xs)
else:
    xs, ys = xx, yy

# --- PATCH: Use a relative threshold (half-maximum) to define the Goldilocks zone ---
peak_index = int(np.argmax(ys))
peak_value = float(ys[peak_index])

# Use threshold from MASTER_CTRL (default 0.5 if not set)
threshold = MASTER_CTRL.get("GOLDILOCKS_THRESHOLD", 0.5)
half_max = threshold * peak_value

# Select the region around the peak where stability >= half of maximum
valid_region = xs[ys >= half_max]
if len(valid_region) > 0:
    E_c_low, E_c_high = float(valid_region.min()), float(valid_region.max())
else:
    # fallback: narrow window around the peak
    peak_x = float(xs[peak_index])
    E_c_low, E_c_high = peak_x * 0.9, peak_x * 1.1
    print("⚠️ No wide peak region found, using ±10% around the peak.")

# --- Plot Goldilocks stabilization curve ---
plt.figure(figsize=(8,5))
plt.scatter(xx, yy, s=30, c="blue", alpha=0.7, label="bin means")
plt.plot(xs, ys, "r-", lw=2, label="spline fit")
plt.axvline(E_c_low,  color='g', ls='--', label=f"E_c_low = {E_c_low:.2f}")
plt.axvline(E_c_high, color='m', ls='--', label=f"E_c_high = {E_c_high:.2f}")
plt.xlabel("X = E·I")
plt.ylabel("P(stable)")
plt.title("Goldilocks zone: stabilization curve (KL × Shannon)")
plt.legend()
savefig(os.path.join(FIG_DIR, "stability_curve.png"))

# ======================================================
# 7) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
sc = plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information parameter (I: KL×Shannon)")
plt.title("Universe outcomes in (E, I) space")
plt.colorbar(sc, label="Stable=1 / Unstable=0")
savefig(os.path.join(FIG_DIR, "scatter_EI.png"))

# ======================================================
# 8) Save stability summary
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = len(df) - stable_count
lockin_count = int((df["lock_epoch"] >= 0).sum())

summary = {
    "params": MASTER_CTRL,
    "master_seed": master_seed,
    "seeds": {
        "universe_seeds_csv": "universe_seeds.csv"
    },
    "stability_summary": {
        "total_universes": len(df),
        "stable_universes": stable_count,
        "unstable_universes": unstable_count,
        "lockin_universes": lockin_count,
        "stable_percent": float(stable_count/len(df)*100),
        "unstable_percent": float(unstable_count/len(df)*100),
        "lockin_percent": float(lockin_count/len(df)*100)
    }
}

with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n🌌 Universe Stability Summary (KL × Shannon)")
print(f"Total universes: {len(df)}")
print(f"Stable:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")
print(f"Lock-in:  {lockin_count} ({lockin_count/len(df)*100:.2f}%)")

# ======================================================
# 8b) Universe Stability Distribution (bar chart)
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
savefig(os.path.join(FIG_DIR, "stability_distribution.png"))

# ======================================================
# 9) PATCH: Stability by I (exact zero vs eps sweep)
# ======================================================
def _stability_stats(mask: pd.Series, label: str):
    total = int(mask.sum())
    stables = int(df.loc[mask, "stable"].sum())
    ratio = (stables / total) if total > 0 else float("nan")
    return {"group": label, "n": total, "stable_n": stables, "stable_ratio": ratio}

# Exact split
mask_I_eq0 = (df["I"] == 0.0)
mask_I_gt0 = (df["I"]  > 0.0)
zero_split_rows = [
    _stability_stats(mask_I_eq0, "I == 0"),
    _stability_stats(mask_I_gt0, "I > 0"),
]
zero_split_df = pd.DataFrame(zero_split_rows)
zero_split_path = os.path.join(SAVE_DIR, "stability_by_I_zero.csv")
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
eps_path = os.path.join(SAVE_DIR, "stability_by_I_eps_sweep.csv")
eps_df.to_csv(eps_path, index=False)
print("\n📈 Epsilon sweep (near-zero thresholds, preview):")
print(eps_df.head(12).to_string(index=False))
print(f"\n📝 Saved breakdowns to:\n - {zero_split_path}\n - {eps_path}")

# ======================================================
# 10) Save summary (PATCH: more fields)
# ======================================================
summary = {
    "params": MASTER_CTRL,
    "N_samples": int(len(df)),
    "stable_count": stable_count,
    "unstable_count": unstable_count,
    "stable_ratio": float(df["stable"].mean()),
    "unstable_ratio": float(1.0 - df["stable"].mean()),
    "E_c_low": E_c_low,
    "E_c_high": E_c_high,
    "master_seed": master_seed,
    "figures": {
        "stability_curve": os.path.join(FIG_DIR, "stability_curve.png"),
        "scatter_EI": os.path.join(FIG_DIR, "scatter_EI.png"),
        "stability_distribution": os.path.join(FIG_DIR, "stability_distribution.png")
    }
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

print("\n✅ DONE (core).")
print(f"Runs: {len(df)}")
print(f"Stable universes: {summary['stable_count']}")
print(f"Unstable universes: {summary['unstable_count']}")
print(f"Stability ratio: {summary['stable_ratio']:.3f}")
print(f"Goldilocks zone: {E_c_low:.2f} – {E_c_high:.2f}")
print(f"📂 Directory: {SAVE_DIR}")

# ======================================================
# 11) XAI (SHAP + LIME) — stratify guard + CSV saves
# ======================================================
def _savefig_safe(path):
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def _save_df_safe(df_in, path):
    try:
        df_in.to_csv(path, index=False)
        print(f"[SAVE] CSV: {path}")
    except Exception as e:
        print(f"[ERR] CSV save failed: {path} -> {e}")

# Features & targets
X_feat = df[["E", "I", "X"]].copy()
y_cls  = df["stable"].astype(int).values
reg_mask = df["lock_epoch"] >= 0
X_reg = X_feat[reg_mask]
y_reg = df.loc[reg_mask, "lock_epoch"].values

# Sanity checks
assert not np.isnan(X_feat.values).any(), "NaN in X_feat!"
if len(X_reg) > 0:
    assert not np.isnan(X_reg.values).any(), "NaN in X_reg!"

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# --------- Stratify guard ----------
vals, cnts = np.unique(y_cls, return_counts=True)
can_stratify = (len(vals) == 2) and (cnts.min() >= 2)
stratify_arg = y_cls if can_stratify else None
if not can_stratify:
    print(f"[WARN] Skipping stratify: class counts = {dict(zip(vals, cnts))}")

# Classification split
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls,
    test_size=MASTER_CTRL["TEST_SIZE"],
    random_state=42,
    stratify=stratify_arg
)

# Regression split
have_reg = len(X_reg) >= MASTER_CTRL["REGRESSION_MIN"]
if have_reg:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_reg, y_reg,
        test_size=MASTER_CTRL["TEST_SIZE"],
        random_state=42
    )

# ---------- Train models ----------
rf_cls = RandomForestClassifier(
    n_estimators=MASTER_CTRL["RF_N_ESTIMATORS"],
    random_state=42,
    n_jobs=-1
)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

rf_reg, reg_r2 = None, None
if have_reg:
    rf_reg = RandomForestRegressor(
        n_estimators=MASTER_CTRL["RF_N_ESTIMATORS"],
        random_state=42,
        n_jobs=-1
    )
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R^2 (lock_at): {reg_r2:.3f}")
else:
    print("[XAI] Not enough locked samples for regression (need ~30+).")

# ---------- SHAP classification ----------
if MASTER_CTRL["RUN_XAI"]:
    try:
        X_plot = Xte_c.copy()
        try:
            expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional", model_output="raw")
            sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
        except Exception:
            expl_cls = shap.Explainer(rf_cls, Xtr_c)
            sv_cls = expl_cls(X_plot).values

        if isinstance(sv_cls, list):
            sv_cls = sv_cls[1]  # positive class
        sv_cls = np.asarray(sv_cls)
        if sv_cls.ndim == 3 and sv_cls.shape[0] == X_plot.shape[0]:
            sv_cls = sv_cls[:, :, 1]
        elif sv_cls.ndim == 3 and sv_cls.shape[-1] == X_plot.shape[1]:
            sv_cls = sv_cls[1, :, :]
        assert sv_cls.shape == X_plot.shape, f"SHAP shape mismatch: {sv_cls.shape} != {X_plot.shape}"

        plt.figure()
        shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
        _savefig_safe(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"))

        _save_df_safe(pd.DataFrame(sv_cls, columns=X_plot.columns),
                      os.path.join(FIG_DIR, "shap_values_classification.csv"))
        np.save(os.path.join(FIG_DIR, "shap_values_cls.npy"), sv_cls)

    except Exception as e:
        print(f"[ERR] SHAP classification failed: {e}")

# ---------- SHAP regression ----------
if MASTER_CTRL["RUN_XAI"] and rf_reg is not None:
    try:
        X_plot_r = Xte_r.copy()
        try:
            expl_reg = shap.TreeExplainer(rf_reg, feature_perturbation="interventional", model_output="raw")
            sv_reg = expl_reg.shap_values(X_plot_r, check_additivity=False)
        except Exception:
            expl_reg = shap.Explainer(rf_reg, Xtr_r)
            sv_reg = expl_reg(X_plot_r).values

        sv_reg = np.asarray(sv_reg)
        if sv_reg.ndim == 3 and sv_reg.shape[0] == X_plot_r.shape[0]:
            sv_reg = sv_reg[:, :, 0]
        elif sv_reg.ndim == 3 and sv_reg.shape[-1] == X_plot_r.shape[1]:
            sv_reg = sv_reg[0, :, :]
        assert sv_reg.shape == X_plot_r.shape, f"SHAP shape mismatch: {sv_reg.shape} != {X_plot_r.shape}"

        plt.figure()
        shap.summary_plot(sv_reg, X_plot_r.values, feature_names=X_plot_r.columns.tolist(), show=False)
        _savefig_safe(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"))

        _save_df_safe(pd.DataFrame(sv_reg, columns=X_plot_r.columns),
                      os.path.join(FIG_DIR, "shap_values_regression.csv"))
        np.save(os.path.join(FIG_DIR, "shap_values_reg.npy"), sv_reg)

    except Exception as e:
        print(f"[ERR] SHAP regression failed: {e}")

# ---------- LIME ----------
if MASTER_CTRL["RUN_XAI"] and len(np.unique(y_cls)) > 1:
    try:
        lime_explainer = LimeTabularExplainer(
            training_data=Xtr_c.values,
            feature_names=X_feat.columns.tolist(),
            discretize_continuous=True,
            mode='classification'
        )
        exp = lime_explainer.explain_instance(
            Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=min(5, X_feat.shape[1])
        )
        lime_list = exp.as_list(label=1)
        lime_df = pd.DataFrame(lime_list, columns=["feature", "weight"])
        _save_df_safe(lime_df, os.path.join(FIG_DIR, "lime_example_classification.csv"))

        # --- PATCH: add distinct colors for each feature ---
        colors = plt.cm.Set2(np.linspace(0, 1, len(lime_df)))

        plt.figure(figsize=(6,4))
        plt.barh(lime_df["feature"], lime_df["weight"], color=colors, edgecolor="black")
        plt.xlabel("LIME weight")
        plt.ylabel("Feature")
        plt.title("LIME explanation (stable=1)")
        plt.tight_layout()
        _savefig_safe(os.path.join(FIG_DIR, "lime_example_classification.png"))

    except Exception as e:
        print(f"[ERR] LIME failed: {e}")

# ======================================================
# 12) PATCH: Robust copy to Google Drive (counts + .txt allowed)
# ======================================================
if MASTER_CTRL["SAVE_DRIVE_COPY"]:
    print("\n[INFO] Files in FIG_DIR before Drive copy:")
    for fn in sorted(os.listdir(FIG_DIR)):
        print("   -", fn)

    GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_KL_Shannon"
    GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)
    os.makedirs(GOOGLE_DIR, exist_ok=True)

    copied, skipped = [], []
    for root, dirs, files in os.walk(SAVE_DIR):
        dst_dir = os.path.join(GOOGLE_DIR, os.path.relpath(root, SAVE_DIR))
        os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if not file.endswith((".png", ".fits", ".csv", ".json", ".txt", ".npy")):
                continue
            src = os.path.join(root, file)
            dst = os.path.join(dst_dir, file)
            try:
                if os.path.exists(dst) and os.path.samefile(src, dst):
                    skipped.append(dst)
                    continue
            except Exception:
                pass
            shutil.copy2(src, dst)
            copied.append(dst)

    print("☁️ Copy finished.")
    print(f"Copied: {len(copied)} files")
    print(f"Skipped (same path): {len(skipped)} files")
    print("Google Drive folder:", GOOGLE_DIR)
