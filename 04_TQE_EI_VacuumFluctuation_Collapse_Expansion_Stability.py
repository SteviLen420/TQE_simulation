# =============================================================================
# Theory of the Question of Existence (TQE)
# Energy–Information Coupling Simulation — KL × Shannon (multiplicative)
# =============================================================================
# Author: Stefan Len 
# Purpose: Monte Carlo simulation with Goldilocks + XAI (SHAP + LIME)
# =============================================================================
# SUMMARY
# This script simulates universes with random (E,I), checks stability,
# extracts Goldilocks zones, runs seed search, and generates explainability
# (XAI) with SHAP + LIME. Outputs are saved (CSV, PNG, JSON).
# =============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, warnings, sys, subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt

# --- Dependency check ---
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

# --- XAI stack: SHAP + LIME ---
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "shap==0.45.0", "lime==0.2.0.1", "scikit-learn==1.5.2", "-q"])
    import shap
    from lime.lime_tabular import LimeTabularExplainer

# ======================================================
# 1) Parameters
# ======================================================
params = {
    "N_samples": 1000,    # Monte Carlo universes
    "N_epoch": 30,        # time steps
    "rel_eps": 0.05,      # lock-in threshold
    "sigma0": 0.5,        # baseline noise
    "alpha": 1.5,         # noise growth toward edges
    "seed": 42            # RNG seed
}

rng = np.random.default_rng(seed=params["seed"])

# Output dirs
run_id  = time.strftime("TQE_(E,I)_SUPERPOSITION_%Y%m%d_%H%M%S")
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
# 2) Information parameter I = g(KL, Shannon)
# ======================================================
def sample_information_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1 = np.abs(psi1.full().flatten())**2
    p2 = np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12

    KL = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)))
    I_kl = KL / (1.0 + KL)

    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))

    I_raw = I_kl * I_shannon
    I = I_raw / (1.0 + I_raw)
    return float(I)

# ======================================================
# 3) Energy sampling
# ======================================================
def sample_energy_lognormal(mu=2.5, sigma=0.9):
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ======================================================
# 4) Goldilocks noise
# ======================================================
def sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high):
    if E_c_low is None or E_c_high is None:
        return sigma0
    if X < E_c_low or X > E_c_high:
        return sigma0 * 1.5
    mid = 0.5 * (E_c_low + E_c_high)
    width = max(0.5 * (E_c_high - E_c_low), 1e-12)
    dist = abs(X - mid) / width
    return sigma0 * (1 + alpha * dist**2)

# ======================================================
# 5) Lock-in simulation
# ======================================================
def simulate_lock_in(X, N_epoch, rel_eps=0.02, sigma0=0.2, alpha=1.0, E_c_low=None, E_c_high=None):
    A, ns, H = rng.normal(50, 5), rng.normal(0.8, 0.05), rng.normal(0.7, 0.08)
    locked_at, consecutive = None, 0
    for n in range(1, N_epoch + 1):
        sigma = sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high)
        A_prev, ns_prev, H_prev = A, ns, H
        A  += rng.normal(0, sigma)
        ns += rng.normal(0, sigma/10)
        H  += rng.normal(0, sigma/5)
        delta_rel = (abs(A - A_prev)/abs(A_prev) +
                     abs(ns - ns_prev)/abs(ns_prev) +
                     abs(H - H_prev)/abs(H_prev)) / 3.0
        if delta_rel < rel_eps:
            consecutive += 1
            if consecutive >= 15 and locked_at is None:
                locked_at = n
        else:
            consecutive = 0
    stable = 1 if (locked_at is not None and locked_at <= N_epoch) else 0
    return stable, (locked_at if locked_at is not None else -1)

# ======================================================
# 6) Monte Carlo universes
# ======================================================
rows = []
for i in range(params["N_samples"]):
    E = sample_energy_lognormal()
    I = sample_information_param(dim=8)
    X = E * I
    stable, lock_at = simulate_lock_in(X, params["N_epoch"], params["rel_eps"],
                                       params["sigma0"], params["alpha"])
    rows.append({"E": E, "I": I, "X": X, "stable": stable, "lock_at": lock_at})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_DIR, "samples.csv"), index=False)

# ======================================================
# 7) Stability curve + Goldilocks zone
# ======================================================
bins = np.linspace(df["X"].min(), df["X"].max(), 40)
df["bin"] = np.digitize(df["X"], bins)

bin_stats = df.groupby("bin").agg(
    mean_X=("X", "mean"),
    stable_rate=("stable", "mean"),
    count=("stable", "size")
).dropna()

xx = bin_stats["mean_X"].values
yy = bin_stats["stable_rate"].values

if len(xx) > 3:
    spline = make_interp_spline(xx, yy, k=3)
    xs = np.linspace(xx.min(), xx.max(), 300)
    ys = spline(xs)
else:
    xs, ys = xx, yy

peak_index = int(np.argmax(ys))
peak_x = float(xs[peak_index])
half_max = float(ys[peak_index] * 0.5)
valid_peak = xs[ys >= half_max]
if len(valid_peak) > 0:
    E_c_low, E_c_high = float(valid_peak.min()), float(valid_peak.max())
else:
    E_c_low, E_c_high = peak_x, peak_x

plt.figure(figsize=(8,5))
plt.scatter(xx, yy, s=30, alpha=0.7, label="bin means")
plt.plot(xs, ys, lw=2, label="spline fit")
plt.axvline(E_c_low,  ls='--', label=f"E_c_low = {E_c_low:.2f}")
plt.axvline(E_c_high, ls='--', label=f"E_c_high = {E_c_high:.2f}")
plt.xlabel("X = E·I"); plt.ylabel("P(stable)")
plt.title("Goldilocks zone: stabilization curve (KL × Shannon)")
plt.legend()
savefig(os.path.join(FIG_DIR, "stability_curve.png"))

# ======================================================
# 8) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information (I)")
plt.title("Universe outcomes in (E, I) space")
plt.colorbar(label="Stable=1 / Unstable=0")
savefig(os.path.join(FIG_DIR, "scatter_EI.png"))

# ======================================================
# 9) Stability summary
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = int(len(df) - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total: {len(df)}")
print(f"Stable:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")

plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")
plt.xticks([0, 1], [
    f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)"
])
savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# 10) Seed search (Top-5)
# ======================================================
NUM_SEEDS = 50
UNIVERSES_PER_SEED = 300

seed_scores = []
_old_rng = rng
for s in range(NUM_SEEDS):
    rng = np.random.default_rng(seed=s)
    np.random.seed(s)
    rows_s = []
    for i in range(UNIVERSES_PER_SEED):
        E = sample_energy_lognormal()
        I = sample_information_param(dim=8)
        X = E * I
        stable, lock_at = simulate_lock_in(X, params["N_epoch"], params["rel_eps"],
                                           params["sigma0"], params["alpha"])
        rows_s.append({"E":E, "I":I, "X":X, "stable":stable, "lock_at":lock_at})
    df_s = pd.DataFrame(rows_s)
    ratio = float(df_s["stable"].mean())
    locked_mask = df_s["lock_at"] >= 0
    locked_frac = float(locked_mask.mean())
    mean_lock = float(df_s.loc[locked_mask, "lock_at"].mean()) if locked_mask.any() else None
    seed_scores.append({
        "seed": s,
        "stable_ratio": ratio,
        "locked_fraction": locked_frac,
        "mean_lock_at": mean_lock
    })

rng = _old_rng
seed_scores_sorted = sorted(seed_scores, key=lambda r: r["stable_ratio"], reverse=True)

print("\n🏆 Top-5 seeds")
for r in seed_scores_sorted[:5]:
    print(f"Seed {r['seed']:3d} → stability={r['stable_ratio']:.3f}, "
          f"locked_frac={r['locked_fraction']:.3f}, mean_lock_at={r['mean_lock_at']}")

pd.DataFrame(seed_scores_sorted).to_csv(os.path.join(SAVE_DIR, "seed_search_top.csv"), index=False)

# ======================================================
# 11) XAI (SHAP + LIME)
# ======================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

X_feat = df[["E", "I", "X"]].copy()
y_cls  = df["stable"].astype(int).values
reg_mask = df["lock_at"] >= 0
X_reg = X_feat[reg_mask]; y_reg = df.loc[reg_mask, "lock_at"].values

Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)

have_reg = len(X_reg) >= 30
if have_reg:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)

rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classifier accuracy: {cls_acc:.3f}")

rf_reg, reg_r2 = None, None
if have_reg:
    rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R²: {reg_r2:.3f}")

# SHAP classification
try:
    expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional", model_output="raw")
    sv_cls = expl_cls.shap_values(Xte_c, check_additivity=False)
    if isinstance(sv_cls, list):
        sv_cls = sv_cls[1]
    plt.figure()
    shap.summary_plot(sv_cls, Xte_c.values, feature_names=Xte_c.columns.tolist(), show=False)
    savefig(os.path.join(FIG_DIR, "shap_summary_cls.png"))
except Exception as e:
    print("[ERR] SHAP classification failed:", e)

# LIME classification
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
    lime_df.to_csv(os.path.join(FIG_DIR, "lime_example.csv"), index=False)
    plt.figure(figsize=(6,4))
    plt.barh(lime_df["feature"], lime_df["weight"])
    plt.xlabel("LIME weight"); plt.ylabel("Feature")
    plt.title("LIME explanation (stable=1)")
    plt.tight_layout()
    savefig(os.path.join(FIG_DIR, "lime_example.png"))
except Exception as e:
    print("[ERR] LIME failed:", e)

# ======================================================
# 12) Save summary + Google Drive
# ======================================================
summary = {
    "params": params,
    "N_samples": int(len(df)),
    "stable_count": stable_count,
    "unstable_count": unstable_count,
    "stable_ratio": float(df["stable"].mean()),
    "E_c_low": E_c_low,
    "E_c_high": E_c_high,
    "top5_seeds": seed_scores_sorted[:5]
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_SUPERPOSITION"
GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)
os.makedirs(GOOGLE_DIR, exist_ok=True)

for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        if file.endswith((".png",".json",".csv")):
            src = os.path.join(root,file)
            dst = os.path.join(GOOGLE_DIR, os.path.relpath(root,SAVE_DIR))
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src,dst)

print(f"\n☁️ All results saved to Google Drive: {GOOGLE_DIR}")
print("✅ DONE.")
