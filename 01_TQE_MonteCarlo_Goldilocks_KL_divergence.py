# =============================================================================
# Theory of the Question of Existence (TQE)
# Energy–Information Coupling Simulation
# =============================================================================
# Author: Stefan Len
# Purpose: Monte Carlo simulation with Goldilocks_KL divergence
# =============================================================================
# SUMMARY
# This notebook implements a Monte Carlo simulation pipeline that models the
# coupling between energy (E) and information (I). The information parameter I
# is normalized from the KL divergence between random quantum states (0..1).
# We analyze stabilization (“law lock-in”) on the composite variable X = E·I,
# estimating both its probability and timing. The Goldilocks zone is detected
# via spline fitting on the P(stable | X) curve.
# =============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, math, warnings, sys, subprocess, shutil
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

# --- XAI stack: SHAP + LIME only (no eli5/captum/interpret) ---
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
# --- Parameters controlling the simulation ---
params = {
    "N_samples": 1000,   # number of universes (Monte Carlo runs)
    "N_epoch": 30,        # number of time steps (30 gives the most precise Goldilocks zone)
    "rel_eps": 0.05,      # lock-in threshold: max allowed relative change for stability
    "sigma0": 0.5,        # baseline noise amplitude
    "alpha": 1.5,         # noise growth factor toward the edges of the Goldilocks zone
    "seed":  None         # random seed for reproducibility
}

rng = np.random.default_rng(seed=params["seed"])

# Output dirs
run_id  = time.strftime("TQE_(E,I)KL_divergence_%Y%m%d_%H%M%S")
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
# 2) Information parameter (I) via KL divergence
# ======================================================
def sample_information_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12
    KL = np.sum(p1 * np.log((p1+eps) / (p2+eps)))
    return KL / (1.0 + KL)   # 0 ≤ I ≤ 1

# ======================================================
# 3) Energy sampling
# ======================================================
def sample_energy_lognormal(mu=2.5, sigma=0.9):
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ======================================================
# 4) Goldilocks noise function
# ======================================================
def sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high):
    """
    Noise function:
    - If X is outside the zone → high noise (unstable)
    - If inside → noise increases as you approach the edges
    """
    if E_c_low is None or E_c_high is None:
        return sigma0
    if X < E_c_low or X > E_c_high:
        return sigma0 * 1.5
    else:
        mid = 0.5 * (E_c_low + E_c_high)
        width = 0.5 * (E_c_high - E_c_low)
        dist = abs(X - mid) / width   # 0 in center, 1 at edges
        return sigma0 * (1 + alpha * dist**2)

# ======================================================
# 5) Lock-in simulation
# ======================================================
def simulate_lock_in(X, N_epoch, rel_eps=0.02, sigma0=0.2, alpha=1.0, E_c_low=None, E_c_high=None):
    A, ns, H = rng.normal(50, 5), rng.normal(0.8, 0.05), rng.normal(0.7, 0.08)
    locked_at, consecutive = None, 0

    for n in range(1, N_epoch+1):
        sigma = sigma_goldilocks(X, sigma0, alpha, E_c_low, E_c_high)

        A_prev, ns_prev, H_prev = A, ns, H
        A  += rng.normal(0, sigma)
        ns += rng.normal(0, sigma/10)
        H  += rng.normal(0, sigma/5)

        # relative change
        delta_rel = (abs(A - A_prev)/abs(A_prev) +
                     abs(ns - ns_prev)/abs(ns_prev) +
                     abs(H - H_prev)/abs(H_prev)) / 3.0

        if delta_rel < rel_eps:         # much stricter threshold (2%)
            consecutive += 1            # count consecutive calm steps
            if consecutive >= 15 and locked_at is None:  # need at least 20 calm steps
                locked_at = n
        else:
            consecutive = 0

    stable = 1 if (locked_at is not None and locked_at <= N_epoch) else 0
    return stable, locked_at if locked_at is not None else -1

# ======================================================
# 6) Monte Carlo universes
# ======================================================
rows = []
for i in range(params["N_samples"]):
    E   = sample_energy_lognormal()
    I   = sample_information_param(dim=8)
    X   = E * I
    stable, lock_at = simulate_lock_in(X,
                                       params["N_epoch"],
                                       params["rel_eps"],
                                       params["sigma0"],
                                       params["alpha"])
    rows.append({"E":E, "I":I, "X":X, "stable":stable, "lock_at":lock_at})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SAVE_DIR, "samples.csv"), index=False)

# ======================================================
# 7) Stability curve (binned) + dynamic Goldilocks zone
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

# --- Detect Goldilocks zone around main peak ---
peak_index = np.argmax(ys)
peak_x = xs[peak_index]

half_max = ys[peak_index] * 0.5
valid_peak = xs[ys >= half_max]

if len(valid_peak) > 0:
    E_c_low, E_c_high = valid_peak.min(), valid_peak.max()
else:
    E_c_low, E_c_high = peak_x, peak_x
    print("⚠️ No clear peak zone found, defaulting to peak only.")

# --- Plot stability curve ---
plt.figure(figsize=(8,5))
plt.scatter(xx, yy, s=30, c="blue", alpha=0.7, label="bin means")
plt.plot(xs, ys, "r-", lw=2, label="spline fit")

# always draw the zone boundary lines
plt.axvline(E_c_low, color='g', ls='--', label=f"E_c_low = {E_c_low:.1f}")
plt.axvline(E_c_high, color='m', ls='--', label=f"E_c_high = {E_c_high:.1f}")

plt.xlabel("X = E·I")
plt.ylabel("P(stable)")
plt.title("Goldilocks zone: stabilization curve")
plt.legend()
savefig(os.path.join(FIG_DIR, "stability_curve.png"))

# ======================================================
# 8) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information parameter (I)")
plt.title("Universe outcomes in (E, I) space")
cbar = plt.colorbar(label="Stable=1 / Unstable=0")
savefig(os.path.join(FIG_DIR, "scatter_EI.png"))

# ======================================================
# 9) Stability summary (counts + percentages)
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = int(len(df) - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {len(df)}")
print(f"Stable universes:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")

# --- Save bar chart ---
plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")
labels = [
f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)"
]
plt.xticks([0, 1], labels)

savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# 10) Save summary
# ======================================================
summary = {
    "params": params,
    "N_samples": int(len(df)),
    "stable_count": int(df["stable"].sum()),              # number of stable universes
    "unstable_count": int((1 - df["stable"]).sum()),      # number of unstable universes
    "stable_ratio": float(df["stable"].mean()),           # fraction of stable universes
    "unstable_ratio": float(1 - df["stable"].mean()),
    "E_c_low": E_c_low,
    "E_c_high": E_c_high,
    "figures": {
        "stability_curve": os.path.join(FIG_DIR, "stability_curve.png"),
        "scatter_EI": os.path.join(FIG_DIR, "scatter_EI.png"),
        "stability_summary": os.path.join(FIG_DIR, "stability_summary.png")
    }
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# Print summary to console
print("\n✅ DONE.")
print(f"Runs: {len(df)}")
print(f"Stable universes: {summary['stable_count']}")
print(f"Unstable universes: {summary['unstable_count']}")
print(f"Stability ratio: {summary['stable_ratio']:.3f}")
print(f"Goldilocks zone: {E_c_low:.1f} – {E_c_high:.1f}" if E_c_low else "No stable zone found")
print(f"📂 Directory: {SAVE_DIR}")

save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# Print summary to console
print("\n✅ DONE.")
print(f"Runs: {len(df)}")
print(f"Stable universes: {summary['stable_count']}")
print(f"Unstable universes: {summary['unstable_count']}")
print(f"Stability ratio: {summary['stable_ratio']:.3f}")
print(f"Goldilocks zone: {E_c_low:.1f} – {E_c_high:.1f}" if E_c_low else "No stable zone found")
print(f"📂 Directory: {SAVE_DIR}")

# ======================================================
# 11) XAI (SHAP + LIME) + DARWIN prompt preparation
# ======================================================

# Install on demand (only if missing)
try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "lime", "scikit-learn", "-q"])
    import shap
    from lime.lime_tabular import LimeTabularExplainer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# ---------- Features and targets ----------
# Global feature matrix
X_feat = df[["E", "I", "X"]].copy()

# Classification target: stability (0/1)
y_cls = df["stable"].astype(int).values

# Regression target: lock-in time (only where it happened: lock_at >= 0)
reg_mask = df["lock_at"] >= 0
X_reg = X_feat[reg_mask]
y_reg = df.loc[reg_mask, "lock_at"].values

# ---------- Train/Test split ----------
# Classification
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)

# Regression (only if we have enough lock-in samples)
have_reg = len(X_reg) >= 30
if have_reg:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

# ---------- Train models ----------
# Classifier: what drives stability?
rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

# Regressor: if enough data, what affects lock-in time?
if have_reg:
    rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R^2 (lock_at): {reg_r2:.3f}")
else:
    rf_reg, reg_r2 = None, None
    print("[XAI] Not enough locked samples for regression (need ~30+).")

# ---------- SHAP: global explanations ----------
# Classification SHAP (positive class = stable=1)
expl_cls = shap.TreeExplainer(rf_cls)
shap_vals_cls = expl_cls.shap_values(X_feat)[1]  # class 1

plt.figure()
shap.summary_plot(shap_vals_cls, X_feat, show=False)
plt.title("SHAP summary – classification (stable)")
plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"), dpi=220, bbox_inches="tight")
plt.close()

# Regression SHAP (if trained)
if rf_reg is not None:
    expl_reg = shap.TreeExplainer(rf_reg)
    shap_vals_reg = expl_reg.shap_values(X_reg)

    plt.figure()
    shap.summary_plot(shap_vals_reg, X_reg, show=False)
    plt.title("SHAP summary – regression (lock_at)")
    plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # Local SHAP example for a single sample
    i = 0
    shap.force_plot(
        expl_reg.expected_value, shap_vals_reg[i, :], X_reg.iloc[i, :],
        matplotlib=True, show=False
    )
    plt.savefig(os.path.join(FIG_DIR, "shap_force_reg_example.png"), dpi=220, bbox_inches="tight")
    plt.close()

# ---------- LIME: local explanation (classification) ----------
lime_explainer = LimeTabularExplainer(
    training_data=Xtr_c.values,
    feature_names=X_feat.columns.tolist(),
    discretize_continuous=True,
    mode='classification'
)

exp = lime_explainer.explain_instance(
    Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=5
)
lime_list = exp.as_list(label=1)  # explanation for class 'stable=1'
pd.DataFrame(lime_list, columns=["feature", "weight"]).to_csv(
    os.path.join(FIG_DIR, "lime_example_classification.csv"), index=False
)

# ---------- DARWIN LLM prompt generation ----------
# Create a standalone prompt you can feed into DARWIN
darwin_prompt = f"""
Give a full scientific interpretation of the following Monte Carlo simulation.

Context:
- Samples (universes): {len(df)}
- Stability ratio: {summary['stable_ratio']:.3f}
- Goldilocks zone (X = E·I): [{summary['E_c_low']:.3f}, {summary['E_c_high']:.3f}]
- Features available: E (energy), I (information via KL), X = E·I

What to cover:
1) Global behavior of the stability curve in X, including the detected Goldilocks window.
2) The role of E and I individually, and their interaction through X.
3) Connections to physical concepts (criticality, phase transitions, decoherence, cosmology).
4) Hypotheses for why stabilization occurs in this narrow window.
5) Limitations of the simulation setup and suggestions for further tests (e.g., ablations).

Optional (if helpful): Relate to literature on lock-in/critical phenomena and CMB-like anomalies.
"""

with open(os.path.join(SAVE_DIR, "DARWIN_prompt.txt"), "w") as f:
    f.write(darwin_prompt)

# Context JSON (numbers) you can attach alongside the prompt
darwin_ctx = {
    "N": int(len(df)),
    "stable_ratio": float(summary["stable_ratio"]),
    "E_c_low": float(summary["E_c_low"]),
    "E_c_high": float(summary["E_c_high"]),
    "cls_accuracy": float(cls_acc),
    "reg_r2": float(reg_r2) if reg_r2 is not None else None,
    "features": ["E", "I", "X"],
    "xai_artifacts": {
        "shap_cls": os.path.join(FIG_DIR, "shap_summary_cls_stable.png"),
        "shap_reg": os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png") if reg_r2 is not None else None,
        "shap_force_reg_example": os.path.join(FIG_DIR, "shap_force_reg_example.png") if reg_r2 is not None else None,
        "lime_cls_csv": os.path.join(FIG_DIR, "lime_example_classification.csv")
    }
}
save_json(os.path.join(SAVE_DIR, "DARWIN_context.json"), darwin_ctx)

print("🧠 XAI (SHAP+LIME) done. ✍️ DARWIN prompt saved:", os.path.join(SAVE_DIR, "DARWIN_prompt.txt"))

# ======================================================
# 12) Save all outputs to Google Drive
# ======================================================
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_KL_divergence"
GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)  # separate folder for each run
os.makedirs(GOOGLE_DIR, exist_ok=True)

for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        if file.endswith((".png", ".fits", ".csv", ".json")):
            src = os.path.join(root, file)
            dst_dir = os.path.join(GOOGLE_DIR, os.path.relpath(root, SAVE_DIR))
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, dst_dir)

print(f"☁️ All results saved to Google Drive: {GOOGLE_DIR}")
