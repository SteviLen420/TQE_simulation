# ===========================================
# Theory of the Question of Existence (TQE)
# Energy–Information Coupling Simulation
# ===========================================
# Author: Stefan Lengyel
# Purpose: Monte Carlo simulation with Goldilocks_KL divergence
# ===========================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import shap, lime, eli5
from captum.attr import IntegratedGradients
from interpret import show
import os, time, json, math, warnings, sys, subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt

# -- Ensure dependencies
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["qutip", "pandas", "scikit-learn", "scipy"]:
    _ensure(pkg)

import qutip as qt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from scipy.interpolate import make_interp_spline

warnings.filterwarnings("ignore")

# ======================================================
# Parameters
# ======================================================
# --- Parameters controlling the simulation ---
params = {
    "N_samples": 10000,   # number of universes (Monte Carlo runs)
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
# 1) Information parameter (I) via KL divergence
# ======================================================
def sample_information_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12
    KL = np.sum(p1 * np.log((p1+eps) / (p2+eps)))
    return KL / (1.0 + KL)   # 0 ≤ I ≤ 1

# ======================================================
# 2) Energy sampling
# ======================================================
def sample_energy_lognormal(mu=2.5, sigma=0.9):
    return float(rng.lognormal(mean=mu, sigma=sigma))

# ======================================================
# 3) Goldilocks noise function
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
# 4) Lock-in simulation
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
# 5) Monte Carlo universes
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
# 6) Stability curve (binned) + dynamic Goldilocks zone
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
# 7) Scatter E vs I
# ======================================================
plt.figure(figsize=(7,6))
plt.scatter(df["E"], df["I"], c=df["stable"], cmap="coolwarm", s=10, alpha=0.5)
plt.xlabel("Energy (E)"); plt.ylabel("Information parameter (I)")
plt.title("Universe outcomes in (E, I) space")
cbar = plt.colorbar(label="Stable=1 / Unstable=0")
savefig(os.path.join(FIG_DIR, "scatter_EI.png"))

# ======================================================
# Stability summary (counts + percentages)
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
# 8) Save summary
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

# ======================================================
# 9) Save all outputs to Google Drive
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
