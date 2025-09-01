# ===========================================================================
# Theory of the Question of Existence (TQE) — E-only
# Vacuum fluctuation → Collapse → Expansion → Stability → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Energy-only (I = 0) simulation with a complete pipeline:
#   - Quantum superposition (diagnostic)
#   - Collapse snapshot at t = 0
#   - Monte Carlo over universes 
#   - Stability + law lock-in detection
#   - Optional averaged lock-in plots
#   - Full CSV/JSON/PNG outputs
#   - SHAP/LIME explainability (classification; regression if enough lock-ins)
# ===========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import qutip as qt
import warnings, sys, subprocess
warnings.filterwarnings("ignore")

# ======================================================
# 0) Dependency check (install on demand)
# ======================================================
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
for pkg in ["pandas", "numpy", "matplotlib", "qutip", "scikit-learn", "shap", "lime"]:
    _ensure(pkg)

import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

# ======================================================
# Setup: directories, helpers, and global controls
# ======================================================
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E)_UNIVERSE_SIMULATION"
run_id = time.strftime("TQE_(E)_UNIVERSE_SIMULATION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p): 
    plt.savefig(p, dpi=180, bbox_inches="tight"); plt.close()

def savejson(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# Plot flags (set False to skip PNGs while still saving CSVs)
PLOT_AVG_LOCKIN  = True     # average c(t) across lock-in histories
PLOT_LOCKIN_HIST = True     # histogram of lock-in epochs

# ======================================================
# MASTER SIMULATION CONTROLS
# ======================================================

MASTER_CTRL = {
    "NUM_UNIVERSES":     5000,
    "TIME_STEPS":        800,
    "LOCKIN_EPOCHS":     500,
    "EXPANSION_EPOCHS":  800,
    "BEST_STEPS":        800,
    "BEST_NUM_REGIONS":  10,      
    "BEST_NUM_STATES":   500, 
    "STABILITY_THRESHOLD": 3.5,

    # Energy distribution
    "E_LOG_MU":          2.5,
    "E_LOG_SIGMA":       0.8,
    "E_CENTER":          2.0,   # Goldilocks center
    "E_WIDTH":           0.3,   # Goldilocks width

    # Stability thresholds
    "REL_EPS_STABLE":    0.05,
    "REL_EPS_LOCKIN":    1e-3,
    "CALM_STEPS":        5,

    # Randomness
    "SEED": None,              # or set an int for reproducibility

    # Plot / XAI flags
    "PLOT_AVG_LOCKIN":   True,
    "PLOT_LOCKIN_HIST":  True,
    "RUN_XAI":           True,
}

# ======================================================
# 1) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
Nlev = 12
a = qt.destroy(Nlev)

# Slightly perturbed Hamiltonian (adds random noise to the ladder operator)
H0 = a.dag()*a + 0.05*(np.random.randn()*a + np.random.randn()*a.dag())

# Random initial ket and density matrix
psi0 = qt.rand_ket(Nlev)
rho = psi0 * psi0.dag()

# Short evolutions with a time-varying dissipation rate
tlist  = np.linspace(0, 10, 200)
gammas = 0.02 + 0.01*np.sin(0.5*tlist) + 0.005*np.random.randn(len(tlist))

states = []
for g in gammas:
    res = qt.mesolve(H0, rho, np.linspace(0,0.5,5), [np.sqrt(abs(g))*a], [])
    states.append(res.states[-1])

def purity(r): 
    return float((r*r).tr().real) if qt.isoper(r) else float((r*r.dag()).tr().real)

S = np.array([qt.entropy_vn(r) for r in states])
P = np.array([purity(r) for r in states])

plt.figure()
plt.plot(tlist, S, label="Entropy")
plt.plot(tlist, P, label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend()
savefig(os.path.join(FIG_DIR, "superposition.png"))

pd.DataFrame({"time": tlist, "Entropy": S, "Purity": P}).to_csv(
    os.path.join(SAVE_DIR, "superposition.csv"), index=False
)

# ======================================================
# 2) t = 0 : Collapse snapshot (E only)
# ======================================================
def sample_energy(mu=None, sigma=None):
    if mu is None: mu = MASTER_CTRL["E_LOG_MU"]
    if sigma is None: sigma = MASTER_CTRL["E_LOG_SIGMA"]
    return float(np.random.lognormal(mean=mu, sigma=sigma))

E0 = sample_energy()
X0 = E0  # E-only: composite X ≡ E

collapse_t = np.linspace(-0.2, 0.2, 200)
X_series = X0 + 0.5*np.random.randn(len(collapse_t))          # fluctuation before t=0
X_series[collapse_t >= 0] = X0 + 0.05*np.random.randn(np.sum(collapse_t >= 0))  # calm after t=0

plt.figure()
plt.plot(collapse_t, X_series, alpha=0.7, label="fluctuation → lock-in")
plt.axhline(X0, color="r", ls="--", label=f"Lock-in X≈{X0:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E only)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

pd.DataFrame({"time": collapse_t, "X_vals": X_series}).to_csv(
    os.path.join(SAVE_DIR, "collapse.csv"), index=False
)

# ======================================================
# 3) Stability criterion (toy model, E only)
# ======================================================
def is_stable(E, n_epoch=None):
    if n_epoch is None:
        n_epoch = MASTER_CTRL["TIME_STEPS"]
    A, calm = 20.0, 0
    for _ in range(n_epoch):
        A_prev = A
        A = A*1.02 + np.random.normal(0, 2.0)
        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        calm = calm + 1 if delta < MASTER_CTRL["REL_EPS_STABLE"] else 0
        if calm >= MASTER_CTRL["CALM_STEPS"]:
            return 1
    return 0

# ======================================================
# 4) Law lock-in dynamics (E only)
# ======================================================
def law_lock_in(E, n_epoch=None):
    if n_epoch is None:
        n_epoch = MASTER_CTRL["LOCKIN_EPOCHS"]
    Ec = MASTER_CTRL["E_CENTER"]
    sigma = MASTER_CTRL["E_WIDTH"]

    f = np.exp(-(E - Ec)**2 / (2 * sigma**2))  # one-dimensional Goldilocks (no I)
    if f < 0.2:
        return -1, []  # too far from Goldilocks → no lock-in at all

    c_val = np.random.normal(3e8, 1e7)
    calm, locked_at = 0, None
    history = []

    for n in range(n_epoch):
        prev = c_val
        noise = 1e6 * (1 + abs(E - 5) / 10) * np.random.uniform(0.8, 1.2)
        c_val += np.random.normal(0, noise)
        history.append(c_val)

        delta = abs(c_val - prev) / max(abs(prev), 1e-9)
        if delta < MASTER_CTRL["REL_EPS_LOCKIN"]:
            calm += 1
            if calm >= MASTER_CTRL["CALM_STEPS"] and locked_at is None:
                locked_at = n
        else:
            calm = 0

    return locked_at if locked_at is not None else -1, history

# ======================================================
# 5) Monte Carlo over universes (or single universe)
# ======================================================
E_vals, X_vals, f_vals = [], [], []
stables, law_epochs, final_cs, all_histories = [], [], [], []

for _ in range(MASTER_CTRL["NUM_UNIVERSES"]):
    Ei = sample_energy()
    fi = np.exp(-(Ei - MASTER_CTRL["E_CENTER"])**2 / (2 * MASTER_CTRL["E_WIDTH"]**2))
    E_vals.append(Ei)
    X_vals.append(Ei)
    f_vals.append(fi)

    s = is_stable(Ei)
    stables.append(s)

    if s == 1:
        lock_epoch, c_hist = law_lock_in(Ei, n_epoch=MASTER_CTRL["LOCKIN_EPOCHS"])
        law_epochs.append(lock_epoch)
        if len(c_hist) > 0:
            final_cs.append(c_hist[-1])
            all_histories.append(c_hist)
        else:
            final_cs.append(np.nan)
    else:
        law_epochs.append(-1)
        final_cs.append(np.nan)

valid_epochs = [e for e in law_epochs if e >= 0]
median_epoch = float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None

# Save master run table
df = pd.DataFrame({
    "E": E_vals,
    "X": X_vals,
    "stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs
})
df.to_csv(os.path.join(SAVE_DIR, "tqe_runs.csv"), index=False)

# Quick diagnostics
num_stable = int(np.sum(stables))
num_lockin = int(np.sum([e >= 0 for e in law_epochs]))
print(f"\n🔒 Universes with lock-in: {num_lockin} / {MASTER_CTRL['NUM_UNIVERSES']}")

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {MASTER_CTRL['NUM_UNIVERSES']}")
print(f"Stable universes:   {num_stable} ({num_stable/MASTER_CTRL['NUM_UNIVERSES']*100:.2f}%)")
print(f"Unstable universes: {MASTER_CTRL['NUM_UNIVERSES']-num_stable} "
      f"({100*(MASTER_CTRL['NUM_UNIVERSES']-num_stable)/MASTER_CTRL['NUM_UNIVERSES']:.2f}%)")

# Convenience export (stability-only view)
df[["E","X","stable","lock_epoch","final_c"]].to_csv(
    os.path.join(SAVE_DIR,"stability.csv"), index=False
)

# ======================================================
# 6) Stability summary bar chart  — with lock-in split
# ======================================================
num_unstable = MASTER_CTRL["NUM_UNIVERSES"] - num_stable
num_locked   = int(np.sum([e >= 0 for e in law_epochs]))
num_stable_no_lock = max(num_stable - num_locked, 0)               # stable but no lock-in

# percentages for labels
pct_stable_no_lock = 100.0 * num_stable_no_lock / MASTER_CTRL["NUM_UNIVERSES"]
pct_locked         = 100.0 * num_locked / MASTER_CTRL["NUM_UNIVERSES"]
pct_unstable       = 100.0 * num_unstable / MASTER_CTRL["NUM_UNIVERSES"]

# --- plot (stacked bar for Stable) ---
plt.figure()

# x positions: 0 = Stable (stacked), 1 = Unstable
plt.bar(0, num_stable_no_lock, color="#1f77b4", label=f"Stable (no lock-in) [{num_stable_no_lock}, {pct_stable_no_lock:.1f}%]")
plt.bar(0, num_locked, bottom=num_stable_no_lock, color="#9467bd", label=f"Stable (lock-in) [{num_locked}, {pct_locked:.1f}%]")
plt.bar(1, num_unstable, color="#d62728", label=f"Unstable [{num_unstable}, {pct_unstable:.1f}%]")

plt.xticks([0, 1], ["Stable (split by lock-in)", "Unstable"])
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution (E-only) — with Lock-in")
plt.legend(loc="upper right", frameon=False)
plt.tight_layout()
savefig(os.path.join(FIG_DIR, "stability_summary_with_lockin.png"))

# ======================================================
# 7) Average law lock-in dynamics across all universes
# ======================================================
if len(all_histories) > 0:
    min_len   = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]
    avg_c = np.mean(truncated, axis=0)
    std_c = np.std(truncated, axis=0)

    # Always save CSV
    pd.DataFrame({"epoch": np.arange(min_len), "avg_c": avg_c, "std_c": std_c}).to_csv(
        os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False
    )

    # Optional PNG
    if PLOT_AVG_LOCKIN and (median_epoch is not None):
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c, alpha=0.3, label="±1σ")
        plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (Monte Carlo, E-only)")
        plt.xlabel("epoch"); plt.ylabel("c value (m/s)"); plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 8) t > 0 : Expansion dynamics (E only)
# ======================================================
def evolve(E, n_epoch=None):
    if n_epoch is None:
        n_epoch = MASTER_CTRL["EXPANSION_EPOCHS"]
    A_series, A = [], 20.0
    for _ in range(n_epoch):
        A = A*1.005 + np.random.normal(0, 1.0)
        A_series.append(A)
    return A_series

A_series = evolve(E0)
plt.figure()
plt.plot(A_series, label="Amplitude A")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.6, label="Equilibrium A")
title_suffix = "" if (median_epoch is not None) else " (no lock-in observed)"
if median_epoch is not None:
    plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {int(median_epoch)}")
plt.title("t > 0 : Expansion dynamics" + title_suffix)
plt.xlabel("epoch"); plt.ylabel("Amplitude A"); plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# ======================================================
# 9) Histogram of lock-in epochs (CSV always, PNG optional)
# ======================================================
pd.DataFrame({"lock_epoch": valid_epochs}).to_csv(
    os.path.join(SAVE_DIR, "law_lockin_epochs.csv"), index=False
)

if PLOT_LOCKIN_HIST and len(valid_epochs) > 0:
    plt.figure()
    plt.hist(valid_epochs, bins=50, alpha=0.75)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Median lock-in = {int(median_epoch)}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (Monte Carlo, E-only)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# 10) Summary JSON (key aggregates)
# ======================================================
summary = {
    "simulation": {
        "total_universes": MASTER_CTRL["NUM_UNIVERSES"],
        "stable_fraction": float(np.mean(stables)),
        "unstable_fraction": 1.0 - float(np.mean(stables))
    },
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "mean_X": float(np.mean(X_vals)),
        "std_X": float(np.std(X_vals))
    },
    "law_lockin": {
        "mean_lock_epoch": (float(np.mean(valid_epochs)) if len(valid_epochs) > 0 else None),
        "median_lock_epoch": (float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None),
        "locked_fraction": float(np.mean([1 if e >= 0 else 0 for e in law_epochs])),
        "mean_final_c": (float(np.nanmean(final_cs)) if len(final_cs) > 0 else None),
        "std_final_c": (float(np.nanstd(final_cs)) if len(final_cs) > 0 else None)
    }
}
savejson(os.path.join(SAVE_DIR, "summary.json"), summary)

# ======================================================
# 10b) "Best universe" deep-dive — entropy plot like the screenshot
#       (pick earliest-locking stable universe, then simulate regions)
# ======================================================

# ----- Scoring rule -----
# among stable universes with lock-in (lock_epoch >= 0), pick the one with
# the *earliest* lock-in. if none locked, fall back to the highest f(E,I).
locked_idxs = [i for i, e in enumerate(law_epochs) if e >= 0 and stables[i] == 1]

if len(locked_idxs) > 0:
    best_idx = locked_idxs[int(np.argmin([law_epochs[i] for i in locked_idxs]))]
    reason = f"earliest lock-in (epoch={law_epochs[best_idx]})"
else:
    best_idx = int(np.argmax(f_vals))
    reason = "no lock-ins → picked max f(E)"

E_best = E_vals[best_idx]
print(f"[BEST] Universe index={best_idx} chosen by {reason}; E*={E_best:.3f}")

# ----- Single-universe entropy simulator (same style as your screenshot) -----

def simulate_entropy_universe(E, I=0.0,
                              steps=None,
                              num_regions=None,
                              num_states=None):
    """
    Runs a single-universe entropy evolution with f(E,I) modulation.
    Returns: (region_entropies_list, global_entropy_list, lock_in_step)
    """
    if steps is None: steps = MASTER_CTRL["BEST_STEPS"]
    if num_regions is None: num_regions = MASTER_CTRL["BEST_NUM_REGIONS"]
    if num_states is None: num_states = MASTER_CTRL["BEST_NUM_STATES"]

    def f_EI_local(E, I, E_c=MASTER_CTRL["E_CENTER"], sigma=MASTER_CTRL["E_WIDTH"]):
        return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + I)

    from scipy.stats import entropy

    # init states: break symmetry
    states = np.zeros((num_regions, num_states))
    states[0, :] = 1.0

    region_entropies, global_entropy = [], []
    lock_in_step, consecutive_calm = None, 0

    # dynamic vars
    A = 1.0
    orient = float(I)
    E_run = float(E)

    for step in range(steps):
        noise_scale = max(0.02, 1.0 - step / steps)

        if step > 0:
            A = A * 1.01 + np.random.normal(0, 0.02)
            orient += (0.5 - orient) * 0.10 + np.random.normal(0, 0.02)
            orient = np.clip(orient, 0, 1)

        E_run += np.random.normal(0, 0.05)
        f_step_base = f_EI_local(E_run, I)

        for r in range(num_regions):
            noise = np.random.normal(0, noise_scale * 5.0, num_states)
            if np.random.rand() < 0.05:
                noise += np.random.normal(0, 8.0, num_states)
            f_step = f_step_base * (1 + np.random.normal(0, 0.1))
            states[r] += f_step * noise
            states[r] = np.clip(states[r], 0, 1)

        region_entropies.append([entropy(states[r]) for r in range(num_regions)])
        global_entropy.append(entropy(states.flatten()))

        if step > 0:
            prev, cur = global_entropy[-2], global_entropy[-1]
            delta = abs(cur - prev) / max(prev, 1e-9)
            if delta < 0.001:
                consecutive_calm += 1
                if consecutive_calm >= 10 and lock_in_step is None:
                    lock_in_step = step
            else:
                consecutive_calm = 0

    return region_entropies, global_entropy, lock_in_step

# ----- Run the deep-dive sim on the chosen (E*, I*) -----
best_region_entropies, best_global_entropy, best_lock = simulate_entropy_universe(
    E_best, I=0.0, steps=MASTER_CTRL["BEST_STEPS"]
)

# ----- Save CSVs -----
# global entropy
pd.DataFrame({
    "time": np.arange(len(best_global_entropy)),
    "global_entropy": best_global_entropy
}).to_csv(os.path.join(SAVE_DIR, "best_universe_global_entropy.csv"), index=False)

# per-region entropy (wide)
best_re_mat = np.array(best_region_entropies)  # shape: (steps, regions)
re_cols = [f"region_{i}_entropy" for i in range(best_re_mat.shape[1])]
pd.DataFrame(best_re_mat, columns=re_cols).assign(time=np.arange(best_re_mat.shape[0])) \
  .to_csv(os.path.join(SAVE_DIR, "best_universe_region_entropies.csv"), index=False)

# ----- Plot like the screenshot -----
plt.figure(figsize=(12, 6))
time_axis = np.arange(len(best_global_entropy))

# region curves
for r in range(min(MASTER_CTRL["BEST_NUM_REGIONS"], best_re_mat.shape[1])):
    plt.plot(time_axis, best_re_mat[:, r], lw=1, label=f"Region {r} entropy")

plt.axhline(y=MASTER_CTRL["STABILITY_THRESHOLD"], color="red", linestyle="--", label="Stability threshold")

# lock-in indicator
if best_lock is not None:
    plt.axvline(x=best_lock, color="purple", linestyle="--", linewidth=2,
                label=f"Lock-in step = {best_lock}")

plt.title("Best-universe entropy evolution (chosen from MC)")
plt.xlabel("Time step"); plt.ylabel("Entropy"); plt.legend(ncol=2)
plt.grid(True, alpha=0.3)
savefig(os.path.join(FIG_DIR, "best_universe_entropy_evolution.png"))

# ======================================================
# 11) XAI: SHAP + LIME (classification; regression if enough data)
# ======================================================
# Features/targets (E-only: keep X for symmetry; X == E)
X_feat = df[["E","X"]].copy()
y_cls  = df["stable"].astype(int).values

# Lock-in regression mask
reg_mask = df["lock_epoch"] >= 0
X_reg = X_feat[reg_mask]
y_reg = df.loc[reg_mask, "lock_epoch"].values

# Train/test split for classification
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls, test_size=0.25, random_state=42, stratify=y_cls
)
rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

# SHAP (classification)
X_plot = Xte_c.copy()
try:
    expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional", model_output="raw")
    sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
except Exception:
    expl_cls = shap.Explainer(rf_cls, Xtr_c)
    sv_cls = expl_cls(X_plot).values

# --- SHAP (classification) ---
X_plot = Xte_c.copy()

# Get SHAP values; fall back to generic explainer if TreeExplainer fails
try:
    expl_cls = shap.TreeExplainer(
        rf_cls,
        feature_perturbation="interventional",
        model_output="raw"
    )
    sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
except Exception:
    expl_cls = shap.Explainer(rf_cls, Xtr_c)
    sv_cls = expl_cls(X_plot).values  # may already be (n_samples, n_features)

# Normalize to shape (n_samples, n_features)
if isinstance(sv_cls, list):
    # Binary classification returns [class0, class1]; keep the positive class
    sv_cls = sv_cls[1]

sv_cls = np.asarray(sv_cls)
if sv_cls.ndim == 3:
    # Common case: (n_samples, n_classes, n_features) → take class 1 slice
    # Adjust the index if your positive class is different
    sv_cls = sv_cls[:, 1, :]

# Save SHAP values and global importance
pd.DataFrame(sv_cls, columns=X_plot.columns).to_csv(
    os.path.join(FIG_DIR, "shap_values_classification.csv"),
    index=False
)

cls_importance = pd.Series(np.mean(np.abs(sv_cls), axis=0), index=X_plot.columns) \
                   .sort_values(ascending=False)
cls_importance.to_csv(
    os.path.join(FIG_DIR, "shap_feature_importance_classification.csv"),
    header=["mean_|shap|"]
)

# SHAP summary plot
plt.figure()
shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
plt.title("SHAP summary – classification (stable, E-only)")
savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"))
cls_importance = pd.Series(np.mean(np.abs(sv_cls), axis=0), index=X_plot.columns).sort_values(ascending=False)
cls_importance.to_csv(os.path.join(FIG_DIR, "shap_feature_importance_classification.csv"), header=["mean_|shap|"])

plt.figure()
shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
plt.title("SHAP summary – classification (stable, E-only)")
savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"))

# SHAP (regression) only if enough lock-ins exist (≥ 30 samples)
if len(X_reg) >= 30:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_reg, y_reg, test_size=0.25, random_state=42)
    rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R^2 (lock_epoch): {reg_r2:.3f}")

    try:
        expl_reg = shap.TreeExplainer(rf_reg, feature_perturbation="interventional", model_output="raw")
        sv_reg = expl_reg.shap_values(Xte_r, check_additivity=False)
    except Exception:
        expl_reg = shap.Explainer(rf_reg, Xtr_r)
        sv_reg = expl_reg(Xte_r).values

    pd.DataFrame(np.asarray(sv_reg), columns=X_reg.columns).to_csv(
        os.path.join(FIG_DIR, "shap_values_regression_lock_epoch.csv"), index=False
    )
    reg_importance = pd.Series(np.mean(np.abs(sv_reg), axis=0), index=X_reg.columns).sort_values(ascending=False)
    reg_importance.to_csv(os.path.join(FIG_DIR, "shap_feature_importance_regression_lock_epoch.csv"), header=["mean_|shap|"])

    plt.figure()
    shap.summary_plot(sv_reg, Xte_r.values, feature_names=X_reg.columns.tolist(), show=False)
    plt.title("SHAP summary – regression (lock_epoch, E-only)")
    savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_epoch.png"))
else:
    print("[XAI] Not enough lock-in samples for regression (need ≥ 30).")

# LIME (classification) — only if at least 2 classes exist
if len(np.unique(y_cls)) > 1:
    lime_explainer = LimeTabularExplainer(
        training_data=Xtr_c.values,
        feature_names=X_feat.columns.tolist(),
        discretize_continuous=True,
        mode='classification'
    )
    exp = lime_explainer.explain_instance(
        Xte_c.iloc[0].values,
        rf_cls.predict_proba,
        num_features=5
    )
    label_index = 1 if rf_cls.n_classes_ > 1 else 0
    pd.DataFrame(exp.as_list(label=label_index), columns=["feature", "weight"]).to_csv(
        os.path.join(FIG_DIR, "lime_example_classification.csv"), index=False
    )
else:
    print("[XAI] Skipping LIME: only one class present in y_cls.")

print("\n✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
