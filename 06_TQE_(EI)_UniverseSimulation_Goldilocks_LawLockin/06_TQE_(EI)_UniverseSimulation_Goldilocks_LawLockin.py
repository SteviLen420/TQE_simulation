# ===========================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Vacuum fluctuation → Superposition → Collapse → Expansion → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Full Monte Carlo pipeline starting from many-universe code
# Focus: Stable, law-locked universes via Energy–Information dynamics
# Includes: MC, law_lock_in, averaged c(t), CSV/PNG saves, summary.json, SHAP/LIME
# ===========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ---- Minimal auto-install (Colab) ----
import sys, subprocess, warnings
warnings.filterwarnings("ignore")

def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["qutip", "pandas", "scikit-learn", "shap", "lime"]:
    _ensure(pkg)

# ---- Imports ----
import os, time, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import qutip as qt
import shap
from lime.lime_tabular import LimeTabularExplainer

# ---- Directories ----
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_UNIVERSE_SIMULATION"
run_id = time.strftime("TQE_(E,I)_UNIVERSE_SIMULATION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p):
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()

# ===== Master flags (tune as you like) =====
PLOT_AVG_LOCKIN  = False   # average lock-in c(t) plot toggle
PLOT_LOCKIN_HIST = False   # histogram of lock-in epochs plot toggle
RUN_XAI          = True    # SHAP + LIME
RUN_SEED_SEARCH  = False   # heavy; enable when needed

# ======================================================
# MASTER SIMULATION CONTROLS
# ======================================================

TIME_STEPS          = 5000   # <--- Main control knob: adjust this, and it propagates everywhere
NUM_UNIVERSES       = 1000   # Number of universes for Monte Carlo run (set to 1 for single-universe mode)
LOCKIN_EPOCHS       = TIME_STEPS   # Epochs used in law lock-in simulation
EXPANSION_EPOCHS    = TIME_STEPS  # Expansion dynamics length, tied to TIME_STEPS
BEST_STEPS          = TIME_STEPS  # Steps for "best-universe" entropy deep dive
BEST_NUM_REGIONS    = 10     # Number of spatial regions in the entropy simulation
BEST_NUM_STATES     = 500    # Number of microstates per region
STABILITY_THRESHOLD = 3.5    # Entropy threshold used to define stability

# ===== Goldilocks window base params =====
E_C   = 2.0
SIGMA = 0.5
ALPHA = 0.8

# ======================================================
# 1) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
Nlev = 12
a = qt.destroy(Nlev)

# Perturbed Hamiltonian with small random noise
H0 = a.dag()*a + 0.05*(np.random.randn()*a + np.random.randn()*a.dag())

# Initial state: random superposition
psi0 = qt.rand_ket(Nlev)
rho0 = psi0 * psi0.dag()

# Coarse time evolution windows across a longer timeline
tlist = np.linspace(0, 10, 200)
gammas = 0.02 + 0.01*np.sin(0.5*tlist) + 0.005*np.random.randn(len(tlist))

states = []
for g in gammas:
    res = qt.mesolve(H0, rho0, np.linspace(0,0.5,5), [np.sqrt(abs(g))*a], [])
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
# 2) t = 0 : Collapse (E·I coupling + Goldilocks factor)
# ======================================================

# KL divergence helper (safe)
def KL(p, q, eps=1e-12):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

# Information parameter I (KL × Shannon, squashed to [0,1])
def info_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12
    KL_val = np.sum(p1 * np.log((p1 + eps)/(p2 + eps)))
    I_kl   = KL_val / (1.0 + KL_val)
    H      = -np.sum(p1 * np.log(p1 + eps))
    I_sh   = H / np.log(len(p1))
    I_raw  = I_kl * I_sh
    return I_raw / (1.0 + I_raw)

# Energy sampling
def sample_energy(mu=2.5, sigma=0.8):
    return float(np.random.lognormal(mean=mu, sigma=sigma))

# Goldilocks modulation f(E,I)
def f_EI(E, I, E_c=E_C, sigma=SIGMA, alpha=ALPHA):
    return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

# Draw one (E, I) pair for the collapse demo
E0 = sample_energy()
I0 = info_param()
f0 = f_EI(E0, I0)
X0 = E0 * I0 * f0

collapse_t = np.linspace(-0.2, 0.2, 200)
X_series = X0 + 0.5 * np.random.randn(len(collapse_t))               # pre-collapse fluctuation
X_series[collapse_t >= 0] = X0 + 0.05*np.random.randn((collapse_t >= 0).sum())  # post-collapse calm

plt.figure()
plt.plot(collapse_t, X_series, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X0, color="r", ls="--", label=f"Lock-in X={X0:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E·I + Goldilocks)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E·I·f(E,I)"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

pd.DataFrame({"time": collapse_t, "X_vals": X_series}).to_csv(
    os.path.join(SAVE_DIR, "collapse.csv"), index=False
)

# ======================================================
# 3) Law lock-in model: c(t) stabilization with calmness check
# ======================================================
def law_lock_in(E, I, n_epoch=LOCKIN_EPOCHS):
    """
    Simulates lock-in of a 'law' proxy (e.g., c) driven by E,I via f(E,I).
    Lock when relative step change < 1e-3 for 5 consecutive epochs.
    Returns: (locked_at_epoch or -1, history_list)
    """
    f = f_EI(E, I)
    if f < 0.1:   # too far from Goldilocks → no lock-in
        return -1, []

    c_val = np.random.normal(3e8, 1e7)  # initial c
    calm = 0
    locked_at = None
    history = []

    for n in range(n_epoch):
        prev = c_val
        # noise scale decreases when E*I is near ~5 and increases otherwise
        noise = 1e6 * (1 + abs(E*I - 5)/10) * np.random.uniform(0.8, 1.2)
        c_val += np.random.normal(0, noise)
        history.append(c_val)

        delta = abs(c_val - prev) / max(abs(prev), 1e-9)
        if delta < 1e-3:
            calm += 1
            if calm >= 5 and locked_at is None:
                locked_at = n
        else:
            calm = 0

    return locked_at if locked_at is not None else -1, history

# Stability predicate used in MC (amplitude calmness gated by f(E,I))
def is_stable(E, I, n_epoch=200):
    f = f_EI(E, I)
    if f < 0.2:
        return 0
    A, calm = 20.0, 0
    for _ in range(n_epoch):
        A_prev = A
        A = A*1.02 + np.random.normal(0, 2.0)
        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        calm = calm + 1 if delta < 0.05 else 0
        if calm >= 5:
            return 1
    return 0

# ======================================================
# 4) Monte Carlo: Stability + Law lock-in for many universes
# ======================================================
def sample_I(dim=8):
    return info_param(dim=dim)

E_vals, I_vals, f_vals, X_vals = [], [], [], []
stables, law_epochs, final_cs, all_histories = [], [], [], []

for _ in range(NUM_UNIVERSES):
    Ei = sample_energy()
    Ii = sample_I(dim=8)
    fi = f_EI(Ei, Ii)
    Xi = Ei * Ii * fi

    E_vals.append(Ei)
    I_vals.append(Ii)
    f_vals.append(fi)
    X_vals.append(Xi)

    s = is_stable(Ei, Ii)
    stables.append(s)

    if s == 1:
        lock_epoch, c_hist = law_lock_in(Ei, Ii, n_epoch=LOCKIN_EPOCHS)
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
mean_epoch   = float(np.mean(valid_epochs))   if len(valid_epochs) > 0 else None

print(f"\n🔒 Universes with lock-in: {len(valid_epochs)} / {NUM_UNIVERSES}")

# ======================================================
# 5) Master DataFrame and saves
# ======================================================
df = pd.DataFrame({
    "E": E_vals,
    "I": I_vals,
    "fEI": f_vals,
    "X": X_vals,
    "stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs,
})
df.to_csv(os.path.join(SAVE_DIR, "tqe_runs.csv"), index=False)

# Diagnostics
stable_total = int(np.sum(stables))
valid_lockins = int(np.sum([e >= 0 for e in law_epochs]))
valid_lockins_among_stable = int(np.sum([e >= 0 for e, s in zip(law_epochs, stables) if s == 1]))

print("\n[DIAG] Stability vs Law lock-in")
print(f"Stable universes: {stable_total}/{NUM_UNIVERSES} ({100*stable_total/NUM_UNIVERSES:.1f}%)")
print(f"Lock-ins (any):   {valid_lockins}/{NUM_UNIVERSES} ({100*valid_lockins/NUM_UNIVERSES:.1f}%)")
if stable_total > 0:
    print(f"Lock-ins among stable: {valid_lockins_among_stable}/{stable_total} "
          f"({100*valid_lockins_among_stable/stable_total:.1f}%)")

# ======================================================
# 6) Stability summary — with lock-in split (E,I case)
# ======================================================
stable_total   = int(np.sum(stables))
unstable_total = int(NUM_UNIVERSES - stable_total)

# universes with lock-in (law_epochs >= 0)
locked_total = int(np.sum([e >= 0 for e in law_epochs]))
stable_no_lock = max(stable_total - locked_total, 0)

# percentages
pct_stable_no_lock = 100.0 * stable_no_lock / NUM_UNIVERSES
pct_locked         = 100.0 * locked_total / NUM_UNIVERSES
pct_unstable       = 100.0 * unstable_total / NUM_UNIVERSES

print("\n🌌 Universe Stability Summary (E,I)")
print(f"Total universes simulated: {NUM_UNIVERSES}")
print(f"Stable universes (lock-in):     {locked_total} ({pct_locked:.2f}%)")
print(f"Stable universes (no lock-in):  {stable_no_lock} ({pct_stable_no_lock:.2f}%)")
print(f"Unstable universes:             {unstable_total} ({pct_unstable:.2f}%)")

# --- plot stacked bar ---
plt.figure()

# x positions: 0 = Stable (stacked), 1 = Unstable
plt.bar(0, stable_no_lock, color="#1f77b4",
        label=f"Stable (no lock-in) [{stable_no_lock}, {pct_stable_no_lock:.1f}%]")
plt.bar(0, locked_total, bottom=stable_no_lock, color="#9467bd",
        label=f"Stable (lock-in) [{locked_total}, {pct_locked:.1f}%]")
plt.bar(1, unstable_total, color="#d62728",
        label=f"Unstable [{unstable_total}, {pct_unstable:.1f}%]")

plt.xticks([0, 1], ["Stable (split by lock-in)", "Unstable"])
plt.ylabel("Number of Universes")
plt.title("Universe Stability Distribution (E,I) — with Lock-in")
plt.legend(loc="upper right", frameon=False)
plt.tight_layout()
savefig(os.path.join(FIG_DIR, "stability_summary_with_lockin.png"))

# ======================================================
# 7) Average law lock-in dynamics across all universes
# ======================================================
if all_histories:
    min_len = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]
    avg_c = np.mean(truncated, axis=0)
    std_c = np.std(truncated, axis=0)

    pd.DataFrame({
        "epoch": np.arange(min_len),
        "avg_c": avg_c,
        "std_c": std_c
    }).to_csv(os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False)

    if PLOT_AVG_LOCKIN and (median_epoch is not None):
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c,
                         alpha=0.3, color="blue", label="±1σ")
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (Monte Carlo)")
        plt.xlabel("epoch"); plt.ylabel("c value (m/s)"); plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 8) t > 0 : Expansion dynamics (demo; uses single E0,I0 and median lock)
# ======================================================
def evolve(E, I, n_epoch=EXPANSION_EPOCHS):
    A_series, I_series = [], []
    A, orient = 20.0, I
    for _ in range(n_epoch):
        A = A * 1.005 + np.random.normal(0, 1.0)
        noise = 0.25 * (1 + 1.5 * abs(orient - 0.5))
        orient += (0.5 - orient) * 0.35 + np.random.normal(0, noise)
        orient = np.clip(orient, 0, 1)
        A_series.append(A); I_series.append(orient)
    return A_series, I_series

A_series, I_series = evolve(E0, I0, n_epoch=EXPANSION_EPOCHS)
plt.figure()
plt.plot(A_series, label="Amplitude A")
plt.plot(I_series, label="Orientation I")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")
if median_epoch is not None:
    plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {int(median_epoch)}")
    title_suffix = ""
else:
    title_suffix = " (no lock-in observed)"
plt.title("t > 0 : Expansion dynamics" + title_suffix)
plt.xlabel("epoch"); plt.ylabel("Parameters"); plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# Save expansion CSV
pd.DataFrame({
    "epoch": np.arange(len(A_series)),
    "Amplitude_A": A_series,
    "Orientation_I": I_series
}).to_csv(os.path.join(SAVE_DIR, "expansion.csv"), index=False)

# ======================================================
# 9) Histogram of lock-in epochs (CSV always, PNG optional)
# ======================================================
pd.DataFrame({"lock_epoch": valid_epochs}).to_csv(
    os.path.join(SAVE_DIR, "law_lockin_epochs.csv"), index=False
)

if PLOT_LOCKIN_HIST and len(valid_epochs) > 0:
    plt.figure()
    plt.hist(valid_epochs, bins=50, color="blue", alpha=0.7)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in = {int(median_epoch)}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (Monte Carlo)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# 10) Save additional CSVs and summary.json
# ======================================================
# Stability outcomes CSV (compact)
pd.DataFrame({
    "E": E_vals,
    "I": I_vals,
    "X": X_vals,
    "Stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs
}).to_csv(os.path.join(SAVE_DIR, "stability.csv"), index=False)

summary = {
    "simulation": {
        "total_universes": NUM_UNIVERSES,
        "stable_fraction": float(np.mean(stables)),
        "unstable_fraction": 1.0 - float(np.mean(stables))
    },
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "E0": float(E0),
        "I0": float(I0),
        "f0": float(f0),
        "mean_X": float(np.mean(X_vals)),
        "std_X": float(np.std(X_vals))
    },
    "law_lockin": {
        "mean_lock_epoch": float(np.mean(valid_epochs)) if len(valid_epochs) > 0 else None,
        "median_lock_epoch": float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None,
        "locked_fraction": float(np.mean([1 if e >= 0 else 0 for e in law_epochs])),
        "mean_final_c": float(np.nanmean(final_cs)) if len(final_cs) > 0 else None,
        "std_final_c": float(np.nanstd(final_cs)) if len(final_cs) > 0 else None
    },
    "diag": {
        "stable_total": stable_total,
        "valid_lockins": valid_lockins,
        "valid_lockins_among_stable": valid_lockins_among_stable
    }
}
with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# ======================================================
# 10b) "Best universe" deep-dive — entropy plot like the screenshot
#       (pick earliest-locking stable universe, then simulate regions)
# ======================================================

# ----- Scoring rule -----
# among stable universes with lock-in (lock_epoch >= 0), pick the one with
# the *earliest* lock-in. if none locked, fall back to the highest f(E,I).
locked_idxs = [i for i, e in enumerate(law_epochs) if e >= 0 and stables[i] == 1]

if len(locked_idxs) > 0:
    # earliest lock-in epoch
    best_idx = locked_idxs[int(np.argmin([law_epochs[i] for i in locked_idxs]))]
    reason = f"earliest lock-in (epoch={law_epochs[best_idx]})"
else:
    # fallback: pick universe with the largest Goldilocks modulation f(E,I)
    best_idx = int(np.argmax(f_vals))
    reason = "no lock-ins → picked max f(E,I)"

E_best = E_vals[best_idx]
I_best = I_vals[best_idx]
print(f"[BEST] Universe index={best_idx} chosen by {reason}; E*={E_best:.3f}, I*={I_best:.3f}")

# ----- Single-universe entropy simulator (same style as your screenshot) -----

def simulate_entropy_universe(E, I, steps=BEST_STEPS,
                              num_regions=BEST_NUM_REGIONS, num_states=BEST_NUM_STATES):
    """
    Runs a single-universe entropy evolution with f(E,I) modulation.
    Returns: (region_entropies_list, global_entropy_list, lock_in_step)
    - region_entropies_list: list over time, each item is length `num_regions`
    - global_entropy_list: list over time (scalar entropy per step)
    - lock_in_step: first step where calmness criterion is satisfied (or None)
    """
    def f_EI_local(E, I, E_c=E_C, sigma=SIGMA, alpha=ALPHA):
        return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

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
        # cooling schedule for noise
        noise_scale = max(0.02, 1.0 - step / steps)

        # amplitude + orientation drift
        if step > 0:
            A = A * 1.01 + np.random.normal(0, 0.02)
            orient += (0.5 - orient) * 0.10 + np.random.normal(0, 0.02)
            orient = np.clip(orient, 0, 1)

        # energy random walk + recompute f
        E_run += np.random.normal(0, 0.05)
        f_step_base = f_EI_local(E_run, I)

        # update regions with noise and occasional shocks
        for r in range(num_regions):
            noise = np.random.normal(0, noise_scale * 5.0, num_states)
            if np.random.rand() < 0.05:
                noise += np.random.normal(0, 8.0, num_states)
            f_step = f_step_base * (1 + np.random.normal(0, 0.1))
            states[r] += f_step * noise
            states[r] = np.clip(states[r], 0, 1)

        # entropies
        region_entropies.append([entropy(states[r]) for r in range(num_regions)])
        global_entropy.append(entropy(states.flatten()))

        # lock-in detection: calm relative change of global entropy
        if step > 0:
            prev = global_entropy[-2]
            cur  = global_entropy[-1]
            delta = abs(cur - prev) / max(prev, 1e-9)
            if delta < 0.001:
                consecutive_calm += 1
                if consecutive_calm >= 10 and lock_in_step is None:
                    lock_in_step = step
            else:
                consecutive_calm = 0

    return region_entropies, global_entropy, lock_in_step

# ----- Run the deep-dive sim on the chosen (E*, I*) -----
best_region_entropies, best_global_entropy, best_lock = simulate_entropy_universe(E_best, I_best)

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
for r in range(min(BEST_NUM_REGIONS, best_re_mat.shape[1])):
    plt.plot(time_axis, best_re_mat[:, r], lw=1, label=f"Region {r} entropy")

# global + thresholds
plt.plot(time_axis, best_global_entropy, color="black", linewidth=2, label="Global entropy")
plt.axhline(y=STABILITY_THRESHOLD, color="red", linestyle="--", label="Stability threshold")

# lock-in indicator
if best_lock is not None:
    plt.axvline(x=best_lock, color="purple", linestyle="--", linewidth=2,
                label=f"Lock-in step = {best_lock}")

plt.title("Best-universe entropy evolution (chosen from MC)")
plt.xlabel("Time step"); plt.ylabel("Entropy"); plt.legend(ncol=2)
plt.grid(True, alpha=0.3)
savefig(os.path.join(FIG_DIR, "best_universe_entropy_evolution.png"))

# ======================================================
# 11) XAI (SHAP + LIME) — classification on stability, regression on lock_epoch
# ======================================================
if RUN_XAI:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import r2_score, accuracy_score

    # Features and targets
    X_feat = df[["E", "I", "X"]].copy()
    y_cls  = df["stable"].astype(int).values

    reg_mask = df["lock_epoch"] >= 0
    X_reg = X_feat[reg_mask]
    y_reg = df.loc[reg_mask, "lock_epoch"].values

    # Train/Test split
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
        X_feat, y_cls, test_size=0.25, random_state=42, stratify=y_cls
    )
    have_reg = len(X_reg) >= 30
    if have_reg:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
            X_reg, y_reg, test_size=0.25, random_state=42
        )

    # Train models
    rf_cls = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    rf_cls.fit(Xtr_c, ytr_c)
    cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
    print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

    if have_reg:
        rf_reg = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        rf_reg.fit(Xtr_r, ytr_r)
        reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
        print(f"[XAI] Regression R^2 (lock_epoch): {reg_r2:.3f}")
    else:
        rf_reg, reg_r2 = None, None
        print("[XAI] Not enough locked samples for regression (need ~30+).")

    # SHAP — classification
    X_plot = Xte_c.copy()
    try:
        expl_cls = shap.TreeExplainer(
            rf_cls, feature_perturbation="interventional", model_output="raw"
        )
        sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
    except Exception:
        expl_cls = shap.Explainer(rf_cls, Xtr_c)
        sv_cls = expl_cls(X_plot).values

    if isinstance(sv_cls, list):
        sv_cls = sv_cls[1]  # positive class
    sv_cls = np.asarray(sv_cls)
    # Normalize possible shapes
    if sv_cls.ndim == 3 and sv_cls.shape[0] == X_plot.shape[0]:
        sv_cls = sv_cls[:, :, 1]
    elif sv_cls.ndim == 3 and sv_cls.shape[-1] == X_plot.shape[1]:
        sv_cls = sv_cls[1, :, :]
    assert sv_cls.shape == X_plot.shape, f"SHAP shape {sv_cls.shape} != data shape {X_plot.shape}"

    plt.figure()
    shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
    plt.title("SHAP summary – classification (stable)")
    plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"), dpi=220, bbox_inches="tight")
    plt.close()

    # SHAP — regression (if trained)
    if rf_reg is not None:
        X_plot_r = Xte_r.copy()
        try:
            expl_reg = shap.TreeExplainer(
                rf_reg, feature_perturbation="interventional", model_output="raw"
            )
            sv_reg = expl_reg.shap_values(X_plot_r, check_additivity=False)
        except Exception:
            expl_reg = shap.Explainer(rf_reg, Xtr_r)
            sv_reg = expl_reg(X_plot_r).values

        sv_reg = np.asarray(sv_reg)
        if sv_reg.ndim == 3 and sv_reg.shape[0] == X_plot_r.shape[0]:
            sv_reg = sv_reg[:, :, 0]
        elif sv_reg.ndim == 3 and sv_reg.shape[-1] == X_plot_r.shape[1]:
            sv_reg = sv_reg[0, :, :]
        assert sv_reg.shape == X_plot_r.shape, f"SHAP shape {sv_reg.shape} != data shape {X_plot_r.shape}"

        plt.figure()
        shap.summary_plot(sv_reg, X_plot_r.values, feature_names=X_plot_r.columns.tolist(), show=False)
        plt.title("SHAP summary – regression (lock_epoch)")
        plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_at.png"), dpi=220, bbox_inches="tight")
        plt.close()

        # CSV exports
        pd.DataFrame(sv_reg, columns=X_plot_r.columns).to_csv(
            os.path.join(FIG_DIR, "shap_values_regression_lock_at.csv"), index=False
        )
        reg_importance = pd.Series(np.mean(np.abs(sv_reg), axis=0), index=X_plot_r.columns)\
                         .sort_values(ascending=False)
        reg_importance.to_csv(
            os.path.join(FIG_DIR, "shap_feature_importance_regression_lock_at.csv"),
            header=["mean_|shap|"]
        )
    else:
        print("[XAI] Skipping SHAP regression CSV export (not enough lock-in universes).")

    # SHAP classification CSVs
    pd.DataFrame(np.asarray(sv_cls), columns=X_plot.columns).to_csv(
        os.path.join(FIG_DIR, "shap_values_classification.csv"), index=False
    )
    cls_importance = pd.Series(np.mean(np.abs(sv_cls), axis=0), index=X_plot.columns)\
                     .sort_values(ascending=False)
    cls_importance.to_csv(
        os.path.join(FIG_DIR, "shap_feature_importance_classification.csv"),
        header=["mean_|shap|"]
    )

    # LIME — quick local explanation example
    lime_explainer = LimeTabularExplainer(
        training_data=X_feat.values,
        feature_names=X_feat.columns.tolist(),
        discretize_continuous=True,
        mode='classification'
    )
    exp = lime_explainer.explain_instance(Xte_c.iloc[0].values, rf_cls.predict_proba, num_features=5)
    lime_list = exp.as_list(label=1)
    pd.DataFrame(lime_list, columns=["feature", "weight"]).to_csv(
        os.path.join(FIG_DIR, "lime_example_classification.csv"), index=False
    )

# ======================================================
# 12) EXTRA: Seed search — Top-5 seeds (stability ratio)
# ======================================================
if RUN_SEED_SEARCH:
    NUM_SEEDS = 100
    UNIVERSES_PER_SEED = 1000

    def _sample_energy_lognormal_rng(rng, mu=2.5, sigma=0.8):
        return float(rng.lognormal(mean=mu, sigma=sigma))

    def _sample_information_param_KL_only(dim=8):
        psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
        p1 = np.abs(psi1.full().flatten())**2
        p2 = np.abs(psi2.full().flatten())**2
        p1 /= p1.sum(); p2 /= p2.sum()
        eps = 1e-12
        KLv = np.sum(p1 * np.log((p1 + eps) / (p2 + eps)))
        return KLv / (1.0 + KLv)

    seed_scores = []
    for s in range(NUM_SEEDS):
        rng_local = np.random.default_rng(seed=s)
        np.random.seed(s)
        stable_flags = []
        for _ in range(UNIVERSES_PER_SEED):
            E = _sample_energy_lognormal_rng(rng_local)
            I = _sample_information_param_KL_only(dim=8)
            stable_flags.append(is_stable(E, I))
        ratio = float(np.mean(stable_flags))
        seed_scores.append({"seed": s, "stable_ratio": ratio})

    seed_scores_sorted = sorted(seed_scores, key=lambda r: r["stable_ratio"], reverse=True)

    print("\n🏆 Top-5 seeds by stability ratio")
    for r in seed_scores_sorted[:5]:
        print(f"Seed {r['seed']:3d} → stability={r['stable_ratio']:.3f}")

    top_csv_path = os.path.join(SAVE_DIR, "seed_search_top.csv")
    pd.DataFrame(seed_scores_sorted).to_csv(top_csv_path, index=False)
    print("Seed search table saved to:", top_csv_path)

    # Append to summary.json
    summary["seed_search"] = {
        "num_seeds": NUM_SEEDS,
        "universes_per_seed": UNIVERSES_PER_SEED,
        "top5": seed_scores_sorted[:5],
        "csv_path": top_csv_path,
    }
    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

print("\n✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
