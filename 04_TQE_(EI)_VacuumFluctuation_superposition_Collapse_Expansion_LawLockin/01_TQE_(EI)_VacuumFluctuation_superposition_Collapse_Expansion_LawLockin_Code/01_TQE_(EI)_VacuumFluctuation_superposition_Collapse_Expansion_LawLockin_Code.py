# ===========================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Vacuum fluctuation → superposition → Collapse → Expansion → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Full model simulation of energy-information (E,I) dynamics
# Improvements: MASTER_CTRL, reproducible seeds, eps sweep, XAI guards,
#               unified summary, robust Drive copy
# ===========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import sys, subprocess, warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.unicode_minus": False
})

# ensure core deps (only if needed)
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
for pkg in ["qutip", "pandas", "scikit-learn", "shap", "lime"]:
    _ensure(pkg)

# -- (optional) Pin versions for publication --
PINNED = {
    "numpy": "1.26.4",
    "scipy": "1.11.4",
    "qutip": "5.0.3",
    "scikit-learn": "1.3.2",
    "shap": "0.43.0",
    "lime": "0.2.0.1"
}
def _pin(pkg, ver):
    try:
        import importlib
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"{pkg}=={ver}", "-q"])

for _p, _v in PINNED.items():
    _pin(_p, _v)

import qutip as qt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# ======================================================
# 1) MASTER CONTROLLER – central parameters
# ======================================================
DEMO_MODE = False  # Set to True for fast run
if DEMO_MODE:
    MASTER_CTRL.update({"N_universes": 800, "N_epoch": 200, "expansion_epochs": 200})

MASTER_CTRL = {
    # --- Simulation core ---
    "N_universes": 5000,
    "N_epoch": 500,
    "expansion_epochs": 500,
    "rel_eps": 0.05,
    "seed": None,   # master seed (generated if None)

    # --- Law lock-in ---
    "lock_consecutive": 5,
    "regression_min": 30,

    # --- Train/test split ---
    "test_size": 0.25,
    "rf_n_estimators": 400,

    # --- XAI controls ---
    "enable_SHAP": True,
    "enable_LIME": True,

    # --- Seed search ---
    "enable_seed_search": False,
    "seed_search_num": 100,
    "seed_search_universes": 500,

    # --- Outputs ---
    "save_drive_copy": True,
    "save_figs": True,
    "save_json": True,

    # --- Plot controls ---
    "PLOT_AVG_LOCKIN": True,
    "PLOT_LOCKIN_HIST": True,

    "PLOT_STABILITY_BASIC": False,
}

# --- Energy distribution & Goldilocks (linear scale) ---
E_LOG_MU    = 2.5
E_LOG_SIGMA = 0.8
E_CENTER    = float(np.exp(E_LOG_MU))   # ~12.18 (median of lognormal)
E_WIDTH     = 6.0                       # try 6–8 for a reasonable window
ALPHA_I     = 0.8

# --- Master RNG + sync qutip/np.random (for reproducibility) ---
if MASTER_CTRL["seed"] is None:
    MASTER_CTRL["seed"] = int(np.random.SeedSequence().generate_state(1)[0])
master_seed = MASTER_CTRL["seed"]
rng = np.random.default_rng(master_seed)
np.random.seed(master_seed)  # sync for qutip.rand_ket()
print(f"🎲 Using master seed: {master_seed}")
summary = {"params": MASTER_CTRL, "master_seed": master_seed}

# --- Directories ---
run_id = time.strftime("TQE_(E,I)_law_lockin_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("/content/drive/MyDrive/TQE_(E,I)_law_lockin", run_id)
FIG_DIR = os.path.join(SAVE_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p):
    if not MASTER_CTRL["save_figs"]: 
        return
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()

def save_json(path, obj):
    if not MASTER_CTRL["save_json"]:
        return
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ======================================================
# 2) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
Nlev = 12
a = qt.destroy(Nlev)

# perturbed Hamiltonian with small random noise
H0 = a.dag()*a + 0.05*(rng.normal()*a + rng.normal()*a.dag())

# initial state: random superposition (not just vacuum)
psi0 = qt.rand_ket(Nlev)
rho = psi0 * psi0.dag()

# time scale
tlist = np.linspace(0, 10, 200)

# time-dependent gamma (fluctuating environment)
gammas = 0.02 + 0.01*np.sin(0.5*tlist) + 0.005*rng.normal(size=len(tlist))

states = []
for g in gammas:
    # actual time evolution in a small window
    res = qt.mesolve(H0, rho, np.linspace(0, 0.5, 5), [np.sqrt(abs(g))*a], [])
    states.append(res.states[-1])

# purity and entropy
def purity(r): 
    return float((r*r).tr().real) if qt.isoper(r) else float((r*r.dag()).tr().real)

S = np.array([qt.entropy_vn(r) for r in states])
P = np.array([purity(r) for r in states])

# plot
plt.plot(tlist, S, label="Entropy")
plt.plot(tlist, P, label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend()
savefig(os.path.join(FIG_DIR, "superposition.png"))

# Save superposition results to CSV
superposition_df = pd.DataFrame({
    "time": tlist,
    "Entropy": S,
    "Purity": P
})
superposition_df.to_csv(os.path.join(SAVE_DIR, "superposition.csv"), index=False)

# ======================================================
# 3) t = 0 : Collapse (E·I coupling + Goldilocks factor)
# ======================================================

# Kullback–Leibler divergence
def KL(p, q, eps=1e-12):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p /= p.sum(); q /= q.sum()
    return np.sum(p * np.log(p / q))

# Goldilocks modulation factor
def f_EI(E, I, E_c=E_CENTER, sigma=E_WIDTH, alpha=ALPHA_I):
    """
    Gaussian Goldilocks window around E_c (LINEAR E), with information coupling.
    """
    return np.exp(-((E - E_c) ** 2) / (2.0 * (sigma ** 2))) * (1.0 + alpha * I)

# Generate two random quantum states and compute Information I (KL × Shannon)
psi1, psi2 = qt.rand_ket(8), qt.rand_ket(8)
p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
p1 /= p1.sum(); p2 /= p2.sum()
eps = 1e-12

# KL divergence normalized
# Compute KL divergence between the two amplitude distributions.
# (Using the helper makes the intent explicit and avoids duplicating math.)
KL_val = KL(p1, p2, eps=eps)  # KL-divergencia a helperrel

# Convert raw KL to a bounded information factor in [0,1] for stability.
I_kl = KL_val / (1.0 + KL_val)  # normalizált KL → [0,1]

# Normalized Shannon entropy of psi1
H = -np.sum(p1 * np.log(p1 + eps))
I_shannon = H / np.log(len(p1))

# Multiplicative fusion (squash back to [0,1])
I_raw = I_kl * I_shannon
I = I_raw / (1.0 + I_raw)

# Energy fluctuation (lognormal from master rng)
E = float(rng.lognormal(mean=2.5, sigma=0.8))

# Apply Goldilocks filter
f = f_EI(E, I)

# Coupled parameter
X = E * I * f

# Collapse dynamics (before t=0 fluctuation, after lock-in)
collapse_t = np.linspace(-0.2, 0.2, 200)
collapse_X_curve = X + 0.5 * rng.normal(size=len(collapse_t))
collapse_X_curve[collapse_t >= 0] = X + 0.05 * rng.normal(size=np.sum(collapse_t >= 0))

# plot
plt.plot(collapse_t, collapse_X_curve, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X, color="r", ls="--", label=f"Lock-in X={X:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E·I coupling + Goldilocks)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E·I·f")
plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

# Save collapse results to CSV
collapse_df = pd.DataFrame({
    "time": collapse_t,
    "X_curve": collapse_X_curve
})
collapse_df.to_csv(os.path.join(SAVE_DIR, "collapse.csv"), index=False)
    
# ======================================================
# 5) Monte Carlo Simulation: Stability + Law lock-in for many universes
# ======================================================

def sample_information_param_KLxShannon(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1 = np.abs(psi1.full().flatten())**2
    p2 = np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12

    KL_val = KL(p1, p2, eps=eps)
    I_kl = KL_val / (1.0 + KL_val)

    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))

    I_raw = I_kl * I_shannon
    return I_raw / (1.0 + I_raw)


# --- Stability check uses ONLY the provided rng ---
def is_stable(E, I, n_epoch=None, rel_eps=None, lock_consec=None, rng=None):
    """
    Returns 1 if the universe stabilizes; 0 otherwise.
    All randomness (noise) must come from the given rng to keep per-universe determinism.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_epoch is None: n_epoch = MASTER_CTRL["N_epoch"]
    if rel_eps is None: rel_eps = MASTER_CTRL["rel_eps"]
    if lock_consec is None: lock_consec = MASTER_CTRL["lock_consecutive"]

    f = f_EI(E, I)
    if f < 0.2:    # too far from Goldilocks -> unstable immediately
        return 0

    A, calm = 20.0, 0
    for _ in range(n_epoch):
        A_prev = A
        A = A * 1.02 + rng.normal(0, 2.0)       # <-- use rng, not np.random
        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        calm = calm + 1 if delta < rel_eps else 0
        if calm >= lock_consec:
            return 1
    return 0


# --- Law lock-in uses ONLY the provided rng ---
def law_lock_in(E, I, n_epoch=500, rng=None):
    """
    Simulates lock-in of a 'law' proxy (e.g., c).
    Lock when relative step change stays below 1e-3 for 5 consecutive epochs.
    All randomness uses rng to ensure per-universe reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    f = f_EI(E, I)
    if f < 0.1:
        return -1, []   # too far from Goldilocks -> no lock-in

    c_val = rng.normal(3e8, 1e7)  # initial value for 'c'
    calm, locked_at = 0, None
    history = []

    for n in range(n_epoch):
        prev = c_val
        # noise intensity modulated by E*I distance from ~5
        noise = 1e6 * (1 + abs(E * I - 5) / 10.0) * rng.uniform(0.8, 1.2)
        c_val += rng.normal(0, noise)
        history.append(c_val)

        delta = abs(c_val - prev) / max(abs(prev), 1e-9)
        calm = calm + 1 if delta < 1e-3 else 0
        if calm >= 5 and locked_at is None:
            locked_at = n

    return locked_at if locked_at is not None else -1, history


# --------- MAIN MC LOOP (per-universe RNG) ---------
N = MASTER_CTRL["N_universes"]

X_vals, I_vals, E_vals, f_vals = [], [], [], []
stables, law_epochs, final_cs, all_histories = [], [], [], []
universe_seeds = []

for _ in range(N):
    # 1) draw a unique seed from the master rng and keep it
    uni_seed = int(rng.integers(0, 2**32 - 1))
    universe_seeds.append(uni_seed)

    # 2) build a per-universe RNG and sync legacy np.random for qutip
    rng_uni = np.random.default_rng(uni_seed)
    np.random.seed(uni_seed)   # ensures qt.rand_ket() etc. are reproducible

    # 3) sample everything using the per-universe rng
    Ei = float(rng_uni.lognormal(E_LOG_MU, E_LOG_SIGMA))
    Ii = sample_information_param_KLxShannon(dim=8)   # uses qutip + np.random (already seeded)
    fi = f_EI(Ei, Ii)
    Xi = Ei * Ii * fi

    E_vals.append(Ei); I_vals.append(Ii); f_vals.append(fi); X_vals.append(Xi)

    # 4) call the updated functions, passing rng_uni
    stable = is_stable(Ei, Ii, rng=rng_uni)
    stables.append(stable)

    if stable == 1:
        lock_epoch, c_hist = law_lock_in(Ei, Ii, n_epoch=MASTER_CTRL["N_epoch"], rng=rng_uni)
    else:
        lock_epoch, c_hist = -1, []

    law_epochs.append(lock_epoch)
    if c_hist:
        final_cs.append(c_hist[-1])
        all_histories.append(c_hist)
    else:
        final_cs.append(np.nan)

# central statistics
valid_epochs = [e for e in law_epochs if e >= 0]
mean_lock   = float(np.mean(valid_epochs))   if valid_epochs else None
median_epoch = float(np.median(valid_epochs)) if valid_epochs else None

np.random.seed(master_seed)

# ======================================================
# 6) Build master DataFrame and save
# ======================================================

df = pd.DataFrame({
    "universe_id": np.arange(N),
    "seed": universe_seeds,
    "E": E_vals,
    "I": I_vals,
    "fEI": f_vals,
    "X": X_vals,
    "stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs,
    # "seed": universe_seeds  
})

df.to_csv(os.path.join(SAVE_DIR, "tqe_runs.csv"), index=False)
pd.DataFrame({"universe_id": np.arange(N), "seed": universe_seeds}) \
  .to_csv(os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False)

summary.setdefault("seeds", {})
summary["seeds"]["master_seed"] = master_seed
summary["seeds"]["universe_seeds_csv"] = "universe_seeds.csv"

# Friss summary információ
summary["runs"] = {
    "csv_path": os.path.join(SAVE_DIR, "tqe_runs.csv"),
    "total": int(len(df)),
    "stable_count": int(df["stable"].sum()),
    "unstable_count": int(len(df) - df["stable"].sum()),
    "stable_ratio": float(df["stable"].mean()),
    "mean_lock_epoch": float(np.mean([e for e in df["lock_epoch"] if e >= 0])) if any(df["lock_epoch"] >= 0) else None,
    "median_lock_epoch": float(np.median([e for e in df["lock_epoch"] if e >= 0])) if any(df["lock_epoch"] >= 0) else None
}

# ======================================================
# 7) [DIAG] Stability vs Law lock-in (extra check)
# ======================================================
stable_total = int(df["stable"].sum())
valid_lockins = int(np.sum(df["lock_epoch"] >= 0))
valid_lockins_among_stable = int(np.sum((df["lock_epoch"] >= 0) & (df["stable"] == 1)))

print("\n[DIAG] Stability vs Law lock-in")
print(f"Stable universes: {stable_total}/{N} ({100*stable_total/N:.1f}%)")
print(f"Lock-ins (any): {valid_lockins}/{N} ({100*valid_lockins/N:.1f}%)")
if stable_total > 0:
    print(f"Lock-ins among stable: {valid_lockins_among_stable}/{stable_total} "
          f"({100*valid_lockins_among_stable/stable_total:.1f}%)")

# --- Add to summary ---
summary["diagnostics"] = {
    "stable_total": stable_total,
    "valid_lockins": valid_lockins,
    "valid_lockins_among_stable": valid_lockins_among_stable,
    "stable_ratio": float(stable_total / N),
    "lockin_ratio": float(valid_lockins / N),
    "lockin_ratio_among_stable": float(valid_lockins_among_stable / stable_total) if stable_total > 0 else None
}
        
# ======================================================
# 8) Stability summary (counts + percentages)
# ======================================================
total_universes = len(df)
stable_count = int(df["stable"].sum())
unstable_count = total_universes - stable_count

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {total_universes}")
print(f"Stable universes:   {stable_count} ({stable_count/total_universes*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/total_universes*100:.2f}%)")

# --- Save to summary JSON (extend existing summary dict) ---
summary["stability_counts"] = {
    "total_universes": total_universes,
    "stable_universes": stable_count,
    "unstable_universes": unstable_count,
    "stable_percent": float(stable_count/total_universes*100),
    "unstable_percent": float(unstable_count/total_universes*100)
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# ======================================================
# 9) Average law lock-in dynamics across all universes
# ======================================================
if all_histories:
    min_len = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]
    avg_c = np.mean(truncated, axis=0)
    std_c = np.std(truncated, axis=0)

    avg_df = pd.DataFrame({
        "epoch": np.arange(len(avg_c)),
        "avg_c": avg_c,
        "std_c": std_c
    })
    avg_df.to_csv(os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False)

    # Add to summary JSON (in-memory; a file-t később amúgy is írjuk)
    summary["law_lockin_avg"] = {
        "epochs": len(avg_c),
        "mean_final_c": float(avg_c[-1]),
        "std_final_c": float(std_c[-1])
    }

    # (OPTIONAL) Plot only if enabled
    if MASTER_CTRL["PLOT_AVG_LOCKIN"] and (median_epoch is not None):
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(len(avg_c)), avg_c-std_c, avg_c+std_c,
                         alpha=0.3, color="blue", label="±1σ")
        if median_epoch is not None:
            plt.axvline(median_epoch, color="r", ls="--", lw=2,
                        label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (Monte Carlo)")
        plt.xlabel("epoch")
        plt.ylabel("c value (m/s)")
        plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 10) t > 0 : Expansion dynamics (reference universe E,I)
# ======================================================
def evolve(E, I, n_epoch=None):   
    """Simulate expansion dynamics after law lock-in."""
    if n_epoch is None:
        n_epoch = MASTER_CTRL["expansion_epochs"]

    A_series = []
    I_series = []
    A = 20
    orient = I
    for n in range(n_epoch):
        # Amplitude growth with noise
        A = A * 1.005 + rng.normal(0, 1.0)

        # Orientation dynamics with convergence + noise
        noise = 0.25 * (1 + 1.5 * abs(orient - 0.5))
        orient += (0.5 - orient) * 0.35 + rng.normal(0, noise)

        # Clamp orientation between [0,1]
        orient = max(0, min(1, orient))

        A_series.append(A)
        I_series.append(orient)

    return A_series, I_series

# --- Pick reference universe for expansion (any stable one)
if df["stable"].sum() > 0:
    ref_universe = df[df["stable"] == 1].sample(1, random_state=MASTER_CTRL["seed"])
    E_ref, I_ref = ref_universe["E"].values[0], ref_universe["I"].values[0]
else:
    E_ref, I_ref = E, I   # fallback: collapse values

# Run expansion
A_series, I_series = evolve(E_ref, I_ref)

# Save expansion dynamics
expansion_df = pd.DataFrame({
    "epoch": np.arange(len(A_series)),
    "Amplitude_A": A_series,
    "Orientation_I": I_series
})
expansion_df.to_csv(os.path.join(SAVE_DIR, "expansion.csv"), index=False)

# Update summary
summary["expansion"] = {
    "E_ref": float(E_ref),
    "I_ref": float(I_ref),
    "mean_A": float(np.mean(A_series)),
    "mean_I": float(np.mean(I_series))
}

# Plot expansion
plt.figure()
plt.plot(A_series, label="Amplitude A")
plt.plot(I_series, label="Orientation I")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")
if median_epoch is not None:
    plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {median_epoch:.0f}")
plt.title("t > 0 : Expansion dynamics (reference universe)")
plt.xlabel("epoch"); plt.ylabel("Parameters")
plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# ======================================================
# 11) Histogram of lock-in epochs 
# ======================================================

if len(valid_epochs) > 0:
    # Save raw lock-in epochs to CSV
    pd.DataFrame({"lock_epoch": valid_epochs}).to_csv(
        os.path.join(SAVE_DIR, "law_lockin_epochs.csv"), index=False
    )

    # Update summary
    summary["law_lockin_epochs"] = {
        "count": len(valid_epochs),
        "mean": float(np.mean(valid_epochs)),
        "median": float(np.median(valid_epochs)),
        "min": int(np.min(valid_epochs)),
        "max": int(np.max(valid_epochs))
    }

    # (OPTIONAL) Plot only if enabled
    if MASTER_CTRL.get("PLOT_LOCKIN_HIST", False):
        plt.figure()
        bins = min(50, len(valid_epochs))  # adaptive binning
        plt.hist(valid_epochs, bins=bins, color="blue", alpha=0.7)

        if median_epoch is not None:
            plt.axvline(median_epoch, color="r", ls="--", lw=2,
                        label=f"Median lock-in = {median_epoch:.0f}")
            plt.legend()

        plt.title("Distribution of law lock-in epochs (Monte Carlo)")
        plt.xlabel("Epoch of lock-in")
        plt.ylabel("Count")
        savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))
else:
    print("[INFO] No valid lock-in epochs to save or plot.")
    
# ======================================================
# 12) Stability summary (counts + percentages)
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = int(len(df) - stable_count)

# Count lock-in universes (only those with lock_epoch >= 0)
lockin_count = int((df["lock_epoch"] >= 0).sum())

print("\n🌌 Universe Stability Summary (final)")
print(f"Total universes simulated: {len(df)}")
print(f"Stable universes:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")
print(f"Lock-in universes:  {lockin_count} ({lockin_count/len(df)*100:.2f}%)")

# --- Update summary JSON ---
summary["stability_summary"] = {
    "total_universes": int(len(df)),
    "stable_universes": stable_count,
    "unstable_universes": unstable_count,
    "lockin_universes": lockin_count,
    "stable_percent": float(stable_count/len(df)*100),
    "unstable_percent": float(unstable_count/len(df)*100),
    "lockin_percent": float(lockin_count/len(df)*100)
}

save_json(os.path.join(SAVE_DIR,"summary.json"), summary)

# --- Save bar chart ---
plt.figure()
plt.bar(
    ["Stable", "Unstable", "Lock-in"],
    [stable_count, unstable_count, lockin_count],
    color=["green", "red", "blue"]
)
plt.title("Universe Stability Distribution (Final)")
plt.ylabel("Number of Universes")
plt.xlabel("Category")

# Labels with counts + percentages UNDER the bars
labels = [
    f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)",
    f"Lock-in ({lockin_count}, {lockin_count/len(df)*100:.1f}%)"
]
plt.xticks([0, 1, 2], labels)

savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# 13) Save results (JSON + CSV + Figures)
# ======================================================

# Save stability outcomes to CSV (redundant but simplified)
stability_df = pd.DataFrame({
    "X": X_vals,
    "Stable": stables
})
stability_df.to_csv(os.path.join(SAVE_DIR, "stability.csv"), index=False)

# --- Extend summary dict instead of overwriting ---
valid_epochs = [e for e in law_epochs if e >= 0]
summary.update({
    "simulation": {
        "total_universes": N,
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
        "mean_lock_epoch": float(np.mean(valid_epochs)) if valid_epochs else None,
        "median_lock_epoch": float(np.median(valid_epochs)) if valid_epochs else None,
        "locked_fraction": float(len(valid_epochs) / len(law_epochs)) if law_epochs else 0.0,
        "mean_final_c": float(np.nanmean(final_cs)),
        "std_final_c": float(np.nanstd(final_cs))
    }
})

save_json(os.path.join(SAVE_DIR,"summary.json"), summary)

# ======================================================
# 14) XAI (SHAP + LIME) 
# ======================================================

# ---------- Features and targets ----------
X_feat = df[["E", "I", "X"]].copy()
y_cls = df["stable"].astype(int).values
reg_mask = df["lock_epoch"] >= 0
X_reg = X_feat[reg_mask]
y_reg = df.loc[reg_mask, "lock_epoch"].values

# --- Sanity checks ---
assert not np.isnan(X_feat.values).any(), "NaN in X_feat!"
if len(X_reg) > 0:
    assert not np.isnan(X_reg.values).any(), "NaN in X_reg!"

# --------- Stratify guard ---------
vals, cnts = np.unique(y_cls, return_counts=True)
can_stratify = (len(vals) == 2) and (cnts.min() >= 2)
stratify_arg = y_cls if can_stratify else None
if not can_stratify:
    print(f"[WARN] Skipping stratify: class counts = {dict(zip(vals, cnts))}")

# --- Classification split ---
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_feat, y_cls,
    test_size=MASTER_CTRL["test_size"],
    random_state=42,
    stratify=stratify_arg
)

# --- Regression split (only if enough samples) ---
have_reg = len(X_reg) >= MASTER_CTRL["regression_min"]
if have_reg:
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_reg, y_reg,
        test_size=MASTER_CTRL["test_size"],
        random_state=42
    )

# ---------- Train models ----------
rf_cls = RandomForestClassifier(n_estimators=MASTER_CTRL["rf_n_estimators"],
                                random_state=42, n_jobs=-1)
rf_cls.fit(Xtr_c, ytr_c)
cls_acc = accuracy_score(yte_c, rf_cls.predict(Xte_c))
print(f"[XAI] Classification accuracy (stable): {cls_acc:.3f}")

rf_reg, reg_r2 = None, None
if have_reg:
    rf_reg = RandomForestRegressor(n_estimators=MASTER_CTRL["rf_n_estimators"],
                                   random_state=42, n_jobs=-1)
    rf_reg.fit(Xtr_r, ytr_r)
    reg_r2 = r2_score(yte_r, rf_reg.predict(Xte_r))
    print(f"[XAI] Regression R^2 (lock_epoch): {reg_r2:.3f}")
else:
    print("[XAI] Not enough lock-in samples for regression.")

# ---------- SHAP ----------
if MASTER_CTRL["enable_SHAP"]:
    # Classification
    try:
        X_plot = Xte_c.copy()
        try:
            expl_cls = shap.TreeExplainer(rf_cls, feature_perturbation="interventional",
                                          model_output="raw")
            sv_cls = expl_cls.shap_values(X_plot, check_additivity=False)
        except Exception:
            expl_cls = shap.Explainer(rf_cls, Xtr_c)
            sv_cls = expl_cls(X_plot).values

        if isinstance(sv_cls, list) and len(sv_cls) > 1:
            sv_cls = sv_cls[1]  # positive class only
        sv_cls = np.asarray(sv_cls)
        if sv_cls.ndim == 3 and sv_cls.shape[0] == X_plot.shape[0]:
            sv_cls = sv_cls[:, :, 1]
        elif sv_cls.ndim == 3 and sv_cls.shape[-1] == X_plot.shape[1]:
            sv_cls = sv_cls[1, :, :]
        assert sv_cls.shape == X_plot.shape, f"SHAP mismatch {sv_cls.shape} vs {X_plot.shape}"

        plt.figure()
        shap.summary_plot(sv_cls, X_plot.values, feature_names=X_plot.columns.tolist(), show=False)
        savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"))

        pd.DataFrame(sv_cls, columns=X_plot.columns).to_csv(
            os.path.join(SAVE_DIR, "shap_values_classification.csv"), index=False
        )
        cls_importance = pd.Series(np.mean(np.abs(sv_cls), axis=0), index=X_plot.columns) \
                           .sort_values(ascending=False)
        cls_importance.to_csv(
            os.path.join(SAVE_DIR, "shap_feature_importance_classification.csv"),
            header=["mean_|shap|"]
        )
    except Exception as e:
        print(f"[ERR] SHAP classification failed: {e}")

    # Regression 
    if rf_reg is not None:
        try:
            X_plot_r = Xte_r.copy()
            try:
                expl_reg = shap.TreeExplainer(rf_reg, feature_perturbation="interventional",
                                              model_output="raw")
                sv_reg = expl_reg.shap_values(X_plot_r, check_additivity=False)
            except Exception:
                expl_reg = shap.Explainer(rf_reg, Xtr_r)
                sv_reg = expl_reg(X_plot_r).values

            sv_reg = np.asarray(sv_reg)
            if sv_reg.ndim == 3 and sv_reg.shape[0] == X_plot_r.shape[0]:
                sv_reg = sv_reg[:, :, 0]
            elif sv_reg.ndim == 3 and sv_reg.shape[-1] == X_plot_r.shape[1]:
                sv_reg = sv_reg[0, :, :]
            assert sv_reg.shape == X_plot_r.shape, f"SHAP mismatch {sv_reg.shape} vs {X_plot_r.shape}"

            plt.figure()
            shap.summary_plot(sv_reg, X_plot_r.values, feature_names=X_plot_r.columns.tolist(), show=False)
            savefig(os.path.join(FIG_DIR, "shap_summary_reg_lock_epoch.png"))

            pd.DataFrame(sv_reg, columns=X_plot_r.columns).to_csv(
                os.path.join(SAVE_DIR, "shap_values_regression.csv"), index=False
            )
            reg_importance = pd.Series(np.mean(np.abs(sv_reg), axis=0), index=X_plot_r.columns) \
                               .sort_values(ascending=False)
            reg_importance.to_csv(
                os.path.join(SAVE_DIR, "shap_feature_importance_regression.csv"),
                header=["mean_|shap|"]
            )
        except Exception as e:
            print(f"[ERR] SHAP regression failed: {e}")

# ---------- LIME ----------
if MASTER_CTRL["enable_LIME"] and (len(np.unique(y_cls)) > 1):
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
        lime_list = exp.as_list(label=1 if 1 in np.unique(y_cls) else 0)
        pd.DataFrame(lime_list, columns=["feature", "weight"]).to_csv(
            os.path.join(SAVE_DIR, "lime_example_classification.csv"), index=False
        )
    except Exception as e:
        print(f"[ERR] LIME failed: {e}")

print(f"☁️ All XAI results saved to Google Drive: {SAVE_DIR}")
# -- ensure final summary on disk --
save_json(os.path.join(SAVE_DIR,"summary.json"), summary)
print(f"📦 Artifacts saved to: {SAVE_DIR}")
