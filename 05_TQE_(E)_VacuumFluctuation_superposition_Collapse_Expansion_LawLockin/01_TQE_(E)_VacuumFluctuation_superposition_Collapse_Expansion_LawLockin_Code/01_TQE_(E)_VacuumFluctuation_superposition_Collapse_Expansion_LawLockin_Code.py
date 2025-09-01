# ===========================================================================
# Theory of the Question of Existence (TQE)
# Energy-only Simulation — Vacuum fluctuation → Collapse → Expansion → Law lock-in
# ===========================================================================
# Author: Stefan Len
#
# SUMMARY
# Monte Carlo framework modeling cosmogenesis from energy (E) alone, without 
# informational orientation (I=0). Starting from vacuum fluctuations and 
# quantum superposition (t < 0), the system undergoes collapse (t = 0), 
# stabilization, and expansion (t > 0). A Goldilocks window defines critical 
# thresholds where law lock-in emerges. Outputs include entropy/purity traces, 
# collapse dynamics, stability fractions, law lock-in epochs, and expansion 
# trajectories. Reproducible with master + per-universe seeds, results are 
# saved as CSV/JSON/PNG for analysis.
# ===========================================================================


# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os, time, json, numpy as np, matplotlib.pyplot as plt,
import sys, subprocess

# ======================================================
# Auto-install required packages (only if missing)
# ======================================================
def install_if_missing(packages):
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[INSTALL] Missing package: {pkg} → installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# List of required packages
required_packages = ["numpy", "pandas", "matplotlib", "qutip"]

# Install if missing
install_if_missing(required_packages)

# --- Imports after ensuring install ---
import qutip as qt
import pandas as pd

# --- Directories ---
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E)_law_lockin"
run_id = time.strftime("TQE_(E)law_lockin_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p): 
    plt.savefig(p,dpi=150,bbox_inches="tight")
    plt.close()

# ======================================================
# 0) MASTER CONTROLLER – central parameters (E-only)
# ======================================================
MASTER_CTRL = {
    "N_universes": 5000,      # number of universes to simulate
    "N_epoch": 500,           # number of epochs for law lock-in
    "expansion_epochs": 500,  # number of epochs for expansion
    "rel_eps": 0.05,          # stability tolerance
    "lock_consecutive": 5,    # consecutive calm epochs required
    "seed": None,             # master RNG seed

    # Output controls
    "save_drive_copy": True,
    "save_figs": True,
    "save_json": True,

    # Plot controls
    "PLOT_AVG_LOCKIN": True,
    "PLOT_LOCKIN_HIST": True,   
}

# --- Goldilocks window tuned to the E distribution ---
# These are GLOBAL constants, not inside MASTER_CTRL.
E_MU = 2.5           # lognormal parameter (mean in log-space)
E_SIGMA = 0.8        # lognormal sigma
E_CENTER = float(np.exp(E_MU))   # ~12.18, median in linear space
E_WIDTH  = 6.0       # try 6–8 for ~40%+ stability; smaller = stricter

# Alias for readability
NUM_UNIVERSES = MASTER_CTRL["N_universes"]

# Create master RNG
master_rng = np.random.default_rng(MASTER_CTRL["seed"])

# --- Master seed (reproducibility) ---
if MASTER_CTRL["seed"] is None:
    MASTER_CTRL["seed"] = int(np.random.SeedSequence().generate_state(1)[0])
master_seed = MASTER_CTRL["seed"]
master_rng = np.random.default_rng(master_seed)
np.random.seed(master_seed)  # keep legacy np.random in sync for QuTiP
print(f"🎲 Using master seed: {master_seed}")

# ======================================================
# 1) t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
Nlev = 12
a = qt.destroy(Nlev)

H0 = a.dag()*a + 0.05*(np.random.randn()*a + np.random.randn()*a.dag())
psi0 = qt.rand_ket(Nlev)
rho = psi0 * psi0.dag()

tlist = np.linspace(0,10,200)
gammas = 0.02 + 0.01*np.sin(0.5*tlist) + 0.005*np.random.randn(len(tlist))

states = []
for g in gammas:
    res = qt.mesolve(H0, rho, np.linspace(0,0.5,5), [np.sqrt(abs(g))*a], [])
    states.append(res.states[-1])

def purity(r): 
    return float((r*r).tr().real) if qt.isoper(r) else float((r*r.dag()).tr().real)

S = np.array([qt.entropy_vn(r) for r in states])
P = np.array([purity(r) for r in states])

plt.plot(tlist,S,label="Entropy")
plt.plot(tlist,P,label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend(); savefig(os.path.join(FIG_DIR,"superposition.png"))

pd.DataFrame({"time": tlist, "Entropy": S, "Purity": P}).to_csv(
    os.path.join(SAVE_DIR, "superposition.csv"), index=False)

# ======================================================
# 2) t = 0 : Collapse (E only)
# ======================================================
# Use the same lognormal parameters as the global E distribution
E = float(np.random.lognormal(mean=E_MU, sigma=E_SIGMA))
X = E

# Collapse trace around t=0 (pre-lock fluctuation, post-lock calm)
collapse_t = np.linspace(-0.2, 0.2, 200)
X_curve = X + 0.5 * np.random.randn(len(collapse_t))
X_curve[collapse_t >= 0] = X + 0.05 * np.random.randn(np.sum(collapse_t >= 0))

plt.plot(collapse_t, X_curve, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X, color="r", ls="--", label=f"Lock-in X={X:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E only)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

pd.DataFrame({"time": collapse_t, "X_vals": X_curve}).to_csv(
    os.path.join(SAVE_DIR, "collapse.csv"), index=False)
    
# ======================================================
# 3) Stability check (Energy-dependent)
# ======================================================
def is_stable(E, n_epoch=30, E_center=E_CENTER, E_width=E_WIDTH,
              rel_eps=None, lock_consec=None):
    """
    Energy-only stability check controlled by MASTER_CTRL thresholds.
    1) Quick Goldilocks cutoff by |E - E_center|
    2) Internal dynamics with E-dependent noise; stability = consecutive small steps
    """
    if rel_eps is None:
        rel_eps = MASTER_CTRL["rel_eps"]
    if lock_consec is None:
        lock_consec = MASTER_CTRL["lock_consecutive"]

    # 1) Quick cutoff: too far from Goldilocks → unstable
    if abs(E - E_center) > 3 * E_width:
        return 0

    # 2) Dynamics with E-dependent noise
    A, calm = 20.0, 0
    for _ in range(n_epoch):
        A_prev = A
        dist = abs(E - E_center) / max(E_width, 1e-12)
        noise_scale = 2.0 * (1.0 + 0.5 * dist**2)  # more noise further from center
        A = A * 1.02 + np.random.normal(0, noise_scale)

        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        calm = calm + 1 if delta < rel_eps else 0
        if calm >= lock_consec:
            return 1
    return 0

# ======================================================
# 4) Law lock-in (E only, energy-dependent noise)
# ======================================================
def law_lock_in(E, n_epoch=None):
    """Simulate law lock-in using Energy E only; returns (locked_epoch or -1, history)."""
    if n_epoch is None:
        n_epoch = MASTER_CTRL["N_epoch"]

    # Goldilocks alignment around E_CENTER
    f = np.exp(-((E - E_CENTER)**2) / (2.0 * (E_WIDTH**2)))
    if f < 0.2:
        return -1, []  # too far from the window → no lock-in

    # Initial 'c' candidate value (e.g., speed of light proxy)
    c_val = np.random.normal(3e8, 1e7)
    calm, locked_at = 0, None
    history = []

    for n in range(n_epoch):
        prev = c_val
        dist = abs(E - E_CENTER) / max(E_WIDTH, 1e-12)
        noise = 1e6 * (1.0 + 0.4 * dist) * np.random.uniform(0.8, 1.2)

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

# ======================================================
# 5) Monte Carlo: Stability + Law lock-in 
#   (Energy E only, all universes tracked)
# ======================================================

E_vals, stables, law_epochs, final_cs, all_histories = [], [], [], [], []
X_vals = []  # E-only: X = E
sub_seeds = []     # store per-universe seeds

for i in range(NUM_UNIVERSES):
    # Each universe gets its own seed from the master RNG
    sub_seed = master_rng.integers(0, 2**32)
    sub_seeds.append(sub_seed)  # <<< save the sub-seed!
    rng = np.random.default_rng(sub_seed)

    # 🔧 Sync legacy RNG so functions using np.random become per-universe reproducible
    np.random.seed(int(sub_seed))

    # Sample Energy E using universe-specific RNG
    Ei = float(rng.lognormal(E_MU, E_SIGMA))
    E_vals.append(Ei)
    X_vals.append(Ei)

    # Stability check uses rng too if you want reproducibility
    s = is_stable(Ei)
    stables.append(s)

    if s == 1:
        lock_epoch, c_hist = law_lock_in(Ei, n_epoch=MASTER_CTRL["N_epoch"])
        law_epochs.append(lock_epoch)

        if len(c_hist) > 0:
            final_cs.append(c_hist[-1])
            all_histories.append(c_hist)
        else:
            final_cs.append(np.nan)
    else:
        law_epochs.append(-1)
        final_cs.append(np.nan)

# --- Median epoch of law lock-in (only for universes that actually locked) ---
valid_epochs = [e for e in law_epochs if e >= 0]
median_epoch = float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None

print(f"🔒 Universes with lock-in: {len(valid_epochs)} / {NUM_UNIVERSES}")

# --- Stability table ---
stability_df = pd.DataFrame({
    "E": E_vals,
    "Stable": stables,
    "lock_epoch": law_epochs,
    "final_c": final_cs
})
stability_df.to_csv(os.path.join(SAVE_DIR, "stability.csv"), index=False)
assert len(E_vals) == len(stables) == len(law_epochs) == len(final_cs), \
    f"Length mismatch: E={len(E_vals)}, S={len(stables)}, lock={len(law_epochs)}, c={len(final_cs)}"

# --- Save per-universe seeds (for reproducibility) ---
pd.DataFrame({"universe_id": np.arange(NUM_UNIVERSES), "seed": sub_seeds}).to_csv(
    os.path.join(SAVE_DIR, "universe_seeds.csv"), index=False
)
    
# ======================================================
# 6) Stability summary (counts + percentages) - unified
# ======================================================
stable_count = int(sum(stables))
unstable_count = int(NUM_UNIVERSES - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {NUM_UNIVERSES}")
print(f"Stable universes:   {stable_count} ({stable_count/NUM_UNIVERSES*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/NUM_UNIVERSES*100:.2f}%)")

# --- Save to summary JSON ---
summary = {
    "stability_counts": {
        "total_universes": NUM_UNIVERSES,
        "stable_universes": stable_count,
        "unstable_universes": unstable_count,
        "stable_percent": float(stable_count/NUM_UNIVERSES*100),
        "unstable_percent": float(unstable_count/NUM_UNIVERSES*100)
    },
    "seeds": {
        "master_seed": master_seed,
        "universe_seeds_csv": "universe_seeds.csv"
    }
}
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

# --- Save bar chart with Lock-in ---
lockin_count = len(valid_epochs)   # universes that actually locked in

plt.figure()
plt.bar(
    ["Lock-in", "Stable", "Unstable"],
    [lockin_count, stable_count, unstable_count],
    color=["blue", "green", "red"]
)
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")

labels = [
    f"Lock-in ({lockin_count}, {lockin_count/NUM_UNIVERSES*100:.1f}%)",
    f"Stable ({stable_count}, {stable_count/NUM_UNIVERSES*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/NUM_UNIVERSES*100:.1f}%)"
]
plt.xticks([0, 1, 2], labels)

savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# 7) Average law lock-in dynamics across all universes (E-only)
# ======================================================
if len(all_histories) > 0:
    # Truncate all histories to the shortest length
    min_len = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]

    # Compute mean and standard deviation
    avg_c = np.mean(truncated, axis=0)
    std_c = np.std(truncated, axis=0)

    # ALWAYS save to CSV
    pd.DataFrame({
        "epoch": np.arange(min_len),
        "avg_c": avg_c,
        "std_c": std_c
    }).to_csv(os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False)

    # Optional plot
    if MASTER_CTRL["PLOT_AVG_LOCKIN"] and (median_epoch is not None):
        plt.figure()
        plt.plot(avg_c, label="Average c value")
        plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c,
                         alpha=0.3, color="blue", label="±1σ")
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in ≈ {median_epoch:.0f}")
        plt.title("Average law lock-in dynamics (Monte Carlo, E-only)")
        plt.xlabel("epoch"); plt.ylabel("c value (m/s)"); plt.legend()
        savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

# ======================================================
# 8) Expansion dynamics (E only)
# ======================================================
def evolve(E, n_epoch=None):
    if n_epoch is None:
        n_epoch = MASTER_CTRL["expansion_epochs"]
    A_series = []
    A = 20
    for n in range(n_epoch):
        A = A * 1.005 + np.random.normal(0, 1.0)
        A_series.append(A)
    return A_series

A_series = evolve(E, n_epoch=MASTER_CTRL["expansion_epochs"])
plt.figure()
plt.plot(A_series, label="Amplitude A")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")

if median_epoch is not None:
    plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {int(median_epoch)}")
    title_suffix = ""
else:
    title_suffix = " (no lock-in observed)"

plt.title("t > 0 : Expansion dynamics" + title_suffix)
plt.xlabel("epoch"); plt.ylabel("Amplitude A"); plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# ======================================================
# 9) Histogram of lock-in epochs  (CSV always, PNG optional)
# ======================================================
# Save raw epochs to CSV
pd.DataFrame({"lock_epoch": valid_epochs}).to_csv(
    os.path.join(SAVE_DIR, "law_lockin_epochs.csv"), index=False
)

# Optional plot
if MASTER_CTRL["PLOT_LOCKIN_HIST"] and len(valid_epochs) > 0:
    plt.figure()
    plt.hist(valid_epochs, bins=50, color="blue", alpha=0.7)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in = {int(median_epoch)}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (Monte Carlo, E-only)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# Final summary update (E-only)
# ======================================================
summary.update({
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
    "seeds": {
        "master_seed": MASTER_CTRL["seed"],
        "sub_seeds": sub_seeds
    }
})

# Save summary JSON (fix: convert numpy types)
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(
        summary, f, indent=2,
        default=lambda x: float(x) if isinstance(x, (np.floating,))
                        else int(x) if isinstance(x, (np.integer,))
                        else str(x)
    )

print("✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
