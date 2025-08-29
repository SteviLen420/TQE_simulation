# ===========================================================================
# Theory of the Question of Existence (TQE)
# (E) Vacuum fluctuation → Collapse → Expansion → Stability → Law lock-in
# ===========================================================================
# Author: Stefan Len
# Description: Energy-only model of cosmogenesis via vacuum fluctuation
# Focus: Stability emergence without informational orientation (I = 0)
# Mechanisms: Goldilocks window, critical thresholds, stochastic dynamics
# ===========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import shap
from lime.lime_tabular import LimeTabularExplainer
import os, time, json, numpy as np, matplotlib.pyplot as plt, shutil
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

# ===== Plot control flags =====
PLOT_AVG_LOCKIN  = False   # average lock-in dynamics plot (optional)
PLOT_LOCKIN_HIST = False   # histogram of lock-in epochs plot (optional)

# number of epochs for expansion dynamics
EXPANSION_EPOCHS = 500

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
E = float(np.random.lognormal(mean=2.5, sigma=0.8))
X = E

collapse_t = np.linspace(-0.2, 0.2, 200)
X_vals = X + 0.5 * np.random.randn(len(collapse_t))
X_vals[collapse_t >= 0] = X + 0.05 * np.random.randn(np.sum(collapse_t >= 0))

plt.plot(collapse_t, X_vals, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X, color="r", ls="--", label=f"Lock-in X={X:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E only)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

pd.DataFrame({"time": collapse_t,"X_vals": X_vals}).to_csv(
    os.path.join(SAVE_DIR, "collapse.csv"), index=False)
    
# ======================================================
# 3) Stability check (Energy-only)
# ======================================================
def is_stable(E, n_epoch=30):
    """Check if a universe is stable based only on A dynamics."""
    A = 20
    calm = 0
    for n in range(n_epoch):
        A_prev = A
        A = A * 1.02 + np.random.normal(0, 2)
        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)

        if delta < 0.05:
            calm += 1
        else:
            calm = 0

        if calm >= 5:
            return 1   # stable universe

    return 0   # unstable

# ======================================================
# 4) Law lock-in (E only)
# ======================================================
def law_lock_in(E, n_epoch=500):
    """Simulates the law lock-in process based only on Energy E."""
    f = np.exp(-(E - 2.0)**2 / (2 * 0.3**2))  

    if f < 0.2:
        return -1, []   # too far from Goldilocks → no lock-in

    c_val = np.random.normal(3e8, 1e7)  # initial c
    calm, locked_at = 0, None
    history = []

    for n in range(n_epoch):
        prev = c_val
        noise = 1e6 * (1 + abs(E - 5) / 10) * np.random.uniform(0.8, 1.2)
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

N = 1000
E_vals, stables, law_epochs, final_cs, all_histories = [], [], [], [], []
X_vals = []  # E-only: X = E

for _ in range(N):
    # Sample Energy E from a log-normal distribution
    Ei = float(np.random.lognormal(2.5, 0.8))
    E_vals.append(Ei)

    # Check if this universe is stable
    s = is_stable(Ei)
    stables.append(s)

    if s == 1:
        # Run law lock-in only for stable universes
        lock_epoch, c_hist = law_lock_in(Ei, n_epoch=500)  # use 500 epochs if desired
        law_epochs.append(lock_epoch)

        if len(c_hist) > 0:
            # If lock-in occurred, save the final c value and full trajectory
            final_cs.append(c_hist[-1])
            all_histories.append(c_hist)
        else:
            # Stable, but NO lock-in → still append NaN to keep array lengths equal
            final_cs.append(np.nan)
    else:
        # Unstable universe → mark no lock-in and store NaN for c
        law_epochs.append(-1)
        final_cs.append(np.nan)

# --- Median epoch of law lock-in (only for universes that actually locked) ---
valid_epochs = [e for e in law_epochs if e >= 0]
median_epoch = float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None

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
    
# ======================================================
# 6) Stability summary (counts + percentages) - unified
# ======================================================
stable_count = int(sum(stables))
unstable_count = int(N - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {N}")
print(f"Stable universes:   {stable_count} ({stable_count/N*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/N*100:.2f}%)")

# --- Save to summary JSON ---
summary = {
    "stability_counts": {
        "total_universes": N,
        "stable_universes": stable_count,
        "unstable_universes": unstable_count,
        "stable_percent": float(stable_count/N*100),
        "unstable_percent": float(unstable_count/N*100)
    }
}
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

# --- Save bar chart ---
plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")
labels = [
    f"Stable ({stable_count}, {stable_count/N*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/N*100:.1f}%)"
]
plt.xticks([0, 1], labels)

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
    if PLOT_AVG_LOCKIN and (median_epoch is not None):
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
def evolve(E, n_epoch=EXPANSION_EPOCHS):
    A_series = []
    A = 20
    for n in range(n_epoch):
        A = A * 1.005 + np.random.normal(0, 1.0)
        A_series.append(A)
    return A_series

A_series = evolve(E, n_epoch=EXPANSION_EPOCHS)
plt.plot(A_series, label="Amplitude A")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")
plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {median_epoch:.0f}")
plt.title("t > 0 : Expansion dynamics")
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
if PLOT_LOCKIN_HIST and len(valid_epochs) > 0:
    plt.figure()
    plt.hist(valid_epochs, bins=50, color="blue", alpha=0.7)
    if median_epoch is not None:
        plt.axvline(median_epoch, color="r", ls="--", lw=2,
                    label=f"Median lock-in = {median_epoch:.0f}")
        plt.legend()
    plt.title("Distribution of law lock-in epochs (Monte Carlo, E-only)")
    plt.xlabel("Epoch of lock-in"); plt.ylabel("Count")
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))

# ======================================================
# Final summary update (E-only)
# ======================================================
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
        "mean_lock_epoch": float(np.mean(valid_epochs)) if len(valid_epochs) > 0 else None,
        "median_lock_epoch": float(np.median(valid_epochs)) if len(valid_epochs) > 0 else None,
        "locked_fraction": float(np.mean([1 if e >= 0 else 0 for e in law_epochs])),
        "mean_final_c": float(np.nanmean(final_cs)) if len(final_cs) > 0 else None,
        "std_final_c": float(np.nanstd(final_cs)) if len(final_cs) > 0 else None
    }
})

# Save summary JSON
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

print("✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
