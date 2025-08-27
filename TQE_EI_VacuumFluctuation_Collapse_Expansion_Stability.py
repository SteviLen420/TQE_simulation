# =================================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Vacuum fluctuation → Collapse → Expansion → Stability
# =================================================================================
# Author: Stefan Lengyel
# Description: Monte Carlo simulation of energy-information dynamics
# Model: E, I coupling, KL divergence, Shannon entropy, Goldilocks zone emergence
# =================================================================================

import shap, lime, eli5
from captum.attr import IntegratedGradients
from interpret import show
import os, time, json, numpy as np, matplotlib.pyplot as plt, shutil
import qutip as qt
import pandas as pd

# --- Directories ---
run_id = time.strftime("TQE_(E,I)_SUPERPOSITION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(os.getcwd(), run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)
def savefig(p): plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()

# ======================================================
# Goldilocks modulation factor
# ======================================================
def f_EI(E, I, E_c=2.0, sigma=0.3, alpha=0.8):
    """
    Gaussian window around E_c with information coupling.
    - E_c: preferred energy (Goldilocks center)
    - sigma: width of stability window
    - alpha: strength of I-coupling
    """
    return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

# ======================================================
# t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
Nlev = 12
a = qt.destroy(Nlev)

# perturbed Hamiltonian with small random noise
H0 = a.dag()*a + 0.05*(np.random.randn()*a + np.random.randn()*a.dag())

# initial state: random superposition (not just vacuum)
psi0 = qt.rand_ket(Nlev)
rho = psi0 * psi0.dag()

# time scale
tlist = np.linspace(0,10,200)

# time-dependent gamma (fluctuating environment)
gammas = 0.02 + 0.01*np.sin(0.5*tlist) + 0.005*np.random.randn(len(tlist))

states = []
for g in gammas:
    # actual time evolution in a small window
    res = qt.mesolve(H0, rho, np.linspace(0,0.5,5), [np.sqrt(abs(g))*a], [])
    states.append(res.states[-1])

# purity and entropy
def purity(r): 
    return float((r*r).tr().real) if qt.isoper(r) else float((r*r.dag()).tr().real)

S = np.array([qt.entropy_vn(r) for r in states])
P = np.array([purity(r) for r in states])

# plot
plt.plot(tlist,S,label="Entropy")
plt.plot(tlist,P,label="Purity")
plt.title("t < 0 : Quantum superposition (vacuum fluctuation)")
plt.xlabel("time"); plt.legend(); savefig(os.path.join(FIG_DIR,"superposition.png"))
# Save superposition results to CSV
superposition_df = pd.DataFrame({
    "time": tlist,
    "Entropy": S,
    "Purity": P
})
superposition_df.to_csv(os.path.join(SAVE_DIR, "superposition.csv"), index=False)

# ======================================================
# t = 0 : Collapse (E·I coupling + Goldilocks)
# ======================================================

def KL(p, q, eps=1e-12):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p /= p.sum(); q /= q.sum()
    return np.sum(p * np.log(p / q))

psi1, psi2 = qt.rand_ket(8), qt.rand_ket(8)
p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
p1 /= p1.sum(); p2 /= p2.sum()

# Shannon entropy (normalized)
eps = 1e-12
H = -np.sum(p1 * np.log(p1 + eps))
I_shannon = H / np.log(len(p1))

# KL divergence (normalized)
KL_val = KL(p1, p2)
I_kl = KL_val / (1 + KL_val)

# Combined information parameter
I = (I_shannon * I_kl) / (1 + I_shannon * I_kl)

# Energy fluctuation
E = float(np.random.lognormal(mean=2.5, sigma=0.8))

# Goldilocks factor
f = f_EI(E, I)

# Coupled parameter with Goldilocks
X = E * I * f

# ======================================================
# Collapse dynamics (save as PNG + CSV)
# ======================================================
collapse_t = np.linspace(-0.2, 0.2, 200)

# Before t=0 → big fluctuations, after t=0 → lock-in around X
X_vals = X + 0.5 * np.random.randn(len(collapse_t))
X_vals[collapse_t >= 0] = X + 0.05 * np.random.randn(np.sum(collapse_t >= 0))

# Plot collapse
plt.plot(collapse_t, X_vals, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X, color="r", ls="--", label=f"Lock-in X={X:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E·I·f coupling)")
plt.xlabel("time (collapse)")
plt.ylabel("X = E·I·f")
plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

# Save collapse to CSV
collapse_df = pd.DataFrame({
    "time": collapse_t,
    "X_vals": X_vals
})
collapse_df.to_csv(os.path.join(SAVE_DIR, "collapse.csv"), index=False)

# ======================================================
# t > 0 : Expansion dynamics
# ======================================================
def evolve(E,I,n_epoch=30):
    A_series=[]; I_series=[]
    A=20; orient=I
    for n in range(n_epoch):
        A = A*1.03 + np.random.normal(0,2.0)
        f = f_EI(E,I)  # Goldilocks factor
        noise = f * 0.25 * (1 + 1.5 * abs(orient - 0.5))
        orient += (0.5 - orient)*0.35 + np.random.normal(0, noise)
        orient = max(0, min(1, orient))
        A_series.append(A); I_series.append(orient)
    return A_series, I_series


# Run expansion
A_series, I_series = evolve(E, I)

# Plot expansion dynamics
plt.plot(A_series, label="Amplitude A")
plt.plot(I_series, label="Orientation I")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5)  # equilibrium line
plt.title("t > 0 : Expansion dynamics")
plt.xlabel("epoch")
plt.ylabel("Parameters")
plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# ======================================================
# Stability summary (with Goldilocks filter)
# ======================================================
def is_stable(E, I, n_epoch=30):
    """Check if a universe is stable given energy E and information I."""
    # Apply Goldilocks factor
    f = np.exp(-(E - 2.0)**2 / (2 * 0.3**2)) * (1 + 0.8 * I)

    if f < 0.2:  
        # Too far from Goldilocks zone → unstable by definition
        return 0

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

    return 0   # unstable if threshold not reached


# Run many universes
N = 10000
X_vals = []
stables = []

for _ in range(N):
    Ei = float(np.random.lognormal(2.5, 0.8))
    Ii = np.random.rand()
    X_vals.append(Ei * Ii)
    stables.append(is_stable(Ei, Ii))

# Plot results
colors = ["green" if s == 1 else "red" for s in stables]
plt.scatter(X_vals, stables, c=colors, alpha=0.7)
plt.title("Final stability outcome (with Goldilocks)")
plt.xlabel("X = E·I")
plt.yticks([0, 1], ["Unstable", "Stable"])
savefig(os.path.join(FIG_DIR, "stability.png"))

# Count results
stable_count = sum(stables)
unstable_count = N - stable_count

print("\n🌌 Universe Stability Summary (with Goldilocks)")
print(f"Total universes simulated: {N}")
print(f"Stable:   {stable_count} ({stable_count/N*100:.2f}%)")
print(f"Unstable: {unstable_count} ({unstable_count/N*100:.2f}%)")

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
# Save results (JSON + CSV + Figures)
# ======================================================

# Save expansion dynamics to CSV
expansion_df = pd.DataFrame({
    "epoch": np.arange(len(A_series)),
    "Amplitude_A": A_series,
    "Orientation_I": I_series
})
expansion_df.to_csv(os.path.join(SAVE_DIR, "expansion.csv"), index=False)

# Save stability outcomes to CSV
stability_df = pd.DataFrame({
    "X": X_vals,
    "Stable": stables
})
stability_df.to_csv(os.path.join(SAVE_DIR, "stability.csv"), index=False)

# Save summary JSON (only once, full info)
summary = {
    "E": E,
    "I": I,
    "X": X,
    "stable": int(is_stable(E, I)),
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "lock_in_value": float(X),
        "mean_fluctuation": float(np.mean(X_vals)),
        "std_fluctuation": float(np.std(X_vals))
    }
}
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)
    
# ======================================================
# Save all outputs to Google Drive
# ======================================================
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_SUPERPOSITION"
GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)
os.makedirs(GOOGLE_DIR, exist_ok=True)
for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        if file.endswith((".png",".json",".npz",".csv")):
            src=os.path.join(root,file)
            dst=os.path.join(GOOGLE_DIR,os.path.relpath(root,SAVE_DIR))
            os.makedirs(dst,exist_ok=True); shutil.copy2(src,dst)
print(f"☁️ All results saved to Google Drive: {GOOGLE_DIR}")
