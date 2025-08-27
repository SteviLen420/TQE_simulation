# ================================================================
# Theory of the Question of Existence (TQE)
# (E) Vacuum fluctuation → Collapse → Expansion → Stability
# ================================================================
# Author: Stefan Len
# Description: Monte Carlo simulation using energy (E) only
# Focus: Emergence of structure without informational orientation
# ================================================================

import shap, lime, eli5
from captum.attr import IntegratedGradients
from interpret import show
import os, time, json, numpy as np, matplotlib.pyplot as plt, shutil
import qutip as qt
import pandas as pd

# --- Directories ---
run_id = time.strftime("TQE_(E)SUPERPOSITION_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(os.getcwd(), run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)
def savefig(p): plt.savefig(p,dpi=150,bbox_inches="tight"); plt.close()

# ======================================================
# t < 0 : Quantum superposition (vacuum fluctuation)
# ======================================================
Nlev = 12
a = qt.destroy(Nlev)

H0 = a.dag()*a + 0.05*(np.random.randn()*a + np.random.randn()*a.dag())  # perturbed Hamiltonian

psi0 = qt.rand_ket(Nlev)   # initial random superposition
rho = psi0 * psi0.dag()

tlist = np.linspace(0,10,200)

gammas = 0.02 + 0.01*np.sin(0.5*tlist) + 0.005*np.random.randn(len(tlist))  # fluctuating environment

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
# --- Save superposition results ---
superposition_df = pd.DataFrame({
    "time": tlist,
    "Entropy": S,
    "Purity": P
})
superposition_df.to_csv(os.path.join(SAVE_DIR, "superposition.csv"), index=False)

# ======================================================
# t = 0 : Collapse (Energy lock-in)
# ======================================================
E = float(np.random.lognormal(mean=2.5, sigma=0.8))  # Energy fluctuation
X = E   # only energy

collapse_t = np.linspace(-0.2, 0.2, 200)
X_vals = X + 0.5 * np.random.randn(len(collapse_t))
X_vals[collapse_t >= 0] = X + 0.05 * np.random.randn(np.sum(collapse_t >= 0))

plt.plot(collapse_t, X_vals, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X, color="r", ls="--", label=f"Lock-in X={X:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (Energy lock-in)")
plt.xlabel("time (collapse)"); plt.ylabel("X = Energy"); plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))
# --- Save collapse results ---
collapse_df = pd.DataFrame({
    "time": collapse_t,
    "X_vals": X_vals
})
collapse_df.to_csv(os.path.join(SAVE_DIR, "collapse.csv"), index=False)

# ======================================================
# t > 0 : Expansion dynamics
# ======================================================
def evolve(E,n_epoch=30):
    A_series=[]
    A=20
    for n in range(n_epoch):
        A = A*1.03 + np.random.normal(0,2.0)   # amplitude growth with noise
        A_series.append(A)
    return A_series

A_series = evolve(E)

plt.plot(A_series, label="Amplitude A")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5)
plt.title("t > 0 : Expansion dynamics")
plt.xlabel("epoch")
plt.ylabel("Amplitude")
plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# ======================================================
# Stability summary
# ======================================================
def is_stable(E,n_epoch=30):
    A = 20
    calm = 0
    for n in range(n_epoch):
        A_prev = A
        A = A*1.02 + np.random.normal(0,2)
        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)
        if delta < 0.05:
            calm += 1
        else:
            calm = 0
        if calm >= 5:
            return 1   # stable
    return 0          # unstable

N=10000
X_vals=[]; stables=[]
for _ in range(N):
    Ei = float(np.random.lognormal(2.5,0.8))
    X_vals.append(Ei)
    stables.append(is_stable(Ei))

colors = ["green" if s==1 else "red" for s in stables]
plt.scatter(X_vals, stables, c=colors, alpha=0.7)
plt.title("Final stability outcome")
plt.xlabel("X = Energy")
plt.yticks([0,1], ["Unstable","Stable"])
savefig(os.path.join(FIG_DIR,"stability.png"))

stable_count = sum(stables)
unstable_count = N - stable_count
print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {N}")
print(f"Stable:   {stable_count} ({stable_count/N*100:.2f}%)")
print(f"Unstable: {unstable_count} ({unstable_count/N*100:.2f}%)")

# ======================================================
# Stability summary (counts + percentages)
# ======================================================
stable_count = int(sum(stables))        
unstable_count = int(N - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {N}")
print(f"Stable universes:   {stable_count} ({stable_count/len(df)*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/len(df)*100:.2f}%)")

# --- Save bar chart ---
plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")

# Labels with counts + percentages next to categories
labels = [
    f"Stable ({stable_count}, {stable_count/len(df)*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/len(df)*100:.1f}%)"
]
plt.xticks([0, 1], labels)

savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# Save results (JSON + CSV + Figures)
# ======================================================
summary = {
    "E": E,
    "X": X,
    "stable": int(is_stable(E)),
    "superposition": {
        "mean_entropy": float(np.mean(S)),
        "mean_purity": float(np.mean(P))
    },
    "collapse": {
        "lock_in_value": float(X)
    }
}
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Simulation complete")
print(f"E={E:.3f}, X={X:.3f}, Stable={summary['stable']}")
print(f"📂 Results in {SAVE_DIR}")

# --- Save CSVs ---
superposition_df.to_csv(os.path.join(SAVE_DIR, "superposition.csv"), index=False)
collapse_df.to_csv(os.path.join(SAVE_DIR, "collapse.csv"), index=False)

expansion_df = pd.DataFrame({
    "epoch": np.arange(len(A_series)),
    "Amplitude_A": A_series
})
expansion_df.to_csv(os.path.join(SAVE_DIR, "expansion.csv"), index=False)

stability_df = pd.DataFrame({
    "X": X_vals,
    "Stable": stables
})
stability_df.to_csv(os.path.join(SAVE_DIR, "stability.csv"), index=False)

# ======================================================
# Save all outputs to Google Drive
# ======================================================
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E)_SUPERPOSITION"
GOOGLE_DIR = os.path.join(GOOGLE_BASE, run_id)
os.makedirs(GOOGLE_DIR, exist_ok=True)
for root, dirs, files in os.walk(SAVE_DIR):
    for file in files:
        if file.endswith((".png",".json",".npz",".csv")):
            src=os.path.join(root,file)
            dst=os.path.join(GOOGLE_DIR,os.path.relpath(root,SAVE_DIR))
            os.makedirs(dst,exist_ok=True); shutil.copy2(src,dst)
print(f"☁️ All results saved to Google Drive: {GOOGLE_DIR}")
