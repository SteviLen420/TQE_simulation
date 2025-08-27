# ===========================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Vacuum fluctuation → Collapse → Expansion → Stability → Law lock-in
# ===========================================================================
# Author: Stefan Lengyel
# Description: Full model simulation of energy-information (E,I) dynamics
# Focus: How vacuum fluctuation leads to stable law-locked universes
# Mechanisms: Goldilocks zone emergence, KL divergence, lock-in detection
# ===========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import shap, lime, eli5
from captum.attr import IntegratedGradients
from interpret import show
import os, time, json, numpy as np, matplotlib.pyplot as plt, shutil
import qutip as qt
import pandas as pd

# --- Directories ---
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_law_lockin"
run_id = time.strftime("TQE_(E,I)law_lockin_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p): 
    plt.savefig(p,dpi=150,bbox_inches="tight")
    plt.close()

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
# t = 0 : Collapse (E·I coupling + Goldilocks factor)
# ======================================================

# Kullback–Leibler divergence
def KL(p, q, eps=1e-12):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p /= p.sum(); q /= q.sum()
    return np.sum(p * np.log(p / q))

# Goldilocks modulation factor
def f_EI(E, I, E_c=2.0, sigma=0.3, alpha=0.8):
    """
    Gaussian Goldilocks window centered at E_c with I-coupling.
    - E_c: preferred energy (Goldilocks center)
    - sigma: width of stability window
    - alpha: strength of I-coupling
    """
    return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

# Generate two random quantum states and compute KL-based orientation
psi1, psi2 = qt.rand_ket(8), qt.rand_ket(8)
p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
I = KL(p1, p2) / (1 + KL(p1, p2))  # Information orientation (0–1)

# Energy fluctuation
E = float(np.random.lognormal(mean=2.5, sigma=0.8))

# Apply Goldilocks filter
f = f_EI(E, I)

# Coupled parameter
X = E * I * f

# Collapse dynamics (before t=0 fluctuation, after lock-in)
collapse_t = np.linspace(-0.2, 0.2, 200)
X_vals = X + 0.5 * np.random.randn(len(collapse_t))
X_vals[collapse_t >= 0] = X + 0.05 * np.random.randn(np.sum(collapse_t >= 0))

plt.plot(collapse_t, X_vals, "k-", alpha=0.6, label="fluctuation → lock-in")
plt.axhline(X, color="r", ls="--", label=f"Lock-in X={X:.2f}")
plt.axvline(0, color="r", lw=2)
plt.title("t = 0 : Collapse (E·I coupling + Goldilocks)")
plt.xlabel("time (collapse)"); plt.ylabel("X = E·I·f")
plt.legend()
savefig(os.path.join(FIG_DIR, "collapse.png"))

collapse_df = pd.DataFrame({
    "time": collapse_t,
    "X_vals": X_vals
})
collapse_df.to_csv(os.path.join(SAVE_DIR, "collapse.csv"), index=False)


# ======================================================
# Additional lock-in: Physical laws (speed of light c)
# ======================================================
def law_lock_in(E, I, n_epoch=200):
    """
    Simulates the lock-in of physical laws with Goldilocks modulation.
    """
    f = f_EI(E, I)
    if f < 0.2:   # Outside Goldilocks → no lock-in
        return -1, []

    c_val = np.random.normal(3e8, 1e7)  # initial speed of light
    calm = 0
    locked_at = None
    history = []

    for n in range(n_epoch):
        prev = c_val
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
    
# ======================================================
# Monte Carlo Simulation: Stability + Law lock-in for many universes
# ======================================================

# Number of universes to simulate
N = 1000

# Storage lists
X_vals, I_vals, stables, law_epochs, final_cs, all_histories = [], [], [], [], [], []
E_vals, f_vals = [], []

def is_stable(E, I, n_epoch=200):
    """
    Checks whether a universe stabilizes.
    Stability is determined by amplitude convergence inside the Goldilocks zone.
    """
    f = f_EI(E, I)
    if f < 0.2:  
        return 0   # too far from Goldilocks → unstable
    
    A = 20
    calm = 0
    for n in range(n_epoch):
        A_prev = A
        A = A*1.02 + np.random.normal(0, 2)  # amplitude update with noise
        delta = abs(A - A_prev) / max(abs(A_prev), 1e-6)

        if delta < 0.05:
            calm += 1
        else:
            calm = 0

        if calm >= 5:   # stabilized for 5 consecutive steps
            return 1
    return 0

# Run the simulation for N universes
for _ in range(N):
    Ei = float(np.random.lognormal(2.5,0.8))
    Ii = np.random.rand()
    fi = f_EI(Ei, Ii) 
    Xi = Ei * Ii

    # Save parameters
    X_vals.append(Xi)
    I_vals.append(Ii)
    E_vals.append(Ei)             
    f_vals.append(fi)

    # Stability check
    stable = is_stable(Ei, Ii)
    stables.append(stable)

    # Law lock-in
    lock_epoch, c_hist = law_lock_in(Ei, Ii) 
    law_epochs.append(lock_epoch)

   # --- Save results (only for stable universes in Goldilocks) ---
    if stables[-1] == 1 and len(c_hist) > 0:   # <-- only stable universes inside Goldilocks
        final_cs.append(c_hist[-1])
        all_histories.append(c_hist)
    else:
        final_cs.append(np.nan)
# Median epoch of law lock-in (only stable universes with lock-in)
median_epoch = np.median([e for e in law_epochs if e >= 0])

# ======================================================
# Build master DataFrame and save
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
        
# ======================================================
# Stability summary (counts + percentages)
# ======================================================
stable_count = int(df["stable"].sum())
unstable_count = int(len(df) - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {N}")
print(f"Stable universes:   {stable_count} ({stable_count/N*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/N*100:.2f}%)")

# --- Save to summary JSON (extend existing summary dict) ---
summary["stability_counts"] = {
    "total_universes": N,
    "stable_universes": stable_count,
    "unstable_universes": unstable_count,
    "stable_percent": float(stable_count/N*100),
    "unstable_percent": float(unstable_count/N*100)
}
with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

# --- Plot stability as bar chart ---
plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")
plt.text(0, stable_count/2, f"{stable_count}\n({stable_count/N*100:.1f}%)",
         ha="center", va="center", color="white", fontsize=12, fontweight="bold")
plt.text(1, unstable_count/2, f"{unstable_count}\n({unstable_count/N*100:.1f}%)",
         ha="center", va="center", color="white", fontsize=12, fontweight="bold")
savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ======================================================
# Average law lock-in dynamics across all universes
# ======================================================
if all_histories:
    # Truncate to the shortest history length
    min_len = min(len(h) for h in all_histories)
    truncated = [h[:min_len] for h in all_histories]

    # Compute average and standard deviation
    avg_c = np.mean(truncated, axis=0)
    std_c = np.std(truncated, axis=0)

    # Plot average dynamics
    plt.figure()
    plt.plot(avg_c, label="Average c value")
    plt.fill_between(np.arange(min_len), avg_c-std_c, avg_c+std_c, 
                     alpha=0.3, color="blue", label="±1σ")
    plt.axvline(median_epoch, color="r", ls="--", lw=2,
                label=f"Median lock-in ≈ {median_epoch:.0f}")
    plt.title("Average law lock-in dynamics (Monte Carlo)")
    plt.xlabel("epoch")
    plt.ylabel("c value (m/s)")
    plt.legend()
    savefig(os.path.join(FIG_DIR, "law_lockin_avg.png"))

    # Save average to CSV
    avg_df = pd.DataFrame({
        "epoch": np.arange(min_len),
        "avg_c": avg_c,
        "std_c": std_c
    })
    avg_df.to_csv(os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False)

# ======================================================
# t > 0 : Expansion dynamics (reference universe E,I)
# ======================================================
def evolve(E, I, n_epoch=200):   
    A_series = []
    I_series = []
    A = 20
    orient = I
    for n in range(n_epoch):
        # Amplitude growth with stronger noise
        A = A * 1.005 + np.random.normal(0, 1.0)

        # Orientation: stronger convergence + larger noise
        noise = 0.25 * (1 + 1.5 * abs(orient - 0.5))
        orient += (0.5 - orient) * 0.35 + np.random.normal(0, noise)

        # Clamp orientation between 0 and 1
        orient = max(0, min(1, orient))

        A_series.append(A)
        I_series.append(orient)

    return A_series, I_series


# Run expansion
A_series, I_series = evolve(E, I, n_epoch=200)

# Plot expansion dynamics with law lock-in marker
plt.plot(A_series, label="Amplitude A")
plt.plot(I_series, label="Orientation I")
plt.axhline(np.mean(A_series), color="gray", ls="--", alpha=0.5, label="Equilibrium A")

# red dashed line for the median law lock-in
plt.axvline(median_epoch, color="r", ls="--", lw=2, label=f"Law lock-in ≈ {median_epoch:.0f}")

plt.title("t > 0 : Expansion dynamics")
plt.xlabel("epoch")
plt.ylabel("Parameters")
plt.legend()
savefig(os.path.join(FIG_DIR, "expansion.png"))

# ======================================================
# Histogram of lock-in epochs
# ======================================================
# Histogram of lock-in epochs
valid_epochs = [e for e in law_epochs if e >= 0]

if len(valid_epochs) > 0:
    median_epoch = np.median(valid_epochs)
    plt.figure()
    plt.hist(valid_epochs, bins=50, color="blue", alpha=0.7)
    plt.axvline(median_epoch, color="r", ls="--", lw=2,
                label=f"Median lock-in = {median_epoch:.0f}")
    plt.title("Distribution of law lock-in epochs (Monte Carlo)")
    plt.xlabel("Epoch of lock-in")
    plt.ylabel("Count")
    plt.legend()
    savefig(os.path.join(FIG_DIR, "law_lockin_mc.png"))
    
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
        "mean_lock_epoch": float(np.mean([e for e in law_epochs if e >= 0])),
        "median_lock_epoch": float(np.median([e for e in law_epochs if e >= 0])),
        "locked_fraction": float(np.mean([1 if e >= 0 else 0 for e in law_epochs])),
        "mean_final_c": float(np.mean(final_cs)),
        "std_final_c": float(np.std(final_cs))
    }
}

with open(os.path.join(SAVE_DIR,"summary.json"),"w") as f:
    json.dump(summary, f, indent=2)

# Save law lock-in dynamics to CSV
if all_histories:
    law_df = pd.DataFrame({
        "epoch": np.arange(min_len),
        "avg_c": avg_c,
        "std_c": std_c
    })
    law_df.to_csv(os.path.join(SAVE_DIR, "law_lockin_avg.csv"), index=False)
    
# ======================================================
# 🔍 XAI (SHAP + LIME) analysis
# ======================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer

# Feature mátrix és célok
X_feat = df[["E","I","fEI","X"]].fillna(0.0)
y_reg  = df["X"]                       # példa: a kompozit paraméter előrejelzése
y_cls  = df["stable"].astype(int)      # stabil-e (0/1)

# Train / test split
Xtr, Xte, ytr_reg, yte_reg = train_test_split(X_feat, y_reg, test_size=0.25, random_state=42)
_,  _,  ytr_cls, yte_cls   = train_test_split(X_feat, y_cls, test_size=0.25, random_state=42)

# Baseline modellek
rf_reg = RandomForestRegressor(n_estimators=400, random_state=42).fit(Xtr, ytr_reg)
rf_cls = RandomForestClassifier(n_estimators=400, random_state=42).fit(Xtr, ytr_cls)

print("Reg R^2:", r2_score(yte_reg, rf_reg.predict(Xte)))
print("Cls Acc:", accuracy_score(yte_cls, rf_cls.predict(Xte)))

# --- SHAP: regresszió ---
expl_reg = shap.TreeExplainer(rf_reg)
shap_vals_reg = expl_reg.shap_values(X_feat)

plt.figure()
shap.summary_plot(shap_vals_reg, X_feat, show=False)
plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- SHAP: osztályozás (class=1, stable) ---
expl_cls = shap.TreeExplainer(rf_cls)
shap_vals_cls = expl_cls.shap_values(X_feat)[1]

plt.figure()
shap.summary_plot(shap_vals_cls, X_feat, show=False)
plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- SHAP: lokális példa ---
i = 0
shap.force_plot(expl_reg.expected_value, shap_vals_reg[i,:], X_feat.iloc[i,:], matplotlib=True, show=False)
plt.savefig(os.path.join(FIG_DIR, "shap_force_example.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- LIME: lokális magyarázat (regresszió) ---
explainer = LimeTabularExplainer(
    training_data=Xtr.values,
    feature_names=X_feat.columns.tolist(),
    discretize_continuous=True,
    mode='regression'
)
exp = explainer.explain_instance(Xte.iloc[0].values, rf_reg.predict, num_features=5)
lime_pairs = exp.as_list()
pd.DataFrame(lime_pairs, columns=["feature","weight"]).to_csv(
    os.path.join(FIG_DIR, "lime_example.csv"), index=False
)
print("XAI artifacts saved to:", FIG_DIR)
    
print("✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
