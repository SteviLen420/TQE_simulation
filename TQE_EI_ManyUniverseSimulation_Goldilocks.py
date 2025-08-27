# ========================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Many-universe simulation with Goldilocks principle
# ========================================================================
# Author: Stefan Len
# Description: Monte Carlo simulation of multiple universes evolving from
# vacuum fluctuations, influenced by energy (E) and information (I) dynamics.
# Focus: Statistical emergence of law-consistent universes within Goldilocks zone
# Mechanisms: KL divergence-based orientation, law-lock-in detection, entropy tracking
# ========================================================================

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import shap, lime, eli5
from captum.attr import IntegratedGradients
from interpret import show
import os, time, json, numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.stats import entropy
import pandas as pd

# --- Directories ---
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_UNI_MANY"
run_id = time.strftime("TQE_(E,I)_UNI_MANY_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p): 
    plt.savefig(p,dpi=150,bbox_inches="tight")
    plt.close()

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        
# ========== Parameters ==========
NUM_REGIONS = 8
NUM_STATES  = 50
STEPS       = 250
STABILITY_THRESHOLD = 6

# TQE parameters
E_c   = 2.0
sigma = 0.5
alpha = 0.8

# ========== Information parameter (I) ==========
def info_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12
    
    # KL divergence
    KL = np.sum(p1 * np.log((p1+eps)/(p2+eps)))
    I_kl = KL / (1 + KL)
    
    # Shannon entropy (normalized)
    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))
    
    # Multiplicative combination
    I = (I_kl * I_shannon) / (1 + I_kl * I_shannon)
    return I

# ========== Energy sampling ==========
def sample_energy(mu=2.5, sigma=0.9):
    return float(np.random.lognormal(mean=mu, sigma=sigma))

# ========== TQE modulation factor f(E,I) ==========
def f_EI(E, I, E_c=E_c, sigma=sigma, alpha=alpha):
    return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

# ========== Single universe simulation ==========
def run_universe():
    # --- Energy and information (classic version) ---
    E = sample_energy()     # energy drawn from lognormal distribution
    I = info_param()        # information parameter (KL + Shannon)
    f = f_EI(E, I)          # Goldilocks modulation

    # Initial states (break symmetry)
    states = np.zeros((NUM_REGIONS, NUM_STATES))   # all zeros initially
    states[0,:] = 1.0   # one-hot initialization for first region

    # Reset lists at the start of each run
    region_entropies = []
    global_entropy   = []
    amplitude        = []
    orientation      = []
    purities         = []

    lock_in_step, consecutive_calm = None, 0

    for step in range(STEPS):
        noise_scale = max(0.02, 1.0 - step / STEPS)  # large noise at beginning (Big Bang), small noise later (cooling)

        # --- amplitude growth ---
        if step == 0:
            A = 1.0
            orient = I
        else:
            A = A * 1.01 + np.random.normal(0, 0.02)    # exponential growth with noise
            orient += (0.5 - orient) * 0.1 + np.random.normal(0, 0.02) 
            orient = np.clip(orient, 0, 1)

        amplitude.append(A)
        orientation.append(orient)

        # --- dynamic energy drift ---
        E += np.random.normal(0, 0.05)   # small random walk for energy
        f = f_EI(E, I)                   # recompute modulation factor each step

        # --- region updates with noise ---
        for r in range(NUM_REGIONS):
            # base noise depending on time (cooling effect)
            noise = np.random.normal(0, noise_scale * 3.0, NUM_STATES)   # stronger noise

            # occasional large "catastrophic event" (supernova, black hole, etc.)
            if np.random.rand() < 0.05:  
                noise += np.random.normal(0, 8.0, NUM_STATES)

            # dynamic fluctuation in f(E,I)
            f_step = f * (1 + np.random.normal(0, 0.1))

            # update region states
            states[r] += f_step * noise
            states[r] = np.clip(states[r], 0, 1)

        # --- entropies ---
        region_entropies.append([entropy(states[r]) for r in range(NUM_REGIONS)])
        global_entropy.append(entropy(states.flatten()))

        # --- purity of random state ---
        psi = qt.rand_ket(NUM_STATES)
        rho = psi * psi.dag()
        purities.append((rho*rho).tr().real)

        # --- lock-in detection ---
        if step > 0:
            delta = abs(global_entropy[-1] - global_entropy[-2]) / max(global_entropy[-2], 1e-9)
            if delta < 0.001:   # threshold for calmness
                consecutive_calm += 1
                if consecutive_calm >= 10 and lock_in_step is None:
                    lock_in_step = step
            else:
                consecutive_calm = 0

    return region_entropies, global_entropy, amplitude, orientation, purities, (E, I, f), lock_in_step

# ======================================================
# Monte Carlo: Run many universes
# ======================================================
N = 100  # number of universes
results = []

for i in range(N):
    region_entropies, global_entropy, amplitude, orientation, purities, params, lock_in_step = run_universe()
    
    results.append({
    "E": params[0],                  # E
    "I": params[1],                  # I
    "fEI": params[2],                # f(E,I)
    "lock_in_step": lock_in_step,
    "stable": float(np.mean(global_entropy) < STABILITY_THRESHOLD),
    "mean_entropy": float(np.mean(global_entropy)),
    "mean_amplitude": float(np.mean(amplitude)),
    "mean_orientation": float(np.mean(orientation)),
    "mean_purity": float(np.mean(purities))
})

# Convert to DataFrame
df_mc = pd.DataFrame(results)
df_mc.to_csv(os.path.join(SAVE_DIR, "montecarlo_results.csv"), index=False)

# Quick statistics
stable_count = df_mc["stable"].sum()
unstable_count = N - stable_count
print("\n🌌 Monte Carlo Universe Stability")
print(f"Total universes: {N}")
print(f"Stable:   {stable_count} ({stable_count/N*100:.2f}%)")
print(f"Unstable: {unstable_count} ({unstable_count/N*100:.2f}%)")

time_axis = np.arange(STEPS)

# ======================================================
# Average entropy evolution across many universes
# ======================================================

# Collect all global entropies into one array
all_entropies = np.array([run_universe()[1] for _ in range(N)])  # shape: (N, STEPS)

# Compute average and standard deviation
mean_entropy = np.mean(all_entropies, axis=0)
std_entropy = np.std(all_entropies, axis=0)

# Plot average entropy curve
plt.figure(figsize=(12,6))
plt.plot(time_axis, mean_entropy, color="blue", linewidth=2, label="Average global entropy")

# Add uncertainty band (mean ± std)
plt.fill_between(time_axis,
                 mean_entropy - std_entropy,
                 mean_entropy + std_entropy,
                 color="blue", alpha=0.2, label="±1 std deviation")

# Add stability threshold
plt.axhline(y=STABILITY_THRESHOLD, color="red", linestyle="--", label="Stability threshold")

plt.title(f"Average Global Entropy across {N} universes")
plt.xlabel("Time step")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)

# Save figure
savefig(os.path.join(FIG_DIR, "average_entropy_many.png"))

# ========== Run & plot ==========
(
    region_entropies,   # 1
    global_entropy,     # 2
    amplitude,          # 3
    orientation,        # 4
    purities,           # 5
    params,             # 6  (E, I, f))
    lock_in_step        # 7
) = run_universe()

time_axis = range(STEPS)

plt.figure(figsize=(12,6))
for r in range(min(NUM_REGIONS, 10)):  
    plt.plot(time_axis, [region_entropies[t][r] for t in time_axis],
             alpha=1.0, lw=1, label=f"Region {r} entropy")

plt.plot(time_axis, global_entropy, color="black", linewidth=2, label="Global entropy")
plt.axhline(y=STABILITY_THRESHOLD, color="red", linestyle="--", label="Stability threshold")

plt.title("TQE Universe Simulation with f(E,I)")
plt.xlabel("Time step")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
savefig(os.path.join(FIG_DIR, "entropy_evolution.png"))


# ======================================================
# Stability summary (counts + percentages)
# ======================================================
stable_count = int(df_mc["stable"].sum())
unstable_count = int(N - stable_count)

print("\n🌌 Universe Stability Summary")
print(f"Total universes simulated: {N}")
print(f"Stable universes:   {stable_count} ({stable_count/N*100:.2f}%)")
print(f"Unstable universes: {unstable_count} ({unstable_count/N*100:.2f}%)")

# --- Save bar chart ---
plt.figure()
plt.bar(["Stable", "Unstable"], [stable_count, unstable_count], color=["green", "red"])
plt.title("Universe Stability Distribution")
plt.ylabel("Number of Universes")
plt.xlabel("Category")

# Labels with counts + percentages next to categories
labels = [
    f"Stable ({stable_count}, {stable_count/N*100:.1f}%)",
    f"Unstable ({unstable_count}, {unstable_count/N*100:.1f}%)"
]
plt.xticks([0, 1], labels)

savefig(os.path.join(FIG_DIR, "stability_summary.png"))

# ========== Save data ==========
df = pd.DataFrame({"time": time_axis, "global_entropy": global_entropy})
df.to_csv(os.path.join(SAVE_DIR, "global_entropy.csv"), index=False)

summary = {
    "params": {"E": params[0], "I": params[1], "f(E,I)": params[2]},
    "lock_in_step": lock_in_step,
    "stable": float(np.mean(global_entropy) < STABILITY_THRESHOLD),
    "mean_entropy": float(np.mean(global_entropy)),
    "mean_amplitude": float(np.mean(amplitude)),
    "mean_orientation": float(np.mean(orientation)),
    "mean_purity": float(np.mean(purities))
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# ======================================================
# 🔍 XAI (SHAP + LIME) analysis on Monte Carlo table
# ======================================================

import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

# Feature-mátrix és célok
X_feat = df_mc[["E","I","fEI"]].copy()
# Regressziós cél: mean_entropy (vagy lock_in_step, ha sok a nem-NaN)
y_reg  = df_mc["mean_entropy"].values
# Osztályozási cél: stable (0/1)
y_cls  = df_mc["stable"].astype(int).values

# Train/test
Xtr, Xte, ytr_reg, yte_reg = train_test_split(X_feat, y_reg, test_size=0.25, random_state=42)
_,  _,  ytr_cls, yte_cls   = train_test_split(X_feat, y_cls, test_size=0.25, random_state=42)

# Modellek
rf_reg = RandomForestRegressor(n_estimators=400, random_state=42).fit(Xtr, ytr_reg)
rf_cls = RandomForestClassifier(n_estimators=400, random_state=42).fit(Xtr, ytr_cls)

print("Reg R^2:", r2_score(yte_reg, rf_reg.predict(Xte)))
print("Cls Acc:", accuracy_score(yte_cls, rf_cls.predict(Xte)))

# --- SHAP: regresszió ---
expl_reg = shap.TreeExplainer(rf_reg)
shap_vals_reg = expl_reg.shap_values(X_feat)

plt.figure()
shap.summary_plot(shap_vals_reg, X_feat, show=False)
plt.title("SHAP summary – regression (mean_entropy)")
plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_mean_entropy.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- SHAP: osztályozás (class=1 = stable) ---
expl_cls = shap.TreeExplainer(rf_cls)
# A TreeExplainer osztályonként ad shap_values listát → a pozitív (1) osztály kell
shap_vals_cls = expl_cls.shap_values(X_feat)[1]

plt.figure()
shap.summary_plot(shap_vals_cls, X_feat, show=False)
plt.title("SHAP summary – classification (stable)")
plt.savefig(os.path.join(FIG_DIR, "shap_summary_cls_stable.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- SHAP: lokális példa (első sor) ---
i = 0
shap.force_plot(expl_reg.expected_value, shap_vals_reg[i,:], X_feat.iloc[i,:], matplotlib=True, show=False)
plt.savefig(os.path.join(FIG_DIR, "shap_force_example.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- LIME: lokális regressziós magyarázat ---
explainer = LimeTabularExplainer(
    training_data=Xtr.values,
    feature_names=X_feat.columns.tolist(),
    discretize_continuous=True,
    mode='regression'
)
exp = explainer.explain_instance(Xte.iloc[0].values, rf_reg.predict, num_features=5)
lime_pairs = exp.as_list()
pd.DataFrame(lime_pairs, columns=["feature","weight"]).to_csv(
    os.path.join(FIG_DIR, "lime_example_regression.csv"), index=False
)
print("XAI artifacts saved to:", FIG_DIR)

print("✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
