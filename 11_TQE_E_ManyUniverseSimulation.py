# ================================================================
# Theory of the Question of Existence (TQE)
# (E) Many-universe simulation
# ================================================================
# Author: Stefan Len
# Description: Monte Carlo simulation of multiple universes using
# only vacuum energy fluctuation (E), without informational orientation (I).
# Focus: Spontaneous emergence of stability and physical laws across universes
# without any directional bias in quantum state evolution.
# ================================================================

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
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E)_ONLY_UNI_SINGLE"
run_id = time.strftime("TQE_(E)_ONLY_UNI_SINGLE_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p): 
    plt.savefig(p,dpi=150,bbox_inches="tight")
    plt.close()

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        
# ========== Parameters ==========
NUM_REGIONS = 10
NUM_STATES  = 250
STEPS       = 500
STABILITY_THRESHOLD = 3.5

# TQE parameters
E_c   = 2.0
sigma = 0.5
alpha = 0

# ========== Energy sampling ==========
def sample_energy(mu=2.5, sigma=0.9):
    return float(np.random.lognormal(mean=mu, sigma=sigma))

# ========== TQE modulation factor E ==========
def f_E(E, E_c=E_c, sigma=sigma):
    return np.exp(-(E - E_c)**2 / (2 * sigma**2))

# ========== Single universe simulation ==========
def run_universe():
    # --- Energy and information (classic version) ---
    E = sample_energy()     # energy drawn from lognormal distribution
    f = f_E(E)              # Goldilocks modulation (only E)

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
            orient = 0.5
        else:
            A = A * 1.01 + np.random.normal(0, 0.02)    # exponential growth with noise
            orient += (0.5 - orient) * 0.1 + np.random.normal(0, 0.02) 
            orient = np.clip(orient, 0, 1)

        amplitude.append(A)
        orientation.append(orient)

        # --- dynamic energy drift ---
        E += np.random.normal(0, 0.05)   # small random walk for energy
        f = f_E(E)                   # recompute modulation factor each step

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

    return region_entropies, global_entropy, amplitude, orientation, purities, (E, f), lock_in_step

# ======================================================
# Monte Carlo: Run many universes
# ======================================================
N = 1000  # number of universes
results = []

for i in range(N):
    region_entropies, global_entropy, amplitude, orientation, purities, params, lock_in_step = run_universe()
    
    results.append({
        "E": params[0],
        "f(E)": params[1],
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

# Histogram of lock-in steps
plt.figure()
plt.hist(df_mc["lock_in_step"].dropna(), bins=50, color="blue", alpha=0.7)
plt.title("Distribution of lock-in steps (Monte Carlo)")
plt.xlabel("Lock-in step")
plt.ylabel("Count")
savefig(os.path.join(FIG_DIR, "lockin_hist.png"))

# ========== Run & plot ==========
(
    region_entropies,   # 1
    global_entropy,     # 2
    amplitude,          # 3
    orientation,        # 4
    purities,           # 5
    E,                  # 6 
    lock_in_step        # 7
) = run_universe()

time_axis = range(STEPS)

plt.figure(figsize=(12,6))
for r in range(min(NUM_REGIONS, 10)):  
    plt.plot(time_axis, [region_entropies[t][r] for t in time_axis],
             alpha=0.8, lw=1, label=f"Region {r} entropy")

plt.plot(time_axis, global_entropy, color="black", linewidth=2, label="Global entropy")
plt.axhline(y=STABILITY_THRESHOLD, color="red", linestyle="--", label="Stability threshold")

if lock_in_step is not None:
    plt.axvline(x=lock_in_step, color="purple", linestyle="--", linewidth=2,
                label=f"Lock-in step = {lock_in_step}")

plt.title("TQE Universe Simulation only with (E)")
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
    "params": {"E": params[0], "f(E)": params[1]},
    "lock_in_step": lock_in_step,
    "stable": float(np.mean(global_entropy) < STABILITY_THRESHOLD),
    "mean_entropy": float(np.mean(global_entropy)),
    "mean_amplitude": float(np.mean(amplitude)),
    "mean_orientation": float(np.mean(orientation)),
    "mean_purity": float(np.mean(purities))
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

print("✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
