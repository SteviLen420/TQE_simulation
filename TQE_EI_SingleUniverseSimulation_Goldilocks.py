# =====================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Single-universe simulation with Goldilocks principle
# =====================================================================
# Author: Stefan Lengyel
# Description: Stochastic simulation of a single universe model
# Focus: Emergence of law-consistent dynamics in the Goldilocks zone
# Variables: Energy fluctuation (E), Information orientation (I)
# =====================================================================

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
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_UNI_SINGLE"
run_id = time.strftime("TQE_(E,I)_UNI_SINGLE_%Y%m%d_%H%M%S")
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
            noise = np.random.normal(0, noise_scale * 5.0, NUM_STATES)   # stronger noise

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

if lock_in_step is not None:
    plt.axvline(x=lock_in_step, color="purple", linestyle="--", linewidth=2,
                label=f"Lock-in step = {lock_in_step}")

plt.title("TQE Universe Simulation with f(E,I)")
plt.xlabel("Time step")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True)
savefig(os.path.join(FIG_DIR, "entropy_evolution.png"))

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

print("✅ DONE.")
print(f"☁️ All results saved to Google Drive: {SAVE_DIR}")
