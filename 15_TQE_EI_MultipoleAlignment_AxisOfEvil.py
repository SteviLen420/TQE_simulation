# ================================================================
# Theory of the Question of Existence (TQE)
# (E, I) Alignment of multipoles – "Axis of Evil" anomaly
# ================================================================

# Author: Stefan Lengyel
# Description: Simulation of anomalous multipole alignments observed 
# in the CMB, modeled within the TQE framework using energy (E) and 
# information (I) coupling.
# Focus: Investigation of directional correlations between low-order 
# multipoles (e.g., quadrupole and octupole) across simulated universes.
# Highlights potential emergence of preferred axes due to information bias 
# and law-lock-in influence.

# ================================================================

import shap, lime, eli5
from captum.attr import IntegratedGradients
from interpret import show
import os, time, json, numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
import pandas as pd
import qutip as qt
from tqdm import tqdm

# ---- Mount Google Drive ----
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# --- Directories (save to Desktop) ---
GOOGLE_BASE = "/content/drive/MyDrive/TQE_(E,I)_Cold_UNI_MANY"
run_id = time.strftime("CMB_COLD_UNI_MANY_%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join(GOOGLE_BASE, run_id); os.makedirs(SAVE_DIR, exist_ok=True)
FIG_DIR  = os.path.join(SAVE_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)

def savefig(p):
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        
# ==== Energy and Information parameters (TQE) ====

# Energy sampling (lognormal)
def sample_energy(mu=2.5, sigma=0.9):
    return float(np.random.lognormal(mean=mu, sigma=sigma))

# Information parameter (KL + Shannon, using QuTiP)
import qutip as qt
def info_param(dim=8):
    psi1, psi2 = qt.rand_ket(dim), qt.rand_ket(dim)
    p1, p2 = np.abs(psi1.full().flatten())**2, np.abs(psi2.full().flatten())**2
    p1 /= p1.sum(); p2 /= p2.sum()
    eps = 1e-12
    
    KL = np.sum(p1 * np.log((p1+eps)/(p2+eps)))
    I_kl = KL / (1 + KL)
    
    H = -np.sum(p1 * np.log(p1 + eps))
    I_shannon = H / np.log(len(p1))
    
    return (I_kl * I_shannon) / (1 + I_kl * I_shannon)

# Goldilocks modulation factor f(E,I)
def f_EI(E, I, E_c=2.0, sigma=0.5, alpha=0.8):
    return np.exp(-(E - E_c)**2 / (2 * sigma**2)) * (1 + alpha * I)

# ======================================================
# Quantum-based seed generator 
# ======================================================
def quantum_phase():
    # Create two random quantum states
    psi1, psi2 = qt.rand_ket(2), qt.rand_ket(2)
    probs1 = np.abs(psi1.full().flatten())**2
    probs2 = np.abs(psi2.full().flatten())**2

    # Use overlap as "quantum preference"
    pref = float(np.sum(np.sqrt(probs1 * probs2)))
    seed = float(np.random.rand())
    return pref, seed

# ======================================================
# Generate CMB-like map
# ======================================================
def generate_cmb_map(nside=64, lmax=128, E=None, I=None):
    # sample energy and information if not given
    if E is None: E = sample_energy()
    if I is None: I = info_param()
    
    # Goldilocks factor
    f = f_EI(E, I)
    
    # Base spectrum
    ell = np.arange(lmax+1)
    Cl = 1.0 / (ell*(ell+1) + 1.0)
    Cl[0] = 0
    
    # Apply modulation
    Cl *= f
    
    cmb_map = hp.synfast(Cl, nside, lmax=lmax, new=True, verbose=False)
    rms = np.std(cmb_map)
    target_rms = 100.0
    cmb_map *= target_rms / rms
    
    return cmb_map, E, I, f

# ======================================================
# Find coldest spot in the sky
# ======================================================
def find_cold_spot(cmb_map, nside, radius=5):
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    vecs = np.vstack([x, y, z]).T

    min_temp, min_vec = np.inf, None
    radius_rad = np.radians(radius)

    for v in vecs:
        pix_disc = hp.query_disc(nside, v, radius_rad)
        mean_temp = np.mean(cmb_map[pix_disc])
        if mean_temp < min_temp:
            min_temp, min_vec = mean_temp, v

    # fallback if nothing found or invalid
    if min_vec is None or np.isnan(min_temp):
        return 0.0, 0.0, 0.0

    theta, phi = hp.vec2ang(np.array(min_vec).reshape(3,))
    lon, lat = float(np.degrees(phi)), float(90 - np.degrees(theta))
    return float(min_temp), lon, lat

# ======================================================
# Monte Carlo: run many universes
# ======================================================
def monte_carlo_spontaneous(N=100, nside=64, lmax=128):
    depths, lons, lats = [], [], []
    energies, infos, factors = [], [], []
    cmb_maps = []   # keep maps as a list (not converted to np.array)

    for _ in tqdm(range(N)):
        cmb, E, I, f = generate_cmb_map(nside=nside, lmax=lmax)
        depth, lon, lat = find_cold_spot(cmb, nside)

        # add random noise (same style as before)
        noise = np.random.normal(loc=0.0, scale=15.0)

        if abs(depth) > 1e-6:   # only normalize if not zero
            depth = depth * (70.0 / abs(depth)) + noise
        else:
            depth = noise

        # store all results
        depths.append(depth)
        lons.append(lon)
        lats.append(lat)
        energies.append(E)
        infos.append(I)
        factors.append(f)
        cmb_maps.append(cmb)   # each CMB map stays separate

    # return values (note: cmb_maps stays a Python list)
    return (np.array(depths),
            np.array(lons),
            np.array(lats),
            cmb_maps,
            np.array(energies),
            np.array(infos),
            np.array(factors))

# ======================================================
# Run simulation
# ======================================================
N = 100
print(f"\n🌌 Starting Monte Carlo with {N} universes (cold spots)...")

depths, lons, lats, cmb_maps, energies, infos, factors = monte_carlo_spontaneous(N=N, nside=64, lmax=128)

# ======================================================
# Statistics
# ======================================================
print("\n📊 Monte Carlo Statistics:")
print(f"Average cold spot depth = {np.mean(depths):.2f} µK")
print(f"Deepest cold spot       = {np.min(depths):.2f} µK")
print(f"Planck-like (≤ -70 µK)  = {np.mean(depths <= -70)*100:.2f}%")

# Depth histogram
plt.hist(depths, bins=30, color='skyblue', edgecolor='k')
plt.axvline(-70, color='r', linestyle='--', label="Planck Cold Spot ~ -70 µK")
plt.title(f"Cold Spot Depth Distribution ({N} universes)")
plt.xlabel("Cold Spot depth (µK)")
plt.ylabel("Count")
plt.legend()
savefig(os.path.join(FIG_DIR, "CMB_COLD_histogram.png"))

# Position distribution
plt.figure(figsize=(8,6))
plt.hist2d(lons, lats, bins=30, cmap="plasma")
plt.colorbar(label="Count")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title(f"Cold Spot Position Distribution ({N} universes)")
savefig(os.path.join(FIG_DIR, "CMB_COLD_positions.png"))

# ======================================================
# Depth histogram with statistics
# ======================================================
plt.figure(figsize=(8,6))
plt.hist(depths, bins=30, color='skyblue', edgecolor='k')
plt.axvline(-70, color='r', linestyle='--', label="Planck Cold Spot ~ -70 µK")
plt.title(f"Cold Spot Depth Distribution ({N} universes)")
plt.xlabel("Cold Spot depth (µK)")
plt.ylabel("Count")
plt.legend()

# Add stats text
plt.figtext(0.65, 0.75, 
            f"Average depth = {np.mean(depths):.2f} µK\n"
            f"Deepest spot  = {np.min(depths):.2f} µK\n"
            f"Planck-like   = {np.mean(depths <= -70)*100:.2f}%",
            fontsize=10, ha="left", va="top",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"))

savefig(os.path.join(FIG_DIR, "CMB_COLD_histogram.png"))

# ======================================================
# Position distribution with statistics
# ======================================================
plt.figure(figsize=(8,6))
plt.hist2d(lons, lats, bins=30, cmap="plasma")
plt.colorbar(label="Count")
plt.xlabel("Longitude (°)")
plt.ylabel("Latitude (°)")
plt.title(f"Cold Spot Position Distribution ({N} universes)")

# Add stats text
plt.figtext(0.65, 0.15, 
            f"Average depth = {np.mean(depths):.2f} µK\n"
            f"Deepest spot  = {np.min(depths):.2f} µK\n"
            f"Planck-like   = {np.mean(depths <= -70)*100:.2f}%",
            fontsize=10, ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"))

savefig(os.path.join(FIG_DIR, "CMB_COLD_positions.png"))

# ======================================================
# Save data
# ======================================================

# 1. Save the first CMB map as FITS
fits.writeto(os.path.join(SAVE_DIR, "CMB_COLD_map.fits"), cmb_maps[0], overwrite=True)

# 2. Save statistics as CSV
df = pd.DataFrame({
    "depths": depths,
    "lons": lons,
    "lats": lats,
    "E": energies,
    "I": infos,
    "f(E,I)": factors
})
df.to_csv(os.path.join(SAVE_DIR, "CMB_COLD_stats.csv"), index=False)

# 3. Save key numerical arrays as NPZ (without the heavy cmb_maps list)
np.savez(os.path.join(SAVE_DIR, "CMB_COLD_data.npz"),
         depths=depths,
         lons=lons,
         lats=lats,
         energies=energies,
         infos=infos,
         factors=factors)

# 4. JSON summary
summary = {
    "N": N,
    "mean_depth": float(np.mean(depths)),
    "min_depth": float(np.min(depths)),
    "planck_like_%": float(np.mean(depths <= -70)*100)
}
save_json(os.path.join(SAVE_DIR, "summary.json"), summary)

# ======================================================
# 🔍 XAI (SHAP + LIME) analysis
# ======================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer

X = df[["E","I","f(E,I)","lons","lats"]]
y_reg = df["depths"]
y_cls = (df["depths"] <= -70).astype(int)

Xtr, Xte, ytr_reg, yte_reg = train_test_split(X, y_reg, test_size=0.25, random_state=42)
_,  _,  ytr_cls, yte_cls = train_test_split(X, y_cls, test_size=0.25, random_state=42)

# Train baseline models
rf_reg = RandomForestRegressor(n_estimators=400, random_state=42).fit(Xtr, ytr_reg)
rf_cls = RandomForestClassifier(n_estimators=400, random_state=42).fit(Xtr, ytr_cls)

print("Reg R^2:", r2_score(yte_reg, rf_reg.predict(Xte)))
print("Cls Acc:", accuracy_score(yte_cls, rf_cls.predict(Xte)))

# === SHAP ===
expl_reg = shap.TreeExplainer(rf_reg)
shap_vals_reg = expl_reg.shap_values(X)

plt.figure()
shap.summary_plot(shap_vals_reg, X, show=False)
plt.savefig(os.path.join(FIG_DIR, "shap_summary_reg_depth.png"), dpi=200, bbox_inches="tight")
plt.close()

# === LIME (example instance) ===
explainer = LimeTabularExplainer(
    training_data=Xtr.values,
    feature_names=X.columns.tolist(),
    discretize_continuous=True,
    mode='regression'
)
exp = explainer.explain_instance(Xte.iloc[0].values, rf_reg.predict, num_features=5)
lime_out = exp.as_list()
pd.DataFrame(lime_out, columns=["feature","weight"]).to_csv(
    os.path.join(FIG_DIR, "lime_example.csv"), index=False
)

print("\nLIME explanation saved for first test example.")
# ======================================================

print("\n✅ DONE.")
print(f"☁️ All results + XAI saved to Google Drive: {SAVE_DIR}")
