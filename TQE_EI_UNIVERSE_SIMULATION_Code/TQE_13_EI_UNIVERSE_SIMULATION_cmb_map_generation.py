# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_13_EI_UNIVERSE_SIMULATION_cmb_map_generation.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script functions as an observational data synthesizer. Its primary purpose is
# to generate a synthetic, physically-motivated Cosmic Microwave Background (CMB)
# map for each universe produced by the simulation. This stage is crucial for
# bridging the gap between the theoretical model and real-world astronomical
# observations.
#
# The synthesis follows a standard procedure for creating mock cosmological maps:
# 1. It begins with a field of Gaussian white noise, unique to each universe via
#    its dedicated random seed.
# 2. It applies a spectral shaping filter in Fourier space to imbue the map
#    with realistic large-scale correlations (a 1/k^alpha power spectrum).
# 3. It smooths the result with a Gaussian blur to simulate the finite
#    resolution of a telescope's "beam."
#
# The generated maps are mock "observables" that subsequent stages can analyze for
# features like anomalies. The full map data for every universe is saved as a
# .npy file, while a limited number of .png visual previews are also created.
#
# ===================================================================================

from typing import Dict, Optional
import os, json, math, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + paths (stable run_id within a pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds

# Try to import universe_rngs; if not available, build RNGs from seeds locally
try:
    from TQE_04_EI_UNIVERSE_SIMULATION_seeding import universe_rngs  # preferred
except Exception:
    universe_rngs = None


# ---------------------------
# Math helpers (no SciPy)
# ---------------------------
def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Return a normalized 1D Gaussian kernel with standard deviation `sigma` in pixels."""
    if sigma <= 0:
        return np.array([1.0], dtype=float)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    s = k.sum()
    if s > 0:
        k /= s
    return k

def _gaussian_blur2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (dependency-free)."""
    if sigma <= 0:
        return img
    k = _gaussian_kernel1d(sigma)
    pad = len(k) // 2
    # horizontal pass
    tmp = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    out = np.empty_like(img)
    for i in range(img.shape[0]):
        out[i] = np.convolve(tmp[i], k, mode="valid")
    # vertical pass
    tmp2 = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
    out2 = np.empty_like(out)
    for j in range(out.shape[1]):
        out2[:, j] = np.convolve(tmp2[:, j], k, mode="valid")
    return out2

def _spectral_shaping_white_to_1overk(field: np.ndarray, alpha: float) -> np.ndarray:
    """
    Isotropic spectral shaping: multiply FFT by a radial (1 / k^alpha) filter.
    alpha=0 → white; alpha in [~0.5..2] makes the field redder (more large-scale power).
    """
    if alpha <= 0:
        return field

    ny, nx = field.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    eps = 1e-9  # avoid division by zero at k=0
    gain = 1.0 / np.maximum(k, eps) ** alpha
    # Normalize filter energy to keep overall variance reasonable
    gain /= np.sqrt(np.mean(gain * gain))

    F = np.fft.fft2(field)
    Ff = F * gain
    shaped = np.fft.ifft2(Ff).real
    return shaped

def _fwhm_deg_to_sigma_pix(fwhm_deg: float, nside: int) -> float:
    """
    Convert Gaussian beam FWHM (degrees) to sigma in pixels, assuming the square spans ~180°.
    """
    fwhm_pix = (nside / 180.0) * float(fwhm_deg)
    sigma_pix = fwhm_pix / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return float(sigma_pix)


# ---------------------------
# Core generator
# ---------------------------
def _synthesize_cmb_proxy(nside: int,
                          rng: np.random.Generator,
                          psd_alpha: float,
                          beam_sigma_pix: float) -> np.ndarray:
    """
    Build a CMB-like map:
      1) white Gaussian noise
      2) optional 1/k^alpha spectral shaping
      3) Gaussian beam smoothing
      4) normalize to mean=0, std=1
    """
    sky = rng.normal(0.0, 1.0, size=(nside, nside)).astype(np.float64)
    if psd_alpha > 0:
        sky = _spectral_shaping_white_to_1overk(sky, psd_alpha)
    if beam_sigma_pix > 0:
        sky = _gaussian_blur2d(sky, beam_sigma_pix)
    sky -= np.mean(sky)
    std = np.std(sky)
    if std > 0:
        sky /= std
    return sky


# ---------------------------
# Public API
# ---------------------------
def run_cmb_map_generation(active_cfg: Dict = ACTIVE,
                           arrays: Optional[Dict[str, np.ndarray]] = None,
                           preview_max: Optional[int] = None) -> Dict:
    """
    Generate a CMB-like map per universe and save .npy (plus a limited number of PNG previews).

    Inputs:
      - active_cfg: config dict (reads ANOMALY.map.* for resolution/beam; ANOMALY.psd_alpha optional)
      - arrays: optional, currently unused (reserved for future joins)
      - preview_max: cap on the number of PNG previews (default: min(N, 24))

    Returns:
      dict(csv, json, previews, table)
    """
    # Use cached paths to keep run_id consistent with the rest of the pipeline
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths.get("mirrors", [])

    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # EI/E filename tag
    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Config
    an_map = active_cfg["ANOMALY"]["CMB_MAP"]
    nside = int(an_map.get("resolution_nside", 128))
    beam_fwhm_deg = float(an_map.get("beam_fwhm_deg", 1.0))
    beam_sigma_pix = _fwhm_deg_to_sigma_pix(beam_fwhm_deg, nside)

    # optional spectral shaping
    psd_alpha = float(active_cfg["ANOMALY"].get("psd_alpha", 1.0))  # 0=white, 1..2=redder

    # Seeds → one RNG per universe
    seeds = load_or_create_run_seeds(active_cfg)
    uni_seeds = seeds.get("universe_seeds", [])
    if universe_rngs is not None:
        rngs = universe_rngs(uni_seeds)
    else:
        # Fallback: build RNGs locally from the provided seeds
        if not isinstance(uni_seeds, (list, tuple, np.ndarray)) or len(uni_seeds) == 0:
            uni_seeds = [int(seeds.get("master_seed", 1234567))]
        rngs = [np.random.default_rng(int(s)) for s in uni_seeds]

    N = int(active_cfg["ENERGY"]["num_universes"])
    # Make sure we have at least N RNGs
    if len(rngs) < N:
        # deterministically extend with offsets of master seed
        base = int(seeds.get("master_seed", 1234567))
        for i in range(len(rngs), N):
            rngs.append(np.random.default_rng(base + 10007 * i))

    preview_cap = min(N, 24) if preview_max is None else int(preview_max)

    rows = []
    previews = []

    for i in range(N):
        rng = rngs[i]
        sky = _synthesize_cmb_proxy(nside, rng, psd_alpha, beam_sigma_pix)

        # Save .npy map
        npy_path = run_dir / f"{tag}__cmb_map_u{i}.npy"
        np.save(npy_path, sky)

        rows.append({
            "universe_id": i,
            "nside": nside,
            "beam_fwhm_deg": beam_fwhm_deg,
            "beam_sigma_pix": beam_sigma_pix,
            "psd_alpha": psd_alpha,
            "map_path": str(npy_path),
        })

        # Limited previews to keep disk footprint reasonable
        if len(previews) < preview_cap:
            plt.figure()
            # fixed color scale for comparability across universes
            plt.imshow(sky, origin="lower", interpolation="nearest", vmin=-3, vmax=3, cmap="coolwarm")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"CMB-like map (u={i})")
            p = fig_dir / f"{tag}__cmb_map_u{i}.png"
            plt.tight_layout()
            plt.savefig(p, dpi=active_cfg["RUNTIME"].get("matplotlib_dpi", 180))
            plt.close()
            previews.append(str(p))

    # CSV
    df = pd.DataFrame(rows)
    csv_path = run_dir / f"{tag}__cmb_maps.csv"
    df.to_csv(csv_path, index=False)

    # Summary JSON
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": N,
        "nside": nside,
        "beam_fwhm_deg": beam_fwhm_deg,
        "beam_sigma_pix": beam_sigma_pix,
        "psd_alpha": psd_alpha,
        "preview_count": len(previews),
        "files": {"csv": str(csv_path), "previews": previews},
    }
    json_path = run_dir / f"{tag}__cmb_maps_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Mirroring (CSV/JSON to mirror root; PNGs to <mirror>/<fig_subdir>/)
    from shutil import copy2
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            if previews:
                m_fig = pathlib.Path(m) / fig_sub
                m_fig.mkdir(parents=True, exist_ok=True)
                for fp in previews:
                    copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[CMB] Generated {N} maps @ nside={nside}, beam={beam_fwhm_deg}° (alpha={psd_alpha}).")
    return {"csv": str(csv_path), "json": str(json_path), "previews": previews, "table": df}

# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_cmb_map_generation_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_cmb_map_generation(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_cmb_map_generation_stage(ACTIVE)
