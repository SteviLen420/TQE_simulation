# ===================================================================================
# 13_TQE_EI_UNIVERSE_SIMULATION_cmb_map_generation.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
from seeding import load_or_create_run_seeds, universe_rngs

import os, json, math, pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Math helpers (no SciPy)
# ---------------------------
def _gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    """Return a normalized 1D Gaussian kernel of std=sigma (pixels)."""
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
    Isotropic spectral shaping: multiply FFT by (1 / k^alpha) radial filter.
    alpha=0 → white; alpha in [0.5..2] gives redder fields (CMB-like large-scale power).
    """
    if alpha <= 0:
        return field

    ny, nx = field.shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    # Avoid singularity at k=0: set gain(0) = gain at nearest nonzero bin
    eps = 1e-9
    gain = 1.0 / np.maximum(k, eps) ** alpha
    # Normalize filter energy to keep overall variance in a reasonable range
    gain /= np.sqrt(np.mean(gain * gain))

    F = np.fft.fft2(field)
    Ff = F * gain
    shaped = np.fft.ifft2(Ff).real
    return shaped

def _fwhm_deg_to_sigma_pix(fwhm_deg: float, nside: int) -> float:
    """
    Convert Gaussian beam FWHM (deg) → sigma in pixels, assuming the square spans ~180°.
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
      - start white Gaussian noise
      - apply 1/k^alpha spectral shaping (optional)
      - apply Gaussian beam smoothing
      - normalize (mean=0, std=1)
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
    Generate a CMB-like map per universe and save .npy (and sampled PNG previews).

    Inputs:
      - active_cfg: config dict (uses ANOMALY.map.* for nside/beam; optional ANOMALY.psd_alpha)
      - arrays: optional, currently unused (kept for future joins)
      - preview_max: cap on the number of PNG previews (default: min(N, 24))

    Returns:
      dict(csv, json, previews, table)
    """
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    # EI/E filename tag
    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Config
    an_map = active_cfg["ANOMALY"]["map"]
    nside = int(an_map.get("resolution_nside", 128))
    beam_fwhm_deg = float(an_map.get("beam_fwhm_deg", 1.0))
    beam_sigma_pix = _fwhm_deg_to_sigma_pix(beam_fwhm_deg, nside)

    # optional spectral shaping
    psd_alpha = float(active_cfg["ANOMALY"].get("psd_alpha", 1.0))  # 0=white, 1..2 redder

    # Seeds
    seeds = load_or_create_run_seeds(active_cfg)
    rngs = universe_rngs(seeds["universe_seeds"])

    N = int(active_cfg["ENERGY"]["num_universes"])
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
            plt.imshow(sky, origin="lower", interpolation="nearest")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"CMB-like map (u={i})")
            p = fig_dir / f"{tag}__cmb_map_u{i}.png"
            plt.tight_layout()
            plt.savefig(p, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
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

    # Mirroring
    from shutil import copy2
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
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


# Standalone
if __name__ == "__main__":
    run_cmb_map_generation(ACTIVE)
