# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_17_EI_UNIVERSE_SIMULATION_anomaly_LackOfLargeAngleCorrelation.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script is a scientific analysis module that tests for the "Lack of
# Large-Angle Correlation" (LLAC) anomaly in the simulated CMB maps. This
# anomaly refers to the observed suppression of correlations at large angular
# scales (theta > 60 degrees) in the real sky.
#
# The script's core calculation involves transforming the statistical information
# from each map's power spectrum (C_l) into the two-point angular correlation
# function (C(theta)) via a Legendre transform. It then computes the S_1/2
# statistic, which is the integrated power of C(theta)^2 over large angles. A
# small S_1/2 value indicates a lack of correlation.
#
# To determine if a low S_1/2 value is statistically significant, the script
# performs a Monte Carlo null test. It generates many random skies based on the
# theoretical average to build an expected distribution of S_1/2 values. This
# allows it to assign a p-value to each simulated universe, quantifying the
# rarity of its correlation properties. The full analysis requires the
# healpy library.
#
# ===================================================================================

from typing import Dict, Optional, Tuple, List
import os, json, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + resolved paths (stable run_id within a pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR

# Prefer the project seeding utilities; fall back softly if not present
try:
    from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds, universe_rngs
except Exception:
    load_or_create_run_seeds = None
    universe_rngs = None

# Optional: Healpy for CMB-specific ops (map synthesis, Cl, etc.)
try:
    import healpy as hp
except Exception:
    hp = None


# ---------------------------
# Utilities
# ---------------------------
def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a reproducible NumPy Generator (PCG64)."""
    return np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()


def _legendre_transform_cl_to_Ctheta(cl: np.ndarray, costheta: np.ndarray) -> np.ndarray:
    """
    Compute C(θ) = sum_{l=0}^{lmax} (2l+1)/(4π) C_l P_l(cos θ) using Legendre polynomials.
    Implemented with numpy.polynomial.legendre.legval (no SciPy dependency).
    """
    from numpy.polynomial.legendre import legval
    l = np.arange(len(cl), dtype=float)
    coeffs = (2.0 * l + 1.0) * cl / (4.0 * np.pi)  # coefficients for Σ a_l P_l(x)
    return legval(costheta, coeffs)


def _s_one_half_from_Ctheta(costh_grid: np.ndarray, Cth: np.ndarray, theta_min_deg: float = 60.0) -> float:
    """
    Compute S_{1/2} = ∫_{-1}^{cos θ_min} C(θ)^2 d(cos θ) using a trapezoidal rule.
    costh_grid must be monotonically increasing in [-1, 1].
    """
    x = costh_grid
    y2 = Cth * Cth
    x_cut = float(np.clip(np.cos(np.deg2rad(theta_min_deg)), -1.0, 1.0))
    mask = x <= x_cut  # integrate from -1 up to cos(theta_min)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(y2[mask], x[mask]))


def _default_cl(lmax: int = 64) -> np.ndarray:
    """
    Simple decaying baseline C_l (∝ 1/[l(l+1)] for l>=2); monopole/dipole set to 0.
    Light normalization to ~unit variance.
    """
    cl = np.zeros(lmax + 1, dtype=float)
    l = np.arange(2, lmax + 1, dtype=float)
    cl[2:] = 1.0 / (l * (l + 1.0))
    norm = np.sum((2 * l + 1) * cl[2:]) / (4 * np.pi)
    if norm > 0:
        cl[2:] /= norm
    return cl


def _make_maps_from_cl(cl: np.ndarray, nside: int, rng: np.random.Generator, n_maps: int) -> np.ndarray:
    """
    Generate Gaussian isotropic skies with given power spectrum (healpy required).
    Returns array of shape (n_maps, npix).
    """
    if hp is None:
        raise RuntimeError("healpy is not available; cannot synthesize CMB maps.")
    npix = hp.nside2npix(nside)
    maps = np.empty((n_maps, npix), dtype=float)
    # healpy.synfast uses numpy.random under the hood → seed NumPy's global RNG for reproducibility
    for i in range(n_maps):
        local_seed = int(rng.integers(0, 2**31 - 1))
        np.random.seed(local_seed)
        maps[i] = hp.synfast(cl, nside=nside, lmax=len(cl) - 1, new=True, verbose=False)
    return maps


def _compute_cl_from_map(m: np.ndarray, lmax: int) -> np.ndarray:
    """Compute C_l from a HEALPix map with hp.anafast; monopole/dipole removed by default."""
    if hp is None:
        raise RuntimeError("healpy is not available; cannot compute C_l from a map.")
    return hp.anafast(m, lmax=lmax, pol=False)


def _save_mirrors(files: List[str], mirrors: List[str], fig_sub: str):
    """Copy generated files to mirror targets; figures go under <mirror>/<fig_sub>/."""
    from shutil import copy2
    for m in mirrors or []:
        try:
            m_path = pathlib.Path(m)
            m_path.mkdir(parents=True, exist_ok=True)
            for fp in files:
                p = pathlib.Path(fp)
                if p.suffix.lower() in {".png", ".jpg"}:
                    tdir = m_path / fig_sub
                    tdir.mkdir(parents=True, exist_ok=True)
                    copy2(p, tdir / p.name)
                else:
                    copy2(p, m_path / p.name)
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")


# ---------------------------
# Public API
# ---------------------------
def run_llac(active_cfg: Dict = ACTIVE,
             cmb_maps: Optional[np.ndarray] = None,
             cl_spectrum: Optional[np.ndarray] = None) -> Dict:
    """
    Detect 'Lack of Large-Angle Correlation' (LLAC) on CMB skies.

    Inputs (optional):
      - cmb_maps: array of shape (N, npix) in HEALPix RING ordering. If provided and healpy is
                  present, we compute C_l from each map. If not provided, we synthesize skies.
      - cl_spectrum: array of C_l (length lmax+1) used for C(θ) transform and MC nulls.

    Outputs:
      - CSV/JSON in run dir (tagged with EI/E), PNGs in figs dir (+ mirrors), and a result dict.
    """
    # Stage toggle
    if not active_cfg.get("ANOMALY", {}).get("enabled", True):
        print("[LLAC] ANOMALY.enabled=False → skipping.")
        return {}

    # Use cached paths (keeps run_id consistent with the whole pipeline)
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(FIG_DIR); fig_dir.mkdir(parents=True, exist_ok=True)
    mirrors = paths.get("mirrors", [])
    # Safe fetch of figure subdir (avoid KeyError if OUTPUTS/local missing)
    fig_sub = active_cfg.get("OUTPUTS", {}).get("local", {}).get("fig_subdir", "figs")

    # EI/E filename tag
    tag = "EI" if active_cfg["PIPELINE"].get("use_information", True) else "E"
    dpi = int(active_cfg["RUNTIME"].get("matplotlib_dpi", 180))

    # Read LLAC controls from config
    a_cfg   = active_cfg.get("ANOMALY", {})
    map_cfg = a_cfg.get("map", {})
    targets = a_cfg.get("targets", [])
    llac_cfg = None
    for t in targets:
        if t.get("name") in {"lack_large_angle", "llac"}:
            llac_cfg = t
            break
    if llac_cfg is None:
        llac_cfg = {"name": "lack_large_angle", "enabled": True, "theta_min_deg": 60.0,
                    "lmax": 64, "n_mc": 200, "p_percentile": 0.05}
    if not llac_cfg.get("enabled", True):
        print("[LLAC] target disabled → skipping.")
        return {}

    # Parameters
    nside         = int(map_cfg.get("resolution_nside", 128))
    lmax          = int(llac_cfg.get("lmax", 64))
    theta_min_deg = float(llac_cfg.get("theta_min_deg", 60.0))
    n_mc          = int(llac_cfg.get("n_mc", 200))
    p_perc        = float(llac_cfg.get("p_percentile", 0.05))
    seed_per_map  = bool(map_cfg.get("seed_per_map", True))

    # Seeding
    if load_or_create_run_seeds is not None:
        seeds       = load_or_create_run_seeds(active_cfg)
        master_seed = seeds.get("master_seed", 1234567)
        uni_seeds   = seeds.get("universe_seeds", [])
        rng_master  = _rng(master_seed)
    else:
        rng_master  = _rng(active_cfg.get("ENERGY", {}).get("seed", None))
        uni_seeds   = []

    # Number of universes
    N = int(active_cfg["ENERGY"].get("num_universes", 1000))
    if cmb_maps is not None:
        N = int(cmb_maps.shape[0])

    # Power spectrum (input or default)
    if cl_spectrum is not None:
        cl_in = np.asarray(cl_spectrum, dtype=float)
        lmax_eff = min(lmax, len(cl_in) - 1)
        cl_in = cl_in[:lmax_eff + 1]
    else:
        cl_in = _default_cl(lmax)

    # If no maps provided but healpy is available → synthesize
    synthesized = False
    if cmb_maps is not None and hp is None:
        print("[LLAC] healpy not available …")
        cl  = _compute_cl_from_map(m, lmax=lmax)  # <-- ez hp nélkül kivételt dob
        synthesized = True
        if seed_per_map and universe_rngs is not None and len(uni_seeds) > 0:
            rngs = universe_rngs(uni_seeds[:N])
            maps_list = []
            for i in range(N):
                maps_list.append(_make_maps_from_cl(cl_in, nside, rngs[i], 1)[0])
            cmb_maps = np.vstack(maps_list)
        else:
            cmb_maps = _make_maps_from_cl(cl_in, nside, rng_master, N)

    # Dense grid in cos(theta) for stable S_{1/2} integration
    n_grid     = 2048
    costh_grid = np.linspace(-1.0, 1.0, n_grid, dtype=float)
    theta_grid = np.rad2deg(np.arccos(np.clip(costh_grid, -1.0, 1.0)))

    # Per-universe metrics
    S12 = np.empty(N, dtype=float)
    Cth_at_tmin = np.empty(N, dtype=float)
    # Pick up to 6 evenly spaced, unique indices for plotting C(theta)
    keep_idx = np.unique(np.linspace(0, max(N - 1, 0), num=min(N, 6), dtype=int))
    kept_curves: List[Tuple[int, np.ndarray]] = []

    # Warn if maps are provided but healpy is unavailable → fall back to theoretical Cl
    if cmb_maps is not None and hp is None:
        print("[LLAC] healpy not available → cannot compute C_l from provided maps; using theoretical C_l for all universes.")
        # Path: map → C_l → C(θ)
        for i in range(N):
            m   = cmb_maps[i]
            cl  = _compute_cl_from_map(m, lmax=lmax)
            Cth = _legendre_transform_cl_to_Ctheta(cl, costh_grid)
            S12[i] = _s_one_half_from_Ctheta(costh_grid, Cth, theta_min_deg=theta_min_deg)
            ccut = math.cos(math.radians(theta_min_deg))
            Cth_at_tmin[i] = float(np.interp(ccut, costh_grid, Cth))
            if i in keep_idx:
                kept_curves.append((i, Cth))
    else:
        # Path: theoretical C_l → same C(θ) for all universes (no map/HP)
        Cth = _legendre_transform_cl_to_Ctheta(cl_in, costh_grid)
        base_S12 = _s_one_half_from_Ctheta(costh_grid, Cth, theta_min_deg=theta_min_deg)
        base_Ctmin = float(np.interp(math.cos(math.radians(theta_min_deg)), costh_grid, Cth))
        S12[:] = base_S12
        Cth_at_tmin[:] = base_Ctmin
        kept_curves.append((0, Cth))

    # Monte Carlo null (optional; requires healpy)
    p_values = np.full(N, np.nan, dtype=float)
    if hp is not None and n_mc > 0:
        mc_rng = _rng(int(rng_master.integers(0, 2**31 - 1)))
        mc_S = np.empty(n_mc, dtype=float)
        for j in range(n_mc):
            m = _make_maps_from_cl(cl_in, nside, mc_rng, 1)[0]
            cl = _compute_cl_from_map(m, lmax=lmax)
            Cth_mc = _legendre_transform_cl_to_Ctheta(cl, costh_grid)
            mc_S[j] = _s_one_half_from_Ctheta(costh_grid, Cth_mc, theta_min_deg=theta_min_deg)
        # p-value: fraction of null S below observed (lack-of-correlation = small S)
        for i in range(N):
            p_values[i] = float((1 + np.sum(mc_S <= S12[i])) / (1 + n_mc))

        flag_llac = (p_values <= p_perc).astype(int)
    else:
        flag_llac = np.zeros(N, dtype=int)  # unknown without nulls

    # Build output table
    used_map_flag = int(cmb_maps is not None and hp is not None)
    out_df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "S_half": S12,
        "C_theta_min": Cth_at_tmin,
        "theta_min_deg": float(theta_min_deg),
        "p_value": p_values,
        "llac_flag": flag_llac,
        "used_map": used_map_flag,
        "synthesized_map": int(synthesized),
    })

    # Tagged outputs
    csv_path  = run_dir / f"{tag}__LLAC_metrics.csv"
    json_path = run_dir / f"{tag}__LLAC_summary.json"
    out_df.to_csv(csv_path, index=False)

    # Summary JSON
    def _stats(x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        if x.size == 0 or not np.isfinite(x).any():
            return {
                "min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "std": float("nan"),
                "p25": float("nan"), "median": float("nan"), "p75": float("nan"),
            }
        x = x[np.isfinite(x)]
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
        }
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": int(N),
        "theta_min_deg": float(theta_min_deg),
        "lmax": int(lmax),
        "has_healpy": bool(hp is not None),
        "synthesized": bool(synthesized),
        "S_half": _stats(S12),
        "p_values_summary": _stats(p_values[np.isfinite(p_values)]) if np.isfinite(p_values).any() else None,
        "files": {"csv": str(csv_path), "json": str(json_path)},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots (tagged)
    figs: List[str] = []

    # 1) Histogram of S1/2
    plt.figure()
    plt.hist(S12, bins=40)
    plt.xlabel(r"$S_{1/2}$"); plt.ylabel("count"); plt.title("Lack of Large-Angle Correlation — $S_{1/2}$")
    f1 = fig_dir / f"{tag}__LLAC_S12_hist.png"
    plt.tight_layout(); plt.savefig(f1, dpi=dpi); plt.close()
    figs.append(str(f1))

    # 2) A few C(theta) curves
    if len(kept_curves) > 0:
        plt.figure()
        for idx, Cth in kept_curves:
            plt.plot(theta_grid, Cth, linewidth=1.0, alpha=0.85, label=f"u={idx}")
        plt.axvline(theta_min_deg, linestyle="--", alpha=0.6)
        plt.xlabel(r"$\theta$ [deg]"); plt.ylabel(r"$C(\theta)$"); plt.title("Two-point correlation C(θ) — samples")
        plt.legend(ncols=2, fontsize=8)
        f2 = fig_dir / f"{tag}__LLAC_Ctheta_samples.png"
        plt.tight_layout(); plt.savefig(f2, dpi=dpi); plt.close()
        figs.append(str(f2))

    # Mirror copies (CSV/JSON to mirror root; PNGs to <mirror>/<fig_subdir>/)
    _save_mirrors([str(csv_path), str(json_path), *figs], mirrors, fig_sub)

    print(f"[LLAC] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")
    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": out_df}


# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_anomaly_llac_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")
    return run_llac(active_cfg=cfg, **kwargs)
    
if __name__ == "__main__":
    run_anomaly_llac_stage(ACTIVE)
