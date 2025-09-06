# ===================================================================================
# TQE_17_EI_UNIVERSE_SIMULATION_anomaly_LackOfLargeAngleCorrelation.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional, Tuple, List
import os, json, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
try:
    from seeding import load_or_create_run_seeds, universe_rngs
except Exception:
    # Soft fallback if seeding module is not present
    load_or_create_run_seeds = None
    universe_rngs = None

# Optional healpy for CMB-specific operations
try:
    import healpy as hp
except Exception:
    hp = None


# ---------------------------
# Utilities
# ---------------------------
def _rng(seed: Optional[int] = None) -> np.random.Generator:
    """Create a reproducible RNG (PCG64)."""
    return np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()


def _legendre_transform_cl_to_Ctheta(cl: np.ndarray, costheta: np.ndarray) -> np.ndarray:
    """
    Compute C(θ) = sum_{l=0}^{lmax} (2l+1)/(4π) C_l P_l(cos θ) using Legendre polynomials.
    Implemented with numpy.polynomial.legendre.legval (no SciPy dependency).
    """
    from numpy.polynomial.legendre import legval
    l = np.arange(len(cl), dtype=float)
    a = (2.0 * l + 1.0) * cl / (4.0 * np.pi)  # coefficients for Legendre series
    # legval evaluates Σ a_l P_l(x)
    return legval(costheta, a)


def _s_one_half_from_Ctheta(costh_grid: np.ndarray, Cth: np.ndarray, theta_min_deg: float = 60.0) -> float:
    """
    Compute S_{1/2} = ∫_{-1}^{cos θ_min} C(θ)^2 d(cos θ) using trapezoidal rule on a dense grid.
    costh_grid must be monotonically increasing; we integrate over the required sub-interval.
    """
    x = costh_grid
    y2 = Cth * Cth
    x_cut = np.clip(math.cos(np.deg2rad(theta_min_deg)), -1.0, 1.0)
    mask = x <= x_cut
    if not np.any(mask):
        return 0.0
    return float(np.trapz(y2[mask], x[mask]))


def _default_cl(lmax: int = 64) -> np.ndarray:
    """
    Provide a simple, decaying baseline power spectrum (if nothing else is supplied),
    roughly C_l ∝ 1/(l(l+1)) for l >= 2; set C0,C1 to zero to avoid monopole/dipole.
    """
    cl = np.zeros(lmax + 1, dtype=float)
    l = np.arange(2, lmax + 1, dtype=float)
    cl[2:] = 1.0 / (l * (l + 1.0))
    # Normalize to unit variance-ish scale (optional; harmless if left as-is)
    norm = np.sum((2*l + 1) * cl[2:]) / (4*np.pi)
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
    # healpy uses its own RNG; to keep reproducibility, draw unique seeds from our rng
    for i in range(n_maps):
        local_seed = int(rng.integers(0, 2**31 - 1))
        hp.random.seed(local_seed)
        maps[i] = hp.synfast(cl, nside=nside, lmax=len(cl)-1, new=True, verbose=False)
    return maps


def _compute_cl_from_map(m: np.ndarray, lmax: int) -> np.ndarray:
    """Compute C_l from a HEALPix map via hp.anafast; if not available, raise."""
    if hp is None:
        raise RuntimeError("healpy is not available; cannot compute C_l from a map.")
    # Remove monopole/dipole by default to mimic large-angle analyses
    return hp.anafast(m, lmax=lmax, pol=False)


def _save_mirrors(files: List[str], mirrors: List[str], fig_sub: str):
    """Copy generated files to mirror targets; figures go under <mirror>/<fig_sub>/."""
    from shutil import copy2
    for m in mirrors:
        try:
            m_path = pathlib.Path(m)
            for fp in files:
                fp = pathlib.Path(fp)
                if fp.suffix.lower() in {".png", ".jpg"}:
                    tdir = m_path / fig_sub
                    tdir.mkdir(parents=True, exist_ok=True)
                    copy2(fp, tdir / fp.name)
                else:
                    copy2(fp, m_path / fp.name)
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
      - cl_spectrum: array of C_l (length lmax+1). If provided, used for transform and MC.

    Returns:
      dict with CSV/JSON paths, plot paths, and the per-universe dataframe.
    """
    if not active_cfg.get("ANOMALY", {}).get("enabled", True):
        print("[LLAC] ANOMALY.enabled=False → skipping.")
        return {}

    # Resolve outputs and environment
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")

    # Read LLAC controls (we place them under ANOMALY.targets 'lack_large_angle'; add sane defaults)
    a_cfg = active_cfg.get("ANOMALY", {})
    map_cfg = a_cfg.get("map", {})
    targets = a_cfg.get("targets", [])
    # Find or define LLAC block
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
    nside = int(map_cfg.get("resolution_nside", 128))
    lmax = int(llac_cfg.get("lmax", 64))
    theta_min_deg = float(llac_cfg.get("theta_min_deg", 60.0))
    n_mc = int(llac_cfg.get("n_mc", 200))
    p_perc = float(llac_cfg.get("p_percentile", 0.05))  # for boolean flag
    seed_per_map = bool(map_cfg.get("seed_per_map", True))

    # Seeding
    if load_or_create_run_seeds is not None:
        seeds = load_or_create_run_seeds(active_cfg)
        master_seed = seeds["master_seed"]
        uni_seeds = seeds["universe_seeds"]
        rng_master = _rng(master_seed)
    else:
        rng_master = _rng(active_cfg.get("ENERGY", {}).get("seed", None))
        uni_seeds = None

    # Determine number of universes
    N = int(active_cfg["ENERGY"].get("num_universes", 1000))
    if cmb_maps is not None:
        N = int(cmb_maps.shape[0])

    # Prepare power spectrum
    if cl_spectrum is not None:
        cl_in = np.asarray(cl_spectrum, dtype=float)
        lmax_eff = min(lmax, len(cl_in) - 1)
        cl_in = cl_in[:lmax_eff + 1]
    else:
        cl_in = _default_cl(lmax)

    # If no maps provided: synthesize (only when healpy available)
    synthesized = False
    if cmb_maps is None and hp is not None:
        synthesized = True
        # Build per-universe RNGs if requested; else use a single rng
        if seed_per_map and universe_rngs is not None and uni_seeds is not None:
            rngs = universe_rngs(uni_seeds[:N])
            # synth map one-by-one to keep strict reproducibility
            maps_list = []
            for i in range(N):
                local_maps = _make_maps_from_cl(cl_in, nside, rngs[i], 1)
                maps_list.append(local_maps[0])
            cmb_maps = np.vstack(maps_list)
        else:
            cmb_maps = _make_maps_from_cl(cl_in, nside, rng_master, N)

    # Grid for C(θ)
    # Use dense cosine grid for numerically stable S_{1/2} integration
    n_grid = 2048
    costh_grid = np.linspace(-1.0, 1.0, n_grid, dtype=float)
    theta_grid = np.rad2deg(np.arccos(np.clip(costh_grid, -1.0, 1.0)))

    # Compute per-universe C(θ) and S1/2
    S12 = np.empty(N, dtype=float)
    Cth_at_tmin = np.empty(N, dtype=float)
    # We won't store full C(θ) curves for all N (too big). Keep a small panel to plot.
    keep_idx = np.linspace(0, N - 1, num=min(N, 6), dtype=int)
    kept_curves: List[Tuple[int, np.ndarray]] = []

    if cmb_maps is not None and hp is not None:
        # Path: map → C_l → C(θ)
        for i in range(N):
            m = cmb_maps[i]
            cl = _compute_cl_from_map(m, lmax=lmax)
            Cth = _legendre_transform_cl_to_Ctheta(cl, costh_grid)
            S12[i] = _s_one_half_from_Ctheta(costh_grid, Cth, theta_min_deg=theta_min_deg)
            # value at theta_min for quick sanity check
            ccut = math.cos(math.radians(theta_min_deg))
            Cth_at_tmin[i] = float(np.interp(ccut, costh_grid, Cth))
            if i in keep_idx:
                kept_curves.append((i, Cth))
    else:
        # Path: C_l (theoretical or provided) → same C(θ) for all universes
        Cth = _legendre_transform_cl_to_Ctheta(cl_in, costh_grid)
        base_S12 = _s_one_half_from_Ctheta(costh_grid, Cth, theta_min_deg=theta_min_deg)
        base_Ctmin = float(np.interp(math.cos(math.radians(theta_min_deg)), costh_grid, Cth))
        S12[:] = base_S12
        Cth_at_tmin[:] = base_Ctmin
        kept_curves.append((0, Cth))

    # Monte Carlo p-values (optional; requires healpy)
    p_values = np.full(N, np.nan, dtype=float)
    if hp is not None and n_mc > 0:
        # We'll use the same cl_in for null skies to test S1/2 distribution
        # Draw MC using master RNG (reproducible). Costly but manageable at small n_mc.
        mc_rng = _rng(int(rng_master.integers(0, 2**31 - 1)))
        # Precompute null S1/2 distribution
        mc_S = np.empty(n_mc, dtype=float)
        for j in range(n_mc):
            # For speed, synthesize a single sky and compute S1/2
            m = _make_maps_from_cl(cl_in, nside, mc_rng, 1)[0]
            cl = _compute_cl_from_map(m, lmax=lmax)
            Cth_mc = _legendre_transform_cl_to_Ctheta(cl, costh_grid)
            mc_S[j] = _s_one_half_from_Ctheta(costh_grid, Cth_mc, theta_min_deg=theta_min_deg)

        # p-value as fraction of null S below observed (lack-of-correlation = small S)
        # Use rank-based with +1 smoothing
        for i in range(N):
            p_values[i] = float((1 + np.sum(mc_S <= S12[i])) / (1 + n_mc))

        # Boolean flag if in the bottom p_percentile
        flag_llac = (p_values <= p_perc).astype(int)
    else:
        flag_llac = np.zeros(N, dtype=int)  # unknown; keep zeros

    # Build output table
    out_df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "S_half": S12,
        "C_theta_min": Cth_at_tmin,
        "theta_min_deg": float(theta_min_deg),
        "p_value": p_values,
        "llac_flag": flag_llac,
        "used_map": int(cmb_maps is not None),
        "synthesized_map": int(synthesized),
    })

    # Save CSV
    csv_path = run_dir / "LLAC__metrics.csv"
    out_df.to_csv(csv_path, index=False)

    # Summary JSON
    def _stats(x: np.ndarray) -> Dict[str, float]:
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
        "N": int(N),
        "theta_min_deg": float(theta_min_deg),
        "lmax": int(lmax),
        "has_healpy": bool(hp is not None),
        "synthesized": bool(synthesized),
        "S_half": _stats(S12),
        "p_values_summary": _stats(p_values[np.isfinite(p_values)]) if np.isfinite(p_values).any() else None,
        "files": {"csv": str(csv_path)}
    }
    json_path = run_dir / "LLAC__summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    figs: List[str] = []

    # 1) Histogram of S1/2
    plt.figure()
    plt.hist(S12, bins=40)
    plt.xlabel(r"$S_{1/2}$")
    plt.ylabel("count")
    plt.title("Lack of Large-Angle Correlation — $S_{1/2}$")
    f1 = fig_dir / "LLAC__S12_hist.png"
    plt.tight_layout()
    plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.close()
    figs.append(str(f1))

    # 2) A few C(theta) curves
    if len(kept_curves) > 0:
        plt.figure()
        for idx, Cth in kept_curves:
            plt.plot(theta_grid, Cth, linewidth=1.0, alpha=0.85, label=f"u={idx}")
        plt.axvline(theta_min_deg, linestyle="--", alpha=0.6)
        plt.xlabel(r"$\theta$ [deg]")
        plt.ylabel(r"$C(\theta)$")
        plt.title("Two-point correlation C(θ) — samples")
        plt.legend(ncols=2, fontsize=8)
        f2 = fig_dir / "LLAC__Ctheta_samples.png"
        plt.tight_layout()
        plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f2))

    # Mirror copies
    _save_mirrors([str(csv_path), str(json_path), *figs], mirrors, fig_sub)

    print(f"[LLAC] saved CSV/JSON/PNGs under:\n  {run_dir}")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "plots": figs,
        "table": out_df,
    }


# Allow standalone execution
if __name__ == "__main__":
    run_llac(ACTIVE)
