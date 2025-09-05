# 18_TQE_(E,I)_UNIVERSE_SIMULATION_anomaly_HemisphericalAsymmetry.py
# ===================================================================================
# Hemispherical Power Asymmetry (HPA) anomaly detector for the TQE pipeline
# -----------------------------------------------------------------------------------
# - Splits each CMB sky into two hemispheres along a fixed axis (default: z-axis).
# - Computes total power in each hemisphere from C_l or directly from maps.
# - Reports asymmetry ratio (max/min) as metric.
# - (If healpy available) estimates Monte Carlo p-values with isotropic null skies.
# - Saves CSV (metrics per universe), JSON (summary), and PNGs (histogram + sample maps).
#
# Compatible with:
#   - config.ACTIVE (ANOMALY target settings; OUTPUT routing; REPRO seeding)
#   - io_paths.resolve_output_paths / ensure_colab_drive_mounted
#   - seeding.load_or_create_run_seeds / universe_rngs
#
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional, List
import os, json, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
try:
    from seeding import load_or_create_run_seeds, universe_rngs
except Exception:
    load_or_create_run_seeds = None
    universe_rngs = None

# healpy optional
try:
    import healpy as hp
except Exception:
    hp = None


# ---------------------------
# Utilities
# ---------------------------
def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()


def _compute_power_map(m: np.ndarray, mask: np.ndarray) -> float:
    """Compute variance (power) of a CMB map restricted to given mask (boolean array)."""
    sub = m[mask]
    return float(np.mean(sub**2))


def _default_cl(lmax: int = 64) -> np.ndarray:
    """Fallback C_l spectrum ~ 1/(l(l+1)) normalized."""
    cl = np.zeros(lmax + 1, dtype=float)
    l = np.arange(2, lmax + 1, dtype=float)
    cl[2:] = 1.0 / (l * (l + 1))
    norm = np.sum((2*l + 1) * cl[2:]) / (4*np.pi)
    if norm > 0:
        cl[2:] /= norm
    return cl


def _make_map_from_cl(cl: np.ndarray, nside: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single Gaussian sky map from C_l."""
    if hp is None:
        raise RuntimeError("healpy required for map synthesis.")
    local_seed = int(rng.integers(0, 2**31 - 1))
    hp.random.seed(local_seed)
    return hp.synfast(cl, nside=nside, lmax=len(cl)-1, new=True, verbose=False)


def _save_mirrors(files: List[str], mirrors: List[str], fig_sub: str):
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
def run_hpa(active_cfg: Dict = ACTIVE,
            cmb_maps: Optional[np.ndarray] = None,
            cl_spectrum: Optional[np.ndarray] = None) -> Dict:
    """
    Hemispherical Power Asymmetry detection.

    Inputs:
      - cmb_maps: array (N, npix) in HEALPix RING order. If None, maps synthesized.
      - cl_spectrum: optional C_l for synthesis.

    Returns:
      dict with CSV/JSON/plots and dataframe.
    """
    if not active_cfg.get("ANOMALY", {}).get("enabled", True):
        print("[HPA] ANOMALY disabled → skipping.")
        return {}

    # Resolve outputs
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")

    # Config
    a_cfg = active_cfg.get("ANOMALY", {})
    map_cfg = a_cfg.get("map", {})
    targets = a_cfg.get("targets", [])
    hpa_cfg = None
    for t in targets:
        if t.get("name") in {"hemispheric_asymmetry", "hpa"}:
            hpa_cfg = t
            break
    if hpa_cfg is None:
        hpa_cfg = {"name": "hemispheric_asymmetry", "enabled": True,
                   "l_max": 40, "n_mc": 200, "pval_thresh": 0.05}

    if not hpa_cfg.get("enabled", True):
        print("[HPA] target disabled → skipping.")
        return {}

    nside = int(map_cfg.get("resolution_nside", 128))
    lmax = int(hpa_cfg.get("l_max", 40))
    n_mc = int(hpa_cfg.get("n_mc", 200))
    seed_per_map = bool(map_cfg.get("seed_per_map", True))

    # Seeds
    if load_or_create_run_seeds is not None:
        seeds = load_or_create_run_seeds(active_cfg)
        master_seed = seeds["master_seed"]
        uni_seeds = seeds["universe_seeds"]
        rng_master = _rng(master_seed)
    else:
        rng_master = _rng(active_cfg.get("ENERGY", {}).get("seed", None))
        uni_seeds = None

    # N universes
    N = int(active_cfg["ENERGY"].get("num_universes", 1000))
    if cmb_maps is not None:
        N = cmb_maps.shape[0]

    # Spectrum
    if cl_spectrum is not None:
        cl_in = np.asarray(cl_spectrum, dtype=float)
    else:
        cl_in = _default_cl(lmax)

    # Synthesize if needed
    if cmb_maps is None and hp is not None:
        maps = []
        if seed_per_map and universe_rngs is not None and uni_seeds is not None:
            rngs = universe_rngs(uni_seeds[:N])
            for i in range(N):
                maps.append(_make_map_from_cl(cl_in, nside, rngs[i]))
        else:
            for i in range(N):
                maps.append(_make_map_from_cl(cl_in, nside, rng_master))
        cmb_maps = np.vstack(maps)

    # Hemisphere masks
    if hp is None:
        raise RuntimeError("healpy required for hemisphere splitting.")
    npix = hp.nside2npix(nside)
    vecs = hp.pix2vec(nside, np.arange(npix))
    z = vecs[2]
    mask_north = z >= 0
    mask_south = ~mask_north

    # Metrics
    ratios = np.empty(N, dtype=float)
    p_values = np.full(N, np.nan)
    keep_idx = np.linspace(0, N-1, num=min(N, 6), dtype=int)
    kept_maps = []

    for i in range(N):
        m = cmb_maps[i]
        pN = _compute_power_map(m, mask_north)
        pS = _compute_power_map(m, mask_south)
        ratio = max(pN, pS) / max(1e-18, min(pN, pS))
        ratios[i] = ratio
        if i in keep_idx:
            kept_maps.append((i, m, ratio))

    # Null distribution via MC
    if hp is not None and n_mc > 0:
        mc_rng = _rng(int(rng_master.integers(0, 2**31 - 1)))
        mc_ratios = np.empty(n_mc, dtype=float)
        for j in range(n_mc):
            m = _make_map_from_cl(cl_in, nside, mc_rng)
            pN = _compute_power_map(m, mask_north)
            pS = _compute_power_map(m, mask_south)
            mc_ratios[j] = max(pN, pS) / max(1e-18, min(pN, pS))
        # Compute p-values
        for i in range(N):
            p_values[i] = float((1 + np.sum(mc_ratios >= ratios[i])) / (1 + n_mc))

    # Dataframe
    out_df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "ratio": ratios,
        "p_value": p_values,
        "flag_asym": (p_values <= hpa_cfg.get("pval_thresh", 0.05)).astype(int),
    })

    # Save CSV
    csv_path = run_dir / "HPA__metrics.csv"
    out_df.to_csv(csv_path, index=False)

    # JSON summary
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "N": int(N),
        "lmax": int(lmax),
        "ratios": {
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)),
        },
        "p_values": {
            "mean": float(np.nanmean(p_values)),
            "frac_sig": float(np.mean(p_values <= hpa_cfg.get("pval_thresh", 0.05))),
        },
        "files": {"csv": str(csv_path)},
    }
    json_path = run_dir / "HPA__summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    figs = []
    plt.figure()
    plt.hist(ratios, bins=40)
    plt.xlabel("Power ratio (max/min)")
    plt.ylabel("count")
    plt.title("Hemispherical Power Asymmetry")
    f1 = fig_dir / "HPA__ratio_hist.png"
    plt.tight_layout()
    plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.close()
    figs.append(str(f1))

    # Mirror
    _save_mirrors([str(csv_path), str(json_path), *figs], mirrors, fig_sub)

    print(f"[HPA] results saved under:\n  {run_dir}")
    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": out_df}


if __name__ == "__main__":
    run_hpa(ACTIVE)
