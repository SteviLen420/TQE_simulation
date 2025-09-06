# ===================================================================================
# TQE_18_EI_UNIVERSE_SIMULATION_anomaly_HemisphericalAsymmetry.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional, List
import os, json, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + paths (stable run_id within a pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR

# Seeding (prefer project path; safe fallback to None)
try:
    from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds, universe_rngs
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
    """Variance (power) over a boolean mask."""
    sub = m[mask]
    return float(np.mean(sub * sub)) if sub.size else float("nan")


def _default_cl(lmax: int = 64) -> np.ndarray:
    """Fallback C_l spectrum ~ 1/(l(l+1)) normalized (C0=C1=0)."""
    cl = np.zeros(lmax + 1, dtype=float)
    l = np.arange(2, lmax + 1, dtype=float)
    cl[2:] = 1.0 / (l * (l + 1))
    norm = np.sum((2 * l + 1) * cl[2:]) / (4 * np.pi)
    if norm > 0:
        cl[2:] /= norm
    return cl


def _make_map_from_cl(cl: np.ndarray, nside: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single Gaussian sky map from C_l (healpy required)."""
    if hp is None:
        raise RuntimeError("healpy required for map synthesis.")
    local_seed = int(rng.integers(0, 2**31 - 1))
    hp.random.seed(local_seed)
    return hp.synfast(cl, nside=nside, lmax=len(cl) - 1, new=True, verbose=False)


def _save_mirrors(files: List[str], mirrors: List[str], fig_sub: str):
    """Copy CSV/JSON to mirror root; PNG/JPG to <mirror>/<fig_sub>/."""
    from shutil import copy2
    for m in mirrors or []:
        try:
            m_path = pathlib.Path(m)
            m_path.mkdir(parents=True, exist_ok=True)
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
    Hemispherical Power Asymmetry (HPA).
      - If `cmb_maps` is None and healpy is available → synthesize maps from C_ℓ.
      - Otherwise use provided maps (HEALPix RING).

    Outputs (under run dir):
      - CSV:  <tag>__anomaly_hpa_metrics.csv
      - JSON: <tag>__anomaly_hpa_summary.json
      - PNG:  <tag>__hpa_ratio_hist.png
    """
    if not active_cfg.get("ANOMALY", {}).get("enabled", True):
        print("[HPA] ANOMALY disabled → skipping.")
        return {}

    # Use cached paths (stable run_id)
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(FIG_DIR); fig_dir.mkdir(parents=True, exist_ok=True)
    mirrors = paths.get("mirrors", [])
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
    dpi     = int(active_cfg["RUNTIME"].get("matplotlib_dpi", 180))

    # EI/E tag
    tag = "EI" if active_cfg["PIPELINE"].get("use_information", True) else "E"

    # Config
    a_cfg   = active_cfg.get("ANOMALY", {})
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

    nside        = int(map_cfg.get("resolution_nside", 128))
    lmax         = int(hpa_cfg.get("l_max", 40))
    n_mc         = int(hpa_cfg.get("n_mc", 200))
    p_thr        = float(hpa_cfg.get("pval_thresh", 0.05))
    seed_per_map = bool(map_cfg.get("seed_per_map", True))

    # Seeds
    if load_or_create_run_seeds is not None:
        seeds       = load_or_create_run_seeds(active_cfg)
        master_seed = seeds.get("master_seed", None)
        uni_seeds   = seeds.get("universe_seeds", [])
        rng_master  = _rng(master_seed)
    else:
        rng_master  = _rng(active_cfg.get("ENERGY", {}).get("seed", None))
        uni_seeds   = []

    # N universes
    N = int(active_cfg["ENERGY"].get("num_universes", 1000))
    if cmb_maps is not None:
        N = int(cmb_maps.shape[0])

    # Spectrum
    cl_in = np.asarray(cl_spectrum, dtype=float) if cl_spectrum is not None else _default_cl(lmax)

    # Synthesize if needed
    synthesized = False
    if cmb_maps is None:
        if hp is None:
            raise RuntimeError("healpy is required to synthesize maps (cmb_maps=None).")
        synthesized = True
        maps = []
        if seed_per_map and universe_rngs is not None and len(uni_seeds) > 0:
            rngs = universe_rngs(uni_seeds[:N])
            for i in range(N):
                maps.append(_make_map_from_cl(cl_in, nside, rngs[i]))
        else:
            for i in range(N):
                maps.append(_make_map_from_cl(cl_in, nside, rng_master))
        cmb_maps = np.vstack(maps)

    # Hemisphere masks (requires healpy)
    if hp is None:
        raise RuntimeError("healpy required for hemisphere splitting and pixel geometry.")
    npix = hp.nside2npix(nside)

    # If maps provided, sanity check pixel count vs nside
    if cmb_maps is not None:
        if cmb_maps.ndim != 2 or cmb_maps.shape[1] != npix:
            raise ValueError(f"[HPA] cmb_maps must have shape (N, {npix}) for nside={nside}.")

    vecs = hp.pix2vec(nside, np.arange(npix))
    z    = vecs[2]
    mask_north = z >= 0
    mask_south = ~mask_north

    # Metrics
    ratios   = np.empty(N, dtype=float)
    p_values = np.full(N, np.nan)

    for i in range(N):
        m  = cmb_maps[i]
        pN = _compute_power_map(m, mask_north)
        pS = _compute_power_map(m, mask_south)
        ratios[i] = max(pN, pS) / max(1e-18, min(pN, pS))

    # Null distribution via MC (healpy required)
    if n_mc > 0:
        mc_rng     = _rng(int(rng_master.integers(0, 2**31 - 1)))
        mc_ratios  = np.empty(n_mc, dtype=float)
        for j in range(n_mc):
            m = _make_map_from_cl(cl_in, nside, mc_rng)
            pN = _compute_power_map(m, mask_north)
            pS = _compute_power_map(m, mask_south)
            mc_ratios[j] = max(pN, pS) / max(1e-18, min(pN, pS))
        # p-value = fraction of null >= observed (asymmetry → large ratio)
        for i in range(N):
            p_values[i] = float((1 + np.sum(mc_ratios >= ratios[i])) / (1 + n_mc))

    # Dataframe
    out_df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "ratio": ratios,
        "p_value": p_values,
        "flag_asym": (p_values <= p_thr).astype(int),
        "synthesized_map": int(synthesized),
    })

    # Tagged outputs
    csv_path  = run_dir / f"{tag}__anomaly_hpa_metrics.csv"
    json_path = run_dir / f"{tag}__anomaly_hpa_summary.json"
    out_df.to_csv(csv_path, index=False)

    # JSON summary
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": int(N),
        "nside": int(nside),
        "lmax": int(lmax),
        "ratios": {
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "mean": float(np.mean(ratios)),
            "median": float(np.median(ratios)),
            "std": float(np.std(ratios)),
        },
        "p_values": {
            "mean": float(np.nanmean(p_values)) if np.isfinite(p_values).any() else float("nan"),
            "frac_sig": float(np.mean(p_values <= p_thr)) if np.isfinite(p_values).any() else 0.0,
            "threshold": p_thr,
        },
        "files": {"csv": str(csv_path)},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot
    figs: List[str] = []
    plt.figure()
    plt.hist(ratios, bins=40)
    plt.xlabel("Power ratio (max/min)")
    plt.ylabel("count")
    plt.title("Hemispherical Power Asymmetry")
    f1 = fig_dir / f"{tag}__hpa_ratio_hist.png"
    plt.tight_layout(); plt.savefig(f1, dpi=dpi); plt.close()
    figs.append(str(f1))

    # Mirrors
    _save_mirrors([str(csv_path), str(json_path), *figs], mirrors, fig_sub)

    print(f"[HPA] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")
    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": out_df}


# Standalone
if __name__ == "__main__":
    run_hpa(ACTIVE)
