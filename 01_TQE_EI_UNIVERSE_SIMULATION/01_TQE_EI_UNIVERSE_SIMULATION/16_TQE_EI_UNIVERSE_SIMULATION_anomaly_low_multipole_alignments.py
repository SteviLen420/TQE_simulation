# ===================================================================================
# 16_TQE_EI_UNIVERSE_SIMULATION_anomaly_low_multipole_alignments.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional, Tuple
import os, json, pathlib, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
from seeding import load_or_create_run_seeds, universe_rngs

# Optional deps
try:
    import healpy as hp
except Exception:
    hp = None

try:
    from scipy.special import sph_harm
except Exception:
    sph_harm = None


# ---------------------------
# Utilities
# ---------------------------
def _tag_ei(cfg: dict) -> str:
    """Return filename tag 'EI' or 'E' depending on information channel."""
    return "EI" if cfg["PIPELINE"].get("use_information", True) else "E"

def _fibonacci_sphere(n_pts: int) -> np.ndarray:
    """Even-ish sampling over the unit sphere. Returns (N,3) unit vectors."""
    # Golden-angle based Fibonacci lattice
    i = np.arange(n_pts, dtype=float)
    z = (2.0*i + 1.0)/n_pts - 1.0
    phi = (math.pi*(3.0 - math.sqrt(5.0))) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    vecs = np.stack([x, y, z], axis=1)
    # Normalize (defensive)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs

def _theta_phi_from_vecs(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (N,3) unit vectors to (theta, phi) with theta∈[0,π], phi∈[0,2π)."""
    x, y, z = v[:,0], v[:,1], v[:,2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0*np.pi)
    return theta, phi

def _default_Cl(l: int) -> float:
    """Very simple low-ℓ power model; scale ∝ 1/[ℓ(ℓ+1)] for ℓ≥2."""
    return 1.0 / (l*(l+1)) if l >= 2 else 0.0

def _draw_alms_low_l(rng: np.random.Generator, l_vals=(2,3), cl_scale=1.0) -> dict:
    """
    Sample complex a_{ℓm} for low multipoles with Gaussian stats and variance C_ℓ.
    Enforce a_{ℓ,-m} = (-1)^m a_{ℓm}^* for real fields.
    Returns: {ℓ: array shape (2ℓ+1,) ordered m=-ℓ..+ℓ}
    """
    alms = {}
    for l in l_vals:
        C = cl_scale * _default_Cl(l)
        # m=0 real ~ N(0, C); m>0 complex with var C/2 per real/imag part
        vals = np.zeros(2*l+1, dtype=np.complex128)
        # m=0
        vals[l+0] = rng.normal(0.0, np.sqrt(C))
        # m=1..l
        for m in range(1, l+1):
            re = rng.normal(0.0, np.sqrt(C/2.0))
            im = rng.normal(0.0, np.sqrt(C/2.0))
            vals[l+m] = re + 1j*im
            # a_{l,-m} = (-1)^m a_{lm}^*
            vals[l-m] = ((-1)**m) * np.conjugate(vals[l+m])
        alms[l] = vals
    return alms

def _bandmap_from_alms(theta: np.ndarray, phi: np.ndarray, l: int, alm_vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a band-limited map for a single ℓ on directions (theta,phi):
        T_ℓ(n) = Σ_{m=-ℓ}^{+ℓ} a_{ℓm} Y_{ℓm}(n)
    """
    if sph_harm is None:
        raise RuntimeError("scipy.special.sph_harm is required if healpy is unavailable.")
    m_vals = np.arange(-l, l+1, dtype=int)
    Y = np.stack([sph_harm(m, l, phi, theta) for m in m_vals], axis=1)  # (N, 2l+1)
    T = (Y @ alm_vec).real  # real field
    return np.asarray(T, dtype=float)

def _inertia_axis_and_conc(dirs_xyz: np.ndarray, T: np.ndarray, weights: Optional[np.ndarray]=None) -> Tuple[np.ndarray, float]:
    """
    Power-weighted inertia tensor on the sphere:
        I = Σ w T^2 n n^T
    Returns the dominant eigenvector (unit axis) and a concentration metric λ_max / trace(I).
    """
    N = dirs_xyz.shape[0]
    if weights is None:
        weights = np.full(N, 4.0*np.pi/N)
    w = (weights * (T*T)).reshape(-1,1)                # (N,1)
    I = (dirs_xyz[:,:,None] * dirs_xyz[:,None,:])      # (N,3,3) -> n n^T
    I = (w[:,None,:] * I).sum(axis=0)                  # (3,3)
    # Symmetrize numerically
    I = 0.5*(I + I.T)
    vals, vecs = np.linalg.eigh(I)
    idx = int(np.argmax(vals))
    v = vecs[:, idx]
    v = v / np.linalg.norm(v)
    conc = float(np.max(vals) / max(1e-18, np.trace(I)))
    return v, conc

def _angle_between_axes(a: np.ndarray, b: np.ndarray) -> float:
    """Return angle in degrees between two axes, sign-invariant (0..90°)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    cosang = abs(float(np.dot(a, b)))
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------
# Main entry
# ---------------------------
def run_anomaly_low_multipole_alignments(active_cfg: Dict = ACTIVE) -> Dict:
    """
    Compute quadrupole–octopole alignment for each universe and save CSV/JSON/PNGs.
    Returns a dict with file paths and the DataFrame.
    """
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    tag = _tag_ei(active_cfg)
    N = int(active_cfg["ENERGY"]["num_universes"])

    # Read align threshold from config.ANOMALY.targets (quad_oct_align)
    align_thresh_deg = 20.0
    for t in active_cfg["ANOMALY"].get("targets", []):
        if t.get("name") in ("quad_oct_align", "low_multipole_align"):
            align_thresh_deg = float(t.get("l2l3_align_deg", align_thresh_deg))

    # Directions on sphere (healpy or Fibonacci)
    if hp is not None:
        nside = int(active_cfg["ANOMALY"]["map"].get("resolution_nside", 64))
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        dirs = np.column_stack(hp.ang2vec(theta, phi))
        weights = np.full(npix, hp.nside2pixarea(nside))
    else:
        # Portable fallback
        npix = max(2048, 16 * 64)  # ~ few thousand points
        dirs = _fibonacci_sphere(npix)
        theta, phi = _theta_phi_from_vecs(dirs)
        weights = np.full(npix, 4.0*np.pi/npix)

    # Seeds → per-universe RNGs
    seeds = load_or_create_run_seeds(active_cfg)
    rngs = universe_rngs(seeds["universe_seeds"])

    # Storage
    axis_q = np.zeros((N,3), dtype=float)
    axis_o = np.zeros((N,3), dtype=float)
    conc_q = np.zeros(N, dtype=float)
    conc_o = np.zeros(N, dtype=float)
    angle_deg = np.zeros(N, dtype=float)

    # Optional global scale for low-ℓ variance (can be tuned later)
    cl_scale = 1.0

    # Loop universes
    for i in range(N):
        rng = rngs[i]
        alms = _draw_alms_low_l(rng, l_vals=(2,3), cl_scale=cl_scale)

        # Reconstruct band-limited maps
        T2 = _bandmap_from_alms(theta, phi, 2, alms[2])
        T3 = _bandmap_from_alms(theta, phi, 3, alms[3])

        # Preferred axes & concentration
        v2, c2 = _inertia_axis_and_conc(dirs, T2, weights)
        v3, c3 = _inertia_axis_and_conc(dirs, T3, weights)

        axis_q[i,:] = v2
        axis_o[i,:] = v3
        conc_q[i] = c2
        conc_o[i] = c3
        angle_deg[i] = _angle_between_axes(v2, v3)

    # Build table
    df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "axis_q_x": axis_q[:,0], "axis_q_y": axis_q[:,1], "axis_q_z": axis_q[:,2],
        "axis_o_x": axis_o[:,0], "axis_o_y": axis_o[:,1], "axis_o_z": axis_o[:,2],
        "conc_q": conc_q, "conc_o": conc_o,
        "angle_deg": angle_deg,
        "aligned_flag": (angle_deg <= align_thresh_deg).astype(int),
    })

    # Save CSV
    csv_path = run_dir / f"{tag}__anomaly_low_multipole_align.csv"
    df.to_csv(csv_path, index=False)

    # Summary JSON
    def _stats(x):
        return {
            "min": float(np.min(x)),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    # Top-K most aligned universes (smallest angle)
    K = min(10, N)
    top_idx = np.argsort(angle_deg)[:K]
    top_list = [{"universe_id": int(i), "angle_deg": float(angle_deg[i])} for i in top_idx]

    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": N,
        "align_threshold_deg": align_thresh_deg,
        "counts": {
            "aligned_n": int((angle_deg <= align_thresh_deg).sum()),
        },
        "angle_deg": _stats(angle_deg),
        "concentration": {
            "conc_q": _stats(conc_q),
            "conc_o": _stats(conc_o),
        },
        "top_aligned": top_list,
        "files": {
            "csv": str(csv_path),
        },
    }

    json_path = run_dir / f"{tag}__anomaly_low_multipole_align_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plots
    figs = []

    # 1) Histogram of alignment angles
    plt.figure()
    plt.hist(angle_deg, bins=36)
    plt.axvline(align_thresh_deg, color="red", linestyle="--", label=f"threshold = {align_thresh_deg:.1f}°")
    plt.xlabel("Quadrupole–octopole alignment angle (deg)")
    plt.ylabel("count")
    plt.title("Low-ℓ alignment angle distribution")
    plt.legend()
    f1 = fig_dir / f"{tag}__low_multipole_alignment_hist.png"
    plt.tight_layout()
    plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.close()
    figs.append(str(f1))

    # 2) Optional: scatter X vs angle if X available from prior stages (nice diagnostic)
    # We only draw if `montecarlo` or prior table exported X into a CSV in the same run dir.
    try:
        # Look for expansion or collapse CSV quickly
        candidates = [
            run_dir / f"{tag}__expansion.csv",
            run_dir / f"{tag}__collapse_lockin.csv",
        ]
        X = None
        for c in candidates:
            if c.exists():
                tmp = pd.read_csv(c)
                if "X" in tmp.columns:
                    X = tmp["X"].to_numpy(dtype=float)
                    break
        if X is not None and len(X) == N:
            plt.figure()
            plt.scatter(X, angle_deg, s=6, alpha=0.6)
            plt.xlabel("X")
            plt.ylabel("alignment angle (deg)")
            plt.title("X vs low-ℓ alignment")
            f2 = fig_dir / f"{tag}__low_multipole_alignment_scatter.png"
            plt.tight_layout()
            plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
            plt.close()
            figs.append(str(f2))
    except Exception as _:
        pass

    # Mirror copies
    from shutil import copy2
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            m_fig = pathlib.Path(m) / fig_sub
            m_fig.mkdir(parents=True, exist_ok=True)
            for fp in figs:
                copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[ANOM-LM] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": df}


# Standalone
if __name__ == "__main__":
    run_anomaly_low_multipole_alignments(ACTIVE)
