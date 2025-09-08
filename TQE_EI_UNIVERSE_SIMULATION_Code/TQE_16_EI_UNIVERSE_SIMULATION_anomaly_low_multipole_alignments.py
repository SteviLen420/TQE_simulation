# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_16_EI_UNIVERSE_SIMULATION_anomaly_low_multipole_alignments.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script is a specialized scientific analysis module designed to test for
# the "Low Multipole Alignment" anomaly (sometimes called the "Axis of Evil")
# in the simulated universes. This anomaly refers to the unexpected alignment
# of the largest-scale features in the CMB.
#
# Instead of analyzing full sky maps, this script simulates the anomaly from
# first principles. For each universe, it independently generates a random
# quadrupole (l=2) and a random octopole (l=3) by drawing their spherical
# harmonic coefficients (a_lm). Using an inertia tensor method, it then
# calculates the preferred spatial axis for each of these two shapes and
# computes the angle between them.
#
# This analysis directly tests the cosmological principle of statistical
# isotropy. A significant number of universes with a small alignment angle
# would challenge this assumption, mirroring a contentious debate in modern
# cosmology. The script outputs a .csv with the alignment angles, a .json
# summary, and a key diagnostic histogram of the angles.
#
# ===================================================================================

from typing import Dict, Optional, Tuple
import os, json, pathlib, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + paths (stable run_id within a pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds, universe_rngs

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
    return "EI" if cfg.get("PIPELINE",{}).get("use_information", True) else "E"


def _fibonacci_sphere(n_pts: int) -> np.ndarray:
    """Even-ish sampling over the unit sphere. Returns (N,3) unit vectors."""
    i = np.arange(n_pts, dtype=float)
    z = (2.0 * i + 1.0) / n_pts - 1.0
    phi = (math.pi * (3.0 - math.sqrt(5.0))) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    vecs = np.stack([x, y, z], axis=1)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)  # defensive normalization
    return vecs


def _theta_phi_from_vecs(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (N,3) unit vectors to (theta, phi) with theta∈[0,π], phi∈[0,2π)."""
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    return theta, phi


def _default_Cl(l: int) -> float:
    """Very simple low-ℓ power model; scale ∝ 1/[ℓ(ℓ+1)] for ℓ≥2."""
    return 1.0 / (l * (l + 1)) if l >= 2 else 0.0


def _draw_alms_low_l(rng: np.random.Generator, l_vals=(2, 3), cl_scale=1.0) -> dict:
    """
    Sample complex a_{ℓm} for low multipoles with Gaussian stats and variance C_ℓ.
    Enforce a_{ℓ,-m} = (-1)^m a_{ℓm}^* for real fields.
    Returns: {ℓ: array shape (2ℓ+1,) ordered m=-ℓ..+ℓ}
    """
    alms = {}
    for l in l_vals:
        C = cl_scale * _default_Cl(l)
        vals = np.zeros(2 * l + 1, dtype=np.complex128)
        # m = 0 is real N(0, C)
        vals[l + 0] = rng.normal(0.0, np.sqrt(C))
        # m = 1..l are complex, with Re/Im ~ N(0, C/2)
        for m in range(1, l + 1):
            re = rng.normal(0.0, np.sqrt(C / 2.0))
            im = rng.normal(0.0, np.sqrt(C / 2.0))
            vals[l + m] = re + 1j * im
            vals[l - m] = ((-1) ** m) * np.conjugate(vals[l + m])  # reality condition
        alms[l] = vals
    return alms


def _bandmap_from_alms(theta: np.ndarray, phi: np.ndarray, l: int, alm_vec: np.ndarray) -> np.ndarray:
    """
    Reconstruct a band-limited map for a single ℓ on directions (theta, phi):
        T_ℓ(n) = Σ_{m=-ℓ}^{+ℓ} a_{ℓm} Y_{ℓm}(n)
    Note: requires scipy.special.sph_harm.
    """
    if sph_harm is None:
        raise RuntimeError("scipy.special.sph_harm is required to reconstruct band-limited maps.")
    m_vals = np.arange(-l, l + 1, dtype=int)
    # SciPy sph_harm signature: sph_harm(m, l, phi, theta)
    Y = np.stack([sph_harm(m, l, phi, theta) for m in m_vals], axis=1)  # (N, 2l+1)
    T = (Y @ alm_vec).real
    return np.asarray(T, dtype=float)


def _inertia_axis_and_conc(dirs_xyz: np.ndarray, T: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Power-weighted inertia tensor on the sphere:
       I = Σ_i (weights[i] * T[i]^2) * n_i n_i^T
       Returns (principal_axis_unit_vector, concentration = λ_max / trace(I)).
       Falls back to +Z and 0.0 if total power is ~0.
    """
    N = dirs_xyz.shape[0]
    if weights is None:
        weights = np.full(N, 4.0 * np.pi / N)

    power = weights * (T * T)
    total_power = float(np.sum(power))
    if not np.isfinite(total_power) or total_power <= 0.0:
        # Fallback: no directional preference → arbitrary unit axis, zero concentration
        return np.array([0.0, 0.0, 1.0], dtype=float), 0.0

    # I = Σ w T^2 n n^T
    # Use einsum to avoid large temp arrays: (N),(N,3),(N,3) -> (3,3)
    I = np.einsum("i,ij,ik->jk", power, dirs_xyz, dirs_xyz, optimize=True)
    I = 0.5 * (I + I.T)

    vals, vecs = np.linalg.eigh(I)
    idx = int(np.argmax(vals))
    v = vecs[:, idx]
    v = v / max(1e-18, np.linalg.norm(v))
    conc = float(np.max(vals) / max(1e-18, np.trace(I)))
    return v.astype(float), conc


def _angle_between_axes(a: np.ndarray, b: np.ndarray) -> float:
    """Return angle in degrees between two axes, sign-invariant (0..90°)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
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
    # --- mandatory SciPy (here, with correct indentation) ---
    if sph_harm is None:
        raise RuntimeError("This stage requires SciPy (scipy.special.sph_harm). Please install scipy.")

    # -- stable, cached paths --
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(FIG_DIR); fig_dir.mkdir(parents=True, exist_ok=True)
    mirrors = paths.get("mirrors", [])

    tag = _tag_ei(active_cfg)
    N   = int(active_cfg.get("ENERGY", {}).get("num_universes", 0))
    dpi = int(active_cfg.get("RUNTIME", {}).get("matplotlib_dpi", 180))

    # -- threshold from ANOMALY.targets --
    align_thresh_deg = 20.0
    for t in active_cfg.get("ANOMALY", {}).get("targets", []):
        if t.get("name") in ("quad_oct_align", "low_multipole_align"):
            align_thresh_deg = float(t.get("l2l3_align_deg", align_thresh_deg))

    # -- directions on the sphere (healpy or Fibonacci) --
    if hp is not None:
        nside = int(active_cfg.get("ANOMALY", {}).get("map", {}).get("resolution_nside", 64))
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        vec = hp.ang2vec(theta, phi)
        dirs = vec.T if (vec.ndim == 2 and vec.shape[0] == 3) else vec  # (npix,3)
        weights = np.full(npix, hp.nside2pixarea(nside))
    else:
        npix = max(2048, 16 * 64)
        dirs = _fibonacci_sphere(npix)
        theta, phi = _theta_phi_from_vecs(dirs)
        weights = np.full(npix, 4.0 * np.pi / npix)

    # -- precompute Y_{lm} (once) --
    m2 = np.arange(-2, 3)
    m3 = np.arange(-3, 4)
    Y2 = np.stack([sph_harm(m, 2, phi, theta) for m in m2], axis=1)  # (npix, 5)
    Y3 = np.stack([sph_harm(m, 3, phi, theta) for m in m3], axis=1)  # (npix, 7)

    # -- seeds / RNGs --
    seeds = load_or_create_run_seeds(active_cfg)
    rngs  = universe_rngs(seeds.get("universe_seeds", []))
    if len(rngs) < N:
        base = int(seeds.get("master_seed", 1234567))
        for i in range(len(rngs), N):
            rngs.append(np.random.default_rng(base + 10007 * i))

    # -- storage buffers --
    axis_q = np.zeros((N, 3), dtype=np.float64)
    axis_o = np.zeros((N, 3), dtype=np.float64)
    conc_q = np.zeros(N, dtype=np.float64)
    conc_o = np.zeros(N, dtype=np.float64)
    angle_deg = np.zeros(N, dtype=np.float64)

    cl_scale = 1.0  # optional scale for low-ℓ variance

    # -- main loop --
    for i in range(N):
        rng  = rngs[i]
        alms = _draw_alms_low_l(rng, l_vals=(2, 3), cl_scale=cl_scale)

        # band-limited map reconstruction (matrix multiply with precomputed Y)
        T2 = (Y2 @ alms[2]).real
        T3 = (Y3 @ alms[3]).real

        v2, c2 = _inertia_axis_and_conc(dirs, T2, weights)
        v3, c3 = _inertia_axis_and_conc(dirs, T3, weights)

        axis_q[i, :] = v2; conc_q[i] = c2
        axis_o[i, :] = v3; conc_o[i] = c3
        angle_deg[i] = _angle_between_axes(v2, v3)

    # -- table + CSV --
    df = pd.DataFrame({
        "universe_id": np.arange(N, dtype=int),
        "axis_q_x": axis_q[:, 0], "axis_q_y": axis_q[:, 1], "axis_q_z": axis_q[:, 2],
        "axis_o_x": axis_o[:, 0], "axis_o_y": axis_o[:, 1], "axis_o_z": axis_o[:, 2],
        "conc_q": conc_q, "conc_o": conc_o,
        "angle_deg": angle_deg,
        "aligned_flag": (angle_deg <= align_thresh_deg).astype(int),
    })
    csv_path = run_dir / f"{tag}__anomaly_low_multipole_align.csv"
    df.to_csv(csv_path, index=False)

    def _stats(x):
        x = np.asarray(x)
        if x.size == 0:
            return {"min": float("nan"), "p25": float("nan"), "median": float("nan"),
                    "p75": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
        return {
            "min": float(np.min(x)),
            "p25": float(np.percentile(x, 25)),
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    K = min(10, N)
    if K > 0:
        top_idx = np.argpartition(angle_deg, K - 1)[:K]
        top_idx = top_idx[np.argsort(angle_deg[top_idx])]
        top_list = [{"universe_id": int(i), "angle_deg": float(angle_deg[i])} for i in top_idx]
    else:
        top_list = []

    # -- plots --
    figs = []
    if N > 0:
        plt.figure()
        plt.hist(angle_deg, bins=36, range=(0, 90))
        plt.axvline(align_thresh_deg, color="red", linestyle="--", label=f"threshold = {align_thresh_deg:.1f}°")
        plt.xlabel("Quadrupole–octopole alignment angle (deg)")
        plt.ylabel("count")
        plt.title("Low-ℓ alignment angle distribution")
        plt.legend()
        f1 = fig_dir / f"{tag}__low_multipole_alignment_hist.png"
        plt.tight_layout(); plt.savefig(f1, dpi=dpi); plt.close()
        figs.append(str(f1))

        # optional scatter (if column X exists from an earlier stage)
        try:
            for c in [run_dir / f"{tag}__expansion.csv", run_dir / f"{tag}__collapse_lockin.csv"]:
                if c.exists():
                    tmp = pd.read_csv(c)
                    if "X" in tmp.columns and len(tmp["X"]) >= N:
                        X = tmp["X"].to_numpy(dtype=float)[:N]
                        plt.figure()
                        plt.scatter(X, angle_deg, s=6, alpha=0.6)
                        plt.xlabel("X"); plt.ylabel("alignment angle (deg)")
                        plt.title("X vs low-ℓ alignment")
                        f2 = fig_dir / f"{tag}__low_multipole_alignment_scatter.png"
                        plt.tight_layout(); plt.savefig(f2, dpi=dpi); plt.close()
                        figs.append(str(f2))
                        break
        except Exception:
            pass

    summary = {
        "env": paths.get("env", ""),
        "run_id": paths.get("run_id", ""),
        "mode": tag,
        "N": int(N),
        "align_threshold_deg": float(align_thresh_deg),
        "counts": {"aligned_n": int((angle_deg <= align_thresh_deg).sum())},
        "angle_deg": _stats(angle_deg),
        "concentration": {"conc_q": _stats(conc_q), "conc_o": _stats(conc_o)},
        "top_aligned": top_list,
        "files": {"csv": str(csv_path), "plots": figs},
    }
    json_path = run_dir / f"{tag}__anomaly_low_multipole_align_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -- mirroring --
    from shutil import copy2
    fig_sub = active_cfg.get("OUTPUTS", {}).get("local", {}).get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            mpath = pathlib.Path(m); mpath.mkdir(parents=True, exist_ok=True)
            copy2(csv_path, mpath / csv_path.name)
            copy2(json_path, mpath / json_path.name)
            if figs:
                m_fig = mpath / fig_sub
                m_fig.mkdir(parents=True, exist_ok=True)
                for fp in figs:
                    copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[ANOM-LM] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")
    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": df}

# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_anomaly_low_multipole_alignments_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_anomaly_low_multipole_alignments(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_anomaly_low_multipole_alignments_stage(ACTIVE)
