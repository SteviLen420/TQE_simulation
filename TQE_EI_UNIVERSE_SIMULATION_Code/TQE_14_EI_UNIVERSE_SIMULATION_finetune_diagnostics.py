# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_14_EI_UNIVERSE_SIMULATION_finetune_diagnostics.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script performs an advanced diagnostic analysis to evaluate the "fine-tuning"
# of the simulated universes. It measures sophisticated statistical properties of
# the generated CMB maps and scores each universe based on how closely it matches
# a set of ideal cosmological parameters.
#
# The analysis is restricted to the subset of universes that successfully achieved
# "lock-in," focusing on the most viable outcomes of the simulation. For each of
# these universes, it computes a suite of key metrics from its CMB map, including
# the RMS, power spectral slope (alpha), correlation length, skewness, and
# kurtosis. It then calculates a "fine-tuning score" representing the weighted
# distance between these measured metrics and pre-defined target values.
#
# The outputs include a ranked .csv with the diagnostic metrics for all locked-in
# universes, a .json summary, and crucially, detailed multi-plot diagnostic
# panels (.png) for the top-K best-tuned universes, providing a deep visual
# analysis of their properties.
#
# ===================================================================================

from typing import Dict, Optional, Tuple
import os, json, math, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + paths (stable run_id within a pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR


# ---------------------------
# Helpers (pure analysis)
# ---------------------------
def _safe_load_map(path: str) -> np.ndarray:
    """Load a map from .npy or .npz; returns 2D float64 ndarray."""
    if path.endswith(".npy"):
        arr = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        arr = data["map"] if "map" in data else data[list(data.keys())[0]]
    else:
        raise ValueError(f"Unsupported map format: {path}")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Map must be 2D, got shape {arr.shape}")
    return arr


def _rms(arr: np.ndarray) -> float:
    """RMS after mean removal."""
    a = arr - np.mean(arr)
    return float(np.sqrt(np.mean(a * a)))


def _isotropic_psd_and_alpha(arr: np.ndarray,
                             kmin_frac: float = 0.02,
                             kmax_frac: float = 0.45) -> Tuple[float, float]:
    """
    Estimate isotropic power spectral slope alpha on a flat 2D map.
    Steps:
      1) Remove mean, 2D FFT, power |F|^2.
      2) Build radial wavenumber k = sqrt(kx^2 + ky^2).
      3) Fit log P ~ -alpha * log k in a safe k-range (fractions of Nyquist).
    Returns: (alpha, r2)
    Note: kmin_frac/kmax_frac a Nyquist (0.5) arányai; a kód ezért szoroz 0.5-tel.
    """
    a = arr - np.mean(arr)
    F = np.fft.rfft2(a)           # speed-up on real input
    P = (F.real**2 + F.imag**2)

    ny, nx_r = P.shape            # rfft2 => nx_r = nx//2 + 1
    ky = np.fft.fftfreq(ny)       # [-0.5..0.5)
    kx = np.fft.rfftfreq((nx_r - 1) * 2)  # [0..0.5]
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    KR = np.sqrt(KX**2 + KY**2)

    kr = KR.ravel()
    p  = P.ravel()
    mask = (kr > 1e-9)
    kr = kr[mask]; p = p[mask]

    kmin = kmin_frac * 0.5
    kmax = kmax_frac * 0.5
    fit = (kr >= kmin) & (kr <= kmax) & np.isfinite(p) & (p > 0)
    if fit.sum() < 50:
        return float("nan"), float("nan")

    x = np.log(kr[fit]); y = np.log(p[fit])
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = -m
    return float(alpha), float(r2)


def _radial_autocorr_length(arr: np.ndarray) -> float:
    """
    Radial correlation length ℓ_c from normalized autocorrelation:
      A = ifft2(|F|^2), normalized so A[0,0] = 1.
      Radially average A(r); ℓ_c = first radius where A(r) <= 1/e.
    Returns ℓ_c in pixels (float) or NaN if not found.
    """
    a = arr - np.mean(arr)
    F = np.fft.fft2(a)
    S = np.abs(F)**2
    A = np.real(np.fft.ifft2(S))
    A = A / np.max(A)

    ny, nx = A.shape
    cy, cx = ny // 2, nx // 2
    Ash = np.fft.fftshift(A)

    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    r = R.ravel()
    val = Ash.ravel()
    r_int = np.floor(r).astype(int)
    rmax = r_int.max()
    sums = np.bincount(r_int, weights=val, minlength=rmax + 1)
    cnts = np.bincount(r_int, minlength=rmax + 1)
    with np.errstate(invalid="ignore"):
        prof = sums / np.maximum(cnts, 1)

    target = 1.0 / math.e
    idx = np.where(prof <= target)[0]
    return float(idx[0]) if idx.size > 0 else float("nan")


def _skew_kurt(arr: np.ndarray) -> Tuple[float, float]:
    """Skewness and excess kurtosis (Fisher) after mean removal."""
    a = arr - np.mean(arr)
    s = np.std(a)
    if s <= 0:
        return 0.0, 0.0
    z = a / s
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def _finetune_score(metrics: Dict[str, float],
                    targets: Dict[str, Dict[str, float]]) -> float:
    """
    Weighted distance to target values/ranges:
      Each metric m has either {'target': m0, 'tol': t} or {'min': a, 'max': b}.
      Accumulate normalized squared deviations with optional 'weight'.
    """
    score = 0.0
    for key, cfg in targets.items():
        w = float(cfg.get("weight", 1.0))
        val = metrics.get(key, float("nan"))
        if not np.isfinite(val):
            score += w * 10.0
            continue

        if "target" in cfg:
            t = float(cfg["target"])
            tol = float(cfg.get("tol", 1.0))
            dz = (val - t) / max(tol, 1e-12)
            score += w * (dz * dz)
        else:
            lo = cfg.get("min", -np.inf)
            hi = cfg.get("max",  np.inf)
            tol = float(cfg.get("tol", 1.0))
            if val < lo:
                dz = (lo - val) / tol
                score += w * (dz * dz)
            elif val > hi:
                dz = (val - hi) / tol
                score += w * (dz * dz)
    return float(score)


# ---------------------------
# Public API
# ---------------------------
def run_finetune_diagnostics(active_cfg: Dict = ACTIVE,
                             collapse_csv: Optional[str] = None,
                             cmb_manifest_csv: Optional[str] = None):
    """
    Measure fine-tuning indicators on CMB maps for LOCKED-IN universes only.

    Inputs:
      - collapse_csv: path to collapse output CSV (must have 'universe_id', 'lockin_at')
      - cmb_manifest_csv: path to CMB generator manifest CSV (cols: 'universe_id','map_path')

    Outputs (under run directory):
      - CSV:  <tag>__finetune_metrics.csv
      - JSON: <tag>__finetune_summary.json
      - PNGs: <tag>__ft_hist_*.png and <tag>__ft_panel_rankXX_uidYY.png
    """
    # Use cached paths to keep run_id consistent
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    mirrors = paths.get("mirrors", [])

    # EI/E filename tag
    pipe_cfg = active_cfg.get("PIPELINE", {}) if isinstance(active_cfg, dict) else {}
    use_info = bool(pipe_cfg.get("use_information", True))
    tag_prefix = "EI__" if use_info else "E__"

    runtime_cfg = active_cfg.get("RUNTIME", {}) if isinstance(active_cfg, dict) else {}
    dpi = int(runtime_cfg.get("matplotlib_dpi", 180))

    # -------- Config for targets (override via ACTIVE["FINETUNE_DIAG"])
    cfg_ft = active_cfg.get("FINETUNE_DIAG", {})
    tcfg = cfg_ft if isinstance(cfg_ft, dict) else {}
    targets = tcfg.get("targets", {
        "rms":      {"target": 1.0, "tol": 0.25, "weight": 1.0},
        "alpha":    {"target": 2.9, "tol": 0.6,  "weight": 1.0},
        "corr_len": {"min": 2.0,   "max": 40.0, "tol": 2.0, "weight": 0.7},
        "skew":     {"target": 0.0, "tol": 0.15, "weight": 0.5},
        "kurt":     {"target": 0.0, "tol": 0.30, "weight": 0.5},
    })
    top_k = int(tcfg.get("top_k", 5))

    # -------- Load inputs
    if collapse_csv is None:
        collapse_csv = run_dir / f"{tag_prefix}collapse_lockin.csv"
    if cmb_manifest_csv is None:
        cmb_manifest_csv = run_dir / f"{tag_prefix}cmb_maps.csv"

    collapse_csv = pathlib.Path(collapse_csv)
    cmb_manifest_csv = pathlib.Path(cmb_manifest_csv)

    if not collapse_csv.is_file():
        raise FileNotFoundError(f"collapse CSV not found: {collapse_csv}")
    if not cmb_manifest_csv.is_file():
        raise FileNotFoundError(f"CMB manifest CSV not found: {cmb_manifest_csv}")

    df_col = pd.read_csv(str(collapse_csv))
    df_map = pd.read_csv(str(cmb_manifest_csv))

    # required columns
    req_col = {"universe_id", "lockin_at"}
    if not req_col.issubset(df_col.columns):
        raise ValueError(f"collapse CSV must contain {req_col}, got {df_col.columns.tolist()}")

    req_map = {"universe_id", "map_path"}
    if not req_map.issubset(df_map.columns):
        raise ValueError(f"CMB manifest must contain {req_map}, got {df_map.columns.tolist()}")

    locked = df_col.loc[df_col["lockin_at"].ge(0)].copy()
    if locked.empty:
        print("[FINETUNE] No locked-in universes to analyze.")
        return {}

    df = pd.merge(locked, df_map, on="universe_id", how="inner")
    if df.empty:
        print("[FINETUNE] No overlap between locked-in universes and manifest.")
        return {}

    # -------- Per-universe metrics
    rows = []
    for _, row in df.iterrows():
        uid = int(row["universe_id"])
        path = str(row["map_path"])
        try:
            m = _safe_load_map(path)
            rms = _rms(m)
            alpha, r2 = _isotropic_psd_and_alpha(m)
            corr = _radial_autocorr_length(m)
            skew, kurt = _skew_kurt(m)
            metrics = dict(rms=rms, alpha=alpha, corr_len=corr, skew=skew, kurt=kurt)
            score = _finetune_score(metrics, targets)
            rows.append({
                "universe_id": uid,
                "map_path": path,
                "rms": rms,
                "alpha": alpha,
                "alpha_r2": r2,
                "corr_len": corr,
                "skew": skew,
                "kurt": kurt,
                "finetune_score": score,
                "lockin_at": int(row.get("lockin_at", -1)),
            })
        except Exception as e:
            rows.append({
                "universe_id": uid,
                "map_path": path,
                "rms": np.nan, "alpha": np.nan, "alpha_r2": np.nan,
                "corr_len": np.nan, "skew": np.nan, "kurt": np.nan,
                "finetune_score": np.inf,
                "lockin_at": int(row.get("lockin_at", -1)),
                "error": str(e),
            })

    out = pd.DataFrame(rows)
    # Tagged outputs
    out_csv  = run_dir / f"{tag_prefix}finetune_metrics.csv"
    out_json = run_dir / f"{tag_prefix}finetune_summary.json"
    out.to_csv(out_csv, index=False)

    # -------- Summary & Top-K
    def S(x):
        x = np.asarray(x); x = x[np.isfinite(x)]
        if x.size == 0:
            return {}
        return {
            "min": float(np.min(x)), "max": float(np.max(x)),
            "mean": float(np.mean(x)), "std": float(np.std(x)),
            "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
        }

    top = out[np.isfinite(out["finetune_score"])].sort_values("finetune_score").head(top_k).copy()

    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "N_locked": int(len(locked)),
        "N_analyzed": int(len(out)),
        "targets": targets,
        "stats": {
            "rms": S(out["rms"]),
            "alpha": S(out["alpha"]),
            "alpha_r2": S(out["alpha_r2"]),
            "corr_len": S(out["corr_len"]),
            "skew": S(out["skew"]),
            "kurt": S(out["kurt"]),
            "finetune_score": S(out["finetune_score"]),
        },
        "top_k": int(len(top)),
        "top": top[["universe_id", "finetune_score", "rms", "alpha", "corr_len", "skew", "kurt", "map_path"]]
                    .to_dict(orient="records"),
        "files": {"csv": str(out_csv)},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -------- Plots: histograms (tagged)
    figs = []
    def _hist(series, title, fname, xlabel):
        s = np.asarray(series)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return
        plt.figure()
        plt.hist(s, bins=40)
        plt.xlabel(xlabel); plt.ylabel("count"); plt.title(title)
        p = fig_dir / f"{tag_prefix}{fname}"
        plt.tight_layout(); plt.savefig(p, dpi=dpi); plt.close()
        figs.append(str(p))

    _hist(out["rms"],       "RMS distribution (locked-in)",                 "ft_hist_rms.png",   "RMS")
    _hist(out["alpha"],     "Spectral slope α distribution (locked-in)",    "ft_hist_alpha.png", "alpha")
    _hist(out["corr_len"],  "Correlation length distribution (locked-in)",  "ft_hist_corr.png",  "corr length [px]")
    _hist(out["finetune_score"], "Fine-tuning score distribution (locked-in)", "ft_hist_score.png", "score (lower is better)")

    # -------- Top-K panels: map + PSD fit + autocorr (tagged)
    def _panel(uid: int, map_path: str, alpha: float, r2: float, corr: float, rms: float, rank: int):
        m = _safe_load_map(map_path)

        # 1) Map preview
        plt.figure(figsize=(10, 3.2))
        plt.subplot(1, 3, 1)
        plt.imshow(m, origin="lower", vmin=-3, vmax=3, cmap="coolwarm")
        plt.title(f"u{uid}: map\nRMS={rms:.3g}")
        plt.xticks([]); plt.yticks([])

        # 2) PSD radial + crude binning for visualization
        a = m - np.mean(m)
        F = np.fft.rfft2(a); P = (F.real**2 + F.imag**2).ravel()
        ny, nx_r = F.shape
        ky = np.fft.fftfreq(ny); kx = np.fft.rfftfreq((nx_r - 1) * 2)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        kr = np.sqrt(KX**2 + KY**2).ravel()
        mask = (kr > 1e-9) & np.isfinite(P) & (P > 0)
        sort = np.argsort(kr[mask])
        ksorted, psorted = kr[mask][sort], P[mask][sort]
        nb = int(tcfg.get("nbins_psd", 50))
        edges = np.geomspace(ksorted[0], ksorted[-1], nb + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
        means = []
        for i in range(nb):
            sel = (ksorted >= edges[i]) & (ksorted < edges[i+1])
            means.append(np.mean(psorted[sel]) if sel.any() else np.nan)
        centers = centers[np.isfinite(means)]
        means = np.array(means)[np.isfinite(means)]

        plt.subplot(1, 3, 2)
        if centers.size > 0:
            plt.loglog(centers, means, '.', markersize=3)
        plt.title(f"PSD radial\nalpha≈{alpha:.2f}, R²={r2:.2f}")
        plt.xlabel("k"); plt.ylabel("P(k)")

        # 3) Radial autocorrelation + ℓ_c marker
        a = m - np.mean(m)
        F2 = np.fft.fft2(a); S = np.abs(F2)**2
        A = np.real(np.fft.ifft2(S)); A = A / np.max(A)
        Ash = np.fft.fftshift(A)
        ny, nx = Ash.shape; cy, cx = ny//2, nx//2
        yv = np.arange(ny) - cy; xv = np.arange(nx) - cx
        X, Y = np.meshgrid(xv, yv, indexing="xy")
        R = np.sqrt(X**2 + Y**2)
        r = R.ravel(); val = Ash.ravel()
        r_int = np.floor(r).astype(int)
        rmax = r_int.max()
        sums = np.bincount(r_int, weights=val, minlength=rmax+1)
        cnts = np.bincount(r_int, minlength=rmax+1)
        prof = sums / np.maximum(cnts, 1)

        plt.subplot(1, 3, 3)
        plt.plot(prof)
        plt.axhline(1/math.e, ls="--", color="gray")
        plt.axvline(corr if np.isfinite(corr) else 0, ls="--", color="red")
        plt.title(f"Autocorr profile\nℓc≈{corr:.2f} px")
        plt.xlabel("radius [px]"); plt.ylabel("A(r)")
        plt.tight_layout()

        fp = fig_dir / f"{tag_prefix}ft_panel_rank{rank:02d}_uid{uid}.png"
        plt.savefig(fp, dpi=dpi)
        plt.close()
        figs.append(str(fp))

    for rk, r in enumerate(top.itertuples(index=False), start=1):
        _panel(int(r.universe_id), r.map_path, float(r.alpha), float(r.alpha_r2),
               float(r.corr_len), float(r.rms), rk)

    # -------- Mirror copies (CSV/JSON to mirror root; PNGs to <mirror>/<fig_subdir>/)
    from shutil import copy2
    fig_sub = active_cfg.get("OUTPUTS", {}).get("local", {}).get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            mpath = pathlib.Path(m)
            mpath.mkdir(parents=True, exist_ok=True)
            copy2(out_csv,  mpath / out_csv.name)
            copy2(out_json, mpath / out_json.name)  
            if figs:
                m_fig = pathlib.Path(m) / fig_sub
                m_fig.mkdir(parents=True, exist_ok=True)
                for fp in figs:
                    copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[FINETUNE] analyzed {len(out)} locked-in maps → CSV/JSON/PNGs @ {run_dir}")
    return {"csv": str(out_csv), "json": str(out_json), "plots": figs, "table": out}

# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_finetune_diagnostics_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_finetune_diagnostics(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_finetune_diagnostics_stage(ACTIVE)
