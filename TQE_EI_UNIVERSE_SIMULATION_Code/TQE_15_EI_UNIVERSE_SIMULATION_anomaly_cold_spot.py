# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_15_EI_UNIVERSE_SIMULATION_anomaly_cold_spot.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script is a scientific analysis module designed to detect a specific,
# physically-motivated feature—a "Cold Spot" anomaly—within each of the
# generated CMB maps. This analysis directly tests the simulation's ability
# to reproduce a well-known anomaly that exists in the actual observed CMB.
#
# The detection algorithm systematically searches each map to find the circular
# patch of a pre-defined angular size that has the lowest average temperature.
# This is achieved efficiently by convolving the map with a disk-shaped kernel
# using the Fast Fourier Transform (FFT). The script then quantifies the
# statistical significance of this coldest patch by calculating its z-score,
# which measures how many standard deviations its temperature is below the
# map's global average.
#
# The script produces a .csv file with the z-score of the coldest spot for
# every universe, a .json summary of the total number of anomalies found, and
# .png cutout images of the most significant detected spots for visual
# verification.
#
# ===================================================================================

from typing import Optional, Dict
import os, json, math, pathlib
import numpy as np
import matplotlib.pyplot as plt

# Cached config + paths (stable run_id within a pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR

# ---------------------------
# Small helpers
# ---------------------------
def _deg_to_pix_radius(patch_deg: float, H: int) -> int:
    """Convert circular patch radius in degrees to pixels, using pixel_deg ≈ 180/H."""
    pix_per_deg = H / 180.0
    r = int(round(pix_per_deg * patch_deg))
    return max(1, r)

def _make_disk_kernel(r_pix: int) -> np.ndarray:
    """Create a 2D binary disk kernel of radius r_pix (center-included)."""
    d = 2 * r_pix + 1
    yy, xx = np.ogrid[-r_pix:r_pix + 1, -r_pix:r_pix + 1]
    mask = (xx * xx + yy * yy) <= (r_pix * r_pix)
    return mask.astype(np.float64)

def _fft_convolve2d(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    2D convolution via FFT with 'same' output size.
    Zero-pad to avoid circular artifacts, then center-crop.
    """
    H, W = x.shape
    h, w = k.shape
    FH = H + h - 1
    FW = W + w - 1
    fX = np.fft.rfftn(x, s=(FH, FW))
    k_pad = np.zeros((FH, FW), dtype=np.float64)
    k_pad[:h, :w] = k
    fK = np.fft.rfftn(k_pad)
    y_full = np.fft.irfftn(fX * fK, s=(FH, FW))
    y = y_full[(h - 1)//2:(h - 1)//2 + H, (w - 1)//2:(w - 1)//2 + W]
    return np.asarray(y, dtype=np.float64)

def _extract_cutout(img: np.ndarray, cy: int, cx: int, r: int) -> np.ndarray:
    """Extract a (2r+1)x(2r+1) cutout centered at (cy,cx); clip edges with zeros."""
    H, W = img.shape
    y0 = max(0, cy - r); x0 = max(0, cx - r)
    y1 = min(H, cy + r + 1); x1 = min(W, cx + r + 1)
    out = np.zeros((2 * r + 1, 2 * r + 1), dtype=img.dtype)
    oy0 = r - (cy - y0); ox0 = r - (cx - x0)
    out[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)] = img[y0:y1, x0:x1]
    return out

def _plot_cutout_with_circle(cut: np.ndarray, r: int, title: str, save_path: str):
    """Save a cutout PNG with a dashed circle showing the patch boundary."""
    plt.figure()
    # fixed color limits for visual comparability
    v = float(np.nanstd(cut))
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    plt.imshow(cut, origin="lower", vmin=-3*v, vmax=3*v, cmap="coolwarm")
    theta = np.linspace(0, 2 * math.pi, 512)
    cy = cx = r
    ys = cy + r * np.sin(theta)
    xs = cx + r * np.cos(theta)
    plt.plot(xs, ys, "w--", linewidth=1.0, alpha=0.9, label="patch")
    plt.title(title)
    plt.axis("off")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=ACTIVE.get("RUNTIME", {}).get("matplotlib_dpi", 180))
    plt.close()

# ---------------------------
# Core routine
# ---------------------------
def run_anomaly_cold_spot(active_cfg: Dict = ACTIVE,
                          maps: Optional[np.ndarray] = None,
                          arrays: Optional[Dict[str, np.ndarray]] = None):
    """
    Detect a Cold Spot anomaly across a population of CMB maps.

    Inputs:
      - maps: optional array of shape (N, H, W). If not given, we stream from the CMB manifest CSV.
      - arrays: may contain {"cmb_maps": (N,H,W)}.

    Outputs:
      - CSV/JSON files in run dir, cutout PNGs in figs dir, plus a dict return.
    """
    import pandas as pd  # local import to avoid making pandas a hard dependency at import-time

    # Stage toggle
    if not active_cfg.get("PIPELINE", {}).get("run_anomaly_scan", True):
        print("[COLD_SPOT] run_anomaly_scan=False → skipping.")
        return {}

    # Target toggle
    cold_spec = None
    for t in active_cfg.get("ANOMALY", {}).get("targets", []):
        if t.get("name") in {"cold_spot", "coldspot"} and t.get("enabled", False):
            cold_spec = t
            break
    if cold_spec is None:
        print("[COLD_SPOT] target disabled → skipping.")
        return {}

    # Use cached paths (stable run_id)
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR); run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(FIG_DIR); fig_dir.mkdir(parents=True, exist_ok=True)
    mirrors = paths.get("mirrors", [])

    # EI/E tag
    use_I = bool(active_cfg.get("PIPELINE", {}).get("use_information", True))
    tag   = "EI" if use_I else "E"

    # Config
    patch_deg       = float(cold_spec.get("patch_deg", 10.0))
    z_thresh        = float(cold_spec.get("zscore_thresh", 3.0))
    anom_cfg = active_cfg.get("ANOMALY", {}) or {}
    save_cutouts     = bool(anom_cfg.get("save_cutouts", True))
    save_metrics_csv = bool(anom_cfg.get("save_metrics_csv", True))

    # Input maps:
    # 1) explicit argument, 2) arrays["cmb_maps"], 3) stream from manifest CSV made by TQE_13
    df_manifest = None
    if maps is None:
        if arrays and "cmb_maps" in arrays:
            maps = np.asarray(arrays["cmb_maps"], dtype=np.float64)
        else:
            manifest_csv = run_dir / f"{tag}__cmb_maps.csv"
            if not manifest_csv.exists():
                raise FileNotFoundError(
                    f"[COLD_SPOT] No maps provided and manifest not found: {manifest_csv.name}. "
                    "Run CMB map generation first."
                )
            df_manifest = pd.read_csv(manifest_csv)
            if "universe_id" not in df_manifest.columns or "map_path" not in df_manifest.columns:
                raise ValueError("[COLD_SPOT] Manifest CSV must contain 'universe_id' and 'map_path' columns.")

    # Build kernel (we need H to compute r_pix; if streaming, peek first map)
    def _load_map(path: str) -> np.ndarray:
        if path.endswith(".npy"):
            arr = np.load(path)
        elif path.endswith(".npz"):
            data = np.load(path)
            arr = data["map"] if "map" in data else data[list(data.keys())[0]]
        else:
            raise ValueError(f"Unsupported map format: {path}")
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Map must be 2D: {path} has shape {arr.shape}")
        return arr

    records = []
    cutout_paths = []
                              
    if df_manifest is not None:
        # bail out early if manifest is empty
        if df_manifest.empty:
            print("[COLD_SPOT] Manifest is empty.")
            return {}

        # try to prepare kernel from the first map; if it fails, lazily init later
        m0, K, r_pix = None, None, None
        H0 = W0 = None
        try:
            m0 = _load_map(df_manifest.iloc[0]["map_path"])
            H0, W0 = m0.shape
            r_pix = _deg_to_pix_radius(patch_deg, H0)
            K = _make_disk_kernel(r_pix); K /= max(1.0, K.sum())
        except Exception:
            m0 = None  # will fallback to first successful map

        # stream maps row-by-row; robust to per-file failures
        for i, row in enumerate(df_manifest.itertuples(index=False, name="Row"), start=0):
            try:
                # faster row access via namedtuple (no Series allocation)
                uid = int(row.universe_id)
                M = m0 if (i == 0 and m0 is not None) else _load_map(row.map_path)
                H, W = M.shape

                # lazy kernel init if the peek failed or dims differ
                if K is None or (H0 is not None and (H, W) != (H0, W0)):
                    r_pix = _deg_to_pix_radius(patch_deg, H)
                    K = _make_disk_kernel(r_pix); K /= max(1.0, K.sum())
                    H0, W0 = H, W  # update baseline shape

                # compute global stats (ignore NaNs)
                valid  = np.isfinite(M)
                g_mean = float(np.mean(M[valid])) if valid.any() else np.nan
                g_std  = float(np.std(M[valid]))  if np.sum(valid) > 1 else np.nan

                # convolve to find coldest patch mean and z-score
                fill = g_mean if np.isfinite(g_mean) else 0.0
                conv = _fft_convolve2d(np.nan_to_num(M, nan=fill), K)
                cy, cx = np.unravel_index(int(np.argmin(conv)), conv.shape)
                patch_mean = float(conv[cy, cx])
                z = (patch_mean - g_mean) / g_std if (g_std is not None and g_std > 0) else np.nan

                # optional cutout export
                cut_png = ""
                if save_cutouts:
                    cut = _extract_cutout(M, cy, cx, r_pix)
                    cut_path = fig_dir / f"{tag}__cold_spot_u{uid:05d}_r{r_pix}px.png"
                    _plot_cutout_with_circle(cut, r_pix, f"u={uid} | patch≈{patch_deg}° (z={z:.2f})", str(cut_path))
                    cut_png = str(cut_path)
                    cutout_paths.append(cut_png)

                # record metrics
                records.append({
                    "universe_id": uid, "H": H, "W": W,
                    "patch_deg": patch_deg, "r_pix": r_pix,
                    "global_mean": g_mean, "global_std": g_std,
                    "patch_mean": patch_mean, "z_score": z,
                    "is_anomalous": int(np.isfinite(z) and (z <= -abs(z_thresh))),
                    "cutout_png": cut_png, "cy": int(cy), "cx": int(cx),
                })

            except Exception as e:
                # keep going on per-map failure; store error for visibility
                records.append({
                    "universe_id": int(getattr(row, "universe_id", i)),
                    "H": None, "W": None,
                    "patch_deg": patch_deg, "r_pix": -1 if r_pix is None else r_pix,
                    "global_mean": np.nan, "global_std": np.nan,
                    "patch_mean": np.nan, "z_score": np.nan,
                    "is_anomalous": 0, "cutout_png": "", "cy": -1, "cx": -1,
                    "error": str(e),
                })
                continue
    else:
        # batch array mode (maps provided)
        maps = np.asarray(maps, dtype=np.float64)
        if maps.ndim != 3:
            raise ValueError("[COLD_SPOT] Expected maps shape (N,H,W).")
        N, H, W = maps.shape
        r_pix = _deg_to_pix_radius(patch_deg, H)
        K = _make_disk_kernel(r_pix); K /= max(1.0, K.sum())

        for i in range(N):
            M = maps[i]
            valid = np.isfinite(M)
            g_mean = float(np.mean(M[valid])) if valid.any() else np.nan
            g_std  = float(np.std(M[valid]))  if np.sum(valid) > 1 else np.nan
            
            fill = g_mean if np.isfinite(g_mean) else 0.0
            conv = _fft_convolve2d(np.nan_to_num(M, nan=fill), K)
            cy, cx = np.unravel_index(int(np.argmin(conv)), conv.shape)
            patch_mean = float(conv[cy, cx])
            z = (patch_mean - g_mean) / g_std if (g_std is not None and g_std > 0) else np.nan

            cut_png = ""
            if save_cutouts:
                cut = _extract_cutout(M, cy, cx, r_pix)
                cut_path = fig_dir / f"{tag}__cold_spot_u{i:05d}_r{r_pix}px.png"
                _plot_cutout_with_circle(cut, r_pix, f"u={i} | patch≈{patch_deg}° (z={z:.2f})", str(cut_path))
                cut_png = str(cut_path)
                cutout_paths.append(cut_png)

            records.append({
                "universe_id": i,
                "H": H, "W": W,
                "patch_deg": patch_deg, "r_pix": r_pix,
                "global_mean": g_mean, "global_std": g_std,
                "patch_mean": patch_mean, "z_score": z,
                "is_anomalous": int(np.isfinite(z) and (z <= -abs(z_thresh))),
                "cutout_png": cut_png, "cy": int(cy), "cx": int(cx),
            })

    # -> DataFrame + CSV
    df = pd.DataFrame.from_records(records)
    csv_path = run_dir / f"{tag}__anomaly_cold_spot_metrics.csv"
    if save_metrics_csv:
        df.to_csv(csv_path, index=False)

    # -> JSON summary
    n_anom = int(df["is_anomalous"].sum()) if len(df) else 0
    # -- safe stats számítás a summary előtt --
    zv = df["z_score"].to_numpy(dtype=float) if len(df) else np.array([])
    fin = np.isfinite(zv)
    z_min    = float(np.min(zv[fin]))    if fin.any() else float("nan")
    z_median = float(np.median(zv[fin])) if fin.any() else float("nan")
    z_mean   = float(np.mean(zv[fin]))   if fin.any() else float("nan")

    summary = {
        "env": paths["env"], "run_id": paths["run_id"], "mode": tag,
        "N": int(len(df)),
        "patch_deg": patch_deg, "z_thresh": z_thresh,
        "anomalous_count": n_anom,
        "anomalous_frac": float(n_anom / max(1, len(df))),
        "files": {
            "csv": str(csv_path) if save_metrics_csv else "",
            "cutouts": cutout_paths[:50],
        },
        "stats": {"z_min": z_min, "z_median": z_median, "z_mean": z_mean},
        }
    
    json_path = run_dir / f"{tag}__anomaly_cold_spot_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Mirror copies
    from shutil import copy2
    fig_sub = active_cfg.get("OUTPUTS", {}).get("local", {}).get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            mpath = pathlib.Path(m)
            mpath.mkdir(parents=True, exist_ok=True)
            if save_metrics_csv:
                copy2(csv_path, mpath / csv_path.name)
            copy2(json_path, mpath / json_path.name)
            if cutout_paths:
                m_fig = mpath / fig_sub
                m_fig.mkdir(parents=True, exist_ok=True)
                for fp in cutout_paths:
                    try:
                        copy2(fp, m_fig / os.path.basename(fp))
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[COLD_SPOT] Done → CSV/JSON/PNGs saved under:\n  {run_dir}")
    return {"csv": str(csv_path) if save_metrics_csv else "",
            "json": str(json_path),
            "plots": cutout_paths,
            "table": df}

# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_anomaly_cold_spot_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_anomaly_cold_spot(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_anomaly_cold_spot_stage(ACTIVE)
