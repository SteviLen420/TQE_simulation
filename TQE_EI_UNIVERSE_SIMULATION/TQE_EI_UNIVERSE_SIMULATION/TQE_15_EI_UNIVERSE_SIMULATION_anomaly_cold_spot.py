# ===================================================================================
# TQE_15_EI_UNIVERSE_SIMULATION_anomaly_cold_spot.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR

import os
import json
import math
import pathlib
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


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
    Inputs:
      x: (H, W) map
      k: (h, w) kernel
    Returns:
      y: (H, W) convolution result (no padding artifacts thanks to zero-padding).
    """
    H, W = x.shape
    h, w = k.shape
    # Zero-pad to at least H+h-1, W+w-1 then crop to (H, W)
    FH = H + h - 1
    FW = W + w - 1
    fX = np.fft.rfftn(x, s=(FH, FW))
    # Kernel must be zero-padded to same FFT size
    k_pad = np.zeros((FH, FW), dtype=np.float64)
    k_pad[:h, :w] = k
    fK = np.fft.rfftn(k_pad)
    y_full = np.fft.irfftn(fX * fK, s=(FH, FW))
    # 'same' crop centered
    y = y_full[(h - 1)//2:(h - 1)//2 + H, (w - 1)//2:(w - 1)//2 + W]
    return np.asarray(y, dtype=np.float64)


def _extract_cutout(img: np.ndarray, cy: int, cx: int, r: int) -> np.ndarray:
    """
    Extract a square cutout centered at (cy, cx) of radius r (size = 2r+1).
    Handles edges by clipping (result still size (2r+1, 2r+1), with partial data).
    """
    H, W = img.shape
    y0 = max(0, cy - r)
    x0 = max(0, cx - r)
    y1 = min(H, cy + r + 1)
    x1 = min(W, cx + r + 1)
    out = np.zeros((2 * r + 1, 2 * r + 1), dtype=img.dtype)
    oy0 = r - (cy - y0)
    ox0 = r - (cx - x0)
    out[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)] = img[y0:y1, x0:x1]
    return out


def _plot_cutout_with_circle(cut: np.ndarray, r: int, title: str, save_path: str):
    """Save a cutout PNG with a dashed circle showing the patch boundary."""
    plt.figure()
    plt.imshow(cut, origin="lower")
    theta = np.linspace(0, 2 * math.pi, 512)
    cy = cx = r
    ys = cy + r * np.sin(theta)
    xs = cx + r * np.cos(theta)
    plt.plot(xs, ys, "w--", linewidth=1.0, alpha=0.9, label="patch")
    plt.title(title)
    plt.axis("off")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(pad=0.1)
    plt.savefig(save_path, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.close()


# ---------------------------
# Core routine
# ---------------------------
def run_anomaly_cold_spot(active_cfg: Dict = ACTIVE,
                          maps: Optional[np.ndarray] = None,
                          arrays: Optional[Dict[str, np.ndarray]] = None):
    """
    Detect Cold Spot anomaly across a population of CMB maps.

    Inputs (optional):
      - maps: np.ndarray of shape (N, H, W) with CMB temperature maps
      - arrays: dict that may contain {"cmb_maps": (N,H,W)} as an alternative source

    Returns a dict with:
      - 'csv', 'json' paths
      - 'plots': list of PNG paths (cutouts)
      - 'table': metrics DataFrame (if pandas is available)
    """
    import pandas as pd  # local import to keep dependency optional elsewhere

    # Stage switches
    if not active_cfg["PIPELINE"].get("run_anomaly_scan", True):
        print("[COLD_SPOT] run_anomaly_scan=False → skipping.")
        return {}

    # Cold spot target toggle
    cold_spec = None
    for t in active_cfg["ANOMALY"].get("targets", []):
        if t.get("name") in {"cold_spot", "coldspot"} and t.get("enabled", False):
            cold_spec = t
            break
    if cold_spec is None:
        print("[COLD_SPOT] target disabled → skipping.")
        return {}

    # IO paths
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    # Tag for filenames based on EI/E mode
    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # Load maps if not provided
    if maps is None:
        if arrays and "cmb_maps" in arrays:
            maps = np.asarray(arrays["cmb_maps"], dtype=float)
        else:
            # Try to load from standard location produced by module 13
            npy_guess = run_dir / f"{tag}__cmb_maps.npy"
            if npy_guess.exists():
                maps = np.load(npy_guess)
            else:
                raise FileNotFoundError(
                    "[COLD_SPOT] No maps provided and could not find "
                    f"{npy_guess.name}. Pass maps or run CMB map generation first."
                )
    maps = np.asarray(maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError("[COLD_SPOT] Expected maps shape (N,H,W).")

    N, H, W = maps.shape

    # Read parameters from config
    patch_deg = float(cold_spec.get("patch_deg", 10.0))
    z_thresh = float(cold_spec.get("zscore_thresh", 3.0))
    save_cutouts = bool(active_cfg["ANOMALY"].get("save_cutouts", True))
    save_metrics_csv = bool(active_cfg["ANOMALY"].get("save_metrics_csv", True))

    # Build disk kernel in pixel units
    r_pix = _deg_to_pix_radius(patch_deg, H)
    K = _make_disk_kernel(r_pix)
    K = K / max(1.0, K.sum())  # average within the disk

    # Storage
    records = []
    cutout_paths = []

    # Process each universe
    for i in range(N):
        M = maps[i]

        # Global stats (exclude NaNs just in case)
        valid = np.isfinite(M)
        if not valid.any():
            g_mean, g_std = np.nan, np.nan
        else:
            g_mean = float(np.mean(M[valid]))
            g_std = float(np.std(M[valid])) if np.sum(valid) > 1 else np.nan

        # Disk-average map via convolution; get coldest (minimum) position
        conv = _fft_convolve2d(np.nan_to_num(M, nan=g_mean), K)
        min_idx = np.argmin(conv)
        cy, cx = np.unravel_index(min_idx, conv.shape)
        patch_mean = float(conv[cy, cx])

        # z-score of the patch mean against global distribution
        z = (patch_mean - g_mean) / g_std if (g_std is not None and g_std > 0) else np.nan

        # Optional cutout and PNG
        cut_png = None
        if save_cutouts:
            cut = _extract_cutout(M, cy, cx, r_pix)
            cut_path = fig_dir / f"{tag}__cold_spot_u{i:05d}_r{r_pix}px.png"
            _plot_cutout_with_circle(
                cut, r_pix,
                title=f"u={i} | Cold Spot patch ≈ {patch_deg}° (z={z:.2f})",
                save_path=str(cut_path)
            )
            cut_png = str(cut_path)
            cutout_paths.append(cut_png)

        records.append({
            "universe_id": i,
            "H": H,
            "W": W,
            "patch_deg": patch_deg,
            "r_pix": r_pix,
            "global_mean": g_mean,
            "global_std": g_std,
            "patch_mean": patch_mean,
            "z_score": z,
            "is_anomalous": int(np.isfinite(z) and (z <= -abs(z_thresh))),  # cold = negative z
            "cutout_png": cut_png if cut_png else "",
            "cy": int(cy),
            "cx": int(cx),
        })

    # -> DataFrame + CSV
    df = pd.DataFrame.from_records(records)
    csv_path = run_dir / f"{tag}__anomaly_cold_spot_metrics.csv"
    if save_metrics_csv:
        df.to_csv(csv_path, index=False)

    # -> JSON summary
    n_anom = int(df["is_anomalous"].sum())
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": N,
        "map_shape": [int(H), int(W)],
        "patch_deg": patch_deg,
        "z_thresh": z_thresh,
        "anomalous_count": n_anom,
        "anomalous_frac": float(n_anom / max(1, N)),
        "files": {
            "csv": str(csv_path) if save_metrics_csv else "",
            "cutouts": cutout_paths[:50],  # don't flood JSON; keep first 50
        },
        "stats": {
            "z_min": float(np.nanmin(df["z_score"])) if len(df) else np.nan,
            "z_median": float(np.nanmedian(df["z_score"])) if len(df) else np.nan,
            "z_mean": float(np.nanmean(df["z_score"])) if len(df) else np.nan,
        }
    }
    json_path = run_dir / f"{tag}__anomaly_cold_spot_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Mirror copies
    from shutil import copy2
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        try:
            if save_metrics_csv:
                copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            # copy cutouts into mirror/figs/
            m_fig = pathlib.Path(m) / fig_sub
            m_fig.mkdir(parents=True, exist_ok=True)
            for fp in cutout_paths:
                try:
                    copy2(fp, m_fig / os.path.basename(fp))
                except Exception as _:
                    pass
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[COLD_SPOT] Done → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {
        "csv": str(csv_path) if save_metrics_csv else "",
        "json": str(json_path),
        "plots": cutout_paths,
        "table": df,
    }


# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    run_anomaly_cold_spot(ACTIVE)
