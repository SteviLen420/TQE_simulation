# ===================================================================================
# 05_TQE_EI_UNIVERSE_SIMULATION_E_energy_sampling.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from 03_TQE_EI_UNIVERSE_SIMULATION_imports import PATHS, RUN_DIR, FIG_DIR

import os, json, math, time, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _get_rng(seed=None):
    """Create a NumPy Generator; if seed is None, use SeedSequence for fresh entropy."""
    if seed is None:
        ss = np.random.SeedSequence()
        return np.random.default_rng(ss)
    return np.random.default_rng(int(seed))

def _maybe_truncate(arr, low, high):
    """Clamp values if low/high truncation thresholds are provided."""
    if low is not None:
        arr = np.maximum(arr, float(low))
    if high is not None:
        arr = np.minimum(arr, float(high))
    return arr

def _kde_smooth(x, bins=80):
    """Simple histogram centers for plotting; leave true KDE to mpl's density display."""
    hist, edges = np.histogram(x, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

def run_energy_sampling(active=ACTIVE, tag="E"):
    """
    Sample initial energies E0 for NUM_UNIVERSES using params under ACTIVE['ENERGY'].
    Returns a dict with arrays and file paths; also writes CSV/PNG/JSON to disk.
    """
    cfgE = active["ENERGY"]

    n    = int(cfgE["num_universes"])
    mu   = float(cfgE.get("log_mu", 2.5))
    sig  = float(cfgE.get("log_sigma", 0.8))
    low  = cfgE.get("trunc_low", None)
    high = cfgE.get("trunc_high", None)
    seed = cfgE.get("seed", None)

    # Resolve output directories for this run
    run_dir = RUN_DIR
    fig_dir = FIG_DIR
    mirrors = PATHS["mirrors"]
    run_id  = PATHS["run_id"]

    # RNG
    rng = _get_rng(seed)

    # Sample lognormal
    # Note: NumPy's lognormal takes mean/sigma of the underlying normal.
    E0 = rng.lognormal(mean=mu, sigma=sig, size=n).astype(np.float64)
    E0 = _maybe_truncate(E0, low, high)

    # Build DataFrame and save CSV
    df = pd.DataFrame({
        "universe_id": np.arange(n, dtype=np.int64),
        "E0": E0,
    })

    csv_path = os.path.join(run_dir, f"energy_samples__{tag}.csv")
    df.to_csv(csv_path, index=False)

    # Plot histogram with density
    plt.figure(dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.hist(E0, bins=80, density=True, alpha=0.6)
    plt.title(f"E0 distribution ({tag}) — lognormal mu={mu}, sigma={sig}, n={n}")
    plt.xlabel("E0")
    plt.ylabel("Density")
    png_path = os.path.join(fig_dir, f"energy_hist__{tag}.png")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    # Summary JSON
    summary = {
        "run_id": run_id,
        "tag": tag,
        "num_universes": n,
        "distribution": "lognormal",
        "log_mu": mu,
        "log_sigma": sig,
        "trunc_low": low,
        "trunc_high": high,
        "E0_stats": {
            "min": float(np.min(E0)),
            "max": float(np.max(E0)),
            "mean": float(np.mean(E0)),
            "median": float(np.median(E0)),
            "std": float(np.std(E0)),
            "p01": float(np.quantile(E0, 0.01)),
            "p99": float(np.quantile(E0, 0.99)),
        },
        "files": {
            "csv": csv_path,
            "png": png_path,
        },
        "mirrors": [],
    }

    # Mirror to additional targets (if any)
    for mdir in mirrors:
        m_csv = os.path.join(mdir, os.path.basename(csv_path))
        # ensure mirror fig subdir exists (same subdir name as primary 'figs')
        fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
        mirror_fig_dir = os.path.join(mdir, fig_sub)
        pathlib.Path(mirror_fig_dir).mkdir(parents=True, exist_ok=True)
        # write files
        df.to_csv(m_csv, index=False)
        from shutil import copy2
        copy2(png_path, os.path.join(mirror_fig_dir, os.path.basename(png_path)))
        summary["mirrors"].append({"csv": m_csv, "png": os.path.join(mirror_fig_dir, os.path.basename(png_path))})

    # Save JSON
    json_path = os.path.join(run_dir, f"energy_summary__{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "E0": E0,
        "df": df,
        "paths": {
            "csv": csv_path,
            "png": png_path,
            "json": json_path,
        },
        "run": PATHS,
    }

if __name__ == "__main__":
    # Default E-only sampling when executed standalone
    out = run_energy_sampling(ACTIVE, tag="E")
    print("[energy_sampling] done →", out["paths"])
