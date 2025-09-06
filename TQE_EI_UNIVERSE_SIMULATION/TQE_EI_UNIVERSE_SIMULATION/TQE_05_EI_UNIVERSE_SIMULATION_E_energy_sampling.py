# ===================================================================================
# TQE_05_EI_UNIVERSE_SIMULATION_E_energy_sampling.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds

import os, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# 1) RNG helpers
# -----------------------------------------------------------------------------------

def _maybe_truncate(arr, low, high):
    """Clamp values if truncation bounds are given."""
    if low is not None:
        arr = np.maximum(arr, float(low))
    if high is not None:
        arr = np.minimum(arr, float(high))
    return arr

# -----------------------------------------------------------------------------------
# 2) Main routine
# -----------------------------------------------------------------------------------

def run_energy_sampling(active=ACTIVE, tag="E"):
    """
    Sample initial energies E0 for N universes using parameters under ACTIVE['ENERGY'].

    Outputs:
        - CSV: sampled E0 values
        - PNG: histogram plot of E0 distribution
        - JSON: summary with statistics and file paths

    Returns:
        dict with arrays, file paths, and run metadata.
    """
    cfgE = active["ENERGY"]

    n    = int(cfgE["num_universes"])
    mu   = float(cfgE.get("log_mu", 2.5))
    sig  = float(cfgE.get("log_sigma", 0.8))
    low  = cfgE.get("trunc_low", None)
    high = cfgE.get("trunc_high", None)
    seed = cfgE.get("seed", None)

    # Resolve output directories
    run_dir = RUN_DIR
    fig_dir = FIG_DIR
    mirrors = PATHS["mirrors"]
    run_id  = PATHS["run_id"]

    # RNG
    seeds_data = load_or_create_run_seeds(active)
    master_seed = seeds_data["master_seed"]
    rng = np.random.default_rng(master_seed)

    # Draw samples from lognormal distribution
    E0 = rng.lognormal(mean=mu, sigma=sig, size=n).astype(np.float64)
    E0 = _maybe_truncate(E0, low, high)

    # Save CSV
    df = pd.DataFrame({
        "universe_id": np.arange(n, dtype=np.int64),
        "E0": E0,
    })
    csv_path = os.path.join(run_dir, f"energy_samples__{tag}.csv")
    df.to_csv(csv_path, index=False)

    # Plot histogram
    plt.figure(dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    bins = ACTIVE["ENERGY_SAMPLING"].get("hist_bins", 80)
    plt.hist(E0, bins=bins, density=True, alpha=0.6)
    plt.title(f"E0 distribution ({tag}) — lognormal mu={mu}, sigma={sig}, n={n}")
    plt.xlabel("E0")
    plt.ylabel("Density")
    png_path = os.path.join(fig_dir, f"energy_hist__{tag}.png")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    # Build summary JSON
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

    # Mirror outputs if configured
    for mdir in mirrors:
        m_csv = os.path.join(mdir, os.path.basename(csv_path))
        fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
        mirror_fig_dir = os.path.join(mdir, fig_sub)
        pathlib.Path(mirror_fig_dir).mkdir(parents=True, exist_ok=True)

        df.to_csv(m_csv, index=False)
        from shutil import copy2
        copy2(png_path, os.path.join(mirror_fig_dir, os.path.basename(png_path)))
        summary["mirrors"].append({
            "csv": m_csv,
            "png": os.path.join(mirror_fig_dir, os.path.basename(png_path)),
        })

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

# -----------------------------------------------------------------------------------
# 3) Standalone execution
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    out = run_energy_sampling(ACTIVE, tag="E")
    print("[energy_sampling] done →", out["paths"])
