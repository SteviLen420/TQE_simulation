# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_06_EI_UNIVERSE_SIMULATION_I_information_bootstrap.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script executes the second data-generating stage of the pipeline, tasked with
# bootstrapping the initial "information" value (I_seed) for each universe. This
# value complements the initial energy (E0) generated in the preceding stage.
#
# The methodology models the initial information content as the normalized Shannon
# entropy of an underlying, unobserved state. For each universe, the script first
# generates a random probability vector of a specified dimension (`hilbert_dim`),
# typically using a Dirichlet distribution. It then calculates the entropy of this
# vector to yield a single scalar value for I_seed, normalized to the range [0, 1].
#
# Following the pipeline's standard for auditability, it produces three outputs:
# 1. A .csv file containing the generated I_seed value for each universe.
# 2. A .png histogram visualizing the distribution of these initial information values.
# 3. A .json summary file detailing the run parameters, statistics, and file paths.
#
# ===================================================================================

from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds

import os, json, math, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# RNG and small helpers
# ---------------------------

def _normalized_shannon(p):
    """Return H(p)/log(K) in [0,1] for probability vector p (K = len(p))."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    H = -np.sum(p * np.log(p))
    Hmax = math.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0


def _gen_probs_dirichlet(rng, K, alpha=1.05):
    """Sample a single probability vector from Dirichlet(alpha)."""
    a = np.full(K, float(alpha), dtype=np.float64)
    v = rng.gamma(shape=a, scale=1.0)
    return v / v.sum()


def _gen_probs_uniform(K):
    """Return a uniform probability vector of length K."""
    return np.full(K, 1.0 / K, dtype=np.float64)


# ---------------------------
# Main entry
# ---------------------------

def run_information_bootstrap(active=ACTIVE, tag="EIseed"):
    """
    Generate I_seed values for each universe using INFORMATION & INFORMATION_BOOTSTRAP.

    Returns:
        dict {
          "I_seed": np.ndarray (N,),
          "df": pandas.DataFrame,
          "paths": {"csv": str|None, "png": str|None, "json": str|None},
          "run": PATHS
        }
    """
    # Sizes
    n = int(active["ENERGY"]["num_universes"])
    K = int(active["INFORMATION"].get("hilbert_dim", 8))

    # Bootstrap sub-config with robust defaults
    bs_cfg = active.get("INFORMATION_BOOTSTRAP", {}) or {}
    prior  = bs_cfg.get("prior_type", "dirichlet")          # "dirichlet" | "uniform"
    alpha  = float(bs_cfg.get("dirichlet_alpha", 1.05))     # only for dirichlet
    floor  = float(bs_cfg.get("i_seed_floor", 0.0))
    expo   = float(bs_cfg.get("exponent", 1.0))
    bins   = int(bs_cfg.get("hist_bins", 60))
    seed_from_energy = bool(bs_cfg.get("seed_from_energy", True))

    # RNG from central seeder
    seeds_data = load_or_create_run_seeds(active)
    master_seed = seeds_data["master_seed"]
    rng = np.random.default_rng(master_seed)

    # Resolve outputs (paths are pre-resolved & stable for this run)
    run_dir = RUN_DIR
    fig_dir = FIG_DIR
    run_id  = PATHS["run_id"]
    mirrors = PATHS["mirrors"]

    # Ensure directories exist (safe even if already created)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Global output switches
    out_cfg = active.get("OUTPUTS", {})
    save_stage = out_cfg.get("save_per_stage", {}).get("information_bootstrap", True)
    mirroring_enabled = out_cfg.get("mirroring", {}).get("enabled", False)

    # Core computation
    I_seed   = np.empty(n, dtype=np.float64)
    flatness = np.empty(n, dtype=np.float64)  # std of probabilities (compact "shape" proxy)

    for i in range(n):
        if prior == "uniform":
            p = _gen_probs_uniform(K)
        else:
            p = _gen_probs_dirichlet(rng, K, alpha=alpha)

        Hn = _normalized_shannon(p)      # in [0,1]
        I  = max(Hn, floor)              # floor clamp
        if expo != 1.0:
            I = float(np.power(I, expo))

        I_seed[i]   = I
        flatness[i] = float(np.std(p))

    # Assemble DataFrame
    df = pd.DataFrame({
        "universe_id": np.arange(n, dtype=np.int64),
        "I_seed": I_seed,
        "flatness_std": flatness,
        "hilbert_dim": K,
        "prior": prior,
        "alpha": alpha if prior == "dirichlet" else np.nan,
    })

    # Paths (filled only if we actually save)
    csv_path  = os.path.join(run_dir, f"information_seed__{tag}.csv")  if save_stage else None
    png_path  = os.path.join(fig_dir, f"information_seed_hist__{tag}.png") if save_stage else None
    json_path = os.path.join(run_dir, f"information_seed_summary__{tag}.json") if save_stage else None

    # Write CSV
    if save_stage:
        df.to_csv(csv_path, index=False)

    # Plot histogram
    if save_stage:
        plt.figure(dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.hist(I_seed, bins=bins, density=True, alpha=0.7)
        plt.title(f"I_seed distribution ({tag}) — prior={prior}, K={K}")
        plt.xlabel("I_seed (normalized [0,1])")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

    # Build summary JSON
    summary = {
        "run_id": run_id,
        "tag": tag,
        "num_universes": n,
        "hilbert_dim": K,
        "prior": prior,
        "dirichlet_alpha": alpha if prior == "dirichlet" else None,
        "i_seed_floor": floor,
        "exponent": expo,
        "I_seed_stats": {
            "min": float(np.min(I_seed)),
            "max": float(np.max(I_seed)),
            "mean": float(np.mean(I_seed)),
            "median": float(np.median(I_seed)),
            "std": float(np.std(I_seed)),
            "p01": float(np.quantile(I_seed, 0.01)),
            "p99": float(np.quantile(I_seed, 0.99)),
        },
        "files": {"csv": csv_path, "png": png_path},
        "mirrors": [],
    }

    # Save JSON
    if save_stage:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Optional mirroring
    if save_stage and mirroring_enabled and mirrors:
        fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
        for mdir in mirrors:
            try:
                # CSV
                if csv_path:
                    m_csv = os.path.join(mdir, os.path.basename(csv_path))
                    df.to_csv(m_csv, index=False)
                else:
                    m_csv = None

                # PNG
                if png_path:
                    mirror_fig_dir = os.path.join(mdir, fig_sub)
                    pathlib.Path(mirror_fig_dir).mkdir(parents=True, exist_ok=True)
                    from shutil import copy2
                    copy2(png_path, os.path.join(mirror_fig_dir, os.path.basename(png_path)))
                    m_png = os.path.join(mirror_fig_dir, os.path.basename(png_path))
                else:
                    m_png = None

                summary["mirrors"].append({"csv": m_csv, "png": m_png})
            except Exception as e:
                print(f"[WARN] Mirroring failed → {mdir}: {e}")

        # Re-save JSON to include mirrors (optional)
        if json_path:
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            except Exception:
                pass

    return {
        "I_seed": I_seed,
        "df": df,
        "paths": {"csv": csv_path, "png": png_path, "json": json_path},
        "run": PATHS,
    }


# ---------------------------
# Standalone execution
# ---------------------------
if __name__ == "__main__":
    out = run_information_bootstrap(ACTIVE, tag="EIseed")
    print("[information_bootstrap] done →", out["paths"])
