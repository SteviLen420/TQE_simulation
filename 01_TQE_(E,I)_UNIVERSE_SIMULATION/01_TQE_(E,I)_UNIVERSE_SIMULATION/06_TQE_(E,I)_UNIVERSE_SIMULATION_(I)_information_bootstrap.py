# ===================================================================================
# 06_TQE_(E,I)_UNIVERSE_SIMULATION_(I)_information_bootstrap.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths

import os, json, math, time, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _get_rng(seed=None):
    if seed is None:
        ss = np.random.SeedSequence()
        return np.random.default_rng(ss)
    return np.random.default_rng(int(seed))

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

def run_information_bootstrap(active=ACTIVE, tag="EIseed"):
    """
    Build I_seed for each universe based on INFORMATION + INFORMATION_BOOTSTRAP
    (if bootstrap block is missing, sensible defaults are applied).
    Saves CSV/PNG/JSON and returns I_seed as array of shape (N,).
    """
    # Core sizes
    n  = int(active["ENERGY"]["num_universes"])
    K  = int(active["INFORMATION"].get("hilbert_dim", 8))

    # Bootstrap sub-config with defaults
    bs_cfg = active.get("INFORMATION_BOOTSTRAP", {}) or {}
    prior  = bs_cfg.get("prior_type", "dirichlet")      # "dirichlet" | "uniform"
    alpha  = float(bs_cfg.get("dirichlet_alpha", 1.05)) # only for dirichlet
    floor  = float(bs_cfg.get("i_seed_floor", 0.0))
    expo   = float(bs_cfg.get("exponent", 1.0))
    seed   = active["ENERGY"].get("seed", None)  # reuse energy seed for coupling if present

    # Resolve outputs
    paths  = resolve_output_paths(active)
    run_dir = paths["primary_run_dir"]
    fig_dir = paths["fig_dir"]
    mirrors = paths["mirrors"]
    run_id  = paths["run_id"]

    rng = _get_rng(seed)

    # Generate I_seed per universe
    I_seed = np.empty(n, dtype=np.float64)

    # (Optional) also store a compact "flatness" metric from the probs: std deviation
    flatness = np.empty(n, dtype=np.float64)

    for i in range(n):
        if prior == "uniform":
            p = _gen_probs_uniform(K)
        else:
            p = _gen_probs_dirichlet(rng, K, alpha=alpha)

        Hn = _normalized_shannon(p)   # in [0,1]
        I  = max(Hn, floor)           # floor clamp
        if expo != 1.0:
            I = float(np.power(I, expo))

        I_seed[i]   = I
        flatness[i] = float(np.std(p))

    # Save CSV
    df = pd.DataFrame({
        "universe_id": np.arange(n, dtype=np.int64),
        "I_seed": I_seed,
        "flatness_std": flatness,
        "hilbert_dim": K,
        "prior": prior,
        "alpha": alpha if prior == "dirichlet" else np.nan,
    })
    csv_path = os.path.join(run_dir, f"information_seed__{tag}.csv")
    df.to_csv(csv_path, index=False)

    # Plot histogram
    plt.figure(dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
    plt.hist(I_seed, bins=60, density=True, alpha=0.7)
    plt.title(f"I_seed distribution ({tag}) — prior={prior}, K={K}")
    plt.xlabel("I_seed (normalized [0,1])")
    plt.ylabel("Density")
    png_path = os.path.join(fig_dir, f"information_seed_hist__{tag}.png")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    # Summary JSON
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

    # Mirror copies
    for mdir in mirrors:
        # CSV
        m_csv = os.path.join(mdir, os.path.basename(csv_path))
        df.to_csv(m_csv, index=False)
        # PNG
        fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
        mirror_fig_dir = os.path.join(mdir, fig_sub)
        pathlib.Path(mirror_fig_dir).mkdir(parents=True, exist_ok=True)
        from shutil import copy2
        copy2(png_path, os.path.join(mirror_fig_dir, os.path.basename(png_path)))

        summary["mirrors"].append({
            "csv": m_csv,
            "png": os.path.join(mirror_fig_dir, os.path.basename(png_path))
        })

    json_path = os.path.join(run_dir, f"information_seed_summary__{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "I_seed": I_seed,
        "df": df,
        "paths": {
            "csv": csv_path,
            "png": png_path,
            "json": json_path,
        },
        "run": paths,
    }

if __name__ == "__main__":
    out = run_information_bootstrap(ACTIVE, tag="EIseed")
    print("[information_bootstrap] done →", out["paths"])
