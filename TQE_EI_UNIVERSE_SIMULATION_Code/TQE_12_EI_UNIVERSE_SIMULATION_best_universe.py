# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_12_EI_UNIVERSE_SIMULATION_best_universe.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script serves as a ranking and selection module designed to identify the
# "best-performing" universes from the entire simulated population. It synthesizes
# results from previous stages to score and rank each universe.
#
# The core logic calculates a composite "score" for each universe by creating a
# weighted sum of several key performance metrics. These metrics, defined in the
# configuration, typically include the final size of the universe (growth), the
# speed at which its laws stabilized (speed), and whether it achieved stability at
# all. The script normalizes these metrics to ensure they are combined fairly
# before ranking the entire population from best to worst.
#
# A key feature is its ability to re-simulate and plot the evolutionary
# trajectories of the top-K ranked universes. This provides a visual narrative of
# how the most successful universes evolved, offering more insight than a simple
# table of results. The outputs include a ranked .csv file, a .json summary of
# the winner, and detailed plots for the top performers.
#
# ===================================================================================

from typing import Dict, Optional
import os, json, math, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + resolved paths (stable run_id during one pipeline run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR
from TQE_04_EI_UNIVERSE_SIMULATION_seeding import load_or_create_run_seeds


def _simulate_law_trajectory(X_row: float, epochs: int, rng: np.random.Generator, cfg: dict):
    """Recreate a lock-in-like trajectory for a single universe (visualization only)."""
    noise = cfg["NOISE"]

    sigma0     = float(noise.get("exp_noise_base", 0.12))
    ll_floor   = float(noise.get("ll_base_noise", 8e-4))
    tau        = float(noise.get("decay_tau", 500))
    floor_frac = float(noise.get("floor_frac", 0.25))

    Xn = max(0.0, float(X_row))
    xnorm = Xn / (1.0 + Xn)
    sX = 1.0 / (1.0 + 2.0 * xnorm)

    L = np.empty(epochs, dtype=float)
    L[0] = max(1e-9, Xn)

    for t in range(1, epochs):
        decay = math.exp(-t / tau)
        sigma_t = max(ll_floor, sigma0 * (floor_frac + (1.0 - floor_frac) * decay))
        sigma_eff = sigma_t * sX
        kappa = 0.02  # small OU-like drift toward Xn
        drift = kappa * (Xn - L[t - 1])
        eps = rng.normal(0.0, sigma_eff)
        L[t] = max(1e-12, L[t - 1] + drift + eps)
    return L


# --- Utility: safe z-score ---
def _z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return z-scores with a small epsilon to avoid division by ~0."""
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    sd = sd if sd > eps else eps
    return (x - mu) / sd


# --- Public API ---
def run_best_universe(active_cfg: Dict = ACTIVE,
                      collapse_df: Optional[pd.DataFrame] = None,
                      expansion_df: Optional[pd.DataFrame] = None,
                      montecarlo_df: Optional[pd.DataFrame] = None):
    """
    Rank universes and export the best one (+Top-K plots if requested).

    Inputs:
      - collapse_df: expects columns ['universe_id','lockin_at','stable',...]
      - expansion_df: expects columns ['universe_id','S_final','X','S0',...]
      - montecarlo_df: optional; may provide ['X','E0'] for additional context.

    Returns:
      dict(csv, json, plots, top_table, best_row)
    """

    if not active_cfg["PIPELINE"].get("run_best_universe", True):
        print("[BEST] run_best_universe=False → skipping.")
        return {}

    # Use cached paths (consistent run_id)
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths.get("mirrors", [])

    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Seeding (with safe fallback)
    seeds = load_or_create_run_seeds(active_cfg)
    uni_seeds = seeds.get("universe_seeds", [])
    if not isinstance(uni_seeds, (list, tuple, np.ndarray)) or len(uni_seeds) == 0:
        uni_seeds = [int(seeds.get("master_seed", 1234567))]

    # EI/E tag prefix
    ei_tag_enabled = active_cfg["OUTPUTS"].get("tag_ei_in_filenames", True)
    use_info       = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag_prefix     = ("EI__" if use_info else "E__") if ei_tag_enabled else ""

    # --- Collect inputs ---
    cols_cfg = active_cfg.get("BEST_UNIVERSE", {}).get("columns", {})
    col_id   = cols_cfg.get("id", "universe_id")
    col_sfin = cols_cfg.get("s_final", "S_final")
    col_lk   = cols_cfg.get("lockin", "lockin_at")
    col_stab = cols_cfg.get("stable_flag", "stable")

    parts = []
    if isinstance(collapse_df, pd.DataFrame):
        parts.append(collapse_df[[c for c in [col_id, col_lk, col_stab] if c in collapse_df.columns]])
    if isinstance(expansion_df, pd.DataFrame):
        cols = [c for c in [col_id, col_sfin, "X", "S0"] if c in expansion_df.columns]
        if cols:
            parts.append(expansion_df[cols])
    if isinstance(montecarlo_df, pd.DataFrame):
        cols = [c for c in [col_id, "X", "E0"] if c in montecarlo_df.columns]
        if cols:
            parts.append(montecarlo_df[cols])

    if not parts:
        print("[BEST] No input tables provided. Nothing to rank.")
        return {}

    # Outer-join all provided parts on universe_id
    df = parts[0].copy()
    for p in parts[1:]:
        df = pd.merge(df, p, on=col_id, how="outer")

    # --- Compute weighted score ---
    w = active_cfg.get("BEST_UNIVERSE", {}).get("weights", {})
    w_growth = float(w.get("growth", 1.0))
    w_speed  = float(w.get("speed", 0.7))
    w_stab   = float(w.get("stability", 0.3))
    eps      = float(active_cfg.get("BEST_UNIVERSE", {}).get("eps", 1e-9))

    s_final = df[col_sfin].to_numpy(float) if col_sfin in df.columns else np.full(len(df), np.nan)
    lockin  = df[col_lk].to_numpy(float)   if col_lk in df.columns else np.full(len(df), np.nan)
    stable  = df[col_stab].to_numpy(float) if col_stab in df.columns else np.zeros(len(df), dtype=float)

    # smaller lock-in epoch is better → negate before z-scoring
    speed_term  = _z(-np.where(np.isfinite(lockin), lockin, np.nan), eps)
    growth_term = _z(s_final, eps)
    stab_term   = stable.astype(float)

    score = w_growth * growth_term + w_speed * speed_term + w_stab * stab_term
    df["score"] = score

    # Rank high→low
    df_rank = df.sort_values("score", ascending=False).reset_index(drop=True)

    # --- Handle empty result early ---
    if df_rank.empty:
        print("[BEST] No rows to rank after merge. Check inputs/keys.")
        return {
            "csv": None,
            "json": None,
            "plots": [],
            "top_table": df_rank,
            "best_row": {},
        }

    best = df_rank.iloc[0:1].copy()

    # --- Save CSV/JSON ---
    csv_path = run_dir / f"{tag_prefix}best_universe_ranked.csv"
    df_rank.to_csv(csv_path, index=False)

    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": "EI" if use_info else "E",
        "N": int(len(df_rank)),
        "winner": best.to_dict(orient="records")[0],
        "weights": {"growth": w_growth, "speed": w_speed, "stability": w_stab},
        "files": {"csv": str(csv_path)},
    }
    json_path = run_dir / f"{tag_prefix}best_universe_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Plots: Top-K lock-in-style trajectories and overview ---
    figs = []
    plot_cfg = active_cfg.get("BEST_UNIVERSE", {}).get("plot", {})
    dpi  = int(plot_cfg.get("dpi", 180))
    annot_color = plot_cfg.get("annot_color", "red")

    top_k  = int(active_cfg.get("BEST_UNIVERSE", {}).get("top_k_png", 5))
    epochs = int(active_cfg["ENERGY"].get("lockin_epochs", 500))

    if top_k > 0:
        k = min(top_k, len(df_rank))
        for idx in range(k):
            row = df_rank.iloc[idx]
            uid = int(row[col_id])

            # prefer X, fall back to E0, otherwise 1.0
            if "X" in row.index and np.isfinite(row["X"]):
                X_val = float(row["X"])
            elif "E0" in row.index and np.isfinite(row["E0"]):
                X_val = float(row["E0"])
            else:
                X_val = 1.0

            uni_seed = int(uni_seeds[uid % len(uni_seeds)])
            rng_universe = np.random.default_rng(uni_seed)
            L = simulate_law_trajectory(X_val, epochs, rng_universe, active_cfg)

            plt.figure()
            plt.plot(L, label=f"Universe {uid}")
            plt.xlabel("epoch")
            plt.ylabel("law amplitude")
            plt.title(f"Top {idx+1} — score={row['score']:.3f}")

            # --- use configured column names instead of hardcoded ones ---
            if (col_sfin in row.index) and pd.notna(row[col_sfin]) and np.isfinite(row[col_sfin]):
                plt.axhline(y=float(row[col_sfin]), color="gray", linestyle="--", linewidth=1)
            if (col_lk in row.index) and pd.notna(row[col_lk]) and np.isfinite(row[col_lk]):
                plt.axvline(x=float(row[col_lk]), color=annot_color, linestyle="--", linewidth=1,
                            label=f"lock-in≈{int(row[col_lk])}")
                plt.legend()

            fig_p = fig_dir / f"{tag_prefix}best_u{uid:05d}.png"
            plt.tight_layout()
            plt.savefig(fig_p, dpi=dpi)
            plt.close()
            figs.append(str(fig_p))

        # Overview barplot for Top-K scores
        plt.figure()
        plt.bar(range(k), df_rank["score"].head(k))
        plt.xticks(range(k), [int(v) for v in df_rank[col_id].head(k)], rotation=0)
        plt.xlabel("universe_id")
        plt.ylabel("score")
        plt.title(f"Top-{k} universes by score")
        fig_over = fig_dir / f"{tag_prefix}best_top{k}_overview.png"
        plt.tight_layout()
        plt.savefig(fig_over, dpi=dpi)
        plt.close()
        figs.append(str(fig_over))

    # --- Mirror outputs (CSV/JSON to root; PNGs into <mirror>/<fig_subdir>/) ---
    from shutil import copy2
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors or []:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            m_fig = pathlib.Path(m) / fig_sub
            m_fig.mkdir(parents=True, exist_ok=True)
            for fp in figs:
                copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[BEST] mode={'EI' if use_info else 'E'} → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "plots": figs,
        "top_table": df_rank,
        "best_row": best.iloc[0].to_dict(),
    }

# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_best_universe_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_best_universe(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_best_universe_stage(ACTIVE)
