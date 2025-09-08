# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_11_EI_UNIVERSE_SIMULATION_montecarlo.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script functions as the primary analysis and aggregation stage for the
# entire Monte Carlo simulation. It does not perform new simulations but instead
# synthesizes and summarizes the results from the preceding `collapse` and
# `expansion` stages.
#
# Its core operation is to take the output DataFrames from the previous two stages
# and merge them into a single, comprehensive master table that describes the
# full lifecycle of every simulated universe.
#
# The script then calculates aggregate summary statistics (e.g., mean, median,
# quartiles) for key outcome variables across the entire population, such as
# lock-in time and final universe size. It generates high-level visualizations,
# including histograms and scatter plots, to reveal the overall behavior of the
# model and explore correlations between different phases of evolution. The final
# outputs (a master .csv, a summary .json, and plots) represent a consolidated
# overview of the entire simulation run.
#
# ===================================================================================

from typing import Dict, Optional
import os, json, pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cached config + paths for the current run (stable run_id within a run)
from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR


# ---------------------------
# Small helpers
# ---------------------------

def _stats_safe(x: np.ndarray) -> Dict:
    """Return robust summary stats. Handles empty inputs gracefully."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None,
                "median": None, "p25": None, "p75": None}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
    }


def _mirror_file(src: pathlib.Path, mirrors: list, put_in_figs: bool, cfg: dict) -> None:
    """Copy file to mirror roots (CSV/JSON) or mirror/figs (PNGs)."""
    if not mirrors:
        return
    fig_sub = cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        try:
            mpath = pathlib.Path(m)
            if put_in_figs:
                (mpath / fig_sub).mkdir(parents=True, exist_ok=True)
                (mpath / fig_sub / src.name).write_bytes(src.read_bytes())
            else:
                mpath.mkdir(parents=True, exist_ok=True)
                (mpath / src.name).write_bytes(src.read_bytes())
        except Exception as e:
            print(f"[WARN] Mirror copy failed → {m}: {e}")


# ---------------------------
# Public API
# ---------------------------

def run_montecarlo(active_cfg: dict = ACTIVE,
                   collapse_df: Optional[pd.DataFrame] = None,
                   expansion_df: Optional[pd.DataFrame] = None):
    """
    Monte Carlo aggregation across universes.

    Inputs:
      - collapse_df: output of run_collapse (e.g., stable_at, lockin_at, final_L, X)
      - expansion_df: output of run_expansion (e.g., S0, S_final, growth_rate_eff)

    Returns:
      dict(csv, json, plots, table)
    """

    # Use cached, already-resolved paths (keeps run_id stable across stages)
    paths   = PATHS
    run_dir = pathlib.Path(RUN_DIR)
    fig_dir = pathlib.Path(FIG_DIR)
    mirrors = paths.get("mirrors", [])

    # Filename tag prefix (EI__/E__) if requested
    ei_tag_enabled = active_cfg["OUTPUTS"].get("tag_ei_in_filenames", True)
    use_info       = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag_prefix     = ("EI__" if use_info else "E__") if ei_tag_enabled else ""

    # Stage-level save switches, aligned with other modules
    per_stage          = active_cfg["OUTPUTS"].get("save_per_stage", {})
    save_stage         = bool(per_stage.get("montecarlo", True))
    save_csv           = save_stage and bool(active_cfg["OUTPUTS"].get("save_csv", True))
    save_json          = save_stage and bool(active_cfg["OUTPUTS"].get("save_json", True))
    save_figs          = save_stage and bool(active_cfg["OUTPUTS"].get("save_figs", True))
    mirroring_enabled  = bool(active_cfg["OUTPUTS"].get("mirroring", {}).get("enabled", True))

    # --- Join collapse + expansion if both are given ---
    if collapse_df is not None and expansion_df is not None:
        df = collapse_df.merge(
            expansion_df, on="universe_id", how="inner",
            suffixes=("_collapse", "_expansion")
        )
    elif collapse_df is not None:
        df = collapse_df.copy()
    elif expansion_df is not None:
        df = expansion_df.copy()
    else:
        print("[MONTECARLO] No input DataFrames provided!")
        return {}

    N = len(df)

    # --- Summary JSON payload ---
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": "EI" if use_info else "E",
        "N": N,
        "collapse": {},
        "expansion": {},
    }

    # Collapse stats (only if columns exist)
    if "lockin_at" in df.columns:
        lk = np.asarray(df["lockin_at"])
        summary["collapse"]["lockin_at"] = _stats_safe(lk[lk >= 0])
    if "stable_at" in df.columns:
        st = np.asarray(df["stable_at"])
        summary["collapse"]["stable_at"] = _stats_safe(st[st >= 0])
    if "final_L" in df.columns:
        summary["collapse"]["final_L"] = _stats_safe(df["final_L"])

    # Expansion stats
    if "S_final" in df.columns:
        summary["expansion"]["S_final"] = _stats_safe(df["S_final"])
    if "growth_rate_eff" in df.columns:
        summary["expansion"]["growth_rate_eff"] = _stats_safe(df["growth_rate_eff"])

    # --- Save CSV and JSON ---
    csv_path = run_dir / f"{tag_prefix}montecarlo.csv"
    json_path = run_dir / f"{tag_prefix}montecarlo_summary.json"

    if save_csv:
        df.to_csv(csv_path, index=False)
        if mirroring_enabled:
            _mirror_file(csv_path, mirrors, put_in_figs=False, cfg=active_cfg)

    if save_json:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if mirroring_enabled:
            _mirror_file(json_path, mirrors, put_in_figs=False, cfg=active_cfg)

    # --- Plots ---
    figs = []
    dpi = int(ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))

    if save_figs:
        # Histogram: lock-in epochs
        if "lockin_at" in df.columns:
            lk = np.asarray(df["lockin_at"])
            lk = lk[lk >= 0]
            if lk.size > 0:
                plt.figure()
                plt.hist(lk, bins=40)
                plt.xlabel("Lock-in epoch")
                plt.ylabel("Count")
                plt.title("Distribution of lock-in epochs")
                f1 = fig_dir / f"{tag_prefix}montecarlo_lockin_hist.png"
                plt.tight_layout()
                plt.savefig(f1, dpi=dpi)
                plt.close()
                figs.append(str(f1))
                if mirroring_enabled:
                    _mirror_file(f1, mirrors, put_in_figs=True, cfg=active_cfg)

        # Histogram: final expansion sizes
        if "S_final" in df.columns and np.isfinite(df["S_final"]).any():
            plt.figure()
            plt.hist(df["S_final"], bins=40)
            plt.xlabel("Final size S_final")
            plt.ylabel("Count")
            plt.title("Distribution of final expansion sizes")
            f2 = fig_dir / f"{tag_prefix}montecarlo_expansion_hist.png"
            plt.tight_layout()
            plt.savefig(f2, dpi=dpi)
            plt.close()
            figs.append(str(f2))
            if mirroring_enabled:
                _mirror_file(f2, mirrors, put_in_figs=True, cfg=active_cfg)

        # Scatter: lock-in vs final expansion
        if "lockin_at" in df.columns and "S_final" in df.columns:
            lk = np.asarray(df["lockin_at"], dtype=float)
            sf = np.asarray(df["S_final"], dtype=float)
            ok = np.isfinite(lk) & np.isfinite(sf) & (lk >= 0)
            if np.any(ok):
                plt.figure()
                plt.scatter(lk[ok], sf[ok], alpha=0.5, s=8)
                plt.xlabel("Lock-in epoch")
                plt.ylabel("Final size S_final")
                plt.title("Lock-in vs Expansion size")
                # Optional: log-scale for highly skewed S_final
                if np.nanmax(sf[ok]) / max(1e-12, np.nanmin(sf[ok])) > 1e3:
                    plt.yscale("log")
                f3 = fig_dir / f"{tag_prefix}montecarlo_scatter.png"
                plt.tight_layout()
                plt.savefig(f3, dpi=dpi)
                plt.close()
                figs.append(str(f3))
                if mirroring_enabled:
                    _mirror_file(f3, mirrors, put_in_figs=True, cfg=active_cfg)

    print(f"[MONTECARLO] mode={'EI' if use_info else 'E'} → Saved under:\n  {run_dir}")

    return {"csv": str(csv_path) if save_csv else None,
            "json": str(json_path) if save_json else None,
            "plots": figs,
            "table": df}


# Thin wrapper to match Master Control entrypoint
def run_montecarlo(active=None, active_cfg=None, collapse_df=None, expansion_df=None, **_):
    cfg = active if active is not None else active_cfg


if __name__ == "__main__":
    run_montecarlo(ACTIVE)
