montecarlo.py
# ===================================================================================
# Monte Carlo aggregation for the TQE universe simulation
# -----------------------------------------------------------------------------------
# - Loads results from previous pipeline stages (collapse, expansion, etc.)
# - Aggregates across universes: distributions of stability, lock-in, expansion growth
# - Provides histograms, summary stats, and saves to CSV/JSON
# - This stage does NOT pick the "best" universe yet, it only aggregates.
#
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted

import os, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Public API
# ---------------------------
def run_montecarlo(active_cfg: dict = ACTIVE,
                   collapse_df: pd.DataFrame = None,
                   expansion_df: pd.DataFrame = None):
    """
    Monte Carlo aggregation of universes.
    Inputs:
      - collapse_df: DataFrame from run_collapse (stable_at, lockin_at, final_L)
      - expansion_df: DataFrame from run_expansion (S0, S_final, growth_rate_eff)
    Returns:
      dict(csv, json, plots, table)
    """

    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag = "EI" if use_I else "E"

    # --- Join collapse + expansion if both are given ---
    if collapse_df is not None and expansion_df is not None:
        df = collapse_df.merge(expansion_df, on="universe_id", how="inner", suffixes=("_collapse", "_expansion"))
    elif collapse_df is not None:
        df = collapse_df.copy()
    elif expansion_df is not None:
        df = expansion_df.copy()
    else:
        print("[MONTECARLO] No input DataFrames provided!")
        return {}

    N = len(df)

    # --- Summary statistics ---
    def _stats(x):
        x = np.asarray(x, dtype=float)
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "median": float(np.median(x)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
        }

    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": N,
        "collapse": {},
        "expansion": {}
    }

    if "lockin_at" in df.columns:
        summary["collapse"]["lockin_at"] = _stats(df["lockin_at"][df["lockin_at"] >= 0])
    if "stable_at" in df.columns:
        summary["collapse"]["stable_at"] = _stats(df["stable_at"][df["stable_at"] >= 0])
    if "final_L" in df.columns:
        summary["collapse"]["final_L"] = _stats(df["final_L"])

    if "S_final" in df.columns:
        summary["expansion"]["S_final"] = _stats(df["S_final"])
    if "growth_rate_eff" in df.columns:
        summary["expansion"]["growth_rate_eff"] = _stats(df["growth_rate_eff"])

    # --- Save CSV and JSON ---
    csv_path = run_dir / f"{tag}__montecarlo.csv"
    df.to_csv(csv_path, index=False)

    json_path = run_dir / f"{tag}__montecarlo_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    figs = []

    # Histogram: lock-in epochs
    if "lockin_at" in df.columns:
        lk = df["lockin_at"][df["lockin_at"] >= 0]
        if len(lk) > 0:
            plt.figure()
            plt.hist(lk, bins=40, color="skyblue", edgecolor="black")
            plt.xlabel("Lock-in epoch")
            plt.ylabel("Count")
            plt.title("Distribution of lock-in epochs")
            f1 = fig_dir / f"{tag}__montecarlo_lockin_hist.png"
            plt.tight_layout()
            plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
            plt.close()
            figs.append(str(f1))

    # Histogram: final expansion sizes
    if "S_final" in df.columns:
        plt.figure()
        plt.hist(df["S_final"], bins=40, color="lightgreen", edgecolor="black")
        plt.xlabel("Final size S_final")
        plt.ylabel("Count")
        plt.title("Distribution of final expansion sizes")
        f2 = fig_dir / f"{tag}__montecarlo_expansion_hist.png"
        plt.tight_layout()
        plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f2))

    # Scatter: lock-in vs final expansion
    if "lockin_at" in df.columns and "S_final" in df.columns:
        plt.figure()
        plt.scatter(df["lockin_at"], df["S_final"], alpha=0.5, s=8)
        plt.xlabel("Lock-in epoch")
        plt.ylabel("Final size S_final")
        plt.title("Lock-in vs Expansion size")
        f3 = fig_dir / f"{tag}__montecarlo_scatter.png"
        plt.tight_layout()
        plt.savefig(f3, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(f3))

    # --- Mirror copies ---
    from shutil import copy2
    fig_sub = ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs")
    for m in mirrors:
        try:
            copy2(csv_path, os.path.join(m, csv_path.name))
            copy2(json_path, os.path.join(m, json_path.name))
            m_fig = pathlib.Path(m) / fig_sub
            m_fig.mkdir(parents=True, exist_ok=True)
            for fp in figs:
                copy2(fp, m_fig / os.path.basename(fp))
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")

    print(f"[MONTECARLO] mode={tag} → Aggregated CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {"csv": str(csv_path), "json": str(json_path), "plots": figs, "table": df}


if __name__ == "__main__":
    run_montecarlo(ACTIVE)
