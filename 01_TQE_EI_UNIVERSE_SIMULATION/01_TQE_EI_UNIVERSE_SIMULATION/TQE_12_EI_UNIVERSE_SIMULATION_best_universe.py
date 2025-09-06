# ===================================================================================
# TQE_12_EI_UNIVERSE_SIMULATION_best_universe.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted
from seeding import load_or_create_run_seeds, universe_rngs

import os, json, math, pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Minimal re-implementation of the lock-in trajectory (for plotting) ---
def _rng(seed: Optional[int]):
    """Return a reproducible Generator if seed is given, else fresh entropy."""
    return np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

def _simulate_law_trajectory(X_row: float, epochs: int, seed: Optional[int], cfg: dict):
    """Recreate a lock-in like path for a single universe (purely for visualization)."""
    stab   = cfg["STABILITY"]
    noise  = cfg["NOISE"]

    sigma0    = float(noise.get("exp_noise_base", 0.12))
    ll_floor  = float(noise.get("ll_base_noise", 8e-4))
    tau       = float(noise.get("decay_tau", 500))
    floor_frac= float(noise.get("floor_frac", 0.25))

    Xn = max(0.0, float(X_row))
    xnorm = Xn / (1.0 + Xn)
    sX = 1.0 / (1.0 + 2.0 * xnorm)

    rng = _rng(seed)
    L = np.empty(epochs, dtype=float)
    L[0] = max(1e-9, Xn)

    for t in range(1, epochs):
        decay = math.exp(-t / tau)
        sigma_t = max(ll_floor, sigma0 * (floor_frac + (1.0 - floor_frac) * decay))
        sigma_eff = sigma_t * sX
        kappa = 0.02
        drift = kappa * (Xn - L[t-1])
        eps = rng.normal(0.0, sigma_eff)
        L[t] = max(1e-12, L[t-1] + drift + eps)
    return L


# --- Utility: safe z-score ---
def _z(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return z-scores with small epsi for stability."""
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
    Rank universes and export the winner (+Top-K PNGs if requested).

    Inputs:
      - collapse_df: expects columns ['universe_id','lockin_at','stable',...]
      - expansion_df: expects columns ['universe_id','S_final',...]
      - montecarlo_df: optional; if provided and has 'X'/'E0', it is joined too.

    Returns: dict(csv, json, plots, top_table, best_row)
    """

    if not active_cfg["PIPELINE"].get("run_best_universe", True):
        print("[BEST] run_best_universe=False → skipping.")
        return {}

    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]

    seeds = load_or_create_run_seeds(active_cfg)
    uni_rngs = universe_rngs(seeds["universe_seeds"])

    use_I = bool(active_cfg["PIPELINE"].get("use_information", True))
    tag   = "EI" if use_I else "E"

    # --- Collect inputs (fail-soft defaults) ---
    cols_cfg = active_cfg.get("BEST_UNIVERSE", {}).get("columns", {})
    col_id   = cols_cfg.get("id", "universe_id")
    col_sfin = cols_cfg.get("s_final", "S_final")
    col_lk   = cols_cfg.get("lockin", "lockin_at")
    col_stab = cols_cfg.get("stable_flag", "stable")

    # join collapse + expansion on universe_id
    parts = []
    if isinstance(collapse_df, pd.DataFrame):
        parts.append(collapse_df[[c for c in [col_id, col_lk, col_stab] if c in collapse_df.columns]])
    if isinstance(expansion_df, pd.DataFrame):
        cols = [c for c in [col_id, col_sfin, "X", "S0"] if c in expansion_df.columns]
        if cols: parts.append(expansion_df[cols])
    if isinstance(montecarlo_df, pd.DataFrame):
        cols = [c for c in [col_id, "X", "E0"] if c in montecarlo_df.columns]
        if cols: parts.append(montecarlo_df[cols])

    if not parts:
        print("[BEST] No input tables provided. Nothing to rank.")
        return {}

    # incremental join
    df = parts[0].copy()
    for p in parts[1:]:
        df = pd.merge(df, p, on=col_id, how="outer")

    # --- Compute score ---
    w = active_cfg.get("BEST_UNIVERSE", {}).get("weights", {})
    w_growth = float(w.get("growth", 1.0))
    w_speed  = float(w.get("speed", 0.7))
    w_stab   = float(w.get("stability", 0.3))

    eps = float(active_cfg.get("BEST_UNIVERSE", {}).get("eps", 1e-9))

    # needed arrays (with safe fallbacks)
    s_final = df[col_sfin].to_numpy(float) if col_sfin in df.columns else np.full(len(df), np.nan)
    lockin  = df[col_lk].to_numpy(float)   if col_lk in df.columns else np.full(len(df), np.nan)
    stable  = df[col_stab].to_numpy(float) if col_stab in df.columns else np.zeros(len(df), dtype=float)

    # convert lockin to "speed": smaller is better → negate & z-score only finite values
    lk_pos = np.where(np.isfinite(lockin), lockin, np.nan)
    speed_term  = _z(-lk_pos, eps)
    growth_term = _z(s_final, eps)
    stab_term   = stable.astype(float)  # already 0/1

    score = w_growth * growth_term + w_speed * speed_term + w_stab * stab_term
    df["score"] = score

    # sort high→low
    df_rank = df.sort_values("score", ascending=False).reset_index(drop=True)
    best = df_rank.iloc[0:1].copy()

    # --- Save ranked CSV ---
    csv_path = run_dir / f"{tag}__best_universe_ranked.csv"
    df_rank.to_csv(csv_path, index=False)

    # --- Summary JSON ---
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "mode": tag,
        "N": int(len(df_rank)),
        "winner": best.to_dict(orient="records")[0],
        "weights": {"growth": w_growth, "speed": w_speed, "stability": w_stab},
        "files": {"csv": str(csv_path)}
    }
    json_path = run_dir / f"{tag}__best_universe_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Plots: Top-K detail PNGs + Overview ---
    figs = []
    plot_cfg = active_cfg.get("BEST_UNIVERSE", {}).get("plot", {})
    dpi  = int(plot_cfg.get("dpi", 180))
    annot_color = plot_cfg.get("annot_color", "red")

    top_k = int(active_cfg.get("BEST_UNIVERSE", {}).get("top_k_png", 5))
    epochs = int(active_cfg["ENERGY"].get("lockin_epochs", 500))

    if top_k > 0:
        k = min(top_k, len(df_rank))
        for idx in range(k):
            row = df_rank.iloc[idx]
            uid = int(row[col_id])

            # pick X or fallback
            X_val = None
            if "X" in row and np.isfinite(row["X"]):
                X_val = float(row["X"])
            elif "E0" in row and np.isfinite(row["E0"]):
                X_val = float(row["E0"])
            else:
                X_val = 1.0

            # reconstruct a representative lock-in curve for this universe
            base_seed = seeds["master_seed"]
            # deterministic per-universe seed equal to stored split
            uni_seed = int(seeds["universe_seeds"][uid % len(seeds["universe_seeds"])])
            L = _simulate_law_trajectory(X_val, epochs, uni_seed, active_cfg)

            plt.figure()
            plt.plot(L, label=f"Universe {uid}")
            plt.xlabel("epoch"); plt.ylabel("law amplitude")
            plt.title(f"Top {idx+1} — score={row['score']:.3f}")
            # annotate S_final / lockin if available
            if np.isfinite(row.get(col_sfin, np.nan)):
                plt.axhline(y=float(row[col_sfin]), color="gray", linestyle="--", linewidth=1)
            if np.isfinite(row.get(col_lk, np.nan)):
                plt.axvline(x=float(row[col_lk]), color=annot_color, linestyle="--", linewidth=1,
                            label=f"lock-in≈{int(row[col_lk])}")
                plt.legend()
            fig_p = fig_dir / f"{tag}__best_u{uid:05d}.png"
            plt.tight_layout()
            plt.savefig(fig_p, dpi=dpi)
            plt.close()
            figs.append(str(fig_p))

        # Overview PNG: barplot of Top-K scores
        plt.figure()
        plt.bar(range(k), df_rank["score"].head(k))
        plt.xticks(range(k), [int(v) for v in df_rank[col_id].head(k)], rotation=0)
        plt.xlabel("universe_id"); plt.ylabel("score")
        plt.title(f"Top-{k} universes by score")
        fig_over = fig_dir / f"{tag}__best_top{k}_overview.png"
        plt.tight_layout()
        plt.savefig(fig_over, dpi=dpi)
        plt.close()
        figs.append(str(fig_over))

    # --- mirror copies ---
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

    print(f"[BEST] mode={tag} → CSV/JSON/PNGs saved under:\n  {run_dir}")

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "plots": figs,
        "top_table": df_rank,
        "best_row": best.iloc[0].to_dict(),
    }


# allow standalone run (expects you to pass actual tables in real pipeline)
if __name__ == "__main__":
    # In a real pipeline the controller passes collapse/expansion frames.
    # Here we only guarantee module importability.
    print("[BEST] Module ready. Call run_best_universe(...) from the controller.")
