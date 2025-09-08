# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

# ===================================================================================
# TQE_19_EI_UNIVERSE_SIMULATION_xai.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
#
# SUMMARY:
# This script is an Explainable AI (XAI) analysis module designed to provide a
# deep understanding of the simulation's internal dynamics. It uses machine
# learning to determine which input parameters are the most influential
# drivers of the simulation's outcomes.
#
# The methodology involves several steps:
# 1. It ingests and merges all data from the preceding simulation stages into
#    a single master dataset.
# 2. It iteratively trains a machine learning model (a Random Forest) to
#    predict each key outcome variable (e.g., final size, lock-in time,
#    anomaly flags) based on the initial simulation parameters (features).
# 3. Critically, it employs state-of-the-art XAI frameworks, SHAP and LIME,
#    to analyze these trained models. This reveals the importance of each
#    feature for each outcome, effectively performing a global sensitivity
#    analysis.
#
# This XAI stage moves beyond just observing the simulation's results to
# explaining *why* those results occurred, providing key scientific insights
# into the simulated physics.
#
# ===================================================================================

from typing import Dict, List, Optional, Tuple
import os, json, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TQE_03_EI_UNIVERSE_SIMULATION_imports import ACTIVE, PATHS, RUN_DIR, FIG_DIR

# Repro seeds (optional, if present in your project)
try:
    from seeding import load_or_create_run_seeds
except Exception:
    load_or_create_run_seeds = None

# sklearn core
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    accuracy_score, f1_score, roc_auc_score
)

# SHAP (optional but recommended)
try:
    import shap
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False

# LIME (optional)
try:
    from lime import lime_tabular
    _HAS_LIME = True
except Exception:
    lime_tabular = None
    _HAS_LIME = False


# ---------------------------
# Utilities
# ---------------------------
def _mirror_file(src: pathlib.Path, mirrors: List[str], put_in_figs: bool = False, fig_sub: str = "figs"):
    """Copy freshly written file to mirror directories (fig subdir if requested)."""
    for m in mirrors:
        try:
            m = pathlib.Path(m)
            if put_in_figs:
                dst_dir = m / fig_sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
            else:
                dst = m / src.name
            dst.write_bytes(src.read_bytes())
        except Exception as e:
            print(f"[WARN] Mirror copy failed → {m}: {e}")


def _tag_ei(active_cfg: dict) -> str:
    """Return 'EI' if information channel is ON, otherwise 'E'."""
    return "EI" if active_cfg["PIPELINE"].get("use_information", True) else "E"


def _load_csv_if_exists(path: pathlib.Path) -> Optional[pd.DataFrame]:
    """Load CSV or return None if missing."""
    if path.is_file():
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] Could not load CSV: {path} → {e}")
    return None


def _merge_by_universe_id(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-merge many DataFrames on 'universe_id', coalescing duplicate
    columns when content is equal (best-effort).
    """
    base = None
    for d in dfs:
        if d is None or "universe_id" not in d.columns:
            continue
        if base is None:
            base = d.copy()
        else:
            # Merge with suffixes; later try to coalesce dups
            base = base.merge(d, on="universe_id", how="outer", suffixes=("", "_dup"))
            # Coalesce simple duplicates
            dup_cols = [c for c in base.columns if c.endswith("_dup")]
            for dc in dup_cols:
                orig = dc[:-4]
                if orig in base.columns:
                    try:
                        equal = (base[orig].fillna(np.nan) == base[dc].fillna(np.nan)) | (base[orig].isna() & base[dc].isna())
                        if bool(np.nanmean(equal.astype('float')) > 0.99):
                            base.drop(columns=[dc], inplace=True)
                        else:
                            # Keep both (rename dup -> orig_2)
                            new_name = f"{orig}_2"
                            base.rename(columns={dc: new_name}, inplace=True)
                    except Exception:
                        new_name = f"{orig}_2"
                        base.rename(columns={dc: new_name}, inplace=True)
                else:
                    base.rename(columns={dc: orig}, inplace=True)
    return base if base is not None else pd.DataFrame()


def _available_features(df: pd.DataFrame) -> List[str]:
    """
    Choose feature columns. We prefer a known set if present; fall back to
    'E', 'I', 'X' style columns and avoid obvious targets/IDs.
    """
    preferred = [
        "E0", "E", "logE0", "in_goldilocks_E",
        "I", "I_fused", "I_shannon", "I_kl",
        "X", "S0", "growth_rate_eff", "goldilocks_scale"
    ]
    feats = [c for c in preferred if c in df.columns]

    # Add any other numeric columns that look like candidate inputs
    blocked_prefix = ("S_final", "final_", "lockin", "stable", "cold_spot", "low_l_", "lack_large", "hemi_", "target_", "label_")
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric:
        if c in feats or c == "universe_id":
            continue
        if c.startswith(blocked_prefix):
            continue
        # conservative include: typical input names
        if any(c.lower().startswith(p) for p in ("e", "i_", "x", "s0", "gold", "growth", "energy", "info")):
            feats.append(c)

    # Deduplicate while keeping order
    seen, uniq = set(), []
    for c in feats:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _find_targets(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect target columns and classify as 'reg' or 'clf'.
    - Binary {0,1} or bool → 'clf'
    - Otherwise numeric → 'reg'
    """
    targets: Dict[str, str] = {}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    # Hard block IDs and common features
    blocked = {"universe_id"} | set(_available_features(df))
    # Known typical targets from earlier modules
    hints = [
        "S_final", "final_L", "lockin_at", "stable", "locked_in",
        # anomalies:
        "cold_spot_min", "cold_spot_z", "cold_spot_area",
        "low_l_align_deg", "low_l_pval",
        "lack_large_angle_stat", "lack_large_angle_pval",
        "hemi_asym_frac", "hemi_asym_pval",
    ]
    # Start with hinted names
    for h in hints:
        if h in numeric and h not in blocked:
            if set(pd.Series(df[h].dropna().unique()).astype(int).tolist()) in ({0,1}, {1,0}):
                targets[h] = "clf"
            else:
                targets[h] = "reg"

    # Add any other numeric, non-feature columns
    for c in numeric:
        if c in blocked or c in targets:
            continue
        vals = df[c].dropna().unique()
        if len(vals) == 0:
            continue
        # binary?
        uniq_bin = set(pd.Series(vals).astype(int, errors='ignore'))
        if uniq_bin in ({0,1}, {1,0}):
            targets[c] = "clf"
        else:
            targets[c] = "reg"
    return targets


def _save_fig(fig_path: pathlib.Path, dpi: int):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# ---------------------------
# Main API
# ---------------------------
def run_xai(active_cfg: dict = ACTIVE,
            merged_df: Optional[pd.DataFrame] = None,
            source_run_dir: Optional[str] = None,
            paths: Optional[Dict] = None) -> Dict:
    """
    Run SHAP/LIME XAI across all available outputs merged by universe_id.

    Parameters
    ----------
    active_cfg : dict
        Master Controller active configuration.
    merged_df : pd.DataFrame or None
        If provided, use this dataset directly (must include 'universe_id').
    source_run_dir : str or None
        If provided, read CSVs from this directory.
    paths : dict or None
        If provided (recommended), use the existing run's paths, so outputs land
        alongside previous stage outputs (and mirroring works).

    Returns
    -------
    dict
        Summary info with file paths produced.
    """
    # --- Paths & environment
    ensure_colab_drive_mounted(active_cfg)
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")

    if paths is not None:
        run_dir = pathlib.Path(paths["primary_run_dir"])
        fig_dir = pathlib.Path(paths["fig_dir"])
        mirrors = paths["mirrors"]
    else:
        # Fallback: write to a new run directory (may not contain inputs)
        paths2 = resolve_output_paths(active_cfg)
        run_dir = pathlib.Path(paths2["primary_run_dir"])
        fig_dir = pathlib.Path(paths2["fig_dir"])
        mirrors = paths2["mirrors"]

    tag = _tag_ei(active_cfg)
    dpi = int(active_cfg["RUNTIME"].get("matplotlib_dpi", 180))

    # --- Load & merge data if not provided
    if merged_df is None:
        src_dir = pathlib.Path(source_run_dir) if source_run_dir else run_dir
        candidates = []
        # Expected files (best-effort)
        expected = [
            f"{tag}__expansion.csv",
            f"{tag}__collapse_lockin.csv",
            f"{tag}__montecarlo.csv",
            f"{tag}__best_universe.csv",
            f"{tag}__anomaly_cold_spot.csv",
            f"{tag}__anomaly_low_multipole_alignments.csv",
            f"{tag}__anomaly_LackOfLargeAngleCorrelation.csv",
            f"{tag}__anomaly_HemisphericalAsymmetry.csv",
            f"{tag}__energy_samples.csv",  # if your energy stage writes this
        ]
        for name in expected:
            df = _load_csv_if_exists(src_dir / name)
            if df is not None:
                candidates.append(df)

        # Also sweep any other CSVs in the directory to be forgiving
        for fn in os.listdir(src_dir):
            if fn.endswith(".csv") and fn not in expected:
                df = _load_csv_if_exists(src_dir / fn)
                if df is not None and "universe_id" in df.columns:
                    candidates.append(df)

        merged = _merge_by_universe_id(candidates)
    else:
        merged = merged_df.copy()

    if merged.empty or "universe_id" not in merged.columns:
        raise RuntimeError("[XAI] No merged dataset available (missing CSVs or universe_id).")

    # Drop universes without any feature data
    feats = _available_features(merged)
    if len(feats) == 0:
        raise RuntimeError("[XAI] No usable feature columns found.")
    df_model = merged.dropna(subset=feats, how="all").copy()

    # Persist merged dataset for reproducibility
    dataset_csv = run_dir / f"{tag}__xai_dataset.csv"
    df_model.to_csv(dataset_csv, index=False)
    _mirror_file(dataset_csv, mirrors)

    # Targets
    targets = _find_targets(df_model)
    if len(targets) == 0:
        raise RuntimeError("[XAI] No target columns detected.")

    # Train/test split parameters
    test_size = float(active_cfg["XAI"].get("test_size", 0.25))
    random_state = int(active_cfg["XAI"].get("test_random_state", 42))
    rf_n = int(active_cfg["XAI"].get("rf_n_estimators", 400))
    rf_class_weight = active_cfg["XAI"].get("rf_class_weight", None)
    min_reg = int(active_cfg["XAI"].get("regression_min", 10))
    lime_num_features = int(active_cfg["XAI"].get("lime_num_features", 5))
    n_jobs = int(active_cfg["XAI"].get("sklearn_n_jobs", -1))

    # Optional seeding
    if load_or_create_run_seeds is not None:
        try:
            seeds = load_or_create_run_seeds(active_cfg)
            if seeds and "master_seed" in seeds:
                np.random.seed(int(seeds["master_seed"]))
        except Exception:
            pass

    # Global ranking across targets (SHAP mean |value| averaged)
    global_rank: Dict[str, float] = {f: 0.0 for f in feats}
    global_counts: Dict[str, int] = {f: 0 for f in feats}

    # Store summary
    out_figs: List[str] = []
    per_target_metrics: Dict[str, dict] = {}

    # Ensure figs directory exists
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Loop over targets
    for tgt, kind in targets.items():
        # Build clean modeling table
        sub = df_model.dropna(subset=feats + [tgt]).copy()
        if sub.shape[0] < max(20, len(feats) + 5):
            print(f"[XAI] Skip target {tgt}: not enough rows ({sub.shape[0]}).")
            continue

        X = sub[feats].values
        y = sub[tgt].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        if kind == "clf":
            # Binary RF classifier
            model = RandomForestClassifier(
                n_estimators=rf_n,
                random_state=random_state,
                class_weight=rf_class_weight,
                n_jobs=n_jobs,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=rf_n,
                random_state=random_state,
                n_jobs=n_jobs,
            )

        model.fit(X_train, y_train)

        # Metrics
        metrics = {}
        if kind == "clf":
            y_prob = None
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                pass
            y_pred = model.predict(X_test)
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            if y_prob is not None:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
                except Exception:
                    pass
        else:
            y_pred = model.predict(X_test)
            if len(y_test) >= min_reg:
                metrics["r2"] = float(r2_score(y_test, y_pred))
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

        # SHAP explanations
        shap_summary_csv = None
        if _HAS_SHAP:
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_test)
                # For classifiers, SHAP may return a list → use positive class
                if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                    shap_mat = np.array(shap_vals[1])
                else:
                    shap_mat = np.array(shap_vals)

                # mean |shap| per feature
                mean_abs = np.abs(shap_mat).mean(axis=0)
                shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values(
                    "mean_abs_shap", ascending=False
                )
                shap_summary_csv = run_dir / f"{tag}__xai_shap_importance_{tgt}.csv"
                shap_df.to_csv(shap_summary_csv, index=False)
                _mirror_file(shap_summary_csv, mirrors)

                # Accumulate global rank
                for f, v in zip(feats, mean_abs):
                    global_rank[f] += float(v)
                    global_counts[f] += 1

                # Plot: SHAP summary (bar)
                plt.figure()
                shap.summary_plot(shap_mat, features=X_test, feature_names=feats, show=False, plot_type="bar")
                fig1 = fig_dir / f"{tag}__xai_shap_summary_{tgt}.png"
                _save_fig(fig1, dpi)
                out_figs.append(str(fig1))
                _mirror_file(fig1, mirrors, put_in_figs=True, fig_sub=fig_sub)

                # Plot: SHAP beeswarm
                plt.figure()
                shap.summary_plot(shap_mat, features=X_test, feature_names=feats, show=False)
                fig2 = fig_dir / f"{tag}__xai_shap_beeswarm_{tgt}.png"
                _save_fig(fig2, dpi)
                out_figs.append(str(fig2))
                _mirror_file(fig2, mirrors, put_in_figs=True, fig_sub=fig_sub)

            except Exception as e:
                print(f"[WARN] SHAP failed for target {tgt}: {e}")

        # LIME (optional)
        lime_outputs: List[str] = []
        if _HAS_LIME and active_cfg["XAI"].get("run_lime", True):
            try:
                if kind == "clf":
                    expl = lime_tabular.LimeTabularExplainer(
                        training_data=X_train,
                        feature_names=feats,
                        class_names=["neg", "pos"],
                        mode="classification",
                        discretize_continuous=True
                    )
                else:
                    expl = lime_tabular.LimeTabularExplainer(
                        training_data=X_train,
                        feature_names=feats,
                        mode="regression",
                        discretize_continuous=True
                    )
                # Explain a handful of test samples
                K = min(5, X_test.shape[0])
                for i in range(K):
                    exp = expl.explain_instance(
                        X_test[i], model.predict_proba if kind == "clf" else model.predict,
                        num_features=lime_num_features
                    )
                    fig = exp.as_pyplot_figure()
                    lime_fig = fig_dir / f"{tag}__xai_lime_{tgt}_sample{i}.png"
                    _save_fig(lime_fig, dpi)
                    out_figs.append(str(lime_fig))
                    _mirror_file(lime_fig, mirrors, put_in_figs=True, fig_sub=fig_sub)

                    # Save tabular explanation
                    weights = exp.as_list()
                    lime_csv = run_dir / f"{tag}__xai_lime_{tgt}_sample{i}.csv"
                    pd.DataFrame(weights, columns=["feature", "weight"]).to_csv(lime_csv, index=False)
                    _mirror_file(lime_csv, mirrors)
                    lime_outputs.append(str(lime_csv))
            except Exception as e:
                print(f"[WARN] LIME failed for target {tgt}: {e}")

        # Save per-target metrics JSON
        mt_json = run_dir / f"{tag}__xai_metrics_{tgt}.json"
        payload = {
            "target": tgt,
            "type": kind,
            "metrics": metrics,
            "features_used": feats,
            "shap_importance_csv": str(shap_summary_csv) if shap_summary_csv else None,
            "lime_csvs": lime_outputs,
        }
        with mt_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        _mirror_file(mt_json, mirrors)

        per_target_metrics[tgt] = payload

    # Global feature ranking
    glob_rows = []
    for f in feats:
        if global_counts[f] > 0:
            glob_rows.append((f, global_rank[f] / global_counts[f]))
        else:
            glob_rows.append((f, 0.0))
    glob_df = pd.DataFrame(glob_rows, columns=["feature", "mean_abs_shap_avg"]).sort_values(
        "mean_abs_shap_avg", ascending=False
    )
    global_csv = run_dir / f"{tag}__xai_global_feature_ranking.csv"
    glob_df.to_csv(global_csv, index=False)
    _mirror_file(global_csv, mirrors)

    # Summary JSON
    summary = {
        "env": str(paths["env"] if paths else "unknown"),
        "run_id": str(paths["run_id"] if paths else run_dir.name),
        "tag": tag,
        "features": feats,
        "targets": per_target_metrics,
        "has_shap": bool(_HAS_SHAP),
        "has_lime": bool(_HAS_LIME and ACTIVE["XAI"].get("run_lime", True)),
        "dataset_csv": str(dataset_csv),
        "global_ranking_csv": str(global_csv),
        "figs": out_figs,
    }
    sm_json = run_dir / f"{tag}__xai_summary.json"
    with sm_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _mirror_file(sm_json, mirrors)

    print(f"[XAI] Completed. Outputs under:\n  {run_dir}")
    return {"summary": summary, "dataset_csv": str(dataset_csv), "global_csv": str(global_csv), "figs": out_figs}


# --------------------------------------------------------------
# Wrapper for Master Controller
# --------------------------------------------------------------
def run_xai_stage(active=None, active_cfg=None, **kwargs):
    cfg = active if active is not None else active_cfg
    if cfg is None:
        raise ValueError("Provide 'active' or 'active_cfg'")     
    return run_xai(active_cfg=cfg, **kwargs)  
    
if __name__ == "__main__":
    run_xai_stage(ACTIVE)
