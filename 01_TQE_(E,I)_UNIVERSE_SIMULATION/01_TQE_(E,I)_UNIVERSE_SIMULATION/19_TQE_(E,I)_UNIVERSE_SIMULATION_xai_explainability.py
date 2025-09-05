# 19_TQE_(E,I)_UNIVERSE_SIMULATION_xai_explainability.py
# ===================================================================================
# XAI (SHAP/LIME) explainability module for the TQE pipeline
# -----------------------------------------------------------------------------------
# - Collects features from previous stages (E0, I_*, X, stability/lock-in, S_final).
# - Builds classification label (locked_in) and regression target (S_final).
# - Trains RandomForest models (params from ACTIVE["XAI"]).
# - Computes SHAP values and plots (if shap installed).
# - Optionally generates a few LIME local explanations (if lime installed).
# - Saves CSV (feature importances), JSON (summary), PNG (SHAP plots).
#
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional, List, Tuple
import os, json, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ACTIVE
from io_paths import resolve_output_paths, ensure_colab_drive_mounted

# Optional deps
try:
    import shap
except Exception:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:
    LimeTabularExplainer = None

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error


# ---------------------------
# Small helpers
# ---------------------------
def _save_mirrors(files: List[str], mirrors: List[str], fig_sub: str):
    """Copy files to mirror targets; figures go to <mirror>/<fig_sub>/."""
    from shutil import copy2
    for m in mirrors:
        try:
            m_path = pathlib.Path(m)
            for fp in files:
                p = pathlib.Path(fp)
                if p.suffix.lower() in {".png", ".jpg"}:
                    tdir = m_path / fig_sub
                    tdir.mkdir(parents=True, exist_ok=True)
                    copy2(p, tdir / p.name)
                else:
                    copy2(p, m_path / p.name)
        except Exception as e:
            print(f"[WARN] mirror copy failed for {m}: {e}")


def _load_join_tables(run_dir: pathlib.Path) -> pd.DataFrame:
    """
    Load tables produced earlier (if present) and inner-join by 'universe_id'.
    We try a reasonable set; missing ones are skipped gracefully.
    """
    # Candidate files (optional – we read what exists)
    candidates = [
        "EI__expansion.csv", "E__expansion.csv",
        "EI__collapse_lockin.csv", "E__collapse_lockin.csv",
        "EI__fluctuation_samples.csv", "E__fluctuation_samples.csv",
        "superposition.csv", "superposition_E_only.csv",
        "energy_sampling.csv",  # if you dump such file
    ]
    frames = []
    for name in candidates:
        p = run_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p)
                frames.append(df)
            except Exception as e:
                print(f"[XAI] skip {name}: {e}")

    if not frames:
        raise RuntimeError("No input CSVs found for XAI. Ensure earlier stages saved CSVs.")

    # Progressive inner join on 'universe_id'
    out = frames[0]
    for df in frames[1:]:
        if "universe_id" in df.columns:
            out = out.merge(df, on="universe_id", how="inner")
    return out


def _pick_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select a robust feature set present in most runs.
    Falls back gracefully if some columns are missing.
    """
    candidates = [
        "E0", "logE0",                 # energy
        "I_kl", "I_shannon", "I_fused",# information
        "X",                           # coupled quantity
        "stable", "in_goldilocks_E",   # flags
        "stable_at", "lockin_at",      # dynamics timing
    ]
    feats = [c for c in candidates if c in df.columns]
    if not feats:
        # last resort: use any numeric columns except obvious labels/targets
        blacklist = {"universe_id", "locked_in", "S_final"}
        feats = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feats].copy()
    # sanitize NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, feats


# ---------------------------
# Public API
# ---------------------------
def run_xai(active_cfg: Dict = ACTIVE,
            joined_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Train simple RF models and compute SHAP/LIME explanations.
    Inputs:
      - joined_df: optional; if None, tries to load CSVs from run_dir and join.
    Returns:
      dict with csv/json/plots and the final DataFrame used.
    """
    if not active_cfg.get("PIPELINE", {}).get("run_xai", True):
        print("[XAI] run_xai=False → skipping.")
        return {}

    # Resolve IO
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    fig_dir = pathlib.Path(paths["fig_dir"])
    mirrors = paths["mirrors"]
    fig_sub = active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")

    xcfg = active_cfg.get("XAI", {})
    test_size = float(xcfg.get("test_size", 0.25))
    random_state = int(xcfg.get("test_random_state", 42))
    rf_n = int(xcfg.get("rf_n_estimators", 400))
    n_jobs = int(xcfg.get("sklearn_n_jobs", -1))
    lime_features = int(xcfg.get("lime_num_features", 5))

    # Load/join data
    if joined_df is None:
        df = _load_join_tables(run_dir)
    else:
        df = joined_df.copy()

    # Derive labels/targets
    if "locked_in" not in df.columns:
        if "lockin_at" in df.columns:
            df["locked_in"] = (df["lockin_at"] >= 0).astype(int)
        else:
            # if missing, approximate from 'stable'
            df["locked_in"] = df.get("stable", pd.Series(np.zeros(len(df)))).astype(int)

    if "S_final" not in df.columns:
        # If expansion not run, try 'final_L' (collapse result), else fallback X
        if "final_L" in df.columns:
            df["S_final"] = df["final_L"].astype(float)
        elif "X" in df.columns:
            df["S_final"] = df["X"].astype(float)
        else:
            raise RuntimeError("No S_final/final_L/X available to regress.")

    # Pick features
    X, feat_names = _pick_features(df)
    y_cls = df["locked_in"].astype(int).values
    y_reg = df["S_final"].astype(float).values

    # Split
    Xtr, Xte, ytr_c, yte_c = train_test_split(X.values, y_cls, test_size=test_size,
                                              random_state=random_state, stratify=y_cls if y_cls.sum() and y_cls.sum() < len(y_cls) else None)
    _,  Xte_r, _, yte_r = train_test_split(X.values, y_reg, test_size=test_size,
                                           random_state=random_state)

    # Models
    clf = RandomForestClassifier(n_estimators=rf_n, n_jobs=n_jobs, class_weight=xcfg.get("rf_class_weight", None), random_state=random_state)
    reg = RandomForestRegressor(n_estimators=rf_n, n_jobs=n_jobs, random_state=random_state)

    clf.fit(Xtr, ytr_c)
    reg.fit(X.values, y_reg)  # use all for regression to stabilize SHAP

    # Metrics
    yproba = clf.predict_proba(Xte)[:, 1] if Xte.shape[0] else np.array([])
    ypred_c = (yproba >= 0.5).astype(int) if yproba.size else np.array([])
    auc = float(roc_auc_score(yte_c, yproba)) if yproba.size else float("nan")
    acc = float(accuracy_score(yte_c, ypred_c)) if ypred_c.size else float("nan")

    yhat_r = reg.predict(Xte_r) if Xte_r.shape[0] else np.array([])
    r2 = float(r2_score(yte_r, yhat_r)) if yhat_r.size else float("nan")
    mae = float(mean_absolute_error(yte_r, yhat_r)) if yhat_r.size else float("nan")

    # Feature importances (RF)
    imp_cls = clf.feature_importances_
    imp_reg = reg.feature_importances_
    import_df = pd.DataFrame({
        "feature": feat_names,
        "rf_importance_cls": imp_cls,
        "rf_importance_reg": imp_reg
    }).sort_values("rf_importance_cls", ascending=False)

    # Save CSV
    csv_import = run_dir / "XAI__rf_importances.csv"
    import_df.to_csv(csv_import, index=False)

    # SHAP
    figs = []
    shap_done = False
    if shap is not None:
        try:
            # Use a small background to keep it light
            bg_size = int(xcfg.get("shap_background_size", 200))
            bg_idx = np.random.choice(np.arange(X.shape[0]), size=min(bg_size, X.shape[0]), replace=False)
            explainer_c = shap.TreeExplainer(clf)
            shap_vals_c = explainer_c.shap_values(X.values)[1] if isinstance(explainer_c.model_output, str) or (isinstance(explainer_c.expected_value, list) and len(explainer_c.expected_value)==2) else explainer_c.shap_values(X.values)

            # Summary plot (classification)
            plt.figure()
            shap.summary_plot(shap_vals_c, X, feature_names=feat_names, show=False)
            f1 = fig_dir / "XAI__shap_summary_cls.png"
            plt.tight_layout()
            plt.savefig(f1, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
            plt.close()
            figs.append(str(f1))

            # Bar plot
            plt.figure()
            shap.summary_plot(shap_vals_c, X, feature_names=feat_names, plot_type="bar", show=False)
            f2 = fig_dir / "XAI__shap_bar_cls.png"
            plt.tight_layout()
            plt.savefig(f2, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
            plt.close()
            figs.append(str(f2))

            # Regression SHAP (optional – often similar ordering)
            explainer_r = shap.TreeExplainer(reg)
            shap_vals_r = explainer_r.shap_values(X.values)

            plt.figure()
            shap.summary_plot(shap_vals_r, X, feature_names=feat_names, show=False)
            f3 = fig_dir / "XAI__shap_summary_reg.png"
            plt.tight_layout()
            plt.savefig(f3, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180), bbox_inches="tight")
            plt.close()
            figs.append(str(f3))

            shap_done = True
        except Exception as e:
            print("[XAI] SHAP failed:", e)

    # LIME (few samples)
    lime_files = []
    if LimeTabularExplainer is not None:
        try:
            expl = LimeTabularExplainer(X.values, feature_names=feat_names, class_names=["no_lock", "lock"], discretize_continuous=True)
            n_examples = min(3, Xte.shape[0]) if Xte.shape[0] else 0
            for i in range(n_examples):
                exp = expl.explain_instance(Xte[i], clf.predict_proba, num_features=lime_features)
                html_path = fig_dir / f"XAI__lime_example_{i}.html"
                exp.save_to_file(str(html_path))
                lime_files.append(str(html_path))
        except Exception as e:
            print("[XAI] LIME failed:", e)

    # JSON summary
    summary = {
        "env": paths["env"],
        "run_id": paths["run_id"],
        "n_samples": int(X.shape[0]),
        "n_features": len(feat_names),
        "features": feat_names,
        "classifier_metrics": {"auc": auc, "acc": acc},
        "regressor_metrics": {"r2": r2, "mae": mae},
        "shap_done": bool(shap_done),
        "lime_examples": [os.path.basename(p) for p in lime_files],
        "files": {
            "rf_importances_csv": str(csv_import),
            "plots": [os.path.basename(f) for f in figs] + [os.path.basename(p) for p in lime_files],
        }
    }
    json_path = run_dir / "XAI__summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Mirror copies
    _save_mirrors([str(csv_import), str(json_path), *figs, *lime_files], mirrors, fig_sub)

    print(f"[XAI] saved under:\n  {run_dir}")
    return {"csv": str(csv_import), "json": str(json_path), "plots": figs + lime_files, "table": df}


if __name__ == "__main__":
    run_xai(ACTIVE)
