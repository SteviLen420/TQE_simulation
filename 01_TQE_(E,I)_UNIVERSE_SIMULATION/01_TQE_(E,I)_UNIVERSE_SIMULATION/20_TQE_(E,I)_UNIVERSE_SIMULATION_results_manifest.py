# 20_TQE_(E,I)_UNIVERSE_SIMULATION_results_manifest.py
# ===================================================================================
# Results Manifest (master aggregator) for the TQE simulation
# -----------------------------------------------------------------------------------
# - Collects per-stage mini-manifests (already produced by each module) and embeds
#   them into one hierarchical master JSON: run_manifest.json
# - Builds a per_universe_summary.csv by merging a curated set of columns across
#   stage CSVs (best-effort; missing files are skipped).
# - Saves under the current run directory and mirrors to configured targets.
#
# Author: Stefan Len
# ===================================================================================

from typing import Dict, List, Optional, Tuple
import os, json, glob, pathlib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import ACTIVE                      # unified master config
from io_paths import resolve_output_paths, ensure_colab_drive_mounted

# ---------------------------
# Utilities
# ---------------------------

def _read_json_safe(fp: str) -> Optional[dict]:
    """Best-effort JSON reader; returns None if file is missing or invalid."""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[MANIFEST] Skip JSON ({fp}): {e}")
        return None

def _find_first(patterns: List[str]) -> Optional[str]:
    """Return the first existing file that matches any pattern (glob patterns)."""
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            # prefer the newest if multiple
            hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return hits[0]
    return None

def _load_csv_selected(fp: str, columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Load a CSV and keep only the listed columns that actually exist.
    Returns None if file missing or no requested columns are present.
    """
    if fp is None or not os.path.exists(fp):
        return None
    try:
        df = pd.read_csv(fp)
        keep = [c for c in columns if c in df.columns]
        if not keep:
            return None
        return df[keep].copy()
    except Exception as e:
        print(f"[MANIFEST] Skip CSV ({fp}): {e}")
        return None

def _mirror_file(src: str, mirrors: List[str], put_in_figs: bool = False, fig_subdir: str = "figs"):
    """Copy a file to mirror directories; figures may go under <mirror>/figs/."""
    from shutil import copy2
    for m in mirrors:
        try:
            dst_dir = pathlib.Path(m) / (fig_subdir if put_in_figs else "")
            dst_dir.mkdir(parents=True, exist_ok=True)
            copy2(src, dst_dir / os.path.basename(src))
        except Exception as e:
            print(f"[WARN] Mirror copy failed → {m}: {e}")

# ---------------------------
# Master builder
# ---------------------------

def run_results_manifest(active_cfg: Dict = ACTIVE,
                         run_dir: Optional[str] = None) -> Dict:
    """
    Build the master run manifest and a merged per-universe summary CSV.

    Parameters
    ----------
    active_cfg : dict
        The resolved MASTER_CTRL (ACTIVE).
    run_dir : str or None
        If None, uses the current run directory from resolve_output_paths(...).
        Prefer passing the run_dir from the Master Controller for exactness.

    Returns
    -------
    dict with paths: {"manifest_json", "summary_csv", "figs": [...]}
    """
    # Resolve paths (and optionally mount Drive)
    ensure_colab_drive_mounted(active_cfg)
    paths = resolve_output_paths(active_cfg)

    # If the caller passed a specific run_dir, override the primary one
    if run_dir is not None:
        primary_run_dir = pathlib.Path(run_dir)
        fig_dir = primary_run_dir / active_cfg["OUTPUTS"]["local"].get("fig_subdir", "figs")
        fig_dir.mkdir(parents=True, exist_ok=True)
        mirrors = paths["mirrors"]
    else:
        # Use resolved paths (typical call from Master Controller in the same run)
        primary_run_dir = pathlib.Path(paths["primary_run_dir"])
        fig_dir = pathlib.Path(paths["fig_dir"])
        mirrors = paths["mirrors"]

    print(f"[MANIFEST] Collecting results under:\n  {primary_run_dir}")

    # ---------------------------
    # 1) Locate known mini-manifests (JSON) and important CSVs
    # ---------------------------
    # Patterns are conservative and depend on stage naming we used across modules.
    # Adjust easily by adding patterns below if you rename files.
    j_energy      = _find_first([str(primary_run_dir / "*energy*_summary.json"),
                                 str(primary_run_dir / "*E*_summary.json")])
    j_info        = _find_first([str(primary_run_dir / "*information*_summary.json"),
                                 str(primary_run_dir / "superposition_summary.json")])
    j_fluct       = _find_first([str(primary_run_dir / "*fluctuation*_summary.json")])
    j_collapse    = _find_first([str(primary_run_dir / "*collapse*_summary.json")])
    j_expansion   = _find_first([str(primary_run_dir / "*expansion*_summary.json")])
    j_anom_cold   = _find_first([str(primary_run_dir / "*cold_spot*_summary.json")])
    j_anom_lowell = _find_first([str(primary_run_dir / "*low_multipole*_summary.json")])
    j_anom_lack   = _find_first([str(primary_run_dir / "*LackOfLargeAngleCorrelation*_summary.json")])
    j_anom_hemi   = _find_first([str(primary_run_dir / "*HemisphericalAsymmetry*_summary.json")])
    j_best        = _find_first([str(primary_run_dir / "*best_universe*_summary.json")])
    j_xai         = _find_first([str(primary_run_dir / "*xai*_summary.json")])

    # CSVs to merge per-universe (best effort)
    c_energy     = _find_first([str(primary_run_dir / "*energy*/*.csv"),
                                str(primary_run_dir / "*E*_*.csv"),
                                str(primary_run_dir / "*energy*csv")])
    c_fluct      = _find_first([str(primary_run_dir / "*fluctuation*csv")])
    c_superpos   = _find_first([str(primary_run_dir / "superposition.csv")])
    c_collapse   = _find_first([str(primary_run_dir / "*collapse*_lockin.csv"),
                                str(primary_run_dir / "*collapse*csv")])
    c_expansion  = _find_first([str(primary_run_dir / "*expansion.csv")])
    # anomaly CSVs (each file typically contains per-universe or per-map rows)
    c_cold       = _find_first([str(primary_run_dir / "*cold_spot*csv")])
    c_lowell     = _find_first([str(primary_run_dir / "*low_multipole*csv")])
    c_lack       = _find_first([str(primary_run_dir / "*LackOfLargeAngleCorrelation*csv")])
    c_hemi       = _find_first([str(primary_run_dir / "*HemisphericalAsymmetry*csv")])

    # ---------------------------
    # 2) Load and embed mini-manifests
    # ---------------------------
    stages = {
        "energy":            _read_json_safe(j_energy),
        "information":       _read_json_safe(j_info),
        "fluctuation":       _read_json_safe(j_fluct),
        "collapse_lockin":   _read_json_safe(j_collapse),
        "expansion":         _read_json_safe(j_expansion),
        "best_universe":     _read_json_safe(j_best),
        "anomaly": {
            "cold_spot":                 _read_json_safe(j_anom_cold),
            "low_multipole_alignments":  _read_json_safe(j_anom_lowell),
            "lack_of_large_angle_corr":  _read_json_safe(j_anom_lack),
            "hemispherical_asymmetry":   _read_json_safe(j_anom_hemi),
        },
        "xai":               _read_json_safe(j_xai),
    }

    # ---------------------------
    # 3) Build per_universe_summary.csv (best-effort merge on 'universe_id')
    # ---------------------------
    # Curated column sets we try to extract from each CSV:
    cols_energy    = ["universe_id", "E", "E0", "logE0"]
    cols_fluct     = ["universe_id", "X", "in_goldilocks_E"]
    cols_superpos  = ["universe_id", "I_shannon", "I_kl", "I"]
    cols_collapse  = ["universe_id", "stable_at", "lockin_at", "final_L", "final_rel_delta", "stable", "locked_in"]
    cols_expansion = ["universe_id", "S0", "S_final", "growth_rate_eff"]

    # anomaly columns (names are indicative; modules may choose exact names)
    cols_cold      = ["universe_id", "coldspot_min_z", "coldspot_pval", "coldspot_ra_deg", "coldspot_dec_deg"]
    cols_lowell    = ["universe_id", "l2l3_align_deg", "l2_power", "l3_power"]
    cols_lack      = ["universe_id", "S1_stat", "S1_pval"]
    cols_hemi      = ["universe_id", "hemi_power_ratio", "hemi_pval", "preferred_axis_ra", "preferred_axis_dec"]

    # Load each CSV (subset of columns)
    df_list: List[pd.DataFrame] = []
    for fp, cols in [
        (c_energy,    cols_energy),
        (c_fluct,     cols_fluct),
        (c_superpos,  cols_superpos),
        (c_collapse,  cols_collapse),
        (c_expansion, cols_expansion),
        (c_cold,      cols_cold),
        (c_lowell,    cols_lowell),
        (c_lack,      cols_lack),
        (c_hemi,      cols_hemi),
    ]:
        df = _load_csv_selected(fp, cols)
        if df is not None:
            df_list.append(df)

    if df_list:
        # Progressive outer merge on 'universe_id'
        merged = df_list[0]
        for k in range(1, len(df_list)):
            merged = pd.merge(merged, df_list[k], on="universe_id", how="outer")
        summary_csv_path = primary_run_dir / "per_universe_summary.csv"
        merged.to_csv(summary_csv_path, index=False)
    else:
        summary_csv_path = primary_run_dir / "per_universe_summary.csv"
        # Create an empty placeholder so downstream steps can rely on its presence
        pd.DataFrame({"universe_id": []}).to_csv(summary_csv_path, index=False)
        print("[MANIFEST] No per-universe CSVs found to merge; wrote empty summary.")

    # ---------------------------
    # 4) Optional tiny figure: dataset coverage bars (how many rows per source)
    # ---------------------------
    counts = []
    labels = []
    for label, fp in [
        ("energy", c_energy), ("fluct", c_fluct), ("superpos", c_superpos),
        ("collapse", c_collapse), ("expansion", c_expansion),
        ("cold", c_cold), ("low-ℓ", c_lowell), ("lack", c_lack), ("hemi", c_hemi),
    ]:
        if fp and os.path.exists(fp):
            try:
                n = sum(1 for _ in open(fp, "r", encoding="utf-8")) - 1  # headerless row count
                counts.append(max(0, n))
                labels.append(label)
            except Exception:
                pass

    figs = []
    if counts:
        plt.figure()
        plt.bar(labels, counts)
        plt.ylabel("rows")
        plt.title("Per-source row counts (coverage)")
        plt.tight_layout()
        fig_path = fig_dir / "manifest_coverage.png"
        plt.savefig(fig_path, dpi=ACTIVE["RUNTIME"].get("matplotlib_dpi", 180))
        plt.close()
        figs.append(str(fig_path))

    # ---------------------------
    # 5) Build master JSON (embed stage mini-manifests)
    # ---------------------------
    master = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profile": os.environ.get("TQE_PROFILE", None),
        },
        "paths": {
            "run_dir": str(primary_run_dir),
            "fig_dir": str(fig_dir),
            "summary_csv": str(summary_csv_path),
        },
        "stages": stages,  # embedded mini-manifests
    }

    manifest_json_path = primary_run_dir / "run_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2)

    # ---------------------------
    # 6) Mirror copies
    # ---------------------------
    _mirror_file(str(manifest_json_path), mirrors)
    _mirror_file(str(summary_csv_path), mirrors)
    for fp in figs:
        _mirror_file(fp, mirrors, put_in_figs=True,
                     fig_subdir=ACTIVE["OUTPUTS"]["local"].get("fig_subdir", "figs"))

    print(f"[MANIFEST] Master JSON: {manifest_json_path}")
    print(f"[MANIFEST] Summary CSV: {summary_csv_path}")

    return {
        "manifest_json": str(manifest_json_path),
        "summary_csv": str(summary_csv_path),
        "figs": figs,
    }


# Allow standalone run
if __name__ == "__main__":
    run_results_manifest(ACTIVE)
