"""
Diagnostics helpers for the modular Dark Energy pipeline.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, Optional

try:
    import pandas as _pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _pd = None

MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(MODULE_ROOT, "..", ".."))


def _new_report(phase: str) -> Dict:
    return {
        "phase": phase,
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": [],
    }


def _add_check(
    report: Dict,
    name: str,
    passed: bool,
    severity: str = "info",
    details: Optional[str] = None,
    path: Optional[str] = None,
) -> None:
    severity = severity if severity in {"info", "warning", "error"} else "info"
    report["checks"].append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "details": details or "",
            "path": path,
        }
    )

    if not passed:
        if severity == "error":
            report["status"] = "error"
        elif severity == "warning" and report["status"] == "ok":
            report["status"] = "warning"


def _resolve_path(target: Optional[str]) -> Optional[str]:
    if not target:
        return None
    if os.path.isabs(target):
        return target
    return os.path.abspath(os.path.join(MODULE_ROOT, target))


def _desktop_output_root() -> str:
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    return os.path.join(desktop_path, "TQE_DarkEnergy_Modular_Results")


def run_preflight_diagnostics(config: Dict, feature_flags: Optional[Dict] = None) -> Dict:
    report = _new_report("preflight")
    feature_flags = feature_flags or {}

    output_root = _desktop_output_root()
    test_file = os.path.join(output_root, ".diagnostic_write_test")
    try:
        os.makedirs(output_root, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_file)
        _add_check(
            report,
            "Desktop output write access",
            True,
            details=f"Writable target: {output_root}",
            path=output_root,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        _add_check(
            report,
            "Desktop output write access",
            False,
            severity="error",
            details=f"Cannot write to {output_root}: {exc}",
            path=output_root,
        )

    def _check_data_source(flag_name: str, path_key: str, label: str) -> None:
        if not config.get(flag_name, False):
            _add_check(
                report,
                label,
                True,
                details="Mock dataset enabled (no real file required).",
            )
            return
        resolved = _resolve_path(config.get(path_key))
        if not resolved:
            _add_check(
                report,
                label,
                False,
                severity="warning",
                details=f"{path_key} not provided while {flag_name}=True.",
            )
            return
        exists = os.path.exists(resolved)
        _add_check(
            report,
            label,
            exists,
            severity="error" if not exists else "info",
            details=f"Resolved path: {resolved}",
            path=resolved if exists else None,
        )

    _check_data_source("USE_REAL_SNE_DATA", "PANTHEON_PLUS_DATA_PATH", "Pantheon+ dataset")
    _check_data_source("USE_REAL_BAO_DATA", "BOSS_BAO_DATA_PATH", "BAO dataset")
    _check_data_source("USE_REAL_CMB_DATA", "PLANCK_CMB_DATA_PATH", "Planck CMB dataset")

    if config.get("USE_REAL_CMB_PLANCK_MAPS", False):
        planck_base = _resolve_path(config.get("CMB_PLANCK_BASE_PATH"))
        if planck_base:
            _add_check(
                report,
                "Planck map directory",
                os.path.isdir(planck_base),
                severity="warning" if not os.path.isdir(planck_base) else "info",
                details=f"Resolved path: {planck_base}",
                path=planck_base,
            )
        else:
            _add_check(
                report,
                "Planck map directory",
                True,
                severity="warning",
                details="CMB_PLANCK_BASE_PATH not set; relying on auto-detection.",
            )
    else:
        _add_check(
            report,
            "Planck map directory",
            True,
            details="USE_REAL_CMB_PLANCK_MAPS=False (synthetic validation).",
        )

    if config.get("RUN_GALAXY_STRUCTURE_ANALYSIS", True):
        available = bool(feature_flags.get("galaxy_analysis_available", True))
        _add_check(
            report,
            "Galaxy structure module availability",
            available,
            severity="warning" if not available else "info",
            details="structure.py import succeeded" if available else "structure module not available.",
        )

    if config.get("RUN_MCMC", False):
        available = bool(feature_flags.get("mcmc_available", True))
        _add_check(
            report,
            "Bayesian inference engine",
            available,
            severity="warning" if not available else "info",
            details="inference.BayesianInferenceEngine available"
            if available
            else "MCMC requested but inference dependencies missing.",
        )

    if config.get("USE_REAL_CMB_PLANCK_MAPS", False):
        available = bool(feature_flags.get("cmb_validation_available", True))
        _add_check(
            report,
            "Planck validation stack",
            available,
            severity="warning" if not available else "info",
            details="Planck validation modules imported."
            if available
            else "Planck validation requested but data_loader modules missing.",
        )

    coupling_mode = config.get("COUPLING_MODE", "EplusI")
    if coupling_mode not in {"Eonly", "EplusI", "dual"}:
        _add_check(
            report,
            "Coupling mode validity",
            False,
            severity="error",
            details=f"Invalid COUPLING_MODE='{coupling_mode}'. Expected Eonly/EplusI/dual.",
        )
    else:
        _add_check(
            report,
            "Coupling mode validity",
            True,
            details=f"COUPLING_MODE={coupling_mode}",
        )

    if config.get("RUN_BETA0_SWEEP", False):
        sweep = config.get("BETA0_SWEEP_FINE") or []
        _add_check(
            report,
            "β₀ sweep configuration",
            bool(sweep),
            severity="warning" if not sweep else "info",
            details=f"{len(sweep)} β₀ values configured." if sweep else "RUN_BETA0_SWEEP=True but sweep list empty.",
        )

    return report


def report_preflight_results(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("TQE DARK ENERGY PIPELINE - PREFLIGHT DIAGNOSTICS")
    print("=" * 70)
    print(f"Status: {report['status'].upper()}  @ {report['timestamp']}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
        print(f"{icon} {check['name']}: {check['details']}")
    print("=" * 70 + "\n")


def run_postrun_diagnostics(
    run_dir: str,
    config: Dict,
    feature_flags: Optional[Dict] = None,
    aggregator_results: Optional[Dict] = None,
) -> Dict:
    report = _new_report("postrun")
    feature_flags = feature_flags or {}

    _add_check(
        report,
        "Run directory exists",
        os.path.isdir(run_dir),
        severity="error" if not os.path.isdir(run_dir) else "info",
        details=f"Run dir: {run_dir}",
        path=run_dir,
    )
    if not os.path.isdir(run_dir):
        return report

    summary_path = os.path.join(run_dir, "pipeline_summary.json")
    pipeline_summary = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                pipeline_summary = json.load(handle)
            _add_check(
                report,
                "Pipeline summary JSON",
                True,
                details="pipeline_summary.json parsed successfully.",
                path=summary_path,
            )
        except Exception as exc:
            _add_check(
                report,
                "Pipeline summary JSON",
                False,
                severity="error",
                details=f"Failed to parse pipeline_summary.json: {exc}",
                path=summary_path,
            )
    else:
        _add_check(
            report,
            "Pipeline summary JSON",
            False,
            severity="error",
            details="pipeline_summary.json missing.",
            path=summary_path,
        )

    model_dirs = [
        d
        for d in os.listdir(run_dir)
        if os.path.isdir(os.path.join(run_dir, d))
        and (d.startswith("Model_") or d.startswith("Null_"))
    ]
    _add_check(
        report,
        "Model directories detected",
        len(model_dirs) > 0,
        severity="error" if not model_dirs else "info",
        details=f"{len(model_dirs)} model folders found.",
    )

    if model_dirs:
        missing_results = []
        for model_dir in model_dirs:
            pattern = os.path.join(run_dir, model_dir, "*_TQE_DarkEnergy_Results_*.json")
            if not glob.glob(pattern):
                missing_results.append(model_dir)
        _add_check(
            report,
            "Per-model result JSON files",
            len(missing_results) == 0,
            severity="warning" if missing_results else "info",
            details="All model result files present."
            if not missing_results
            else f"Missing JSON in: {', '.join(missing_results[:5])}",
        )

    png_matches = glob.glob(os.path.join(run_dir, "Model_*", "PNG_Visualizations", "*.png"))
    png_count = len(png_matches)
    _add_check(
        report,
        "Per-model visualization PNGs",
        png_count > 0,
        severity="warning" if png_count == 0 else "info",
        details=f"Located {png_count} PNG files across model folders.",
    )

    if config.get("RUN_AUTO_AGGREGATOR", True):
        agg_dir = os.path.join(run_dir, "Auto_Aggregator_Summary")
        agg_csv = (
            aggregator_results.get("aggregated_csv")
            if aggregator_results
            else os.path.join(agg_dir, "Aggregated_Results_Summary.csv")
        )
        _add_check(
            report,
            "Aggregator CSV",
            agg_csv is not None and os.path.exists(agg_csv),
            severity="warning" if not agg_csv or not os.path.exists(agg_csv) else "info",
            details=f"Path: {agg_csv}",
            path=agg_csv,
        )

        png_dir = (
            aggregator_results.get("png_dir")
            if aggregator_results and aggregator_results.get("png_dir")
            else os.path.join(agg_dir, "PNG_Visualizations")
        )
        png_files = []
        if png_dir and os.path.isdir(png_dir):
            png_files = glob.glob(os.path.join(png_dir, "*.png"))
        _add_check(
            report,
            "Aggregator visualizations",
            len(png_files) >= 3,
            severity="warning" if len(png_files) < 3 else "info",
            details=f"{len(png_files)} aggregator PNG files.",
            path=png_dir,
        )

    coupling_mode = config.get("COUPLING_MODE", "EplusI")
    run_dual = coupling_mode == "dual" or config.get("RUN_DUAL_COMPARISON", False)
    if run_dual:
        comparison_dir = os.path.join(run_dir, "Eonly_vs_EplusI_Comparison")
        comparison_csv = os.path.join(comparison_dir, "Comparison_Table.csv")
        _add_check(
            report,
            "Dual-mode comparison table",
            os.path.exists(comparison_csv),
            severity="warning" if not os.path.exists(comparison_csv) else "info",
            details=f"Path: {comparison_csv}",
            path=comparison_csv,
        )

    if config.get("USE_NESTED_SAMPLING", False) and config.get("COMPUTE_EVIDENCE", True):
        bf_path = os.path.join(run_dir, "Bayes_Factor_Comparison.json")
        _add_check(
            report,
            "Bayes factor output",
            os.path.exists(bf_path),
            severity="warning" if not os.path.exists(bf_path) else "info",
            details=f"Path: {bf_path}",
            path=bf_path,
        )

    if _pd is not None and model_dirs:
        sample_csv = None
        for model_dir in model_dirs:
            candidate = glob.glob(os.path.join(run_dir, model_dir, "*_TQE_DarkEnergy_Results_*.json"))
            if candidate:
                sample_csv = os.path.join(run_dir, model_dir, "TQE_Run_Log.csv")
                break
        if sample_csv and os.path.exists(sample_csv):
            try:
                df = _pd.read_csv(sample_csv)
                nan_count = int(df.isna().sum().sum())
                _add_check(
                    report,
                    "Sample run log NaN scan",
                    nan_count == 0,
                    severity="warning" if nan_count else "info",
                    details=f"NaN count in {os.path.basename(sample_csv)}: {nan_count}",
                    path=sample_csv,
                )
            except Exception as exc:  # pragma: no cover - depends on pandas availability
                _add_check(
                    report,
                    "Sample run log NaN scan",
                    False,
                    severity="warning",
                    details=f"Failed to read {sample_csv}: {exc}",
                    path=sample_csv,
                )

    return report


def report_postrun_results(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("TQE DARK ENERGY PIPELINE - POST-RUN HEALTH REPORT")
    print("=" * 70)
    print(f"Status: {report['status'].upper()}  @ {report['timestamp']}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
        print(f"{icon} {check['name']}: {check['details']}")
    print("=" * 70 + "\n")

