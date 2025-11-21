"""
Universe pipeline diagnostics.

Provides pre-flight (pre-run) and post-run health checks that are
tailored to the modular universe simulation pipeline.
"""
from __future__ import annotations

import glob
import json
import multiprocessing
import os
import time
from typing import Dict, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover - diagnostics only needs pandas when available
    pd = None  # type: ignore


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


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
    """
    Append a single check result and automatically update the aggregate status.
    """
    if severity not in {"info", "warning", "error"}:
        severity = "info"

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


def _resolve_target_output_dir(config: Dict) -> str:
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    modular_results_dir = os.path.join(
        desktop_path, "TQE_Universe_Simulation_Modular_Results"
    )
    if config.get("DRIVE_BASE_DIR"):
        return config["DRIVE_BASE_DIR"]
    if config.get("MULTI_I_ANALYSIS_MODE", False):
        return config.get("MULTI_I_SAVE_DIR", modular_results_dir)
    return modular_results_dir


# ---------------------------------------------------------------------------
# Pre-flight diagnostics
# ---------------------------------------------------------------------------


def run_preflight_diagnostics(config: Dict) -> Dict:
    """
    Execute pre-run diagnostics to catch environmental issues early.
    """
    report = _new_report("preflight")

    target_dir = _resolve_target_output_dir(config)
    test_file = os.path.join(target_dir, ".diagnostic_write_test")
    try:
        os.makedirs(target_dir, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_file)
        _add_check(
            report,
            "Desktop output write access",
            True,
            details=f"Writable target: {target_dir}",
            path=target_dir,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime env
        _add_check(
            report,
            "Desktop output write access",
            False,
            severity="error",
            details=f"Cannot create files in {target_dir}: {exc}",
            path=target_dir,
        )

    cpu_count = multiprocessing.cpu_count()
    requested_workers = config.get("MAX_WORKERS", cpu_count)
    use_mp = config.get("USE_MULTIPROCESSING", True)
    if use_mp and requested_workers:
        severity = "warning" if requested_workers > cpu_count else "info"
        _add_check(
            report,
            "Multiprocessing capacity",
            requested_workers <= cpu_count,
            severity=severity,
            details=f"Configured workers: {requested_workers}, CPU cores: {cpu_count}",
        )
    else:
        _add_check(
            report,
            "Multiprocessing capacity",
            True,
            details="Sequential mode",
        )

    num_universes = config.get("NUM_UNIVERSES", 0)
    run_mode = config.get("RUN_MODE", "single_ei")
    if run_mode == "batch_all" and num_universes < 400:
        _add_check(
            report,
            "Universe sample size",
            False,
            severity="warning",
            details=(
                f"NUM_UNIVERSES={num_universes} is low for batch_all mode "
                "(recommended ≥ 400)."
            ),
        )
    else:
        _add_check(
            report,
            "Universe sample size",
            True,
            details=f"{num_universes} universes configured for {run_mode}.",
        )

    planck_required = config.get("RUN_PLANCK_VALIDATION", True)
    planck_path = config.get("PLANCK_DATA_PATH")
    if planck_required and planck_path:
        if not os.path.isabs(planck_path):
            candidate = os.path.join(REPO_ROOT, planck_path)
        else:
            candidate = planck_path
        exists = os.path.exists(candidate)
        severity = "error" if not exists and not config.get(
            "PLANCK_AUTO_DOWNLOAD", True
        ) else "warning"
        _add_check(
            report,
            "Planck dataset availability",
            exists or config.get("PLANCK_AUTO_DOWNLOAD", True),
            severity=severity,
            details=(
                f"Resolved path: {candidate}"
                if planck_path
                else "PLANCK_DATA_PATH not set"
            ),
            path=candidate if planck_path else None,
        )
    elif planck_required:
        _add_check(
            report,
            "Planck dataset availability",
            config.get("PLANCK_AUTO_DOWNLOAD", True)
            or config.get("PLANCK_GENERATE_IF_MISSING", True),
            severity="warning",
            details="PLANCK_DATA_PATH not provided; relying on auto-download/generation.",
        )
    else:
        _add_check(
            report,
            "Planck dataset availability",
            True,
            details="Planck validation disabled in config.",
        )

    output_flags = [
        ("SAVE_JSON", "JSON artifacts"),
        ("SAVE_FIGS", "Matplotlib figures"),
        ("SAVE_AGGREGATES", "Aggregate CSV files"),
    ]
    for flag, label in output_flags:
        enabled = config.get(flag, True)
        _add_check(
            report,
            label,
            enabled,
            severity="warning" if not enabled else "info",
            details=f"{flag}={'ON' if enabled else 'OFF'}",
        )

    return report


def report_preflight_results(report: Dict) -> None:
    """
    Pretty-print the preflight diagnostics.
    """
    print("\n" + "=" * 70)
    print("TQE UNIVERSE PIPELINE - PREFLIGHT DIAGNOSTICS")
    print("=" * 70)
    print(f"Status: {report['status'].upper()}  @ {report['timestamp']}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
        name = check["name"]
        details = check["details"]
        print(f"{icon} {name}: {details}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Post-run diagnostics
# ---------------------------------------------------------------------------


def run_postrun_diagnostics(ctx, summary: Dict) -> Dict:
    """
    Inspect the finished run for missing artifacts or suspicious results.
    """
    report = _new_report("postrun")
    save_dir = ctx.paths.get("SAVE_DIR")
    aggregate_dir = ctx.paths.get("AGGREGATE_DIR")
    png_dir = ctx.paths.get("PNG_VISUALIZATIONS_DIR")

    _add_check(
        report,
        "Run output directory",
        os.path.isdir(save_dir),
        severity="error",
        details=f"Save dir: {save_dir}",
        path=save_dir,
    )

    tqe_runs_path = os.path.join(aggregate_dir, "tqe_runs.csv")
    has_tqe_runs = os.path.exists(tqe_runs_path) and os.path.getsize(tqe_runs_path) > 0
    _add_check(
        report,
        "Monte Carlo output (tqe_runs.csv)",
        has_tqe_runs,
        severity="error" if not has_tqe_runs else "info",
        details="Primary universe dataset.",
        path=tqe_runs_path,
    )

    summary_path = os.path.join(aggregate_dir, "summary_full.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary_json = json.load(handle)
            missing_keys = [
                key
                for key in ("stability_summary", "ei_summary", "pipeline_status")
                if key not in summary_json
            ]
            _add_check(
                report,
                "Summary JSON completeness",
                not missing_keys,
                severity="warning" if missing_keys else "info",
                details=(
                    "Missing keys: " + ", ".join(missing_keys)
                    if missing_keys
                    else "summary_full.json OK"
                ),
                path=summary_path,
            )
        except Exception as exc:  # pragma: no cover - depends on file state
            _add_check(
                report,
                "Summary JSON readability",
                False,
                severity="error",
                details=f"Failed to parse summary_full.json: {exc}",
                path=summary_path,
            )
    else:
        _add_check(
            report,
            "Summary JSON presence",
            False,
            severity="error",
            details="summary_full.json not found.",
            path=summary_path,
        )

    png_files = glob.glob(os.path.join(png_dir, "*.png"))
    _add_check(
        report,
        "Visualization outputs",
        len(png_files) >= 5,
        severity="warning" if len(png_files) < 5 else "info",
        details=f"{len(png_files)} PNG files located.",
        path=png_dir,
    )

    planck_csv = os.path.join(aggregate_dir, "planck_validation.csv")
    if ctx.config.get("RUN_PLANCK_VALIDATION", True):
        _add_check(
            report,
            "Planck validation CSV",
            os.path.exists(planck_csv),
            severity="warning",
            details="Required when RUN_PLANCK_VALIDATION=True.",
            path=planck_csv,
        )
    else:
        _add_check(
            report,
            "Planck validation CSV",
            True,
            details="Planck validation disabled.",
        )

    if ctx.config.get("COMPUTE_ALL_I_DEFINITIONS", False):
        i_def_csv = os.path.join(aggregate_dir, "I_Definitions_Comparison.csv")
        _add_check(
            report,
            "I-definition comparison CSV",
            os.path.exists(i_def_csv),
            severity="warning",
            details="Expected when COMPUTE_ALL_I_DEFINITIONS=True.",
            path=i_def_csv,
        )

    goldilocks_dir = ctx.paths.get("GOLDILOCKS_DIR")
    has_goldilocks = bool(glob.glob(os.path.join(goldilocks_dir, "*.json")))
    _add_check(
        report,
        "Goldilocks artifacts",
        has_goldilocks,
        severity="warning" if not has_goldilocks else "info",
        details="Calibration json files present." if has_goldilocks else "No Goldilocks JSON files detected.",
        path=goldilocks_dir,
    )

    # Phase 16: CMB Anomaly Detection CSV files
    if ctx.variant != "energy_only":
        i_def = ctx.config.get("I_DEFINITION_MODE", "default")
        variant_suffix = i_def if ctx.variant != "energy_only" else "eonly"
        coldspot_csv = os.path.join(aggregate_dir, f"cmb_coldspots_summary_{variant_suffix}.csv")
        aoe_csv = os.path.join(aggregate_dir, f"cmb_aoe_summary_{variant_suffix}.csv")
        _add_check(
            report,
            "Phase 16: CMB anomaly CSV files",
            os.path.exists(coldspot_csv) and os.path.exists(aoe_csv),
            severity="warning",
            details=f"Cold spots: {os.path.exists(coldspot_csv)}, AOE: {os.path.exists(aoe_csv)}",
        )

    # Phase 22: Anomaly PNG files
    if ctx.variant != "energy_only":
        anomaly_pngs = [
            "aggregate_coldspot_density_map.png",
            "aggregate_cmb_anomaly_overlay.png",
            "aoe_alignment_histogram.png",
            "coldspot_position_heatmap.png",
            "coldspot_depth_histogram.png",
            "aggregate_aoe_density_map.png"
        ]
        found_anomaly = 0
        missing_pngs = []
        for png in anomaly_pngs:
            base_path = os.path.join(png_dir, png)
            variant_path = ctx.with_variant(base_path) if hasattr(ctx, 'with_variant') else base_path
            if os.path.exists(base_path) or os.path.exists(variant_path):
                found_anomaly += 1
            else:
                missing_pngs.append(png)
        _add_check(
            report,
            "Phase 22: Anomaly visualization PNGs",
            found_anomaly >= len(anomaly_pngs) * 0.8,  # At least 80% should exist
            severity="warning",
            details=f"{found_anomaly}/{len(anomaly_pngs)} anomaly PNG files found. Missing: {', '.join(missing_pngs) if missing_pngs else 'none'}",
        )

    # Phase 23: Enhanced Physics files
    if ctx.config.get("USE_ENHANCED_PHYSICS", True) and ctx.variant != "energy_only":
        enhanced_files = [
            "enhanced_physics_analysis.json",
            "enhanced_physics_comprehensive_summary.csv",
            "enhanced_physics_entanglement_network.csv",
            "enhanced_physics_friedmann_evolution.csv",
            "enhanced_physics_physical_anomalies.csv",
            "enhanced_physics_quantum_fields.csv"
        ]
        # Check with variant tags and glob patterns
        found_enhanced = 0
        missing_enhanced = []
        for f in enhanced_files:
            base_name = f.split('.')[0]
            ext = f.split('.')[1]
            # Check exact match first
            exact_path = os.path.join(aggregate_dir, f)
            variant_path = ctx.with_variant(exact_path) if hasattr(ctx, 'with_variant') else exact_path
            # Also check glob pattern for variant-tagged files
            pattern = os.path.join(aggregate_dir, f"{base_name}*{ext}")
            if os.path.exists(exact_path) or os.path.exists(variant_path) or glob.glob(pattern):
                found_enhanced += 1
            else:
                missing_enhanced.append(f)
        _add_check(
            report,
            "Phase 23: Enhanced Physics files",
            found_enhanced >= len(enhanced_files) * 0.8,
            severity="warning",
            details=f"{found_enhanced}/{len(enhanced_files)} enhanced physics files found. Missing: {', '.join(missing_enhanced) if missing_enhanced else 'none'}",
        )

    # Phase 21: Advanced Statistics files
    if ctx.variant != "energy_only":
        phase21_files = [
            "comprehensive_statistics.csv",
            "parameter_sensitivity_analysis.csv",
            "universe_classification.csv",
            "performance_metrics.csv"
        ]
        # Check both aggregate_dir and with variant tags
        found_phase21 = 0
        missing_files = []
        for f in phase21_files:
            # Check in aggregate_dir
            base_path = os.path.join(aggregate_dir, f)
            variant_path = ctx.with_variant(base_path) if hasattr(ctx, 'with_variant') else base_path
            if os.path.exists(base_path) or os.path.exists(variant_path):
                found_phase21 += 1
            else:
                missing_files.append(f)
        _add_check(
            report,
            "Phase 21: Advanced Statistics files",
            found_phase21 >= len(phase21_files) * 0.75,
            severity="warning",
            details=f"{found_phase21}/{len(phase21_files)} statistics files found. Missing: {', '.join(missing_files) if missing_files else 'none'}",
        )

    # Phase 24: Comprehensive Data
    if ctx.config.get("USE_ENHANCED_PHYSICS", True):
        comprehensive_pattern = os.path.join(aggregate_dir, "comprehensive_universe_physics_data*.csv")
        has_comprehensive = bool(glob.glob(comprehensive_pattern))
        _add_check(
            report,
            "Phase 24: Comprehensive universe physics data",
            has_comprehensive,
            severity="warning",
            details="Comprehensive data CSV present" if has_comprehensive else "Missing comprehensive data CSV",
        )

    # Phase 25: Advanced Anomaly Detection
    if ctx.config.get("ENABLE_QUANTUM_ANOMALY_DETECTION", True):
        anomaly_csv = os.path.join(aggregate_dir, "advanced_anomaly_detection_results.csv")
        _add_check(
            report,
            "Phase 25: Advanced anomaly detection CSV",
            os.path.exists(anomaly_csv),
            severity="warning",
            details="Advanced anomaly results present" if os.path.exists(anomaly_csv) else "Missing advanced anomaly CSV",
        )

    # Phase 15: Planck best fit JSON
    if ctx.config.get("RUN_PLANCK_VALIDATION", True):
        planck_json = os.path.join(aggregate_dir, "planck_best_fit_summary.json")
        _add_check(
            report,
            "Phase 15: Planck best fit summary JSON",
            os.path.exists(planck_json),
            severity="warning",
            details="Planck best fit JSON present" if os.path.exists(planck_json) else "Missing planck_best_fit_summary.json",
        )

    # Phase 18: Goldilocks optimization JSON (specific to I-definition)
    if ctx.variant != "energy_only":
        i_def = ctx.config.get("I_DEFINITION_MODE", "default")
        goldilocks_json = os.path.join(goldilocks_dir, f"goldilocks_optimization_{i_def}.json")
        _add_check(
            report,
            "Phase 18: Goldilocks optimization JSON",
            os.path.exists(goldilocks_json),
            severity="warning",
            details=f"Goldilocks JSON for {i_def} present" if os.path.exists(goldilocks_json) else f"Missing goldilocks_optimization_{i_def}.json",
        )

    stability = summary.get("stability_summary", {})
    total_universes = int(stability.get("total_universes", 0))
    _add_check(
        report,
        "Stability metrics populated",
        total_universes > 0,
        severity="error" if total_universes == 0 else "info",
        details=f"Total universes recorded: {total_universes}",
    )

    if pd is not None and has_tqe_runs:
        try:
            df = pd.read_csv(tqe_runs_path)
            nan_count = int(df.isna().sum().sum())
            severity = "warning" if nan_count > 0 else "info"
            _add_check(
                report,
                "NaN scan (tqe_runs.csv)",
                nan_count == 0,
                severity=severity,
                details=f"NaN entries found: {nan_count}",
                path=tqe_runs_path,
            )
        except Exception as exc:  # pragma: no cover - depends on pandas availability
            _add_check(
                report,
                "NaN scan (tqe_runs.csv)",
                False,
                severity="warning",
                details=f"Unable to read CSV for NaN scan: {exc}",
                path=tqe_runs_path,
            )

    return report


def report_postrun_results(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("TQE UNIVERSE PIPELINE - POST-RUN HEALTH REPORT")
    print("=" * 70)
    print(f"Status: {report['status'].upper()}  @ {report['timestamp']}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
        name = check["name"]
        details = check["details"]
        print(f"{icon} {name}: {details}")
    print("=" * 70 + "\n")

