"""
Diagnostics helpers for the modular Heisenberg pipeline.
"""
from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, Optional

try:
    import pandas as _pd  # type: ignore
except ImportError:  # pragma: no cover
    _pd = None

MODULE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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


def _desktop_output_root() -> str:
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    return os.path.join(desktop_path, "TQE_Heisenberg_Modular_Results")


def run_preflight_diagnostics(config: Dict) -> Dict:
    report = _new_report("preflight")

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
    except Exception as exc:  # pragma: no cover
        _add_check(
            report,
            "Desktop output write access",
            False,
            severity="error",
            details=f"Cannot write to {output_root}: {exc}",
            path=output_root,
        )

    n_ensemble = config.get("N_ENSEMBLE", 0)
    severity = "warning" if n_ensemble < 64 else "info"
    _add_check(
        report,
        "Ensemble size",
        n_ensemble >= 32,
        severity=severity,
        details=f"N_ENSEMBLE={n_ensemble}",
    )

    n_segments = config.get("N_SEGMENTS", 1)
    _add_check(
        report,
        "Segment configuration",
        n_segments > 0,
        severity="error" if n_segments <= 0 else "info",
        details=f"N_SEGMENTS={n_segments}",
    )

    swap_enabled = config.get("USE_TIME_DEPENDENT_SWAP", False)
    hbar = config.get("HBAR", 1.0)
    if swap_enabled and hbar <= 0:
        _add_check(
            report,
            "Qubit swap logic",
            False,
            severity="warning",
            details="USE_TIME_DEPENDENT_SWAP=True but HBAR<=0.",
        )
    else:
        _add_check(
            report,
            "Qubit swap logic",
            True,
            details=f"USE_TIME_DEPENDENT_SWAP={'ON' if swap_enabled else 'OFF'}",
        )

    return report


def report_preflight_results(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("TQE HEISENBERG PIPELINE - PREFLIGHT DIAGNOSTICS")
    print("=" * 70)
    print(f"Status: {report['status'].upper()}  @ {report['timestamp']}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
        print(f"{icon} {check['name']}: {check['details']}")
    print("=" * 70 + "\n")


def run_postrun_diagnostics(run_dir: str) -> Dict:
    report = _new_report("postrun")

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

    expected_files = [
        ("comparative_analysis.json", "Comparative analysis JSON"),
        ("summary.json", "Summary JSON"),
        (os.path.join("data", "no_law_timeseries.csv"), "No-law timeseries CSV"),
        (os.path.join("data", "with_law_timeseries.csv"), "With-law timeseries CSV"),
        (os.path.join("data", "ensemble_final_energies.csv"), "Final energies CSV"),
        (os.path.join("data", "ensemble_final_entropy_coherence.csv"), "Entropy/coherence CSV"),
    ]
    for rel_path, label in expected_files:
        full_path = os.path.join(run_dir, rel_path)
        _add_check(
            report,
            label,
            os.path.exists(full_path),
            severity="warning" if not os.path.exists(full_path) else "info",
            details=f"Path: {full_path}",
            path=full_path,
        )

    fig_dir = os.path.join(run_dir, "figs")
    png_files = glob.glob(os.path.join(fig_dir, "*.png"))
    _add_check(
        report,
        "Visualization PNGs",
        len(png_files) >= 5,
        severity="warning" if len(png_files) < 5 else "info",
        details=f"{len(png_files)} PNG files in figs/.",
        path=fig_dir,
    )

    if _pd is not None:
        csv_path = os.path.join(run_dir, "data", "ensemble_final_energies.csv")
        if os.path.exists(csv_path):
            try:
                df = _pd.read_csv(csv_path)
                _add_check(
                    report,
                    "Final energy CSV row count",
                    len(df) > 0,
                    severity="warning" if len(df) == 0 else "info",
                    details=f"{len(df)} rows in ensemble_final_energies.csv",
                    path=csv_path,
                )
            except Exception as exc:  # pragma: no cover
                _add_check(
                    report,
                    "Final energy CSV readability",
                    False,
                    severity="warning",
                    details=f"Failed to read ensemble_final_energies.csv: {exc}",
                    path=csv_path,
                )

    return report


def report_postrun_results(report: Dict) -> None:
    print("\n" + "=" * 70)
    print("TQE HEISENBERG PIPELINE - POST-RUN HEALTH REPORT")
    print("=" * 70)
    print(f"Status: {report['status'].upper()}  @ {report['timestamp']}")
    for check in report["checks"]:
        icon = "✅" if check["passed"] else ("⚠️" if check["severity"] == "warning" else "❌")
        print(f"{icon} {check['name']}: {check['details']}")
    print("=" * 70 + "\n")

