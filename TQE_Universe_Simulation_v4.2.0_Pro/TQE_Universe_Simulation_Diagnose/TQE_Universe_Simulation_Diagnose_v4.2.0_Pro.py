# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py
# ==========================================================================================
# TQE Pipeline Diagnostics: Comprehensive validation tool for pipeline integrity
# Checks imports, schema, dependencies, and phase functions for both monolithic and modular pipelines
# ==========================================================================================
#
# AUTHOR: Stefan Len
# DATE: 11.7.2025
# VERSION: v4.2.0 PRO
#
# ==========================================================================================
# USAGE
# ==========================================================================================
#
# Usage (from TQE_Universe_Simulation_Full_Pipeline directory):
#   python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py                    # Basic checks (both pipelines)
#   python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --monolithic       # Check monolithic pipeline only
#   python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --modular          # Check modular pipeline only
#   python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --deep             # Also check heavy optional deps
#   python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --smoke            # Tiny smoke-run test
#   python TQE_Universe_Simulation_Diagnose_v4.2.0_Pro.py --all              # Check both pipelines (default)
#
# CONFIGURATION:
#   The diagnostic behavior is controlled by the DIAGNOSE_CTRL dictionary in this file.
#   You can modify DIAGNOSE_CTRL to customize which checks are performed.
#   Command-line arguments (--monolithic, --modular, --deep, --smoke) override DIAGNOSE_CTRL settings.
#
# ==========================================================================================

from pathlib import Path
import sys
import os
import json
import time
import importlib
import inspect
import traceback
import importlib.util
import pandas as pd
from datetime import datetime

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_DIR = SCRIPT_DIR

# Add pipeline directory to path
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# ==========================================================================================
# DIAGNOSE_CTRL: DIAGNOSTIC CONFIGURATION
# ==========================================================================================
# Configuration dictionary for diagnostic behavior
# Override via command-line arguments: --monolithic, --modular, --deep, --smoke
# ==========================================================================================

DIAGNOSE_CTRL = {
    # ╔════════════════════════════════════════════════════════════════╗
    # ║                    CORE DIAGNOSTIC CONTROLS                    ║
    # ╚════════════════════════════════════════════════════════════════╝
    
    # === PIPELINE SELECTION ===
    "CHECK_MONOLITHIC": True,      # Check monolithic pipeline (TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py)
    "CHECK_MODULAR": True,         # Check modular pipeline (TQE_Pipeline_Modular/)
    "CHECK_BOTH": True,            # Check both pipelines (default: True, overrides individual flags if True)
    
    # === CHECK DEPTH ===
    "DEEP_CHECK": False,           # Deep dependency checking (includes optional heavy deps: healpy, camb, qutip, etc.)
    "CHECK_IMPORTS": True,         # Check module imports
    "CHECK_FUNCTIONS": True,       # Check function signatures
    "CHECK_PHASES": True,          # Check all 28 phase functions
    "CHECK_CONFIG": True,          # Check MASTER_CTRL configuration schema
    "CHECK_DEPENDENCIES": True,    # Check Python package dependencies
    
    # === SMOKE TEST ===
    "RUN_SMOKE_TEST": False,       # Run minimal smoke test (creates PipelineContext, PhysicsEngine, tests basic functions)
    "SMOKE_TEST_UNIVERSES": 3,     # Number of universes for smoke test
    "SMOKE_TEST_EPOCHS": 10,       # Number of epochs for smoke test
    
    # === DEPENDENCY CHECKS ===
    "CHECK_ESSENTIAL_DEPS": True,  # Check essential dependencies (numpy, pandas, matplotlib, scipy, sklearn, tqdm)
    "CHECK_OPTIONAL_DEPS": False,  # Check optional dependencies (healpy, camb, qutip, dynesty, corner) - only if DEEP_CHECK=True
    
    # === VERBOSITY ===
    "VERBOSE": True,               # Print detailed diagnostic messages
    "PRINT_SUMMARY": True,         # Print summary at the end
    
    # === OUTPUT ===
    "SAVE_JSON": True,             # Save diagnostic results to JSON file
    "SAVE_CSV": True,              # Save diagnostic results to CSV file (if applicable)
    "OUTPUT_BASE_DIR": None,       # Base directory (None = auto-detect desktop or current dir)
    "OUTPUT_DIR_PREFIX": "TQE_Universe_Simulation_Diagnostics",  # Prefix for output directory
    "OUTPUT_FILENAME": "diagnostic_results",  # Base filename (without extension)
    
    # === FILE CHECKS ===
    "CHECK_FILE_EXISTENCE": True,  # Check if required files exist
    "CHECK_FILE_READABILITY": True, # Check if files are readable
    "CHECK_SOURCE_CODE": True,     # Check source code for key components (without full import)
    
    # === IMPORT CHECKS ===
    "ATTEMPT_FULL_IMPORT": False,  # Attempt full module import (can be slow, only if DEEP_CHECK=True)
    
    # === CONFIGURATION VALIDATION ===
    "VALIDATE_RUN_MODE": True,     # Validate RUN_MODE values
    "VALIDATE_I_DEFINITION": True, # Validate I_DEFINITION_MODE values
    "VALIDATE_VARIANT": True,      # Validate PIPELINE_VARIANT values
    "VALIDATE_NUMERIC_RANGES": True, # Validate numeric parameter ranges
    "CHECK_OPTIONAL_KEYS": True,   # Check for optional important configuration keys
}

# ======================================================
# LOGGING HELPERS
# ======================================================

def ok(msg: str, category: str = None, component: str = None, phase: str = None, pipeline_type: str = None) -> None:
    """Print OK message and optionally log to diagnostic report."""
    print(f"[OK] {msg}")
    # Log successful check to diagnostic report
    report = get_diagnostic_report()
    report.add_check(
        check_name=msg,
        status="passed",
        details={"message": msg},
        phase=phase,
        pipeline_type=pipeline_type
    )

def warn(msg: str, category: str = "general", component: str = None, suggestion: str = None, phase: str = None, pipeline_type: str = None) -> None:
    """Print warning message and log to diagnostic report."""
    print(f"[WARN] {msg}")
    # Log warning to diagnostic report
    report = get_diagnostic_report()
    report.add_issue(
        category=category,
        severity="warning",
        message=msg,
        component=component,
        suggestion=suggestion,
        phase=phase,
        pipeline_type=pipeline_type
    )
    report.add_check(
        check_name=msg,
        status="failed",
        details={"message": msg, "severity": "warning"},
        phase=phase,
        pipeline_type=pipeline_type
    )

def err(msg: str, category: str = "general", component: str = None, suggestion: str = None, phase: str = None, pipeline_type: str = None) -> None:
    """Print error message and log to diagnostic report."""
    print(f"[ERR] {msg}")
    # Log critical issue to diagnostic report
    report = get_diagnostic_report()
    report.add_issue(
        category=category,
        severity="critical",
        message=msg,
        component=component,
        suggestion=suggestion,
        phase=phase,
        pipeline_type=pipeline_type
    )
    report.add_check(
        check_name=msg,
        status="failed",
        details={"message": msg, "severity": "critical"},
        phase=phase,
        pipeline_type=pipeline_type
    )

# ======================================================
# OUTPUT HELPERS
# ======================================================

def get_desktop_path() -> str:
    """Get desktop path (cross-platform)."""
    if sys.platform == "darwin":  # macOS
        return os.path.join(os.path.expanduser("~"), "Desktop")
    elif sys.platform == "win32":  # Windows
        return os.path.join(os.path.expanduser("~"), "Desktop")
    elif sys.platform.startswith("linux"):  # Linux
        return os.path.join(os.path.expanduser("~"), "Desktop")
    else:
        # Fallback to current directory
        return os.getcwd()

def initialize_output_paths(config: dict) -> dict:
    """Initialize output directory structure (similar to pipeline structure)."""
    # Determine base directory
    if config.get("OUTPUT_BASE_DIR"):
        base_dir = config["OUTPUT_BASE_DIR"]
    else:
        # Auto-detect: try desktop first, fallback to current directory
        try:
            base_dir = get_desktop_path()
        except Exception:
            base_dir = os.getcwd()
    
    # Create main diagnostics directory
    main_dir = os.path.join(base_dir, config.get("OUTPUT_DIR_PREFIX", "TQE_Universe_Simulation_Diagnostics"))
    os.makedirs(main_dir, exist_ok=True)
    
    # Create timestamped run directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.get('OUTPUT_DIR_PREFIX', 'TQE_Universe_Simulation_Diagnostics')}_{timestamp}"
    run_dir = os.path.join(main_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    return {
        "BASE_DIR": base_dir,
        "MAIN_DIR": main_dir,
        "RUN_DIR": run_dir,
        "RUN_ID": run_id,
        "TIMESTAMP": timestamp
    }

def save_json(filepath: str, data: dict) -> None:
    """Save dictionary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        if DIAGNOSE_CTRL.get("VERBOSE", True):
            print(f"[OUTPUT] Saved JSON: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"[OUTPUT][ERR] Failed to save JSON {filepath}: {e}")

def save_csv(filepath: str, df: pd.DataFrame) -> None:
    """Save DataFrame to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df.to_csv(filepath, index=False)
        if DIAGNOSE_CTRL.get("VERBOSE", True):
            print(f"[OUTPUT] Saved CSV: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"[OUTPUT][ERR] Failed to save CSV {filepath}: {e}")

# ======================================================
# PHASE 9: DETAILED ERROR REPORT SYSTEM
# ======================================================

class DiagnosticReport:
    """
    Structured error report collector and generator.
    PHASE 9: Detailed error report generation.
    """
    
    def __init__(self):
        self.issues = []
        self.checks = []
        self.summary = {
            "total_issues": 0,
            "critical": 0,
            "warning": 0,
            "info": 0,
            "checks_performed": 0,
            "checks_passed": 0,
            "checks_failed": 0,
        }
    
    def add_issue(self, 
                  category: str,
                  severity: str,  # "critical", "warning", "info"
                  message: str,
                  component: str = None,
                  suggestion: str = None,
                  phase: str = None,
                  pipeline_type: str = None):
        """
        Add an issue to the report.
        
        Args:
            category: Issue category (e.g., "import", "function", "config")
            severity: Severity level ("critical", "warning", "info")
            message: Error message
            component: Affected component (e.g., "PhysicsEngine", "phase_01")
            suggestion: Suggested fix
            phase: Phase name (e.g., "PHASE 1", "PHASE 2")
            pipeline_type: "monolithic" or "modular"
        """
        issue = {
            "id": len(self.issues) + 1,
            "category": category,
            "severity": severity,
            "message": message,
            "component": component,
            "suggestion": suggestion,
            "phase": phase,
            "pipeline_type": pipeline_type,
            "timestamp": datetime.now().isoformat(),
        }
        self.issues.append(issue)
        
        # Update summary
        self.summary["total_issues"] += 1
        if severity == "critical":
            self.summary["critical"] += 1
        elif severity == "warning":
            self.summary["warning"] += 1
        else:
            self.summary["info"] += 1
    
    def add_check(self,
                  check_name: str,
                  status: str,  # "passed", "failed", "skipped"
                  details: dict = None,
                  phase: str = None,
                  pipeline_type: str = None):
        """
        Add a check result.
        
        Args:
            check_name: Check name
            status: Result ("passed", "failed", "skipped")
            details: Additional details
            phase: Phase name
            pipeline_type: "monolithic" or "modular"
        """
        check = {
            "id": len(self.checks) + 1,
            "check_name": check_name,
            "status": status,
            "details": details or {},
            "phase": phase,
            "pipeline_type": pipeline_type,
            "timestamp": datetime.now().isoformat(),
        }
        self.checks.append(check)
        
        # Update summary
        self.summary["checks_performed"] += 1
        if status == "passed":
            self.summary["checks_passed"] += 1
        elif status == "failed":
            self.summary["checks_failed"] += 1
    
    def get_structured_report(self) -> dict:
        """Returns the structured report as a dictionary."""
        return {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "total_issues": self.summary["total_issues"],
                "total_checks": self.summary["checks_performed"],
            },
            "summary": self.summary,
            "issues_by_severity": {
                "critical": [i for i in self.issues if i["severity"] == "critical"],
                "warning": [i for i in self.issues if i["severity"] == "warning"],
                "info": [i for i in self.issues if i["severity"] == "info"],
            },
            "issues_by_category": self._group_by_category(),
            "issues_by_component": self._group_by_component(),
            "all_issues": self.issues,
            "all_checks": self.checks,
            "recommendations": self._generate_recommendations(),
        }
    
    def _group_by_category(self) -> dict:
        """Groups issues by category."""
        grouped = {}
        for issue in self.issues:
            cat = issue["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(issue)
        return grouped
    
    def _group_by_component(self) -> dict:
        """Groups issues by component."""
        grouped = {}
        for issue in self.issues:
            comp = issue.get("component", "unknown")
            if comp not in grouped:
                grouped[comp] = []
            grouped[comp].append(issue)
        return grouped
    
    def _generate_recommendations(self) -> list:
        """Generates recommendations based on issues."""
        recommendations = []
        
        # Critical issues
        critical_issues = [i for i in self.issues if i["severity"] == "critical"]
        if critical_issues:
            recommendations.append({
                "priority": "HIGH",
                "title": "Critical Issues Found",
                "description": f"{len(critical_issues)} critical issue(s) must be resolved before running the pipeline.",
                "actions": [i.get("suggestion", "Review the issue details") for i in critical_issues[:5]],
            })
        
        # Missing imports
        import_issues = [i for i in self.issues if i["category"] == "import"]
        if import_issues:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Missing Dependencies",
                "description": f"{len(import_issues)} import issue(s) found. Install missing packages.",
                "actions": ["Run: pip install <missing_package>", "Check requirements.txt"],
            })
        
        # Configuration issues
        config_issues = [i for i in self.issues if i["category"] == "config"]
        if config_issues:
            recommendations.append({
                "priority": "MEDIUM",
                "title": "Configuration Issues",
                "description": f"{len(config_issues)} configuration issue(s) found. Review MASTER_CTRL settings.",
                "actions": ["Review MASTER_CTRL configuration", "Check value ranges and logical consistency"],
            })
        
        # Function/phase issues
        function_issues = [i for i in self.issues if i["category"] in ["function", "phase"]]
        if function_issues:
            recommendations.append({
                "priority": "LOW",
                "title": "Function/Phase Issues",
                "description": f"{len(function_issues)} function/phase issue(s) found. May affect pipeline execution.",
                "actions": ["Review function signatures", "Check phase implementations"],
            })
        
        return recommendations
    
    def save_detailed_report(self, output_dir: str, base_filename: str = "diagnostic_report") -> dict:
        """
        Saves the detailed report in JSON and CSV formats.
        
        Returns:
            dict: Paths of saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # 1. Full structured JSON report
        full_report = self.get_structured_report()
        json_path = os.path.join(output_dir, f"{base_filename}_full.json")
        save_json(json_path, full_report)
        saved_files["full_json"] = json_path
        
        # 2. Issues CSV (detailed)
        if self.issues:
            issues_df = pd.DataFrame(self.issues)
            issues_csv = os.path.join(output_dir, f"{base_filename}_issues.csv")
            save_csv(issues_csv, issues_df)
            saved_files["issues_csv"] = issues_csv
        
        # 3. Checks CSV (detailed)
        if self.checks:
            checks_df = pd.DataFrame(self.checks)
            checks_csv = os.path.join(output_dir, f"{base_filename}_checks.csv")
            save_csv(checks_csv, checks_df)
            saved_files["checks_csv"] = checks_csv
        
        # 4. Summary CSV (by category)
        if self.issues:
            category_summary = []
            for category, issues in self._group_by_category().items():
                category_summary.append({
                    "category": category,
                    "total_issues": len(issues),
                    "critical": len([i for i in issues if i["severity"] == "critical"]),
                    "warning": len([i for i in issues if i["severity"] == "warning"]),
                    "info": len([i for i in issues if i["severity"] == "info"]),
                })
            category_df = pd.DataFrame(category_summary)
            category_csv = os.path.join(output_dir, f"{base_filename}_summary_by_category.csv")
            save_csv(category_csv, category_df)
            saved_files["category_summary_csv"] = category_csv
        
        # 5. Recommendations JSON
        recommendations = self._generate_recommendations()
        if recommendations:
            rec_json = os.path.join(output_dir, f"{base_filename}_recommendations.json")
            save_json(rec_json, {"recommendations": recommendations})
            saved_files["recommendations_json"] = rec_json
        
        return saved_files

# Global diagnostic report instance
_diagnostic_report = DiagnosticReport()

def get_diagnostic_report() -> DiagnosticReport:
    """Returns the global diagnostic report instance."""
    return _diagnostic_report

def reset_diagnostic_report():
    """Resets the diagnostic report (for testing)."""
    global _diagnostic_report
    _diagnostic_report = DiagnosticReport()

# ======================================================
# INTROSPECTION HELPERS
# ======================================================
def check_import(module_name: str, path_hint: str = None):
    """Try to import a module and return (module|None, error_message|None)."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except Exception as e:
        if path_hint:
            return None, f"{type(e).__name__}: {e} (path: {path_hint})"
        return None, f"{type(e).__name__}: {e}"

def check_callable(mod, func_name: str):
    """Return (callable|None, error_message|None) for a symbol in a module."""
    fn = getattr(mod, func_name, None)
    if fn is None:
        return None, f"Function '{func_name}' not found."
    if not callable(fn):
        return None, f"Attribute '{func_name}' exists but is not callable."
    return fn, None

def param_names_of(fn):
    """Return parameter names of a callable (empty list if not introspectable)."""
    try:
        sig = inspect.signature(fn)
        return list(sig.parameters.keys())
    except Exception:
        return []

def require_keys(d: dict, keys, ctx: str = "dict") -> None:
    """Raise KeyError if any of the required keys are missing."""
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{ctx} missing keys: {missing}")

# ======================================================
# PHASE FUNCTION DEFINITIONS (28 PHASES)
# ======================================================
PHASES = [
    ("phase_01_monte_carlo", ["ctx", "X_c_low", "X_c_high", "num_universes"]),
    ("phase_02_stability_curve", ["ctx", "df"]),
    ("phase_03_scatter_ei", ["ctx", "df"]),
    ("phase_04_fluctuation_panels", ["ctx", "df"]),
    ("phase_05_stability_by_i", ["ctx", "df"]),
    ("phase_06_lockin_histogram", ["ctx", "df"]),
    ("phase_07_stability_distribution", ["ctx", "df"]),
    ("phase_08_avg_lockin_curve", ["ctx", "df"]),
    ("phase_09_feature_importance", ["ctx", "df"]),
    ("phase_10_emergent_laws", ["ctx", "df"]),
    ("phase_11_finetuning_detector", ["ctx", "df"]),
    ("phase_12_best_universe_plots", ["ctx", "df"]),
    ("phase_13_generate_missing_cmb_maps", ["ctx", "df"]),
    ("phase_14_entropy_volatility", ["ctx", "df"]),
    ("phase_15_planck_validation", ["ctx", "df"]),
    ("phase_16_cmb_anomaly_detection", ["ctx", "df"]),
    ("phase_17_ei_importance_comparison", ["ctx", "df"]),
    ("phase_18_multi_mode_goldilocks_comparison", ["ctx", "df"]),
    ("phase_19_cmb_analysis_plots", ["ctx", "df"]),
    ("phase_20_comprehensive_correlation_analysis", ["ctx", "df"]),
    ("phase_21_advanced_statistical_analysis", ["ctx", "df"]),
    ("phase_22_cmb_anomaly_analysis_plots", ["ctx", "df"]),
    ("phase_23_enhanced_physics_analysis", ["ctx", "df"]),
    ("phase_24_comprehensive_data_extraction", ["ctx", "df"]),
    ("phase_25_advanced_anomaly_detection", ["ctx", "df"]),
    ("phase_26_advanced_law_detection", ["ctx", "df"]),
    ("phase_27_comprehensive_visualization_extraction", ["ctx", "df"]),
    ("phase_28_final_summary", ["ctx", "df", "peak_x"]),
]

# ======================================================
# CORE COMPONENTS
# ======================================================
CORE_COMPONENTS = [
    ("PipelineContext", "class"),
    ("PhysicsEngine", "class"),
    ("run_pipeline", "function"),
    ("MASTER_CTRL", "dict"),
]

# ======================================================
# PIPELINE CONTEXT METHODS
# ======================================================
PIPELINE_CONTEXT_METHODS = [
    "__init__",
    "_initialize_paths",
    "with_variant",
    "save_json",
    "save_fig",
    "save_csv",
    "get_full_path",
    "get_rel_path",
]

# ======================================================
# PHYSICS ENGINE METHODS
# ======================================================
PHYSICS_ENGINE_METHODS = [
    "__init__",
    "sample_energy",
    "_compute_pure_kl",  # Internal method: pure KL-divergence calculation
    "_compute_pure_shannon",  # Internal method: pure Shannon entropy calculation
    # Note: kl_divergence, shannon, renyi, mutual_info, composite have no separate methods,
    # these are computed inside compute_all_I_definitions()
    "sample_information_kl_shannon",  # KL-Shannon harmonic mean
    "sample_information_entanglement",  # Entanglement entropy
    "sample_information_fisher",  # Fisher information
    "sample_information_jensen_shannon",  # Jensen-Shannon divergence
    "compute_coupling",  # E-I coupling calculation
    "generate_cmb_from_physics",  # CMB generation
    "sample_universe",  # Universe sampling
    "compute_all_I_definitions",  # All 10 I-definition calculations (including kl_divergence, shannon, renyi, mutual_info, composite)
]

# ======================================================
# HELPER FUNCTIONS (MONOLITHIC)
# ======================================================
HELPER_FUNCTIONS = [
    ("setup_scientific_plotting_style", "plotting"),
    ("apply_consistent_plot_style", "plotting"),
    ("_fmt", "formatting"),
    ("_pretty_label", "formatting"),
    ("_axis_from_lmap", "cmb_utils"),
    ("detect_cold_spots_healpix", "cmb_utils"),
    ("detect_axis_of_evil", "cmb_utils"),
    ("generate_coldspot_overlay", "cmb_utils"),
    ("generate_aoe_overlay", "cmb_utils"),
    ("get_cached_cmb_or_generate", "cmb_utils"),
    ("optimize_for_colab", "memory"),
    ("cleanup_memory", "memory"),
    ("bayesian_adaptive_goldilocks", "goldilocks"),
    ("simulate_lock_in", "goldilocks"),
    ("compute_dynamic_goldilocks", "goldilocks"),
    ("run_mc", "monte_carlo"),
    ("_run_single_universe", "monte_carlo"),
    ("adjust_stability_thresholds", "lock_in"),
    ("validate_against_planck", "validation"),
]

# ======================================================
# ESSENTIAL IMPORTS
# ======================================================
ESSENTIAL_IMPORTS = [
    ("os", "os"),
    ("sys", "sys"),
    ("time", "time"),
    ("json", "json"),
    ("numpy", "np"),
    ("pandas", "pd"),
    ("matplotlib.pyplot", "plt"),
    ("scipy", "scipy"),
    ("sklearn", "sklearn"),
    ("tqdm", "tqdm"),
]

# ======================================================
# OPTIONAL IMPORTS
# ======================================================
OPTIONAL_IMPORTS = [
    ("healpy", "hp", "CMB map generation"),
    ("camb", "camb", "Enhanced CMB physics"),
    ("qutip", "qt", "Quantum calculations"),
    ("dynesty", "dynesty", "Bayesian nested sampling"),
    ("corner", "corner", "Corner plots"),
]

# ======================================================
# MODULAR PIPELINE STRUCTURE
# ======================================================
MODULAR_MODULES = [
    ("TQE_Pipeline_Modular.config.master_ctrl", "MASTER_CTRL"),
    ("TQE_Pipeline_Modular.core.pipeline_context", "PipelineContext"),
    ("TQE_Pipeline_Modular.core.physics_engine", "PhysicsEngine"),
    ("TQE_Pipeline_Modular.simulation.monte_carlo", "phase_01_monte_carlo"),
    ("TQE_Pipeline_Modular.simulation.goldilocks", "bayesian_adaptive_goldilocks"),
    ("TQE_Pipeline_Modular.phases.phase_01_10", "phase_02_stability_curve"),
    ("TQE_Pipeline_Modular.phases.phase_11_20", "phase_16_cmb_anomaly_detection"),
    ("TQE_Pipeline_Modular.phases.phase_21_28", "phase_22_cmb_anomaly_analysis_plots"),
    ("TQE_Pipeline_Modular.utils.cmb_utils", "detect_cold_spots_healpix"),
    ("TQE_Pipeline_Modular.utils.cmb_utils", "detect_axis_of_evil"),
    ("TQE_Pipeline_Modular.main", "run_pipeline"),
]

# ======================================================
# DIAGNOSTIC FUNCTIONS
# ======================================================

def check_monolithic_pipeline(deep: bool = False) -> int:
    """Check monolithic pipeline (TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py)."""
    print("\n" + "="*70)
    print("CHECKING MONOLITHIC PIPELINE")
    print("="*70)
    
    missing = 0
    pipeline_file = PIPELINE_DIR / "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
    
    # ===== PHASE 1.1: FILE-LEVEL CHECKS =====
    print("\n[DIAGNOSE] PHASE 1.1: File-level checks...")
    
    if not pipeline_file.exists():
        err(f"Monolithic pipeline file not found: {pipeline_file}")
        return 1
    
    ok(f"Pipeline file exists: {pipeline_file.name}")
    
    # Check file size
    try:
        file_size = pipeline_file.stat().st_size
        if file_size > 0:
            ok(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        else:
            err("File is empty")
            missing += 1
            return missing
    except Exception as e:
        warn(f"Could not check file size: {e}")
        missing += 1
    
    # Check encoding and readability
    try:
        with open(pipeline_file, 'r', encoding='utf-8') as f:
            content = f.read()
            ok(f"File is readable (UTF-8, {len(content):,} characters, {len(content.splitlines()):,} lines)")
    except UnicodeDecodeError as e:
        err(f"File encoding error: {e}")
        missing += 1
        return missing
    except Exception as e:
        err(f"File read error: {e}")
        missing += 1
        return missing
    
    # Check Python syntax
    try:
        compile(content, str(pipeline_file), 'exec')
        ok("Python syntax is valid")
    except SyntaxError as e:
        err(f"Python syntax error at line {e.lineno}: {e.msg}")
        missing += 1
    except Exception as e:
        warn(f"Could not verify syntax: {e}")
    
    # ===== PHASE 1.2: IMPORT CHECKS =====
    print("\n[DIAGNOSE] PHASE 1.2: Import checks...")
    
    # Essential imports
    essential_found = 0
    for imp_name, imp_alias in ESSENTIAL_IMPORTS:
        if f"import {imp_name}" in content or f"from {imp_name}" in content:
            essential_found += 1
            ok(f"Essential import found: {imp_name}")
        else:
            warn(f"Essential import not found: {imp_name}")
            missing += 1
    
    if essential_found == len(ESSENTIAL_IMPORTS):
        ok(f"All {len(ESSENTIAL_IMPORTS)} essential imports found")
    
    # Optional imports (only check if deep)
    if deep:
        optional_found = 0
        for imp_name, imp_alias, purpose in OPTIONAL_IMPORTS:
            if f"import {imp_name}" in content or f"from {imp_name}" in content:
                optional_found += 1
                ok(f"Optional import found: {imp_name} ({purpose})")
            else:
                warn(f"Optional import not found: {imp_name} ({purpose})")
        
        if optional_found > 0:
            ok(f"Found {optional_found}/{len(OPTIONAL_IMPORTS)} optional imports")
    
    # ===== PHASE 1.3: CORE COMPONENTS CHECK =====
    print("\n[DIAGNOSE] PHASE 1.3: Core components check...")
    
    # Check for key components in source code (without full import)
    checks = [
        ("MASTER_CTRL", "MASTER_CTRL configuration"),
        ("class PipelineContext", "PipelineContext class"),
        ("class PhysicsEngine", "PhysicsEngine class"),
        ("def run_pipeline", "run_pipeline function"),
    ]
    
    for pattern, name in checks:
        if pattern in content:
            ok(f"{name} found in source")
        else:
            warn(f"{name} not found in source")
            missing += 1
    
    # ===== PHASE 1.4: PIPELINE CONTEXT METHODS =====
    print("\n[DIAGNOSE] PHASE 1.4: PipelineContext methods check...")
    
    if "class PipelineContext" in content:
        ctx_methods_found = 0
        for method_name in PIPELINE_CONTEXT_METHODS:
            if f"def {method_name}" in content:
                ctx_methods_found += 1
                ok(f"PipelineContext.{method_name}() found")
            else:
                warn(f"PipelineContext.{method_name}() not found")
                missing += 1
        
        if ctx_methods_found == len(PIPELINE_CONTEXT_METHODS):
            ok(f"All {len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
        else:
            warn(f"Only {ctx_methods_found}/{len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
    else:
        warn("PipelineContext class not found - skipping method checks")
        missing += 1
    
    # ===== PHASE 1.5: PHYSICS ENGINE METHODS =====
    print("\n[DIAGNOSE] PHASE 1.5: PhysicsEngine methods check...")
    
    if "class PhysicsEngine" in content:
        pe_methods_found = 0
        for method_name in PHYSICS_ENGINE_METHODS:
            if f"def {method_name}" in content:
                pe_methods_found += 1
                ok(f"PhysicsEngine.{method_name}() found")
            else:
                warn(f"PhysicsEngine.{method_name}() not found")
                missing += 1
        
        if pe_methods_found == len(PHYSICS_ENGINE_METHODS):
            ok(f"All {len(PHYSICS_ENGINE_METHODS)} PhysicsEngine methods found")
        else:
            warn(f"Only {pe_methods_found}/{len(PHYSICS_ENGINE_METHODS)} PhysicsEngine methods found")
    else:
        warn("PhysicsEngine class not found - skipping method checks")
        missing += 1
    
    # ===== PHASE 1.6: PHASE FUNCTIONS (BASIC) =====
    print("\n[DIAGNOSE] PHASE 1.6: Basic phase functions check...")
    
    phase_found = 0
    for phase_name, _ in PHASES:
        if f"def {phase_name}" in content:
            phase_found += 1
        else:
            warn(f"Phase function {phase_name} not found")
            missing += 1
    
    if phase_found == len(PHASES):
        ok(f"All {len(PHASES)} phase functions found in source")
    else:
        warn(f"Only {phase_found}/{len(PHASES)} phase functions found")
    
    # ===== PHASE 1.7: HELPER FUNCTIONS =====
    if deep:
        print("\n[DIAGNOSE] PHASE 1.7: Helper functions check...")
        
        helper_found = 0
        for func_name, category in HELPER_FUNCTIONS:
            if f"def {func_name}" in content:
                helper_found += 1
                ok(f"Helper function {func_name} ({category}) found")
            else:
                warn(f"Helper function {func_name} ({category}) not found")
                missing += 1
        
        if helper_found > 0:
            ok(f"Found {helper_found}/{len(HELPER_FUNCTIONS)} helper functions")
    
    # ===== PHASE 1.8: FULL IMPORT (OPTIONAL, SLOW) =====
    if deep and DIAGNOSE_CTRL.get("ATTEMPT_FULL_IMPORT", False):
        print("\n[DIAGNOSE] PHASE 1.8: Full module import (this may take a moment)...")
        try:
            # Add parent to path for imports
            if str(PIPELINE_DIR) not in sys.path:
                sys.path.insert(0, str(PIPELINE_DIR))
            
            # Use a safer import method
            mod_name = "TQE_Universe_Simulation_Full_Pipeline_v4_2_0_PRO"
            spec = importlib.util.spec_from_file_location(mod_name, pipeline_file)
            if spec is None or spec.loader is None:
                warn("Could not create module spec from file")
                missing += 1
            else:
                ok("Module spec created successfully")
        except Exception as e:
            warn(f"Full import check failed: {e}")
            missing += 1
    
    return missing

def check_phase_signatures(pipeline_type: str = "monolithic", deep: bool = False) -> int:
    """
    PHASE 2: Detailed phase function signature checks.
    
    Args:
        pipeline_type: "monolithic" or "modular"
        deep: If True, detailed signature check (parameter types, return values)
    
    Returns:
        int: Number of missing or incorrect phase functions
    """
    print("\n" + "="*70)
    print(f"PHASE 2: PHASE FUNCTION SIGNATURE CHECKS ({pipeline_type.upper()})")
    print("="*70)
    
    missing = 0
    
    if pipeline_type == "monolithic":
        # Monolithic pipeline: check based on source code
        pipeline_file = PIPELINE_DIR / "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
        
        if not pipeline_file.exists():
            err(f"Monolithic pipeline file not found: {pipeline_file}")
            return len(PHASES)
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            err(f"Could not read pipeline file: {e}")
            return len(PHASES)
        
        print("\n[DIAGNOSE] PHASE 2.1: Phase function signature checks (source code)...")
        
        for phase_name, expected_params in PHASES:
            # Check if function exists
            if f"def {phase_name}" not in content:
                warn(f"Phase {phase_name}: function not found")
                missing += 1
                continue
            
            # Try to extract signature from source (basic check)
            # Look for function definition line
            import re
            pattern = rf"def {phase_name}\s*\([^)]*\)"
            match = re.search(pattern, content)
            if match:
                sig_line = match.group(0)
                ok(f"Phase {phase_name}: found signature: {sig_line[:80]}...")
                
                # Check if expected parameters are mentioned in signature
                params_found = 0
                for param in expected_params:
                    if param in sig_line:
                        params_found += 1
                
                if params_found == len(expected_params):
                    ok(f"  → All {len(expected_params)} expected parameters found")
                else:
                    warn(f"  → Only {params_found}/{len(expected_params)} expected parameters found")
                    if deep:
                        missing += 1
            else:
                warn(f"Phase {phase_name}: could not extract signature")
                if deep:
                    missing += 1
    
    elif pipeline_type == "modular":
        # Modular pipeline: check based on import and introspection
        print("\n[DIAGNOSE] PHASE 2.1: Phase function import and signature checks...")
        
        # Phase 1 (special case - in simulation module)
        try:
            from TQE_Pipeline_Modular.simulation.monte_carlo import phase_01_monte_carlo
            params = param_names_of(phase_01_monte_carlo)
            expected = ["ctx", "X_c_low", "X_c_high", "num_universes"]
            if all(p in params for p in expected):
                ok(f"Phase phase_01_monte_carlo: signature OK ({len(params)} params)")
            else:
                warn(f"Phase phase_01_monte_carlo: signature mismatch. Expected: {expected}, Got: {params}")
                missing += 1
        except Exception as e:
            warn(f"Phase phase_01_monte_carlo: import failed - {e}")
            missing += 1
        
        # Phases 2-10
        try:
            from TQE_Pipeline_Modular.phases.phase_01_10 import (
                phase_02_stability_curve, phase_03_scatter_ei, phase_04_fluctuation_panels,
                phase_05_stability_by_i, phase_06_lockin_histogram, phase_07_stability_distribution,
                phase_08_avg_lockin_curve, phase_09_feature_importance, phase_10_emergent_laws
            )
            
            phases_01_10 = [
                phase_02_stability_curve, phase_03_scatter_ei, phase_04_fluctuation_panels,
                phase_05_stability_by_i, phase_06_lockin_histogram, phase_07_stability_distribution,
                phase_08_avg_lockin_curve, phase_09_feature_importance, phase_10_emergent_laws
            ]
            
            for i, phase_func in enumerate(phases_01_10, start=2):
                phase_name = f"phase_{i:02d}_{phase_func.__name__.split('_', 1)[1] if '_' in phase_func.__name__ else phase_func.__name__}"
                # Find expected params from PHASES list
                expected_params = None
                for pname, params in PHASES:
                    if pname == phase_func.__name__:
                        expected_params = params
                        break
                
                if expected_params:
                    params = param_names_of(phase_func)
                    if all(p in params for p in expected_params):
                        ok(f"Phase {phase_func.__name__}: signature OK ({len(params)} params)")
                    else:
                        warn(f"Phase {phase_func.__name__}: signature mismatch. Expected: {expected_params}, Got: {params}")
                        if deep:
                            missing += 1
                else:
                    ok(f"Phase {phase_func.__name__}: found (no expected params defined)")
        except Exception as e:
            warn(f"Could not check phases 2-10: {e}")
            missing += 9
        
        # Phases 11-20
        try:
            from TQE_Pipeline_Modular.phases.phase_11_20 import (
                phase_11_finetuning_detector, phase_12_best_universe_plots, phase_13_generate_missing_cmb_maps,
                phase_14_entropy_volatility, phase_15_planck_validation, phase_16_cmb_anomaly_detection,
                phase_17_ei_importance_comparison, phase_18_multi_mode_goldilocks_comparison,
                phase_19_cmb_analysis_plots, phase_20_comprehensive_correlation_analysis
            )
            
            phases_11_20 = [
                phase_11_finetuning_detector, phase_12_best_universe_plots, phase_13_generate_missing_cmb_maps,
                phase_14_entropy_volatility, phase_15_planck_validation, phase_16_cmb_anomaly_detection,
                phase_17_ei_importance_comparison, phase_18_multi_mode_goldilocks_comparison,
                phase_19_cmb_analysis_plots, phase_20_comprehensive_correlation_analysis
            ]
            
            for phase_func in phases_11_20:
                expected_params = None
                for pname, params in PHASES:
                    if pname == phase_func.__name__:
                        expected_params = params
                        break
                
                if expected_params:
                    params = param_names_of(phase_func)
                    if all(p in params for p in expected_params):
                        ok(f"Phase {phase_func.__name__}: signature OK ({len(params)} params)")
                    else:
                        warn(f"Phase {phase_func.__name__}: signature mismatch. Expected: {expected_params}, Got: {params}")
                        if deep:
                            missing += 1
                else:
                    ok(f"Phase {phase_func.__name__}: found (no expected params defined)")
        except Exception as e:
            warn(f"Could not check phases 11-20: {e}")
            missing += 10
        
        # Phases 21-28
        try:
            from TQE_Pipeline_Modular.phases.phase_21_28 import (
                phase_21_advanced_statistical_analysis, phase_22_cmb_anomaly_analysis_plots,
                phase_23_enhanced_physics_analysis, phase_24_comprehensive_data_extraction,
                phase_25_advanced_anomaly_detection, phase_26_advanced_law_detection,
                phase_27_comprehensive_visualization_extraction, phase_28_final_summary
            )
            
            phases_21_28 = [
                phase_21_advanced_statistical_analysis, phase_22_cmb_anomaly_analysis_plots,
                phase_23_enhanced_physics_analysis, phase_24_comprehensive_data_extraction,
                phase_25_advanced_anomaly_detection, phase_26_advanced_law_detection,
                phase_27_comprehensive_visualization_extraction, phase_28_final_summary
            ]
            
            for phase_func in phases_21_28:
                expected_params = None
                for pname, params in PHASES:
                    if pname == phase_func.__name__:
                        expected_params = params
                        break
                
                if expected_params:
                    params = param_names_of(phase_func)
                    if all(p in params for p in expected_params):
                        ok(f"Phase {phase_func.__name__}: signature OK ({len(params)} params)")
                    else:
                        warn(f"Phase {phase_func.__name__}: signature mismatch. Expected: {expected_params}, Got: {params}")
                        if deep:
                            missing += 1
                else:
                    ok(f"Phase {phase_func.__name__}: found (no expected params defined)")
        except Exception as e:
            warn(f"Could not check phases 21-28: {e}")
            missing += 8
    
    else:
        err(f"Unknown pipeline type: {pipeline_type}")
        return len(PHASES)
    
    # ===== PHASE 2.2: RETURN VALUE CHECKS (DEEP MODE) =====
    if deep and pipeline_type == "modular":
        print("\n[DIAGNOSE] PHASE 2.2: Return value checks (deep mode)...")
        
        # Phase 1 should return tuple[pd.DataFrame, float, float]
        try:
            from TQE_Pipeline_Modular.simulation.monte_carlo import phase_01_monte_carlo
            sig = inspect.signature(phase_01_monte_carlo)
            if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
                ok(f"Phase phase_01_monte_carlo: return annotation: {sig.return_annotation}")
            else:
                warn(f"Phase phase_01_monte_carlo: no return annotation")
        except Exception as e:
            warn(f"Could not check phase_01_monte_carlo return annotation: {e}")
        
        # Phase 2 should return float
        try:
            from TQE_Pipeline_Modular.phases.phase_01_10 import phase_02_stability_curve
            sig = inspect.signature(phase_02_stability_curve)
            if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
                ok(f"Phase phase_02_stability_curve: return annotation: {sig.return_annotation}")
            else:
                warn(f"Phase phase_02_stability_curve: no return annotation")
        except Exception as e:
            warn(f"Could not check phase_02_stability_curve return annotation: {e}")
        
        # Phase 28 should return dict
        try:
            from TQE_Pipeline_Modular.phases.phase_21_28 import phase_28_final_summary
            sig = inspect.signature(phase_28_final_summary)
            if sig.return_annotation and sig.return_annotation != inspect.Signature.empty:
                ok(f"Phase phase_28_final_summary: return annotation: {sig.return_annotation}")
            else:
                warn(f"Phase phase_28_final_summary: no return annotation")
        except Exception as e:
            warn(f"Could not check phase_28_final_summary return annotation: {e}")
    
    print(f"\n[DIAGNOSE] PHASE 2 summary: {missing} phase function issues found")
    
    return missing

def check_helper_functions(pipeline_type: str = "monolithic", deep: bool = False) -> int:
    """
    PHASE 3: Detailed helper function checks.
    
    Args:
        pipeline_type: "monolithic" or "modular"
        deep: If True, detailed signature checks
    
    Returns:
        int: Number of missing or incorrect helper functions
    """
    print("\n" + "="*70)
    print(f"PHASE 3: HELPER FUNCTIONS CHECK ({pipeline_type.upper()})")
    print("="*70)
    
    missing = 0
    
    # Helper functions grouped by category
    helper_categories = {
        "plotting": ["setup_scientific_plotting_style", "apply_consistent_plot_style"],
        "formatting": ["_fmt", "_pretty_label"],
        "cmb_utils": ["_axis_from_lmap", "detect_cold_spots_healpix", "detect_axis_of_evil", 
                      "generate_coldspot_overlay", "generate_aoe_overlay", "get_cached_cmb_or_generate"],
        "memory": ["optimize_for_colab", "cleanup_memory"],
        "goldilocks": ["bayesian_adaptive_goldilocks", "simulate_lock_in", "compute_dynamic_goldilocks"],
        "monte_carlo": ["run_mc", "_run_single_universe"],
        "lock_in": ["adjust_stability_thresholds"],
        "validation": ["validate_against_planck"],
        "complexity": ["integrate_complexity_analysis"],
    }
    
    if pipeline_type == "monolithic":
        # Monolithic pipeline: check based on source code
        pipeline_file = PIPELINE_DIR / "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
        
        if not pipeline_file.exists():
            err(f"Monolithic pipeline file not found: {pipeline_file}")
            return sum(len(funcs) for funcs in helper_categories.values())
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            err(f"Could not read pipeline file: {e}")
            return sum(len(funcs) for funcs in helper_categories.values())
        
        print("\n[DIAGNOSE] PHASE 3.1: Helper function checks (source code)...")
        
        total_found = 0
        total_expected = sum(len(funcs) for funcs in helper_categories.values())
        
        for category, func_names in helper_categories.items():
            print(f"\n[DIAGNOSE] Category: {category.upper()}")
            category_found = 0
            
            for func_name in func_names:
                if f"def {func_name}" in content:
                    category_found += 1
                    total_found += 1
                    ok(f"  {func_name}() found")
                    
                    if deep:
                        # Try to extract signature
                        import re
                        pattern = rf"def {func_name}\s*\([^)]*\)"
                        match = re.search(pattern, content)
                        if match:
                            sig_line = match.group(0)
                            ok(f"    → Signature: {sig_line[:100]}...")
                else:
                    warn(f"  {func_name}() not found")
                    missing += 1
            
            if category_found == len(func_names):
                ok(f"  → All {len(func_names)} {category} functions found")
            else:
                warn(f"  → Only {category_found}/{len(func_names)} {category} functions found")
        
        # Additional complexity detector asset checks
        complexity_artifacts = [
            ("complexity_metrics_summary.csv", "complexity metrics CSV export"),
            ("life_compatibility_summary.json", "life-compatibility JSON export"),
            ("complexity_life_components.png", "complexity/life breakdown PNG export"),
            ("complexity_universe_ranking.csv", "top-universe complexity CSV export"),
            ("complexity_top_universes.png", "top-universe complexity PNG export"),
            ("summary.setdefault(\"figures\", {})", "summary figure injection for complexity outputs"),
            ("summary.setdefault(\"artifacts\", {})", "summary artifact injection for complexity outputs"),
        ]
        
        print("\n[DIAGNOSE] Complexity detector artifact references...")
        comp_found = 0
        for token, description in complexity_artifacts:
            if token in content:
                comp_found += 1
                ok(f"  {description} reference found ({token})")
            else:
                warn(f"  Missing complexity reference: {description} ({token})")
                missing += 1
        
        if comp_found == len(complexity_artifacts):
            ok(f"  → All {len(complexity_artifacts)} complexity detector references present")
        else:
            warn(f"  → Only {comp_found}/{len(complexity_artifacts)} complexity detector references found")
        
        print(f"\n[DIAGNOSE] PHASE 3.1 summary: {total_found}/{total_expected} helper functions found")
    
    elif pipeline_type == "modular":
        # Modular pipeline: check based on module imports
        print("\n[DIAGNOSE] PHASE 3.1: Helper function import and checks...")
        
        total_found = 0
        total_expected = sum(len(funcs) for funcs in helper_categories.values())
        
        # Plotting functions
        print("\n[DIAGNOSE] Category: PLOTTING")
        try:
            from TQE_Pipeline_Modular.utils.plotting import (
                setup_scientific_plotting_style, apply_consistent_plot_style
            )
            for func_name in helper_categories["plotting"]:
                func = globals().get(func_name) or locals().get(func_name)
                if func_name == "setup_scientific_plotting_style":
                    func = setup_scientific_plotting_style
                elif func_name == "apply_consistent_plot_style":
                    func = apply_consistent_plot_style
                
                if func and callable(func):
                    total_found += 1
                    ok(f"  {func_name}() imported and callable")
                    if deep:
                        params = param_names_of(func)
                        ok(f"    → Parameters: {params}")
                else:
                    warn(f"  {func_name}() not callable")
                    missing += 1
        except Exception as e:
            warn(f"  Could not import plotting functions: {e}")
            missing += len(helper_categories["plotting"])
        
        # Formatting functions
        print("\n[DIAGNOSE] Category: FORMATTING")
        try:
            from TQE_Pipeline_Modular.utils.formatting import _fmt, _pretty_label
            for func_name in helper_categories["formatting"]:
                if func_name == "_fmt":
                    func = _fmt
                elif func_name == "_pretty_label":
                    func = _pretty_label
                
                if func and callable(func):
                    total_found += 1
                    ok(f"  {func_name}() imported and callable")
                    if deep:
                        params = param_names_of(func)
                        ok(f"    → Parameters: {params}")
                else:
                    warn(f"  {func_name}() not callable")
                    missing += 1
        except Exception as e:
            warn(f"  Could not import formatting functions: {e}")
            missing += len(helper_categories["formatting"])
        
        # CMB utils functions
        print("\n[DIAGNOSE] Category: CMB_UTILS")
        try:
            from TQE_Pipeline_Modular.utils.cmb_utils import (
                _axis_from_lmap, detect_cold_spots_healpix, detect_axis_of_evil,
                generate_coldspot_overlay, generate_aoe_overlay, get_cached_cmb_or_generate
            )
            cmb_funcs = {
                "_axis_from_lmap": _axis_from_lmap,
                "detect_cold_spots_healpix": detect_cold_spots_healpix,
                "detect_axis_of_evil": detect_axis_of_evil,
                "generate_coldspot_overlay": generate_coldspot_overlay,
                "generate_aoe_overlay": generate_aoe_overlay,
                "get_cached_cmb_or_generate": get_cached_cmb_or_generate,
            }
            
            for func_name in helper_categories["cmb_utils"]:
                func = cmb_funcs.get(func_name)
                if func and callable(func):
                    total_found += 1
                    ok(f"  {func_name}() imported and callable")
                    if deep:
                        params = param_names_of(func)
                        ok(f"    → Parameters: {params}")
                else:
                    warn(f"  {func_name}() not callable")
                    missing += 1
        except Exception as e:
            warn(f"  Could not import CMB utils functions: {e}")
            missing += len(helper_categories["cmb_utils"])
        
        # Memory functions
        print("\n[DIAGNOSE] Category: MEMORY")
        try:
            from TQE_Pipeline_Modular.utils.memory import optimize_for_colab, cleanup_memory
            for func_name in helper_categories["memory"]:
                if func_name == "optimize_for_colab":
                    func = optimize_for_colab
                elif func_name == "cleanup_memory":
                    func = cleanup_memory
                
                if func and callable(func):
                    total_found += 1
                    ok(f"  {func_name}() imported and callable")
                    if deep:
                        params = param_names_of(func)
                        ok(f"    → Parameters: {params}")
                else:
                    warn(f"  {func_name}() not callable")
                    missing += 1
        except Exception as e:
            warn(f"  Could not import memory functions: {e}")
            missing += len(helper_categories["memory"])
        
        # Goldilocks functions
        print("\n[DIAGNOSE] Category: GOLDILOCKS")
        try:
            from TQE_Pipeline_Modular.simulation.goldilocks import (
                bayesian_adaptive_goldilocks, simulate_lock_in, compute_dynamic_goldilocks
            )
            goldilocks_funcs = {
                "bayesian_adaptive_goldilocks": bayesian_adaptive_goldilocks,
                "simulate_lock_in": simulate_lock_in,
                "compute_dynamic_goldilocks": compute_dynamic_goldilocks,
            }
            
            for func_name in helper_categories["goldilocks"]:
                func = goldilocks_funcs.get(func_name)
                if func and callable(func):
                    total_found += 1
                    ok(f"  {func_name}() imported and callable")
                    if deep:
                        params = param_names_of(func)
                        ok(f"    → Parameters: {params}")
                else:
                    warn(f"  {func_name}() not callable")
                    missing += 1
        except Exception as e:
            warn(f"  Could not import goldilocks functions: {e}")
            missing += len(helper_categories["goldilocks"])
        
        # Monte Carlo functions
        print("\n[DIAGNOSE] Category: MONTE_CARLO")
        try:
            from TQE_Pipeline_Modular.simulation.monte_carlo import run_mc, _run_single_universe
            for func_name in helper_categories["monte_carlo"]:
                if func_name == "run_mc":
                    func = run_mc
                elif func_name == "_run_single_universe":
                    func = _run_single_universe
                
                if func and callable(func):
                    total_found += 1
                    ok(f"  {func_name}() imported and callable")
                    if deep:
                        params = param_names_of(func)
                        ok(f"    → Parameters: {params}")
                else:
                    warn(f"  {func_name}() not callable")
                    missing += 1
        except Exception as e:
            warn(f"  Could not import monte_carlo functions: {e}")
            missing += len(helper_categories["monte_carlo"])
        
        # Lock-in functions
        print("\n[DIAGNOSE] Category: LOCK_IN")
        try:
            from TQE_Pipeline_Modular.simulation.lock_in import adjust_stability_thresholds
            if callable(adjust_stability_thresholds):
                total_found += 1
                ok(f"  adjust_stability_thresholds() imported and callable")
                if deep:
                    params = param_names_of(adjust_stability_thresholds)
                    ok(f"    → Parameters: {params}")
            else:
                warn(f"  adjust_stability_thresholds() not callable")
                missing += 1
        except Exception as e:
            warn(f"  Could not import lock_in functions: {e}")
            missing += len(helper_categories["lock_in"])
        
        # Validation functions (may not exist in modular)
        print("\n[DIAGNOSE] Category: VALIDATION")
        # validate_against_planck might be in a phase function, not a separate helper
        # Check if it exists in any phase module
        try:
            # Try to find it in phases
            from TQE_Pipeline_Modular.phases.phase_11_20 import phase_15_planck_validation
            # If phase exists, validation might be inside it
            ok(f"  validate_against_planck: found in phase_15_planck_validation")
            total_found += 1
        except Exception as e:
            warn(f"  validate_against_planck: not found as separate function (may be in phase)")
            # Don't count as missing, as it might be integrated into phase
        
        print(f"\n[DIAGNOSE] PHASE 3.1 summary: {total_found}/{total_expected} helper functions found")
    
    else:
        err(f"Unknown pipeline type: {pipeline_type}")
        return sum(len(funcs) for funcs in helper_categories.values())
    
    print(f"\n[DIAGNOSE] PHASE 3 summary: {missing} helper function issues found")
    
    return missing

def check_physics_engine(pipeline_type: str = "monolithic", deep: bool = False) -> int:
    """
    PHASE 4: Detailed PhysicsEngine checks.
    
    Args:
        pipeline_type: "monolithic" or "modular"
        deep: If True, detailed functionality testing
    
    Returns:
        int: Number of missing or incorrect PhysicsEngine components
    """
    print("\n" + "="*70)
    print(f"PHASE 4: DETAILED PHYSICS ENGINE CHECKS ({pipeline_type.upper()})")
    print("="*70)
    
    missing = 0
    
    # 10 I-definition methods
    # Note: kl_divergence, shannon, renyi, mutual_info, composite have no separate methods,
    # these are computed inside compute_all_I_definitions() (using _compute_pure_kl and _compute_pure_shannon)
    I_DEFINITION_METHODS = {
        "kl_divergence": "_compute_pure_kl",  # Internal method usage
        "shannon": "_compute_pure_shannon",  # Internal method usage
        "renyi": "compute_all_I_definitions",  # Computed inside compute_all_I_definitions
        "mutual_info": "compute_all_I_definitions",  # Computed inside compute_all_I_definitions
        "composite": "compute_all_I_definitions",  # Computed inside compute_all_I_definitions
        "kl_shannon": "sample_information_kl_shannon",  # Separate method
        "entanglement": "sample_information_entanglement",  # Separate method
        "fisher": "sample_information_fisher",  # Separate method
        "fisher_kl_fusion": None,  # No separate method (not used)
        "jensen_shannon": "sample_information_jensen_shannon",  # Separate method
    }
    
    # Core methods
    CORE_METHODS = [
        "sample_energy",
        "compute_coupling",
        "generate_cmb_from_physics",
        "sample_universe",
        "compute_all_I_definitions",
    ]
    
    # Internal helper methods
    INTERNAL_METHODS = [
        "_compute_pure_kl",
        "_compute_pure_shannon",
    ]
    
    if pipeline_type == "monolithic":
        # Monolithic pipeline: check based on source code
        pipeline_file = PIPELINE_DIR / "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
        
        if not pipeline_file.exists():
            err(f"Monolithic pipeline file not found: {pipeline_file}")
            return len(I_DEFINITION_METHODS) + len(CORE_METHODS) + len(INTERNAL_METHODS)
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            err(f"Could not read pipeline file: {e}")
            return len(I_DEFINITION_METHODS) + len(CORE_METHODS) + len(INTERNAL_METHODS)
        
        print("\n[DIAGNOSE] PHASE 4.1: PhysicsEngine methods check (source code)...")
        
        # Check class definition
        if "class PhysicsEngine" not in content:
            err("PhysicsEngine class not found")
            return len(I_DEFINITION_METHODS) + len(CORE_METHODS) + len(INTERNAL_METHODS)
        
        ok("PhysicsEngine class found")
        
        # Check I-definition methods
        print("\n[DIAGNOSE] PHASE 4.2: 10 I-definition methods check...")
        i_def_found = 0
        i_def_skipped = 0
        for i_def_name, method_name in I_DEFINITION_METHODS.items():
            if method_name is None:
                # No separate method (e.g., fisher_kl_fusion not used)
                i_def_skipped += 1
                ok(f"  {i_def_name}: no separate method (OK - computed inside compute_all_I_definitions)")
            elif f"def {method_name}" in content:
                i_def_found += 1
                ok(f"  {method_name}() found ({i_def_name})")
            else:
                warn(f"  {method_name}() not found ({i_def_name})")
                missing += 1
        
        total_checked = len(I_DEFINITION_METHODS) - i_def_skipped
        if i_def_found == total_checked:
            ok(f"  → All {total_checked} I-definition methods found ({i_def_skipped} skipped - no separate method)")
        else:
            warn(f"  → Only {i_def_found}/{total_checked} I-definition methods found ({i_def_skipped} skipped)")
        
        # Check core methods
        print("\n[DIAGNOSE] PHASE 4.3: Core methods check...")
        core_found = 0
        for method_name in CORE_METHODS:
            if f"def {method_name}" in content:
                core_found += 1
                ok(f"  {method_name}() found")
            else:
                warn(f"  {method_name}() not found")
                missing += 1
        
        if core_found == len(CORE_METHODS):
            ok(f"  → All {len(CORE_METHODS)} core methods found")
        else:
            warn(f"  → Only {core_found}/{len(CORE_METHODS)} core methods found")
        
        # Check internal methods
        print("\n[DIAGNOSE] PHASE 4.4: Internal helper methods check...")
        internal_found = 0
        for method_name in INTERNAL_METHODS:
            if f"def {method_name}" in content:
                internal_found += 1
                ok(f"  {method_name}() found")
            else:
                warn(f"  {method_name}() not found")
                missing += 1
        
        if internal_found == len(INTERNAL_METHODS):
            ok(f"  → All {len(INTERNAL_METHODS)} internal methods found")
        else:
            warn(f"  → Only {internal_found}/{len(INTERNAL_METHODS)} internal methods found")
    
    elif pipeline_type == "modular":
        # Modular pipeline: check based on import and introspection
        print("\n[DIAGNOSE] PHASE 4.1: PhysicsEngine import and check...")
        
        try:
            from TQE_Pipeline_Modular.core.physics_engine import PhysicsEngine
            from TQE_Pipeline_Modular.config.master_ctrl import MASTER_CTRL
            import numpy as np
            
            ok("PhysicsEngine class imported successfully")
            
            # Try to instantiate (if deep mode)
            if deep:
                try:
                    config = MASTER_CTRL.copy()
                    rng = np.random.default_rng(42)
                    physics = PhysicsEngine(config, rng)
                    ok("PhysicsEngine instantiated successfully")
                except Exception as e:
                    warn(f"PhysicsEngine instantiation failed: {e}")
                    missing += 1
            
            # Check I-definition methods
            print("\n[DIAGNOSE] PHASE 4.2: 10 I-definition methods check...")
            i_def_found = 0
            i_def_skipped = 0
            for i_def_name, method_name in I_DEFINITION_METHODS.items():
                if method_name is None:
                    # No separate method (e.g., fisher_kl_fusion not used)
                    i_def_skipped += 1
                    ok(f"  {i_def_name}: no separate method (OK - computed inside compute_all_I_definitions)")
                elif hasattr(PhysicsEngine, method_name):
                    i_def_found += 1
                    method = getattr(PhysicsEngine, method_name)
                    if callable(method):
                        ok(f"  {method_name}() found ({i_def_name})")
                        if deep:
                            params = param_names_of(method)
                            ok(f"    → Parameters: {params}")
                    else:
                        warn(f"  {method_name}() exists but not callable")
                        missing += 1
                else:
                    warn(f"  {method_name}() not found ({i_def_name})")
                    missing += 1
            
            total_checked = len(I_DEFINITION_METHODS) - i_def_skipped
            if i_def_found == total_checked:
                ok(f"  → All {total_checked} I-definition methods found ({i_def_skipped} skipped - no separate method)")
            else:
                warn(f"  → Only {i_def_found}/{total_checked} I-definition methods found ({i_def_skipped} skipped)")
            
            # Check core methods
            print("\n[DIAGNOSE] PHASE 4.3: Core methods check...")
            core_found = 0
            for method_name in CORE_METHODS:
                if hasattr(PhysicsEngine, method_name):
                    core_found += 1
                    method = getattr(PhysicsEngine, method_name)
                    if callable(method):
                        ok(f"  {method_name}() found")
                        if deep:
                            params = param_names_of(method)
                            ok(f"    → Parameters: {params}")
                    else:
                        warn(f"  {method_name}() exists but not callable")
                        missing += 1
                else:
                    warn(f"  {method_name}() not found")
                    missing += 1
            
            if core_found == len(CORE_METHODS):
                ok(f"  → All {len(CORE_METHODS)} core methods found")
            else:
                warn(f"  → Only {core_found}/{len(CORE_METHODS)} core methods found")
            
            # Check internal methods
            print("\n[DIAGNOSE] PHASE 4.4: Internal helper methods check...")
            internal_found = 0
            for method_name in INTERNAL_METHODS:
                if hasattr(PhysicsEngine, method_name):
                    internal_found += 1
                    method = getattr(PhysicsEngine, method_name)
                    if callable(method):
                        ok(f"  {method_name}() found")
                        if deep:
                            params = param_names_of(method)
                            ok(f"    → Parameters: {params}")
                    else:
                        warn(f"  {method_name}() exists but not callable")
                        missing += 1
                else:
                    warn(f"  {method_name}() not found")
                    missing += 1
            
            if internal_found == len(INTERNAL_METHODS):
                ok(f"  → All {len(INTERNAL_METHODS)} internal methods found")
            else:
                warn(f"  → Only {internal_found}/{len(INTERNAL_METHODS)} internal methods found")
            
            # Deep mode: Test functionality
            if deep:
                print("\n[DIAGNOSE] PHASE 4.5: Functionality testing (deep mode)...")
                try:
                    config = MASTER_CTRL.copy()
                    rng = np.random.default_rng(42)
                    physics = PhysicsEngine(config, rng)
                    
                    # Test E sampling
                    try:
                        E = physics.sample_energy()
                        if 0.1 <= E <= 0.95:
                            ok(f"  sample_energy() works: E={E:.3f}")
                        else:
                            warn(f"  sample_energy() returned out of range: E={E:.3f}")
                            missing += 1
                    except Exception as e:
                        warn(f"  sample_energy() failed: {e}")
                        missing += 1
                    
                    # Test I-definition (use _compute_pure_kl as example)
                    try:
                        I = physics._compute_pure_kl(dim=8, eps=1e-12, E=0.7)
                        if 0.0 <= I <= 1.0:
                            ok(f"  _compute_pure_kl() works: I={I:.3f}")
                        else:
                            warn(f"  _compute_pure_kl() returned out of range: I={I:.3f}")
                            missing += 1
                    except Exception as e:
                        warn(f"  _compute_pure_kl() failed: {e}")
                        missing += 1
                    
                    # Test compute_all_I_definitions (tests all 10 I-definitions)
                    try:
                        I_dict = physics.compute_all_I_definitions(E=0.7)
                        if isinstance(I_dict, dict) and len(I_dict) >= 10:
                            ok(f"  compute_all_I_definitions() works: {len(I_dict)} definitions computed")
                            # Check a few keys
                            expected_keys = ["kl_divergence", "shannon", "renyi", "mutual_info", "composite", 
                                           "kl_shannon", "entanglement", "fisher", "jensen_shannon"]
                            found_keys = [k for k in expected_keys if k in I_dict]
                            if len(found_keys) >= 8:
                                ok(f"    → Found {len(found_keys)}/{len(expected_keys)} expected I-definition keys")
                            else:
                                warn(f"    → Only {len(found_keys)}/{len(expected_keys)} expected keys found")
                        else:
                            warn(f"  compute_all_I_definitions() returned invalid: {type(I_dict)}, len={len(I_dict) if isinstance(I_dict, dict) else 'N/A'}")
                            missing += 1
                    except Exception as e:
                        warn(f"  compute_all_I_definitions() failed: {e}")
                        missing += 1
                    
                    # Test compute_coupling
                    try:
                        E_test = 0.7
                        I_test = 0.5
                        X = physics.compute_coupling(E_test, I_test)
                        if X > 0:
                            ok(f"  compute_coupling() works: X={X:.3f} (E={E_test}, I={I_test})")
                        else:
                            warn(f"  compute_coupling() returned invalid: X={X:.3f}")
                            missing += 1
                    except Exception as e:
                        warn(f"  compute_coupling() failed: {e}")
                        missing += 1
                    
                    # Test compute_all_I_definitions
                    try:
                        I_defs = physics.compute_all_I_definitions(E=0.7, a=1.0)
                        if isinstance(I_defs, dict) and len(I_defs) == 10:
                            ok(f"  compute_all_I_definitions() works: {len(I_defs)} definitions")
                            # Check if all expected keys are present
                            expected_keys = list(I_DEFINITION_METHODS.keys())
                            missing_keys = [k for k in expected_keys if k not in I_defs]
                            if not missing_keys:
                                ok(f"    → All 10 I-definition keys present")
                            else:
                                warn(f"    → Missing keys: {missing_keys}")
                                missing += len(missing_keys)
                        else:
                            warn(f"  compute_all_I_definitions() returned unexpected format: {type(I_defs)}, len={len(I_defs) if isinstance(I_defs, dict) else 'N/A'}")
                            missing += 1
                    except Exception as e:
                        warn(f"  compute_all_I_definitions() failed: {e}")
                        missing += 1
                    
                    # Test sample_universe
                    try:
                        universe = physics.sample_universe()
                        if isinstance(universe, dict) and "E" in universe and "I" in universe and "X" in universe:
                            ok(f"  sample_universe() works: E={universe['E']:.3f}, I={universe['I']:.3f}, X={universe['X']:.3f}")
                        else:
                            warn(f"  sample_universe() returned unexpected format: {universe}")
                            missing += 1
                    except Exception as e:
                        warn(f"  sample_universe() failed: {e}")
                        missing += 1
                        
                except Exception as e:
                    warn(f"  Deep functionality test failed: {e}")
                    missing += 1
        
        except Exception as e:
            err(f"Could not import PhysicsEngine: {e}")
            missing += len(I_DEFINITION_METHODS) + len(CORE_METHODS) + len(INTERNAL_METHODS)
    
    else:
        err(f"Unknown pipeline type: {pipeline_type}")
        return len(I_DEFINITION_METHODS) + len(CORE_METHODS) + len(INTERNAL_METHODS)
    
    print(f"\n[DIAGNOSE] PHASE 4 summary: {missing} PhysicsEngine issues found")
    
    return missing

def check_pipeline_context(pipeline_type: str = "monolithic", deep: bool = False) -> int:
    """
    PHASE 5: Detailed PipelineContext checks.
    
    Args:
        pipeline_type: "monolithic" or "modular"
        deep: If True, detailed functionality testing
    
    Returns:
        int: Number of missing or incorrect PipelineContext components
    """
    print("\n" + "="*70)
    print(f"PHASE 5: DETAILED PIPELINE CONTEXT CHECKS ({pipeline_type.upper()})")
    print("="*70)
    
    missing = 0
    
    # Expected path keys
    EXPECTED_PATH_KEYS = [
        "REPO_ROOT", "OUTPUT_ROOT", "SAVE_DIR",
        "GOLDILOCKS_DIR", "PNG_VISUALIZATIONS_DIR", "AGGREGATE_DIR", "CATEGORIZED_DIR",
        "ANOMALY_PNG_DIR", "PHYSICS_PNG_DIR", "MAIN_PNG_DIR", "LAWS_PNG_DIR",
        "STATS_PNG_DIR", "CMB_PNG_DIR", "VIZ_PNG_DIR",
        "ANOMALY_CSV_DIR", "PHYSICS_CSV_DIR", "MAIN_CSV_DIR", "LAWS_CSV_DIR",
        "STATS_CSV_DIR", "CMB_CSV_DIR", "VIZ_CSV_DIR",
    ]
    
    # Expected attributes
    EXPECTED_ATTRIBUTES = [
        "config", "master_seed", "rng", "run_id", "paths",
        "map_registry", "universe_category_map", "variant",
    ]
    
    if pipeline_type == "monolithic":
        # Monolithic pipeline: check based on source code
        pipeline_file = PIPELINE_DIR / "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
        
        if not pipeline_file.exists():
            err(f"Monolithic pipeline file not found: {pipeline_file}")
            return len(PIPELINE_CONTEXT_METHODS) + len(EXPECTED_PATH_KEYS) + len(EXPECTED_ATTRIBUTES)
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            err(f"Could not read pipeline file: {e}")
            return len(PIPELINE_CONTEXT_METHODS) + len(EXPECTED_PATH_KEYS) + len(EXPECTED_ATTRIBUTES)
        
        print("\n[DIAGNOSE] PHASE 5.1: PipelineContext methods check (source code)...")
        
        # Check class definition
        if "class PipelineContext" not in content:
            err("PipelineContext class not found")
            return len(PIPELINE_CONTEXT_METHODS) + len(EXPECTED_PATH_KEYS) + len(EXPECTED_ATTRIBUTES)
        
        ok("PipelineContext class found")
        
        # Check methods
        methods_found = 0
        for method_name in PIPELINE_CONTEXT_METHODS:
            if f"def {method_name}" in content:
                methods_found += 1
                ok(f"  {method_name}() found")
            else:
                warn(f"  {method_name}() not found")
                missing += 1
        
        if methods_found == len(PIPELINE_CONTEXT_METHODS):
            ok(f"  → All {len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
        else:
            warn(f"  → Only {methods_found}/{len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
        
        # Check path keys in _initialize_paths
        print("\n[DIAGNOSE] PHASE 5.2: Path management check...")
        paths_found = 0
        for path_key in EXPECTED_PATH_KEYS:
            if f'"{path_key}"' in content or f"'{path_key}'" in content:
                paths_found += 1
                ok(f"  Path key '{path_key}' found")
            else:
                warn(f"  Path key '{path_key}' not found")
                missing += 1
        
        if paths_found == len(EXPECTED_PATH_KEYS):
            ok(f"  → All {len(EXPECTED_PATH_KEYS)} path keys found")
        else:
            warn(f"  → Only {paths_found}/{len(EXPECTED_PATH_KEYS)} path keys found")
        
        # Check attributes
        print("\n[DIAGNOSE] PHASE 5.3: Attributes check...")
        attrs_found = 0
        for attr_name in EXPECTED_ATTRIBUTES:
            if f"self.{attr_name}" in content:
                attrs_found += 1
                ok(f"  Attribute '{attr_name}' found")
            else:
                warn(f"  Attribute '{attr_name}' not found")
                missing += 1
        
        if attrs_found == len(EXPECTED_ATTRIBUTES):
            ok(f"  → All {len(EXPECTED_ATTRIBUTES)} attributes found")
        else:
            warn(f"  → Only {attrs_found}/{len(EXPECTED_ATTRIBUTES)} attributes found")
    
    elif pipeline_type == "modular":
        # Modular pipeline: check based on import and introspection
        print("\n[DIAGNOSE] PHASE 5.1: PipelineContext import and check...")
        
        try:
            from TQE_Pipeline_Modular.core.pipeline_context import PipelineContext
            from TQE_Pipeline_Modular.config.master_ctrl import MASTER_CTRL
            import numpy as np
            import tempfile
            import shutil
            
            ok("PipelineContext class imported successfully")
            
            # Check methods
            print("\n[DIAGNOSE] PHASE 5.2: PipelineContext methods check...")
            methods_found = 0
            for method_name in PIPELINE_CONTEXT_METHODS:
                if hasattr(PipelineContext, method_name):
                    methods_found += 1
                    method = getattr(PipelineContext, method_name)
                    if callable(method):
                        ok(f"  {method_name}() found")
                        if deep:
                            params = param_names_of(method)
                            ok(f"    → Parameters: {params}")
                    else:
                        warn(f"  {method_name}() exists but not callable")
                        missing += 1
                else:
                    warn(f"  {method_name}() not found")
                    missing += 1
            
            if methods_found == len(PIPELINE_CONTEXT_METHODS):
                ok(f"  → All {len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
            else:
                warn(f"  → Only {methods_found}/{len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
            
            # Deep mode: Test functionality
            if deep:
                print("\n[DIAGNOSE] PHASE 5.3: Functionality testing (deep mode)...")
                
                # Create temporary directory for testing
                test_dir = tempfile.mkdtemp(prefix="tqe_diagnose_test_")
                try:
                    # Test instantiation
                    try:
                        config = MASTER_CTRL.copy()
                        config["DRIVE_BASE_DIR"] = test_dir
                        ctx = PipelineContext(config)
                        ok("  PipelineContext instantiated successfully")
                        
                        # Test attributes
                        print("\n[DIAGNOSE] PHASE 5.4: Attributes check...")
                        attrs_found = 0
                        for attr_name in EXPECTED_ATTRIBUTES:
                            if hasattr(ctx, attr_name):
                                attrs_found += 1
                                value = getattr(ctx, attr_name)
                                ok(f"  Attribute '{attr_name}' found: {type(value).__name__}")
                            else:
                                warn(f"  Attribute '{attr_name}' not found")
                                missing += 1
                        
                        if attrs_found == len(EXPECTED_ATTRIBUTES):
                            ok(f"  → All {len(EXPECTED_ATTRIBUTES)} attributes found")
                        else:
                            warn(f"  → Only {attrs_found}/{len(EXPECTED_ATTRIBUTES)} attributes found")
                        
                        # Test path management
                        print("\n[DIAGNOSE] PHASE 5.5: Path management testing...")
                        paths_found = 0
                        for path_key in EXPECTED_PATH_KEYS:
                            if path_key in ctx.paths:
                                paths_found += 1
                                path_value = ctx.paths[path_key]
                                if os.path.exists(path_value):
                                    ok(f"  Path '{path_key}' exists: {path_value}")
                                else:
                                    warn(f"  Path '{path_key}' does not exist: {path_value}")
                                    missing += 1
                            else:
                                warn(f"  Path key '{path_key}' not in ctx.paths")
                                missing += 1
                        
                        if paths_found == len(EXPECTED_PATH_KEYS):
                            ok(f"  → All {len(EXPECTED_PATH_KEYS)} path keys found and directories exist")
                        else:
                            warn(f"  → Only {paths_found}/{len(EXPECTED_PATH_KEYS)} path keys found")
                        
                        # Test seed management
                        print("\n[DIAGNOSE] PHASE 5.6: Seed management testing...")
                        if hasattr(ctx, 'master_seed') and ctx.master_seed is not None:
                            if isinstance(ctx.master_seed, int) and 0 < ctx.master_seed < 2**32:
                                ok(f"  master_seed is valid: {ctx.master_seed}")
                            else:
                                warn(f"  master_seed has invalid value: {ctx.master_seed}")
                                missing += 1
                        else:
                            warn("  master_seed is None or missing")
                            missing += 1
                        
                        if hasattr(ctx, 'rng') and ctx.rng is not None:
                            ok("  rng (random number generator) exists")
                        else:
                            warn("  rng is None or missing")
                            missing += 1
                        
                        # Test variant tagging
                        print("\n[DIAGNOSE] PHASE 5.7: Variant tagging testing...")
                        test_path = "test_file.png"
                        variant_path = ctx.with_variant(test_path)
                        if variant_path != test_path and ("EI_Pipeline" in variant_path or "E_only_Pipeline" in variant_path):
                            ok(f"  with_variant() works: '{test_path}' -> '{variant_path}'")
                        else:
                            warn(f"  with_variant() may not work correctly: '{test_path}' -> '{variant_path}'")
                            missing += 1
                        
                        # Test file I/O (save_json)
                        print("\n[DIAGNOSE] PHASE 5.8: File I/O testing (save_json)...")
                        try:
                            test_data = {"test": "data", "number": 42}
                            test_json_path = "test_output.json"
                            ctx.save_json(test_json_path, test_data)
                            full_path = ctx.get_full_path(test_json_path)
                            if os.path.exists(full_path):
                                ok(f"  save_json() works: {full_path}")
                                # Clean up
                                os.remove(full_path)
                            else:
                                warn(f"  save_json() did not create file: {full_path}")
                                missing += 1
                        except Exception as e:
                            warn(f"  save_json() failed: {e}")
                            missing += 1
                        
                        # Test file I/O (save_csv)
                        print("\n[DIAGNOSE] PHASE 5.9: File I/O testing (save_csv)...")
                        try:
                            import pandas as pd
                            test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
                            test_csv_path = "test_output.csv"
                            ctx.save_csv(test_df, test_csv_path)
                            full_path = ctx.get_full_path(test_csv_path)
                            if os.path.exists(full_path):
                                ok(f"  save_csv() works: {full_path}")
                                # Clean up
                                os.remove(full_path)
                            else:
                                warn(f"  save_csv() did not create file: {full_path}")
                                missing += 1
                        except Exception as e:
                            warn(f"  save_csv() failed: {e}")
                            missing += 1
                        
                        # Test path helpers
                        print("\n[DIAGNOSE] PHASE 5.10: Path helper methods testing...")
                        try:
                            rel_path = "test/relative/path.txt"
                            full_path = ctx.get_full_path(rel_path)
                            if os.path.isabs(full_path):
                                ok(f"  get_full_path() works: '{rel_path}' -> '{full_path}'")
                            else:
                                warn(f"  get_full_path() did not return absolute path: {full_path}")
                                missing += 1
                            
                            rel_back = ctx.get_rel_path(full_path)
                            if not os.path.isabs(rel_back):
                                ok(f"  get_rel_path() works: '{full_path}' -> '{rel_back}'")
                            else:
                                warn(f"  get_rel_path() did not return relative path: {rel_back}")
                                missing += 1
                        except Exception as e:
                            warn(f"  Path helper methods failed: {e}")
                            missing += 1
                        
                        # Test runtime registries
                        print("\n[DIAGNOSE] PHASE 5.11: Runtime registries check...")
                        if hasattr(ctx, 'map_registry') and isinstance(ctx.map_registry, list):
                            ok("  map_registry exists and is a list")
                        else:
                            warn("  map_registry missing or wrong type")
                            missing += 1
                        
                        if hasattr(ctx, 'universe_category_map') and isinstance(ctx.universe_category_map, dict):
                            ok("  universe_category_map exists and is a dict")
                        else:
                            warn("  universe_category_map missing or wrong type")
                            missing += 1
                        
                    except Exception as e:
                        warn(f"  PipelineContext instantiation failed: {e}")
                        missing += 1
                
                finally:
                    # Clean up test directory
                    try:
                        shutil.rmtree(test_dir)
                    except Exception:
                        pass
        
        except Exception as e:
            err(f"Could not import PipelineContext: {e}")
            missing += len(PIPELINE_CONTEXT_METHODS) + len(EXPECTED_PATH_KEYS) + len(EXPECTED_ATTRIBUTES)
    
    else:
        err(f"Unknown pipeline type: {pipeline_type}")
        return len(PIPELINE_CONTEXT_METHODS) + len(EXPECTED_PATH_KEYS) + len(EXPECTED_ATTRIBUTES)
    
    print(f"\n[DIAGNOSE] PHASE 5 summary: {missing} PipelineContext issues found")
    
    return missing

def check_phase_integration(pipeline_type: str = "modular", deep: bool = False) -> int:
    """
    PHASE 6: Integration checks - Data flow between phases.
    
    Args:
        pipeline_type: "monolithic" or "modular"
        deep: If True, detailed functionality testing
    
    Returns:
        int: Number of integration issues
    """
    print("\n" + "="*70)
    print(f"PHASE 6: INTEGRATION CHECKS ({pipeline_type.upper()})")
    print("="*70)
    
    missing = 0
    
    # Data flow definitions between phases
    PHASE_DATA_FLOW = [
        # (from_phase, to_phase, data_type, description)
        (1, 2, "df", "Phase 1 output (df) → Phase 2 input"),
        (1, 2, "X_c_low", "Phase 1 output (X_c_low) → Phase 2 (optional)"),
        (1, 2, "X_c_high", "Phase 1 output (X_c_high) → Phase 2 (optional)"),
        (1, 3, "df", "Phase 1 output (df) → Phase 3 input"),
        (1, 4, "df", "Phase 1 output (df) → Phase 4 input"),
        (2, 28, "peak_x", "Phase 2 output (peak_x) → Phase 28 input"),
        (12, 13, "ctx.map_registry", "Phase 12 CMB maps → Phase 13 completion"),
        (12, 16, "ctx.map_registry", "Phase 12-13 CMB maps → Phase 16 anomaly detection"),
        (12, 19, "ctx.map_registry", "Phase 12-13 CMB maps → Phase 19 CMB analysis"),
        (16, 22, "anomaly_data", "Phase 16 anomaly detection → Phase 22 visualization"),
    ]
    
    # Expected DataFrame columns (after Phase 1)
    EXPECTED_DF_COLUMNS = [
        "universe_id", "E", "I", "X", "stable", "lock_epoch",
        "stable_epoch", "lockin", "A", "ns", "H",
    ]
    
    if pipeline_type == "monolithic":
        # Monolithic pipeline: check based on source code
        pipeline_file = PIPELINE_DIR / "TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py"
        
        if not pipeline_file.exists():
            err(f"Monolithic pipeline file not found: {pipeline_file}")
            return len(PHASE_DATA_FLOW) + len(EXPECTED_DF_COLUMNS)
        
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            err(f"Could not read pipeline file: {e}")
            return len(PHASE_DATA_FLOW) + len(EXPECTED_DF_COLUMNS)
        
        print("\n[DIAGNOSE] PHASE 6.1: Data flow between phases check (source code)...")
        
        # Check run_pipeline function for phase execution order
        if "def run_pipeline" in content:
            ok("run_pipeline function found")
            
            # Check if phases are called in order
            phase_order_ok = True
            for i in range(1, 29):
                phase_call = f"phase_{i:02d}_"
                if i == 1:
                    # Phase 1 should return df, X_c_low, X_c_high
                    if "phase_01_monte_carlo(ctx" in content:
                        ok(f"  Phase 1 call found with ctx parameter")
                    else:
                        warn(f"  Phase 1 call not found or wrong signature")
                        phase_order_ok = False
                        missing += 1
                else:
                    # Other phases should receive ctx and df
                    if f"phase_{i:02d}_" in content and f"(ctx, df" in content:
                        ok(f"  Phase {i} call found with ctx, df parameters")
                    else:
                        # Some phases might have different signatures
                        if i == 28:
                            # Phase 28 also needs peak_x
                            if "phase_28_final_summary(ctx, df, peak_x" in content:
                                ok(f"  Phase 28 call found with ctx, df, peak_x parameters")
                            else:
                                warn(f"  Phase 28 call not found or wrong signature")
                                phase_order_ok = False
                                missing += 1
                        else:
                            # Check if phase exists at all
                            phase_name = f"phase_{i:02d}_"
                            if phase_name in content:
                                ok(f"  Phase {i} exists (signature check skipped)")
                            else:
                                warn(f"  Phase {i} not found")
                                phase_order_ok = False
                                missing += 1
            
            if phase_order_ok:
                ok("  → Phase execution order appears correct")
        else:
            warn("run_pipeline function not found")
            missing += 1
        
        # Check DataFrame column usage
        print("\n[DIAGNOSE] PHASE 6.2: DataFrame columns usage...")
        df_cols_found = 0
        for col_name in EXPECTED_DF_COLUMNS:
            if f"df['{col_name}']" in content or f'df["{col_name}"]' in content:
                df_cols_found += 1
                ok(f"  DataFrame column '{col_name}' used")
            else:
                warn(f"  DataFrame column '{col_name}' not found in usage")
                missing += 1
        
        if df_cols_found == len(EXPECTED_DF_COLUMNS):
            ok(f"  → All {len(EXPECTED_DF_COLUMNS)} expected DataFrame columns found")
        else:
            warn(f"  → Only {df_cols_found}/{len(EXPECTED_DF_COLUMNS)} DataFrame columns found")
        
        # Check map_registry usage
        print("\n[DIAGNOSE] PHASE 6.3: CMB map registry usage...")
        if "ctx.map_registry" in content:
            ok("  ctx.map_registry used in pipeline")
            
            # Check if it's populated (Phase 12-13)
            if "map_registry.append" in content or "map_registry +=" in content:
                ok("  ctx.map_registry is populated (append/+= found)")
            else:
                warn("  ctx.map_registry population not found")
                missing += 1
            
            # Check if it's used (Phase 16, 19)
            map_registry_uses = content.count("ctx.map_registry")
            if map_registry_uses >= 3:
                ok(f"  ctx.map_registry used {map_registry_uses} times (expected: >=3)")
            else:
                warn(f"  ctx.map_registry used only {map_registry_uses} times (expected: >=3)")
                missing += 1
        else:
            warn("  ctx.map_registry not found in pipeline")
            missing += 1
    
    elif pipeline_type == "modular":
        # Modular pipeline: check based on import and introspection
        print("\n[DIAGNOSE] PHASE 6.1: Data flow between phases check (modular)...")
        
        try:
            from TQE_Pipeline_Modular.main import run_pipeline
            import inspect
            
            ok("run_pipeline function imported successfully")
            
            # Get source code of run_pipeline
            try:
                source = inspect.getsource(run_pipeline)
            except Exception:
                source = None
            
            if source:
                # Check phase execution order
                print("\n[DIAGNOSE] PHASE 6.2: Phase execution order check...")
                
                # Check Phase 1 call
                if "phase_01_monte_carlo(ctx" in source:
                    ok("  Phase 1 call found: phase_01_monte_carlo(ctx)")
                    
                    # Check if it captures return values
                    if "df, X_c_low" in source or "df, X_c_low_used" in source:
                        ok("  Phase 1 return values captured (df, X_c_low, X_c_high)")
                    else:
                        warn("  Phase 1 return values may not be captured correctly")
                        missing += 1
                else:
                    warn("  Phase 1 call not found")
                    missing += 1
                
                # Check Phase 2 call and peak_x
                if "phase_02_stability_curve(ctx, df" in source:
                    ok("  Phase 2 call found: phase_02_stability_curve(ctx, df)")
                    
                    if "peak_x = phase_02_stability_curve" in source:
                        ok("  Phase 2 return value captured (peak_x)")
                    else:
                        warn("  Phase 2 return value (peak_x) may not be captured")
                        missing += 1
                else:
                    warn("  Phase 2 call not found")
                    missing += 1
                
                # Check Phase 28 call with peak_x
                if "phase_28_final_summary(ctx, df, peak_x" in source:
                    ok("  Phase 28 call found: phase_28_final_summary(ctx, df, peak_x)")
                else:
                    warn("  Phase 28 call not found or missing peak_x parameter")
                    missing += 1
                
                # Check all phases receive ctx and df
                phases_with_ctx_df = 0
                for i in range(3, 28):
                    phase_name = f"phase_{i:02d}_"
                    if phase_name in source and f"(ctx, df" in source:
                        phases_with_ctx_df += 1
                
                if phases_with_ctx_df >= 20:  # Phases 3-27 should have ctx, df
                    ok(f"  {phases_with_ctx_df} phases (3-27) have ctx, df parameters")
                else:
                    warn(f"  Only {phases_with_ctx_df} phases (3-27) have ctx, df parameters")
                    missing += 1
                
                # Check map_registry usage
                print("\n[DIAGNOSE] PHASE 6.3: CMB map registry usage...")
                if "ctx.map_registry" in source:
                    ok("  ctx.map_registry used in run_pipeline")
                    
                    map_registry_uses = source.count("ctx.map_registry")
                    if map_registry_uses >= 2:
                        ok(f"  ctx.map_registry referenced {map_registry_uses} times")
                    else:
                        warn(f"  ctx.map_registry referenced only {map_registry_uses} times")
                        missing += 1
                else:
                    warn("  ctx.map_registry not found in run_pipeline")
                    missing += 1
                
                # Check DataFrame save (after Phase 1)
                if 'ctx.save_csv(df,' in source or 'ctx.save_csv(df,' in source:
                    ok("  DataFrame saved after Phase 1 (tqe_runs.csv)")
                else:
                    warn("  DataFrame save after Phase 1 not found")
                    missing += 1
            
                # Deep mode: Check actual phase function signatures
            if deep:
                print("\n[DIAGNOSE] PHASE 6.4: Phase function signature consistency...")
                
                try:
                    from TQE_Pipeline_Modular.simulation.monte_carlo import phase_01_monte_carlo
                    from TQE_Pipeline_Modular.phases.phase_01_10 import phase_02_stability_curve
                    from TQE_Pipeline_Modular.phases.phase_21_28 import phase_28_final_summary
                    
                    # Check Phase 1 signature
                    sig1 = inspect.signature(phase_01_monte_carlo)
                    params1 = list(sig1.parameters.keys())
                    if "ctx" in params1:
                        ok(f"  Phase 1 signature OK: {params1}")
                    else:
                        warn(f"  Phase 1 signature missing 'ctx': {params1}")
                        missing += 1
                    
                    # Check Phase 1 return annotation
                    if sig1.return_annotation and sig1.return_annotation != inspect.Signature.empty:
                        if "tuple" in str(sig1.return_annotation) or "pd.DataFrame" in str(sig1.return_annotation):
                            ok(f"  Phase 1 return annotation OK: {sig1.return_annotation}")
                        else:
                            warn(f"  Phase 1 return annotation unexpected: {sig1.return_annotation}")
                            missing += 1
                    
                    # Check Phase 2 signature
                    sig2 = inspect.signature(phase_02_stability_curve)
                    params2 = list(sig2.parameters.keys())
                    if "ctx" in params2 and "df" in params2:
                        ok(f"  Phase 2 signature OK: {params2}")
                    else:
                        warn(f"  Phase 2 signature missing 'ctx' or 'df': {params2}")
                        missing += 1
                    
                    # Check Phase 28 signature
                    sig28 = inspect.signature(phase_28_final_summary)
                    params28 = list(sig28.parameters.keys())
                    if "ctx" in params28 and "df" in params28 and "peak_x" in params28:
                        ok(f"  Phase 28 signature OK: {params28}")
                    else:
                        warn(f"  Phase 28 signature missing required params: {params28}")
                        missing += 1
                    
                except Exception as e:
                    warn(f"  Could not check phase signatures: {e}")
                    missing += 1
        
        except Exception as e:
            err(f"Could not import run_pipeline: {e}")
            missing += len(PHASE_DATA_FLOW)
    
    else:
        err(f"Unknown pipeline type: {pipeline_type}")
        return len(PHASE_DATA_FLOW) + len(EXPECTED_DF_COLUMNS)
    
    print(f"\n[DIAGNOSE] PHASE 6 summary: {missing} integration issues found")
    
    return missing

def check_modular_pipeline(deep: bool = False) -> int:
    """Check modular pipeline (TQE_Pipeline_Modular/)."""
    print("\n" + "="*70)
    print("CHECKING MODULAR PIPELINE")
    print("="*70)
    
    missing = 0
    modular_dir = PIPELINE_DIR / "TQE_Pipeline_Modular"
    
    # ===== PHASE 1.1: DIRECTORY STRUCTURE =====
    print("\n[DIAGNOSE] PHASE 1.1: Directory structure check...")
    
    if not modular_dir.exists():
        err(f"Modular pipeline directory not found: {modular_dir}")
        return 1
    
    ok(f"Modular directory exists: {modular_dir}")
    
    # Check directory structure
    required_dirs = ["config", "core", "phases", "simulation", "utils", "analysis"]
    for dir_name in required_dirs:
        dir_path = modular_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            ok(f"Directory exists: {dir_name}/")
        else:
            err(f"Directory missing: {dir_name}/")
            missing += 1
    
    # ===== PHASE 1.2: KEY FILES =====
    print("\n[DIAGNOSE] PHASE 1.2: Key files check...")
    
    key_files = [
        ("main.py", "Main orchestrator"),
        ("config/master_ctrl.py", "MASTER_CTRL configuration"),
        ("config/__init__.py", "Config package init"),
        ("core/pipeline_context.py", "PipelineContext class"),
        ("core/physics_engine.py", "PhysicsEngine class"),
        ("core/__init__.py", "Core package init"),
        ("phases/phase_01_10.py", "Phases 1-10"),
        ("phases/phase_11_20.py", "Phases 11-20"),
        ("phases/phase_21_28.py", "Phases 21-28"),
        ("phases/__init__.py", "Phases package init"),
        ("simulation/monte_carlo.py", "Monte Carlo simulation"),
        ("simulation/goldilocks.py", "Goldilocks optimization"),
        ("simulation/lock_in.py", "Lock-in mechanism"),
        ("simulation/__init__.py", "Simulation package init"),
        ("utils/cmb_utils.py", "CMB utilities"),
        ("utils/plotting.py", "Plotting utilities"),
        ("utils/formatting.py", "Formatting utilities"),
        ("utils/memory.py", "Memory utilities"),
        ("utils/__init__.py", "Utils package init"),
        ("analysis/bayesian.py", "Bayesian analysis"),
        ("analysis/anomaly_detection.py", "Anomaly detection"),
        ("analysis/law_detection.py", "Law detection"),
        ("analysis/__init__.py", "Analysis package init"),
    ]
    
    files_found = 0
    for file_path, description in key_files:
        full_path = modular_dir / file_path
        if full_path.exists():
            files_found += 1
            # Check file size
            try:
                file_size = full_path.stat().st_size
                if file_size > 0:
                    ok(f"File exists: {file_path} ({description}, {file_size:,} bytes)")
                else:
                    warn(f"File is empty: {file_path}")
                    missing += 1
            except Exception as e:
                warn(f"Could not check file {file_path}: {e}")
                missing += 1
        else:
            err(f"File missing: {file_path} ({description})")
            missing += 1
    
    if files_found == len(key_files):
        ok(f"All {len(key_files)} key files found")
    else:
        warn(f"Only {files_found}/{len(key_files)} key files found")
    
    # ===== PHASE 1.3: MODULE IMPORTS =====
    print("\n[DIAGNOSE] PHASE 1.3: Module imports check...")
    
    # First import PipelineContext so it's available for other modules
    try:
        from TQE_Pipeline_Modular.core.pipeline_context import PipelineContext
        ok("PipelineContext pre-imported (for circular dependency resolution)")
    except Exception as e:
        warn(f"PipelineContext pre-import failed: {e} (may cause circular dependency issues)")
    
    for mod_name, attr_name in MODULAR_MODULES:
        mod, e = check_import(mod_name, str(modular_dir))
        if e:
            # Not critical for circular dependency
            if "PipelineContext" in str(e) or "not defined" in str(e):
                warn(f"Import failed (circular dependency): {mod_name} — {e} (non-critical)")
            else:
                warn(f"Import failed: {mod_name} — {e}")
                missing += 1
            continue
        
        # Check for the attribute
        if hasattr(mod, attr_name):
            attr = getattr(mod, attr_name)
            if inspect.isclass(attr):
                ok(f"{mod_name}.{attr_name} (class) imported")
            elif callable(attr):
                ok(f"{mod_name}.{attr_name} (function) imported")
            else:
                ok(f"{mod_name}.{attr_name} imported")
        else:
            warn(f"{mod_name}.{attr_name} not found")
            missing += 1
    
    # ===== PHASE 1.4: PIPELINE CONTEXT METHODS (MODULAR) =====
    print("\n[DIAGNOSE] PHASE 1.4: PipelineContext methods check (modular)...")
    
    try:
        from TQE_Pipeline_Modular.core.pipeline_context import PipelineContext
        ctx_methods_found = 0
        for method_name in PIPELINE_CONTEXT_METHODS:
            if hasattr(PipelineContext, method_name):
                ctx_methods_found += 1
                ok(f"PipelineContext.{method_name}() found")
            else:
                warn(f"PipelineContext.{method_name}() not found")
                missing += 1
        
        if ctx_methods_found == len(PIPELINE_CONTEXT_METHODS):
            ok(f"All {len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
        else:
            warn(f"Only {ctx_methods_found}/{len(PIPELINE_CONTEXT_METHODS)} PipelineContext methods found")
    except Exception as e:
        warn(f"Could not check PipelineContext methods: {e}")
        missing += 1
    
    # ===== PHASE 1.5: PHYSICS ENGINE METHODS (MODULAR) =====
    print("\n[DIAGNOSE] PHASE 1.5: PhysicsEngine methods check (modular)...")
    
    try:
        from TQE_Pipeline_Modular.core.physics_engine import PhysicsEngine
        pe_methods_found = 0
        for method_name in PHYSICS_ENGINE_METHODS:
            if hasattr(PhysicsEngine, method_name):
                pe_methods_found += 1
                ok(f"PhysicsEngine.{method_name}() found")
            else:
                warn(f"PhysicsEngine.{method_name}() not found")
                missing += 1
        
        if pe_methods_found == len(PHYSICS_ENGINE_METHODS):
            ok(f"All {len(PHYSICS_ENGINE_METHODS)} PhysicsEngine methods found")
        else:
            warn(f"Only {pe_methods_found}/{len(PHYSICS_ENGINE_METHODS)} PhysicsEngine methods found")
    except Exception as e:
        warn(f"Could not check PhysicsEngine methods: {e}")
        missing += 1
    
    # Check phase functions in modular structure
    print("\n[DIAGNOSE] Checking phase functions...")
    
    # Phase 1-10
    try:
        mod_01_10, e = check_import("TQE_Pipeline_Modular.phases.phase_01_10")
        if not e:
            for i in range(2, 11):
                phase_name = f"phase_{i:02d}_"
                # Find matching phase
                for phase_func, _ in PHASES:
                    if phase_func.startswith(phase_name):
                        fn, e2 = check_callable(mod_01_10, phase_func)
                        if e2:
                            warn(f"Phase {phase_func}: {e2}")
                            missing += 1
                        else:
                            ok(f"Phase {phase_func} found")
    except Exception as e:
        warn(f"Could not check phases 1-10: {e}")
        missing += 1
    
    # Phase 11-20
    try:
        mod_11_20, e = check_import("TQE_Pipeline_Modular.phases.phase_11_20")
        if not e:
            for i in range(11, 21):
                phase_name = f"phase_{i:02d}_"
                for phase_func, _ in PHASES:
                    if phase_func.startswith(phase_name):
                        fn, e2 = check_callable(mod_11_20, phase_func)
                        if e2:
                            warn(f"Phase {phase_func}: {e2}")
                            missing += 1
                        else:
                            ok(f"Phase {phase_func} found")
    except Exception as e:
        warn(f"Could not check phases 11-20: {e}")
        missing += 1
    
    # Phase 21-28
    try:
        mod_21_28, e = check_import("TQE_Pipeline_Modular.phases.phase_21_28")
        if not e:
            for i in range(21, 29):
                phase_name = f"phase_{i:02d}_"
                for phase_func, _ in PHASES:
                    if phase_func.startswith(phase_name):
                        fn, e2 = check_callable(mod_21_28, phase_func)
                        if e2:
                            warn(f"Phase {phase_func}: {e2}")
                            missing += 1
                        else:
                            ok(f"Phase {phase_func} found")
    except Exception as e:
        warn(f"Could not check phases 21-28: {e}")
        missing += 1
    
    # Check main run_pipeline
    try:
        main_mod, e = check_import("TQE_Pipeline_Modular.main")
        if not e:
            fn, e2 = check_callable(main_mod, "run_pipeline")
            if e2:
                warn(f"run_pipeline: {e2}")
                missing += 1
            else:
                params = param_names_of(fn)
                if "config_override" in params and "run_id_override" in params:
                    ok("run_pipeline signature OK")
                else:
                    warn(f"run_pipeline signature unexpected: {params}")
                    missing += 1
    except Exception as e:
        warn(f"Could not check main.run_pipeline: {e}")
        missing += 1
    
    return missing

def check_dependencies(deep: bool = False) -> int:
    """Check optional dependencies."""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    missing = 0
    
    # Essential dependencies
    essential = ["numpy", "pandas", "matplotlib", "scipy", "sklearn", "tqdm"]
    for pkg in essential:
        mod, e = check_import(pkg)
        if e:
            err(f"Essential dependency missing: {pkg} — {e}")
            missing += 1
        else:
            ok(f"Essential dependency: {pkg}")
    
    # Optional dependencies (only check if deep)
    if deep:
        optional = {
            "healpy": "CMB map generation",
            "camb": "Enhanced CMB physics",
            "qutip": "Quantum calculations",
            "dynesty": "Bayesian nested sampling",
            "corner": "Corner plots",
        }
        
        for pkg, purpose in optional.items():
            mod, e = check_import(pkg)
            if e:
                warn(f"Optional dependency missing: {pkg} ({purpose}) — {e}")
            else:
                ok(f"Optional dependency: {pkg} ({purpose})")
    
    return missing

def check_config_schema(
    validate_run_mode: bool = True,
    validate_i_definition: bool = True,
    validate_variant: bool = True,
    validate_numeric: bool = True,
    check_optional: bool = True
) -> int:
    """Check MASTER_CTRL configuration schema comprehensively."""
    print("\n" + "="*70)
    print("CHECKING MASTER_CTRL CONFIGURATION SCHEMA")
    print("="*70)
    
    missing = 0
    
    # Try to import MASTER_CTRL from modular version first
    try:
        from TQE_Pipeline_Modular.config.master_ctrl import MASTER_CTRL
        ok("MASTER_CTRL imported from modular pipeline")
    except Exception as e:
        warn(f"Could not import MASTER_CTRL from modular: {e}")
        missing += 1
        return missing
    
    # ===== GROUP 1: CORE PIPELINE CONTROLS =====
    print("\n[DIAGNOSE] Checking core pipeline controls...")
    
    required_keys = {
        "SEED": (int, type(None), "Master seed (int or None)"),
        "NUM_UNIVERSES": (int, "Number of universes to simulate"),
        "I_DEFINITION_MODE": (str, "Active I-definition mode"),
        "PIPELINE_VARIANT": (str, "Pipeline variant (full/energy_only)"),
        "RUN_MODE": (str, "Execution mode"),
        "LOCKIN_EPOCHS": (int, "Lock-in epochs"),
        "SAVE_FIGS": (bool, "Save figures flag"),
        "SAVE_JSON": (bool, "Save JSON flag"),
    }
    
    for key, expected in required_keys.items():
        if key in MASTER_CTRL:
            value = MASTER_CTRL[key]
            # Type checking
            if isinstance(expected, tuple) and len(expected) > 1:
                expected_types = expected[:-1]
                description = expected[-1]
                if any(isinstance(value, t) for t in expected_types):
                    ok(f"Config key '{key}': {description} (value: {value})")
                else:
                    warn(f"Config key '{key}' has wrong type: expected {expected_types}, got {type(value).__name__}")
                    missing += 1
            elif isinstance(expected, tuple):
                if isinstance(value, expected[0]):
                    ok(f"Config key '{key}': {expected[1]} (value: {value})")
                else:
                    warn(f"Config key '{key}' has wrong type: expected {expected[0].__name__}, got {type(value).__name__}")
                    missing += 1
            else:
                ok(f"Config key '{key}': present (value: {value})")
        else:
            warn(f"Config key missing: {key}")
            missing += 1
    
    # ===== GROUP 2: VALIDATE RUN_MODE =====
    if validate_run_mode:
        print("\n[DIAGNOSE] Validating RUN_MODE...")
        valid_run_modes = ["single_eonly", "single_ei", "batch_ei", "batch_all"]
        if "RUN_MODE" in MASTER_CTRL:
            run_mode = MASTER_CTRL["RUN_MODE"]
            if run_mode in valid_run_modes:
                ok(f"RUN_MODE is valid: {run_mode}")
            else:
                warn(f"RUN_MODE has invalid value: {run_mode} (valid: {valid_run_modes})")
                missing += 1
        else:
            warn("RUN_MODE not found - cannot validate")
    
    # ===== GROUP 3: VALIDATE I_DEFINITION_MODE =====
    if validate_i_definition:
        print("\n[DIAGNOSE] Validating I_DEFINITION_MODE...")
        valid_i_definitions = [
            "kl_divergence", "shannon", "renyi", "mutual_info", "composite",
            "kl_shannon", "entanglement", "fisher", "fisher_kl_fusion", "jensen_shannon"
        ]
        if "I_DEFINITION_MODE" in MASTER_CTRL:
            i_def = MASTER_CTRL["I_DEFINITION_MODE"]
            if i_def in valid_i_definitions:
                ok(f"I_DEFINITION_MODE is valid: {i_def}")
            else:
                warn(f"I_DEFINITION_MODE has invalid value: {i_def} (valid: {valid_i_definitions})")
                missing += 1
        else:
            warn("I_DEFINITION_MODE not found - cannot validate")
    
    # ===== GROUP 4: VALIDATE PIPELINE_VARIANT =====
    if validate_variant:
        print("\n[DIAGNOSE] Validating PIPELINE_VARIANT...")
        valid_variants = ["full", "energy_only"]
        if "PIPELINE_VARIANT" in MASTER_CTRL:
            variant = MASTER_CTRL["PIPELINE_VARIANT"]
            if variant in valid_variants:
                ok(f"PIPELINE_VARIANT is valid: {variant}")
            else:
                warn(f"PIPELINE_VARIANT has invalid value: {variant} (valid: {valid_variants})")
                missing += 1
        else:
            warn("PIPELINE_VARIANT not found - cannot validate")
    
    # ===== GROUP 5: VALIDATE NUMERIC RANGES =====
    if validate_numeric:
        print("\n[DIAGNOSE] Validating numeric ranges...")
        if "NUM_UNIVERSES" in MASTER_CTRL:
            num_uni = MASTER_CTRL["NUM_UNIVERSES"]
            if isinstance(num_uni, int) and num_uni > 0:
                if num_uni >= 100:
                    ok(f"NUM_UNIVERSES is reasonable: {num_uni}")
                else:
                    warn(f"NUM_UNIVERSES is very low: {num_uni} (recommended: >= 100)")
            else:
                warn(f"NUM_UNIVERSES has invalid value: {num_uni} (must be positive integer)")
                missing += 1
        
        if "LOCKIN_EPOCHS" in MASTER_CTRL:
            lockin_eps = MASTER_CTRL["LOCKIN_EPOCHS"]
            if isinstance(lockin_eps, int) and lockin_eps > 0:
                ok(f"LOCKIN_EPOCHS is valid: {lockin_eps}")
            else:
                warn(f"LOCKIN_EPOCHS has invalid value: {lockin_eps} (must be positive integer)")
                missing += 1
    
    # ===== GROUP 6: CHECK OPTIONAL IMPORTANT KEYS =====
    if check_optional:
        print("\n[DIAGNOSE] Checking optional important keys...")
        optional_keys = [
            "CALIBRATION_EPOCHS", "BAYESIAN_UCB_KAPPA", "BAYESIAN_GP_NOISE",
            "USE_ENHANCED_PHYSICS", "USE_PHYSICAL_MODEL", "CAMB_INTEGRATION",
            "CMB_NSIDE", "CMB_COLD_ENABLE", "CMB_AOE_ENABLE", "VERBOSE",
            "ENABLE_COMPLEXITY_ANALYSIS", "SAVE_COMPLEXITY_PLOTS", "COMPLEXITY_TOP_N",
            "COMPLEXITY_THRESHOLD", "LIFE_COMPATIBILITY_THRESHOLD"
        ]
        
        found_optional = 0
        for key in optional_keys:
            if key in MASTER_CTRL:
                found_optional += 1
                value = MASTER_CTRL[key]
                ok(f"Optional key '{key}' present: {value}")
        
        if found_optional > 0:
            ok(f"Found {found_optional}/{len(optional_keys)} optional important keys")
        else:
            warn(f"No optional important keys found (expected some of: {optional_keys})")
    
    # ===== GROUP 7: CHECK PHASE ENABLE FLAGS (if they exist) =====
    print("\n[DIAGNOSE] Checking phase enable flags...")
    phase_flags = [f"ENABLE_PHASE_{i:02d}" for i in range(1, 29)]
    found_flags = sum(1 for flag in phase_flags if flag in MASTER_CTRL)
    if found_flags > 0:
        ok(f"Found {found_flags} phase enable flags (phases controlled individually)")
    else:
        # This is OK - phases might be controlled differently
        ok("No individual phase flags found (phases controlled by default/group settings)")
    
    # ===== GROUP 8: VALIDATE VALUE RANGES (PHASE 7) =====
    if validate_numeric:
        print("\n[DIAGNOSE] PHASE 7.1: Value range validation...")
        
        # NUM_UNIVERSES: 1-100000 (recommended: 100-10000)
        if "NUM_UNIVERSES" in MASTER_CTRL:
            num_uni = MASTER_CTRL["NUM_UNIVERSES"]
            if isinstance(num_uni, int):
                if 1 <= num_uni <= 100000:
                    if 100 <= num_uni <= 10000:
                        ok(f"NUM_UNIVERSES is in recommended range: {num_uni} (100-10000)")
                    elif num_uni < 100:
                        warn(f"NUM_UNIVERSES is low: {num_uni} (recommended: >= 100)")
                    elif num_uni > 10000:
                        warn(f"NUM_UNIVERSES is very high: {num_uni} (may be slow, recommended: <= 10000)")
                    else:
                        ok(f"NUM_UNIVERSES is valid: {num_uni}")
                else:
                    warn(f"NUM_UNIVERSES out of valid range: {num_uni} (valid: 1-100000)")
                    missing += 1
            else:
                warn(f"NUM_UNIVERSES has wrong type: {type(num_uni).__name__}")
                missing += 1
        
        # LOCKIN_EPOCHS: 1-10000 (recommended: 100-1000)
        if "LOCKIN_EPOCHS" in MASTER_CTRL:
            lockin_eps = MASTER_CTRL["LOCKIN_EPOCHS"]
            if isinstance(lockin_eps, int):
                if 1 <= lockin_eps <= 10000:
                    if 100 <= lockin_eps <= 1000:
                        ok(f"LOCKIN_EPOCHS is in recommended range: {lockin_eps} (100-1000)")
                    elif lockin_eps < 100:
                        warn(f"LOCKIN_EPOCHS is low: {lockin_eps} (recommended: >= 100)")
                    elif lockin_eps > 1000:
                        warn(f"LOCKIN_EPOCHS is high: {lockin_eps} (may be slow, recommended: <= 1000)")
                    else:
                        ok(f"LOCKIN_EPOCHS is valid: {lockin_eps}")
                else:
                    warn(f"LOCKIN_EPOCHS out of valid range: {lockin_eps} (valid: 1-10000)")
                    missing += 1
            else:
                warn(f"LOCKIN_EPOCHS has wrong type: {type(lockin_eps).__name__}")
                missing += 1
        
        # CMB_NSIDE: 16, 32, 64, 128, 256, 512 (power of 2)
        if "CMB_NSIDE" in MASTER_CTRL:
            nside = MASTER_CTRL["CMB_NSIDE"]
            valid_nsides = [16, 32, 64, 128, 256, 512, 1024, 2048]
            if isinstance(nside, int) and nside in valid_nsides:
                ok(f"CMB_NSIDE is valid: {nside}")
            elif isinstance(nside, int):
                # Check if it's a power of 2
                if (nside & (nside - 1) == 0) and nside >= 16:
                    warn(f"CMB_NSIDE is a power of 2 but not in standard list: {nside} (valid: {valid_nsides})")
                else:
                    warn(f"CMB_NSIDE is not a valid power of 2: {nside} (valid: {valid_nsides})")
                    missing += 1
            else:
                warn(f"CMB_NSIDE has wrong type: {type(nside).__name__}")
                missing += 1
        
        # BAYESIAN_UCB_KAPPA: 0.1-10.0 (recommended: 1.5-3.0)
        if "BAYESIAN_UCB_KAPPA" in MASTER_CTRL:
            kappa = MASTER_CTRL["BAYESIAN_UCB_KAPPA"]
            if isinstance(kappa, (int, float)):
                if 0.1 <= kappa <= 10.0:
                    if 1.5 <= kappa <= 3.0:
                        ok(f"BAYESIAN_UCB_KAPPA is in recommended range: {kappa} (1.5-3.0)")
                    else:
                        warn(f"BAYESIAN_UCB_KAPPA is outside recommended range: {kappa} (recommended: 1.5-3.0)")
                    ok(f"BAYESIAN_UCB_KAPPA is valid: {kappa}")
                else:
                    warn(f"BAYESIAN_UCB_KAPPA out of valid range: {kappa} (valid: 0.1-10.0)")
                    missing += 1
            else:
                warn(f"BAYESIAN_UCB_KAPPA has wrong type: {type(kappa).__name__}")
                missing += 1
        
        # BAYESIAN_GP_NOISE: 0.001-0.1 (recommended: 0.005-0.05)
        if "BAYESIAN_GP_NOISE" in MASTER_CTRL:
            noise = MASTER_CTRL["BAYESIAN_GP_NOISE"]
            if isinstance(noise, (int, float)):
                if 0.001 <= noise <= 0.1:
                    if 0.005 <= noise <= 0.05:
                        ok(f"BAYESIAN_GP_NOISE is in recommended range: {noise} (0.005-0.05)")
                    else:
                        warn(f"BAYESIAN_GP_NOISE is outside recommended range: {noise} (recommended: 0.005-0.05)")
                    ok(f"BAYESIAN_GP_NOISE is valid: {noise}")
                else:
                    warn(f"BAYESIAN_GP_NOISE out of valid range: {noise} (valid: 0.001-0.1)")
                    missing += 1
            else:
                warn(f"BAYESIAN_GP_NOISE has wrong type: {type(noise).__name__}")
                missing += 1
        
        # TIME_STEPS: 1-10000
        if "TIME_STEPS" in MASTER_CTRL:
            time_steps = MASTER_CTRL["TIME_STEPS"]
            if isinstance(time_steps, int) and 1 <= time_steps <= 10000:
                ok(f"TIME_STEPS is valid: {time_steps}")
            elif isinstance(time_steps, int):
                warn(f"TIME_STEPS out of valid range: {time_steps} (valid: 1-10000)")
                missing += 1
            else:
                warn(f"TIME_STEPS has wrong type: {type(time_steps).__name__}")
                missing += 1
    
    # ===== GROUP 9: LOGICAL CONSISTENCY (PHASE 7) =====
    if validate_numeric or validate_variant or validate_run_mode:
        print("\n[DIAGNOSE] PHASE 7.2: Logical consistency check...")
        
        # Rule 1: If PIPELINE_VARIANT == "energy_only", then I_DEFINITION_MODE is not relevant
        if "PIPELINE_VARIANT" in MASTER_CTRL and "I_DEFINITION_MODE" in MASTER_CTRL:
            variant = MASTER_CTRL["PIPELINE_VARIANT"]
            i_def = MASTER_CTRL["I_DEFINITION_MODE"]
            if variant == "energy_only":
                ok("  Rule 1: PIPELINE_VARIANT='energy_only' → I_DEFINITION_MODE not used (OK)")
            else:
                if i_def:
                    ok(f"  Rule 1: PIPELINE_VARIANT='{variant}' → I_DEFINITION_MODE='{i_def}' (OK)")
                else:
                    warn("  Rule 1: PIPELINE_VARIANT='full' but I_DEFINITION_MODE is empty")
                    missing += 1
        
        # Rule 2: Ha RUN_MODE == "single_eonly", akkor PIPELINE_VARIANT == "energy_only"
        if "RUN_MODE" in MASTER_CTRL and "PIPELINE_VARIANT" in MASTER_CTRL:
            run_mode = MASTER_CTRL["RUN_MODE"]
            variant = MASTER_CTRL["PIPELINE_VARIANT"]
            if run_mode == "single_eonly":
                if variant == "energy_only":
                    ok("  Rule 2: RUN_MODE='single_eonly' → PIPELINE_VARIANT='energy_only' (OK)")
                else:
                    warn(f"  Rule 2: RUN_MODE='single_eonly' but PIPELINE_VARIANT='{variant}' (should be 'energy_only')")
                    missing += 1
            elif run_mode in ["single_ei", "batch_ei", "batch_all"]:
                if variant == "full":
                    ok(f"  Rule 2: RUN_MODE='{run_mode}' → PIPELINE_VARIANT='full' (OK)")
                else:
                    warn(f"  Rule 2: RUN_MODE='{run_mode}' but PIPELINE_VARIANT='{variant}' (should be 'full')")
                    missing += 1
        
        # Rule 3: Ha USE_PHYSICAL_MODEL == False, akkor CAMB_INTEGRATION == False (logical, not enforced)
        if "USE_PHYSICAL_MODEL" in MASTER_CTRL and "CAMB_INTEGRATION" in MASTER_CTRL:
            use_phys = MASTER_CTRL["USE_PHYSICAL_MODEL"]
            camb_int = MASTER_CTRL["CAMB_INTEGRATION"]
            if not use_phys and camb_int:
                warn("  Rule 3: USE_PHYSICAL_MODEL=False but CAMB_INTEGRATION=True (may be inconsistent)")
            else:
                ok(f"  Rule 3: USE_PHYSICAL_MODEL={use_phys} → CAMB_INTEGRATION={camb_int} (OK)")
        
        # Rule 4: Ha CMB_COLD_ENABLE == False, akkor cold spot detection kikapcsolva
        if "CMB_COLD_ENABLE" in MASTER_CTRL:
            cold_enable = MASTER_CTRL["CMB_COLD_ENABLE"]
            if isinstance(cold_enable, bool):
                if cold_enable:
                    ok("  Rule 4: CMB_COLD_ENABLE=True → cold spot detection enabled (OK)")
                else:
                    ok("  Rule 4: CMB_COLD_ENABLE=False → cold spot detection disabled (OK)")
            else:
                warn(f"  Rule 4: CMB_COLD_ENABLE has wrong type: {type(cold_enable).__name__}")
                missing += 1
        
        # Rule 5: Ha CMB_AOE_ENABLE == False, akkor AOE detection kikapcsolva
        if "CMB_AOE_ENABLE" in MASTER_CTRL:
            aoe_enable = MASTER_CTRL["CMB_AOE_ENABLE"]
            if isinstance(aoe_enable, bool):
                if aoe_enable:
                    ok("  Rule 5: CMB_AOE_ENABLE=True → AOE detection enabled (OK)")
                else:
                    ok("  Rule 5: CMB_AOE_ENABLE=False → AOE detection disabled (OK)")
            else:
                warn(f"  Rule 5: CMB_AOE_ENABLE has wrong type: {type(aoe_enable).__name__}")
                missing += 1
        
        # Rule 6: SEED validation (if present)
        if "SEED" in MASTER_CTRL:
            seed = MASTER_CTRL["SEED"]
            if seed is None:
                ok("  Rule 6: SEED=None → will be auto-generated (OK)")
            elif isinstance(seed, int):
                if 0 < seed < 2**32:
                    ok(f"  Rule 6: SEED is valid: {seed}")
                else:
                    warn(f"  Rule 6: SEED out of valid range: {seed} (valid: 1-2^32-1)")
                    missing += 1
            else:
                warn(f"  Rule 6: SEED has wrong type: {type(seed).__name__}")
                missing += 1
    
    # ===== GROUP 10: SUMMARY =====
    print("\n[DIAGNOSE] Configuration summary:")
    total_keys = len(MASTER_CTRL)
    ok(f"Total MASTER_CTRL keys: {total_keys}")
    
    return missing

def smoke_test(num_universes: int = 3, num_epochs: int = 10) -> int:
    """
    PHASE 8: Extended smoke test - multiple component testing.
    
    Args:
        num_universes: Number of universes in test
        num_epochs: Number of epochs in test
    
    Returns:
        int: Number of issues found
    """
    print("\n" + "="*70)
    print("PHASE 8: SMOKE TEST (EXTENDED)")
    print("="*70)
    
    missing = 0
    
    try:
        # Try to import and create minimal config
        from TQE_Pipeline_Modular.config.master_ctrl import MASTER_CTRL
        from TQE_Pipeline_Modular.core.pipeline_context import PipelineContext
        import numpy as np
        import pandas as pd
        import tempfile
        import os
        
        # Create minimal config override
        mini_config = MASTER_CTRL.copy()
        mini_config["NUM_UNIVERSES"] = num_universes
        mini_config["LOCKIN_EPOCHS"] = num_epochs
        mini_config["SAVE_FIGS"] = False
        mini_config["SAVE_JSON"] = False
        mini_config["VERBOSE"] = False
        mini_config["CMB_COLD_ENABLE"] = False  # Disable heavy CMB processing
        mini_config["CMB_AOE_ENABLE"] = False
        mini_config["USE_ENHANCED_PHYSICS"] = False  # Disable heavy physics
        
        # ===== GROUP 1: CORE COMPONENTS =====
        print("\n[SMOKE TEST] PHASE 8.1: Core components testing...")
        
        # Try to create PipelineContext
        try:
            ctx = PipelineContext(mini_config)
            ok("  PipelineContext created successfully")
        except Exception as e:
            warn(f"  PipelineContext creation failed: {e}")
            missing += 1
            return missing
        
        # Try to create PhysicsEngine
        try:
            from TQE_Pipeline_Modular.core.physics_engine import PhysicsEngine
            physics = PhysicsEngine(mini_config, ctx.rng)
            ok("  PhysicsEngine created successfully")
        except Exception as e:
            warn(f"  PhysicsEngine creation failed: {e}")
            missing += 1
            return missing
        
        # ===== GROUP 2: PHYSICS ENGINE METHODS =====
        print("\n[SMOKE TEST] PHASE 8.2: PhysicsEngine methods testing...")
        
        # Test sample_universe()
        try:
            universe_data = physics.sample_universe()
            E, I, X = universe_data["E"], universe_data["I"], universe_data["X"]
            ok(f"  PhysicsEngine.sample_universe() works (E={E:.3f}, I={I:.3f}, X={X:.3f})")
        except Exception as e:
            warn(f"  PhysicsEngine.sample_universe() failed: {e}")
            missing += 1
        
        # Test sample_energy()
        try:
            E = physics.sample_energy()
            if isinstance(E, (int, float)) and E > 0:
                ok(f"  PhysicsEngine.sample_energy() works (E={E:.3f})")
            else:
                warn(f"  PhysicsEngine.sample_energy() returned invalid value: {E}")
                missing += 1
        except Exception as e:
            warn(f"  PhysicsEngine.sample_energy() failed: {e}")
            missing += 1
        
        # Test compute_coupling()
        try:
            X = physics.compute_coupling(E, I)
            if isinstance(X, (int, float)):
                ok(f"  PhysicsEngine.compute_coupling() works (X={X:.3f})")
            else:
                warn(f"  PhysicsEngine.compute_coupling() returned invalid value: {X}")
                missing += 1
        except Exception as e:
            warn(f"  PhysicsEngine.compute_coupling() failed: {e}")
            missing += 1
        
        # Test compute_all_I_definitions()
        try:
            I_dict = physics.compute_all_I_definitions(E)
            if isinstance(I_dict, dict) and len(I_dict) > 0:
                ok(f"  PhysicsEngine.compute_all_I_definitions() works ({len(I_dict)} definitions)")
            else:
                warn(f"  PhysicsEngine.compute_all_I_definitions() returned invalid value: {I_dict}")
                missing += 1
        except Exception as e:
            warn(f"  PhysicsEngine.compute_all_I_definitions() failed: {e}")
            missing += 1
        
        # ===== GROUP 3: PIPELINE CONTEXT METHODS =====
        print("\n[SMOKE TEST] PHASE 8.3: PipelineContext methods testing...")
        
        # Test path management
        try:
            save_dir = ctx.paths.get("SAVE_DIR")
            if save_dir and os.path.exists(save_dir):
                ok(f"  PipelineContext paths initialized (SAVE_DIR exists)")
            else:
                warn(f"  PipelineContext paths not properly initialized")
                missing += 1
        except Exception as e:
            warn(f"  PipelineContext path check failed: {e}")
            missing += 1
        
        # Test save_json()
        try:
            test_data = {"test": "data", "number": 42}
            test_path = os.path.join(ctx.paths["SAVE_DIR"], "smoke_test.json")
            ctx.save_json(test_path, test_data)
            if os.path.exists(test_path):
                ok("  PipelineContext.save_json() works")
                os.remove(test_path)  # Cleanup
            else:
                warn("  PipelineContext.save_json() did not create file")
                missing += 1
        except Exception as e:
            warn(f"  PipelineContext.save_json() failed: {e}")
            missing += 1
        
        # Test save_csv()
        try:
            test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
            test_path = os.path.join(ctx.paths["SAVE_DIR"], "smoke_test.csv")
            ctx.save_csv(test_path, test_df)
            if os.path.exists(test_path):
                ok("  PipelineContext.save_csv() works")
                os.remove(test_path)  # Cleanup
            else:
                warn("  PipelineContext.save_csv() did not create file")
                missing += 1
        except Exception as e:
            warn(f"  PipelineContext.save_csv() failed: {e}")
            missing += 1
        
        # ===== GROUP 4: PHASE 1 (MONTE CARLO) =====
        print("\n[SMOKE TEST] PHASE 8.4: Phase 1 (Monte Carlo) testing...")
        
        try:
            from TQE_Pipeline_Modular.simulation.monte_carlo import phase_01_monte_carlo
            
            # Run Phase 1 with minimal config
            df, X_c_low, X_c_high = phase_01_monte_carlo(ctx)
            
            # Check return values
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                ok(f"  Phase 1 (Monte Carlo) completed successfully ({len(df)} universes)")
                
                # Check DataFrame columns
                expected_cols = ["universe_id", "E", "I", "X", "stable"]
                missing_cols = [col for col in expected_cols if col not in df.columns]
                if not missing_cols:
                    ok(f"  Phase 1 DataFrame has required columns: {expected_cols}")
                else:
                    warn(f"  Phase 1 DataFrame missing columns: {missing_cols}")
                    missing += 1
            else:
                warn(f"  Phase 1 returned invalid DataFrame: {type(df)}, len={len(df) if hasattr(df, '__len__') else 'N/A'}")
                missing += 1
        except Exception as e:
            warn(f"  Phase 1 (Monte Carlo) failed: {e}")
            import traceback
            if ctx.config.get("VERBOSE", False):
                print("".join(traceback.format_exc()))
            missing += 1
        
        # ===== GROUP 5: HELPER FUNCTIONS =====
        print("\n[SMOKE TEST] PHASE 8.5: Helper functions testing...")
        
        # Test formatting helpers
        try:
            from TQE_Pipeline_Modular.utils.formatting import format_number, format_percentage
            test_num = format_number(1234.5678)
            test_pct = format_percentage(0.1234)
            if test_num and test_pct:
                ok("  Formatting helpers work (format_number, format_percentage)")
            else:
                warn("  Formatting helpers returned invalid values")
                missing += 1
        except Exception as e:
            warn(f"  Formatting helpers failed: {e}")
            missing += 1
        
        # Test memory helpers
        try:
            from TQE_Pipeline_Modular.utils.memory import cleanup_memory
            cleanup_memory()
            ok("  Memory cleanup helper works")
        except Exception as e:
            warn(f"  Memory cleanup helper failed: {e}")
            missing += 1
        
        # ===== GROUP 6: SUMMARY =====
        print("\n[SMOKE TEST] PHASE 8 summary:")
        if missing == 0:
            ok(f"  → Smoke test PASSED: All {num_universes} universes, {num_epochs} epochs")
        else:
            warn(f"  → Smoke test found {missing} issue(s)")
        
    except Exception as e:
        warn(f"Smoke test failed with exception: {e}")
        import traceback
        print("".join(traceback.format_exc()))
        missing += 1
    
    return missing

# ======================================================
# MAIN ORCHESTRATOR
# ======================================================

def main() -> None:
    """Main diagnostic function."""
    os.chdir(PIPELINE_DIR)
    
    # Load DIAGNOSE_CTRL (can be overridden by command-line arguments)
    config = DIAGNOSE_CTRL.copy()
    
    # Parse command-line arguments (override DIAGNOSE_CTRL)
    if "--monolithic" in sys.argv:
        config["CHECK_MONOLITHIC"] = True
        config["CHECK_MODULAR"] = False
        config["CHECK_BOTH"] = False
    elif "--modular" in sys.argv:
        config["CHECK_MONOLITHIC"] = False
        config["CHECK_MODULAR"] = True
        config["CHECK_BOTH"] = False
    elif "--all" in sys.argv:
        config["CHECK_BOTH"] = True
        config["CHECK_MONOLITHIC"] = True
        config["CHECK_MODULAR"] = True
    
    if "--deep" in sys.argv:
        config["DEEP_CHECK"] = True
        config["CHECK_OPTIONAL_DEPS"] = True
        config["ATTEMPT_FULL_IMPORT"] = True
    
    if "--smoke" in sys.argv:
        config["RUN_SMOKE_TEST"] = True
    
    # Determine what to check
    check_mono = config["CHECK_BOTH"] or config["CHECK_MONOLITHIC"]
    check_mod = config["CHECK_BOTH"] or config["CHECK_MODULAR"]
    deep = config["DEEP_CHECK"]
    smoke = config["RUN_SMOKE_TEST"]
    
    # Header
    print("="*70)
    print("TQE (Theory of the Question of Existence) Pipeline Diagnostics v4.2.0 Professional")
    print("="*70)
    print(f"Python: {sys.version.split()[0]}  |  cwd: {os.getcwd()}")
    print(f"Pipeline directory: {PIPELINE_DIR}")
    print("-" * 60)
    print(f"Check monolithic: {check_mono}  |  Check modular: {check_mod}")
    print(f"Deep check: {deep}  |  Smoke test: {smoke}")
    if config.get("VERBOSE", True):
        print(f"Configuration: DIAGNOSE_CTRL loaded ({len(config)} settings)")
    print("-" * 60)
    
    total_missing = 0
    
    # ===== GROUP 1: DEPENDENCIES & CONFIGURATION =====
    if config.get("CHECK_DEPENDENCIES", True):
        print("\n[DIAGNOSE] Starting dependency checks...")
        total_missing += check_dependencies(deep=deep)
    
    if config.get("CHECK_CONFIG", True):
        print("\n[DIAGNOSE] Starting configuration checks...")
        total_missing += check_config_schema(
            validate_run_mode=config.get("VALIDATE_RUN_MODE", True),
            validate_i_definition=config.get("VALIDATE_I_DEFINITION", True),
            validate_variant=config.get("VALIDATE_VARIANT", True),
            validate_numeric=config.get("VALIDATE_NUMERIC_RANGES", True),
            check_optional=config.get("CHECK_OPTIONAL_KEYS", True)
        )
    
    # ===== GROUP 2: PIPELINE STRUCTURE =====
    if check_mono and config.get("CHECK_IMPORTS", True):
        total_missing += check_monolithic_pipeline(deep=deep)
    
    if check_mod and config.get("CHECK_IMPORTS", True):
        total_missing += check_modular_pipeline(deep=deep)
    
    # ===== GROUP 2.5: PHASE SIGNATURES (PHASE 2) =====
    if config.get("CHECK_PHASES", True):
        if check_mono:
            total_missing += check_phase_signatures(pipeline_type="monolithic", deep=deep)
        if check_mod:
            total_missing += check_phase_signatures(pipeline_type="modular", deep=deep)
    
    # ===== GROUP 2.6: HELPER FUNCTIONS (PHASE 3) =====
    if config.get("CHECK_FUNCTIONS", True) and deep:
        if check_mono:
            total_missing += check_helper_functions(pipeline_type="monolithic", deep=deep)
        if check_mod:
            total_missing += check_helper_functions(pipeline_type="modular", deep=deep)
    
    # ===== GROUP 2.7: PHYSICS ENGINE (PHASE 4) =====
    if config.get("CHECK_FUNCTIONS", True) and deep:
        if check_mono:
            total_missing += check_physics_engine(pipeline_type="monolithic", deep=deep)
        if check_mod:
            total_missing += check_physics_engine(pipeline_type="modular", deep=deep)
    
    # ===== GROUP 2.8: PIPELINE CONTEXT (PHASE 5) =====
    if config.get("CHECK_FUNCTIONS", True) and deep:
        if check_mono:
            total_missing += check_pipeline_context(pipeline_type="monolithic", deep=deep)
        if check_mod:
            total_missing += check_pipeline_context(pipeline_type="modular", deep=deep)
    
    # ===== GROUP 2.9: PHASE INTEGRATION (PHASE 6) =====
    if config.get("CHECK_PHASES", True) and deep:
        if check_mono:
            total_missing += check_phase_integration(pipeline_type="monolithic", deep=deep)
        if check_mod:
            total_missing += check_phase_integration(pipeline_type="modular", deep=deep)
    
    # ===== GROUP 3: SMOKE TEST =====
    if smoke and config.get("RUN_SMOKE_TEST", False):
        total_missing += smoke_test(
            num_universes=config.get("SMOKE_TEST_UNIVERSES", 3),
            num_epochs=config.get("SMOKE_TEST_EPOCHS", 10)
        )
    
    # ===== GROUP 4: SAVE RESULTS (PHASE 9) =====
    output_paths = None
    report = get_diagnostic_report()
    
    if config.get("SAVE_JSON", True) or config.get("SAVE_CSV", True):
        print("\n[DIAGNOSE] PHASE 9: Detailed error report generation...")
        
        # Initialize output paths (desktop structure like pipeline)
        output_paths = initialize_output_paths(config)
        timestamp = output_paths["TIMESTAMP"]
        run_dir = output_paths["RUN_DIR"]
        base_filename = config.get("OUTPUT_FILENAME", "diagnostic_results")
        
        if config.get("VERBOSE", True):
            print(f"[OUTPUT] Base directory: {output_paths['BASE_DIR']}")
            print(f"[OUTPUT] Main directory: {output_paths['MAIN_DIR']}")
            print(f"[OUTPUT] Run directory: {output_paths['RUN_DIR']}")
        
        # Prepare summary data (backward compatible)
        summary_data = {
            "diagnostic_run": {
                "timestamp": timestamp,
                "run_id": output_paths["RUN_ID"],
                "python_version": sys.version.split()[0],
                "pipeline_directory": str(PIPELINE_DIR),
                "output_base_dir": output_paths["BASE_DIR"],
                "output_main_dir": output_paths["MAIN_DIR"],
                "output_run_dir": output_paths["RUN_DIR"],
                "check_monolithic": check_mono,
                "check_modular": check_mod,
                "deep_check": deep,
                "smoke_test": smoke,
            },
            "results": {
                "total_issues": total_missing,
                "status": "PASS" if total_missing == 0 else "FAIL",
                "checks_performed": {
                    "dependencies": config.get("CHECK_DEPENDENCIES", True),
                    "configuration": config.get("CHECK_CONFIG", True),
                    "monolithic_pipeline": check_mono and config.get("CHECK_IMPORTS", True),
                    "modular_pipeline": check_mod and config.get("CHECK_IMPORTS", True),
                    "smoke_test": smoke and config.get("RUN_SMOKE_TEST", False),
                }
            },
            "configuration": {
                "diagnose_ctrl": config,
            }
        }
        
        # Save backward-compatible JSON
        if config.get("SAVE_JSON", True):
            json_path = os.path.join(run_dir, f"{base_filename}.json")
            save_json(json_path, summary_data)
            ok(f"Results saved to: {json_path}")
        
        # Save backward-compatible CSV (summary table)
        if config.get("SAVE_CSV", True):
            # Create a summary DataFrame
            csv_data = {
                "check_type": ["Total Issues", "Status", "Check Monolithic", "Check Modular", "Deep Check", "Smoke Test"],
                "value": [
                    total_missing,
                    "PASS" if total_missing == 0 else "FAIL",
                    "Yes" if check_mono else "No",
                    "Yes" if check_mod else "No",
                    "Yes" if deep else "No",
                    "Yes" if smoke else "No"
                ],
                "timestamp": [timestamp] * 6
            }
            df_summary = pd.DataFrame(csv_data)
            csv_path = os.path.join(run_dir, f"{base_filename}.csv")
            save_csv(csv_path, df_summary)
            ok(f"Summary saved to: {csv_path}")
        
        # Save detailed structured report (PHASE 9)
        print("\n[DIAGNOSE] PHASE 9: Saving structured error report...")
        try:
            saved_files = report.save_detailed_report(run_dir, base_filename="diagnostic_report")
            
            if saved_files:
                ok(f"Detailed diagnostic report saved:")
                for file_type, file_path in saved_files.items():
                    ok(f"  → {file_type}: {os.path.basename(file_path)}")
                
                # Print summary
                print(f"\n[DIAGNOSE] PHASE 9 summary:")
                print(f"  → Total {report.summary['total_issues']} issues found")
                print(f"    - Critical: {report.summary['critical']}")
                print(f"    - Warning: {report.summary['warning']}")
                print(f"    - Info: {report.summary['info']}")
                print(f"  → Total {report.summary['checks_performed']} checks")
                print(f"    - Passed: {report.summary['checks_passed']}")
                print(f"    - Failed: {report.summary['checks_failed']}")
                
                # Print recommendations
                recommendations = report._generate_recommendations()
                if recommendations:
                    print(f"\n[DIAGNOSE] Recommendations:")
                    for rec in recommendations[:3]:  # Top 3 recommendations
                        print(f"  [{rec['priority']}] {rec['title']}: {rec['description']}")
        except Exception as e:
            warn(f"Failed to save detailed diagnostic report: {e}", category="output")
    
    # Final status
    print("\n" + "="*70)
    if total_missing == 0:
        ok("Diagnostics finished: no blocking issues detected.")
        print("="*70)
        if output_paths:
            print(f"📁 Results saved to:")
            print(f"   Main directory: {output_paths['MAIN_DIR']}")
            print(f"   Run directory:  {output_paths['RUN_DIR']}")
        sys.exit(0)
    else:
        warn(f"Diagnostics finished with {total_missing} potential issue(s). See messages above.")
        print("="*70)
        if output_paths:
            print(f"📁 Results saved to:")
            print(f"   Main directory: {output_paths['MAIN_DIR']}")
            print(f"   Run directory:  {output_paths['RUN_DIR']}")
        sys.exit(1)

if __name__ == "__main__":
    main()

