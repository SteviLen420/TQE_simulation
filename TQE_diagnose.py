# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len

"""
tqe_diagnose.py — Pipeline diagnostics (imports, schema, dependencies)
Place this file next to your modules (TQE_* .py) and run:
  python tqe_diagnose.py
Options:
  --deep   : also check heavy optional deps (only if flags say they're needed)
  --smoke  : tiny smoke-run of 1–2 stages with minimal settings (safe & quick)
"""


# --- path bootstrap: repo root + code dir ---
from pathlib import Path
import sys, os

try:
    REPO_ROOT = Path(__file__).resolve().parent
except NameError:  
    REPO_ROOT = Path.cwd()

CODE_DIR = REPO_ROOT / "TQE_EI_UNIVERSE_SIMULATION_Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# Notebook/Colab doesn't define __file__. Handle both cases.
try:
    CODE_DIR = Path(__file__).parent.resolve()
except NameError:
    # Try a few likely locations; fall back to CWD
    candidates = [
        Path(os.environ.get("TQE_CODE_DIR", "")),
        Path("/content/TQE_simulation/TQE_EI_UNIVERSE_SIMULATION_Code"),
        Path.cwd(),
        Path.cwd() / "TQE_EI_UNIVERSE_SIMULATION_Code",
    ]
    CODE_DIR = next((p.resolve() for p in candidates if p and (p / "TQE_01_EI_UNIVERSE_SIMULATION_Master_Control.py").exists()), Path.cwd().resolve())

# ------------- Pretty helpers ------------------------------------------------------
def ok(msg):    print(f"[OK ] {msg}")
def warn(msg):  print(f"[WARN] {msg}")
def err(msg):   print(f"[ERR] {msg}")

def check_import(module_name):
    """Try to import a module and return (module|None, error|None)."""
    try:
        mod = importlib.import_module(module_name)
        return mod, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def check_callable(mod, func_name):
    """Return (callable|None, error|None)."""
    fn = getattr(mod, func_name, None)
    if fn is None:
        return None, f"Function '{func_name}' not found."
    if not callable(fn):
        return None, f"Attribute '{func_name}' exists but is not callable."
    return fn, None

def param_names_of(fn):
    try:
        sig = inspect.signature(fn)
        return list(sig.parameters.keys())
    except Exception:
        return []

def require_keys(d, keys, ctx="dict"):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{ctx} missing keys: {missing}")

# ------------- Stage map (align with your Master Controller) -----------------------
STAGES = [
    ("run_energy_sampling",
     "TQE_05_EI_UNIVERSE_SIMULATION_E_energy_sampling",
     "run_energy_sampling",
     ["active", "tag"]),
    ("run_info_bootstrap",
     "TQE_06_EI_UNIVERSE_SIMULATION_I_information_bootstrap",
     "run_information_bootstrap",
     ["active"]),
    ("run_fluctuation",
     "TQE_07_EI_UNIVERSE_SIMULATION_t_lt_0_fluctuation",
     "run_fluctuation_stage",
     ["active"]),
    ("run_superposition",
     "TQE_08_EI_UNIVERSE_SIMULATION_t_lt_0_superposition",
     "run_superposition_stage",
     ["active"]),
    ("run_lockin",
     "TQE_09_EI_UNIVERSE_SIMULATION_t_eq_0_collapse_LawLockin",
     "run_lockin_stage",
     ["active"]),
    ("run_expansion",
     "TQE_10_EI_UNIVERSE_SIMULATION_t_gt_0_expansion",
     "run_expansion_stage",
     ["active"]),
    ("run_montecarlo",
     "TQE_11_EI_UNIVERSE_SIMULATION_montecarlo",
     "run_montecarlo",
     ["active"]),
    ("run_best_universe",
     "TQE_12_EI_UNIVERSE_SIMULATION_best_universe",
     "run_best_universe",
     ["active"]),
    ("run_cmb_map",
     "TQE_13_EI_UNIVERSE_SIMULATION_cmb_map_generation",
     "run_cmb_generation",
     ["active"]),
    ("run_anomaly_scan",
     "TQE_15_EI_UNIVERSE_SIMULATION_anomaly_cold_spot",
     "run_cold_spot_scan",
     ["active"]),
    ("run_anomaly_scan",
     "TQE_16_EI_UNIVERSE_SIMULATION_anomaly_low_multipole_alignments",
     "run_low_ell_alignments",
     ["active"]),
    ("run_anomaly_scan",
     "TQE_17_EI_UNIVERSE_SIMULATION_anomaly_LackOfLargeAngleCorrelation",
     "run_lack_large_angle",
     ["active"]),
    ("run_anomaly_scan",
     "TQE_18_EI_UNIVERSE_SIMULATION_anomaly_HemisphericalAsymmetry",
     "run_hemi_asymmetry",
     ["active"]),
    ("run_finetune_diag",
     "TQE_14_EI_UNIVERSE_SIMULATION_finetune_diagnostics",
     "run_finetune_diagnostics",
     ["active"]),
    ("run_xai",
     "TQE_19_EI_UNIVERSE_SIMULATION_xai",
     "run_xai",
     ["active"]),
    ("run_manifest",
     "TQE_20_EI_UNIVERSE_SIMULATION_results_manifest",
     "run_results_manifest",
     ["active_cfg", "run_dir"]),
]

# ------------- Main diagnostics ----------------------------------------------------
def main():
    # Ensure code dir on sys.path
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))

    deep = "--deep" in sys.argv
    smoke = "--smoke" in sys.argv

    print("=== TQE pipeline diagnostics ===")
    print(f"Python: {sys.version.split()[0]}  |  cwd: {os.getcwd()}")
    print(f"Code dir: {CODE_DIR}")

    # 1) Load configuration (ACTIVE)
    cfg_mod, e = check_import("TQE_00_EI_UNIVERSE_SIMULATION_config")
    if e:
        err("Cannot import TQE_00_EI_UNIVERSE_SIMULATION_config")
        print("Traceback:\n" + "".join(traceback.format_exc()))
        sys.exit(2)

    try:
        ACTIVE = cfg_mod.ACTIVE
        ok("ACTIVE imported")
    except Exception as ex:
        err(f"Config ACTIVE not available: {ex}")
        sys.exit(2)

    # 2) Basic schema checks
    try:
        require_keys(ACTIVE, ["META", "PIPELINE", "OUTPUTS"], "ACTIVE")
        ok("ACTIVE has required top-level keys")
    except Exception as ex:
        err(str(ex))
        sys.exit(2)

    # Optional sanity on sub-keys
    if "colab_drive" in ACTIVE.get("OUTPUTS", {}):
        require_keys(ACTIVE["OUTPUTS"]["colab_drive"], ["enabled"], "OUTPUTS.colab_drive")
    if "local" in ACTIVE.get("OUTPUTS", {}):
        require_keys(ACTIVE["OUTPUTS"]["local"], ["fig_subdir"], "OUTPUTS.local")

    # 3) Resolve run tag and optional deps needs
    use_info = bool(ACTIVE["PIPELINE"].get("use_information", True))
    tag = "EI" if use_info else "E"
    print(f"Profile: {ACTIVE['META'].get('selected_profile', '<unknown>')} | EI tag: {tag}")

    need_healpy = bool(ACTIVE["PIPELINE"].get("run_cmb_map", False) or ACTIVE["PIPELINE"].get("run_anomaly_scan", False))
    need_qutip  = bool(ACTIVE.get("SUPERPOSITION", {}).get("enabled", False) and ACTIVE.get("SUPERPOSITION", {}).get("use_qutip", False))
    need_shap   = bool(ACTIVE.get("XAI", {}).get("run_shap", False))
    need_lime   = bool(ACTIVE.get("XAI", {}).get("run_lime", False))

    # 4) Check IO helpers (io_paths)
    io_mod, e = check_import("io_paths")
    if e:
        warn(f"io_paths not importable: {e}")
    else:
        for fn in ("resolve_output_paths", "ensure_colab_drive_mounted"):
            if not hasattr(io_mod, fn):
                warn(f"io_paths.{fn} missing")
        ok("io_paths module present")

    # 5) Stage modules: existence, import, function signature
    missing = 0
    for flag, modname, funcname, wanted_params in STAGES:
        if not ACTIVE["PIPELINE"].get(flag, False):
            print(f"[SKIP] {modname} (flag {flag}=False)")
            continue

        py_path = CODE_DIR / f"{modname}.py"
        if not py_path.exists():
            warn(f"File missing: {py_path.name}")
            missing += 1

        mod, e = check_import(modname)
        if e:
            warn(f"Import failed: {modname} — {e}")
            missing += 1
            continue

        fn, e2 = check_callable(mod, funcname)
        if e2:
            warn(f"{modname}.{funcname}: {e2}")
            missing += 1
            continue

        params = param_names_of(fn)
        # Loose check: require all wanted params to be present (order not enforced)
        missing_params = [p for p in wanted_params if p not in params]
        if missing_params:
            warn(f"{modname}.{funcname} signature missing params {missing_params} (has {params})")
            missing += 1
        else:
            ok(f"{modname}.{funcname} import & signature OK")

    # 6) Optional heavy deps (only if --deep)
    if deep:
        def dep_check(pkg, needed):
            if not needed:
                print(f"[SKIP] optional dep '{pkg}' not required by flags")
                return
            try:
                importlib.import_module(pkg)
                ok(f"Optional dependency present: {pkg}")
            except Exception as ex:
                warn(f"Optional dependency MISSING: {pkg} — {ex}")

        dep_check("healpy", need_healpy)
        dep_check("qutip", need_qutip)
        dep_check("shap", need_shap)
        dep_check("lime", need_lime)

    # 7) Optional tiny smoke-run (very small ACTIVE clone)
    if smoke:
        print("--- tiny smoke-run ---")
        try:
            from io_paths import resolve_output_paths
            import copy, tempfile
            mini = copy.deepcopy(ACTIVE)
            # shrink workloads drastically
            mini.setdefault("ENERGY", {})["num_universes"] = 3
            mini["ENERGY"]["time_steps"] = 5
            mini["ENERGY"]["lockin_epochs"] = 5
            mini["ENERGY"]["expansion_epochs"] = 5
            # turn off heavy parts
            mini["PIPELINE"]["run_cmb_map"] = False
            mini["PIPELINE"]["run_anomaly_scan"] = False
            mini["PIPELINE"]["run_xai"] = False
            mini["SUPERPOSITION"]["enabled"] = False

            # temp run dir
            tmp = Path(tempfile.mkdtemp(prefix="tqe_smoke_"))
            # fake a minimal resolve_output_paths contract
            paths = {"primary_run_dir": str(tmp), "fig_dir": str(tmp / "figs")}
            os.makedirs(paths["primary_run_dir"], exist_ok=True)
            os.makedirs(paths["fig_dir"], exist_ok=True)

            # try energy_sampling only (fast)
            mod, _ = check_import("TQE_05_EI_UNIVERSE_SIMULATION_E_energy_sampling")
            if mod and hasattr(mod, "run_energy_sampling"):
                print("[RUN] energy_sampling (mini)")
                mod.run_energy_sampling(active=mini, tag="EI" if mini["PIPELINE"]["use_information"] else "E")
                ok("energy_sampling smoke OK")
            else:
                warn("energy_sampling module/function not found; skipping smoke.")

        except Exception as ex:
            warn(f"Smoke-run failed: {ex}")
            print("".join(traceback.format_exc()))

    # Final status
    if missing == 0:
        ok("Diagnostics finished: no blocking issues detected.")
        sys.exit(0)
    else:
        warn(f"Diagnostics finished with {missing} potential issue(s). See messages above.")
        # non-zero to make CI visibly fail if you wire this into CI
        sys.exit(1)

if __name__ == "__main__":
    main()
