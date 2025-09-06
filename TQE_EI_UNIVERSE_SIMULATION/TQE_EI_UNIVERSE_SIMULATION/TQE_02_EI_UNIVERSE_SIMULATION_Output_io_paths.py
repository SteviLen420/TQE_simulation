# ===================================================================================
# TQE_02_EI_UNIVERSE_SIMULATION_Output_io_paths.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================
"""
Environment and Output Path Resolver for the TQE Simulation.

This module is responsible for the automatic detection of the runtime environment
(e.g., Colab, desktop, cloud) and for creating the necessary folder structure
for saving simulation results (figures, CSV, JSON). For each simulation run,
it generates a unique, timestamped, and profile-tagged directory to ensure
reproducibility and clarity.

The primary purpose of this module is to provide all other pipeline modules
with the correct save paths from a central, reliable source, regardless of
where the code is being executed. It also supports the simultaneous mirroring
of outputs to multiple storage targets (e.g., a local folder and Google Drive).

Key Functions:
  - `resolve_output_paths()`: The central function that detects the environment,
    generates the unique run ID, creates the necessary directories based on the
    config, and returns a dictionary of the fully resolved, absolute paths.
  - `ensure_colab_drive_mounted()`: A helper function that checks if Google
    Drive is mounted in a Colab environment and attempts to mount it if necessary.

Inputs:
  - `config.py`: It uses the `ENV`, `OUTPUTS`, and `META` sections of the `ACTIVE`
    configuration dictionary to function.
  - Environment Variables: It checks for `TQE_PROFILE` and `TQE_DESKTOP_DIR`
    to override the run profile and the desktop save location, respectively.

Outputs:
  - Folder Structure: Creates the run-specific directories on the filesystem.
  - Data Structure: Returns a dictionary (`dict`) containing all generated
    paths (e.g., `primary_run_dir`, `fig_dir`, `mirrors`).

Notes:
  - This module requires write permissions for the output directories specified
    in `config.py`.
  - The "desktop" environment detection is platform-specific (Windows/Linux/macOS).
"""
# ===================================================================================

from config import ACTIVE
import os, platform, time, pathlib
from typing import Dict, List

# -----------------------------------------------------------------------------------
# Helpers for environment detection and path resolution
# -----------------------------------------------------------------------------------

def _is_colab(active_cfg: dict) -> bool:
    """Detect Colab by checking known environment variables."""
    markers = active_cfg["ENV"].get("colab_markers", [])
    return any(k in os.environ for k in markers)

def _desktop_dir(active_cfg: dict) -> str:
    """Resolve a Desktop path for saving when not in Colab/Cloud."""
    env_key = active_cfg["OUTPUTS"]["local"].get("desktop_env_var", "TQE_DESKTOP_DIR")
    if os.environ.get(env_key):
        return os.path.expanduser(os.environ[env_key])

    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        for p in (os.path.join(home, "Desktop"),
                  os.path.join(home, "OneDrive", "Desktop")):
            if os.path.isdir(p):
                return p
    else:
        p = os.path.join(home, "Desktop")
        if os.path.isdir(p):
            return p
    return os.getcwd()

def _run_id(meta_cfg: dict, active_cfg: dict) -> str:
    """Build a run_id with timestamp and optional EI/profile tags."""
    prefix = meta_cfg.get("RUN_ID_PREFIX", "")
    fmt    = meta_cfg.get("RUN_ID_FORMAT", "%Y%m%d_%H%M%S")
    rid    = prefix + time.strftime(fmt)

    # Append -EI / -E tag if requested
    if meta_cfg.get("append_ei_to_run_id", False):
        ei_tag = "EI" if active_cfg["PIPELINE"].get("use_information", True) else "E"
        rid += f"-{ei_tag}"

    # Append profile name if requested
    if active_cfg["OUTPUTS"].get("tag_profile_in_runid", False):
        prof = os.environ.get("TQE_PROFILE", None)
        if prof:
            rid += f"-{prof}"

    return rid

def _resolve_environment(active_cfg: dict) -> str:
    """
    Decide which environment we are in: 'colab' | 'cloud' | 'desktop'.

    Priority:
      1) ENV.force_environment if set
      2) Auto-detect Colab
      3) Cloud if bucket_url is set
      4) Fallback: desktop
    """
    forced = active_cfg["ENV"].get("force_environment")
    if forced in {"colab", "cloud", "desktop"}:
        return forced

    if active_cfg["ENV"].get("auto_detect", True):
        if _is_colab(active_cfg):
            return "colab"
        if active_cfg["OUTPUTS"]["cloud"].get("enabled") and active_cfg["OUTPUTS"]["cloud"].get("bucket_url"):
            return "cloud"
    return "desktop"

# -----------------------------------------------------------------------------------
# Main resolver
# -----------------------------------------------------------------------------------

def resolve_output_paths(active_cfg: dict) -> Dict[str, str]:
    """
    Main function to resolve output directories.

    - Colab: primary = Google Drive, mirror = /content
    - Cloud: primary = Desktop (cloud sync handled separately)
    - Desktop: primary = Desktop or local.base_dir

    Returns:
        dict with keys:
          - env
          - run_id
          - primary_run_dir
          - fig_dir
          - mirrors
          - cloud_bucket
    """
    # Ensure Drive is mounted if needed
    ensure_colab_drive_mounted(active_cfg)

    env     = _resolve_environment(active_cfg)
    outputs = active_cfg["OUTPUTS"]
    meta    = active_cfg["META"]
    run_id  = _run_id(meta, active_cfg)

    # Disable Drive outside Colab
    if env != "colab":
        outputs["colab_drive"]["enabled"] = False

    # Enable cloud if bucket URL exists
    if outputs["cloud"].get("bucket_url") and not outputs["cloud"].get("enabled"):
        outputs["cloud"]["enabled"] = True

    # Primary base dir
    if env == "colab" and outputs["colab_drive"].get("enabled", False):
        primary_base = outputs["colab_drive"]["base_dir"]
    else:
        local_cfg   = outputs.get("local", {})
        local_base  = local_cfg.get("base_dir")
        if local_base and str(local_base).strip():
            primary_base = local_base
        else:
            primary_base = os.path.join(
                _desktop_dir(active_cfg),
                local_cfg.get("desktop_subdir", "TQE_Output")
            )

    # Create directories
    primary_run_dir = os.path.join(primary_base, run_id)
    fig_sub         = outputs["local"].get("fig_subdir", "figs")
    fig_dir         = os.path.join(primary_run_dir, fig_sub)
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # Handle mirrors
    mirrors: List[str] = []
    if outputs.get("mirroring", {}).get("enabled", True):
        targets = outputs["mirroring"].get("targets", ["colab_drive", "local"])
        for tgt in targets:
            if tgt == "local":
                mdir = os.path.join(outputs["local"].get("base_dir", "/content/TQE_Output"), run_id)
            elif tgt == "colab_drive" and outputs["colab_drive"].get("enabled", False):
                mdir = os.path.join(outputs["colab_drive"]["base_dir"], run_id)
            elif tgt == "cloud" and outputs["cloud"].get("enabled") and outputs["cloud"].get("bucket_url"):
                continue
            else:
                continue

            if os.path.abspath(mdir) == os.path.abspath(primary_run_dir):
                continue

            pathlib.Path(os.path.join(mdir, fig_sub)).mkdir(parents=True, exist_ok=True)
            mirrors.append(mdir)

    # Cloud bucket path
    cloud_bucket = None
    if outputs["cloud"].get("enabled") and outputs["cloud"].get("bucket_url"):
        cloud_bucket = outputs["cloud"]["bucket_url"].rstrip("/") + "/" + run_id

    return {
        "env": env,
        "run_id": run_id,
        "primary_run_dir": primary_run_dir,
        "fig_dir": fig_dir,
        "mirrors": mirrors,
        "cloud_bucket": cloud_bucket,
    }

# -----------------------------------------------------------------------------------
# Drive helper
# -----------------------------------------------------------------------------------

def ensure_colab_drive_mounted(active_cfg: dict):
    """If running in Colab and Drive is enabled, attempt to mount it."""
    if _resolve_environment(active_cfg) != "colab":
        return
    if not active_cfg["OUTPUTS"]["colab_drive"].get("enabled", False):
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False)
    except Exception as e:
        print("[WARN] Could not mount Google Drive:", e)
