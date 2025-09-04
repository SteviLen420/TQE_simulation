io_paths.py
# ===================================================================================
# Environment and Output Path Resolver for TQE Simulation
# ===================================================================================
# Handles detection of runtime environment (Colab / Desktop / Cloud),
# automatic path resolution for saving results (figures, CSV, JSON),
# and optional mirroring of outputs across multiple storage targets.
#
# Author: Stefan Len
# ===================================================================================

import os, platform, time, pathlib
from typing import Dict, List

def _is_colab(active_cfg: dict) -> bool:
    """Detect Google Colab by presence of known environment variables."""
    markers = active_cfg["ENV"].get("colab_markers", [])
    return any(k in os.environ for k in markers)

def _desktop_dir(active_cfg: dict) -> str:
    """Resolve Desktop folder (env override -> OS default -> CWD fallback)."""
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
    """Build a run_id using prefix + time format, then append -EI/-E and -profile if requested."""
    prefix = meta_cfg.get("RUN_ID_PREFIX", "")
    fmt    = meta_cfg.get("RUN_ID_FORMAT", "%Y%m%d_%H%M%S")
    rid    = prefix + time.strftime(fmt)

    # EI/E tag a run_id végére (ha kérve)
    if meta_cfg.get("append_ei_to_run_id", False):
        ei_tag = "EI" if active_cfg["PIPELINE"].get("use_information", True) else "E"
        rid += f"-{ei_tag}"

    # profil tag a run_id végére (ha kérve)
    if active_cfg["OUTPUTS"].get("tag_profile_in_runid", False):
        prof = os.environ.get("TQE_PROFILE", None)
        if prof:
            rid += f"-{prof}"

    return rid

def _resolve_environment(active_cfg: dict) -> str:
    """
    Decide primary environment string: 'colab' | 'cloud' | 'desktop'.
    Priority:
      1) ENV.force_environment if set
      2) auto-detect Colab
      3) if cloud.enabled+bucket_url -> 'cloud'
      4) else 'desktop'
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

def resolve_output_paths(active_cfg: dict) -> Dict[str, str]:
    """
    Apply routing rule:
      - if env == 'colab'   => PRIMARY = Colab Drive
      - if env == 'cloud'   => PRIMARY = Desktop
      - if env == 'desktop' => PRIMARY = Desktop

    Also prepares optional mirror targets (local project folder and/or Colab Drive),
    and returns a cloud bucket path (no directories are created for cloud).
    Keys returned: { env, run_id, primary_run_dir, fig_dir, mirrors, cloud_bucket }
    """
    env     = _resolve_environment(active_cfg)
    outputs = active_cfg["OUTPUTS"]
    meta    = active_cfg["META"]
    run_id  = _run_id(meta, active_cfg)

    # Safety: never use Colab Drive when not in Colab
    if env != "colab":
        outputs["colab_drive"]["enabled"] = False

    # Ensure cloud is marked enabled when a bucket is present
    if outputs["cloud"].get("bucket_url") and not outputs["cloud"].get("enabled"):
        outputs["cloud"]["enabled"] = True

    # Primary base dir by env
    if env == "colab":
        primary_base = outputs["colab_drive"]["base_dir"]
    else:
        primary_base = os.path.join(
            _desktop_dir(active_cfg),
            outputs["local"].get("desktop_subdir", "TQE_Output")
        )

    # Create primary run dir and figs subdir
    primary_run_dir = os.path.join(primary_base, run_id)
    fig_sub         = outputs["local"].get("fig_subdir", "figs")
    fig_dir         = os.path.join(primary_run_dir, fig_sub)
    pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)

    # Mirrors
    mirrors: List[str] = []
    if outputs.get("mirroring", {}).get("enabled", True):
        targets = outputs["mirroring"].get("targets", ["local", "colab_drive"])
        for tgt in targets:
            if tgt == "local":
                local_base = outputs["local"].get("base_dir", "./")
                mdir = os.path.join(local_base, run_id)
            elif tgt == "colab_drive" and outputs["colab_drive"].get("enabled", False):
                mdir = os.path.join(outputs["colab_drive"]["base_dir"], run_id)
            elif tgt == "cloud" and outputs["cloud"].get("enabled") and outputs["cloud"].get("bucket_url"):
                # Cloud sync handled by uploader (no mkdir)
                continue
            else:
                continue

            if os.path.abspath(mdir) == os.path.abspath(primary_run_dir):
                continue

            pathlib.Path(os.path.join(mdir, fig_sub)).mkdir(parents=True, exist_ok=True)
            mirrors.append(mdir)

    # Cloud path (for uploader)
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

def ensure_colab_drive_mounted(active_cfg: dict):
    """If in Colab and Drive is enabled, try to mount."""
    if _resolve_environment(active_cfg) != "colab":
        return
    if not active_cfg["OUTPUTS"]["colab_drive"].get("enabled", False):
        return
    try:
        from google.colab import drive  # type: ignore
        drive.mount('/content/drive', force_remount=False)
    except Exception as e:
        print("[WARN] Could not mount Google Drive:", e)
