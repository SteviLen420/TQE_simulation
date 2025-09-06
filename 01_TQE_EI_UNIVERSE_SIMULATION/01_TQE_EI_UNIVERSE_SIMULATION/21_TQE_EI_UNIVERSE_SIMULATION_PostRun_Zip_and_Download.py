# ===================================================================================
# 21_TQE_EI_UNIVERSE_SIMULATION_PostRun_Zip_and_Download.py
# ===================================================================================
# Author: Stefan Len
# ===================================================================================

import os
import shutil
import datetime
import pathlib
from typing import Optional

# ---------------------------
# Configurable root folder
# ---------------------------
# Default location where the pipeline saves runs.
OUT_ROOT = os.environ.get("TQE_OUT_ROOT", "/content/TQE_Output")

# Optional: you can force a specific directory to zip via env
#   export TQE_ZIP_DIR="/content/TQE_Output/TQE_EI_UNIVERSE_SIMULATION_20250905_235959-EI-paper"
FORCED_DIR = os.environ.get("TQE_ZIP_DIR", "").strip()


def _newest_run_dir(root: str) -> Optional[str]:
    """Return the most recently modified subdirectory under `root`, or None if none."""
    if not os.path.isdir(root):
        return None
    subdirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _ensure_dir_exists(path: str) -> None:
    """Create directory if missing (no-op if exists)."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def main():
    print(f"📂 Output root: {OUT_ROOT}")

    # (2) Decide which directory to zip
    if FORCED_DIR:
        target_dir = FORCED_DIR
        print(f"🔧 Using forced directory (TQE_ZIP_DIR): {target_dir}")
    else:
        target_dir = _newest_run_dir(OUT_ROOT)
        if target_dir is None:
            # If there are no subfolders, zip the root itself as a fallback
            target_dir = OUT_ROOT
            print("ℹ️ No run subdirectory found. Will zip the root folder instead.")

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"❌ Target directory not found: {target_dir}")

    print(f"✅ Target to zip: {target_dir}")

    # (3) Create timestamped ZIP in /content
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # with seconds
    zip_base = f"/content/TQE_Output_{ts}"
    _ensure_dir_exists("/content")
    zip_path = shutil.make_archive(zip_base, "zip", target_dir)
    print(f"📦 Archive created: {zip_path}")

    # (4) Try to trigger a download in Colab
    try:
        from google.colab import files  # type: ignore
        files.download(zip_path)
        print("⬇️ Download triggered in Colab.")
    except Exception:
        print("ℹ️ Not running in Colab (or auto-download unavailable). ZIP is ready above.")

    print("✅ Done.")


if __name__ == "__main__":
    main()
