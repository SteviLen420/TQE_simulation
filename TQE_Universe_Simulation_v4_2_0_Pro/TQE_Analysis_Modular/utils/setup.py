# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Package installation utilities

import sys
import subprocess

def _ensure(pkg):
    """Ensure a package is installed before importing."""
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def check_and_install_packages():
    """Check and install essential packages."""
    print("[SETUP] Checking and installing dependencies...")
    essential_packages = ["pandas", "numpy", "matplotlib", "seaborn", "scipy", "tqdm"]
    
    for pkg in essential_packages:
        _ensure(pkg)
    
    print("[SETUP] All dependencies ready!")

