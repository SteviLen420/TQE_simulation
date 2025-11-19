# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# setup.py - Package Installation and Reproducibility Setup
# ==========================================================================================

import sys
import subprocess
import os
import time
import hashlib
import numpy as np

def _ensure(pkg):
    """Ensure a package is installed before importing."""
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def check_and_install_packages():
    """Check and install required packages."""
    print("[SETUP] Checking and installing dependencies...")
    essential_packages = ["numpy", "matplotlib", "scipy", "tqdm"]
    quantum_packages = ["qutip"]

    for pkg in essential_packages:
        print(f"[SETUP] Checking {pkg}...")
        _ensure(pkg)
        print(f"[SETUP] ✓ {pkg} ready")

    # Try quantum packages
    for pkg in quantum_packages:
        try:
            print(f"[SETUP] Checking {pkg}... (this may take 30-60 sec on first install)")
            _ensure(pkg)
            print(f"[SETUP] ✓ {pkg} ready")
        except Exception as e:
            print(f"[SETUP] Warning: Could not install {pkg}: {e}")

    print("[SETUP] ✓ All dependencies ready")

def setup_reproducibility(seed=None, config=None):
    """
    Setup reproducibility with deterministic seeding.
    
    Parameters
    ----------
    seed : int or None
        Master random seed (None = auto-generate)
    config : dict or None
        MASTER_CTRL config dict (optional, for updating SEED)
    
    Returns
    -------
    int
        The seed value used
    numpy.random.Generator
        RNG instance
    """
    if seed is None:
        # Generate truly random seed from multiple entropy sources
        entropy_sources = [
            str(time.time()),
            str(os.urandom(16)),
            str(os.getpid()),
        ]
        entropy_string = "".join(entropy_sources)
        hash_digest = hashlib.sha256(entropy_string.encode()).hexdigest()
        seed = int(hash_digest[:8], 16) % (2**31)  # Use first 8 hex chars as seed
        print(f"[SEED] Generated random master seed: {seed}")
    else:
        print(f"[SEED] Using specified master seed: {seed}")
    
    # Update config if provided
    if config is not None:
        config["SEED"] = seed
    
    # Create RNG instances
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # Also seed legacy numpy RNG for compatibility
    
    # Set environment variables for strict reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"[SEED] ✓ RNG initialized (reproducible)")
    
    return seed, rng

