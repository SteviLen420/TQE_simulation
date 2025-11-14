# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Common imports and setup code
# Extracted from TQE_Universe_Simulation_Full_Pipeline_v4.2.0_PRO.py
#
# ======== CRITICAL: PACKAGE INSTALLATION FIRST ========
import sys
import subprocess

def _ensure(pkg):
    """Ensure a package is installed before importing."""
    try:
        __import__(pkg)
    except ImportError:
        print(f"[SETUP] Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ⚡ INSTALL PACKAGES BEFORE ANY IMPORTS ⚡
print("[SETUP] Checking and installing dependencies...")
# Optimized package list for Colab - install only what's needed
essential_packages = ["pandas", "scipy", "scikit-learn", "numpy", "matplotlib", "tqdm"]
optional_packages = ["qutip", "healpy", "camb"]
bayesian_packages = ["dynesty", "corner"]  # Bayesian model selection

for pkg in essential_packages:
    _ensure(pkg)

# Try optional packages, but don't fail if they're not available
for pkg in optional_packages:
    try:
        _ensure(pkg)
    except Exception as e:
        print(f"[SETUP] Warning: Could not install {pkg}: {e}")

# Try Bayesian packages (PRO features)
for pkg in bayesian_packages:
    try:
        _ensure(pkg)
    except Exception as e:
        print(f"[SETUP] Warning: Could not install {pkg} (Bayesian features will be disabled): {e}")

print("[SETUP] All dependencies ready!")

# ======== NOW SAFE TO IMPORT ========
import os
import time
import json
import warnings
import shutil
import multiprocessing
import gc
from functools import lru_cache
import numpy as np
import pandas as pd

# Set matplotlib backend BEFORE importing pyplot (critical for Colab PNG generation)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Configure matplotlib for proper PNG generation (prevent white/empty images in Colab)
plt.ioff()  # Turn off interactive mode (critical for Colab)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['figure.max_open_warning'] = 0  # Suppress max figure warning

from tqdm.auto import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Optional imports with fallbacks for Colab
try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    print("[WARNING] healpy not available - CMB maps will use fallback mode")
    HEALPY_AVAILABLE = False

try:
    import camb
    CAMB_AVAILABLE = True
except ImportError:
    print("[WARNING] camb not available - using simplified CMB generation")
    CAMB_AVAILABLE = False

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
except ImportError:
    print("[WARNING] qutip not available - using simplified quantum calculations")
    QUTIP_AVAILABLE = False

warnings.filterwarnings("ignore")

