# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# utils.py - Utility Functions Module
# ==========================================================================================
# TQE–ΛSim: Utility functions for package installation, memory management, 
# Google Drive setup, deterministic seeding, and production hardening
# ==========================================================================================

import subprocess
import sys

def install_package(package_name):
    # Install package using pip
    try:
        if 'google.colab' in sys.modules:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--quiet'])
        else:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--quiet'])
        return True
    except Exception as e:
        print(f"⚠ Failed to install {package_name}: {e}")
        return False

def check_and_install_all_packages():
    # List of all required packages for cosmological analysis
    required_packages = [
        'numpy', 'matplotlib', 'scipy', 'tqdm', 'pandas',
        'camb',  # For CMB power spectrum calculations
        'emcee',  # For MCMC Bayesian inference
        'dynesty',  # For nested sampling Bayes Factor
        'corner',  # For corner plots
        'h5py',  # For data storage
        'scikit-learn',  # For cross-validation
        'astropy'  # For cosmological calculations
    ]
    
    print("📦 Checking and installing required packages...")
    
    for package in required_packages:
        try:
            # Try to import the package
            if package == 'numpy':
                import numpy
            elif package == 'matplotlib':
                import matplotlib
            elif package == 'scipy':
                import scipy
            elif package == 'tqdm':
                import tqdm
            elif package == 'pandas':
                import pandas
            elif package == 'camb':
                import camb
            elif package == 'emcee':
                import emcee
            elif package == 'dynesty':
                import dynesty
            elif package == 'corner':
                import corner
            elif package == 'h5py':
                import h5py
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'astropy':
                import astropy
            
            
        except ImportError:
            print(f"📥 Installing {package}...")
            if install_package(package):
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ {package} installation failed")
    
    print("✅ Package installation completed!")

# ==========================================================================================
# IMPORTS
# ==========================================================================================

import numpy as np
import gc
from functools import lru_cache

# Set matplotlib backend BEFORE importing pyplot (critical for Colab PNG generation)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Configure matplotlib for proper PNG generation (prevent white/empty images in Colab)
plt.ioff()  # Turn off interactive mode (critical for Colab)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['figure.max_open_warning'] = 0  # Suppress max figure warning
import scipy.integrate as integrate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize, differential_evolution
import json
import os
from datetime import datetime
import warnings
from tqdm import tqdm
import zipfile
import random
import hashlib
import pandas as pd

# Filter only non-critical warnings (keep RuntimeWarning and UserWarning visible)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

# Cosmological imports
try:
    import camb
    from camb import model, initialpower
    CAMB_AVAILABLE = True
    print("✅ CAMB cosmological code available")
except ImportError:
    CAMB_AVAILABLE = False
    print("⚠ CAMB not available - attempting installation...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'camb', '--quiet'])
        import camb
        from camb import model, initialpower
        CAMB_AVAILABLE = True
        print("✅ CAMB successfully installed")
    except Exception as e:
        CAMB_AVAILABLE = False
        print(f"⚠ CAMB installation failed: {e}")

# Bayesian inference imports
try:
    import emcee
    import corner
    MCMC_AVAILABLE = True
    print("✅ MCMC tools (emcee, corner) available")
except ImportError:
    MCMC_AVAILABLE = False
    print("⚠ MCMC tools not available")

# Astropy for cosmological calculations
try:
    from astropy.cosmology import FlatLambdaCDM
    from astropy import units as u
    from astropy import constants as const
    ASTROPY_AVAILABLE = True
    print("✅ Astropy cosmology tools available")
except ImportError:
    ASTROPY_AVAILABLE = False
    print("⚠ Astropy not available")

# Google Drive integration for Colab environment
try:
    from google.colab import drive
    from google.colab import files
    COLAB = True
    print("✅ Google Colab environment detected")
    if os.path.exists('/content/drive/MyDrive'):
        print("✅ Google Drive already mounted - no setup needed")
    else:
        print("📁 Google Drive not mounted - will be set up during pipeline execution")
        print("💡 One-time authorization will be required when pipeline starts")
except ImportError:
    COLAB = False
    print("❌ Local environment detected - Google Colab required")

# ==========================================================================================
# PERFORMANCE: MEMORY OPTIMIZATION
# ==========================================================================================

def cleanup_memory(MASTER_CTRL=None):
    """Clean up memory between computations."""
    from .config import MASTER_CTRL as _MASTER_CTRL
    if MASTER_CTRL is None:
        MASTER_CTRL = _MASTER_CTRL
    if MASTER_CTRL.get("MEMORY_EFFICIENT_MODE", True):
        gc.collect()
        plt.close('all')

def apply_performance_mode(mode=None, MASTER_CTRL=None):
    """Apply performance mode scaling to grid resolutions."""
    from .config import MASTER_CTRL as _MASTER_CTRL
    if MASTER_CTRL is None:
        MASTER_CTRL = _MASTER_CTRL
    if mode is None:
        mode = MASTER_CTRL.get("PERFORMANCE_MODE", "balanced")
    
    if mode == "fast":
        # Fast mode: Reduced resolution for quick testing
        MASTER_CTRL["A_GRID_N_LOG"] = 1024  # Reduced for speed
        MASTER_CTRL["Z_POINTS"] = 50         # Reduced for speed
        MASTER_CTRL["NESTED_NLIVE"] = 100    # Reduced for speed
        MASTER_CTRL["GALAXY_GRID_SIZE"] = 128  # Reduced for speed
        print("⚡ Fast mode: Reduced resolution for quick testing")
    elif mode == "accurate":
        # Accurate mode: High resolution for production
        MASTER_CTRL["A_GRID_N_LOG"] = 8192  # Increased for accuracy
        MASTER_CTRL["Z_POINTS"] = 200        # Increased for accuracy
        MASTER_CTRL["NESTED_NLIVE"] = 500    # Increased for accuracy
        MASTER_CTRL["GALAXY_GRID_SIZE"] = 512  # Increased for accuracy
        print("🎯 Accurate mode: High resolution for production")
    else:
        # Balanced mode: OPTIMIZED (already set in MASTER_CTRL)
        # A_GRID_N_LOG=2048, Z_POINTS=100, NESTED_NLIVE=250, GALAXY_GRID_SIZE=256
        print("⚖️ Balanced mode: OPTIMIZED resolution (2× faster than v4.2.0)")

# ==========================================================================================
# GOOGLE DRIVE SETUP FUNCTIONS
# ==========================================================================================

def setup_google_drive_automatically():
    # Setup Google Drive - ask for authorization only once at the beginning
    if not COLAB:
        print("⚠ Not in Colab environment - skipping Google Drive setup")
        return True
    
    print("🚀 Setting up Google Drive...")
    
    try:
        # Check if already mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive already mounted - no authorization needed")
            return True
        
        # First time setup - ask for authorization
        print("🔐 FIRST TIME GOOGLE DRIVE SETUP")
        print("="*50)
        print("📋 You need to authorize Google Drive access:")
        print("   1. A popup will appear with an authorization link")
        print("   2. Click the link and sign in to your Google account")
        print("   3. Copy the authorization code")
        print("   4. Paste it in the input field below")
        print("   5. Press Enter")
        print("   6. This is a ONE-TIME setup - won't ask again!")
        print("="*50)
        
        # Mount with user authorization (one-time only)
        print("📁 Mounting Google Drive (one-time authorization required)...")
        drive.mount('/content/drive')
        
        # Verify mount was successful
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive mounted successfully!")
            print("🎉 ONE-TIME AUTHORIZATION COMPLETED!")
            print("💡 Google Drive will stay mounted for this entire Colab session")
            print("💡 No more authorization needed - pipeline can run multiple times")
            return True
        else:
            print("❌ Google Drive mount verification failed")
            return False
        
    except Exception as e:
        print(f"❌ Google Drive setup failed: {e}")
        print("💡 Please try running the cell again")
        return False

def check_google_drive_status():
    # Check Google Drive mount status and provide clear feedback
    if not COLAB:
        return False, "Local environment - Google Colab required"
    
    if os.path.exists('/content/drive/MyDrive'):
        return True, "Google Drive already mounted - ready to use"
    else:
        return False, "Google Drive not mounted - authorization required"

# ==========================================================================================
# DETERMINISTIC SEEDING
# ==========================================================================================

def set_deterministic_seed(seed_string="TQE_DarkEnergy_2025"):
    # Set deterministic seed for reproducible results
    seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest(), 16) % (2**32)
    
    # Set all random seeds
    np.random.seed(seed_hash)
    random.seed(seed_hash)
    return seed_hash

def save_reproducibility_snapshot(run_dir, MASTER_CTRL=None):
    # Save complete reproducibility snapshot with environment info
    from .config import MASTER_CTRL as _MASTER_CTRL
    if MASTER_CTRL is None:
        MASTER_CTRL = _MASTER_CTRL
    
    snapshot_file = os.path.join(run_dir, "Reproducibility_Environment_Snapshot.json")
    
    print(f"\n📸 Saving reproducibility snapshot...")
    
    # Get package versions
    try:
        packages = {
            'numpy': np.__version__,
            'matplotlib': plt.matplotlib.__version__,
            'scipy': 'installed',
            'pandas': pd.__version__,
            'tqdm': 'installed',
            'astropy': 'installed' if ASTROPY_AVAILABLE else 'N/A',
            'camb': 'installed' if CAMB_AVAILABLE else 'N/A',
            'emcee': 'installed' if MCMC_AVAILABLE else 'N/A',
            'h5py': 'installed',
            'scikit-learn': 'installed'
        }
    except Exception as e:
        packages = {'error': f'Could not extract package versions: {e}'}
    
    snapshot = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'master_control_panel': MASTER_CTRL,
        'package_versions': packages,
        'environment': {
            'colab': COLAB,
            'google_drive_mounted': os.path.exists('/content/drive/MyDrive') if COLAB else False,
            'camb_available': CAMB_AVAILABLE,
            'mcmc_available': MCMC_AVAILABLE,
            'astropy_available': ASTROPY_AVAILABLE
        },
        'system_info': {
            'platform': sys.platform,
            'cpu_count': os.cpu_count()
        }
    }
    
    with open(snapshot_file, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)
    
    return snapshot_file

# ==========================================================================================
# PRODUCTION HARDENING UTILITY FUNCTIONS
# ==========================================================================================

def w_eff_CPL(a, MASTER_CTRL=None):
    """
    Effective equation of state from CPL parameterization.
    
    w(a) = w₀ + w_a(1 - a)
    
    Args:
        a: scale factor
        MASTER_CTRL: configuration dictionary (optional)
    
    Returns:
        w_eff: effective dark energy equation of state
        None if CPL fallback is disabled
    """
    from .config import MASTER_CTRL as _MASTER_CTRL
    if MASTER_CTRL is None:
        MASTER_CTRL = _MASTER_CTRL
    if not MASTER_CTRL.get("USE_CPL_FALLBACK", False):
        return None
    
    w0 = MASTER_CTRL.get("CPL_W0", -1.0)
    wa = MASTER_CTRL.get("CPL_WA", 0.0)
    
    return w0 + wa * (1.0 - a)

def _top_hat_W(x):
    """
    Top-hat window function in Fourier space for σ₈ calculation.
    
    W(x) = 3(sin(x) - x·cos(x))/x³
    
    Handles x→0 limit with Taylor expansion.
    
    Args:
        x: k·R (dimensionless)
    
    Returns:
        W: window function value
    """
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    
    # Small x: Taylor expansion W(x) ≈ 1 - x²/10
    small = np.abs(x) < 1e-6
    xs = x[~small]
    
    # Normal x
    out[~small] = 3.0 * (np.sin(xs) - xs * np.cos(xs)) / np.where(xs == 0, 1.0, xs**3)
    
    # Taylor expansion for small x
    out[small] = 1.0 - (x[small]**2) / 10.0
    
    return out

# ==========================================================================================
