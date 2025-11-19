# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# data_loader.py - Data Loading Module
# ==========================================================================================
# TQE–ΛSim: Data loading functions for observational data (Pantheon+, BOSS BAO, Planck CMB)
# ==========================================================================================

import numpy as np
import pandas as pd
import os
from .config import MASTER_CTRL

def load_pantheon_plus_data(filepath=None, cov_filepath=None):
    # Load Pantheon+ SNe Ia data with full covariance matrix
    # Public data: https://github.com/PantheonPlusSH0ES/DataRelease
    # Full sample: 1,701 SNe Ia from Pantheon+ (2022)
    
    # TIER 1 UPGRADE: Try to load REAL Pantheon+ data first
    if filepath is not None and os.path.exists(filepath):
        try:
            print("📊 Loading REAL Pantheon+ SNe Ia data...")
            
            # Load main data file (should contain: zHD, MU_SH0ES, MU_ERR, etc.)
            if filepath.endswith('.txt'):
                data = np.loadtxt(filepath)
                z_sne = data[:, 0]
                mu_obs = data[:, 1]
                sigma_mu = data[:, 2] if data.shape[1] > 2 else np.ones_like(z_sne) * 0.15
            elif filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
                z_sne = data['zHD'].values if 'zHD' in data.columns else data['z'].values
                mu_obs = data['MU_SH0ES'].values if 'MU_SH0ES' in data.columns else data['mu'].values
                sigma_mu = data['MU_ERR'].values if 'MU_ERR' in data.columns else np.ones_like(z_sne) * 0.15
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Load covariance matrix if available
            cov_matrix = None
            if cov_filepath is not None and os.path.exists(cov_filepath):
                print("📊 Loading Pantheon+ covariance matrix...")
                if cov_filepath.endswith('.txt'):
                    cov_matrix = np.loadtxt(cov_filepath)
                elif cov_filepath.endswith('.npy'):
                    cov_matrix = np.load(cov_filepath)
                else:
                    print(f"⚠ Unsupported covariance format, using diagonal")
                    cov_matrix = np.diag(sigma_mu**2)
            else:
                # Use diagonal covariance (uncorrelated errors)
                cov_matrix = np.diag(sigma_mu**2)
            
            print(f"✅ Pantheon+ data loaded: {len(z_sne)} SNe, z ∈ [{z_sne.min():.3f}, {z_sne.max():.3f}]")
            
            return z_sne, mu_obs, sigma_mu, cov_matrix
            
        except Exception as e:
            print(f"⚠ Failed to load Pantheon+ data: {e}")
            print("  Falling back to enhanced mock data")
    
    # PRODUCTION HARDENING: Check if mock data is allowed
    if filepath is None or not os.path.exists(filepath):
        if not MASTER_CTRL.get('ALLOW_MOCK_DATA', False):
            raise FileNotFoundError(
                "❌ PRODUCTION MODE: Pantheon+ data file required!\n"
                f"   Requested path: {filepath}\n"
                "   Please provide real Pantheon+ SNe Ia data.\n"
                "   Set MASTER_CTRL['ALLOW_MOCK_DATA'] = True to use mock data (testing only)."
            )
        
        print("⚠ Pantheon+ data not found, using ENHANCED mock data (50 SNe)")
        print("⚠ WARNING: Mock data for TESTING ONLY - not for publication!")
        
        # Extended redshift range: z = 0.01 → 2.3 (Pantheon+ range)
        z_sne = np.concatenate([
            np.linspace(0.01, 0.1, 5),   # Low-z
            np.linspace(0.1, 0.5, 15),   # Medium-z
            np.linspace(0.5, 1.0, 15),   # High-z
            np.linspace(1.0, 2.3, 15)    # Very high-z
        ])
        
        # Mock μ(z) from approximate ΛCDM
        # μ(z) ≈ 5·log10(D_L) + 25, D_L ≈ c·z·(1 + 0.5·z) for low-z
        c_light = 299792.458  # km/s
        H0_fid = 70.0
        mu_obs = 5.0 * np.log10(c_light * z_sne * (1.0 + 0.5*z_sne) / H0_fid) + 25.0
        
        # Add realistic scatter
        np.random.seed(42)
        mu_obs += np.random.normal(0, 0.15, size=len(z_sne))
        
        # Realistic uncertainties (increase with z)
        sigma_mu = 0.10 + 0.15 * (z_sne / 2.3)  # 0.1 → 0.25 mag
        
        # Diagonal covariance (uncorrelated)
        cov_matrix = np.diag(sigma_mu**2)
        
        return z_sne, mu_obs, sigma_mu, cov_matrix
    
    try:
        # Try to load real Pantheon+ data
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
            z_sne = data['zHD'].values if 'zHD' in data.columns else data['z'].values
            mu_obs = data['MU_SH0ES'].values if 'MU_SH0ES' in data.columns else data['mu'].values
            sigma_mu = data['MU_ERR'].values if 'MU_ERR' in data.columns else data['sigma_mu'].values
        elif filepath.endswith('.txt'):
            data = np.loadtxt(filepath)
            z_sne = data[:, 0]
            mu_obs = data[:, 1]
            sigma_mu = data[:, 2]
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print(f"✓ Pantheon+ data loaded: {len(z_sne)} SNe Ia")
        return z_sne, mu_obs, sigma_mu
    
    except Exception as e:
        print(f"⚠ Failed to load Pantheon+ data: {e}")
        print("  Using mock data instead")
        # Fallback to mock
        z_sne = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
        mu_obs = np.array([33.2, 38.5, 40.8, 41.5, 43.5, 44.8, 45.8, 46.5, 47.0, 47.5])
        sigma_mu = np.array([0.15, 0.15, 0.18, 0.18, 0.20, 0.25, 0.28, 0.30, 0.35, 0.40])
        return z_sne, mu_obs, sigma_mu


def load_boss_bao_data(filepath=None, cov_filepath=None):
    # Load BOSS DR12 / eBOSS / DESI BAO data with covariance
    # Public data: 
    # - BOSS DR12: https://data.sdss.org/sas/dr12/boss/lss/
    # - eBOSS DR16: https://www.sdss.org/dr16/
    # - DESI: https://data.desi.lbl.gov/public/
    
    # TIER 1 UPGRADE: Try to load REAL BAO data first
    if filepath is not None and os.path.exists(filepath):
        try:
            print("📊 Loading REAL BOSS/eBOSS/DESI BAO data...")
            
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
                z_bao = data['z'].values
                
                # D_V (spherically averaged distance) or D_M/r_s
                if 'D_V' in data.columns:
                    DV_obs = data['D_V'].values
                    sigma_DV = data['sigma_D_V'].values if 'sigma_D_V' in data.columns else DV_obs * 0.02
                    DM_over_rd_obs = DV_obs / 147.78  # Approximate conversion (r_s ~ 147.78 Mpc)
                    sigma_DM = sigma_DV / 147.78
                else:
                    DM_over_rd_obs = data['DM_over_rd'].values if 'DM_over_rd' in data.columns else data['DM_rs'].values
                    sigma_DM = data['sigma_DM'].values if 'sigma_DM' in data.columns else DM_over_rd_obs * 0.02
                
                # H(z) measurements
                H_obs = data['H'].values if 'H' in data.columns else np.full(len(z_bao), np.nan)
                sigma_H = data['sigma_H'].values if 'sigma_H' in data.columns else np.full(len(z_bao), np.nan)
                
            elif filepath.endswith('.txt'):
                data = np.loadtxt(filepath)
                z_bao = data[:, 0]
                DM_over_rd_obs = data[:, 1]
                sigma_DM = data[:, 2] if data.shape[1] > 2 else DM_over_rd_obs * 0.02
                H_obs = data[:, 3] if data.shape[1] > 3 else np.full(len(z_bao), np.nan)
                sigma_H = data[:, 4] if data.shape[1] > 4 else np.full(len(z_bao), np.nan)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Load covariance matrix if available
            cov_matrix = None
            if cov_filepath is not None and os.path.exists(cov_filepath):
                print("📊 Loading BAO covariance matrix...")
                cov_matrix = np.loadtxt(cov_filepath) if cov_filepath.endswith('.txt') else np.load(cov_filepath)
            else:
                # Diagonal covariance
                n_meas = len(z_bao)
                cov_matrix = np.diag(np.concatenate([sigma_DM**2, np.where(~np.isnan(sigma_H), sigma_H**2, np.zeros_like(sigma_H))]))
            
            print(f"✅ BAO data loaded: {len(z_bao)} measurements, z ∈ [{z_bao.min():.3f}, {z_bao.max():.3f}]")
            
            return z_bao, DM_over_rd_obs, sigma_DM, H_obs, sigma_H, cov_matrix
            
        except Exception as e:
            print(f"⚠ Failed to load BAO data: {e}")
            print("  Falling back to enhanced mock data")
    
    # ENHANCED MOCK DATA (10 points from BOSS DR12 + eBOSS)
    if filepath is None or not os.path.exists(filepath):
        print("⚠ BOSS BAO data not found, using ENHANCED mock data (10 measurements)")
        
        # Real BOSS DR12 + eBOSS redshifts
        z_bao = np.array([0.15, 0.38, 0.51, 0.61, 0.70, 0.85, 1.00, 1.48, 1.52, 2.33])
        
        # D_M/r_d measurements (BOSS DR12 + eBOSS-like values)
        DM_over_rd_obs = np.array([4.47, 10.27, 13.36, 15.23, 17.01, 18.92, 20.83, 27.79, 28.23, 37.77])
        sigma_DM = np.array([0.17, 0.15, 0.20, 0.24, 0.30, 0.45, 0.50, 0.65, 0.70, 1.20])
        
        # H(z) measurements (km/s/Mpc) - from cosmic chronometers + BAO
        H_obs = np.array([np.nan, 81.5, 90.4, 97.3, 103.0, 113.0, 125.0, 168.0, 172.0, 224.0])
        sigma_H = np.array([np.nan, 1.9, 1.9, 2.1, 2.3, 4.5, 6.0, 17.0, 18.0, 8.0])
        
        # Diagonal covariance
        n_meas = len(z_bao)
        cov_DM = np.diag(sigma_DM**2)
        cov_H = np.diag(np.where(~np.isnan(sigma_H), sigma_H**2, 0.0))
        cov_matrix = np.block([[cov_DM, np.zeros((n_meas, n_meas))],
                               [np.zeros((n_meas, n_meas)), cov_H]])
        
        return z_bao, DM_over_rd_obs, sigma_DM, H_obs, sigma_H, cov_matrix
    
    try:
        # Try to load real BOSS data
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
            z_bao = data['z'].values
            DM_over_rd_obs = data['DM_over_rd'].values
            sigma_DM = data['sigma_DM'].values
            H_obs = data.get('H', np.full(len(z_bao), None)).values
            sigma_H = data.get('sigma_H', np.full(len(z_bao), None)).values
        elif filepath.endswith('.txt'):
            data = np.loadtxt(filepath)
            z_bao = data[:, 0]
            DM_over_rd_obs = data[:, 1]
            sigma_DM = data[:, 2]
            H_obs = data[:, 3] if data.shape[1] > 3 else np.full(len(z_bao), None)
            sigma_H = data[:, 4] if data.shape[1] > 4 else np.full(len(z_bao), None)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        print(f"✓ BOSS BAO data loaded: {len(z_bao)} measurements")
        return z_bao, DM_over_rd_obs, sigma_DM, H_obs, sigma_H
    
    except Exception as e:
        print(f"⚠ Failed to load BOSS data: {e}")
        
        # PRODUCTION HARDENING: Check if mock data is allowed
        if not MASTER_CTRL.get('ALLOW_MOCK_DATA', False):
            raise FileNotFoundError(
                f"❌ PRODUCTION MODE: BOSS BAO data file required!\n"
                f"   Error: {e}\n"
                "   Please provide real BOSS/eBOSS/DESI BAO data.\n"
                "   Set MASTER_CTRL['ALLOW_MOCK_DATA'] = True to use mock data (testing only)."
            )
        
        print("  Using mock data instead")
        print("⚠ WARNING: Mock data for TESTING ONLY - not for publication!")
        
        # Fallback to mock
        z_bao = np.array([0.15, 0.38, 0.51, 0.61, 0.70])
        DM_over_rd_obs = np.array([4.5, 10.3, 13.4, 15.3, 17.0])
        sigma_DM = np.array([0.15, 0.15, 0.20, 0.25, 0.30])
        H_obs = np.array([None, 81.5, 90.4, 97.3, 103.0])
        sigma_H = np.array([None, 1.9, 1.9, 2.1, 2.3])
        return z_bao, DM_over_rd_obs, sigma_DM, H_obs, sigma_H


def load_planck_cmb_data(filepath=None, cov_filepath=None):
    # Load Planck 2018 CMB power spectrum data (binned C_ell with covariance)
    # Public data: https://pla.esac.esa.int/
    # - Planck 2018: TT, TE, EE, low-ell, high-ell
    
    # TIER 1 UPGRADE: Try to load REAL Planck data first
    if filepath is not None and os.path.exists(filepath):
        try:
            print("📊 Loading REAL Planck 2018 CMB data...")
            
            # Planck data format: ell, D_ell (or C_ell), sigma
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
                ell = data['ell'].values
                C_ell = data['C_ell'].values if 'C_ell' in data.columns else data['D_ell'].values / (ell*(ell+1)/(2*np.pi))
                sigma_C_ell = data['sigma'].values if 'sigma' in data.columns else C_ell * 0.02
            elif filepath.endswith('.txt'):
                data = np.loadtxt(filepath)
                ell = data[:, 0]
                C_ell = data[:, 1]
                sigma_C_ell = data[:, 2] if data.shape[1] > 2 else C_ell * 0.02
            else:
                raise ValueError(f"Unsupported format: {filepath}")
            
            # Load covariance if available
            cov_matrix = None
            if cov_filepath is not None and os.path.exists(cov_filepath):
                print("📊 Loading Planck covariance matrix...")
                cov_matrix = np.loadtxt(cov_filepath) if cov_filepath.endswith('.txt') else np.load(cov_filepath)
            else:
                cov_matrix = np.diag(sigma_C_ell**2)
            
            print(f"✅ Planck CMB data loaded: {len(ell)} multipoles, ell ∈ [{int(ell.min())}, {int(ell.max())}]")
            
            return ell, C_ell, sigma_C_ell, cov_matrix
            
        except Exception as e:
            print(f"⚠ Failed to load Planck data: {e}")
            
            # PRODUCTION HARDENING: Check if mock data is allowed
            if not MASTER_CTRL.get('ALLOW_MOCK_DATA', False):
                raise FileNotFoundError(
                    f"❌ PRODUCTION MODE: Planck CMB data file required!\n"
                    f"   Error: {e}\n"
                    "   Please provide real Planck 2018 CMB data.\n"
                    "   Set MASTER_CTRL['ALLOW_MOCK_DATA'] = True to use mock data (testing only)."
                )
            
            print("  Falling back to mock CMB data")
    
    # PRODUCTION HARDENING: Check if mock data is allowed
    if (filepath is None or not os.path.exists(filepath)):
        if not MASTER_CTRL.get('ALLOW_MOCK_DATA', False):
            raise FileNotFoundError(
                "❌ PRODUCTION MODE: Planck CMB data file required!\n"
                f"   Requested path: {filepath}\n"
                "   Please provide real Planck 2018 CMB data.\n"
                "   Set MASTER_CTRL['ALLOW_MOCK_DATA'] = True to use mock data (testing only)."
            )
    
    # ENHANCED MOCK DATA (100 multipoles, binned)
    print("⚠ Planck CMB data not found, using ENHANCED mock data (100 bins)")
    print("⚠ WARNING: Mock data for TESTING ONLY - not for publication!")
    
    # Low-ell (2-30): large bins
    ell_low = np.arange(2, 31, 2)
    # High-ell (30-2500): logarithmic bins
    ell_high = np.logspace(np.log10(30), np.log10(2500), 85).astype(int)
    ell = np.concatenate([ell_low, ell_high])
    
    # Mock C_ell from approximate ΛCDM
    # C_ell(TT) ≈ A / ell^2 at large ell (simplified)
    A_cmb = 5000.0  # μK^2 normalization
    C_ell = A_cmb / (ell**2 + 100)
    
    # Add acoustic peaks (simplified)
    peak_positions = [220, 540, 810]  # First 3 acoustic peaks
    for peak_ell in peak_positions:
        C_ell += 1000.0 * np.exp(-((ell - peak_ell)**2) / (2 * 30**2))
    
    # Realistic uncertainties (increase with ell)
    sigma_C_ell = C_ell * (0.01 + 0.05 * ell / 2500)
    
    # Diagonal covariance (realistic Planck has correlations!)
    cov_matrix = np.diag(sigma_C_ell**2)
    
    return ell, C_ell, sigma_C_ell, cov_matrix


# ==========================================================================================
# CMB PLANCK MAP LOADER & VALIDATION
# ==========================================================================================

class PlanckCMBDataLoader:
    """
    Professional Planck CMB map loader with full preprocessing pipeline.
    
    Loads and processes real Planck 2018 CMB maps from Google Drive:
    - Component-separated maps (SMICA, NILC, SEVEM, Commander)
    - Raw frequency maps (100, 143, 217, 353 GHz)
    - Masks (common mask, missing pixel mask)
    - Foreground maps (NHI Neutral Hydrogen)
    
    Performs standard CMB preprocessing:
    - Mask application (set masked pixels to healpy.UNSEEN)
    - Monopole and dipole removal
    - Power spectrum computation (C_ℓ via healpy.anafast)
    
    Usage:
        # Auto-detect Planck data path (check repo structure)
        base_path = None  # Will auto-detect in repo
        loader = PlanckCMBDataLoader(base_path=base_path)
        skymap = loader.load_smica_map()
        mask = loader.load_common_mask()
        cl = loader.compute_power_spectrum(skymap, mask, lmax=2000)
    """
    
    def __init__(self, base_path=None):
        """Initialize Planck CMB data loader."""
        if base_path is None:
            # Auto-detect Planck data path (check repo structure)
            base_path = MASTER_CTRL.get("CMB_PLANCK_BASE_PATH", None)  # None = auto-detect
        
        self.base_path = base_path
        self.maps = {}          # Store loaded maps
        self.masks = {}         # Store loaded masks
        self.raw_maps = {}      # Store raw frequency maps
        self.nhi_map = None     # NHI foreground map
        
        # Check if healpy is available
        try:
            import healpy as hp
            self.hp = hp
            self.healpy_available = True
        except ImportError:
            print("⚠ healpy not available - CMB map processing disabled")
            self.healpy_available = False
    
    def load_component_separated_map(self, method='smica'):
        """
        Load component-separated CMB map (cleaned, foreground-subtracted).
        
        Parameters:
            method (str): 'smica' (default), 'nilc', 'sevem', or 'commander'
        
        Returns:
            skymap (array): HEALPix map (Temperature in μK)
            nside (int): HEALPix resolution parameter
            npix (int): Number of pixels
        """
        if not self.healpy_available:
            return None, None, None
        
        # Map method to filename
        method_files = {
            'smica': 'COM_CMB_IQU-smica_2048_R3.00_full.fits',
            'nilc': 'COM_CMB_IQU-nilc_2048_R3.00_full.fits',
            'sevem': 'COM_CMB_IQU-sevem_2048_R3.00_full.fits',
            'commander': 'COM_CMB_IQU-commander_2048_R3.00_full.fits'
        }
        
        if method not in method_files:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(method_files.keys())}")
        
        filename = method_files[method]
        filepath = f"{self.base_path}/CMB_Maps/{filename}"
        
        try:
            # Load temperature map (field=0 for IQU maps)
            skymap = self.hp.read_map(filepath, field=0, verbose=False)
            nside = self.hp.get_nside(skymap)
            npix = self.hp.nside2npix(nside)
            
            self.maps[method] = skymap
            print(f"  ✓ Loaded {method.upper()} map: Nside={nside}, Npix={npix:,}")
            
            return skymap, nside, npix
        
        except Exception as e:
            print(f"  ✗ Failed to load {method.upper()} map: {e}")
            return None, None, None
    
    def load_smica_map(self):
        """Load SMICA map (primary Planck map)."""
        return self.load_component_separated_map('smica')
    
    def load_raw_frequency_map(self, frequency):
        """
        Load raw HFI frequency map.
        
        Parameters:
            frequency (int): Frequency in GHz (100, 143, 217, 353)
        
        Returns:
            skymap (array): HEALPix map (Temperature in μK_CMB)
        """
        if not self.healpy_available:
            return None
        
        valid_freqs = [100, 143, 217, 353]
        if frequency not in valid_freqs:
            raise ValueError(f"Invalid frequency {frequency}. Choose from: {valid_freqs}")
        
        filename = f"HFI_SkyMap_{frequency}_2048_R4.00_full.fits"
        # Use standard Google Drive path
        filepath = f"{self.base_path}/CMB_Raw_Skymap/{filename}"
        
        try:
            skymap = self.hp.read_map(filepath, field=0, verbose=False)
            self.raw_maps[frequency] = skymap
            print(f"  ✓ Loaded {frequency} GHz raw map")
            return skymap
        
        except Exception as e:
            print(f"  ✗ Failed to load {frequency} GHz map: {e}")
            return None
    
    def load_common_mask(self, mask_type='Int'):
        """
        Load Planck common mask (galactic + point sources).
        
        Parameters:
            mask_type (str): 'Int' (intensity/temperature) or 'Pol' (polarization)
        
        Returns:
            mask (array): HEALPix mask (1 = good pixel, 0 = masked)
        """
        if not self.healpy_available:
            return None
        
        if mask_type not in ['Int', 'Pol']:
            raise ValueError(f"Invalid mask_type '{mask_type}'. Choose 'Int' or 'Pol'")
        
        filename = f"COM_Mask_CMB-common-Mask-{mask_type}_2048_R3.00.fits"
        filepath = f"{self.base_path}/CMB_Mask/{filename}"
        
        try:
            mask = self.hp.read_map(filepath, field=0, verbose=False)
            self.masks[f'common_{mask_type}'] = mask
            
            # Count masked pixels
            n_good = np.sum(mask > 0.5)
            n_total = len(mask)
            fsky = n_good / n_total
            
            print(f"  ✓ Loaded common mask ({mask_type}): fsky={fsky:.1%} ({n_good:,}/{n_total:,} pixels)")
            return mask
        
        except Exception as e:
            print(f"  ✗ Failed to load common mask: {e}")
            return None
    
    def load_misspix_mask(self, mask_type='Int'):
        """
        Load missing pixel mask (high-multipole data quality mask).
        
        Parameters:
            mask_type (str): 'Int' or 'Pol'
        
        Returns:
            mask (array): HEALPix mask (1 = good, 0 = bad/missing)
        """
        if not self.healpy_available:
            return None
        
        if mask_type not in ['Int', 'Pol']:
            raise ValueError(f"Invalid mask_type '{mask_type}'. Choose 'Int' or 'Pol'")
        
        filename = f"COM_Mask_CMB-HM-Misspix-Mask-{mask_type}_2048_R3.00.fits"
        filepath = f"{self.base_path}/CMB_Mask/{filename}"
        
        try:
            mask = self.hp.read_map(filepath, field=0, verbose=False)
            self.masks[f'misspix_{mask_type}'] = mask
            
            fsky = np.sum(mask > 0.5) / len(mask)
            print(f"  ✓ Loaded misspix mask ({mask_type}): fsky={fsky:.1%}")
            return mask
        
        except Exception as e:
            print(f"  ✗ Failed to load misspix mask: {e}")
            return None
    
    def load_nhi_foreground_map(self):
        """
        Load NHI (Neutral Hydrogen) foreground map from CMB_Anomaly/.
        
        Returns:
            nhi_map (array): HEALPix map (NHI column density)
        """
        if not self.healpy_available:
            return None
        
        filename = "NHI_HPX.fits"
        filepath = f"{self.base_path}/CMB_Anomaly/{filename}"
        
        try:
            nhi_map = self.hp.read_map(filepath, field=0, verbose=False)
            self.nhi_map = nhi_map
            print(f"  ✓ Loaded NHI foreground map: min={np.min(nhi_map):.2e}, max={np.max(nhi_map):.2e}")
            return nhi_map
        
        except Exception as e:
            print(f"  ✗ Failed to load NHI map: {e}")
            return None
    
    def combine_masks(self, masks):
        """
        Combine multiple masks (logical AND).
        
        Parameters:
            masks (list): List of mask arrays
        
        Returns:
            combined_mask (array): Combined mask
        """
        if not masks:
            return None
        
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = combined * mask  # Element-wise multiplication (logical AND)
        
        return combined
    
    def apply_mask(self, skymap, mask):
        """
        Apply mask to skymap (set masked pixels to healpy.UNSEEN).
        
        Parameters:
            skymap (array): HEALPix skymap
            mask (array): HEALPix mask (1 = good, 0 = masked)
        
        Returns:
            masked_skymap (array): Skymap with masked pixels set to UNSEEN
        """
        if not self.healpy_available:
            return skymap
        
        masked_skymap = skymap.copy()
        masked_skymap[mask < 0.5] = self.hp.UNSEEN
        
        n_masked = np.sum(mask < 0.5)
        print(f"  ✓ Applied mask: {n_masked:,} pixels masked")
        
        return masked_skymap
    
    def remove_monopole_dipole(self, skymap, mask=None):
        """
        Remove monopole and dipole from skymap using healpy.remove_dipole().
        
        Parameters:
            skymap (array): HEALPix skymap
            mask (array): HEALPix mask (optional)
        
        Returns:
            cleaned_skymap (array): Skymap with monopole/dipole removed
            monopole (float): Removed monopole value
            dipole (array): Removed dipole vector [x, y, z]
        """
        if not self.healpy_available:
            return skymap, 0.0, np.zeros(3)
        
        # healpy.remove_dipole returns (map, monopole, dipole)
        if mask is not None:
            # Only use unmasked pixels
            gal_mask = (mask > 0.5).astype(bool)
            cleaned_skymap, monopole, dipole = self.hp.remove_dipole(skymap, gal_cut=0, fitval=True, copy=True, bad=self.hp.UNSEEN)
        else:
            cleaned_skymap, monopole, dipole = self.hp.remove_dipole(skymap, fitval=True, copy=True)
        
        dipole_amp = np.sqrt(np.sum(dipole**2))
        print(f"  ✓ Removed monopole: {monopole:.2f} μK")
        print(f"  ✓ Removed dipole: amplitude={dipole_amp:.2f} μK")
        
        return cleaned_skymap, monopole, dipole
    
    def compute_power_spectrum(self, skymap, mask=None, lmax=2000, lmin=2):
        """
        Compute CMB power spectrum C_ℓ using healpy.anafast().
        
        Parameters:
            skymap (array): HEALPix skymap (temperature in μK)
            mask (array): HEALPix mask (optional)
            lmax (int): Maximum multipole
            lmin (int): Minimum multipole
        
        Returns:
            ell (array): Multipole moments ℓ
            cl (array): Power spectrum C_ℓ [μK²]
        """
        if not self.healpy_available:
            return None, None
        
        # Apply mask if provided
        if mask is not None:
            skymap_masked = self.apply_mask(skymap, mask)
        else:
            skymap_masked = skymap
        
        # Remove monopole and dipole
        skymap_cleaned, _, _ = self.remove_monopole_dipole(skymap_masked, mask)
        
        # Compute power spectrum
        cl = self.hp.anafast(skymap_cleaned, lmax=lmax)
        ell = np.arange(len(cl))
        
        # Trim to lmin:lmax
        mask_l = (ell >= lmin) & (ell <= lmax)
        ell = ell[mask_l]
        cl = cl[mask_l]
        
        print(f"  ✓ Computed C_ℓ: ℓ ∈ [{lmin}, {lmax}], mean={np.mean(cl):.2e} μK²")
        
        return ell, cl


class CMBPlanckValidation:
    """
    CMB Planck validation: compare TQE simulated C_ℓ vs real Planck C_ℓ.
    
    Performs:
    - Power spectrum comparison (Pearson correlation, RMS difference)
    - χ² goodness of fit test
    - Fractional residual analysis
    - Anomaly detection (cold/hot spots)
    - NHI foreground correlation
    
    Generates:
    - CMB_Planck_Raw_vs_Cleaned.png (Mollweide projection maps)
    - CMB_Power_Spectrum_Comparison.png (C_ℓ TQE vs Planck)
    - CMB_Residuals_Analysis.png (fractional residuals)
    - CMB_NHI_Correlation.png (CMB anomalies vs NHI)
    - CMB_Planck_Validation.csv (ell, Planck_Cl, TQE_Cl, residuals)
    - CMB_Planck_Statistics.json (correlation, RMS, χ², anomaly count)
    """
    
    def __init__(self, tqe_observable, planck_loader):
        """
        Initialize CMB Planck validation.
        
        Parameters:
            tqe_observable (ObservablePredictions): TQE observable predictions
            planck_loader (PlanckCMBDataLoader): Planck data loader
        """
        self.tqe_obs = tqe_observable
        self.planck = planck_loader
        
        self.planck_cl = None
        self.planck_ell = None
        self.tqe_cl = None
        self.tqe_ell = None
        
        self.statistics = {}
        self.anomalies = []
    
    def compute_planck_power_spectrum(self):
        """
        Compute Planck power spectrum from SMICA map.
        
        Pipeline:
        1. Load SMICA map
        2. Load and combine masks
        3. Remove monopole/dipole
        4. Compute C_ℓ with healpy.anafast
        
        Returns:
            ell (array): Multipole moments
            cl (array): Power spectrum C_ℓ [μK²]
        """
        print("📊 Computing Planck power spectrum from SMICA map...")
        
        # Load SMICA map
        skymap, nside, npix = self.planck.load_smica_map()
        if skymap is None:
            print("  ✗ Failed to load SMICA map")
            return None, None
        
        # Load masks if enabled
        masks = []
        if MASTER_CTRL.get("CMB_USE_COMMON_MASK", True):
            mask_type = MASTER_CTRL.get("CMB_MASK_TYPE", "Int")
            common_mask = self.planck.load_common_mask(mask_type)
            if common_mask is not None:
                masks.append(common_mask)
        
        if MASTER_CTRL.get("CMB_USE_MISSPIX_MASK", True):
            mask_type = MASTER_CTRL.get("CMB_MASK_TYPE", "Int")
            misspix_mask = self.planck.load_misspix_mask(mask_type)
            if misspix_mask is not None:
                masks.append(misspix_mask)
        
        # Combine masks
        if masks:
            combined_mask = self.planck.combine_masks(masks)
            fsky = np.sum(combined_mask > 0.5) / len(combined_mask)
            print(f"  ✓ Combined mask: fsky={fsky:.1%}")
        else:
            combined_mask = None
            print(f"  ⚠ No masks applied (full sky)")
        
        # Compute power spectrum
        lmax = MASTER_CTRL.get("CMB_LMAX", 2000)
        lmin = MASTER_CTRL.get("CMB_LMIN", 2)
        
        ell, cl = self.planck.compute_power_spectrum(skymap, combined_mask, lmax=lmax, lmin=lmin)
        
        if ell is not None:
            self.planck_ell = ell
            self.planck_cl = cl
            print(f"✅ Planck C_ℓ computed: ℓ ∈ [{lmin}, {lmax}], {len(ell)} multipoles")
        
        return ell, cl
    
    def compute_tqe_power_spectrum(self):
        """
        Get TQE simulated power spectrum from ObservablePredictions.
        
        Returns:
            ell (array): Multipole moments
            cl (array): Power spectrum C_ℓ [μK²]
        """
        print("📊 Loading TQE simulated power spectrum...")
        
        # TQE power spectrum from observable predictions
        ell, cl, _ = self.tqe_obs.cmb_power_spectrum(use_camb=False)
        
        if ell is not None:
            self.tqe_ell = ell
            self.tqe_cl = cl
            print(f"✅ TQE C_ℓ loaded: ℓ ∈ [{ell[0]}, {ell[-1]}], {len(ell)} multipoles")
        
        return ell, cl
    
    def compare_power_spectra(self):
        """
        Compare TQE vs Planck power spectra.
        
        Computes:
        - Pearson correlation coefficient
        - RMS difference
        - χ² goodness of fit
        - Fractional residuals
        
        Returns:
            statistics (dict): Comparison statistics
        """
        if self.planck_cl is None or self.tqe_cl is None:
            print("⚠ Cannot compare: Planck or TQE C_ℓ not computed")
            return {}
        
        print("📊 Comparing TQE vs Planck power spectra...")
        
        # Interpolate TQE C_ℓ to Planck ℓ grid
        from scipy.interpolate import interp1d
        tqe_cl_interp = interp1d(self.tqe_ell, self.tqe_cl, kind='cubic', fill_value='extrapolate')
        tqe_cl_resampled = tqe_cl_interp(self.planck_ell)
        
        # Pearson correlation
        from scipy.stats import pearsonr
        r, p_value = pearsonr(self.planck_cl, tqe_cl_resampled)
        
        # RMS difference
        residuals = tqe_cl_resampled - self.planck_cl
        rms = np.sqrt(np.mean(residuals**2))
        
        # Fractional residuals
        frac_residuals = residuals / self.planck_cl
        mean_frac_residual = np.mean(np.abs(frac_residuals))
        
        # χ² (assuming equal weights for simplicity)
        chi2 = np.sum((residuals / self.planck_cl)**2)
        dof = len(self.planck_cl) - 1
        chi2_reduced = chi2 / dof
        
        self.statistics = {
            'correlation_r': float(r),
            'correlation_p': float(p_value),
            'rms_difference': float(rms),
            'mean_fractional_residual': float(mean_frac_residual),
            'chi2': float(chi2),
            'dof': int(dof),
            'chi2_reduced': float(chi2_reduced),
            'n_multipoles': len(self.planck_ell)
        }
        
        print(f"  ✓ Pearson r = {r:.4f} (p={p_value:.2e})")
        print(f"  ✓ RMS difference = {rms:.2f} μK²")
        print(f"  ✓ Mean |Δ/Planck| = {mean_frac_residual:.2%}")
        print(f"  ✓ χ²/dof = {chi2_reduced:.2f}")
        
        return self.statistics
    
    def detect_anomalies(self, skymap, threshold=3.0):
        """
        Detect cold/hot spots (anomalies) in CMB map.
        
        Parameters:
            skymap (array): HEALPix CMB map (μK)
            threshold (float): Detection threshold [σ]
        
        Returns:
            anomalies (list): List of anomaly dicts (pixel, amplitude, type)
        """
        if not self.planck.healpy_available:
            return []
        
        print(f"📊 Detecting CMB anomalies (threshold={threshold}σ)...")
        
        # Compute mean and std (excluding UNSEEN pixels)
        good_pixels = skymap != self.planck.hp.UNSEEN
        mean_temp = np.mean(skymap[good_pixels])
        std_temp = np.std(skymap[good_pixels])
        
        # Detect anomalies
        z_scores = (skymap - mean_temp) / std_temp
        
        cold_spots = (z_scores < -threshold) & good_pixels
        hot_spots = (z_scores > threshold) & good_pixels
        
        n_cold = np.sum(cold_spots)
        n_hot = np.sum(hot_spots)
        
        print(f"  ✓ Detected {n_cold} cold spots (< -{threshold}σ)")
        print(f"  ✓ Detected {n_hot} hot spots (> +{threshold}σ)")
        
        # Store anomaly catalog
        anomalies = []
        for pix in np.where(cold_spots)[0]:
            anomalies.append({
                'pixel': int(pix),
                'amplitude': float(skymap[pix]),
                'z_score': float(z_scores[pix]),
                'type': 'cold'
            })
        
        for pix in np.where(hot_spots)[0]:
            anomalies.append({
                'pixel': int(pix),
                'amplitude': float(skymap[pix]),
                'z_score': float(z_scores[pix]),
                'type': 'hot'
            })
        
        self.anomalies = anomalies
        
        return anomalies
    
    def correlate_with_nhi(self, skymap):
        """
        Correlate CMB map with NHI foreground map.
        
        Parameters:
            skymap (array): HEALPix CMB map
        
        Returns:
            correlation (float): Pearson correlation coefficient
        """
        if self.planck.nhi_map is None:
            print("⚠ NHI map not loaded - skipping correlation")
            return 0.0
        
        print("📊 Correlating CMB with NHI foreground...")
        
        # Get good pixels (not UNSEEN)
        good_pixels = (skymap != self.planck.hp.UNSEEN) & (self.planck.nhi_map != self.planck.hp.UNSEEN)
        
        if np.sum(good_pixels) == 0:
            print("  ✗ No overlapping good pixels")
            return 0.0
        
        # Compute correlation
        from scipy.stats import pearsonr
        r, p_value = pearsonr(skymap[good_pixels], self.planck.nhi_map[good_pixels])
        
        print(f"  ✓ CMB-NHI correlation: r = {r:.4f} (p={p_value:.2e})")
        
        self.statistics['nhi_correlation_r'] = float(r)
        self.statistics['nhi_correlation_p'] = float(p_value)
        
        return r
    
    def generate_validation_plots(self, output_dir, prefix=""):
        """
        Generate all CMB Planck validation plots.
        
        Always creates placeholder plots even if data is missing, to ensure
        all expected files are saved.
        
        Creates 4 PNG visualizations:
        1. CMB_Planck_Power_Spectrum_Comparison.png
        2. CMB_Planck_Residuals_Analysis.png
        3. CMB_Planck_Skymap_Mollweide.png
        4. CMB_Planck_NHI_Correlation.png
        
        Parameters:
            output_dir (str): Output directory path
            prefix (str): File prefix (e.g., "Eonly_" or "EplusI_")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_dir = f"{output_dir}/PNG_Visualizations"
        os.makedirs(png_dir, exist_ok=True)
        plots_generated = False
        
        if not self.planck.healpy_available:
            print("⚠ healpy not available - will create placeholder CMB Planck plots")
        else:
            print("📊 Generating CMB Planck validation plots...")
        
        # 1. Power Spectrum Comparison
        if self.planck_cl is not None and self.tqe_cl is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Upper panel: C_ℓ comparison
            ax1.plot(self.planck_ell, self.planck_cl, 'o', markersize=3, color='#3498db', label='Planck 2018 SMICA', alpha=0.7)
            
            # Interpolate TQE to Planck grid
            from scipy.interpolate import interp1d
            tqe_cl_interp = interp1d(self.tqe_ell, self.tqe_cl, kind='cubic', fill_value='extrapolate')
            tqe_cl_resampled = tqe_cl_interp(self.planck_ell)
            
            ax1.plot(self.planck_ell, tqe_cl_resampled, '-', linewidth=2, color='#e74c3c', label='TQE Simulation', alpha=0.8)
            
            ax1.set_xlabel('Multipole moment ℓ', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax1.set_ylabel('C_ℓ [μK²]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax1.set_title('CMB Power Spectrum: TQE vs Planck 2018', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
            ax1.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='upper right')
            ax1.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            ax1.set_xlim(self.planck_ell[0], self.planck_ell[-1])
            
            # Add correlation text
            if 'correlation_r' in self.statistics:
                r = self.statistics['correlation_r']
                ax1.text(0.05, 0.95, f'Pearson r = {r:.4f}', transform=ax1.transAxes, 
                        verticalalignment='top', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'],
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Lower panel: Fractional residuals
            residuals = tqe_cl_resampled - self.planck_cl
            frac_residuals = residuals / self.planck_cl * 100  # in %
            
            ax2.plot(self.planck_ell, frac_residuals, 'o-', markersize=3, color='#9b59b6', linewidth=1.5)
            ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax2.fill_between(self.planck_ell, -5, 5, color='gray', alpha=0.2, label='±5% band')
            
            ax2.set_xlabel('Multipole moment ℓ', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax2.set_ylabel('Fractional Residual [%]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax2.set_title('Residuals: (TQE - Planck) / Planck', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
            ax2.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'], loc='upper right')
            ax2.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            ax2.set_xlim(self.planck_ell[0], self.planck_ell[-1])
            
            # Add RMS text
            if 'mean_fractional_residual' in self.statistics:
                mean_frac = self.statistics['mean_fractional_residual'] * 100
                ax2.text(0.05, 0.95, f'Mean |Δ/Planck| = {mean_frac:.2f}%', transform=ax2.transAxes, 
                        verticalalignment='top', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'],
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plot_path = f"{png_dir}/{prefix}CMB_Planck_Power_Spectrum_Comparison_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
            plt.close()
            plots_generated = True
        else:
            # Create placeholder plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'CMB Power Spectrum Comparison Not Available\n\n' +
                   'Data not available:\n' +
                   f'Planck C_ℓ: {"✓" if self.planck_cl is not None else "✗"}\n' +
                   f'TQE C_ℓ: {"✓" if self.tqe_cl is not None else "✗"}',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plot_path = f"{png_dir}/{prefix}CMB_Planck_Power_Spectrum_Comparison_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Placeholder plot saved: {plot_path}")
        
        # 2. Skymap Mollweide Projection
        try:
            skymap, _, _ = self.planck.load_smica_map()
        except:
            skymap = None
        if skymap is not None and self.planck.healpy_available:
            fig = plt.figure(figsize=(14, 7))
            
            # Use healpy mollview
            self.planck.hp.mollview(skymap, title='Planck 2018 SMICA Temperature Map', 
                                   unit='μK', cmap='RdBu_r', min=-400, max=400,
                                   fig=fig, hold=True)
            
            plot_path = f"{png_dir}/{prefix}CMB_Planck_Skymap_Mollweide_{timestamp}.png"
            plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
            plt.close()
            plots_generated = True
        else:
            # Create placeholder plot
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.text(0.5, 0.5, 'CMB Skymap Not Available\n\n' +
                   'SMICA map could not be loaded',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plot_path = f"{png_dir}/{prefix}CMB_Planck_Skymap_Mollweide_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Placeholder plot saved: {plot_path}")
        
        # 3. Anomaly Map
        if self.anomalies and len(self.anomalies) > 0:
            try:
                skymap, _, _ = self.planck.load_smica_map()
            except:
                skymap = None
            if skymap is not None and self.planck.healpy_available:
                # Create anomaly mask
                anomaly_map = np.zeros_like(skymap)
                for anom in self.anomalies:
                    pix = anom['pixel']
                    if anom['type'] == 'cold':
                        anomaly_map[pix] = -1
                    else:
                        anomaly_map[pix] = 1
                
                fig = plt.figure(figsize=(14, 7))
                self.planck.hp.mollview(anomaly_map, title=f'CMB Anomalies: Cold/Hot Spots (N={len(self.anomalies)})', 
                                       unit='Type', cmap='RdBu_r', min=-1, max=1,
                                       fig=fig, hold=True)
                
                plot_path = f"{png_dir}/{prefix}CMB_Planck_Anomaly_Map_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                plots_generated = True
        else:
            # Create placeholder plot
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.text(0.5, 0.5, 'CMB Anomaly Map Not Available\n\n' +
                   f'Anomalies detected: {len(self.anomalies) if hasattr(self, "anomalies") and self.anomalies else 0}',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plot_path = f"{png_dir}/{prefix}CMB_Planck_Anomaly_Map_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Placeholder plot saved: {plot_path}")
        
        # 4. NHI Correlation Scatter Plot
        if hasattr(self.planck, 'nhi_map') and self.planck.nhi_map is not None:
            try:
                skymap, _, _ = self.planck.load_smica_map()
            except:
                skymap = None
            if skymap is not None and self.planck.healpy_available:
                # Sample pixels for scatter plot (too many pixels for full plot)
                good_pixels = (skymap != self.planck.hp.UNSEEN) & (self.planck.nhi_map != self.planck.hp.UNSEEN)
                sample_idx = np.random.choice(np.where(good_pixels)[0], size=min(10000, np.sum(good_pixels)), replace=False)
                
                fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
                ax.scatter(self.planck.nhi_map[sample_idx], skymap[sample_idx], 
                          s=1, alpha=0.3, color='#3498db')
                
                ax.set_xlabel('NHI Column Density', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], 
                             fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('CMB Temperature [μK]', fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], 
                             fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('CMB Temperature vs NHI Foreground Correlation', 
                            fontweight=MASTER_CTRL['PLOT_FONTWEIGHT'], fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                
                # Add correlation text
                if 'nhi_correlation_r' in self.statistics:
                    r = self.statistics['nhi_correlation_r']
                    p = self.statistics['nhi_correlation_p']
                    ax.text(0.05, 0.95, f'Pearson r = {r:.4f}\np = {p:.2e}', 
                           transform=ax.transAxes, verticalalignment='top',
                           fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'],
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plot_path = f"{png_dir}/{prefix}CMB_Planck_NHI_Correlation_{timestamp}.png"
                plt.savefig(plot_path, dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                plots_generated = True
        else:
            # Create placeholder plot
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'CMB-NHI Correlation Not Available\n\n' +
                   'NHI foreground map not loaded',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plot_path = f"{png_dir}/{prefix}CMB_Planck_NHI_Correlation_{timestamp}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Placeholder plot saved: {plot_path}")
        
        if plots_generated:
            print("✅ CMB Planck validation plots generated!")
        else:
            print("✅ CMB Planck validation plots generated (placeholders)!")
    
    def save_validation_data(self, output_dir, prefix=""):
        """
        Save CMB Planck validation data (CSV + JSON).
        
        Creates:
        - CMB_Planck_Validation.csv (ell, Planck_Cl, TQE_Cl, residuals)
        - CMB_Planck_Statistics.json (correlation, RMS, χ², anomaly count)
        - CMB_Planck_Anomaly_Catalog.csv (pixel, amplitude, z_score, type)
        
        Parameters:
            output_dir (str): Output directory path
            prefix (str): File prefix (e.g., "Eonly_" or "EplusI_")
        """
        print("📊 Saving CMB Planck validation data...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save power spectrum comparison CSV
        if self.planck_cl is not None and self.tqe_cl is not None:
            try:
                from scipy.interpolate import interp1d
                tqe_cl_interp = interp1d(self.tqe_ell, self.tqe_cl, kind='cubic', fill_value='extrapolate')
                tqe_cl_resampled = tqe_cl_interp(self.planck_ell)
                
                residuals = tqe_cl_resampled - self.planck_cl
                frac_residuals = residuals / self.planck_cl
                
                df = pd.DataFrame({
                    'ell': self.planck_ell,
                    'Planck_Cl_uK2': self.planck_cl,
                    'TQE_Cl_uK2': tqe_cl_resampled,
                    'Residual_uK2': residuals,
                    'Fractional_Residual': frac_residuals
                })
                
                csv_path = f"{output_dir}/{prefix}CMB_Planck_Validation_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                print(f"  ✓ Validation CSV saved: {csv_path} ({len(df)} rows)")
            except Exception as e:
                print(f"  ⚠ Failed to save validation CSV: {e}")
                # Save empty file with status
                csv_path = f"{output_dir}/{prefix}CMB_Planck_Validation_{timestamp}.csv"
                pd.DataFrame({'status': ['data_not_available'], 'error': [str(e)]}).to_csv(csv_path, index=False)
        else:
            # Save empty file with status
            csv_path = f"{output_dir}/{prefix}CMB_Planck_Validation_{timestamp}.csv"
            status_msg = "skipped_no_data"
            if self.planck_cl is None:
                status_msg = "skipped_planck_data_unavailable"
            elif self.tqe_cl is None:
                status_msg = "skipped_tqe_data_unavailable"
            pd.DataFrame({'status': [status_msg]}).to_csv(csv_path, index=False)
            print(f"  ⚠ Validation CSV saved (empty): {csv_path} - {status_msg}")
        
        # 2. Save statistics JSON (always save, even if empty)
        json_path = f"{output_dir}/{prefix}CMB_Planck_Statistics_{timestamp}.json"
        if self.statistics:
            with open(json_path, 'w') as f:
                json.dump(self.statistics, f, indent=2)
            print(f"  ✓ Statistics JSON saved: {json_path}")
        else:
            # Save empty statistics with status
            empty_stats = {
                'status': 'skipped_no_data',
                'message': 'CMB Planck validation data not available',
                'planck_cl_available': self.planck_cl is not None,
                'tqe_cl_available': self.tqe_cl is not None
            }
            with open(json_path, 'w') as f:
                json.dump(empty_stats, f, indent=2)
            print(f"  ⚠ Statistics JSON saved (empty): {json_path}")
        
        # 3. Save anomaly catalog CSV (always save, even if empty)
        csv_anom_path = f"{output_dir}/{prefix}CMB_Planck_Anomaly_Catalog_{timestamp}.csv"
        if self.anomalies and len(self.anomalies) > 0:
            try:
                df_anom = pd.DataFrame(self.anomalies)
                df_anom.to_csv(csv_anom_path, index=False)
                print(f"  ✓ Anomaly catalog saved: {csv_anom_path} ({len(self.anomalies)} anomalies)")
            except Exception as e:
                print(f"  ⚠ Failed to save anomaly catalog: {e}")
                pd.DataFrame({'status': ['error'], 'error': [str(e)]}).to_csv(csv_anom_path, index=False)
        else:
            # Save empty catalog with status
            pd.DataFrame({'status': ['no_anomalies_detected'], 'n_anomalies': [0]}).to_csv(csv_anom_path, index=False)
            print(f"  ⚠ Anomaly catalog saved (empty): {csv_anom_path} - no anomalies detected")
        
        print("✅ CMB Planck validation data saved!")


# ==========================================================================================
# OBSERVABLE PREDICTIONS
# ==========================================================================================

