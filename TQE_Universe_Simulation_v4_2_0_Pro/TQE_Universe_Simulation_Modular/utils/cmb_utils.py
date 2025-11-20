# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# CMB utility functions
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..core.pipeline_context import PipelineContext

try:
    import healpy as hp  # type: ignore
    HEALPY_AVAILABLE = True
except ImportError:
    hp = None  # type: ignore
    HEALPY_AVAILABLE = False

# CMB cache functions
# ======== PERFORMANCE: CMB CACHE ========
# LRU cache for expensive CMB map generation (5-10x speedup for repeated calls)
_cmb_cache = {}
_cmb_cache_enabled = True
_cmb_cache_maxsize = 1000

def _cache_key(E, I, nside, seed):
    """Generate cache key from CMB generation parameters."""
    return (round(float(E), 6), round(float(I), 6), int(nside), int(seed))

def get_cached_cmb_or_generate(E, I, nside, seed, generator_func):
    """Cache wrapper for CMB generation."""
    global _cmb_cache, _cmb_cache_enabled, _cmb_cache_maxsize
    
    if not _cmb_cache_enabled:
        return generator_func(E, I, nside, seed)
    
    key = _cache_key(E, I, nside, seed)
    
    if key in _cmb_cache:
        return _cmb_cache[key].copy()  # Return copy to avoid mutation
    
    # Generate and cache
    cmb_map = generator_func(E, I, nside, seed)
    
    # Limit cache size (LRU-like: remove oldest if full)
    if len(_cmb_cache) >= _cmb_cache_maxsize:
        # Remove first (oldest) entry
        _cmb_cache.pop(next(iter(_cmb_cache)))
    
    _cmb_cache[key] = cmb_map.copy()
    return cmb_map

# TODO: Add other CMB utility functions
# - _axis_from_lmap
# - detect_cold_spots_healpix
# - detect_axis_of_evil
# - generate_coldspot_overlay
# - generate_aoe_overlay

def _axis_from_lmap(alm_full, nside, ell_pick, lmax_used):
    """
    Keep only the requested multipole ell_pick from alm_full and build a map.
    Then return the longitude/latitude (deg) of the max |T| pixel and its value.
    """
    fl = np.zeros(lmax_used + 1, dtype=float)
    if ell_pick >= len(fl):
        print(f"Warning: ell_pick={ell_pick} is out of bounds for lmax_used={lmax_used}. Returning default axis.")
        return (0.0, 0.0, 0.0)
    fl[ell_pick] = 1.0
    alm_l  = hp.almxfl(alm_full, fl)
    m_l    = hp.alm2map(alm_l, nside=nside)
    ip     = int(np.argmax(np.abs(m_l)))
    th, ph = hp.pix2ang(nside, ip)
    return (float(np.degrees(ph) % 360.0),       # lon (deg)
            float(90.0 - np.degrees(th)),       # lat (deg)
            float(m_l[ip]))                      # peak value


def detect_cold_spots_healpix(cmb_map, uid, E_val, I_val, lock_ep, maps_dir, category_name, config):
    """Multi-scale cold spot detection on HEALPix maps."""
    if not HEALPY_AVAILABLE:
        return pd.DataFrame()  # Return empty DataFrame if healpy not available
    
    nside = hp.get_nside(cmb_map)
    npix = hp.nside2npix(nside)
    
    sigma_scales = config.get("CMB_COLD_SIGMA_ARCMIN", [30, 60, 90, 120, 180, 240, 360])
    min_sep_arcmin = config.get("CMB_COLD_MIN_SEP_ARCMIN", 30)
    z_thresh = config.get("CMB_COLD_Z_THRESH", -1.5)
    topk = config.get("CMB_COLD_TOPK", 5)
    
    all_spots = []
    
    for sigma_arcmin in sigma_scales:
        sigma_rad = np.deg2rad(sigma_arcmin / 60.0)
        
        try:
            smoothed = hp.smoothing(cmb_map, fwhm=sigma_rad)
        except Exception as e:
            if config.get("VERBOSE", False): print(f"[COLD][WARN] Smoothing failed at {sigma_arcmin}': {e}")
            continue
        
        mean_T = np.mean(smoothed)
        std_T = np.std(smoothed)
        if std_T < 1e-12: continue
        
        z_map = (smoothed - mean_T) / std_T
        
        for ipix in range(npix):
            z_val = z_map[ipix]
            if z_val > z_thresh: continue
            
            neighbors = hp.get_all_neighbours(nside, ipix)
            neighbors = neighbors[neighbors != -1]
            
            if len(neighbors) > 0 and z_val < np.min(z_map[neighbors]):
                theta, phi = hp.pix2ang(nside, ipix)
                lon = np.degrees(phi) % 360.0
                lat = 90.0 - np.degrees(theta)
                
                all_spots.append({
                    'universe_id': uid, 'E': E_val, 'I': I_val, 'lock_epoch': lock_ep,
                    'scale_arcmin': sigma_arcmin, 'lon': lon, 'lat': lat,
                    'z_score': z_val, 'temp_uK': smoothed[ipix], 'category': category_name
                })
    
    if not all_spots: return pd.DataFrame()
    df_spots = pd.DataFrame(all_spots)
    
    def filter_by_separation(spots_df, min_sep_deg):
        if len(spots_df) == 0: return spots_df
        spots_sorted = spots_df.sort_values('z_score').reset_index(drop=True)
        keep_mask = np.ones(len(spots_sorted), dtype=bool)
        for i in range(len(spots_sorted)):
            if not keep_mask[i]: continue
            lon1, lat1 = spots_sorted.loc[i, ['lon', 'lat']]
            for j in range(i+1, len(spots_sorted)):
                if not keep_mask[j]: continue
                lon2, lat2 = spots_sorted.loc[j, ['lon', 'lat']]
                dlon = np.deg2rad(lon2 - lon1)
                lat1_r, lat2_r = np.deg2rad(lat1), np.deg2rad(lat2)
                sep_rad = np.arccos(np.sin(lat1_r) * np.sin(lat2_r) + np.cos(lat1_r) * np.cos(lat2_r) * np.cos(dlon))
                sep_deg = np.degrees(sep_rad)
                if sep_deg * 60 < min_sep_deg: keep_mask[j] = False
        return spots_sorted[keep_mask]
    
    df_filtered = filter_by_separation(df_spots, min_sep_arcmin)
    df_topk = df_filtered.nsmallest(topk, 'z_score')
    
    ref_z = config.get("CMB_COLD_REF_Z", -70.0)
    uk_thresh = config.get("CMB_COLD_UK_THRESH", -70.0)
    df_topk['cold_flag'] = (df_topk['z_score'] <= ref_z / std_T) | (df_topk['temp_uK'] <= uk_thresh)
    
    return df_topk


def detect_axis_of_evil(cmb_map, uid, E_val, I_val, lock_ep, maps_dir, category_name, config, master_seed: int):
    """Axis-of-Evil alignment detection with Monte Carlo significance test."""
    nside = hp.get_nside(cmb_map)
    lmax = config.get("CMB_AOE_LMAX", 5)
    n_realiz = config.get("CMB_AOE_NREALIZ", 100) # Reduced N_realiz for speed
    
    try:
        alm = hp.map2alm(cmb_map, lmax=lmax, iter=3)
    except Exception as e:
        if config.get("VERBOSE", False): print(f"[AOE][WARN] map2alm failed for UID {uid}: {e}")
        return pd.DataFrame()
    
    def extract_axis(alm_in, ell):
        lmax_in = hp.Alm.getlmax(len(alm_in))
        if ell > lmax_in: return None, None, 0.0
        fl = np.zeros(lmax_in + 1)
        fl[ell] = 1.0
        alm_ell = hp.almxfl(alm_in, fl)
        try:
            map_ell = hp.alm2map(alm_ell, nside=nside)
        except Exception:
            return None, None, 0.0
        ipix_max = np.argmax(np.abs(map_ell))
        theta, phi = hp.pix2ang(nside, ipix_max)
        lon = np.degrees(phi) % 360.0
        lat = 90.0 - np.degrees(theta)
        peak_val = map_ell[ipix_max]
        return lon, lat, peak_val
    
    axes_data = []
    for ell in range(2, lmax + 1):
        lon, lat, peak = extract_axis(alm, ell)
        if lon is not None:
            axes_data.append({
                'universe_id': uid, 'E': E_val, 'I': I_val, 'lock_epoch': lock_ep,
                'ell': ell, 'axis_lon': lon, 'axis_lat': lat, 'peak_value': peak,
                'category': category_name
            })
    
    if len(axes_data) < 2: return pd.DataFrame()
    df_axes = pd.DataFrame(axes_data)
    
    q_lon = df_axes.loc[df_axes['ell'] == 2, 'axis_lon'].values[0]
    q_lat = df_axes.loc[df_axes['ell'] == 2, 'axis_lat'].values[0]
    
    alignment_angle = np.nan
    if 3 in df_axes['ell'].values:
        o_lon = df_axes.loc[df_axes['ell'] == 3, 'axis_lon'].values[0]
        o_lat = df_axes.loc[df_axes['ell'] == 3, 'axis_lat'].values[0]
        dlon = np.deg2rad(o_lon - q_lon)
        q_lat_r, o_lat_r = np.deg2rad(q_lat), np.deg2rad(o_lat)
        alignment_angle = np.degrees(np.arccos(
            np.sin(q_lat_r) * np.sin(o_lat_r) + np.cos(q_lat_r) * np.cos(o_lat_r) * np.cos(dlon)
        ))
    
    def random_alignment():
        alm_rand = alm.copy()
        # Use dedicated RNG for random alignment to maintain determinism
        aoe_rng = np.random.default_rng(master_seed + uid + 999)
        phases = np.exp(2j * np.pi * aoe_rng.random(len(alm_rand)))
        alm_rand *= phases
        
        map_rand = hp.alm2map(alm_rand, nside=nside)
        alm_rand_new = hp.map2alm(map_rand, lmax=lmax, iter=0)
        
        q_lon_r, q_lat_r, _ = extract_axis(alm_rand_new, 2)
        o_lon_r, o_lat_r, _ = extract_axis(alm_rand_new, 3)
        
        if q_lon_r is None or o_lon_r is None: return np.nan
        dlon = np.deg2rad(o_lon_r - q_lon_r)
        q_r, o_r = np.deg2rad(q_lat_r), np.deg2rad(o_lat_r)
        
        return np.degrees(np.arccos(
            np.sin(q_r) * np.sin(o_r) + np.cos(q_r) * np.cos(o_r) * np.cos(dlon)
        ))
    
    p_value = np.nan
    if not np.isnan(alignment_angle):
        # Use dedicated RNG for Monte Carlo realizations to maintain determinism
        mc_rng = np.random.default_rng(master_seed + uid + 888)
        random_angles = [random_alignment() for _ in range(n_realiz)]
        valid_angles = np.array([a for a in random_angles if not np.isnan(a)])
        if len(valid_angles) > 0:
            p_value = np.mean(valid_angles <= alignment_angle)

    df_axes['alignment_angle_deg'] = alignment_angle
    df_axes['p_value'] = p_value
    
    ref_angle = config.get("AOE_REF_ANGLE_DEG", 20.0)
    p_thresh = config.get("AOE_P_THRESHOLD", 0.10)
    df_axes['aoe_flag'] = (alignment_angle <= ref_angle) & (p_value <= p_thresh) if not np.isnan(p_value) else False
    
    return df_axes


def generate_coldspot_overlay(cmb_map, spots_df, uid, maps_dir, ctx: PipelineContext):
    """Generates and saves a single cold spot overlay PNG."""
    try:
        overlay_path = os.path.join(maps_dir, f"cmb_uid{uid:05d}_coldspot_overlay_EI_Pipeline_v4.2.0_Pro.png")
        
        # PUBLICATION: Larger title font and better marker visibility
        hp.mollview(cmb_map, title=f"Cold Spots - Universe {uid}", cmap='RdBu_r', unit='µK', 
                   hold=False)
        
        # PUBLICATION: Larger, more visible markers (was: s=200)
        for idx, spot in spots_df.iterrows():
            theta = np.deg2rad(90 - spot['lat'])
            phi = np.deg2rad(spot['lon'])
            hp.projscatter(theta, phi, marker='X', s=400, c='lime', linewidths=4, edgecolors='black', zorder=10)
            
            # Add spot number annotation (optional, only for top 3 coldest)
            if idx < 3:
                # Text annotation near the marker (offset to avoid overlap)
                hp.projtext(theta + 0.1, phi, f"#{idx+1}", color='yellow', 
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', edgecolor='lime', alpha=0.7))
        
        # Save directly (bypass save_fig to avoid axes check with healpy)
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        plt.savefig(overlay_path, dpi=ctx.config.get('PLOT_SAVE_DPI', 180), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[COLD][PLOT-ERR] Overlay failed for UID {uid}: {e}")


def generate_aoe_overlay(cmb_map, axes_df, uid, maps_dir, ctx: PipelineContext):
    """Generates and saves a single Axis-of-Evil overlay PNG."""
    try:
        overlay_path = os.path.join(maps_dir, f"cmb_uid{uid:05d}_aoe_overlay_EI_Pipeline_v4.2.0_Pro.png")
        
        # PUBLICATION: Larger fonts and better styling
        hp.mollview(cmb_map, title=f"Axis of Evil - Universe {uid}", cmap='RdBu_r', unit='µK', 
                   hold=False)
        
        # PUBLICATION: Larger, more visible markers (was: s=300)
        for _, axis in axes_df.iterrows():
            theta = np.deg2rad(90 - axis['axis_lat'])
            phi = np.deg2rad(axis['axis_lon'])
            color = {2: 'cyan', 3: 'magenta', 4: 'yellow', 5: 'orange'}.get(int(axis['ell']), 'white')
            hp.projscatter(theta, phi, marker='*', s=400, c=color, 
                          edgecolors='black', linewidths=3, zorder=10,
                          label=f"ℓ={int(axis['ell'])}")
        
        plt.legend(fontsize=14, framealpha=0.95, loc='lower left')
        
        # Save directly (bypass save_fig to avoid axes check with healpy)
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        plt.savefig(overlay_path, dpi=ctx.config.get('PLOT_SAVE_DPI', 180), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[AOE][PLOT-ERR] Overlay failed for UID {uid}: {e}")


