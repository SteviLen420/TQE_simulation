# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# structure.py - Galaxy Structure Analysis Module
# ==========================================================================================
# TQE–ΛSim: Galaxy structure analysis and cosmic web classification
# ==========================================================================================

import numpy as np
from .config import MASTER_CTRL

class GalaxyStructureAnalyzer:
    """
    Galaxy Structure Analysis - Cosmic Web Topology Detector
    
    Detects large-scale structure formations:
    - Voids: Underdense regions (δ < -0.8)
    - Filaments: Elongated structures connecting clusters
    - Clusters: Overdense knots (δ > 2.0)
    - Walls/Sheets: 2D flat structures
    - Cosmic Web: Full topology classification
    """
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.density_field = None
        self.grid_size = MASTER_CTRL.get('GALAXY_GRID_SIZE', 128)
        self.box_size = MASTER_CTRL.get('GALAXY_BOX_SIZE', 500.0)
        
        print("✓ Galaxy Structure Analyzer initialized")
        print(f"  Grid: {self.grid_size}³ cells, Box: {self.box_size:.0f} Mpc/h")
    
    def generate_density_field(self):
        # Generate 3D density field from P(k)
        print("🌌 Generating 3D density field...")
        
        from scipy.ndimage import gaussian_filter
        
        # Create Gaussian random field
        delta = np.random.randn(self.grid_size, self.grid_size, self.grid_size)
        
        # Smooth to match P(k) shape
        if 'observables' in self.simulation.results and 'lss' in self.simulation.results['observables']:
            k_arr = np.array(self.simulation.results['observables']['lss']['k'])
            P_k = np.array(self.simulation.results['observables']['lss']['P_k'])
            k_smooth = k_arr[np.argmax(P_k)]
            sigma_pix = (2.0 / k_smooth) * self.grid_size / self.box_size
        else:
            # Use default from MASTER_CTRL
            sigma_pix = MASTER_CTRL.get('GALAXY_SMOOTH_SIGMA', 5.0)
        
        delta_smooth = gaussian_filter(delta, sigma=sigma_pix)
        delta_smooth = (delta_smooth - np.mean(delta_smooth)) / np.std(delta_smooth)
        
        self.density_field = delta_smooth
        print(f"  ✓ Field generated: δ ∈ [{np.min(delta_smooth):.2f}, {np.max(delta_smooth):.2f}]")
    
    def classify_cosmic_web(self):
        # Classify voids/filaments/sheets/knots using MASTER_CTRL thresholds
        print("🕸️ Classifying cosmic web...")
        
        if self.density_field is None:
            self.generate_density_field()
        
        from scipy.ndimage import laplace
        
        laplacian = laplace(self.density_field)
        
        # Get thresholds from MASTER_CTRL (configurable)
        void_thresh = MASTER_CTRL.get('GALAXY_VOID_THRESHOLD', -0.8)
        sheet_min = MASTER_CTRL.get('GALAXY_SHEET_MIN', 0.5)
        sheet_max = MASTER_CTRL.get('GALAXY_SHEET_MAX', 3.0)
        knot_thresh = MASTER_CTRL.get('GALAXY_KNOT_THRESHOLD', 2.0)
        
        # Classification using configurable thresholds
        void_mask = (self.density_field < void_thresh) & (laplacian > 0)
        sheet_mask = (self.density_field > sheet_min) & (self.density_field < sheet_max) & (laplacian < 0)
        knot_mask = (self.density_field > knot_thresh)
        filament_mask = ~(void_mask | sheet_mask | knot_mask)
        
        total = self.grid_size**3
        void_frac = np.sum(void_mask) / total
        filament_frac = np.sum(filament_mask) / total
        sheet_frac = np.sum(sheet_mask) / total
        knot_frac = np.sum(knot_mask) / total
        
        print(f"  ✓ Voids: {void_frac*100:.1f}%, Filaments: {filament_frac*100:.1f}%, Sheets: {sheet_frac*100:.1f}%, Knots: {knot_frac*100:.1f}%")
        
        return {
            'void_fraction': void_frac,
            'filament_fraction': filament_frac,
            'sheet_fraction': sheet_frac,
            'knot_fraction': knot_frac
        }
    
    def find_voids(self):
        # Find void regions using MASTER_CTRL threshold
        print("🕳️ Finding voids...")
        
        if self.density_field is None:
            self.generate_density_field()
        
        from scipy.ndimage import label, find_objects
        
        # Get thresholds from MASTER_CTRL
        void_threshold = MASTER_CTRL.get('GALAXY_VOID_THRESHOLD', -0.8)
        void_min_radius = MASTER_CTRL.get('GALAXY_VOID_MIN_RADIUS', 5.0)
        void_max_radius = MASTER_CTRL.get('GALAXY_VOID_MAX_RADIUS', 100.0)
        
        void_regions = self.density_field < void_threshold
        labeled_voids, n_voids = label(void_regions)
        
        print(f"  DEBUG: Labeled regions before filter: {n_voids}")
        
        void_catalogue = []
        n_too_small = 0
        n_too_large = 0
        
        for void_id, sl in enumerate(find_objects(labeled_voids)):
            if sl is None:
                continue
            vol = np.sum(labeled_voids[sl] == (void_id + 1))
            r_cells = (3.0 * vol / (4.0 * np.pi))**(1.0/3.0)
            r_mpc = r_cells * (self.box_size / self.grid_size)
            
            # SIZE FILTER: Remove too small or too large voids
            if r_mpc < void_min_radius:
                n_too_small += 1
                continue
            if r_mpc > void_max_radius:
                n_too_large += 1
                continue
            
            void_catalogue.append({
                'void_id': len(void_catalogue) + 1,  # Re-indexed after filtering
                'radius_mpc': r_mpc,
                'volume_mpc3': vol * (self.box_size / self.grid_size)**3
            })
        
        print(f"  DEBUG: Filtered out - too small: {n_too_small}, too large: {n_too_large}")
        print(f"  Found {len(void_catalogue)} voids (filtered by {void_min_radius:.1f}-{void_max_radius:.1f} Mpc/h)")
        return void_catalogue
    
    def find_clusters(self):
        # Find cluster regions using MASTER_CTRL threshold
        print("🌟 Finding clusters...")
        
        if self.density_field is None:
            self.generate_density_field()
        
        from scipy.ndimage import label, find_objects
        
        # Get thresholds from MASTER_CTRL
        knot_threshold = MASTER_CTRL.get('GALAXY_KNOT_THRESHOLD', 2.0)
        cluster_min_radius = MASTER_CTRL.get('GALAXY_CLUSTER_MIN_RADIUS', 1.0)
        cluster_max_radius = MASTER_CTRL.get('GALAXY_CLUSTER_MAX_RADIUS', 30.0)
        
        cluster_regions = self.density_field > knot_threshold
        labeled_clusters, n_clusters = label(cluster_regions)
        
        print(f"  DEBUG: Labeled regions before filter: {n_clusters}")
        
        cluster_catalogue = []
        n_too_small = 0
        n_too_large = 0
        
        for cluster_id, sl in enumerate(find_objects(labeled_clusters)):
            if sl is None:
                continue
            vol = np.sum(labeled_clusters[sl] == (cluster_id + 1))
            r_cells = (3.0 * vol / (4.0 * np.pi))**(1.0/3.0)
            r_mpc = r_cells * (self.box_size / self.grid_size)
            
            # SIZE FILTER: Remove too small or too large clusters
            if r_mpc < cluster_min_radius:
                n_too_small += 1
                continue
            if r_mpc > cluster_max_radius:
                n_too_large += 1
                continue
            
            cluster_catalogue.append({
                'cluster_id': len(cluster_catalogue) + 1,  # Re-indexed after filtering
                'radius_mpc': r_mpc,
                'mass_proxy': vol * np.mean(self.density_field[sl])
            })
        
        print(f"  DEBUG: Filtered out - too small: {n_too_small}, too large: {n_too_large}")
        print(f"  Found {len(cluster_catalogue)} clusters (filtered by {cluster_min_radius:.1f}-{cluster_max_radius:.1f} Mpc/h)")
        return cluster_catalogue
    
    def find_filaments(self):
        # Find filament structures using MASTER_CTRL thresholds
        print("🧵 Finding filaments...")
        
        if self.density_field is None:
            self.generate_density_field()
        
        from scipy.ndimage import label, find_objects
        
        # Get thresholds from MASTER_CTRL
        fil_min = MASTER_CTRL.get('GALAXY_FILAMENT_MIN', -0.5)
        fil_max = MASTER_CTRL.get('GALAXY_FILAMENT_MAX', 2.0)
        aspect_min = MASTER_CTRL.get('GALAXY_FILAMENT_ASPECT_MIN', 3.0)
        
        filament_regions = (self.density_field > fil_min) & (self.density_field < fil_max)
        labeled_filaments, n_filaments = label(filament_regions)
        
        filament_catalogue = []
        for fil_id, sl in enumerate(find_objects(labeled_filaments)):
            if sl is None:
                continue
            ex = [sl[i].stop - sl[i].start for i in range(3)]
            ex_sorted = sorted(ex)
            aspect = ex_sorted[2] / max(ex_sorted[0], 1.0)
            
            # Use configurable aspect ratio threshold
            if aspect > aspect_min:
                filament_catalogue.append({
                    'filament_id': fil_id + 1,
                    'length_mpc': ex_sorted[2] * (self.box_size / self.grid_size),
                    'aspect_ratio': aspect
                })
        
        return filament_catalogue
    
    def find_walls(self):
        # Find wall/sheet structures using MASTER_CTRL thresholds
        print("🧱 Finding walls...")
        
        if self.density_field is None:
            self.generate_density_field()
        
        from scipy.ndimage import label, find_objects
        
        # Get thresholds from MASTER_CTRL
        sheet_min = MASTER_CTRL.get('GALAXY_SHEET_MIN', 0.5)
        sheet_max = MASTER_CTRL.get('GALAXY_SHEET_MAX', 3.0)
        flatness_max = MASTER_CTRL.get('GALAXY_WALL_FLATNESS_MAX', 0.3)
        min_size = MASTER_CTRL.get('GALAXY_WALL_MIN_SIZE', 5)
        
        sheet_regions = (self.density_field > sheet_min) & (self.density_field < sheet_max)
        labeled_sheets, n_sheets = label(sheet_regions)
        
        wall_catalogue = []
        for sheet_id, sl in enumerate(find_objects(labeled_sheets)):
            if sl is None:
                continue
            ex = [sl[i].stop - sl[i].start for i in range(3)]
            ex_sorted = sorted(ex)
            flatness = ex_sorted[0] / max(ex_sorted[2], 1.0)
            
            # Use configurable flatness and size thresholds
            if flatness < flatness_max and ex_sorted[2] > min_size:
                wall_catalogue.append({
                    'wall_id': sheet_id + 1,
                    'area_mpc2': ex_sorted[1] * ex_sorted[2] * (self.box_size / self.grid_size)**2,
                    'flatness': flatness
                })
        
        return wall_catalogue
    
    def compute_all_metrics(self):
        # Compute all galaxy structure metrics
        print("\n🌌 GALAXY STRUCTURE ANALYSIS...")
        
        self.generate_density_field()
        cosmic_web = self.classify_cosmic_web()
        voids = self.find_voids()
        clusters = self.find_clusters()
        filaments = self.find_filaments()
        walls = self.find_walls()
        
        summary = {
            'cosmic_web_fractions': cosmic_web,
            'n_voids': len(voids),
            'n_clusters': len(clusters),
            'n_filaments': len(filaments),
            'n_walls': len(walls),
            'mean_void_radius_mpc': np.mean([v['radius_mpc'] for v in voids]) if voids else 0.0,
            'mean_cluster_radius_mpc': np.mean([c['radius_mpc'] for c in clusters]) if clusters else 0.0,
            'total_filament_length_mpc': np.sum([f['length_mpc'] for f in filaments]) if filaments else 0.0,
            'total_wall_area_mpc2': np.sum([w['area_mpc2'] for w in walls]) if walls else 0.0
        }
        
        print("✅ Galaxy structure complete!")
        print(f"  Voids: {len(voids)}, Clusters: {len(clusters)}, Filaments: {len(filaments)}, Walls: {len(walls)}")
        
        return {
            'summary': summary,
            'voids': voids,
            'clusters': clusters,
            'filaments': filaments,
            'walls': walls
        }

# ==========================================================================================
# TQE DARK ENERGY COUPLING SIMULATION CLASS
# ==========================================================================================

