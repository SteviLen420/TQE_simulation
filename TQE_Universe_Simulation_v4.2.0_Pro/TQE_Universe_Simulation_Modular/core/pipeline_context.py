# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# PipelineContext class
#
import os
import json
import glob
import urllib.request
import time
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_PLANCK_WARNING_EMITTED = False

class PipelineContext:
    """
    Encapsulates all transient state and global configurations for a single pipeline run.
    Eliminates global variables like OUTPUT_ROOT, SAVE_DIR, and run_id.
    """
    def __init__(self, config: dict, run_id_override: str = None):
        """
        Initializes the context, sets up RNG, and creates file paths.
        """
        self.config = config.copy()
        
        # --- Reproducibility: Seed management ---
        self.master_seed = self.config.get("SEED")
        if self.master_seed is None:
            self.master_seed = int(np.random.randint(1, 2**32))
        self.config["SEED"] = self.master_seed

        self._pending_planck_source = None

        # Create both modern (rng) and legacy (np.random) RNG streams
        self.rng = np.random.default_rng(self.master_seed)
        np.random.seed(self.master_seed)  # sync legacy RNG for QuTiP calls

        # --- Run ID and Paths ---
        # Generate run_id in format: TQE_Universe_Simulation_Full_Pipeline_EI_YYYYMMDD_HHMMSS or _E_only_
        # Use PIPELINE_VARIANT for consistency (not COUPLING_MODE)
        pipeline_variant = self.config.get('PIPELINE_VARIANT', 'full')  # 'full' (E+I) or 'energy_only'
        if pipeline_variant == 'energy_only':
            mode_suffix = "E_only"
        else:
            mode_suffix = "EI"
        
        if run_id_override:
            self.run_id = run_id_override
        else:
            timestamp = time.strftime(self.config.get("RUN_ID_FORMAT", "%Y%m%d_%H%M%S"))
            self.run_id = f"TQE_Universe_Simulation_Full_Pipeline_{mode_suffix}_{timestamp}"
        
        self.paths = self._initialize_paths()
        self._resolve_planck_path()
        if self._pending_planck_source:
            self._store_planck_dataset(self._pending_planck_source)
            self._pending_planck_source = None
        
        # --- Runtime Data Registries ---
        self.map_registry = []  # CMB map tracking
        self.universe_category_map = {}  # UID -> category (for organizing outputs)
        self.variant = self.config.get("PIPELINE_VARIANT", "full")

    def _initialize_paths(self) -> dict:
        """Determines and creates the directory structure with simple categorization."""
        # Modular pipeline saves to Desktop
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        modular_results_dir = os.path.join(desktop_path, "TQE_Universe_Simulation_Modular_Results")
        
        # Check if DRIVE_BASE_DIR is explicitly set (takes precedence)
        if self.config.get("DRIVE_BASE_DIR"):
            # Use the explicitly set directory (for single run modes)
            save_dir = self.config["DRIVE_BASE_DIR"]
            output_root = os.path.dirname(save_dir)
        # Check if we're in multi-I parameter analysis mode
        elif self.config.get("MULTI_I_ANALYSIS_MODE", False):
            # Use the master save directory from multi-I analysis + run_id as subdirectory
            master_save_dir = self.config.get("MULTI_I_SAVE_DIR", modular_results_dir)
            save_dir = os.path.join(master_save_dir, self.run_id)
            output_root = os.path.dirname(master_save_dir)  # Parent of master_save_dir
        else:
            # Structure: Desktop/TQE_Universe_Simulation_Modular_Results/TQE_Universe_Simulation_{mode}_{timestamp}/
            output_root = modular_results_dir
            save_dir = os.path.join(output_root, self.run_id)

        # Simple directory structure - only PNG_Visualizations folder
        goldilocks_results_dir = os.path.join(save_dir, "Goldilocks_Results")
        png_visualizations_dir = os.path.join(save_dir, "PNG_Visualizations")
        aggregate_dir = os.path.join(save_dir, "Aggregate")
        categorized_results_dir = os.path.join(save_dir, "Categorized_Results")
        
        # All other files (CSV, JSON, TXT, ZIP) go to root directory
        # No complex categorization - everything in save_dir root
        
        # Simple paths structure - only PNG_Visualizations folder + Categorized_Results
        paths = {
            "REPO_ROOT": repo_root,
            "OUTPUT_ROOT": output_root,
            "SAVE_DIR": save_dir,
            "GOLDILOCKS_DIR": goldilocks_results_dir,
            "PNG_VISUALIZATIONS_DIR": png_visualizations_dir,
            "AGGREGATE_DIR": aggregate_dir,
            "CATEGORIZED_DIR": categorized_results_dir,
            "AGGREGATE_FIG_DIR": png_visualizations_dir,
            "PLANCK_DATA_DIR": os.path.join(save_dir, "planck_data"),
            
            # All PNG directories point to PNG_Visualizations
            "ANOMALY_PNG_DIR": png_visualizations_dir,
            "PHYSICS_PNG_DIR": png_visualizations_dir,
            "MAIN_PNG_DIR": png_visualizations_dir,
            "LAWS_PNG_DIR": png_visualizations_dir,
            "STATS_PNG_DIR": png_visualizations_dir,
            "CMB_PNG_DIR": png_visualizations_dir,
            "VIZ_PNG_DIR": png_visualizations_dir,
            
            # All CSV directories point to save_dir root
            "ANOMALY_CSV_DIR": aggregate_dir,
            "PHYSICS_CSV_DIR": aggregate_dir,
            "MAIN_CSV_DIR": aggregate_dir,
            "LAWS_CSV_DIR": aggregate_dir,
            "STATS_CSV_DIR": aggregate_dir,
            "CMB_CSV_DIR": aggregate_dir,
            "VIZ_CSV_DIR": aggregate_dir,
        }

        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        paths["PLANCK_DATA_RUN_PATH"] = None

        return paths

    def with_variant(self, path: str) -> str:
        """Add variant tag to filename: file.png -> file_E+I.png (single mode only)"""
        # In batch modes, directory structure already separates runs - no tags needed
        if self.config.get("MULTI_I_ANALYSIS_MODE", False):
            return path
        
        root, ext = os.path.splitext(path)
        if self.variant == "energy_only":
            tag = "E_only_Pipeline_v4.2.0_Pro"
        elif self.variant == "full":
            tag = "EI_Pipeline_v4.2.0_Pro"
        else:
            tag = self.variant
        return f"{root}_{tag}{ext}"

    def resolve_variant_path(self, path: str):
        """
        Locate an artifact saved through ctx.save_* that may include a variant tag.
        Returns the first existing path or None.
        """
        if not path:
            return None

        full_path = path if os.path.isabs(path) else self.get_full_path(path)
        candidates = []

        try:
            variant_path = self.with_variant(full_path)
        except Exception:
            variant_path = full_path

        if variant_path and variant_path not in candidates:
            candidates.append(variant_path)
        if full_path not in candidates:
            candidates.append(full_path)

        dirname, basename = os.path.dirname(full_path), os.path.basename(full_path)
        base_root, base_ext = os.path.splitext(basename)

        for candidate in candidates:
            try:
                if candidate and os.path.exists(candidate) and (os.path.isdir(candidate) or os.path.getsize(candidate) > 0):
                    return candidate
            except OSError:
                continue

        if dirname and base_root:
            pattern = os.path.join(dirname, f"{base_root}_*{base_ext}")
            for candidate in sorted(glob.glob(pattern)):
                try:
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                        return candidate
                except OSError:
                    continue

        return None

    def _resolve_planck_path(self) -> None:
        global _PLANCK_WARNING_EMITTED
        planck_path = self.config.get("PLANCK_DATA_PATH")
        if not planck_path:
            return

        candidates = []
        download_target = None
        if os.path.isabs(planck_path):
            candidates.append(planck_path)
        else:
            base_dir = self.paths.get("REPO_ROOT", os.getcwd())
            download_target = os.path.join(base_dir, planck_path)
            candidates.append(download_target)
            candidates.append(os.path.join(base_dir, os.path.basename(planck_path)))
            planck_dir = os.path.join(base_dir, os.path.dirname(planck_path))
            if planck_dir and not os.path.exists(planck_dir):
                try:
                    os.makedirs(planck_dir, exist_ok=True)
                except OSError:
                    pass

        for candidate in candidates:
            if os.path.exists(candidate):
                self.config["PLANCK_DATA_PATH"] = candidate
                self._store_planck_dataset(candidate)
                return

        auto_download = self.config.get("PLANCK_AUTO_DOWNLOAD", True)
        auto_generate = self.config.get("PLANCK_GENERATE_IF_MISSING", True)
        if download_target:
            if auto_download and self._download_planck_dataset(download_target):
                self.config["PLANCK_DATA_PATH"] = download_target
                self._store_planck_dataset(download_target)
                return
            if auto_generate and self._generate_planck_dataset(download_target):
                self.config["PLANCK_DATA_PATH"] = download_target
                self._store_planck_dataset(download_target)
                return

        if not _PLANCK_WARNING_EMITTED and self.config.get("VERBOSE", True):
            print(f"[PLANCK][WARN] Planck data file not found at {planck_path}. Validation may be skipped.")
            _PLANCK_WARNING_EMITTED = True

    def _download_planck_dataset(self, target_path: str) -> bool:
        url = self.config.get("PLANCK_DATA_URL")
        if not url:
            return False

        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        except OSError:
            pass

        try:
            if self.config.get("VERBOSE", True):
                print(f"[PLANCK][SETUP] Downloading Planck TT spectrum from {url} ...")
            urllib.request.urlretrieve(url, target_path)
            if self.config.get("VERBOSE", True):
                print(f"[PLANCK][SETUP] Saved Planck data to {target_path}")
            return True
        except Exception as exc:
            if self.config.get("VERBOSE", True):
                print(f"[PLANCK][WARN] Unable to download Planck data from {url}: {exc}")
            try:
                if os.path.exists(target_path):
                    os.remove(target_path)
            except OSError:
                pass
            return False

    def _generate_planck_dataset(self, target_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
        except OSError:
            pass

        if self.config.get("VERBOSE", True):
            print("[PLANCK][SETUP] Generating surrogate Planck TT spectrum...")

        camb_module = None
        if self.config.get("PLANCK_ALLOW_CAMB_SYNTHESIS", True):
            try:
                import camb as camb_module  # type: ignore
            except ImportError:
                camb_module = None

        if camb_module is not None:
            try:
                pars = camb_module.CAMBparams()
                H0 = self.config.get("PLANCK_2018_H0", 67.36)
                omega_b = self.config.get("PLANCK_2018_OMEGA_B", 0.0493)
                omega_m = self.config.get("PLANCK_2018_OMEGA_M", 0.3153)
                ombh2 = omega_b * (H0 / 100.0) ** 2
                omch2 = max(omega_m - omega_b, 0.05) * (H0 / 100.0) ** 2
                pars.set_cosmology(
                    H0=H0,
                    ombh2=ombh2,
                    omch2=omch2,
                    mnu=self.config.get("NEUTRINO_MASS_SUM", 0.12),
                    tau=self.config.get("PLANCK_2018_TAU", 0.0544)
                )
                pars.InitPower.set_params(
                    ns=self.config.get("PLANCK_2018_NS", 0.9649),
                    As=self.config.get("PLANCK_2018_AS", 2.1e-9)
                )
                pars.set_for_lmax(self.config.get("PLANCK_SYNTHETIC_LMAX", 2500), lens_potential_accuracy=0)
                results = camb_module.get_results(pars)
                powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
                tt = powers['total'][:, 0]
                ell = np.arange(tt.size)
                Dl = ell * (ell + 1) * tt / (2 * np.pi)
                mask = ell >= 2
                ell = ell[mask]
                Dl = Dl[mask]
                sigma = np.maximum(Dl * 0.03, 2.0)
                data = np.column_stack([ell, Dl, -sigma, sigma])
                header = "ell Dl err_minus err_plus"
                np.savetxt(target_path, data, header=header, fmt="%-8d %.8e %.8e %.8e")
                if self.config.get("VERBOSE", True):
                    print(f"[PLANCK][SETUP] CAMB-derived Planck surrogate saved to {target_path}")
                return True
            except Exception as exc:
                if self.config.get("VERBOSE", True):
                    print(f"[PLANCK][WARN] CAMB synthesis failed: {exc}")

        try:
            ell = np.arange(2, self.config.get("PLANCK_SYNTHETIC_LMAX", 2500) + 1)
            Dl = self._synthetic_planck_dl(ell)
            sigma = np.maximum(Dl * 0.05, 5.0)
            data = np.column_stack([ell, Dl, -sigma, sigma])
            header = "ell Dl err_minus err_plus"
            np.savetxt(target_path, data, header=header, fmt="%-8d %.8e %.8e %.8e")
            if self.config.get("VERBOSE", True):
                print(f"[PLANCK][SETUP] Analytic Planck surrogate saved to {target_path}")
            return True
        except Exception as exc:
            if self.config.get("VERBOSE", True):
                print(f"[PLANCK][WARN] Analytic Planck surrogate failed: {exc}")
            return False

    def _synthetic_planck_dl(self, ell: np.ndarray) -> np.ndarray:
        ell = np.asarray(ell, dtype=float)
        base = 1200.0 * np.power(np.maximum(ell, 2.0) / 80.0, -0.45)
        peak1 = 5500.0 * np.exp(-0.5 * ((ell - 220.0) / 45.0) ** 2)
        peak2 = 2600.0 * np.exp(-0.5 * ((ell - 540.0) / 60.0) ** 2)
        peak3 = 1700.0 * np.exp(-0.5 * ((ell - 800.0) / 70.0) ** 2)
        peak4 = 900.0  * np.exp(-0.5 * ((ell - 1100.0) / 90.0) ** 2)
        damping = np.exp(-ell / 1800.0)
        return (base + peak1 + peak2 + peak3 + peak4) * damping + 35.0

    def _store_planck_dataset(self, source_path: str) -> None:
        """Copy the resolved Planck dataset into the run directory for reproducibility."""
        if not source_path or not os.path.exists(source_path):
            return

        if not hasattr(self, "paths"):
            self._pending_planck_source = source_path
            return

        dest_dir = self.paths.get("PLANCK_DATA_DIR")
        if not dest_dir:
            self._pending_planck_source = source_path
            return

        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError:
            pass

        dest_path = os.path.join(dest_dir, os.path.basename(source_path))
        try:
            if (
                not os.path.exists(dest_path)
                or os.path.getsize(dest_path) != os.path.getsize(source_path)
                or os.path.getmtime(source_path) > os.path.getmtime(dest_path)
            ):
                import shutil
                shutil.copy2(source_path, dest_path)
            self.paths["PLANCK_DATA_RUN_PATH"] = dest_path
            if self._pending_planck_source == source_path:
                self._pending_planck_source = None
        except Exception as exc:
            if self.config.get("VERBOSE", True):
                print(f"[PLANCK][WARN] Failed to mirror Planck dataset into run folder: {exc}")

    def save_json(self, path: str, obj: dict):
        """Centralized JSON saving with error handling."""
        if obj is None or (isinstance(obj, dict) and not obj):
            print(f"[CTX][JSON][WARN] Skipping empty object: {os.path.basename(path)}")
            return
        
        full_path = self.get_full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)
            return full_path
        except Exception as e:
            print(f"[CTX][JSON] ERROR writing {full_path}: {e}")
        return None

    def save_fig(self, path: str, category: str = None, fig: Optional["plt.Figure"] = None, close: bool = True) -> Optional[str]:
        """Centralized figure saving with variant tag, categorization, and error handling."""
        if not self.config.get("SAVE_FIGS", True):
            if fig is not None and close:
                plt.close(fig)
            elif close:
                plt.close()
            return None

        figure = fig if fig is not None else plt.gcf()

        full_path = self.get_full_path(path)
        full_path_variant = self.with_variant(full_path)
        
        # If category is specified, save to categorized directory
        if category:
            filename = os.path.basename(full_path_variant)
            if category == "anomaly":
                full_path_variant = os.path.join(self.paths["ANOMALY_PNG_DIR"], filename)
            elif category == "physics":
                full_path_variant = os.path.join(self.paths["PHYSICS_PNG_DIR"], filename)
            elif category == "main":
                full_path_variant = os.path.join(self.paths["MAIN_PNG_DIR"], filename)
            elif category == "laws":
                full_path_variant = os.path.join(self.paths["LAWS_PNG_DIR"], filename)
            elif category == "stats":
                full_path_variant = os.path.join(self.paths["STATS_PNG_DIR"], filename)
            elif category == "cmb":
                full_path_variant = os.path.join(self.paths["CMB_PNG_DIR"], filename)
            elif category == "viz":
                full_path_variant = os.path.join(self.paths["VIZ_PNG_DIR"], filename)
        
        os.makedirs(os.path.dirname(full_path_variant), exist_ok=True)
        try:
            # Check if figure has content (axes exist and not empty)
            if not figure.get_axes():
                print(f"[CTX][FIG][WARN] Skipping empty figure: {os.path.basename(full_path_variant)}")
                if close:
                    plt.close(figure)
                return None

            figure.savefig(full_path_variant, dpi=self.config.get('PLOT_SAVE_DPI', 180), bbox_inches="tight")
            if self.config.get("VERBOSE", False):
                print(f"[FIG] Saved: {os.path.basename(full_path_variant)}")
            return full_path_variant
        except Exception as e:
            print(f"[CTX][FIG] ERROR saving {full_path_variant}: {e}")
            return None
        finally:
            if close:
                plt.close(figure)

    def save_csv(self, df: pd.DataFrame, path: str, category: str = None, **kwargs):
        """Centralized CSV saving with variant tag, categorization, and error handling."""
        if df is None or df.empty:
            print(f"[CTX][CSV][WARN] Skipping empty DataFrame: {os.path.basename(path)}")
            return
        
        full_path = self.get_full_path(path)
        full_path_variant = self.with_variant(full_path)
        
        # If category is specified, save to categorized directory
        if category:
            filename = os.path.basename(full_path_variant)
            if category == "anomaly":
                full_path_variant = os.path.join(self.paths["ANOMALY_CSV_DIR"], filename)
            elif category == "physics":
                full_path_variant = os.path.join(self.paths["PHYSICS_CSV_DIR"], filename)
            elif category == "main":
                full_path_variant = os.path.join(self.paths["MAIN_CSV_DIR"], filename)
            elif category == "laws":
                full_path_variant = os.path.join(self.paths["LAWS_CSV_DIR"], filename)
            elif category == "stats":
                full_path_variant = os.path.join(self.paths["STATS_CSV_DIR"], filename)
            elif category == "cmb":
                full_path_variant = os.path.join(self.paths["CMB_CSV_DIR"], filename)
            elif category == "viz":
                full_path_variant = os.path.join(self.paths["VIZ_CSV_DIR"], filename)
        
        os.makedirs(os.path.dirname(full_path_variant), exist_ok=True)
        try:
            # Remove index from kwargs if present to avoid duplicate parameter
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('index', None)
            df.to_csv(full_path_variant, index=False, **kwargs_copy)
            if self.config.get("VERBOSE", False):
                print(f"[CSV] Saved: {os.path.basename(full_path_variant)}")
            return full_path_variant
        except Exception as e:
            print(f"[CTX][CSV] ERROR writing {full_path_variant}: {e}")
        return None

    def get_full_path(self, relative_path: str) -> str:
        """Converts a path relative to SAVE_DIR or a sub-directory into an absolute path."""
        # If already an absolute path, return as-is
        if os.path.isabs(relative_path):
            return relative_path
        
        # Simple heuristic: if path contains an AGGREGATE/CATEGORIZED token, use that base
        if "AGGREGATE_RESULTS" in relative_path or "figs" in relative_path:
            return os.path.join(self.paths["AGGREGATE_DIR"], relative_path)
        if "CATEGORIZED_RESULTS" in relative_path:
             # This path is usually fully constructed already (e.g. within _plot_best_universe)
             # but this acts as a safe path join for phase functions.
             return os.path.join(self.paths["SAVE_DIR"], relative_path) 
        
        # Default to saving within the main SAVE_DIR for top-level artifacts
        return os.path.join(self.paths["SAVE_DIR"], relative_path)
    
    def get_rel_path(self, full_path: str) -> str:
        """Makes a path relative to the run's SAVE_DIR for inclusion in the summary JSON."""
        try:
            return os.path.relpath(full_path, self.paths["SAVE_DIR"])
        except Exception:
            return full_path # Fallback


