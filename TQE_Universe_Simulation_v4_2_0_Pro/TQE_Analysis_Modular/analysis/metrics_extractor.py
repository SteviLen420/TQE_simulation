# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Metrics extraction functions

from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

def extract_extended_metrics(data: Dict, i_def: str) -> Dict:
    """
    Extract EXTENDED metrics from additional CSV files.
    
    Extracts from:
    - tqe_runs.csv: Friedmann, quantum fields, entanglement, entropy
    - emergent_law_summary.csv: Power-law, phase transitions
    - parameter_sensitivity.csv: E/I/X sensitivity
    - cmb_coldspots.csv: Cold spot metrics
    - cmb_aoe.csv: Axis of Evil metrics
    - i_definitions_comparison.csv: I(E) curves
    
    Returns:
        dict: Extended metrics (30-50 additional fields)
    """
    extended = {}
    
    # EMERGENT LAWS METRICS
    if data.get("emergent_laws") is not None:
        law_df = data["emergent_laws"]
        if len(law_df) > 0:
            extended["power_law_exponent"] = law_df.get("power_law_exponent", pd.Series([np.nan])).mean()
            extended["correlation_strength"] = law_df.get("correlation_strength", pd.Series([np.nan])).mean()
            # Phase transitions: check if phase_transition_detected column exists
            if "phase_transition_detected" in law_df.columns:
                extended["phase_transition_detected"] = law_df.get("phase_transition_detected", pd.Series([0])).sum() > 0
                extended["n_phase_transitions"] = law_df.get("phase_transition_detected", pd.Series([0])).sum()
            else:
                # Fallback: check advanced_law_detection_results.csv for phase transition type laws
                advanced_laws = data.get("advanced_laws")
                if isinstance(advanced_laws, pd.DataFrame) and len(advanced_laws) > 0:
                    if "law_type" in advanced_laws.columns:
                        phase_transitions = advanced_laws[advanced_laws["law_type"].str.contains("phase|transition", case=False, na=False)]
                        extended["n_phase_transitions"] = len(phase_transitions)
                        extended["phase_transition_detected"] = len(phase_transitions) > 0
                    else:
                        extended["phase_transition_detected"] = False
                        extended["n_phase_transitions"] = 0
                else:
                    extended["phase_transition_detected"] = False
                    extended["n_phase_transitions"] = 0
        else:
            extended.update({"power_law_exponent": np.nan, "correlation_strength": np.nan, 
                           "phase_transition_detected": False, "n_phase_transitions": 0})
    else:
        # Fallback: check advanced_law_detection_results.csv for phase transition type laws
        advanced_laws = data.get("advanced_laws")
        if isinstance(advanced_laws, pd.DataFrame) and len(advanced_laws) > 0:
            if "law_type" in advanced_laws.columns:
                phase_transitions = advanced_laws[advanced_laws["law_type"].str.contains("phase|transition", case=False, na=False)]
                extended["n_phase_transitions"] = len(phase_transitions)
                extended["phase_transition_detected"] = len(phase_transitions) > 0
            else:
                extended["phase_transition_detected"] = False
                extended["n_phase_transitions"] = 0
        else:
            extended.update({"power_law_exponent": np.nan, "correlation_strength": np.nan, 
                       "phase_transition_detected": False, "n_phase_transitions": 0})
    
    # FRIEDMANN COSMOLOGY METRICS (from tqe_runs.csv)
    if data.get("tqe_runs") is not None:
        runs_df = data["tqe_runs"]
        if "age_Gyr" in runs_df.columns:
            extended["age_Gyr_mean"] = runs_df["age_Gyr"].mean()
            extended["age_Gyr_std"] = runs_df["age_Gyr"].std()
            extended["age_deviation_from_planck"] = abs(runs_df["age_Gyr"].mean() - 13.8)  # Planck: 13.8 Gyr
        else:
            extended.update({"age_Gyr_mean": np.nan, "age_Gyr_std": np.nan, "age_deviation_from_planck": np.nan})
        
        if "H_today" in runs_df.columns:
            extended["H0_mean"] = runs_df["H_today"].mean()
            extended["H0_std"] = runs_df["H_today"].std()
            extended["H0_deviation_from_planck"] = abs(runs_df["H_today"].mean() - 67.4)  # Planck: 67.4 km/s/Mpc
        else:
            extended.update({"H0_mean": np.nan, "H0_std": np.nan, "H0_deviation_from_planck": np.nan})
        
        # Omega parameters
        for omega in ["Omega_m", "Omega_b", "Omega_k"]:
            if omega in runs_df.columns:
                extended[f"{omega}_mean"] = runs_df[omega].mean()
                extended[f"{omega}_std"] = runs_df[omega].std()
            else:
                extended.update({f"{omega}_mean": np.nan, f"{omega}_std": np.nan})
        
        # LOCK-IN DYNAMICS
        if "lock_epoch" in runs_df.columns and "stable_epoch" in runs_df.columns:
            lockin_data = runs_df[runs_df["lockin"] == 1]
            if len(lockin_data) > 0:
                extended["lock_epoch_mean"] = lockin_data["lock_epoch"].mean()
                extended["lock_epoch_std"] = lockin_data["lock_epoch"].std()
                extended["lockin_efficiency"] = (lockin_data["lock_epoch"] - lockin_data["stable_epoch"]).mean()
                # Early lock-in: use <150 threshold instead of <100, because minimum lock_epoch is typically >100
                # This gives meaningful data: 2.78% - 84.29% across different runs
                extended["early_lockin_rate"] = (lockin_data["lock_epoch"] < 150).sum() / len(lockin_data) * 100
            else:
                extended.update({"lock_epoch_mean": np.nan, "lock_epoch_std": np.nan, 
                               "lockin_efficiency": np.nan, "early_lockin_rate": 0.0})
        else:
            extended.update({"lock_epoch_mean": np.nan, "lock_epoch_std": np.nan, 
                           "lockin_efficiency": np.nan, "early_lockin_rate": 0.0})
        
        # QUANTUM FIELDS
        if "vacuum_energy_density" in runs_df.columns:
            extended["vacuum_energy_mean"] = runs_df["vacuum_energy_density"].mean()
            extended["vacuum_energy_std"] = runs_df["vacuum_energy_density"].std()
        else:
            extended.update({"vacuum_energy_mean": np.nan, "vacuum_energy_std": np.nan})
        
        if "zero_point_energy" in runs_df.columns:
            extended["zero_point_energy_mean"] = runs_df["zero_point_energy"].mean()
        else:
            extended["zero_point_energy_mean"] = np.nan
        
        if "quantum_fluctuation_amplitude" in runs_df.columns:
            extended["quantum_fluctuation_mean"] = runs_df["quantum_fluctuation_amplitude"].mean()
        else:
            extended["quantum_fluctuation_mean"] = np.nan
        
        # ENTANGLEMENT
        if "entanglement_entropy" in runs_df.columns:
            extended["entanglement_entropy_mean"] = runs_df["entanglement_entropy"].mean()
            extended["entanglement_entropy_std"] = runs_df["entanglement_entropy"].std()
        else:
            extended.update({"entanglement_entropy_mean": np.nan, "entanglement_entropy_std": np.nan})
        
        if "holographic_entropy" in runs_df.columns:
            extended["holographic_entropy_mean"] = runs_df["holographic_entropy"].mean()
        else:
            extended["holographic_entropy_mean"] = np.nan
        
        # ENTROPY & INFORMATION
        if "entropy_volatility" in runs_df.columns:
            extended["entropy_volatility_mean"] = runs_df["entropy_volatility"].mean()
            extended["entropy_volatility_std"] = runs_df["entropy_volatility"].std()
        else:
            extended.update({"entropy_volatility_mean": np.nan, "entropy_volatility_std": np.nan})
        
        # TOPOLOGY
        if "curvature_radius" in runs_df.columns:
            extended["curvature_radius_mean"] = runs_df["curvature_radius"].mean()
        else:
            extended["curvature_radius_mean"] = np.nan
        
        if "topological_defects" in runs_df.columns:
            extended["topological_defect_rate"] = (runs_df["topological_defects"] > 0).sum() / len(runs_df) * 100
        else:
            extended["topological_defect_rate"] = 0.0
    else:
        # No tqe_runs.csv - fill with NaN
        extended.update({
            "age_Gyr_mean": np.nan, "age_Gyr_std": np.nan, "age_deviation_from_planck": np.nan,
            "H0_mean": np.nan, "H0_std": np.nan, "H0_deviation_from_planck": np.nan,
            "Omega_m_mean": np.nan, "Omega_m_std": np.nan,
            "Omega_b_mean": np.nan, "Omega_b_std": np.nan,
            "Omega_k_mean": np.nan, "Omega_k_std": np.nan,
            "lock_epoch_mean": np.nan, "lock_epoch_std": np.nan, "lockin_efficiency": np.nan, "early_lockin_rate": 0.0,
            "vacuum_energy_mean": np.nan, "vacuum_energy_std": np.nan,
            "zero_point_energy_mean": np.nan, "quantum_fluctuation_mean": np.nan,
            "entanglement_entropy_mean": np.nan, "entanglement_entropy_std": np.nan, "holographic_entropy_mean": np.nan,
            "entropy_volatility_mean": np.nan, "entropy_volatility_std": np.nan,
            "curvature_radius_mean": np.nan, "topological_defect_rate": 0.0
        })
    
    # PARAMETER SENSITIVITY
    if data.get("parameter_sensitivity") is not None:
        sens_df = data["parameter_sensitivity"]
        if len(sens_df) > 0:
            # Try to find parameter column and sensitivity column
            if "parameter" in sens_df.columns and "sensitivity" in sens_df.columns:
                e_vals = sens_df[sens_df["parameter"] == "E"]["sensitivity"].values if len(sens_df[sens_df["parameter"] == "E"]) > 0 else []
                i_vals = sens_df[sens_df["parameter"] == "I"]["sensitivity"].values if len(sens_df[sens_df["parameter"] == "I"]) > 0 else []
                x_vals = sens_df[sens_df["parameter"] == "X"]["sensitivity"].values if len(sens_df[sens_df["parameter"] == "X"]) > 0 else []
                extended["E_sensitivity"] = e_vals[0] if len(e_vals) > 0 else np.nan
                extended["I_sensitivity"] = i_vals[0] if len(i_vals) > 0 else np.nan
                extended["X_sensitivity"] = x_vals[0] if len(x_vals) > 0 else np.nan
            elif "E" in sens_df.columns or "I" in sens_df.columns or "X" in sens_df.columns:
                # Direct columns with E/I/X sensitivity values
                extended["E_sensitivity"] = sens_df["E"].mean() if "E" in sens_df.columns else np.nan
                extended["I_sensitivity"] = sens_df["I"].mean() if "I" in sens_df.columns else np.nan
                extended["X_sensitivity"] = sens_df["X"].mean() if "X" in sens_df.columns else np.nan
        else:
            extended.update({"E_sensitivity": np.nan, "I_sensitivity": np.nan, "X_sensitivity": np.nan})
    else:
        extended.update({"E_sensitivity": np.nan, "I_sensitivity": np.nan, "X_sensitivity": np.nan})
    
    # CMB ANOMALIES
    if data.get("cmb_coldspots") is not None:
        coldspots_df = data["cmb_coldspots"]
        if len(coldspots_df) > 0:
            # Count cold spots: count unique universes with cold_flag == True, or count cold_flag == True rows
            if "cold_flag" in coldspots_df.columns:
                cold_spots = coldspots_df[coldspots_df["cold_flag"] == True]
                extended["n_coldspots_mean"] = cold_spots["universe_id"].nunique() if len(cold_spots) > 0 else 0.0
            else:
                # Fallback: count total rows as cold spots
                extended["n_coldspots_mean"] = len(coldspots_df)
            
            # Cold spot depth: use z_score if available, otherwise temp_uK
            if "z_score" in coldspots_df.columns:
                cold_spots = coldspots_df[coldspots_df.get("cold_flag", pd.Series([True])) == True] if "cold_flag" in coldspots_df.columns else coldspots_df
                extended["coldspot_depth_mean"] = abs(cold_spots["z_score"]).mean() if len(cold_spots) > 0 else np.nan
            elif "temp_uK" in coldspots_df.columns:
                cold_spots = coldspots_df[coldspots_df.get("cold_flag", pd.Series([True])) == True] if "cold_flag" in coldspots_df.columns else coldspots_df
                extended["coldspot_depth_mean"] = abs(cold_spots["temp_uK"]).mean() if len(cold_spots) > 0 else np.nan
            else:
                extended["coldspot_depth_mean"] = np.nan
        else:
            extended.update({"n_coldspots_mean": 0.0, "coldspot_depth_mean": np.nan})
    else:
        extended.update({"n_coldspots_mean": 0.0, "coldspot_depth_mean": np.nan})
    
    if data.get("cmb_aoe") is not None:
        aoe_df = data["cmb_aoe"]
        if len(aoe_df) > 0:
            # Alignment angle: use alignment_angle_deg if available, otherwise alignment_angle
            if "alignment_angle_deg" in aoe_df.columns:
                alignment_angles = aoe_df["alignment_angle_deg"].dropna()
            elif "alignment_angle" in aoe_df.columns:
                alignment_angles = aoe_df["alignment_angle"].dropna()
            else:
                alignment_angles = pd.Series([np.nan])
            
            if len(alignment_angles) > 0:
                extended["alignment_angle_mean"] = alignment_angles.mean()
                extended["alignment_angle_std"] = alignment_angles.std()
            else:
                extended.update({"alignment_angle_mean": np.nan, "alignment_angle_std": np.nan})
        else:
            extended.update({"alignment_angle_mean": np.nan, "alignment_angle_std": np.nan})
    else:
        extended.update({"alignment_angle_mean": np.nan, "alignment_angle_std": np.nan})
    
    # I-DEFINITIONS COMPARISON
    if data.get("i_definitions_comparison") is not None:
        i_comp_df = data["i_definitions_comparison"]
        if i_def in i_comp_df.columns and len(i_comp_df) > 0:
            i_values = i_comp_df[i_def].dropna()
            if len(i_values) > 0:
                extended["I_value_mean"] = i_values.mean()
                extended["I_value_std"] = i_values.std()
                extended["I_value_range"] = i_values.max() - i_values.min()
            else:
                extended.update({"I_value_mean": np.nan, "I_value_std": np.nan, "I_value_range": np.nan})
        else:
            extended.update({"I_value_mean": np.nan, "I_value_std": np.nan, "I_value_range": np.nan})
    else:
        extended.update({"I_value_mean": np.nan, "I_value_std": np.nan, "I_value_range": np.nan})
    
    # LIFE COMPATIBILITY SUMMARY
    life_summary = data.get("life_compatibility")
    if life_summary and isinstance(life_summary, dict):
        metrics_block = life_summary.get("metrics", {})
        extended["life_score_json"] = metrics_block.get("life_compatibility_score", np.nan)
        extended["complexity_score_json"] = metrics_block.get("complexity_score", np.nan)
        extended["information_richness_json"] = metrics_block.get("information_richness", np.nan)
        life_components = life_summary.get("life_components", {})
        extended["life_planck_component"] = life_components.get("planck_fit_quality", np.nan)
        extended["life_stability_component"] = life_components.get("stability_quality", np.nan)
        extended["life_goldilocks_component"] = life_components.get("goldilocks_robustness", np.nan)
    else:
        extended.update({
            "life_score_json": np.nan,
            "complexity_score_json": np.nan,
            "information_richness_json": np.nan,
            "life_planck_component": np.nan,
            "life_stability_component": np.nan,
            "life_goldilocks_component": np.nan
        })

    # PLANCK VALIDATION
    planck_data = data.get("planck_validation")
    if planck_data:
        planck_summary = planck_data.get("summary") or {}
        
        # Try to get from summary first (JSON), then from CSV
        val_df = planck_data.get("validation")
        if val_df is not None and isinstance(val_df, pd.DataFrame) and len(val_df) > 0:
            # Extract from CSV DataFrame
            extended["planck_E"] = val_df["E"].mean() if "E" in val_df.columns else (planck_summary.get("E", np.nan))
            extended["planck_I"] = val_df["I"].mean() if "I" in val_df.columns else (planck_summary.get("I", np.nan))
            extended["planck_alpha"] = val_df["alpha"].mean() if "alpha" in val_df.columns else (planck_summary.get("alpha", np.nan))
            extended["planck_chi2_total"] = val_df["chi2_total"].mean() if "chi2_total" in val_df.columns else (planck_summary.get("chi2_total", np.nan))
            extended["planck_chi2_reduced"] = val_df["chi2_reduced"].mean() if "chi2_reduced" in val_df.columns else (planck_summary.get("chi2_reduced", np.nan))
            extended["planck_score"] = val_df["planck_score"].mean() if "planck_score" in val_df.columns else (planck_summary.get("planck_score", np.nan))
            extended["planck_validation_chi2_mean"] = val_df["chi2"].mean() if "chi2" in val_df.columns else np.nan
        else:
            # Fallback to summary JSON if CSV not available
            extended["planck_E"] = planck_summary.get("E", np.nan)
            extended["planck_I"] = planck_summary.get("I", np.nan)
            extended["planck_alpha"] = planck_summary.get("alpha", np.nan)
            extended["planck_chi2_total"] = planck_summary.get("chi2_total", np.nan)
            extended["planck_chi2_reduced"] = planck_summary.get("chi2_reduced", np.nan)
            extended["planck_score"] = planck_summary.get("planck_score", np.nan)
            extended["planck_validation_chi2_mean"] = np.nan
        
        # Ell span from power spectrum if available
        if val_df is not None and isinstance(val_df, pd.DataFrame) and "ell" in val_df.columns:
            extended["planck_validation_ell_span"] = val_df["ell"].max() - val_df["ell"].min()
        else:
            extended["planck_validation_ell_span"] = np.nan
    else:
        extended.update({
            "planck_E": np.nan,
            "planck_I": np.nan,
            "planck_alpha": np.nan,
            "planck_chi2_total": np.nan,
            "planck_chi2_reduced": np.nan,
            "planck_score": np.nan,
            "planck_validation_chi2_mean": np.nan,
            "planck_validation_ell_span": np.nan
        })

    # ENTROPY VOLATILITY SUMMARY (aggregated)
    entropy_df = data.get("entropy_volatility")
    if isinstance(entropy_df, pd.DataFrame) and "volatility" in entropy_df.columns:
        extended["entropy_volatility_global_mean"] = entropy_df["volatility"].mean()
        extended["entropy_volatility_global_std"] = entropy_df["volatility"].std()
        extended["entropy_volatility_max"] = entropy_df["volatility"].max()
    else:
        extended.update({
            "entropy_volatility_global_mean": np.nan,
            "entropy_volatility_global_std": np.nan,
            "entropy_volatility_max": np.nan
        })

    # STABILITY SWEEPS
    def _compute_sweep_metrics(df: Optional[pd.DataFrame], column: str) -> Tuple[float, float]:
        if df is None or column not in df.columns:
            return np.nan, np.nan
        try:
            eps_vals = pd.to_numeric(df.get("eps"), errors="coerce")
            ratios = pd.to_numeric(df[column], errors="coerce")
            mask = (~eps_vals.isna()) & (~ratios.isna())
            eps_vals = eps_vals[mask]
            ratios = ratios[mask]
            if len(eps_vals) > 1:
                log_eps = np.log10(eps_vals.replace(0, np.nan))
                valid_mask = ~log_eps.isna()
                log_eps = log_eps[valid_mask]
                ratios = ratios[valid_mask]
                if len(log_eps) > 1:
                    slope, intercept = np.polyfit(log_eps, ratios, 1)
                    return slope, intercept
        except Exception:
            return np.nan, np.nan
        return np.nan, np.nan

    eps_sweep = data.get("stability_sweep_eps")
    zero_sweep = data.get("stability_sweep_zero")
    slope, intercept = _compute_sweep_metrics(eps_sweep, "stable_ratio")
    extended["stability_eps_slope"] = slope
    extended["stability_eps_intercept"] = intercept
    if zero_sweep is not None and "stable_ratio" in zero_sweep.columns:
        try:
            extended["stability_zero_baseline"] = pd.to_numeric(zero_sweep["stable_ratio"], errors="coerce").max()
        except Exception:
            extended["stability_zero_baseline"] = np.nan
    else:
        extended["stability_zero_baseline"] = np.nan

    # ADVANCED ANOMALIES
    adv_anomalies = data.get("advanced_anomalies")
    if adv_anomalies:
        adv_df = adv_anomalies.get("advanced_anomalies")
        if isinstance(adv_df, pd.DataFrame) and "deviation_sigma" in adv_df.columns:
            sigmas = pd.to_numeric(adv_df["deviation_sigma"], errors="coerce")
            extended["advanced_anomaly_sigma_mean"] = sigmas.mean()
            extended["advanced_anomaly_sigma_max"] = sigmas.max()
        else:
            extended["advanced_anomaly_sigma_mean"] = np.nan
            extended["advanced_anomaly_sigma_max"] = np.nan
        # Physical anomalies: extract from advanced_anomalies DataFrame (not from separate physical_anomalies file)
        # The advanced_anomaly_detection_results.csv contains all anomalies including physical ones
        adv_df = adv_anomalies.get("advanced_anomalies")
        if isinstance(adv_df, pd.DataFrame) and len(adv_df) > 0:
            # Count physical anomalies (non-CMB anomalies, or all anomalies if we want total count)
            # For now, count all anomalies as "physical" (they're detected in the simulation)
            extended["physical_anomaly_count"] = len(adv_df)
            # Optionally filter by anomaly_type if we want only specific types:
            # physical_anomalies = adv_df[adv_df["anomaly_type"].str.contains("physical|information|quantum", case=False, na=False)]
            # extended["physical_anomaly_count"] = len(physical_anomalies) if len(physical_anomalies) > 0 else 0
        else:
            # Fallback: try separate physical_anomalies file if it exists
            phys_df = adv_anomalies.get("physical_anomalies")
            extended["physical_anomaly_count"] = len(phys_df) if isinstance(phys_df, pd.DataFrame) and len(phys_df) > 0 else 0
        
        # CMB Gaussianity: extract skewness and kurtosis (no p_value column exists)
        gaussian_df = adv_anomalies.get("cmb_gaussianity")
        if isinstance(gaussian_df, pd.DataFrame) and len(gaussian_df) > 0:
            if "skewness" in gaussian_df.columns:
                extended["cmb_gaussianity_skewness_mean"] = gaussian_df["skewness"].mean()
            else:
                extended["cmb_gaussianity_skewness_mean"] = np.nan
            if "kurtosis" in gaussian_df.columns:
                extended["cmb_gaussianity_kurtosis_mean"] = gaussian_df["kurtosis"].mean()
            else:
                extended["cmb_gaussianity_kurtosis_mean"] = np.nan
            # For backward compatibility, use skewness as gaussianity measure
            extended["cmb_gaussianity_p_mean"] = abs(gaussian_df["skewness"].mean()) if "skewness" in gaussian_df.columns else np.nan
        else:
            extended["cmb_gaussianity_skewness_mean"] = np.nan
            extended["cmb_gaussianity_kurtosis_mean"] = np.nan
            extended["cmb_gaussianity_p_mean"] = np.nan
        
        # CMB Isotropy: extract MSE (Mean Squared Error) as anisotropy index (no anisotropy_index column exists)
        isotropy_df = adv_anomalies.get("cmb_isotropy")
        if isinstance(isotropy_df, pd.DataFrame) and len(isotropy_df) > 0:
            if "MSE" in isotropy_df.columns:
                extended["cmb_anisotropy_index_mean"] = isotropy_df["MSE"].mean()
            elif "MSE_north_south" in isotropy_df.columns:
                extended["cmb_anisotropy_index_mean"] = isotropy_df["MSE_north_south"].mean()
            else:
                extended["cmb_anisotropy_index_mean"] = np.nan
        else:
            extended["cmb_anisotropy_index_mean"] = np.nan
    else:
        extended.update({
            "advanced_anomaly_sigma_mean": np.nan,
            "advanced_anomaly_sigma_max": np.nan,
            "physical_anomaly_count": 0,
            "cmb_gaussianity_skewness_mean": np.nan,
            "cmb_gaussianity_kurtosis_mean": np.nan,
            "cmb_gaussianity_p_mean": np.nan,
            "cmb_anisotropy_index_mean": np.nan
        })

    # NESTED SAMPLING
    nested_df = data.get("nested_sampling")
    if isinstance(nested_df, pd.DataFrame):
        extended["nested_sampling_iterations"] = len(nested_df)
        if "logZ" in nested_df.columns:
            extended["nested_logZ_final"] = nested_df["logZ"].iloc[-1]
            extended["nested_logZ_span"] = nested_df["logZ"].max() - nested_df["logZ"].min()
        else:
            extended["nested_logZ_final"] = np.nan
            extended["nested_logZ_span"] = np.nan
    else:
        extended.update({
            "nested_sampling_iterations": 0,
            "nested_logZ_final": np.nan,
            "nested_logZ_span": np.nan
        })

    # COMPREHENSIVE PHYSICS DATA (from comprehensive_universe_physics_data.csv)
    # This contains: age_Gyr, vacuum_energy, quantum_correction, entanglement_entropy, etc.
    comprehensive_physics = data.get("comprehensive_physics")
    if isinstance(comprehensive_physics, pd.DataFrame) and len(comprehensive_physics) > 0:
        # Friedmann cosmology (override tqe_runs if available)
        if "age_Gyr" in comprehensive_physics.columns:
            age_vals = pd.to_numeric(comprehensive_physics["age_Gyr"], errors="coerce")
            age_vals = age_vals.dropna()
            if len(age_vals) > 0:
                extended["age_Gyr_mean"] = age_vals.mean()
                extended["age_Gyr_std"] = age_vals.std()
                extended["age_deviation_from_planck"] = abs(age_vals.mean() - 13.8)  # Planck: 13.8 Gyr
        
        # Quantum fields
        if "vacuum_energy" in comprehensive_physics.columns:
            vac_vals = pd.to_numeric(comprehensive_physics["vacuum_energy"], errors="coerce")
            vac_vals = vac_vals.dropna()
            if len(vac_vals) > 0:
                extended["vacuum_energy_mean"] = vac_vals.mean()
                extended["vacuum_energy_std"] = vac_vals.std()
        
        if "quantum_correction" in comprehensive_physics.columns:
            qc_vals = pd.to_numeric(comprehensive_physics["quantum_correction"], errors="coerce")
            qc_vals = qc_vals.dropna()
            if len(qc_vals) > 0:
                extended["zero_point_energy_mean"] = qc_vals.mean()
                extended["quantum_fluctuation_mean"] = qc_vals.std()  # Use std as fluctuation measure
        
        # Entanglement
        if "entanglement_entropy" in comprehensive_physics.columns:
            ent_vals = pd.to_numeric(comprehensive_physics["entanglement_entropy"], errors="coerce")
            ent_vals = ent_vals.dropna()
            if len(ent_vals) > 0:
                extended["entanglement_entropy_mean"] = ent_vals.mean()
                extended["entanglement_entropy_std"] = ent_vals.std()
        
        if "holographic_entropy" in comprehensive_physics.columns:
            holo_vals = pd.to_numeric(comprehensive_physics["holographic_entropy"], errors="coerce")
            holo_vals = holo_vals.dropna()
            if len(holo_vals) > 0:
                extended["holographic_entropy_mean"] = holo_vals.mean()
        
        # Entanglement network
        if "entanglement_density" in comprehensive_physics.columns:
            ed_vals = pd.to_numeric(comprehensive_physics["entanglement_density"], errors="coerce")
            ed_vals = ed_vals.dropna()
            if len(ed_vals) > 0:
                extended["entanglement_density_mean"] = ed_vals.mean()
        
        if "causal_scale" in comprehensive_physics.columns:
            cs_vals = pd.to_numeric(comprehensive_physics["causal_scale"], errors="coerce")
            cs_vals = cs_vals.dropna()
            if len(cs_vals) > 0:
                extended["causal_scale_mean"] = cs_vals.mean()
        
        # Topology
        if "topological_defects" in comprehensive_physics.columns:
            td_vals = pd.to_numeric(comprehensive_physics["topological_defects"], errors="coerce")
            td_vals = td_vals.dropna()
            if len(td_vals) > 0:
                extended["topological_defect_rate"] = (td_vals > 0).sum() / len(td_vals) * 100
    
    # STATISTICAL FINETUNING (from statistical_finetuning_summary.csv)
    statistical_finetuning = data.get("statistical_finetuning")
    if isinstance(statistical_finetuning, pd.DataFrame) and len(statistical_finetuning) > 0:
        if "lockin_rate" in statistical_finetuning.columns:
            finetune_vals = pd.to_numeric(statistical_finetuning["lockin_rate"], errors="coerce")
            finetune_vals = finetune_vals.dropna()
            if len(finetune_vals) > 0:
                extended["statistical_finetuning_rate_mean"] = finetune_vals.mean()
                extended["statistical_finetuning_rate_max"] = finetune_vals.max()
        if "group_name" in statistical_finetuning.columns:
            # Count E≈I groups (finetuning indicator)
            ei_groups = statistical_finetuning[statistical_finetuning["group_name"].str.contains("E≈I|E=I|E_eq_I", case=False, na=False)]
            extended["statistical_finetuning_E_eq_I_count"] = len(ei_groups)
    
    # CMB POWER SPECTRUM
    cmb_power_spectrum = data.get("cmb_power_spectrum")
    if isinstance(cmb_power_spectrum, pd.DataFrame) and len(cmb_power_spectrum) > 0:
        if "C_ell_scaled" in cmb_power_spectrum.columns:
            c_ell_vals = pd.to_numeric(cmb_power_spectrum["C_ell_scaled"], errors="coerce")
            c_ell_vals = c_ell_vals.dropna()
            if len(c_ell_vals) > 0:
                extended["cmb_power_spectrum_mean"] = c_ell_vals.mean()
                extended["cmb_power_spectrum_max"] = c_ell_vals.max()
        if "fit_R_squared" in cmb_power_spectrum.columns:
            r2_vals = pd.to_numeric(cmb_power_spectrum["fit_R_squared"], errors="coerce")
            r2_vals = r2_vals.dropna()
            if len(r2_vals) > 0:
                extended["cmb_power_spectrum_fit_R2_mean"] = r2_vals.mean()
        if "ell" in cmb_power_spectrum.columns:
            ell_vals = pd.to_numeric(cmb_power_spectrum["ell"], errors="coerce")
            ell_vals = ell_vals.dropna()
            if len(ell_vals) > 0:
                extended["cmb_power_spectrum_ell_span"] = ell_vals.max() - ell_vals.min()
    else:
        extended.update({
            "cmb_power_spectrum_mean": np.nan,
            "cmb_power_spectrum_max": np.nan,
            "cmb_power_spectrum_fit_R2_mean": np.nan,
            "cmb_power_spectrum_ell_span": np.nan
        })
    
    # PRE-FLUCTUATION / SEEDS
    pre_pairs = data.get("pre_fluctuation_pairs")
    extended["pre_fluctuation_pairs"] = len(pre_pairs) if isinstance(pre_pairs, pd.DataFrame) else 0
    
    universe_seeds = data.get("universe_seeds")
    if isinstance(universe_seeds, pd.DataFrame):
        if "seed" in universe_seeds.columns:
            extended["unique_seed_count"] = universe_seeds["seed"].nunique()
        else:
            extended["unique_seed_count"] = len(universe_seeds)
    else:
        extended["unique_seed_count"] = np.nan

    # TOP UNIVERSE SNAPSHOT (from summary)
    summary_block = data.get("summary")
    if summary_block and summary_block.get("complexity_analysis", {}).get("top_universes"):
        top_universe = summary_block["complexity_analysis"]["top_universes"][0]
        extended["top_universe_seed"] = top_universe.get("seed")
        extended["top_universe_lock_epoch"] = top_universe.get("lock_epoch")
        extended["top_universe_I"] = top_universe.get("I")
    else:
        extended["top_universe_seed"] = None
        extended["top_universe_lock_epoch"] = np.nan
        extended["top_universe_I"] = np.nan

    return extended


def extract_metrics_from_summary(summary: Dict, i_def: str) -> Dict:
    """
    Extract all relevant metrics from summary JSON.
    
    Includes:
    - Basic stability metrics
    - Goldilocks and Bayesian parameters
    - Advanced complexity indicators
    - Life-compatibility indicators
    """
    stab_sum = summary.get("stability_summary", {})
    gold = summary.get("goldilocks_window_used", {})
    bayes = summary.get("bayesian_model_selection", {})
    
    # Basic metrics
    total_univ = summary.get("N_samples", 0)
    stable_count = stab_sum.get("stable_universes", 0)
    lockin_count = stab_sum.get("lockin_universes", 0)
    
    metrics = {
        "i_definition": i_def,
        
        # Stability metrics
        "total_universes": total_univ,
        "stable_count": stable_count,
        "unstable_count": stab_sum.get("unstable_universes", 0),
        "lockin_count": lockin_count,
        "stable_percent": stab_sum.get("stable_percent", 0.0),
        "unstable_percent": stab_sum.get("unstable_percent", 0.0),
        "lockin_percent": stab_sum.get("lockin_percent", 0.0),
        
        # Goldilocks metrics
        "X_peak": gold.get("X_peak", 0.0),
        "X_peak_uncertainty": gold.get("X_peak_uncertainty", 0.0),
        "X_low": gold.get("X_low_plot_est", 0.0),
        "X_high": gold.get("X_high_plot_est", 0.0),
        "goldilocks_width": gold.get("X_high_plot_est", 0.0) - gold.get("X_low_plot_est", 0.0),
        
        # Bayesian metrics
        "ucb_kappa": gold.get("ucb_kappa", 0.0),
        "gp_noise": gold.get("gp_noise", 0.0),
        "bayesian_samples": gold.get("total_sampled", 0),
        
        # Bayesian model selection
        "BIC": bayes.get("BIC", np.nan),
        "AIC": bayes.get("AIC", np.nan),
        "log_evidence": bayes.get("log_evidence", np.nan),
        "chi_squared_reduced": bayes.get("chi_squared_reduced", np.nan),
    }
    
    # ═══════════════════════════════════════════════════════════════
    # ADVANCED METRICS: COMPLEXITY & LIFE-COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════
    
    # COMPLEXITY SCORE (0-100):
    # Measures structural richness and information content
    complexity_components = []
    
    # 1. Lock-in quality (not quantity!) - Fast, decisive lock-in = complex
    if lockin_count > 0 and total_univ > 0:
        lockin_rate = lockin_count / total_univ
        # Higher lock-in rate among TOTAL (not just stable) = better
        complexity_components.append(min(lockin_rate * 200, 100))  # Scale: 0-50% → 0-100
    else:
        complexity_components.append(0)
    
    # 2. Goldilocks precision (lower uncertainty = sharper, more interesting physics)
    if gold.get("X_peak", 0) > 0:
        rel_uncertainty = gold.get("X_peak_uncertainty", 0) / gold.get("X_peak", 1)
        precision_score = max(0, 100 - rel_uncertainty * 1000)  # Lower uncertainty = higher score
        complexity_components.append(min(precision_score, 100))
    else:
        complexity_components.append(50)
    
    # 3. Information richness (E+I specific) - Is I-coupling effective?
    if i_def != "energy_only":
        # E+I: Effectiveness = stable % relative to E-only (will be computed in comparison)
        # Use lock-in rate as proxy for I-parameter effectiveness
        info_richness = min(stab_sum.get("lockin_percent", 0) * 5, 100)  # 20% lock-in = 100 score
        complexity_components.append(info_richness)
    else:
        complexity_components.append(0)  # E-only has no I-coupling
    
    # Average complexity components
    metrics["complexity_score"] = np.mean(complexity_components) if complexity_components else 0
    
    # LIFE-COMPATIBILITY SCORE (0-100):
    # Measures potential for structure formation and life
    life_components = []
    
    # 1. Planck fit quality (observationally compatible universe)
    chi2 = bayes.get("chi_squared_reduced", np.nan)
    if not np.isnan(chi2):
        # Perfect fit = 1.0, good fit < 2.0
        planck_score = max(0, 100 - abs(chi2 - 1.0) * 25)  # |χ²-1| = 0 → 100, |χ²-1| = 4 → 0
        life_components.append(min(planck_score, 100))
    else:
        life_components.append(50)  # Neutral if no data
    
    # 2. Stability quality (stable AND lock-in is best)
    if stable_count > 0:
        # Proportion of stable universes that lock-in (high = good)
        lockin_among_stable = lockin_count / stable_count if stable_count > 0 else 0
        stability_quality = lockin_among_stable * 100
        life_components.append(min(stability_quality, 100))
    else:
        life_components.append(0)
    
    # 3. Goldilocks robustness (wider, more forgiving zone = life-compatible)
    gold_width = gold.get("X_high_plot_est", 0) - gold.get("X_low_plot_est", 0)
    if gold_width > 0:
        # Wider zone = more robust (normalize to ~10-20 typical width)
        robustness = min(gold_width / 20.0 * 100, 100)
        life_components.append(robustness)
    else:
        life_components.append(50)
    
    # Average life-compatibility components
    metrics["life_compatibility_score"] = np.mean(life_components) if life_components else 0
    
    # INFORMATION RICHNESS (E+I specific):
    # How effective is the I-parameter at directing complexity?
    if i_def != "energy_only":
        # Lock-in rate is proxy for I-parameter effectiveness
        info_effectiveness = min(stab_sum.get("lockin_percent", 0) * 5, 100)
        metrics["information_richness"] = info_effectiveness
    else:
        metrics["information_richness"] = 0
    
    return metrics


def build_metrics_dataframe(collected_data: Dict) -> pd.DataFrame:
    """Build comprehensive metrics DataFrame from collected data (EXTENDED)."""
    metrics_list = []
    
    print("\n  Building extended metrics DataFrame...")
    
    # E-only metrics
    for dirname, data in collected_data["eonly"].items():
        metrics = extract_metrics_from_summary(data["summary"], "energy_only")
        metrics["run_type"] = "E-only"
        # Add extended metrics
        extended = extract_extended_metrics(data, "energy_only")
        metrics.update(extended)
        metrics_list.append(metrics)
        print(f"    ✅ Extracted extended metrics: energy_only")
    
    # E+I metrics
    for i_def, data in collected_data["ei"].items():
        metrics = extract_metrics_from_summary(data["summary"], i_def)
        metrics["run_type"] = "E+I"
        # Add extended metrics
        extended = extract_extended_metrics(data, i_def)
        metrics.update(extended)
        metrics_list.append(metrics)
        print(f"    ✅ Extracted extended metrics: {i_def}")
    
    df = pd.DataFrame(metrics_list)
    print(f"  ✅ DataFrame built: {len(df)} runs, {len(df.columns)} columns (50-80 extended)")
    return df

