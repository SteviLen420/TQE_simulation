# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Model selection and ranking functions

import os
import time
import json
import numpy as np
import pandas as pd

def select_best_model(df_metrics: pd.DataFrame, output_dir: str, config: dict):
    """
    PHASE 5: Best model selection using weighted multi-metric ranking.
    
    Applies configurable weights to normalized metrics (0-100 scale) to identify
    the best-performing I-definition for TQE simulations.
    
    Ranking methodology:
    - Stability rate: 30% (higher is better)
    - Lock-in rate: 20% (higher is better)
    - Planck χ² fit: 20% (lower is better, inverted for scoring)
    - Goldilocks precision: 15% (lower uncertainty is better, inverted)
    - CMB anomaly match: 10% (anomaly detection rates)
    - Bayesian efficiency: 5% (GP performance)
    
    Generates:
    - weighted_ranking.csv (all models with component scores)
    - top_3_models.json (top 3 ranked models with full metrics)
    - recommendation_report.md (scientific justification and usage guide)
    """
    print("\n" + "="*70)
    print("PHASE 5: BEST MODEL SELECTION & TRIPLE RANKING")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"].copy()
    
    # Get ranking weights from config
    ranking_weights_stability = config.get("RANKING_WEIGHTS_STABILITY", {})
    ranking_weights_complexity = config.get("RANKING_WEIGHTS_COMPLEXITY", {})
    ranking_weights_physical_laws = config.get("RANKING_WEIGHTS_PHYSICAL_LAWS", {})
    
    # ==================================================================
    # TRIPLE RANKING SYSTEM: STABILITY vs COMPLEXITY vs PHYSICAL-LAWS
    # ==================================================================
    
    # RANKING 1: STABILITY-FOCUSED (Traditional)
    df_stability_rank = pd.DataFrame()
    df_stability_rank['i_definition'] = df_ei["i_definition"]
    
    # Normalize components
    df_stability_rank['stability_score'] = (df_ei["stable_percent"] / df_ei["stable_percent"].max()) * 100 if df_ei["stable_percent"].max() > 0 else 50.0
    
    if df_ei["lockin_percent"].max() > 0:
        df_stability_rank['lockin_score'] = (df_ei["lockin_percent"] / df_ei["lockin_percent"].max()) * 100
    else:
        df_stability_rank['lockin_score'] = 50.0
    
    relative_uncertainty = df_ei["X_peak_uncertainty"] / df_ei["X_peak"]
    df_stability_rank['precision_score'] = (1 - (relative_uncertainty / relative_uncertainty.max()).clip(0, 1)) * 100
    
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_values = df_ei["chi_squared_reduced"].fillna(df_ei["chi_squared_reduced"].max() * 2)
        df_stability_rank['planck_score'] = (1 - (chi_values / chi_values.max()).clip(0, 1)) * 100
    else:
        df_stability_rank['planck_score'] = 50.0
    
    # CMB anomaly score (from cold spots and alignment)
    if "n_coldspots_mean" in df_ei.columns and "alignment_angle_mean" in df_ei.columns:
        # Normalize cold spots (Planck ~1-2 major cold spots)
        coldspot_match = 100 / (1 + abs(df_ei["n_coldspots_mean"].fillna(0) - 1.5))
        # Normalize alignment (lower angle = better alignment)
        alignment_match = 100 / (1 + df_ei["alignment_angle_mean"].fillna(90))
        df_stability_rank['anomaly_score'] = (coldspot_match + alignment_match) / 2
    else:
        df_stability_rank['anomaly_score'] = 50.0  # Default if no data
    
    # Bayesian efficiency score (from Goldilocks precision and Bayesian samples)
    if "X_peak_uncertainty" in df_ei.columns and "bayesian_samples" in df_ei.columns:
        # Lower uncertainty and more samples = better efficiency
        rel_unc = df_ei["X_peak_uncertainty"] / df_ei["X_peak"].replace(0, 1)
        precision_score = (1 - rel_unc.clip(0, 1)) * 100
        sample_score = (df_ei["bayesian_samples"].fillna(0) / 100 * 100).clip(upper=100)  # Normalize to 100
        df_stability_rank['bayesian_score'] = (precision_score + sample_score) / 2
    else:
        df_stability_rank['bayesian_score'] = 50.0  # Default if no data
    
    # Calculate stability-weighted score
    df_stability_rank['stability_total_score'] = (
        df_stability_rank['stability_score'] * ranking_weights_stability.get('stability_rate', 0.30) +
        df_stability_rank['lockin_score'] * ranking_weights_stability.get('lockin_rate', 0.20) +
        df_stability_rank['planck_score'] * ranking_weights_stability.get('planck_chi2_fit', 0.20) +
        df_stability_rank['precision_score'] * ranking_weights_stability.get('goldilocks_precision', 0.15) +
        df_stability_rank['anomaly_score'] * ranking_weights_stability.get('cmb_anomaly_match', 0.10) +
        df_stability_rank['bayesian_score'] * ranking_weights_stability.get('bayesian_efficiency', 0.05)
    )
    
    df_stability_rank = df_stability_rank.sort_values('stability_total_score', ascending=False)
    
    # RANKING 2: COMPLEXITY-FOCUSED (TQE-Consistent)
    df_complexity_rank = pd.DataFrame()
    df_complexity_rank['i_definition'] = df_ei["i_definition"]
    
    # Use already-computed advanced scores
    df_complexity_rank['complexity_score'] = df_ei["complexity_score"]
    df_complexity_rank['life_compatibility_score'] = df_ei["life_compatibility_score"]
    df_complexity_rank['information_richness'] = df_ei["information_richness"]
    
    # Stability quality (not quantity!)
    df_complexity_rank['stability_quality'] = df_ei["lockin_percent"] * 2  # Lock-in among total
    
    # Observational match
    if not df_ei["chi_squared_reduced"].isna().all():
        chi_values = df_ei["chi_squared_reduced"].fillna(df_ei["chi_squared_reduced"].max() * 2)
        df_complexity_rank['observational_score'] = (1 - (chi_values / chi_values.max()).clip(0, 1)) * 100
    else:
        df_complexity_rank['observational_score'] = 50.0
    
    # Calculate complexity-weighted score
    df_complexity_rank['complexity_total_score'] = (
        df_complexity_rank['complexity_score'] * ranking_weights_complexity.get('complexity_score', 0.35) +
        df_complexity_rank['life_compatibility_score'] * ranking_weights_complexity.get('life_compatibility', 0.25) +
        df_complexity_rank['information_richness'] * ranking_weights_complexity.get('information_richness', 0.20) +
        df_complexity_rank['stability_quality'] * ranking_weights_complexity.get('stability_quality', 0.10) +
        df_complexity_rank['observational_score'] * ranking_weights_complexity.get('observational_match', 0.10)
    )
    
    df_complexity_rank = df_complexity_rank.sort_values('complexity_total_score', ascending=False)
    
    # RANKING 3: PHYSICAL-LAWS-FOCUSED
    df_physical_rank = pd.DataFrame()
    df_physical_rank['i_definition'] = df_ei["i_definition"]
    
    # 1. Emergent laws quality
    if "power_law_exponent" in df_ei.columns and "n_phase_transitions" in df_ei.columns:
        # Normalize power-law exponent closeness to 1.0 (linear)
        power_law_quality = 100 / (1 + abs(df_ei["power_law_exponent"].fillna(1.0) - 1.0))
        # Normalize phase transitions
        phase_trans_quality = (df_ei["n_phase_transitions"].fillna(0) / df_ei["n_phase_transitions"].max()) * 100 if df_ei["n_phase_transitions"].max() > 0 else 50.0
        df_physical_rank['emergent_laws_quality'] = (power_law_quality + phase_trans_quality) / 2
    else:
        df_physical_rank['emergent_laws_quality'] = 50.0
    
    # 2. Friedmann consistency
    if "age_deviation_from_planck" in df_ei.columns and "H0_deviation_from_planck" in df_ei.columns:
        age_consistency = 100 / (1 + df_ei["age_deviation_from_planck"].fillna(10))
        h0_consistency = 100 / (1 + df_ei["H0_deviation_from_planck"].fillna(10))
        df_physical_rank['friedmann_consistency'] = (age_consistency + h0_consistency) / 2
    else:
        df_physical_rank['friedmann_consistency'] = 50.0
    
    # 3. CMB anomaly match
    if "n_coldspots_mean" in df_ei.columns:
        # Normalize cold spots (Planck ~1-2 major cold spots)
        coldspot_match = 100 / (1 + abs(df_ei["n_coldspots_mean"].fillna(0) - 1.5))
        df_physical_rank['cmb_anomaly_match'] = coldspot_match
    else:
        df_physical_rank['cmb_anomaly_match'] = 50.0
    
    # 4. Lock-in efficiency (fast, decisive)
    if "lockin_efficiency" in df_ei.columns:
        # Lower efficiency = faster lock-in = better
        eff_values = df_ei["lockin_efficiency"].fillna(df_ei["lockin_efficiency"].max() * 2)
        df_physical_rank['lockin_efficiency'] = (1 - (eff_values / eff_values.max()).clip(0, 1)) * 100
    else:
        df_physical_rank['lockin_efficiency'] = df_ei["lockin_percent"]  # Fallback to lock-in rate
    
    # 5. Quantum field realism (vacuum energy consistency + fluctuation amplitude)
    if "vacuum_energy_mean" in df_ei.columns and "quantum_fluctuation_mean" in df_ei.columns:
        # Check if vacuum energy is in reasonable range (0.1-1.0)
        vacuum_consistency = 100 / (1 + abs(df_ei["vacuum_energy_mean"].fillna(0.5) - 0.5) * 2)
        # Check if fluctuations are non-zero (indicates quantum activity)
        fluctuation_activity = (df_ei["quantum_fluctuation_mean"].fillna(0) > 0).astype(float) * 100
        df_physical_rank['quantum_field_realism'] = (vacuum_consistency + fluctuation_activity) / 2
    elif "vacuum_energy_mean" in df_ei.columns:
        # Fallback: only vacuum energy
        vacuum_consistency = 100 / (1 + abs(df_ei["vacuum_energy_mean"].fillna(0.5) - 0.5) * 2)
        df_physical_rank['quantum_field_realism'] = vacuum_consistency
    else:
        df_physical_rank['quantum_field_realism'] = 50.0  # Default if no data
    
    # Calculate physical-laws-weighted score
    df_physical_rank['physical_laws_total_score'] = (
        df_physical_rank['emergent_laws_quality'] * ranking_weights_physical_laws.get('emergent_laws_quality', 0.30) +
        df_physical_rank['friedmann_consistency'] * ranking_weights_physical_laws.get('friedmann_consistency', 0.25) +
        df_physical_rank['cmb_anomaly_match'] * ranking_weights_physical_laws.get('cmb_anomaly_match', 0.20) +
        df_physical_rank['lockin_efficiency'] * ranking_weights_physical_laws.get('lockin_efficiency', 0.15) +
        df_physical_rank['quantum_field_realism'] * ranking_weights_physical_laws.get('quantum_field_realism', 0.10)
    )
    
    df_physical_rank = df_physical_rank.sort_values('physical_laws_total_score', ascending=False)
    
    # Merge all THREE rankings for comprehensive view
    df_scores = df_stability_rank.copy()
    df_scores['complexity_total_score'] = df_complexity_rank.set_index('i_definition').loc[df_scores['i_definition'], 'complexity_total_score'].values
    df_scores['physical_laws_total_score'] = df_physical_rank.set_index('i_definition').loc[df_scores['i_definition'], 'physical_laws_total_score'].values
    
    # Save weighted ranking
    print("\n5.1 Weighted Ranking")
    df_scores.to_csv(os.path.join(output_dir, "weighted_ranking.csv"), index=False)
    print("   ✅ weighted_ranking.csv")
    
    # Save all THREE rankings separately
    print("\n5.2 Triple Ranking CSVs")
    df_stability_rank.to_csv(os.path.join(output_dir, "ranking_stability_focused.csv"), index=False)
    print("   ✅ ranking_stability_focused.csv")
    
    df_complexity_rank.to_csv(os.path.join(output_dir, "ranking_complexity_focused.csv"), index=False)
    print("   ✅ ranking_complexity_focused.csv")
    
    df_physical_rank.to_csv(os.path.join(output_dir, "ranking_physical_laws_focused.csv"), index=False)
    print("   ✅ ranking_physical_laws_focused.csv")
    
    # Top 3 models for each system
    print("\n5.3 Top 3 Models (Triple Rankings)")
    top_3_stability = df_stability_rank.head(3).to_dict('records')
    top_3_complexity = df_complexity_rank.head(3).to_dict('records')
    top_3_physical = df_physical_rank.head(3).to_dict('records')
    
    top_models = {
        "stability_focused": top_3_stability,
        "complexity_focused": top_3_complexity,
        "physical_laws_focused": top_3_physical
    }
    
    with open(os.path.join(output_dir, "top_3_models_triple.json"), 'w') as f:
        json.dump(top_models, f, indent=2)
    print("   ✅ top_3_models_triple.json")
    
    # Triple Recommendation Report
    print("\n5.4 Triple Ranking Recommendation Report")
    report = []
    report.append("# TQE ANALYSIS PIPELINE v4.2.0 PRO - TRIPLE RANKING REPORT")
    report.append("=" * 70)
    report.append("")
    report.append("## CRITICAL INSIGHT: THREE RANKING PERSPECTIVES")
    report.append("")
    report.append("**Different goals require different I-definitions!**")
    report.append("")
    report.append("- **Stability-Focused**: Maximizes stable universe percentage")
    report.append("- **Complexity-Focused**: Maximizes structural complexity and life-compatibility")
    report.append("- **Physical-Laws-Focused**: Maximizes observational realism (Planck, CMB, emergent laws)")
    report.append("")
    report.append("For TQE theory validation, PHYSICAL-LAWS RANKING is most observationally consistent!")
    report.append("")
    report.append("=" * 70)
    report.append("")
    
    # RANKING 1: STABILITY-FOCUSED
    report.append("## RANKING 1: STABILITY-FOCUSED (Traditional Approach)")
    report.append("")
    report.append("### Methodology")
    report.append(f"- Stability Rate: {ranking_weights_stability.get('stability_rate', 0.30)*100:.0f}%")
    report.append(f"- Lock-in Rate: {ranking_weights_stability.get('lockin_rate', 0.20)*100:.0f}%")
    report.append(f"- Planck χ² Fit: {ranking_weights_stability.get('planck_chi2_fit', 0.20)*100:.0f}%")
    report.append(f"- Goldilocks Precision: {ranking_weights_stability.get('goldilocks_precision', 0.15)*100:.0f}%")
    report.append(f"- CMB Anomaly: {ranking_weights_stability.get('cmb_anomaly_match', 0.10)*100:.0f}%")
    report.append(f"- Bayesian Efficiency: {ranking_weights_stability.get('bayesian_efficiency', 0.05)*100:.0f}%")
    report.append("")
    report.append("### Top 3 Models")
    report.append("")
    
    for rank, model in enumerate(top_3_stability, 1):
        i_def = model['i_definition']
        score = model['stability_total_score']
        orig = df_ei[df_ei['i_definition'] == i_def].iloc[0]
        
        report.append(f"**{rank}. {i_def.upper()}** (Score: {score:.2f}/100)")
        report.append(f"   - Stable: {orig['stable_percent']:.2f}%, Lock-in: {orig['lockin_percent']:.2f}%")
        report.append("")
    
    report.append("=" * 70)
    report.append("")
    
    # RANKING 2: COMPLEXITY-FOCUSED
    report.append("## RANKING 2: COMPLEXITY-FOCUSED (TQE-Consistent Approach)")
    report.append("")
    report.append("### Methodology")
    report.append(f"- Complexity Score: {ranking_weights_complexity.get('complexity_score', 0.35)*100:.0f}%")
    report.append(f"- Life-Compatibility: {ranking_weights_complexity.get('life_compatibility', 0.25)*100:.0f}%")
    report.append(f"- Information Richness: {ranking_weights_complexity.get('information_richness', 0.20)*100:.0f}%")
    report.append(f"- Stability Quality: {ranking_weights_complexity.get('stability_quality', 0.10)*100:.0f}%")
    report.append(f"- Observational Match: {ranking_weights_complexity.get('observational_match', 0.10)*100:.0f}%")
    report.append("")
    report.append("### Top 3 Models")
    report.append("")
    
    for rank, model in enumerate(top_3_complexity, 1):
        i_def = model['i_definition']
        score = model['complexity_total_score']
        orig = df_ei[df_ei['i_definition'] == i_def].iloc[0]
        
        report.append(f"**{rank}. {i_def.upper()}** (Score: {score:.2f}/100)")
        report.append(f"   - Complexity: {orig['complexity_score']:.2f}, Life: {orig['life_compatibility_score']:.2f}")
        report.append(f"   - Stable: {orig['stable_percent']:.2f}%, Lock-in: {orig['lockin_percent']:.2f}%")
        report.append("")
    
    report.append("=" * 70)
    report.append("")
    
    # RANKING 3: PHYSICAL-LAWS-FOCUSED
    report.append("## RANKING 3: PHYSICAL-LAWS-FOCUSED (Observational Realism)")
    report.append("")
    report.append("### Methodology")
    report.append(f"- Emergent Laws Quality: {ranking_weights_physical_laws.get('emergent_laws_quality', 0.30)*100:.0f}%")
    report.append(f"- Friedmann Consistency: {ranking_weights_physical_laws.get('friedmann_consistency', 0.25)*100:.0f}%")
    report.append(f"- CMB Anomaly Match: {ranking_weights_physical_laws.get('cmb_anomaly_match', 0.20)*100:.0f}%")
    report.append(f"- Lock-in Efficiency: {ranking_weights_physical_laws.get('lockin_efficiency', 0.15)*100:.0f}%")
    report.append(f"- Quantum Field Realism: {ranking_weights_physical_laws.get('quantum_field_realism', 0.10)*100:.0f}%")
    report.append("")
    report.append("### Top 3 Models")
    report.append("")
    
    for rank, model in enumerate(top_3_physical, 1):
        i_def = model['i_definition']
        score = model['physical_laws_total_score']
        orig = df_ei[df_ei['i_definition'] == i_def].iloc[0]
        
        report.append(f"**{rank}. {i_def.upper()}** (Score: {score:.2f}/100)")
        report.append(f"   - Emergent Laws: {model.get('emergent_laws_quality', 50):.1f}, Friedmann: {model.get('friedmann_consistency', 50):.1f}")
        report.append("")
    
    report.append("=" * 70)
    report.append("")
    
    # RECOMMENDATION
    best_stability = top_3_stability[0]['i_definition']
    best_complexity = top_3_complexity[0]['i_definition']
    best_physical = top_3_physical[0]['i_definition']
    
    report.append("## 🏆 FINAL RECOMMENDATION")
    report.append("")
    
    report.append(f"**Stability-Focused Winner: `{best_stability}`**")
    report.append(f"**Complexity-Focused Winner: `{best_complexity}`**")
    report.append(f"**Physical-Laws-Focused Winner: `{best_physical}`**")
    report.append("")
    report.append("### Which to choose?")
    report.append("")
    report.append(f"✅ **For OBSERVATIONAL VALIDATION: USE `{best_physical}`**")
    report.append("   - Best match with Planck 2018 cosmology")
    report.append("   - Realistic emergent laws")
    report.append("   - CMB anomaly reproduction")
    report.append("")
    report.append(f"✅ **For TQE theory validation: USE `{best_complexity}`**")
    report.append("   - More complex, life-compatible universes")
    report.append("   - Information-driven structure formation")
    report.append("")
    report.append(f"⚠️ For maximum stability: USE `{best_stability}`")
    report.append("   - More stable configurations")
    report.append("")
    report.append("### USAGE RECOMMENDATION")
    report.append("")
    report.append(f"For Planck-consistent simulations:")
    report.append("```python")
    report.append(f'I_DEFINITION_MODE = "{best_physical}"')
    report.append("```")
    report.append("")
    report.append(f"For TQE complexity studies:")
    report.append("```python")
    report.append(f'I_DEFINITION_MODE = "{best_complexity}"')
    report.append("```")
    report.append("")
    report.append("=" * 70)
    report.append(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_text = "\n".join(report)
    with open(os.path.join(output_dir, "recommendation_report.md"), 'w') as f:
        f.write(report_text)
    print("   ✅ recommendation_report.md")
    
    # Print best models from all THREE rankings
    print("\n" + "="*70)
    print("🏆 TRIPLE RANKING RESULTS")
    print("="*70)
    print(f"Stability-Focused Winner: {best_stability} (Score: {top_3_stability[0]['stability_total_score']:.2f}/100)")
    print(f"Complexity-Focused Winner: {best_complexity} (Score: {top_3_complexity[0]['complexity_total_score']:.2f}/100)")
    print(f"Physical-Laws-Focused Winner: {best_physical} (Score: {top_3_physical[0]['physical_laws_total_score']:.2f}/100)")
    
    print(f"\n✨ RECOMMENDATION:")
    print(f"  • For OBSERVATIONAL VALIDATION: Use {best_physical}")
    print(f"  • For TQE complexity studies: Use {best_complexity}")
    print(f"  • For maximum stability: Use {best_stability}")
    
    print("="*70)
    
    print("\n✅ Triple Ranking System Complete!")

