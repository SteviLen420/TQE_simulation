# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Report generation functions

import os
import time
from typing import Dict
import pandas as pd

def generate_extended_reports(df_metrics: pd.DataFrame, collected_data: Dict, output_dir: str, config: dict):
    """
    PHASE 6: Produce extended markdown report summarizing key findings.
    """
    print("\n" + "="*70)
    print("PHASE 6: EXTENDED ANALYSIS REPORTS")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append("# TQE Analysis Pipeline v4.2.0 PRO — Extended Report")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Target mode: {collected_data['metadata']['target_mode']}")
    lines.append(f"- Source directory: `{collected_data['metadata']['target_path']}`")
    lines.append("")
    lines.append("## Run Inventory")
    lines.append(f"- Total runs analyzed: **{len(df_metrics)}**")
    lines.append(f"- E-only runs: **{collected_data['metadata']['n_eonly_runs']}**")
    lines.append(f"- E+I runs: **{collected_data['metadata']['n_ei_runs']}**")
    lines.append("")
    
    df_ei = df_metrics[df_metrics["run_type"] == "E+I"]
    if len(df_ei) > 0:
        best_stability = df_ei.sort_values("stable_percent", ascending=False).iloc[0]
        best_complexity = df_ei.sort_values("complexity_score", ascending=False).iloc[0]
        if "planck_chi2_reduced" in df_ei.columns:
            best_physical = df_ei.sort_values("planck_chi2_reduced", ascending=True).iloc[0]
        else:
            best_physical = best_stability
        
        lines.append("## Highlighted I-Definitions")
        lines.append(f"- **Stability leader**: `{best_stability['i_definition']}` ({best_stability['stable_percent']:.2f}% stable)")
        lines.append(f"- **Complexity leader**: `{best_complexity['i_definition']}` (score {best_complexity['complexity_score']:.2f})")
        lines.append(f"- **Planck proximity leader**: `{best_physical['i_definition']}` (χ²={best_physical.get('planck_chi2_reduced', 'N/A')})")
        lines.append("")
    
    lines.append("## Artifact Coverage")
    lines.append("- summary_full.json, tqe_runs.csv, Bayesian calibration")
    lines.append("- Planck validation (scatter + χ² bar + CSV export)")
    lines.append("- Life compatibility, entropy volatility, anomaly diagnostics")
    lines.append("- Nested sampling traces, stability sweeps, seed registries")
    lines.append("")
    
    lines.append("## Suggested Follow-ups")
    lines.append("1. Inspect `04_best_model_selection/recommendation_report.md` for model choices.")
    lines.append("2. Review `02_detailed_metrics/all_runs_metrics.csv` for downstream ML.")
    lines.append("3. Compare E-only vs E+I in `01_comparative_analysis/eonly_vs_ei/` if available.")
    lines.append("")
    
    report_path = os.path.join(output_dir, "extended_report.md")
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"   ✅ Extended report written to {report_path}")

