# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Summary data loading functions

import os
import json
import glob
from typing import Optional, Dict
import pandas as pd
from ..core.path_setup import smart_find_file

def load_summary_json(run_dir: str) -> Optional[Dict]:
    """
    Intelligently load summary JSON from a run directory.
    Searches multiple locations and filename patterns automatically.
    """
    # Use smart file finder with multiple patterns
    summary_file = smart_find_file(
        run_dir, 
        filename_patterns=["summary_full.json", "summary.json", "*summary*.json"],
        recursive=True
    )
    
    if summary_file:
        try:
            with open(summary_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {summary_file}: {e}")
            return None
    
    # Nothing found
    print(f"⚠️  WARNING: No summary JSON found in {os.path.basename(run_dir)}")
    return None


def load_tqe_runs_csv(run_dir: str) -> Optional[pd.DataFrame]:
    """
    Intelligently load TQE runs CSV from a run directory.
    Searches multiple locations and filename patterns automatically.
    """
    # Use smart file finder with multiple patterns
    runs_file = smart_find_file(
        run_dir,
        filename_patterns=["tqe_runs.csv", "runs.csv", "*run*.csv"],
        recursive=True
    )
    
    if runs_file:
        # Filter out bayesian/calibration files
        if "bayesian" in runs_file.lower() or "calibration" in runs_file.lower():
            # Try to find another file
            all_run_files = glob.glob(os.path.join(run_dir, "**", "*run*.csv"), recursive=True)
            filtered = [f for f in all_run_files 
                       if "bayesian" not in f.lower() and "calibration" not in f.lower()]
            if filtered:
                runs_file = sorted(filtered, key=os.path.getmtime, reverse=True)[0]
            else:
                return None  # No valid runs file found
        
        try:
            return pd.read_csv(runs_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {runs_file}: {e}")
    
    # Not found (this is OK - not all runs have trajectory data)
    return None

