# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Bayesian data loading functions

from typing import Optional
import pandas as pd
from ..core.path_setup import smart_find_file

def load_bayesian_calibration_csv(run_dir: str) -> Optional[pd.DataFrame]:
    """
    Intelligently load Bayesian calibration CSV from a run directory.
    Searches multiple locations and filename patterns automatically.
    """
    # Use smart file finder with multiple patterns
    bayesian_file = smart_find_file(
        run_dir,
        filename_patterns=[
            "bayesian_calibration_*.csv",
            "bayesian*.csv", 
            "calibration*.csv",
            "*bayesian*.csv"
        ],
        recursive=True
    )
    
    if bayesian_file:
        try:
            return pd.read_csv(bayesian_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {bayesian_file}: {e}")
    
    # Not found (this is OK - not all runs have Bayesian data)
    return None

