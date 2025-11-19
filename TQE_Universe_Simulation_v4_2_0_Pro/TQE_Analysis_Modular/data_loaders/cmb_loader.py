# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# CMB data loading functions

from typing import Optional
import pandas as pd
from ..core.path_setup import smart_find_file

def load_cmb_coldspots(run_dir: str, i_def: str) -> Optional[pd.DataFrame]:
    """Load CMB cold spots summary CSV."""
    coldspot_file = smart_find_file(
        run_dir,
        filename_patterns=[f"cmb_coldspots_summary_{i_def}.csv", "*coldspots*.csv"],
        recursive=True
    )
    if coldspot_file:
        try:
            return pd.read_csv(coldspot_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {coldspot_file}: {e}")
    return None


def load_cmb_aoe(run_dir: str, i_def: str) -> Optional[pd.DataFrame]:
    """Load CMB Axis of Evil summary CSV."""
    aoe_file = smart_find_file(
        run_dir,
        filename_patterns=[f"cmb_aoe_summary_{i_def}.csv", "*aoe*.csv"],
        recursive=True
    )
    if aoe_file:
        try:
            return pd.read_csv(aoe_file)
        except Exception as e:
            print(f"⚠️  WARNING: Could not parse {aoe_file}: {e}")
    return None

