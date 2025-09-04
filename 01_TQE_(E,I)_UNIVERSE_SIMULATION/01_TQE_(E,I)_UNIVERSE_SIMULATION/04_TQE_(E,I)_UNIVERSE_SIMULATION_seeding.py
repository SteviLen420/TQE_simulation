seeding.py
# ===================================================================================
# Run-level and per-universe seeding for reproducibility across the whole pipeline.
# - One "master_seed" per run (fixed if config.ENERGY.seed is set; otherwise auto)
# - Deterministic per-universe seeds derived via SeedSequence.spawn(N)
# - Persist seeds to seeds_run.json and seeds_universes.csv in the run directory.
#
# Author: Stefan Len
# ===================================================================================

from typing import Dict, Optional, List
import os, json, pathlib
import numpy as np

from config import ACTIVE
from io_paths import resolve_output_paths

SEEDS_JSON_NAME = "seeds_run.json"
SEEDS_CSV_NAME  = "seeds_universes.csv"

def _auto_master_seed() -> int:
    """Create an automatic master seed (stable 64-bit int) for this run if none is provided."""
    # Using NumPy SeedSequence to draw a 64-bit value; reproducible for a given process state.
    # This is only called when config.ENERGY.seed is None, i.e., user wants a fresh run seed.
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])

def _spawn_universe_seeds(master_seed: int, n: int) -> np.ndarray:
    """Deterministically derive per-universe seeds from the master seed."""
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n)
    # Convert each child SeedSequence into a single 64-bit seed integer
    uni_seeds = np.empty(n, dtype=np.uint64)
    for i, cs in enumerate(children):
        # generate_state(...) is deterministic for a given child; we store the 64-bit integer
        uni_seeds[i] = cs.generate_state(1, dtype=np.uint64)[0]
    return uni_seeds.astype(np.uint64)

def _save_seeds_files(run_dir: pathlib.Path, master_seed: int, uni_seeds: np.ndarray):
    """Write seeds_run.json and seeds_universes.csv for audit/replay."""
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "master_seed": int(master_seed),
        "num_universes": int(len(uni_seeds)),
        "universe_seeds_uint64": [int(x) for x in uni_seeds.tolist()],
    }
    with open(run_dir / SEEDS_JSON_NAME, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # CSV
    import pandas as pd
    df = pd.DataFrame({
        "universe_id": np.arange(len(uni_seeds), dtype=np.int64),
        "seed_uint64": uni_seeds.astype(np.uint64).astype(object),  # keep full 64-bit
    })
    df.to_csv(run_dir / SEEDS_CSV_NAME, index=False)

def _load_seeds_files(run_dir: pathlib.Path) -> Optional[Dict]:
    """Return dict with seeds if both files exist and are valid; else None."""
    run_dir = pathlib.Path(run_dir)
    jpath = run_dir / SEEDS_JSON_NAME
    cpath = run_dir / SEEDS_CSV_NAME
    if not jpath.exists() or not cpath.exists():
        return None
    try:
        with open(jpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
        uni = np.array(meta.get("universe_seeds_uint64", []), dtype=np.uint64)
        if uni.size == 0:
            return None
        out = {
            "master_seed": int(meta.get("master_seed")),
            "universe_seeds": uni,
            "json_path": str(jpath),
            "csv_path": str(cpath),
        }
        return out
    except Exception:
        return None

def load_or_create_run_seeds(active: Dict = ACTIVE) -> Dict:
    """
    Load existing seeds for this run (if present), otherwise create and persist them.
    Returns:
      {
        "master_seed": int,
        "universe_seeds": np.ndarray[uint64] shape (N,),
        "json_path": str,
        "csv_path": str,
        "paths": resolve_output_paths(active)
      }
    """
    paths = resolve_output_paths(active)
    run_dir = pathlib.Path(paths["primary_run_dir"])
    n = int(active["ENERGY"]["num_universes"])

    # Try to load existing files (so multiple stages share the same seeds)
    loaded = _load_seeds_files(run_dir)
    if loaded is not None:
        loaded["paths"] = paths
        # If loaded N mismatches, we still return what is on disk (truth source for the run)
        return loaded

    # Else create now
    cfg_seed = active["ENERGY"].get("seed", None)
    master_seed = int(cfg_seed) if cfg_seed is not None else _auto_master_seed()
    uni_seeds = _spawn_universe_seeds(master_seed, n)
    _save_seeds_files(run_dir, master_seed, uni_seeds)

    return {
        "master_seed": master_seed,
        "universe_seeds": uni_seeds,
        "json_path": str(run_dir / SEEDS_JSON_NAME),
        "csv_path":  str(run_dir / SEEDS_CSV_NAME),
        "paths": paths,
    }

def universe_rng(seed_uint64: int) -> np.random.Generator:
    """Convenience: create a NumPy Generator from a stored 64-bit seed."""
    return np.random.default_rng(np.uint64(seed_uint64))

def universe_rngs(uni_seeds: np.ndarray) -> List[np.random.Generator]:
    """Convenience: return list of Generators, one per universe."""
    return [universe_rng(int(s)) for s in uni_seeds.tolist()]
