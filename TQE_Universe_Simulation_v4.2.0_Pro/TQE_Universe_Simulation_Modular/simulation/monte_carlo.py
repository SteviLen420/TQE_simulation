# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# Monte Carlo simulation module
#
import numpy as np
import pandas as pd
import multiprocessing
from tqdm.auto import tqdm
from ..config.master_ctrl import MASTER_CTRL
from ..core.pipeline_context import PipelineContext
from ..core.physics_engine import PhysicsEngine

def _run_single_universe(args):
    """Multiprocessing worker function."""
    uni_seed, X_c_low, X_c_high, universe_id, config = args # Added config to args
    
    # Restore seed for reproducibility in subprocess
    rng_uni = np.random.default_rng(uni_seed)
    np.random.seed(uni_seed)
    
    # Verify seed determinism (optional debug check)
    if config.get("VERBOSE", False) and universe_id < 3:  # Only check first few universes
        test_val = rng_uni.random()
        print(f"[SEED-VERIFY] Universe {universe_id}: seed={uni_seed}, test_value={test_val:.6f}")

    # Re-initialize a local PhysicsEngine instance
    # NOTE: The PhysicsEngine automatically handles the correct use of config/RNG.
    local_physics = PhysicsEngine(config, rng_uni) 
    
    # Sample universe parameters (E+I coupling computed here)
    uni_params = local_physics.sample_universe()
    E, I, X = uni_params["E"], uni_params["I"], uni_params["X"]

    # Simulation using pre-computed E+I coupling (X)
    # This ensures the entire lock-in dynamics use the correct E+I interaction
    # FIX: Pass E and local_physics for I-parameter tracking (horizon_entropy fix)
    stable, lockin, stable_epoch, lock_epoch = simulate_lock_in(
        X, config["LOCKIN_EPOCHS"], config,
        sigma0=config["EXP_NOISE_BASE"],
        alpha=config.get("SIGMA_ALPHA", 1.0),
        X_c_low=X_c_low,
        X_c_high=X_c_high,
        rng=rng_uni,
        E=E,
        physics_engine=local_physics  # Pass local physics engine for I tracking
    )

    rec = {
        "universe_id": universe_id, "seed": uni_seed, "E": E, "I": I, "X": X,
        "stable": stable, "lockin": lockin, "stable_epoch": stable_epoch, "lock_epoch": lock_epoch
    }
    pre_pair = {"universe_id": universe_id, "E": E, "I": I, "X": X}
    return rec, pre_pair



def run_mc(ctx: PipelineContext, X_c_low: float = None, X_c_high: float = None, num_universes: int = None) -> pd.DataFrame:
    """Monte Carlo run for one pipeline phase, uses multiprocessing pool for local execution."""
    
    # Preserve/Restore numpy random state across parallel calls
    prev_state = np.random.get_state()
    try:
        n_runs = num_universes if num_universes is not None else ctx.config["NUM_UNIVERSES"]
        
        # Check if we need to adjust stability thresholds
        max_adjustments = ctx.config.get("MAX_STABILITY_ADJUSTMENTS", 10)
        adjustment_iterations = 0
        
        while adjustment_iterations < max_adjustments:
            # Use the context's master RNG for generating *per-universe* seeds
            universe_seeds = [int(ctx.rng.integers(0, 2**32 - 1)) for _ in range(n_runs)]
            
            # Pass the full configuration to the worker process
            tasks = [(seed, X_c_low, X_c_high, i, ctx.config) for i, seed in enumerate(universe_seeds)] 
            
            results = []
            # Use multiprocessing for local execution
            use_mp = ctx.config.get("USE_MULTIPROCESSING", True) and len(tasks) > 10
            if use_mp:
                max_workers = ctx.config.get("MAX_WORKERS", None)
                n_workers = max_workers if max_workers else min(multiprocessing.cpu_count() or 2, len(tasks), 8)
                try:
                    # Use multiprocessing for local execution
                    with multiprocessing.Pool(processes=n_workers) as pool:
                        results = list(tqdm(
                            pool.imap(_run_single_universe, tasks),
                            total=len(tasks),
                            desc=f"TQE Simulating Universes ({n_workers} workers)"
                        ))
                except Exception as e:
                    print(f"[MP][WARN] Parallel execution unavailable, falling back to sequential: {e}")
                    for task in tqdm(tasks, desc="TQE Simulating Universes (sequential fallback)"):
                        results.append(_run_single_universe(task))
            else:
                # Sequential fallback (for debugging or small batches)
                for task in tqdm(tasks, desc="TQE Simulating Universes (sequential)"):
                    results.append(_run_single_universe(task))
            
            rows = [res[0] for res in results]
            pre_pairs = [res[1] for res in results]

            df_out = pd.DataFrame(rows)
            
            # Check if we need to adjust thresholds
            if ctx.config.get("ADJUST_STABILITY_THRESHOLDS", False) and adjustment_iterations < max_adjustments - 1:
                ctx.config = adjust_stability_thresholds(df_out, ctx.config)
                adjustment_iterations += 1
                
                # OPTIMIZED: Early stopping if stability rates converged
                total_universes = len(df_out)
                stable_count = df_out['stable'].sum()
                lockin_count = df_out['lock_epoch'].ge(0).sum()
                
                current_unstable_rate = (total_universes - stable_count) / total_universes
                current_stable_rate = stable_count / total_universes
                current_lockin_rate = lockin_count / max(stable_count, 1)
                
                target_unstable = ctx.config.get("TARGET_UNSTABLE_RATE", 0.60)
                target_stable = ctx.config.get("TARGET_STABLE_RATE", 0.40)
                target_lockin = ctx.config.get("TARGET_LOCKIN_RATE", 0.60)
                
                # Early stopping if converged
                if ctx.config.get("STABILITY_EARLY_STOP", True) and adjustment_iterations > 2:
                    tolerance = ctx.config.get("STABILITY_TOLERANCE", 0.02)
                    stable_converged = abs(current_stable_rate - target_stable) < tolerance
                    lockin_converged = abs(current_lockin_rate - target_lockin) < tolerance
                    
                    if stable_converged and lockin_converged:
                        if ctx.config.get("VERBOSE", True):
                            print(f"[STABILITY] Converged after {adjustment_iterations} iterations")
                        break
                
                if (abs(current_unstable_rate - target_unstable) < 0.05 and 
                    abs(current_stable_rate - target_stable) < 0.05 and 
                    abs(current_lockin_rate - target_lockin) < 0.05):
                    print(f"[STABILITY ADJUSTMENT] Target rates achieved after {adjustment_iterations} iterations")
                    break
            else:
                break
        
        # Save per-universe seed and pre-fluctuation pairs using context methods
        ctx.save_csv(pd.DataFrame({"universe_id": np.arange(len(df_out)), "seed": universe_seeds}),
                     os.path.join(ctx.paths["AGGREGATE_DIR"], "universe_seeds.csv"))
        ctx.save_csv(pd.DataFrame(pre_pairs),
                     os.path.join(ctx.paths["AGGREGATE_DIR"], "pre_fluctuation_pairs.csv"))
        
        return df_out
    finally:
        np.random.set_state(prev_state)



def phase_01_monte_carlo(ctx: PipelineContext, X_c_low: float = None, X_c_high: float = None, num_universes: int = None) -> tuple[pd.DataFrame, float, float]:
    """
    Phase 1: Monte Carlo Simulation with INTEGRATED Goldilocks Calibration.
    
    NEW LOGIC (simplified, integrated):
    1. Generate NUM_UNIVERSES universes (E+I coupling)
    2. Compute Goldilocks zone FROM THESE SAME UNIVERSES
    3. Save Goldilocks plot
    4. Return df with Goldilocks parameters
    
    No separate calibration step! Everything happens with the same universe set.
    """
    n_runs = num_universes if num_universes is not None else ctx.config["NUM_UNIVERSES"]
    
    # Determine I-definition name for E-only vs E+I
    if ctx.config.get("PIPELINE_VARIANT", "full") == "energy_only":
        i_def = "energy_only"
    else:
                i_def = ctx.config.get("I_DEFINITION_MODE", "kl_shannon")
                
    print(f"\n[PHASE 1] Monte Carlo Simulation + Bayesian Goldilocks: {i_def} ({n_runs} universes total)")
    
    # STEP 1: BAYESIAN ADAPTIVE GOLDILOCKS OPTIMIZATION
    # Intelligently finds optimal Goldilocks zone using a fraction of the total budget
    calibration_fraction = 0.30  # 30% for Bayesian exploration
    calibration_budget = max(30, int(n_runs * calibration_fraction))  # Minimum 30 universes for GP
    simulation_budget = n_runs - calibration_budget  # Remaining budget for full simulation
    
    # Safety check: ensure at least 20 universes for full simulation
    if simulation_budget < 20:
        calibration_budget = n_runs - 20
        simulation_budget = 20
    
    print(f"[PHASE 1] Budget allocation: {calibration_budget} (Bayesian) + {simulation_budget} (full sim) = {n_runs} total")
    
    X_c_low, X_c_high, X_peak, X_peak_std, df_cal = bayesian_adaptive_goldilocks(ctx, total_budget=calibration_budget)
    
    print(f"[PHASE 1] Goldilocks zone discovered: X_peak={X_peak:.2f}±{X_peak_std:.2f}, Window=[{X_c_low:.2f}, {X_c_high:.2f}]")
    
    # STEP 2: RUN FULL SIMULATION with discovered Goldilocks zone
    print(f"[PHASE 1] Running full simulation with {simulation_budget} universes in discovered zone")
    df = run_mc(ctx, X_c_low=X_c_low, X_c_high=X_c_high, num_universes=simulation_budget)
    
    print(f"[PHASE 1] Full simulation complete: {len(df)} universes, {df['stable'].mean():.1%} stable")
    
    return df, X_c_low, X_c_high



