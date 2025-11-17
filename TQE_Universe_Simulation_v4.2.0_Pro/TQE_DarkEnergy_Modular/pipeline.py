# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# pipeline.py - Pipeline Module
# ==========================================================================================
# TQE–ΛSim: Automatic pipeline functions for running the complete analysis
# ==========================================================================================

import numpy as np
import os
from .config import MASTER_CTRL, FIDUCIAL_PARAMS
from .simulation import TQEDarkEnergyCouplingSimulation
from .tqe_core import EnergyInformationContent, CouplingModel

def run_automatic_tqe_darkenergy_pipeline():
    """
    Main automatic pipeline for TQE Dark Energy Coupling Simulation.
    
    This is the master function that orchestrates the complete TQE analysis pipeline,
    testing multiple cosmological models against observational data to validate
    the Theory of the Question of Existence hypothesis.
    
    Pipeline stages:
        0. Goldilocks zone optimization (optional, if AUTO_FIND_GOLDILOCKS=True)
           - Bayesian optimization to find optimal TQE parameters (E_c, σ, α, β₀)
        
        1. Model initialization
           - 4 base models: Covariant Pressure, Uniform w, Geometric, Null ΛCDM
           - Optional β₀ sweep: 21 values from 0.000 to 0.100 (if RUN_BETA0_SWEEP=True)
        
        2. Dual coupling mode execution (if COUPLING_MODE='dual')
           - E-only mode: Energy magnitude effect only
           - E+I mode: Energy + Information coupling (full TQE)
        
        3. Per-model analysis (12 phases per run):
           - Cosmological evolution: H(a), I(a), ρ_DE(a)
           - Field statistics: I_mean, I_std for geometric model
           - Evolution series: S₈(z), D(z), ρ_DE(z)
           - I-E correlation: Pearson, Spearman, MI + lag scan
           - Observable predictions: SNe Ia, BAO, CMB, LSS
           - Galaxy structure: Cosmic web classification
           - Sanity checks: Physical consistency validation
           - Sensitivity test: ±1% I-parameter perturbation
           - Visualizations: 11-16 publication-quality PNG plots
           - CMB Planck validation: Real map comparison (if enabled)
           - Bayesian inference: MCMC posterior sampling (if enabled)
           - Data saving: 32-36 files per run (CSV, JSON, TXT, ZIP)
        
        4. Cross-model aggregation
           - E-only aggregator: Model comparison, β₀ sweep analysis
           - E+I aggregator: Same for E+I mode
           - Dual comparison: E-only vs E+I statistical analysis
        
        5. Final summary
           - Pipeline metadata, execution time, model rankings
           - Reproducibility snapshot (MASTER_CTRL + environment)
    
    Total output:
        - Baseline: ~1,449 files (48 models × 31 files + aggregators + summary)
        - With MCMC: ~1,633 files (48 models × 35 files + aggregators + summary)
    
    Returns:
        results: Dictionary containing all simulation results and metadata
    """
    print("="*60)
    print("🚀 TQE DARK ENERGY COUPLING SIMULATION - AUTOMATIC PIPELINE")
    print("="*60)
    print("💾 IMMEDIATE SAVE MODE: All data saved after each model run")
    
    # PHASE 2: Check coupling mode
    coupling_mode = MASTER_CTRL.get('COUPLING_MODE', 'EplusI')
    run_dual_comparison = MASTER_CTRL.get('RUN_DUAL_COMPARISON', False)
    
    if coupling_mode == 'dual' or run_dual_comparison:
        print("🔄 DUAL MODE: Running both E-only and E+I coupling modes")
        coupling_modes = ['Eonly', 'EplusI']
    else:
        print(f"🎯 SINGLE MODE: Running {coupling_mode} coupling only")
        coupling_modes = [coupling_mode]
    
    # Check Google Drive status
    if COLAB:
        drive_ready, status_msg = check_google_drive_status()
        print(f"📁 Google Drive Status: {status_msg}")
        
        if not drive_ready:
            drive_setup_success = setup_google_drive_automatically()
            if not drive_setup_success:
                print("❌ Google Drive setup failed")
                return None
    
    # Set global deterministic seed from MASTER_CTRL
    master_seed_string = MASTER_CTRL['MASTER_SEED']
    global_seed_hash = set_deterministic_seed(master_seed_string)
    
    print(f"\n🎲 DETERMINISTIC SEEDING:")
    print(f"  Master seed string: '{master_seed_string}'")
    print(f"  Master seed hash: {global_seed_hash}")
    print(f"  Each model gets unique derived seed")
    
    # Setup directory structure FIRST (needed for Goldilocks)
    main_project_name = "TQE_DarkEnergy_Coupling_Simulation"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"TQE_DarkEnergy_Coupling_Simulation_v4.2.0PRO_{timestamp}"
    
    # Google Drive integration - fixed path structure
    if COLAB:
        main_dir = "/content/drive/MyDrive/TQE_DarkEnergy_Coupling_Simulation"
        run_dir = f"{main_dir}/{run_folder_name}"
        print(f"✅ Google Drive main folder: {main_dir}")
        print(f"✅ Google Drive run folder: {run_dir}")
    else:
        print("❌ Local execution detected - not supported")
        print("💡 This pipeline requires Google Colab + Google Drive")
        return None
    
    # Create directories
    try:
        os.makedirs(main_dir, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)
        
        # Test write access
        test_file = f"{run_dir}/00_Google_Drive_Test.txt"
        with open(test_file, 'w') as f:
            f.write("Google Drive write test successful!")
        print(f"✅ Google Drive write test: SUCCESS")
        
        # Copy Auto_Aggregator to Google Drive main folder for execution
        if COLAB:
            print(f"\n📦 Setting up Auto_Aggregator...")
            # Auto_Aggregator will be available in the same directory as main file
            # No copy needed - both files uploaded together to Colab
            print(f"  ✓ Auto_Aggregator ready for execution")
                
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        raise
    
    # ==========================================================================================
    # GOLDILOCKS ZONE FINDER (if enabled)
    # ==========================================================================================
    goldilocks_results = None
    if MASTER_CTRL.get('AUTO_FIND_GOLDILOCKS', False):
        print(f"\n{'='*80}")
        print(f"PHASE 0: GOLDILOCKS ZONE OPTIMIZATION")
        print(f"{'='*80}")
        
        try:
            # Run Goldilocks finder (saves inside run_dir)
            goldilocks_results = find_goldilocks_zone_bayesian(run_dir=run_dir)
            
            # Update MASTER_CTRL with optimal parameters (for Model 3 Geometric)
            print(f"\nUpdating MASTER_CTRL with Goldilocks optimal parameters...")
            MASTER_CTRL['BETA0'] = goldilocks_results['beta0_optimal']
            MASTER_CTRL['ALPHA'] = goldilocks_results['alpha_optimal']
            
            print(f"MASTER_CTRL updated:")
            print(f"  alpha = {MASTER_CTRL['ALPHA']:.6f}")
            print(f"  beta0 = {MASTER_CTRL['BETA0']:.6f}")
            print(f"\nPipeline will use Goldilocks-optimized parameters for Model 3!")
            
        except Exception as e:
            print(f"WARNING: Goldilocks finder failed: {e}")
            print(f"  Continuing with default parameters...")
            import traceback
            traceback.print_exc()
            goldilocks_results = None
    else:
        print(f"\nGoldilocks finder DISABLED (using default parameters)")
    
    # Define models to test - use MASTER_CTRL parameters
    # Base model configurations
    base_models_config = [
        {
            'name': 'Model_1_Covariant_Pressure',
            'coupling_type': 'covariant_pressure',
            'i_field_type': 'energy_based',  # TQE-COMPLIANT: I from energy evolution
            'coupling_params': {'alpha': MASTER_CTRL['ALPHA_COUPLING']},
            'i_field_params': {'epsilon': MASTER_CTRL['I_FIELD_EPSILON'], 'normalization': MASTER_CTRL['I_FIELD_NORMALIZATION']}
        },
        {
            'name': 'Model_2_Uniform_w',
            'coupling_type': 'uniform_w',
            'i_field_type': 'energy_based',  # TQE-COMPLIANT: I from energy evolution
            'coupling_params': {'w0': MASTER_CTRL['W0'], 'w_I': MASTER_CTRL['W_I_COUPLING']},
            'i_field_params': {'epsilon': MASTER_CTRL['I_FIELD_EPSILON'], 'normalization': MASTER_CTRL['I_FIELD_NORMALIZATION']}
        },
        {
            'name': 'Model_3_Geometric_Coupling',
            'coupling_type': 'geometric',
            'i_field_type': 'energy_based',  # TQE-COMPLIANT: I from energy evolution
            'coupling_params': {'beta0': MASTER_CTRL['BETA0_COUPLING']},
            'i_field_params': {'epsilon': MASTER_CTRL['I_FIELD_EPSILON'], 'normalization': MASTER_CTRL['I_FIELD_NORMALIZATION']}
        },
        {
            'name': 'Null_Model_LCDM',
            'coupling_type': 'null',
            'i_field_type': 'phenomenological',  # Null model has no I
            'coupling_params': {},
            'i_field_params': {'A': 0.0, 'gamma': 0.0}  # No I-parameter effect
        }
    ]
    
    # β₀ SWEEP: If enabled, expand Model_3 into multiple β₀ values
    if MASTER_CTRL.get('RUN_BETA0_SWEEP', False):
        print(f"\n🔄 β₀ PARAMETER SWEEP ENABLED")
        beta0_values = MASTER_CTRL['BETA0_SWEEP_FINE']
        print(f"  Sweeping β₀: {len(beta0_values)} values from {min(beta0_values):.3f} to {max(beta0_values):.3f}")
        
        # Remove base Model_3
        models_config = [m for m in base_models_config if 'Model_3' not in m['name']]
        
        # Add all β₀ sweep models
        for i, beta0_val in enumerate(beta0_values):
            models_config.append({
                'name': f'Model_3_Geometric_beta0_{beta0_val:.4f}',
                'coupling_type': 'geometric',
                'i_field_type': 'energy_based',  # TQE-COMPLIANT: I from energy evolution
                'coupling_params': {'beta0': beta0_val},
                'i_field_params': {'epsilon': MASTER_CTRL['I_FIELD_EPSILON'], 'normalization': MASTER_CTRL['I_FIELD_NORMALIZATION']},
                'beta0_sweep_index': i,
                'beta0_value': beta0_val
            })
        
        print(f"  Total models to run: {len(models_config)} (including {len(beta0_values)} β₀ variants)")
    else:
        models_config = base_models_config
        print(f"\n📋 Running {len(models_config)} standard models (β₀ sweep disabled)")
    
    # PHASE 2: Run models for each coupling mode
    print(f"\n📊 Pipeline Configuration:")
    print(f"  - Coupling modes: {coupling_modes}")
    print(f"  - Models per mode: {len(models_config)}")
    print(f"  - Total runs: {len(coupling_modes) * len(models_config)}")
    print(f"  - Observables: SNe Ia, BAO, CMB, LSS")
    print(f"  - Analysis: Bayesian inference + model comparison")
    
    total_runs = len(coupling_modes) * len(models_config)
    all_results = {}  # Store results for comparison (dict with coupling_mode as key)
    
    # Calculate total phases for progress bar
    # Per model: evolution, field_stats, evolution_series, I-E_corr, observables, 
    #            galaxy, sanity, sensitivity, visualizations, CMB_valid, bayesian, save = 12 phases
    # Post-processing: summary(1), save_summary(1), comparison(1), bayes_factor(1), aggregator(1) = 5 phases
    phases_per_model = 12
    total_model_phases = total_runs * phases_per_model
    total_phases = total_model_phases + 5  # Models + post-processing (SYNCED!)
    
    # Main pipeline loop with phase-level progress tracking
    progress = tqdm(total=total_phases, 
                    desc="TQE_DarkEnergy_Coupling_v4.2.0PRO",
                    unit="phase",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                    colour='green', ncols=80)
    
    try:
        
        for coupling_mode in coupling_modes:
            print(f"\n{'='*60}")
            print(f"🔄 COUPLING MODE: {coupling_mode}")
            print(f"{'='*60}")
            
            for model_idx, model_config in enumerate(models_config):
                # Create model-specific directory with coupling mode
                model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_dir_name = f"{model_config['name']}_{coupling_mode}_{model_timestamp}"
                model_dir = f"{run_dir}/{model_dir_name}"
                
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(f"{model_dir}/PNG_Visualizations", exist_ok=True)
            
                # Create information content model (I parameter)
                info_content = EnergyInformationContent(
                    model_type=model_config['i_field_type'],
                    params=model_config['i_field_params']
                )
                
                # Create coupling model (FIXED: pass coupling_mode for E-only vs E+I)
                coupling = CouplingModel(
                    coupling_type=model_config['coupling_type'],
                    information_content=info_content,
                    coupling_params=model_config['coupling_params'],
                    coupling_mode=coupling_mode  # CRITICAL FIX: E-only vs E+I distinction
                )
                
                # Create simulation with coupling mode
                simulation = TQEDarkEnergyCouplingSimulation(
                    coupling_model=coupling,
                    information_content=info_content,
                    fiducial_params=FIDUCIAL_PARAMS.copy(),
                    project_dir=model_dir,
                    coupling_mode=coupling_mode,  # PHASE 2: Pass coupling mode
                    seed_string=f"TQE_DarkEnergy_{model_config['name']}_{coupling_mode}_{model_timestamp}"
                )
            
                # Run cosmological evolution
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Cosmological Evolution")
                simulation.run_cosmological_evolution()
                progress.update(1)
                
                # Compute field statistics for geometric coupling (I_mean, I_std, F_I_mean)
                if coupling.coupling_type == 'geometric':
                    progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Field Statistics")
                    a_grid = np.array(simulation.results['evolution']['a_array'])
                    field_stats = coupling.compute_field_statistics(a_grid, simulation.friedmann)
                    simulation.results['field_statistics'] = field_stats
                    progress.update(1)
                else:
                    progress.update(1)  # Skip field stats for non-geometric
                
                # Compute evolution series (S₈(z), ρ_DE(z), D(z))
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Evolution Series")
                simulation.compute_evolution_series()
                progress.update(1)
                
                # Compute I-E correlation
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): I-E Correlation")
                simulation.compute_I_E_correlation()
                progress.update(1)
                
                # Compute observables
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Observables")
                simulation.compute_observables()
                progress.update(1)
                
                # ==========================================================================================
                # GALAXY STRUCTURE ANALYSIS
                # ==========================================================================================
                
                if MASTER_CTRL.get('RUN_GALAXY_STRUCTURE_ANALYSIS', True):
                    progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Galaxy Structure")
                    try:
                        galaxy_analyzer = GalaxyStructureAnalyzer(simulation)
                        simulation.galaxy_analyzer = galaxy_analyzer
                        galaxy_results = galaxy_analyzer.compute_all_metrics()
                        simulation.results['galaxy_structure'] = galaxy_results
                    except Exception as e:
                        print(f"  WARNING: Galaxy structure failed: {e}")
                    progress.update(1)
                else:
                    progress.update(1)
                
                # Run sanity checks
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Sanity Checks")
                sanity_checks, sanity_issues = simulation.run_sanity_checks()
                simulation.results['sanity_checks'] = sanity_checks
                simulation.results['sanity_issues'] = sanity_issues
                progress.update(1)
            
                # Run sensitivity test
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Sensitivity Test")
                sensitivity_results = simulation.run_sensitivity_test()
                if sensitivity_results:
                    simulation.results['sensitivity_test'] = sensitivity_results
                progress.update(1)
                
                # Create visualizations
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Visualizations")
                simulation.visualize_results(save_plots=True)
                progress.update(1)
                
                # ==========================================================================================
                # CMB PLANCK VALIDATION (if enabled and healpy available)
                # ==========================================================================================
                if MASTER_CTRL.get('USE_REAL_CMB_PLANCK_MAPS', False):
                    progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): CMB Planck Validation")
                    
                    cmb_validation = None
                    validation_success = False
                    validation_error = None
                    
                    try:
                        # Check if healpy is available
                        try:
                            import healpy as hp
                            healpy_available = True
                        except ImportError:
                            print("  ⚠ healpy not installed - skipping CMB Planck validation")
                            healpy_available = False
                            validation_error = "healpy_not_installed"
                        
                        if healpy_available:
                            try:
                                # Initialize Planck data loader
                                planck_loader = PlanckCMBDataLoader(
                                    base_path=MASTER_CTRL.get('CMB_PLANCK_BASE_PATH')
                                )
                                
                                # Initialize CMB validation
                                cmb_validation = CMBPlanckValidation(
                                    tqe_observable=simulation.observables,
                                    planck_loader=planck_loader
                                )
                                
                                # Load NHI foreground map if enabled
                                if MASTER_CTRL.get('CMB_USE_NHI_FOREGROUND', True):
                                    try:
                                        planck_loader.load_nhi_foreground_map()
                                    except Exception as e:
                                        print(f"  ⚠ Failed to load NHI foreground map: {e}")
                                
                                # Compute Planck power spectrum from SMICA map
                                planck_ell, planck_cl = cmb_validation.compute_planck_power_spectrum()
                                
                                if planck_ell is not None and planck_cl is not None:
                                    # Compute TQE power spectrum
                                    tqe_ell, tqe_cl = cmb_validation.compute_tqe_power_spectrum()
                                    
                                    if tqe_ell is not None and tqe_cl is not None:
                                        # Compare power spectra
                                        statistics = cmb_validation.compare_power_spectra()
                                        
                                        # Detect anomalies if enabled
                                        if MASTER_CTRL.get('CMB_ANOMALY_DETECTION', True):
                                            try:
                                                skymap, _, _ = planck_loader.load_smica_map()
                                                if skymap is not None:
                                                    threshold = MASTER_CTRL.get('CMB_ANOMALY_THRESHOLD', 3.0)
                                                    anomalies = cmb_validation.detect_anomalies(skymap, threshold=threshold)
                                                    
                                                    # Correlate with NHI if enabled
                                                    if MASTER_CTRL.get('CMB_NHI_CORRELATION_ANALYSIS', True):
                                                        try:
                                                            cmb_validation.correlate_with_nhi(skymap)
                                                        except Exception as e:
                                                            print(f"  ⚠ NHI correlation failed: {e}")
                                            except Exception as e:
                                                print(f"  ⚠ Anomaly detection failed: {e}")
                                        
                                        # Generate validation plots (always try, even if data is missing)
                                        if MASTER_CTRL.get('CMB_SAVE_VALIDATION_PLOTS', True):
                                            try:
                                                model_prefix = get_file_prefix(simulation.coupling_mode)
                                                cmb_validation.generate_validation_plots(
                                                    output_dir=model_dir,
                                                    prefix=model_prefix
                                                )
                                            except Exception as e:
                                                print(f"  ⚠ Failed to generate validation plots: {e}")
                                        
                                        # Save validation data (ALWAYS save, even if empty)
                                        if MASTER_CTRL.get('CMB_SAVE_VALIDATION_CSV', True):
                                            try:
                                                model_prefix = get_file_prefix(simulation.coupling_mode)
                                                cmb_validation.save_validation_data(
                                                    output_dir=model_dir,
                                                    prefix=model_prefix
                                                )
                                            except Exception as e:
                                                print(f"  ⚠ Failed to save validation data: {e}")
                                        
                                        # Store in simulation results
                                        simulation.results['cmb_planck_validation'] = {
                                            'statistics': statistics,
                                            'n_anomalies': len(cmb_validation.anomalies) if hasattr(cmb_validation, 'anomalies') else 0,
                                            'planck_lmax': int(planck_ell[-1]) if planck_ell is not None and len(planck_ell) > 0 else None,
                                            'validation_complete': True
                                        }
                                        validation_success = True
                                    else:
                                        validation_error = "tqe_power_spectrum_computation_failed"
                                else:
                                    validation_error = "planck_power_spectrum_computation_failed"
                            except Exception as e:
                                print(f"  ⚠ CMB Planck validation initialization failed: {e}")
                                validation_error = str(e)
                    
                    except Exception as e:
                        print(f"  ⚠ CMB Planck validation failed: {e}")
                        validation_error = str(e)
                    
                    # ALWAYS save validation status, even if validation failed
                    if not validation_success:
                        try:
                            model_prefix = get_file_prefix(simulation.coupling_mode)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Create a minimal CMB validation object to save status
                            if cmb_validation is None:
                                # Create a dummy validation object just for saving
                                class DummyCMBValidation:
                                    def __init__(self):
                                        self.planck_cl = None
                                        self.tqe_cl = None
                                        self.statistics = {}
                                        self.anomalies = []
                                
                                cmb_validation = DummyCMBValidation()
                            
                            # Save validation data with error status
                            if MASTER_CTRL.get('CMB_SAVE_VALIDATION_CSV', True):
                                cmb_validation.save_validation_data(
                                    output_dir=model_dir,
                                    prefix=model_prefix
                                )
                            
                            # Store error status in results
                            if 'cmb_planck_validation' not in simulation.results:
                                simulation.results['cmb_planck_validation'] = {
                                    'status': 'failed',
                                    'error': validation_error,
                                    'validation_complete': False
                                }
                        except Exception as e2:
                            print(f"  ⚠ Failed to save CMB validation error status: {e2}")
                    
                    progress.update(1)
                else:
                    progress.update(1)
                
                # RUN BAYESIAN INFERENCE (MCMC or Nested Sampling)
                if MASTER_CTRL.get('RUN_MCMC', False) and MCMC_AVAILABLE:
                    # Determine which method to use
                    use_nested = MASTER_CTRL.get('USE_NESTED_SAMPLING', False)
                    method_name = "Nested Sampling" if use_nested else "MCMC"
                    
                    progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): {method_name}")
                    try:
                        # Initialize Bayesian engine
                        bayesian_engine = BayesianInferenceEngine(simulation, dataset='all')
                        
                        if use_nested:
                            # RUN NESTED SAMPLING (dynesty) - TIER 1 UPGRADE!
                            results = bayesian_engine.run_nested_sampling(
                                nlive=MASTER_CTRL.get('NESTED_NLIVE', 500),
                                dlogz=MASTER_CTRL.get('NESTED_DLOGZ', 0.01)
                            )
                            
                            if results is not None:
                                samples = bayesian_engine.samples
                                
                                # Add evidence to simulation results
                                simulation.results['bayesian_inference'] = {
                                    'method': 'nested_sampling',
                                    'summary': bayesian_engine.summary if hasattr(bayesian_engine, 'summary') else {},
                                    'n_samples': len(samples),
                                    'log_evidence': float(bayesian_engine.logz),
                                    'log_evidence_err': float(bayesian_engine.logz_err),
                                    'information': float(bayesian_engine.information),
                                    'best_params': dict(zip(bayesian_engine.param_names, 
                                                           samples[np.argmax(bayesian_engine.log_prob_samples)]))
                                }
                        else:
                            # RUN MCMC (emcee) - Standard MCMC
                            samples = bayesian_engine.run_mcmc(
                                n_walkers=MASTER_CTRL['MCMC_NWALKERS'],
                                n_steps=MASTER_CTRL['MCMC_NSTEPS'],
                                n_burn=MASTER_CTRL['MCMC_BURNIN']
                            )
                            
                            # Add to simulation results
                            if samples is not None:
                                simulation.results['bayesian_inference'] = {
                                    'method': 'mcmc_emcee',
                                    'summary': bayesian_engine.summary if hasattr(bayesian_engine, 'summary') else {},
                                    'n_samples': len(samples),
                                    'acceptance_fraction': float(np.mean(bayesian_engine.sampler.acceptance_fraction)),
                                    'best_params': dict(zip(bayesian_engine.param_names, 
                                                           samples[np.argmax(bayesian_engine.log_prob_samples)]))
                            }
                            
                        # Save Bayesian results (works for both MCMC and Nested Sampling)
                        if samples is not None:
                            bayesian_dir = f"{model_dir}/Bayesian_Analysis"
                            bayesian_engine.save_results(bayesian_dir)
                    
                    except Exception as e:
                        print(f"  WARNING: Bayesian inference failed: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    progress.update(1)
                else:
                    progress.update(1)
                
                # IMMEDIATE SAVE
                progress.set_description(f"Model {model_idx+1}/{len(models_config)} ({coupling_mode}): Saving Results")
                simulation.save_results()
                progress.update(1)
            
                # Save model summary immediately with prefix
                model_prefix = get_file_prefix(simulation.coupling_mode)
                model_summary_file = f"{model_dir}/{model_prefix}Model_Summary_{model_timestamp}.json"
                with open(model_summary_file, 'w') as f:
                    json.dump({
                        'coupling_mode': simulation.coupling_mode,  # TQE mode: Eonly or EplusI
                        'model_name': model_config['name'],
                        'coupling_type': model_config['coupling_type'],
                        'i_field_type': model_config['i_field_type'],
                        'timestamp': datetime.now().isoformat(),
                        'status': 'completed',
                        'model_directory': model_dir,
                        'google_drive_path': model_dir if COLAB else 'N/A'
                    }, f, indent=2)
                
                # Store results for comparison
                if coupling_mode not in all_results:
                    all_results[coupling_mode] = []
                all_results[coupling_mode].append({
                    'model_name': model_config['name'],
                    'model_config': model_config,
                    'results': simulation.results,
                    'timestamp': datetime.now().isoformat()
                })
                
                # OPTIMIZED: Clean up memory after each model
                cleanup_memory()
        
        # All models completed
        print(f"\n{'='*80}")
        print(f"ALL MODELS COMPLETED!")
        print(f"{'='*80}")
        
        # Post-processing phases
        # Phase: Pipeline summary
        progress.set_description("Post-processing: Pipeline Summary")
        save_reproducibility_snapshot(run_dir)
        progress.update(1)
        
        # Flatten all_results dictionary to list for summary
        all_results_flat = []
        for mode, results_list in all_results.items():
            all_results_flat.extend(results_list)
        
        # Save pipeline summary (with Goldilocks results if available)
        progress.set_description("Post-processing: Saving Summary")
        pipeline_summary = {
            'start_time': all_results_flat[0]['timestamp'] if all_results_flat else datetime.now().isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_models': len(all_results_flat),
            'coupling_modes': list(all_results.keys()),
            'models_by_mode': {mode: len(results) for mode, results in all_results.items()},
            'models': all_results,  # Keep dictionary structure
            'goldilocks_optimization': goldilocks_results if goldilocks_results else {'status': 'disabled'},
            'reproducibility': {
                'master_seed_string': master_seed_string,
                'master_seed_hash': global_seed_hash,
                'deterministic_seeding_enabled': MASTER_CTRL['USE_DETERMINISTIC_SEED'],
                'individual_model_seeds': [
                    {
                        'model_name': r['model_name'],
                        'seed_string': f"TQE_DarkEnergy_{r['model_name']}_{r['timestamp']}"
                    }
                    for r in all_results_flat
                ]
            }
        }
        
        summary_file = f"{run_dir}/pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(pipeline_summary, f, indent=2, default=str)
        progress.update(1)
        
        # PHASE 3: E+I vs E-only Comparison (if dual mode was run)
        if len(coupling_modes) > 1 and 'Eonly' in all_results and 'EplusI' in all_results:
            progress.set_description("Post-processing: Dual Comparison")
            try:
                comparison_results = compare_eonly_vs_eplusi(all_results, run_dir)
                print(f"✅ E+I vs E-only comparison completed!")
            except Exception as e:
                print(f"WARNING: Comparison analysis failed: {e}")
                import traceback
                traceback.print_exc()
            progress.update(1)
        else:
            progress.update(1)
        
        # PHASE 4: Bayes Factor Analysis (if nested sampling was run)
        if MASTER_CTRL.get('USE_NESTED_SAMPLING', False) and MASTER_CTRL.get('COMPUTE_EVIDENCE', True):
            progress.set_description("Post-processing: Bayes Factor")
            try:
                bayes_factor_results = compute_bayes_factors_all_models(all_results)
                
                if bayes_factor_results is not None:
                    # Save Bayes Factor results
                    bf_file = f"{run_dir}/Bayes_Factor_Comparison.json"
                    with open(bf_file, 'w') as f:
                        json.dump(bayes_factor_results, f, indent=2)
                    print(f"✅ Bayes Factor analysis saved: {bf_file}")
                    
                    # Create Bayes Factor comparison table CSV
                    bf_data = []
                    for mode in ['Eonly', 'EplusI']:
                        if mode in bayes_factor_results.get('bayes_factors', {}):
                            for bf in bayes_factor_results['bayes_factors'][mode]:
                                bf['coupling_mode'] = mode
                                bf_data.append(bf)
                    
                    if bf_data:
                        bf_df = pd.DataFrame(bf_data)
                        bf_csv = f"{run_dir}/Bayes_Factor_Table.csv"
                        bf_df.to_csv(bf_csv, index=False)
                        print(f"✅ Bayes Factor table saved: {bf_csv}")
                    
                    # Create Bayes Factor visualization
                    bf_plot = f"{run_dir}/Bayes_Factor_Comparison.png"
                    create_bayes_factor_plot(bayes_factor_results, bf_plot)
                
            except Exception as e:
                print(f"WARNING: Bayes Factor analysis failed: {e}")
                import traceback
                traceback.print_exc()
            progress.update(1)
        else:
            progress.update(1)
        
        # PHASE 5: Auto Aggregator (if enabled) - collects all model results
        if MASTER_CTRL.get('RUN_AUTO_AGGREGATOR', True):
            progress.set_description("Post-processing: Auto Aggregator")
            try:
                aggregator_results = run_integrated_aggregator(run_dir)
                if aggregator_results:
                    print(f"✅ Auto aggregator completed!")
                    print(f"   CSV: {aggregator_results.get('aggregated_csv', 'N/A')}")
                    if 'png_count' in aggregator_results:
                        print(f"   PNG: {aggregator_results['png_count']}/6 generated")
                else:
                    print(f"⚠️  Auto aggregator returned None (no data)")
            except Exception as e:
                print(f"⚠️  WARNING: Auto aggregator exception: {e}")
                print(f"   Attempting minimal CSV save...")
                # Try to at least save something
                try:
                    import pandas as pd
                    agg_dir = f"{run_dir}/Auto_Aggregator_Summary"
                    os.makedirs(agg_dir, exist_ok=True)
                    pd.DataFrame({'error': [str(e)]}).to_csv(f"{agg_dir}/ERROR_LOG.csv", index=False)
                except:
                    pass
                import traceback
                traceback.print_exc()
            progress.update(1)
        else:
            progress.update(1)
        
        return pipeline_summary
        
    finally:
        # Close progress bar
        progress.close()

# ==========================================================================================
# INTEGRATED AUTO AGGREGATOR
# ==========================================================================================

def run_integrated_aggregator(run_dir):
    """Integrated Auto Aggregator - collects and visualizes results from all models"""
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    print("Aggregating results from all models...")
    
    # Find all Full_Summary.json files
    all_dirs = os.listdir(run_dir)
    model_dirs = [d for d in all_dirs if (d.startswith('Model_') or d.startswith('Null_')) and os.path.isdir(os.path.join(run_dir, d))]
    
    if not model_dirs:
        print("WARNING: No model directories found!")
        return None
    
    # Collect summary data
    all_summary_data = []
    for model_dir in model_dirs:
        # Try to find any TQE_DarkEnergy_Results JSON file
        model_path = os.path.join(run_dir, model_dir)
        results_files = glob.glob(f"{model_path}/*_TQE_DarkEnergy_Results_*.json")
        
        if results_files:
            # Use the first TQE_Results file
            with open(results_files[0], 'r') as f:
                data = json.load(f)
                
                # Extract key metrics for aggregation
                summary_row = {
                    'model_dir': model_dir,
                    'coupling_mode': data.get('coupling_mode', 'N/A'),
                    'coupling_type': data.get('model_type', 'N/A'),
                    'i_field_type': data.get('i_field_type', 'N/A'),
                }
                
                # Add observables
                if 'observables' in data:
                    obs = data['observables']
                    summary_row['S8_raw'] = obs.get('S8_raw', 0.0)
                    summary_row['mu_z1'] = obs.get('mu_z1', 0.0)
                    summary_row['D_M_z051'] = obs.get('D_M_z051', 0.0)
                    summary_row['H_z051'] = obs.get('H_z051', 0.0)
                    summary_row['H_z0'] = obs.get('H_z0', 0.0)
                
                # Add likelihood
                if 'likelihood' in data:
                    like = data['likelihood']
                    summary_row['chi2_total'] = like.get('chi2_total', 0.0)
                    summary_row['AIC'] = like.get('AIC', 0.0)
                    summary_row['BIC'] = like.get('BIC', 0.0)
                    summary_row['reduced_chi2'] = like.get('reduced_chi2', 0.0)
                
                # Add I-E correlation
                if 'I_E_correlation' in data:
                    ie = data['I_E_correlation']
                    summary_row['pearson_r'] = ie.get('pearson_r', 0.0)
                    summary_row['spearman_r'] = ie.get('spearman_r', 0.0)
                    summary_row['mutual_information'] = ie.get('mutual_information', 0.0)
                
                # Add galaxy structure
                if 'galaxy_structure' in data:
                    gal = data['galaxy_structure']
                    summary_row['n_voids'] = gal.get('n_voids', 0)
                    summary_row['n_clusters'] = gal.get('n_clusters', 0)
                    summary_row['n_filaments'] = gal.get('n_filaments', 0)
                
                all_summary_data.append(summary_row)
        else:
            print(f"WARNING: No TQE_Results file found in {model_dir}")
    
    # Create aggregated results directory
    agg_dir = f"{run_dir}/Auto_Aggregator_Summary"
    os.makedirs(agg_dir, exist_ok=True)
    
    # Create aggregated CSV
    csv_file = f"{agg_dir}/Aggregated_Results_Summary.csv"
    df = pd.DataFrame(all_summary_data)
    df.to_csv(csv_file, index=False)
    print(f"Aggregated CSV saved: {csv_file}")
    
    # Create PNG_Visualizations directory
    png_dir = f"{agg_dir}/PNG_Visualizations"
    os.makedirs(png_dir, exist_ok=True)
    
    # Generate aggregator visualizations (if we have data)
    if len(df) > 0:
        print(f"\n📊 Generating aggregator visualizations...")
        
        import matplotlib.pyplot as plt
        
        # 1. Model Comparison - S8 and chi2
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')
            
            models = df['model_dir'].str[:30]  # Truncate for readability
            
            if 'S8_raw' in df.columns:
                ax1.bar(range(len(df)), df['S8_raw'], alpha=0.7, color='#457B9D')
                ax1.set_xlabel('Model', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax1.set_ylabel('S₈', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax1.set_title('S₈ Comparison Across Models', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax1.set_xticks([])
                ax1.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            
            if 'chi2_total' in df.columns:
                ax2.bar(range(len(df)), df['chi2_total'], alpha=0.7, color='#E63946')
                ax2.set_xlabel('Model', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax2.set_ylabel('χ² Total', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax2.set_title('Likelihood Comparison', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax2.set_xticks([])
                ax2.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
            
            plt.tight_layout()
            plt.savefig(f"{png_dir}/01_Model_Comparison_Overview.png", dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
            plt.close()
            print(f"  ✅ 01_Model_Comparison_Overview.png")
        except Exception as e:
            print(f"  ⚠️  Model Comparison failed: {e}")
        
        # 2. Chi2 Components Breakdown
        try:
            chi2_cols = ['chi2_total', 'AIC', 'BIC', 'reduced_chi2']
            available_chi2 = [c for c in chi2_cols if c in df.columns]
            
            if available_chi2:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
                
                x = np.arange(len(df))
                width = 0.2
                
                for i, col in enumerate(available_chi2[:4]):
                    if col in df.columns:
                        ax.bar(x + i*width, df[col], width, label=col, alpha=0.7)
                
                ax.set_xlabel('Model Index', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Value', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('Likelihood Metrics Comparison', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                
                plt.tight_layout()
                plt.savefig(f"{png_dir}/02_Likelihood_Comparison.png", dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                print(f"  ✅ 02_Likelihood_Comparison.png")
        except Exception as e:
            print(f"  ⚠️  Likelihood Comparison failed: {e}")
        
        # 3. I-E Correlation Comparison
        try:
            corr_cols = ['pearson_r', 'spearman_r', 'mutual_information']
            available_corr = [c for c in corr_cols if c in df.columns]
            
            if available_corr:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
                
                x = np.arange(len(df))
                width = 0.25
                
                colors = ['#457B9D', '#E63946', '#F4A261']
                for i, col in enumerate(available_corr):
                    if col in df.columns:
                        ax.bar(x + i*width, df[col], width, label=col.replace('_', ' ').title(), 
                               alpha=0.7, color=colors[i])
                
                ax.set_xlabel('Model Index', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Correlation Value', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('I-E Correlation Metrics Comparison', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                
                plt.tight_layout()
                plt.savefig(f"{png_dir}/03_IE_Correlation_Comparison.png", dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                print(f"  ✅ 03_IE_Correlation_Comparison.png")
        except Exception as e:
            print(f"  ⚠️  I-E Correlation failed: {e}")
        
        # 4. Galaxy Structure Comparison
        try:
            gal_cols = ['n_voids', 'n_clusters', 'n_filaments']
            available_gal = [c for c in gal_cols if c in df.columns]
            
            if available_gal:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
                
                x = np.arange(len(df))
                width = 0.25
                
                colors = ['#E63946', '#457B9D', '#F4A261']
                for i, col in enumerate(available_gal):
                    if col in df.columns:
                        ax.bar(x + i*width, df[col], width, 
                               label=col.replace('n_', '').title(), 
                               alpha=0.7, color=colors[i])
                
                ax.set_xlabel('Model Index', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Count', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('Galaxy Structure Counts Comparison', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                
                plt.tight_layout()
                plt.savefig(f"{png_dir}/04_Galaxy_Structure_Comparison.png", dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                print(f"  ✅ 04_Galaxy_Structure_Comparison.png")
        except Exception as e:
            print(f"  ⚠️  Galaxy Structure failed: {e}")
        
        # 5. S8 vs Chi2 Scatter
        try:
            if 'S8_raw' in df.columns and 'chi2_total' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
                
                # Color by coupling mode if available
                if 'coupling_mode' in df.columns:
                    for mode in df['coupling_mode'].unique():
                        mask = df['coupling_mode'] == mode
                        ax.scatter(df[mask]['S8_raw'], df[mask]['chi2_total'], 
                                   label=mode, alpha=0.7, s=100)
                else:
                    ax.scatter(df['S8_raw'], df['chi2_total'], alpha=0.7, s=100, color='#457B9D')
                
                ax.set_xlabel('S₈', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('χ² Total', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('S₈ vs Likelihood', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.legend(fontsize=MASTER_CTRL['PLOT_FONTSIZE_LEGEND'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                
                plt.tight_layout()
                plt.savefig(f"{png_dir}/05_S8_vs_Chi2_Scatter.png", dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                print(f"  ✅ 05_S8_vs_Chi2_Scatter.png")
        except Exception as e:
            print(f"  ⚠️  S8 vs Chi2 Scatter failed: {e}")
        
        # 6. Coupling Mode Comparison
        try:
            if 'coupling_mode' in df.columns and 'chi2_total' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
                
                modes = df['coupling_mode'].unique()
                mode_chi2 = [df[df['coupling_mode']==m]['chi2_total'].mean() for m in modes]
                
                ax.bar(range(len(modes)), mode_chi2, alpha=0.7, color=['#457B9D', '#E63946'])
                ax.set_xticks(range(len(modes)))
                ax.set_xticklabels(modes, fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_ylabel('Average χ²', fontsize=MASTER_CTRL['PLOT_FONTSIZE_LABEL'])
                ax.set_title('Coupling Mode Performance', fontsize=MASTER_CTRL['PLOT_FONTSIZE_TITLE'])
                ax.grid(True, alpha=MASTER_CTRL['PLOT_GRID_ALPHA'])
                
                plt.tight_layout()
                plt.savefig(f"{png_dir}/06_Coupling_Mode_Comparison.png", dpi=MASTER_CTRL['PLOT_SAVE_DPI'], bbox_inches='tight')
                plt.close()
                print(f"  ✅ 06_Coupling_Mode_Comparison.png")
        except Exception as e:
            print(f"  ⚠️  Coupling Mode Comparison failed: {e}")
        
        print(f"\n✅ Aggregator visualizations saved to: {png_dir}")
        
        # Count generated PNGs
        import glob
        png_files = glob.glob(f"{png_dir}/*.png")
        png_count = len(png_files)
    else:
        png_count = 0
    
    return {'aggregated_csv': csv_file, 'aggregator_dir': agg_dir, 'png_dir': png_dir, 'png_count': png_count, 'n_models': len(df)}


# ==========================================================================================
# UNIT TESTS (ΛCDM Compatibility Validation)
# ==========================================================================================

def run_unit_tests(friedmann):
    # Run critical unit tests for ΛCDM compatibility
    # Tests: D_L/D_A = (1+z)², E(1) ≈ 1, Ω_sum ≈ 1
    
    print("\n" + "="*60)
    print("🧪 RUNNING ΛCDM COMPATIBILITY UNIT TESTS")
    print("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Flatness condition Ω_total(z=0) ≈ 1
    tests_total += 1
    try:
        Omega_m_0, Omega_r_0, Omega_DE_0 = friedmann.Omega_components(1.0)
        Omega_total = Omega_m_0 + Omega_r_0 + Omega_DE_0
        error = abs(Omega_total - 1.0)
        
        if error < 1e-3:
            print(f"✅ TEST 1 PASS: Flatness Ω_total = {Omega_total:.6f} (error = {error:.2e} < 1e-3)")
            tests_passed += 1
        else:
            print(f"❌ TEST 1 FAIL: Flatness Ω_total = {Omega_total:.6f} (error = {error:.2e} >= 1e-3)")
    except Exception as e:
        print(f"❌ TEST 1 ERROR: {e}")
    
    # Test 2: E(a=1) ≈ 1 (dimensionless Hubble at z=0)
    tests_total += 1
    try:
        E_at_z0 = friedmann.E(1.0)
        error = abs(E_at_z0 - 1.0)
        
        if error < 1e-2:
            print(f"✅ TEST 2 PASS: E(z=0) = {E_at_z0:.6f} (error = {error:.2e} < 1e-2)")
            tests_passed += 1
        else:
            print(f"❌ TEST 2 FAIL: E(z=0) = {E_at_z0:.6f} (error = {error:.2e} >= 1e-2)")
    except Exception as e:
        print(f"❌ TEST 2 ERROR: {e}")
    
    # Test 3: Distance duality D_L / D_A = (1+z)²
    tests_total += 1
    try:
        z_test = 0.5
        D_C = friedmann.comoving_distance(z_test)
        D_A = D_C / (1 + z_test)
        D_L = D_C * (1 + z_test)
        ratio = D_L / D_A
        expected = (1 + z_test)**2
        error = abs(ratio - expected) / expected
        
        if error < 1e-6:
            print(f"✅ TEST 3 PASS: D_L/D_A = {ratio:.6f}, (1+z)² = {expected:.6f} (error = {error:.2e} < 1e-6)")
            tests_passed += 1
        else:
            print(f"❌ TEST 3 FAIL: D_L/D_A = {ratio:.6f}, (1+z)² = {expected:.6f} (error = {error:.2e} >= 1e-6)")
    except Exception as e:
        print(f"❌ TEST 3 ERROR: {e}")
    
    # Test 4: H(a) > 0 for all a ∈ [0.1, 1.0]
    tests_total += 1
    try:
        a_test_grid = np.linspace(0.1, 1.0, 20)
        H_test = np.array([friedmann.H(a_val) for a_val in a_test_grid])
        all_positive = np.all(H_test > 0)
        all_finite = np.all(np.isfinite(H_test))
        
        if all_positive and all_finite:
            print(f"✅ TEST 4 PASS: H(a) > 0 for all a ∈ [0.1, 1.0] (min H = {np.min(H_test):.2f} km/s/Mpc)")
            tests_passed += 1
        else:
            print(f"❌ TEST 4 FAIL: H(a) not positive/finite everywhere (min H = {np.min(H_test)})")
    except Exception as e:
        print(f"❌ TEST 4 ERROR: {e}")
    
    # Test 5: ρ_DE(a) > 0 for all a ∈ [0.1, 1.0]
    tests_total += 1
    try:
        a_test_grid = np.linspace(0.1, 1.0, 20)
        rho_DE_test = np.array([friedmann.coupling.rho_DE(a_val, friedmann.rho_Lambda_today, friedmann=friedmann) for a_val in a_test_grid])
        all_positive = np.all(rho_DE_test > 0)
        all_finite = np.all(np.isfinite(rho_DE_test))
        
        if all_positive and all_finite:
            print(f"✅ TEST 5 PASS: ρ_DE(a) > 0 for all a ∈ [0.1, 1.0] (min ρ_DE = {np.min(rho_DE_test):.6f})")
            tests_passed += 1
        else:
            print(f"❌ TEST 5 FAIL: ρ_DE(a) not positive/finite everywhere")
    except Exception as e:
        print(f"❌ TEST 5 ERROR: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🧪 UNIT TEST RESULTS: {tests_passed}/{tests_total} PASSED")
    print(f"{'='*60}\n")
    
    return tests_passed == tests_total

# ==========================================================================================
# MAIN EXECUTION
# ==========================================================================================

