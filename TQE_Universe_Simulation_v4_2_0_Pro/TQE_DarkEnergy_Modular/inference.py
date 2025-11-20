# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# inference.py - Bayesian Inference Module
# ==========================================================================================
# TQE–ΛSim: Bayesian inference engine for parameter estimation and model comparison
# ==========================================================================================

import os
import numpy as np
from .config import MASTER_CTRL

class BayesianInferenceEngine:
    # Advanced Bayesian parameter estimation with MCMC/Nested Sampling
    # Professional-grade implementation with full posterior analysis
    
    def __init__(self, simulation, dataset='all'):
        self.simulation = simulation
        self.dataset = dataset
        self.param_names = []
        self.param_bounds = []
        self.param_labels = []
        self._setup_parameters()
        
        self.sampler = None
        self.samples = None
        self.log_prob_samples = None
    
    def _setup_parameters(self):
        # Setup free parameters
        coupling_type = self.simulation.coupling.coupling_type
        
        # Use MASTER_CTRL priors (ΛCDM-compatible bounds)
        self.param_names = ['Omega_m', 'H0']
        self.param_bounds = [
            tuple(MASTER_CTRL.get('PRIOR_OMEGA_M', [0.2, 0.4])),
            tuple(MASTER_CTRL.get('PRIOR_H0', [60.0, 75.0]))
        ]
        self.param_labels = [r'$\Omega_m$', r'$H_0$']
        
        if coupling_type == 'covariant_pressure':
            self.param_names.append('alpha')
            self.param_bounds.append(tuple(MASTER_CTRL.get('PRIOR_ALPHA', [0.0, 0.3])))
            self.param_labels.append(r'$\alpha$')
        elif coupling_type == 'uniform_w':
            self.param_names.extend(['w0', 'w_I'])
            self.param_bounds.extend([
                tuple(MASTER_CTRL.get('PRIOR_W0', [-1.3, -0.7])),
                tuple(MASTER_CTRL.get('PRIOR_W_I', [-0.5, 0.5]))
            ])
            self.param_labels.extend([r'$w_0$', r'$w_I$'])
        elif coupling_type == 'geometric':
            self.param_names.append('beta0')
            self.param_bounds.append(tuple(MASTER_CTRL.get('PRIOR_BETA0', [0.0, 0.3])))
            self.param_labels.append(r'$\beta_0$')
    
    def log_prior(self, params):
        for param, bounds in zip(params, self.param_bounds):
            if not (bounds[0] <= param <= bounds[1]):
                return -np.inf
        return 0.0
    
    def log_likelihood(self, params):
        try:
            param_dict = dict(zip(self.param_names, params))
            
            # Update parameters
            if 'Omega_m' in param_dict:
                self.simulation.friedmann.Omega_m = param_dict['Omega_m']
            if 'H0' in param_dict:
                self.simulation.friedmann.H0 = param_dict['H0']
            if 'alpha' in param_dict:
                self.simulation.coupling.alpha = param_dict['alpha']
            if 'w0' in param_dict:
                self.simulation.coupling.w0 = param_dict['w0']
            if 'w_I' in param_dict:
                self.simulation.coupling.w_I = param_dict['w_I']
            if 'beta0' in param_dict:
                self.simulation.coupling.beta0 = param_dict['beta0']
            
            # Recompute
            self.simulation.friedmann.compute_evolution_grid()
            likelihood_results = self.simulation.observables.compute_likelihood()
            chi2_total = likelihood_results['chi2_total']
            
            return -0.5 * chi2_total
        except (ValueError, KeyError, RuntimeError) as e:
            # Return -inf if likelihood computation fails (invalid parameter space)
            return -np.inf
    
    def log_posterior(self, params):
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(params)
    
    def run_mcmc(self, n_walkers=50, n_steps=5000, n_burn=1000):
        if not MCMC_AVAILABLE:
            print("❌ emcee not available")
            return None
        
        n_dim = len(self.param_names)
        print(f"🔬 MCMC: {n_walkers} walkers × {n_steps} steps, {n_dim}D")
        
        initial = np.array([(b[0] + b[1]) / 2.0 for b in self.param_bounds])
        pos = initial + 1e-4 * np.random.randn(n_walkers, n_dim)
        
        self.sampler = emcee.EnsembleSampler(n_walkers, n_dim, self.log_posterior)
        
        print("  🔥 Burn-in...")
        pos, _, _ = self.sampler.run_mcmc(pos, n_burn, progress=True)
        self.sampler.reset()
        
        print("  ⚙️ Production...")
        self.sampler.run_mcmc(pos, n_steps, progress=True)
        
        self.samples = self.sampler.get_chain(flat=True)
        self.log_prob_samples = self.sampler.get_log_prob(flat=True)
        
        print(f"✅ MCMC done: {len(self.samples)} samples, acceptance={np.mean(self.sampler.acceptance_fraction):.3f}")
        self._compute_summary()
        return self.samples
    
    def _compute_summary(self):
        self.summary = {}
        for i, name in enumerate(self.param_names):
            s = self.samples[:, i]
            self.summary[name] = {
                'mean': np.mean(s),
                'median': np.median(s),
                'std': np.std(s),
                'q16': np.percentile(s, 16),
                'q84': np.percentile(s, 84)
            }
        print("\n📊 POSTERIOR ESTIMATES:")
        for name in self.param_names:
            st = self.summary[name]
            print(f"  {name}: {st['median']:.4f} ± {st['std']:.4f} [{st['q16']:.4f}, {st['q84']:.4f}]")
    
    def make_corner_plot(self, save_path=None):
        if not MCMC_AVAILABLE or self.samples is None:
            return
        import corner
        fig = corner.corner(self.samples, labels=self.param_labels, quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 10})
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Corner plot: {save_path}")
        plt.close()
        return fig
    
    def compute_ic(self):
        idx_best = np.argmax(self.log_prob_samples)
        chi2_best = -2.0 * self.log_prob_samples[idx_best]
        n_params = len(self.param_names)
        likelihood_results = self.simulation.observables.compute_likelihood()
        n_data = likelihood_results['n_data']
        
        AIC = chi2_best + 2 * n_params
        BIC = chi2_best + n_params * np.log(n_data)
        D_bar = np.mean(-2.0 * self.log_prob_samples)
        DIC = 2 * D_bar - chi2_best
        
        print(f"\n📊 INFO CRITERIA: AIC={AIC:.2f}, BIC={BIC:.2f}, DIC={DIC:.2f}")
        return {'AIC': AIC, 'BIC': BIC, 'DIC': DIC, 'n_params': n_params, 'n_data': n_data}
    
    def run_nested_sampling(self, nlive=500, dlogz=0.01):
        """
        Run Nested Sampling with dynesty for Bayesian evidence calculation.
        
        Nested Sampling advantages over MCMC:
        - Computes evidence log Z (for Bayes Factor)
        - Better for multimodal posteriors
        - More robust parameter estimation
        
        Args:
            nlive: Number of live points (higher = more accurate evidence)
            dlogz: Evidence tolerance (stopping criterion)
        
        Returns:
            results: Nested sampling results with samples and log evidence
        """
        try:
            from dynesty import NestedSampler
            from dynesty import plotting as dyplot
        except ImportError:
            print("❌ dynesty not available - attempting installation...")
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dynesty', '--quiet'])
                from dynesty import NestedSampler
                from dynesty import plotting as dyplot
                print("✅ dynesty successfully installed")
            except Exception as e:
                print(f"❌ dynesty installation failed: {e}")
                return None
        
        n_dim = len(self.param_names)
        print(f"🔬 NESTED SAMPLING: {nlive} live points, {n_dim}D parameter space")
        print(f"  Bound: {MASTER_CTRL.get('NESTED_BOUND', 'multi')}, Sample: {MASTER_CTRL.get('NESTED_SAMPLE', 'rwalk')}")
        
        # Prior transform (uniform priors)
        def prior_transform(u):
            """Transform unit cube to physical parameter space."""
            params = np.zeros(n_dim)
            for i, bounds in enumerate(self.param_bounds):
                params[i] = bounds[0] + (bounds[1] - bounds[0]) * u[i]
            return params
        
        # Likelihood function (already defined as log_likelihood)
        def likelihood_func(params):
            return self.log_likelihood(params)
        
        # Initialize sampler
        sampler = NestedSampler(
            likelihood_func,
            prior_transform,
            n_dim,
            nlive=nlive,
            bound=MASTER_CTRL.get('NESTED_BOUND', 'multi'),
            sample=MASTER_CTRL.get('NESTED_SAMPLE', 'rwalk')
        )
        
        # Run nested sampling
        print("  🔥 Running nested sampling...")
        sampler.run_nested(dlogz=dlogz, print_progress=True)
        
        # Extract results
        results = sampler.results
        
        # Samples (equal-weighted)
        weights = np.exp(results['logwt'] - results['logz'][-1])
        self.samples = results['samples']
        self.weights = weights
        self.log_prob_samples = results['logl']
        
        # Evidence and information
        self.logz = results['logz'][-1]
        self.logz_err = results['logzerr'][-1]
        self.information = results['information'][-1]
        
        print(f"✅ NESTED SAMPLING COMPLETE!")
        print(f"  log Z = {self.logz:.2f} ± {self.logz_err:.2f}")
        print(f"  Information H = {self.information:.2f} nats")
        print(f"  Samples: {len(self.samples)}")
        
        # Compute summary statistics from weighted samples
        self._compute_summary_weighted()
        
        return results
    
    def _compute_summary_weighted(self):
        """Compute summary statistics from weighted nested sampling samples."""
        self.summary = {}
        for i, name in enumerate(self.param_names):
            s = self.samples[:, i]
            w = self.weights / np.sum(self.weights)  # Normalize weights
            
            # Weighted statistics
            mean_weighted = np.sum(w * s)
            var_weighted = np.sum(w * (s - mean_weighted)**2)
            std_weighted = np.sqrt(var_weighted)
            
            # Quantiles (use weighted percentile)
            sorted_indices = np.argsort(s)
            cumsum = np.cumsum(w[sorted_indices])
            q16 = s[sorted_indices[np.searchsorted(cumsum, 0.16)]]
            q50 = s[sorted_indices[np.searchsorted(cumsum, 0.50)]]
            q84 = s[sorted_indices[np.searchsorted(cumsum, 0.84)]]
            
            self.summary[name] = {
                'mean': float(mean_weighted),
                'median': float(q50),
                'std': float(std_weighted),
                'q16': float(q16),
                'q84': float(q84)
            }
        
        print("\n📊 NESTED SAMPLING POSTERIOR ESTIMATES:")
        for name in self.param_names:
            st = self.summary[name]
            print(f"  {name}: {st['median']:.4f} ± {st['std']:.4f} [{st['q16']:.4f}, {st['q84']:.4f}]")
    
    def compute_bayes_factor(self, logz_reference):
        """
        Compute Bayes Factor relative to reference model.
        
        Bayes Factor interpretation (Kass & Raftery 1995):
        - log BF > 5: Very strong evidence
        - log BF > 3: Strong evidence
        - log BF > 1: Substantial evidence
        - log BF < 1: Weak evidence
        
        Args:
            logz_reference: log evidence of reference model (e.g., ΛCDM)
        
        Returns:
            dict with Bayes Factor and interpretation
        """
        if not hasattr(self, 'logz'):
            print("⚠️ Nested sampling not run yet, cannot compute Bayes Factor")
            return None
        
        log_BF = self.logz - logz_reference
        BF = np.exp(log_BF)
        
        # Interpretation
        if log_BF > 5:
            interpretation = "VERY STRONG evidence for this model"
        elif log_BF > 3:
            interpretation = "STRONG evidence for this model"
        elif log_BF > 1:
            interpretation = "SUBSTANTIAL evidence for this model"
        elif log_BF > -1:
            interpretation = "WEAK evidence (models comparable)"
        elif log_BF > -3:
            interpretation = "SUBSTANTIAL evidence AGAINST this model"
        else:
            interpretation = "STRONG evidence AGAINST this model"
        
        result = {
            'log_evidence_model': float(self.logz),
            'log_evidence_reference': float(logz_reference),
            'log_bayes_factor': float(log_BF),
            'bayes_factor': float(BF),
            'interpretation': interpretation
        }
        
        print(f"\n🎯 BAYES FACTOR ANALYSIS:")
        print(f"  log Z (this model):  {self.logz:.2f} ± {self.logz_err:.2f}")
        print(f"  log Z (reference):   {logz_reference:.2f}")
        print(f"  log BF:              {log_BF:+.2f}")
        print(f"  BF:                  {BF:.2e}")
        print(f"  → {interpretation}")
        
        return result
    
    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Samples CSV
        pd.DataFrame(self.samples, columns=self.param_names).to_csv(
            f"{output_dir}/mcmc_samples.csv", index=False)
        
        # Summary JSON
        if hasattr(self, 'summary'):
            with open(f"{output_dir}/mcmc_summary.json", 'w') as f:
                json.dump(self.summary, f, indent=2)
        
        # Corner plot
        self.make_corner_plot(f"{output_dir}/corner_plot.png")
        
        # IC
        ic = self.compute_ic()
        with open(f"{output_dir}/information_criteria.json", 'w') as f:
            json.dump(ic, f, indent=2)
        
        # Nested sampling evidence (if available)
        if hasattr(self, 'logz'):
            evidence_data = {
                'log_evidence': float(self.logz),
                'log_evidence_err': float(self.logz_err),
                'information': float(self.information),
                'method': 'nested_sampling_dynesty',
                'nlive': MASTER_CTRL.get('NESTED_NLIVE', 500),
                'dlogz': MASTER_CTRL.get('NESTED_DLOGZ', 0.01)
            }
            with open(f"{output_dir}/nested_sampling_evidence.json", 'w') as f:
                json.dump(evidence_data, f, indent=2)
            print(f"✓ Evidence saved: log Z = {self.logz:.2f} ± {self.logz_err:.2f}")
        
        print(f"✅ Bayesian results saved: {output_dir}")


# ==========================================================================================
# GALAXY STRUCTURE ANALYZER
# ==========================================================================================

