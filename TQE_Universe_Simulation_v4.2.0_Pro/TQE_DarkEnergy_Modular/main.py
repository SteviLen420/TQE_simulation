# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Stefan Len
#
# ==========================================================================================
# main.py - Main Entry Point
# ==========================================================================================
# TQE–ΛSim: Main entry point for the TQE Dark Energy Coupling Simulation
# ==========================================================================================

from .pipeline import run_automatic_tqe_darkenergy_pipeline

def main():
    """
    TQE Dark Energy Coupling Pipeline - Main Entry Point
    
    THEORY OF THE QUESTION OF EXISTENCE (TQE):
    ──────────────────────────────────────────
    Central Question: "Why do stable, complexity-permitting physical laws exist at all?"
    
    TQE Hypothesis: Stable physical laws emerge from the coupling of vacuum energy
                    fluctuations (E) with an information-theoretic orientation
                    parameter (I).
    
    PIPELINE OBJECTIVE:
    ───────────────────
    Test whether I affects dark energy density evolution in our universe:
    
        P'(ψ) = P(ψ) · f(E,I)  where  f(E,I) = exp(-(E-E_c)²/(2σ²)) · (1 + α·I)
    
    METHODOLOGY:
    ────────────
    Compare 4 rival models against Planck/Pantheon+/BOSS observations:
    
        1. Covariant E-pressure: ρ_DE = ρ_Λ·exp(-α·E·(1-I))  [E+I coupling]
        2. Uniform w(I): w_DE = w₀ + w_I·I(a)
        3. Geometric: ρ_DE = ρ_Λ·exp(β₀·F[I,∇I,∂I])
        4. Null (ΛCDM): ρ_DE = ρ_Λ, w = -1  [control]
    
    FALSIFIABLE PREDICTIONS:
    ────────────────────────
    If TQE is correct:
        • S₈ parameter differs between E-only and E+I modes
        • CMB anomalies show non-random statistical signatures
        • Matter power spectrum P(k) exhibits scale-dependent features
    """
    
    print("="*80)
    print("🚀 TQE–ΛSim: Dark Energy Coupling Pipeline v4.2.0 PRO + BUGFIX")
    print("   Theory of the Question of Existence (TQE)")
    print("   CRITICAL UPDATE: 2025-10-29 (16 bug fixes, TQE-compliant I-parameter)")
    print("="*80)
    print("   I-parameter: ENERGY INFORMATION CONTENT (I = |dE/da| / (E + |dE/da|))")
    print("   E-only vs E+I: NOW properly distinguished (was identical!)")
    print("="*80)
    print("   Testing 4 cosmological models:")
    print("   1. Covariant E-pressure: ρ_DE = ρ_Λ·exp(-α·E·(1-I))  [E+I coupling]")
    print("   2. Uniform w(I): w_DE = w₀ + w_I·I(a)")
    print("   3. Geometric: ρ_DE = ρ_Λ·exp(β₀·F[I,∇I,∂I])")
    print("   4. Null model: Pure ΛCDM (w=-1, baseline)")
    print("="*80)
    print()
    
    # Run automatic pipeline
    try:
        results = run_automatic_tqe_darkenergy_pipeline()
        print("\n🎉 Pipeline completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()

