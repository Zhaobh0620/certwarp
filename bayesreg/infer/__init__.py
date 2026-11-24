"""B-line posterior inference (contract C14): Laplace approximation N(u_hat, H^-1) by perturb-and-MAP; the prior-side
samples N(0, Pi_0^-1) of contract C15-9 (eb_count = "gull_mackay")."""
from .laplace import (DIFFUSION_FACTORS, PASS_RATE_THRESHOLD, PCG_MAXITER, PCG_RTOL, PRIOR_BLOCKS,
                      PRIOR_PCG_NOT_CONVERGED, S_GROUP, S_NATIVE, LaplaceConfig, LaplaceSystem, PCGResult, Perturbation,
                      PerturbationSet, PosteriorMaps, PriorSystem, TraceEstimate, curl_gram_apply,
                      default_trace_operators, hutchinson_trace, iter_perturbations, pcg, perturbation_rhs,
                      posterior_maps, prior_perturbation_rhs, rng_record, run_laplace, sample_perturbations,
                      sample_trace, seed_from_run_id, stiffness_apply_scaled, stiffness_sqrt_apply,
                      stiffness_sqrt_spectrum, write_posterior_outputs)

__all__ = ["DIFFUSION_FACTORS", "PASS_RATE_THRESHOLD", "PCG_MAXITER", "PCG_RTOL", "PRIOR_BLOCKS",
           "PRIOR_PCG_NOT_CONVERGED", "S_GROUP", "S_NATIVE", "LaplaceConfig", "LaplaceSystem", "PCGResult",
           "Perturbation", "PerturbationSet", "PosteriorMaps", "PriorSystem", "TraceEstimate", "curl_gram_apply",
           "default_trace_operators", "hutchinson_trace", "iter_perturbations", "pcg", "perturbation_rhs",
           "posterior_maps", "prior_perturbation_rhs", "rng_record", "run_laplace", "sample_perturbations",
           "sample_trace", "seed_from_run_id", "stiffness_apply_scaled", "stiffness_sqrt_apply",
           "stiffness_sqrt_spectrum", "write_posterior_outputs"]
