"""Population learning of the prior parameters eta (contract C15): empirical Bayes over the B-line EB training
pairs, with the leak guard of C15-1 / D23 / D24; the effective-count option eb_count = "gull_mackay" of C15-9
(the default with P2 since BZ 10-30; "dim" = C15-1 is the explicit option).  `population` is pure numpy (no torch);
`cli` is the entry point for real runs (python -m bayesreg.learn.cli)."""
from .population import (B12_NOTE, B17_NOTE, DEFAULT_EB_COUNT, EB_COUNTS, EB_CRITERION_MET, EB_FAILED, EB_ROUND_CAP,
                         EB_STATUSES, EB_WORDING, EB_WORDING_GM, GM_FALLBACK, GM_FORM, LAPLACE_ENDINGS, MAX_ROUNDS,
                         PARAMETRIZATIONS, REL_TOL, TRAINING_SOURCE, EBConfig, EBPair, EBPairRefused, EBResult, EBUpdateFailed,
                         GaussStatistics, GMStatistics, MapOutcome, PairStatistics, admit_eb_pairs,
                         alpha_nug_coupling, alpha_nugget, check_structure_models, eb_update, eta_components,
                         gauss_point_statistics, gm_precision, gm_tau2, gull_mackay_update,
                         inverse_gamma_from_moments, inverse_gamma_from_pseudo_counts, inverse_gamma_moments,
                         multilevel_map, outcome_statistics, pair_statistics, precision_update, prior_round_seed,
                         prior_statistics, pseudo_count_form, read_eb_pairs, relative_changes, rho_weights,
                         round_seed, run_population_eb, sigma_prior, tau2_update, check_sigma_prior_domain,
                         sigma_prior_support)
from .population import __all__ as __all__   # noqa: F401
