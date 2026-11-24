"""EB under sigma_domain support / support_shared (work package SWLI; contract C04-6 gate 11, C15-5 appended note; draft
SWL 2026-10-31).  The learn CLI switch and its record are tested in solve/test_run_pair_sigma_domain.py; the brain EB
path is covered by the existing EB goldens (learn/test_eb_count.py).

- support_counts of a ModelState equals the SupportLevel's N_c bit for bit;
- sigma_prior_support: support -> one (a_c, b_c) per structure channel on the pairs with a non-empty native support (the
  others left out of that channel), support_shared -> exactly one (a_S, b_S); D29 counts N_c / N_S (median);
- the driver uses it when the pairs carry a support (and records sigma_domain, theta);
- experiments/eta/probe.py is frozen (not changed here; the report names the gap for the lead).
"""
from dataclasses import replace

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.learn import EBConfig, PairStatistics, run_population_eb, sigma_prior, sigma_prior_support
from bayesreg.model import support_level
from bayesreg.model.support import support_counts

from ..model import _swl_cases as cases
from .test_population import DIM, fake_pairs, p2_eta, solve_ready


def stats(pair_id, sigma_post, *, domain="support", n_c=(500., 300., 0.), n_q=1000., theta=.05):
    support = dict(sigma_domain=domain, support_threshold=theta, N_q=n_q, N_c=list(n_c),
                   N_S=float(sum(x for x in n_c if x > 0)), empty_support_channels=[c+1 for c, x in enumerate(n_c)
                                                                                   if x == 0])
    return PairStatistics(pair_id=pair_id, n_free=300, n_curl=300, n_q=n_q, kq_quadratic=1., curl_quadratic=1.,
                          kq_trace=0., kq_trace_stderr=0., curl_trace=0., curl_trace_stderr=0., tau_hat=(1.,)*4,
                          tau_var=(0.,)*4, tau_den=(1.,)*4, sigma_used=tuple(sigma_post), sigma_post=tuple(sigma_post),
                          n_samples=16, n_converged=16, n_passed=16, pass_rate=1., pass_rate_low=False, support=support)


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_support_counts_equal_the_level_counts(domain):
    grid, state, _, _ = cases.template()
    level, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    counts = support_counts(st)
    assert counts["N_c"] == list(level.n_c) and counts["N_q"] == level.n_q and counts["N_S"] == level.n_s()
    assert counts["sigma_domain"] == domain and counts["empty_support_channels"] == [3]
    assert support_counts(state) is None


def test_support_prior_per_channel_with_exclusions():
    rows = [stats("p0", (.01, .02, .03, None)), stats("p1", (.012, .025, None, None), n_c=(400., 0., 0.)),
            stats("p2", (.011, .021, .032, None)), stats("p3", (.013, .019, .028, None))]
    prior = sigma_prior_support(rows, EBConfig(eb_count="dim"))
    assert prior["sigma_domain"] == "support" and prior["support_threshold"] == .05
    params = prior["parameters"]
    assert [p["parameter"] for p in params] == ["I", 1, 2, 3]
    assert params[2]["n_T"] == 3 and params[2]["pairs"] == ["p0", "p2", "p3"]
    assert prior["excluded_empty_support"]["2"] == ["p1"] and params[3]["n_T"] == 0
    assert params[1]["n_T"] == 4 and params[1]["a"] > 0                       # moment matching on 4 pairs
    m, v = np.mean([.02, .025, .021, .019]), np.var([.02, .025, .021, .019], ddof=1)
    assert params[1]["mean"] == pytest.approx(m) and params[1]["a"] == pytest.approx(m*m/v+2)
    assert params[3]["a"] == 0. and "not applicable" in params[3]["rule"]
    assert params[1]["count_median"] == np.median([500., 400., 500., 500.])
    # the intensity channel is the brain rule with N_q
    brain = sigma_prior([[s.sigma_post[0]] for s in rows], [s.n_q for s in rows], EBConfig(eb_count="dim"))
    assert brain["channels"][0]["a"] == params[0]["a"] and brain["channels"][0]["b"] == params[0]["b"]


def test_support_shared_prior_is_one_parameter_with_n_s():
    rows = [stats("p0", (.01, .02, .02, None), domain="support_shared"),
            stats("p1", (.012, .03, .03, None), domain="support_shared", n_c=(600., 200., 0.)),
            stats("p2", (.011, None, None, None), domain="support_shared", n_c=(0., 0., 0.)),
            stats("p3", (.013, .025, .025, None), domain="support_shared")]
    config = EBConfig(eb_count="dim", sigma_prior="pseudo_count", kappa0=.1)
    prior = sigma_prior_support(rows, config)
    params = prior["parameters"]
    assert [p["parameter"] for p in params] == ["I", "S"]                     # exactly one (a_S, b_S)
    assert params[1]["pairs"] == ["p0", "p1", "p3"] and prior["excluded_empty_support"]["S"] == ["p2"]
    n_s = np.median([800., 800., 800.])
    assert params[1]["count_median"] == n_s
    assert params[1]["n0"] == pytest.approx(.1*n_s) and params[1]["share_n0_over_N"] == pytest.approx(.1)
    assert params[1]["sigma0_sq"] == pytest.approx(np.mean([.02, .03, .025]))


def test_support_prior_refuses_mixed_domains():
    rows = [stats("p0", (.01, .02, .03, None)), stats("p1", (.01, .02, .03, None), domain="support_shared")]
    with pytest.raises(InputValidationError):
        sigma_prior_support(rows)
    with pytest.raises(InputValidationError):
        sigma_prior_support([stats("p0", (.01, .02, .03, None), theta=.05), stats("p1", (.01, .02, .03, None),
                                                                                    theta=.1)])


def test_driver_matches_the_prior_on_the_support(tmp_path):
    ids = [p.pair_id for p in fake_pairs(3)]

    def statistics(outcome, *, pair_id, config, seed, backend):
        k = ids.index(pair_id)
        return stats(pair_id, (.01+.001*k, .02+.002*k, .03, None)), dict(seed=seed)
    result = run_population_eb(fake_pairs(3), solve_ready, eta0=p2_eta(), config=replace(DIM, max_rounds=1),
                               run_id="t", out_dir=tmp_path, statistics=statistics)
    prior = result.sigma_prior
    assert prior["sigma_domain"] == "support" and [p["parameter"] for p in prior["parameters"]] == ["I", 1, 2, 3]
    assert prior["pairs"] == [p.pair_id for p in fake_pairs(3)]


def test_multilevel_map_records_the_support_and_keeps_v():
    from bayesreg.learn import multilevel_map
    from bayesreg.solve import Caps, SolverConfig, eta_temporary
    from ..solve import _swl_synthetic as S
    _, levels, _ = S.four_levels_structure()
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)
    config = SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(2, 1), sigma_domain="support_shared")
    outcome = multilevel_map(list(levels), eta, solver_config=config)
    assert outcome.usable and outcome.state.sigma_domain == "support_shared"
    record = outcome.record
    assert record["sigma_policy"]["sigma_domain"] == "support_shared" and len(record["support_levels"]) == 4
    assert record["sigma_used"][3] is None and outcome.sigma_post[3] is None
    assert outcome.state.support_v_sha256 == record["support_levels"][3]["v_sha256"]
    assert support_counts(outcome.state)["N_S"] == record["support_levels"][3]["N_S"]


def test_fallback_prior_names_the_run_domain(tmp_path):
    """Review SWLI minor 7: a non-brain run without any usable sigma_post writes the brain-form record without a prior,
    marked with the run's sigma_domain and theta (never read back as brain); a brain run's record is unchanged."""
    def no_sigma(outcome, *, pair_id, config, seed, backend):
        return replace(stats(pair_id, (.01, .02, .03, None), domain="support_shared"), sigma_post=None), {}
    result = run_population_eb(fake_pairs(2), solve_ready, eta0=p2_eta(), config=replace(DIM, max_rounds=1),
                               run_id="t", statistics=no_sigma)
    prior = result.sigma_prior
    assert prior["sigma_domain"] == "support_shared" and prior["support_threshold"] == .05
    assert "sigma_domain_fallback" in prior and prior["pairs"] == []
    given = run_population_eb(fake_pairs(2), solve_ready, eta0=p2_eta(), config=replace(DIM, max_rounds=1),
                              run_id="t", statistics=no_sigma, sigma_domain="support_shared", support_threshold=.05)
    assert given.sigma_prior["sigma_domain"] == "support_shared"
    with pytest.raises(InputValidationError):
        run_population_eb(fake_pairs(2), solve_ready, eta0=p2_eta(), config=replace(DIM, max_rounds=1), run_id="t",
                          statistics=no_sigma, sigma_domain="support", support_threshold=.05)

    def brain_no_sigma(outcome, *, pair_id, config, seed, backend):
        return replace(stats(pair_id, (.01, .02, .03, None)), sigma_post=None, support=None), {}
    brain = run_population_eb(fake_pairs(2), solve_ready, eta0=p2_eta(), config=replace(DIM, max_rounds=1),
                              run_id="t", statistics=brain_no_sigma)
    assert "sigma_domain" not in brain.sigma_prior and "sigma_domain_fallback" not in brain.sigma_prior
