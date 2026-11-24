"""bayesreg.learn.population against contract C15 (update formulas, P1/P2/P3, tau_0 fixed, rounds and the 5%
criterion, sigma prior) on hand-made statistics and a linear Gaussian toy; the Gauss-point statistics against
the operators of bayesreg.infer.laplace (B7-1) and N_C against the curl rank (B11).  No certificate, no data/."""
from dataclasses import replace

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.three_d.grid import make_grid_3d

from bayesreg.geom import gauss_geometry, stiffness_apply
from bayesreg.infer.laplace import curl_gram_apply, stiffness_apply_scaled
from bayesreg.learn import population as pop
from bayesreg.learn import (EB_CRITERION_MET, EB_FAILED, EB_ROUND_CAP, EBConfig, EBPair, MapOutcome, PairStatistics,
                            alpha_nugget, eb_update, gauss_point_statistics, run_population_eb)
from bayesreg.solve import Eta

from ..model._gauss27 import make_state  # DEF: the C05-2 model ("gauss27", the default before DEF), explicit
from ..model._helpers import aniso, feasible_field

TAU2 = (1., .25, .36, .16)
DIM = EBConfig(eb_count="dim")        # the C15-1 update (explicit since EBD: the default is gull_mackay, BZ 10-30)


def stats(pair_id="p", *, n=300, kq=(2., .5), curl=(1.5, .5), tau_hat=(1., 2., 3., 4.), tau_var=(.5, .5, .5, .5),
          tau_den=(10., 20., 30., 40.), conv=16, passed=16, sigma_post=(.01, .02), n_q=1000.):
    return PairStatistics(pair_id=pair_id, n_free=n, n_curl=n, n_q=n_q, kq_quadratic=kq[0], curl_quadratic=curl[0],
                          kq_trace=kq[1] if conv else float("nan"), kq_trace_stderr=.01,
                          curl_trace=curl[1] if conv else float("nan"), curl_trace_stderr=.01, tau_hat=tau_hat,
                          tau_var=tau_var if passed >= 2 else None, tau_den=tau_den, sigma_used=(.01, .02),
                          sigma_post=sigma_post, n_samples=16, n_converged=conv, n_passed=passed,
                          pass_rate=passed/16, pass_rate_low=passed/16 < .8)


def p2_eta(nu=2., tau2=TAU2):
    return Eta(alpha_nugget(tau2), nu, tau2)


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------

def test_config_defaults_are_contract_c15():
    config = EBConfig()
    assert (config.parametrization, config.learn_tau0, config.max_rounds, config.rel_tol) == ("P2", False, 3, .05)
    assert config.eb_count == "gull_mackay"                    # C15-9 / D45 default (BZ 10-30, EBD)
    assert config.n_samples == 16 and config.laplace.n_samples == 16 and config.laplace.rtol == 1e-3
    assert config.alpha_nug_factor == 1e-2 and config.sigma_prior == "moment_match" and config.kappa0 is None
    assert config.learned(4) == ["nu", "tau_1", "tau_2", "tau_3"]
    assert EBConfig(eb_count="dim", learn_tau0=True).learned(4) == ["nu", "tau_0", "tau_1", "tau_2", "tau_3"]
    assert EBConfig(eb_count="dim", parametrization="P1").learned(3) == ["alpha", "nu", "tau_0", "tau_1", "tau_2"]
    assert EBConfig(eb_count="dim", parametrization="P3").learned(3) == ["alpha", "nu", "tau_1", "tau_2"]
    record = config.record()
    assert "B12" in record["note"] and record["n_curl"].startswith("N_C = 3 N_int")
    for bad in (dict(parametrization="P4"), dict(parametrization="P1", learn_tau0=False),
                dict(parametrization="P3", learn_tau0=True), dict(sigma_prior="pseudo_count"),
                dict(kappa0=1.), dict(n_samples=1), dict(rel_tol=0.), dict(max_rounds=0), dict(variance_ddof=2),
                dict(sigma0_statistic="mode")):
        for eb_count in ("dim", "gull_mackay"):
            with pytest.raises(InputValidationError):
                EBConfig(**bad, eb_count=eb_count)


def test_alpha_nugget_components_and_relative_changes():
    assert alpha_nugget(TAU2) == pytest.approx(1e-2)           # min_k 1/tau_k^2 = 1 (tau_0 = 1)
    assert alpha_nugget((1., 4.)) == pytest.approx(.25e-2)
    eta = Eta(.01, 2., (1., .25))
    assert pop.eta_components(eta) == dict(alpha=.01, nu=2., tau_0=1., tau_1=.5)
    new = Eta(.01, 2.2, (1., .36))
    changes, largest = pop.relative_changes(eta, new, ["nu", "tau_1"])
    assert changes == pytest.approx(dict(nu=.1, tau_1=.2)) and largest == pytest.approx(.2)   # tau as a std (C05)


# ----------------------------------------------------------------------------------------------
# Update formulas (C15-1/4)
# ----------------------------------------------------------------------------------------------

def test_precision_and_tau_updates_by_hand():
    value, num, den = pop.precision_update([300, 300], [2., 3.], [.5, .5])
    assert (num, den) == (600., 6.) and value == 100.
    value, num, den = pop.tau2_update([1.5, 2.5], [10., 30.])
    assert value == pytest.approx(.1) and (num, den) == (4., 40.)
    with pytest.raises(pop.EBUpdateFailed):
        pop.precision_update([3], [0.], [0.])
    with pytest.raises(InputValidationError):
        pop.precision_update([], [], [])


def test_p2_update_nu_tau_alpha_nugget_and_fixed_tau0():
    a, b = stats("a"), stats("b", curl=(2.5, .5), tau_hat=(2., 4., 6., 8.))
    eta = p2_eta()
    new, record = eb_update(eta, [a, b], DIM)
    assert new.nu == pytest.approx(600/5.)                     # sum N_C / sum (quad + trace)
    for k in (1, 2, 3):
        expected = (a.tau_hat[k]+a.tau_var[k]+b.tau_hat[k]+b.tau_var[k])/(a.tau_den[k]+b.tau_den[k])
        assert new.tau2[k] == pytest.approx(expected, rel=1e-15)
    assert new.tau2[0] == eta.tau2[0]                          # tau_0 fixed, bitwise (C15-4, D28)
    assert record["candidates"]["tau_0"]["tau2"] == pytest.approx((1+.5+2+.5)/20)     # recorded, not used
    assert record["not_learned"]["tau_0"] == 1.
    assert new.alpha == alpha_nugget(new.tau2) and record["alpha_source"] == "alpha_nug(tau_new)"
    assert new.alpha == eta.alpha                              # min_k 1/tau_k^2 still at tau_0 = 1: unchanged
    assert record["candidates"]["alpha"]["value"] == pytest.approx(600/5.)   # EM candidate recorded only
    # a tau_k above tau_0 moves the nugget (recomputed every round from the current tau, C05-6)
    big = stats("c", tau_hat=(1., 2., 3., 400.), tau_den=(10., 20., 30., 40.))
    new, _ = eb_update(eta, [big], DIM)
    assert new.tau2[3] > 1 and new.alpha == pytest.approx(1e-2/new.tau2[3])


def test_p1_p3_and_learned_tau0():
    a = stats("a")
    eta = Eta(.5, 2., TAU2)
    new, record = eb_update(eta, [a], EBConfig(eb_count="dim", parametrization="P3"))
    assert new.alpha == pytest.approx(300/2.5) and record["alpha_source"] == "EM update" and new.tau2[0] == 1.
    new, _ = eb_update(eta, [a], EBConfig(eb_count="dim", parametrization="P1"))
    assert new.alpha == pytest.approx(300/2.5) and new.tau2[0] == pytest.approx(1.5/10)
    new, _ = eb_update(p2_eta(), [a], EBConfig(eb_count="dim", learn_tau0=True))
    assert new.tau2[0] == pytest.approx(1.5/10) and new.alpha == alpha_nugget(new.tau2)


def test_exclusions_and_failure():
    good, no_trace, no_var = stats("good"), stats("no_trace", conv=0, passed=0), stats("no_var", passed=1)
    new, record = eb_update(p2_eta(), [good, no_trace, no_var, None], DIM)
    assert record["pairs_used_alpha_nu"] == ["good", "no_var"] and record["pairs_used_tau"] == ["good"]
    assert {r["pair_id"] for r in record["pairs_excluded"]} == {"no_trace", "no_var"}
    assert new.nu == pytest.approx(600/4.)
    with pytest.raises(pop.EBUpdateFailed, match="nu"):
        eb_update(p2_eta(), [no_trace, None], DIM)
    with pytest.raises(pop.EBUpdateFailed, match="tau"):
        eb_update(p2_eta(), [no_var], DIM)
    with pytest.raises(InputValidationError):
        eb_update(p2_eta(), [stats(tau_den=(1., 1.))], DIM)
    no_background = stats(tau_hat=(0., 2., 3., 4.), tau_var=(0., .5, .5, .5), tau_den=(0., 20., 30., 40.))
    new, record = eb_update(p2_eta(), [no_background], DIM)             # tau_0 not learned: diagnostic only
    assert "tau_0" in record["candidates_unavailable"] and new.tau2[0] == 1.
    with pytest.raises(pop.EBUpdateFailed):
        eb_update(p2_eta(), [no_background], EBConfig(eb_count="dim", learn_tau0=True))   # learned: the round fails


# ----------------------------------------------------------------------------------------------
# sigma prior (C15-5, C04-4, D29)
# ----------------------------------------------------------------------------------------------

def test_inverse_gamma_round_trips(record_property):
    for m, v in ((.01, 4e-6), (2.5, .3), (1e-4, 1e-9)):
        a, b = pop.inverse_gamma_from_moments(m, v)
        m2, v2 = pop.inverse_gamma_moments(a, b)
        assert abs(m2-m) <= 1e-12*m and abs(v2-v) <= 1e-12*v
        n0, s02 = pop.pseudo_count_form(a, b)
        a2, b2 = pop.inverse_gamma_from_pseudo_counts(n0, s02)
        assert abs(a2-a) <= 1e-12*a and abs(b2-b) <= 1e-12*b
    kappa0, n_q, s0 = .1, 1e6, .02                              # kappa0 -> n0 -> (a, b) -> n0
    a, b = pop.inverse_gamma_from_pseudo_counts(kappa0*n_q, s0)
    n0, s02 = pop.pseudo_count_form(a, b)
    assert abs(n0-kappa0*n_q) <= 1e-12*n0 and abs(s02-s0) <= 1e-12*s0
    record_property("checked", "moments and pseudo counts to 1e-12")


def test_sigma_prior_rules():
    sigma = np.array([[.010, .020], [.012, .018], [.011, .025], [.009, .021]])
    n_q = np.array([1e6, 1.1e6, .9e6, 1.05e6])
    prior = pop.sigma_prior(sigma, n_q, EBConfig())
    for c, row in enumerate(prior["channels"]):
        m, v = sigma[:, c].mean(), sigma[:, c].var(ddof=1)
        assert row["mean"] == pytest.approx(m) and row["variance"] == pytest.approx(v)
        assert row["a"] == pytest.approx(m*m/v+2) and row["b"] == pytest.approx(m*(row["a"]-1))
        assert row["n0"] == pytest.approx(2*row["a"]+2) and row["share_n0_over_Nq"] < 1e-3     # C11-5
    assert all(r["a"] == 0 for r in pop.sigma_prior(sigma[:2], n_q[:2], EBConfig())["channels"])      # n_T < 3
    flat = np.full((4, 1), .01)
    assert pop.sigma_prior(flat, n_q, EBConfig())["channels"][0]["a"] == 0                             # v = 0
    pseudo = pop.sigma_prior(sigma, n_q, EBConfig(sigma_prior="pseudo_count", kappa0=.05))
    for c, row in enumerate(pseudo["channels"]):
        assert row["n0"] == pytest.approx(.05*np.median(n_q)) and row["sigma0_sq"] == pytest.approx(sigma[:, c].mean())
    none = pop.sigma_prior(sigma, n_q, EBConfig(sigma_prior="none"))
    assert all(r["a"] == 0 and r["b"] == 0 for r in none["channels"])


# ----------------------------------------------------------------------------------------------
# Gauss-point statistics (B7-1), tau reads only the tissue layer, N_C (B11)
# ----------------------------------------------------------------------------------------------

def test_gauss_statistics_match_operators(record_property):
    grid = aniso()
    state = make_state(grid, seed=4)
    u = feasible_field(grid, np.random.default_rng(5), .05)
    theta = restrict_free(u, grid)
    out = gauss_point_statistics(state, u)
    kq = float(theta @ stiffness_apply_scaled(state, theta))
    curl = float(theta @ curl_gram_apply(state, theta))
    record_property("kq_rel", abs(out.kq_quadratic-kq)/kq)
    assert abs(out.kq_quadratic-kq) <= 1e-12*kq and abs(out.curl_quadratic-curl) <= 1e-12*curl
    assert out.tau_var is None and len(out.tau_den) == state.tissue.shape[0]
    total = grid.n_cells*grid.omega/state.omega0                  # sum_q omega w_g / omega0
    assert abs(sum(out.tau_den)-total) <= 1e-12*total
    geo = gauss_geometry(u, state.ops)
    logj2 = np.sum((state.ops.weights/state.omega0)*np.log(geo.J)**2)
    assert abs(sum(out.tau_hat)-logj2) <= 1e-12*logj2              # sum_k P^k = 1 at every Gauss point
    var = np.full((grid.n_cells, 27), .5)
    with_var = gauss_point_statistics(state, u, var)
    assert with_var.tau_var == pytest.approx(tuple(.5*d for d in out.tau_den), rel=1e-12)
    with pytest.raises(InputValidationError):
        gauss_point_statistics(state, u, -var)


def test_tau_statistics_read_only_the_tissue_layer():
    grid = aniso()
    a = make_state(grid, seed=4)
    other_channels = make_state(grid, seed=4, fixed=a.fixed*1.7+.3)
    assert other_channels.tissue.tobytes() == a.tissue.tobytes()
    u = feasible_field(grid, np.random.default_rng(6), .05)
    s1, s2 = gauss_point_statistics(a, u), gauss_point_statistics(other_channels, u)
    assert (s1.tau_hat, s1.tau_den, s1.kq_quadratic) == (s2.tau_hat, s2.tau_den, s2.kq_quadratic)    # bitwise
    other_tissue = make_state(grid, seed=9)
    assert gauss_point_statistics(other_tissue, u).tau_den != s1.tau_den


def test_n_curl_is_the_curl_rank():
    """C15-2 / B11: N_C = 3 N_int = rank(C^T W C) on the zero-boundary Q1 space (small grid)."""
    grid = aniso(6, 6, 6)
    state = make_state(grid, seed=1)
    n = 3*grid.n_free_scalar
    gram = np.stack([curl_gram_apply(state, e) for e in np.eye(n)], axis=1)
    assert np.linalg.matrix_rank(gram) == n
    stiff = np.stack([stiffness_apply(e, grid) for e in np.eye(n)], axis=1)
    assert np.linalg.matrix_rank(stiff) == n


# ----------------------------------------------------------------------------------------------
# The driver: rounds, criterion, cap, failure, MAP exclusions (injected statistics)
# ----------------------------------------------------------------------------------------------

class ReadyOutcome(MapOutcome):
    """A MapOutcome the driver treats as having a native output (the statistics step is injected)."""

    @property
    def usable(self):
        return self.native and self.status in pop.LAPLACE_ENDINGS


def fake_pairs(n=2):
    return tuple(EBPair(f"OASIS_OAS1_09{k:02d}_MR1", f"OASIS_OAS1_09{k+50:02d}_MR1", "A", {"check": "test"})
                 for k in range(n))


def contraction_statistics(target, seeds):
    """Per-pair statistics whose update is nu_new = target(nu), tau unchanged (one pair carries everything)."""
    def statistics(outcome, *, pair_id, config, seed, backend):
        eta = outcome.record["eta"]
        seeds.append((pair_id, seed))
        n = 300
        tau2 = eta.tau2
        s = PairStatistics(pair_id=pair_id, n_free=n, n_curl=n, n_q=1e3, kq_quadratic=1., curl_quadratic=n/target(eta.nu),
                           kq_trace=0., kq_trace_stderr=0., curl_trace=0., curl_trace_stderr=0.,
                           tau_hat=tuple(t*10. for t in tau2), tau_var=(0.,)*len(tau2), tau_den=(10.,)*len(tau2),
                           sigma_used=(.01,), sigma_post=(.01+.001*len(seeds),), n_samples=16, n_converged=16,
                           n_passed=16, pass_rate=1., pass_rate_low=False)
        return s, dict(seed=seed)
    return statistics


def solve_ready(pair, eta, *, round_index):
    return ReadyOutcome("converged_regularized_gradient_B", "test", True, None, None, [.01], dict(eta=eta))


def test_driver_stops_when_the_criterion_is_met(tmp_path):
    seeds = []
    target = lambda nu: 100+.5*(nu-100)                      # noqa: E731
    result = run_population_eb(fake_pairs(), solve_ready, eta0=p2_eta(nu=104.), config=DIM, run_id="t",
                               out_dir=tmp_path, statistics=contraction_statistics(target, seeds))
    assert result.status == EB_CRITERION_MET and len(result.rounds) == 1        # 104 -> 102: 1.9 % < 5 %
    assert result.eta.nu == pytest.approx(102.) and result.eta.alpha == p2_eta().alpha
    assert result.record["max_relative_change"] == [pytest.approx(2/104)]
    assert (tmp_path/"eb_result.json").is_file() and (tmp_path/"rounds"/"round_0.json").is_file()
    assert sorted(p.name for p in (tmp_path/"pairs"/"round_0").iterdir()) == sorted(f"{p.pair_id}.json"
                                                                                  for p in fake_pairs())
    assert len({s for _, s in seeds}) == len(seeds) and seeds[0][1] == pop.round_seed("t", 0, seeds[0][0])


def test_driver_reports_the_trajectory_at_the_round_cap(tmp_path):
    seeds = []
    target = lambda nu: 100+.5*(nu-100)                      # noqa: E731
    result = run_population_eb(fake_pairs(), solve_ready, eta0=p2_eta(nu=300.), config=DIM, run_id="t",
                               out_dir=tmp_path, statistics=contraction_statistics(target, seeds))
    assert result.status == EB_ROUND_CAP and len(result.rounds) == 3                # 300 -> 200 -> 150 -> 125
    assert [t["nu"] for t in result.trajectory] == pytest.approx([300., 200., 150., 125.])
    assert result.record["max_relative_change"] == pytest.approx([1/3, .25, 1/6])
    assert not any(r["criterion_met"] for r in result.rounds)
    assert len(seeds) == 6 and len({s for _, s in seeds}) == 6                     # per (round, pair)
    prior = result.sigma_prior                                                     # last round's sigma_post
    assert prior["n_T"] == 2 and prior["channels"][0]["a"] == 0                    # n_T < 3: no prior
    assert all(r["eta_out"]["alpha"] == p2_eta().alpha for r in result.rounds)    # P2: alpha not learned


def test_driver_excludes_failed_maps_and_fails_without_usable_pairs(tmp_path):
    seeds = []
    statistics = contraction_statistics(lambda nu: nu, seeds)

    def solve_mixed(pair, eta, *, round_index):
        if pair.pair_id.startswith("OASIS_OAS1_0900"):
            return ReadyOutcome("level_failed", "test", True, None, None, None, dict(eta=eta))
        return solve_ready(pair, eta, round_index=round_index)
    result = run_population_eb(fake_pairs(), solve_mixed, eta0=p2_eta(), config=DIM, run_id="t",
                               statistics=statistics)
    assert result.status == EB_CRITERION_MET
    rows = result.rounds[0]["pairs"]
    assert rows[0]["excluded"].startswith("MAP ended level_failed") and rows[1]["excluded"] is None

    def no_variance(outcome, **kwargs):
        s, record = statistics(outcome, **kwargs)
        return replace(s, n_passed=1, tau_var=None), record
    failed = run_population_eb(fake_pairs(), solve_ready, eta0=p2_eta(), config=DIM, run_id="t",
                               out_dir=tmp_path, statistics=no_variance)
    assert failed.status == EB_FAILED and "tau" in failed.rounds[0]["reason"]
    assert (tmp_path/"eb_result.json").is_file()


def test_driver_refuses_unadmitted_pairs_and_wrong_p2_alpha():
    with pytest.raises(InputValidationError):
        run_population_eb([("a", "b")], solve_ready, eta0=p2_eta(), run_id="t")
    bare = EBPair("OASIS_OAS1_0901_MR1", "OASIS_OAS1_0902_MR1", "A")            # no admission record
    with pytest.raises(InputValidationError):
        run_population_eb([bare], solve_ready, eta0=p2_eta(), run_id="t")
    with pytest.raises(InputValidationError, match="alpha_nug"):
        run_population_eb(fake_pairs(), solve_ready, eta0=Eta(.5, 2., TAU2), run_id="t")


# ----------------------------------------------------------------------------------------------
# Linear Gaussian toy (C15 gate): all blocks linear, known alpha
# ----------------------------------------------------------------------------------------------

def test_linear_gaussian_toy_recovers_alpha(record_property):
    """Prior u ~ N(0, (alpha* K)^-1), K = K_Q/omega0 (the diffusion block alone, so the block-wise
    normalization (N) is exact); data y = u + e, e ~ N(0, s^2 I); posterior H = alpha K + I/s^2.  Traces by
    perturb-and-MAP with S = 16 (C15-3).  The driver (P3: alpha learned; the curl and tau statistics are held at
    their fixed points) must meet the 5 % criterion within 10 rounds with alpha within 10 % of alpha* (C15 gate)."""
    grid = make_grid_3d(nx=7, ny=7, nz=7, hx=1., hy=1., hz=1., x0=0., y0=0., z0=0., unit="mm")
    n = 3*grid.n_free_scalar
    K = np.stack([stiffness_apply(e, grid) for e in np.eye(n)], axis=1)/grid.omega
    K = .5*(K+K.T)
    lam, Q = np.linalg.eigh(K)
    K_half = (Q*np.sqrt(lam))@Q.T
    alpha_true, s2, n_pairs = 3., .1, 8
    rng = np.random.default_rng(7)
    L = np.linalg.cholesky(alpha_true*K)
    data = {}
    pairs = tuple(EBPair(f"OASIS_OAS1_09{k:02d}_MR1", f"OASIS_OAS1_09{k+50:02d}_MR1", "A", {"check": "toy"})
                  for k in range(n_pairs))
    for pair in pairs:
        u = np.linalg.solve(L.T, rng.standard_normal(n))
        data[pair.pair_id] = u+np.sqrt(s2)*rng.standard_normal(n)

    def statistics(outcome, *, pair_id, config, seed, backend):
        eta = outcome.record["eta"]
        H = eta.alpha*K+np.eye(n)/s2
        y = data[pair_id]
        u_hat = np.linalg.solve(H, y/s2)
        g = np.random.default_rng(seed)
        b = np.sqrt(eta.alpha)*K_half@g.standard_normal((n, config.n_samples)) + \
            g.standard_normal((n, config.n_samples))/np.sqrt(s2)
        delta = np.linalg.solve(H, b)                                   # Cov = H^-1 (B5-1)
        trace = float(np.mean(np.einsum("is,ij,js->s", delta, K, delta)))
        s = PairStatistics(pair_id=pair_id, n_free=n, n_curl=n, n_q=1., kq_quadratic=float(u_hat@K@u_hat),
                           curl_quadratic=n/eta.nu, kq_trace=trace, kq_trace_stderr=0., curl_trace=0.,
                           curl_trace_stderr=0., tau_hat=tuple(10*t for t in eta.tau2), tau_var=(0., 0.),
                           tau_den=(10., 10.), sigma_used=(s2,), sigma_post=(s2,), n_samples=config.n_samples,
                           n_converged=config.n_samples, n_passed=config.n_samples, pass_rate=1., pass_rate_low=False)
        return s, {}
    eta0 = Eta(.3, 1., (1., .25))                                       # alpha 10x too small
    result = run_population_eb(pairs, solve_ready, eta0=eta0,
                               config=EBConfig(eb_count="dim", parametrization="P3", max_rounds=10), run_id="toy",
                               statistics=statistics)
    alphas = [t["alpha"] for t in result.trajectory]
    record_property("alpha_trajectory", [round(a, 4) for a in alphas])
    assert result.status == EB_CRITERION_MET and len(result.rounds) <= 10
    assert abs(result.eta.alpha/alpha_true-1) <= .10
    assert result.eta.nu == 1. and result.eta.tau2 == (1., .25)
