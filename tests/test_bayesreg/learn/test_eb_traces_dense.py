"""FB1 (review M5_EB item A2): the trace and Var_s(log J_q) paths of the EB statistics against a dense H.

The path under test is `pair_statistics` exactly as an EB round calls it: LaplaceSystem -> iter_perturbations (PCG,
contract settings) -> posterior_maps (sample traces over the PCG-converged samples; Welford Var_s(log J_q), ddof 1,
over the certified samples) -> gauss_point_statistics (the weights omega w_g/omega0 and the tissue layer P^k).
Case: the 6^3 Laplace case of tests/test_bayesreg/infer (3 N_int = 192), every precision x 400 so that the posterior
sd of log J is a few percent (linearization), with the independent dense oracle of tests/test_bayesreg/model/_helpers
(dense_model; nothing in it calls bayesreg):
  (a) deterministic: the same seed reproduces the samples (sample_perturbations on a fresh system); kq_trace and
      curl_trace equal (1/S) sum_s delta_s^T A delta_s with the dense A = K_D^T K_D / alpha (= K_Q/omega0) and
      K_C^T K_C / nu (= C^T W C/omega0) to 1e-12; tau_var[k] equals sum_q (omega w_g/omega0) P^k_q Var_s(log J_q)
      with log J_q of the oracle at u_hat + delta_s (ddof 1) to 1e-10;
  (b) statistical, against Sigma = H^-1 (dense): |trace - tr(A Sigma)| <= 5 SE, SE = sqrt(2 tr((A Sigma)^2) / S) (a
      Gaussian quadratic form; the PCG covariance bias of B5-3, O(rtol = 1e-3), is not modelled); tau_var[k] against
      the linearized tr(B_k Sigma), B_k = K_J^T diag(tau_q^2 P^k_q) K_J (K_J the dense log-volume block, row
      q = s_q grad log J_q with s_q^2 tau_q^2 = omega w_g/omega0), SE = sqrt(2 tr((B_k Sigma)^2) / (S - 1)), plus 2 %
      of tr(B_k Sigma) for the nonlinearity of log J; SE <= 20 % of the exact value is asserted (S = 64: about
      1.5 %, so the check resolves a path error of a few percent).  Seed fixed: deterministic.  Development run
      (2026-10-30, this machine): |z| <= 1.9 for every quantity in both regimes;
  in two regimes: case_sigma (the case's sigma, u_hat of amplitude .05 h) and large_sigma (sigma^2 x 1e4 and u_hat
  of amplitude .005 h: the sampled terms are most of the nu, alpha and tau denominators -- share > .5 asserted;
  .98, .97, .94 in the development run -- so this is the path the update divides by).
Certificates: a scripted all-pass backend with the real C07 configuration (the real certificate per sample is
test_laplace's slow test); posterior_maps still requires every Gauss J_q > 0 of a sample (all are, here).
"""
import numpy as np
import pytest

from jd_curl.three_d.boundary import inject_free

from bayesreg.infer import laplace as lp
from bayesreg.learn import EBConfig, pair_statistics

from ..infer import _cases_gauss27 as cs  # DEF: the C05-2 model ("gauss27", the default before DEF), explicit
from ..model._helpers import dense_model, dense_operators

FACTOR = 400.
S = 64
SEED = 20261030


def _setup(regime):
    """case_sigma: the case's sigma, u_hat of amplitude .05 h; large_sigma: sigma^2 x 1e4 and u_hat of amplitude
    .005 h (so the u_hat quadratic terms are small next to the sampled ones)."""
    grid, state = cs.cube6(factor=FACTOR)
    if regime == "large_sigma":
        state = state.replace(sigma2=np.asarray(state.sigma2)*1e4)
        return grid, state, cs.u_hat_of(grid, scale=.005)
    return grid, state, cs.u_hat_of(grid)


def _oracle(state, u):
    """Dense K_i, H, Sigma, A_K, A_C, and per tissue class B_k = K_J^T diag(tau^2 P^k) K_J; W P^k at the Gauss points."""
    blocks = dense_model(state, u)
    K, H = cs.dense_blocks(state, u)
    sigma = np.linalg.inv(H)
    sigma = .5*(sigma+sigma.T)
    A_K = K["diffusion"].T@K["diffusion"]/state.alpha
    A_C = K["curl"].T@K["curl"]/state.nu
    P = np.stack([blocks["B"]@state.tissue[k].ravel() for k in range(state.tissue.shape[0])])     # (K_T+1, |Q|)
    K_J = K["logvol"]
    B = [K_J.T@((blocks["tau2"]*P[k])[:, None]*K_J) for k in range(P.shape[0])]
    WP = (blocks["wq"]/state.omega0)[None]*P
    return dict(blocks=blocks, H=H, sigma=sigma, A_K=A_K, A_C=A_C, B=B, WP=WP)


def _expect(A, sigma, dof):
    AS = A@sigma
    return float(np.trace(AS)), float(np.sqrt(2*np.sum(AS*AS.T)/dof))


@pytest.mark.parametrize("regime", ["case_sigma", "large_sigma"])
def test_trace_and_logj_variance_paths_against_dense_h(regime, record_property):
    grid, state, u = _setup(regime)
    config = EBConfig(n_samples=S, eb_count="dim")              # the C15-1 statistics (dim explicit since EBD)
    backend = cs.ScriptedCertificate([True]*S)
    stats, record = pair_statistics(state, u, pair_id=f"dense_{regime}", config=config, seed=SEED, backend=backend)
    assert backend.calls == S and stats.n_samples == S
    assert stats.n_converged == S and stats.n_passed == S and stats.usable_precisions and stats.usable_tau
    oracle = _oracle(state, u)

    # (a) deterministic: the samples of this seed, the dense operators, the oracle's log J
    samples = lp.sample_perturbations(state, u, SEED, S, system=lp.LaplaceSystem(state, u, config=config.laplace))
    deltas = samples.deltas
    for name, A, got in (("kq", oracle["A_K"], stats.kq_trace), ("curl", oracle["A_C"], stats.curl_trace)):
        direct = float(np.mean(np.einsum("si,ij,sj->s", deltas, A, deltas)))
        assert abs(got-direct) <= 1e-12*abs(direct), name
    D = dense_operators(grid)[0]
    logj = np.stack([np.log(_dense_J(D, u+inject_free(d, grid))) for d in deltas])
    var = np.var(logj, axis=0, ddof=1)
    for k in range(oracle["WP"].shape[0]):
        direct = float(oracle["WP"][k]@var)
        assert abs(stats.tau_var[k]-direct) <= 1e-10*direct, k

    # (b) statistical: the dense Sigma = H^-1
    out = {}
    for name, A, got in (("kq", oracle["A_K"], stats.kq_trace), ("curl", oracle["A_C"], stats.curl_trace)):
        exact, se = _expect(A, oracle["sigma"], S)
        out[name] = dict(estimate=got, exact=exact, se=se, z=(got-exact)/se)
        assert abs(got-exact) <= 5*se, (name, out[name])
        assert se <= .2*exact                                    # the check resolves a 20 % (or larger) path error
    for k, Bk in enumerate(oracle["B"]):
        exact, se = _expect(Bk, oracle["sigma"], S-1)
        out[f"tau_var_{k}"] = dict(estimate=stats.tau_var[k], exact=exact, se=se, z=(stats.tau_var[k]-exact)/se)
        assert abs(stats.tau_var[k]-exact) <= 5*se+.02*exact, (k, out[f"tau_var_{k}"])
        assert se <= .2*exact
    shares = dict(curl=stats.curl_trace/(stats.curl_trace+stats.curl_quadratic),
                  kq=stats.kq_trace/(stats.kq_trace+stats.kq_quadratic),
                  tau=[v/(v+h) for v, h in zip(stats.tau_var, stats.tau_hat)])
    record_property("dense_check", out)
    record_property("trace_share_of_denominator", shares)
    if regime == "large_sigma":                     # the sampled terms dominate what the update divides by
        assert shares["curl"] > .5 and shares["kq"] > .5 and min(shares["tau"]) > .5, shares
    assert record["statistics"]["S_conv"] == S and record["statistics"]["S_pass"] == S


def _dense_J(D, u):
    """J_q at the Gauss points from the oracle's shape derivatives D (3, |Q|, N) (the formula of dense_model)."""
    F = np.einsum("jqn,nr->qrj", D, np.asarray(u).reshape(-1, 3))+np.eye(3)
    return np.linalg.det(F)
