"""Contract C15-10 (logvol_points = "gauss27+closed27", Simpson; work package IMPL27b): the EB statistics on the real
path (`pair_statistics`: LaplaceSystem -> posterior samples -> posterior_maps; PriorSystem -> prior samples at u = 0;
the tau sums over Gauss27 U closed27) against the independent dense oracles (tests/test_bayesreg/model/_helpers for the
Gauss27 rows, dense_closed27_operators of tests/test_bayesreg/model/test_logvol_closed27 for the closed27 rows).
The 6^3 case of tests/test_bayesreg/infer with every precision x 400, S = S0 = 64, scripted all-pass certificates
(the gates of test_eb_traces_dense and test_eb_gm_dense applied to the closed27 block):
  dim  (C15-1/3 under C15-10): tau_hat[k] = sum_{x in Q u S} c_x P^k(x) (log J_x(u_hat))^2, tau_den[k] = sum c_x P^k(x)
       (1e-12), tau_var[k] = sum c_x P^k(x) Var_s(log J_x) with the oracle's log J at both point sets (1e-10);
       c_q = omega w_g theta_G/omega0, c_p = omega s_p theta_S/omega0; tau_den equals the class volume, i.e. the
       gauss27 state's tau_den (1e-13); kq / curl quadratics unchanged against gauss27 (bitwise); tau_var against the
       linearized tr(B_k H^-1) within 5 SE + 2 %;
  gull_mackay (C15-9 under C15-10): m_prior[k] = sum_x c_x rho_xk mean (D_div d0)_x^2, m_hat, m_var with
       rho_xk = P^k(x) (tau_k^2/tau_x^2)^2 at both point sets (1e-12 / 1e-10); per prior sample alpha a + nu c +
       sum_k m_k/tau_k^2 = d0^T Pi_0 d0 with the dense Pi_0 including the closed27 block (1e-10; recorded maximum too);
       m_prior[k] against tr(B~_k Pi_0^-1) within 5 SE; the count check within 5 sqrt(2/(3N_int S0)) of 1.
The comparison state `plain` is the C05-2 model (logvol_points "gauss27") from infer/_cases_gauss27: since DEF the
default state of infer/_cases is gauss27+closed27 (beta_S 1), which would compare closed27 with closed27 and lose the
C05-2 comparison (MIN1, review IMPL27b minor 3).  The closed27 states are plain.replace(logvol_points=CLOSED, beta_S),
as in infer/test_laplace_closed27.
"""
import json

import numpy as np

from jd_curl.three_d.boundary import inject_free

from bayesreg.infer import laplace as lp
from bayesreg.learn import EBConfig, pair_statistics

from ..infer import _cases_gauss27 as cs     # plain = C05-2 ("gauss27", the default before DEF), explicit (MIN1)
from ..infer.test_laplace_closed27 import dense_closed_blocks
from ..model._helpers import dense_operators, selection
from ..model.test_logvol_closed27 import dense_closed27_operators, thetas

CLOSED = "gauss27+closed27"
FACTOR = 400.
S = 64
SEED = 20261030
PRIOR_SEED = 20261031


def _setup(share=1.):
    grid, plain = cs.cube6(factor=FACTOR)
    assert plain.logvol_points == "gauss27" and not plain.logvol_closed27        # the C05-2 comparison (MIN1)
    return grid, plain, plain.replace(logvol_points=CLOSED, logvol_closed27_share=share), cs.u_hat_of(grid)


def _points(state):
    """Dense rows of both point sets: D (3, rows, N), weights c (rows,), P (rows, K_T + 1), tau_x^2 (rows,)."""
    grid = state.grid
    Dg, Bg, wg = dense_operators(grid)
    Ds, Bs, ws = dense_closed27_operators(grid, state.closed27_weights)
    theta_G, theta_S = thetas(state)
    D = np.concatenate((Dg, Ds), axis=1)
    B = np.concatenate((Bg, Bs))
    c = np.concatenate((theta_G*wg, theta_S*ws))/state.omega0
    P = np.stack([B@state.tissue[k].ravel() for k in range(state.tissue.shape[0])], axis=1)
    tau2 = P@np.asarray(state.tau2)
    return D, c, P, tau2


def _J(D, u):
    F = np.einsum("jqn,nr->qrj", D, np.asarray(u).reshape(-1, 3))+np.eye(3)
    return np.linalg.det(F)


def _expect(A, sigma, dof):
    AS = A@sigma
    return float(np.trace(AS)), float(np.sqrt(2*np.sum(AS*AS.T)/dof))


def test_dim_tau_statistics_over_both_point_sets(record_property):
    grid, plain, state, u = _setup(3.)
    config = EBConfig(n_samples=S, eb_count="dim")
    stats, record = pair_statistics(state, u, pair_id="closed27_dim", config=config, seed=SEED,
                                    backend=cs.ScriptedCertificate([True]*S))
    plain_stats, _ = pair_statistics(plain, u, pair_id="plain_dim", config=config, seed=SEED,
                                     backend=cs.ScriptedCertificate([True]*S))
    assert stats.n_passed == S and record["n_closed27"] == 27*grid.n_cells
    D, c, P, _ = _points(state)
    logj_hat = np.log(_J(D, u))
    deltas = lp.sample_perturbations(state, u, SEED, S, system=lp.LaplaceSystem(state, u, config=config.laplace)).deltas
    logj = np.stack([np.log(_J(D, u+inject_free(d, grid))) for d in deltas])
    var = np.var(logj, axis=0, ddof=1)
    out = {}
    for k in range(P.shape[1]):
        wp = c*P[:, k]
        assert abs(stats.tau_hat[k]-wp@logj_hat**2) <= 1e-12*(wp@logj_hat**2), k
        assert abs(stats.tau_den[k]-wp.sum()) <= 1e-12*wp.sum(), k
        assert abs(stats.tau_var[k]-wp@var) <= 1e-10*(wp@var), k
        out[f"tau_den_vs_class_volume_{k}"] = abs(stats.tau_den[k]-plain_stats.tau_den[k])/plain_stats.tau_den[k]
        assert out[f"tau_den_vs_class_volume_{k}"] <= 1e-13
    assert (stats.kq_quadratic, stats.curl_quadratic) == (plain_stats.kq_quadratic, plain_stats.curl_quadratic)
    # statistical: Var_s against the linearized tr(B_k H^-1), B_k = K_J^T diag(tau_x^2 P^k_x) K_J over both sets
    K, H = dense_closed_blocks(state, u)
    sigma = np.linalg.inv(H)
    sigma = .5*(sigma+sigma.T)
    K_J = np.concatenate((K["logvol"], K["logvol_closed27"]))
    _, _, _, tau2 = _points(state)
    for k in range(P.shape[1]):
        Bk = K_J.T@((tau2*P[:, k])[:, None]*K_J)
        exact, se = _expect(Bk, sigma, S-1)
        out[f"tau_var_{k}"] = dict(estimate=stats.tau_var[k], exact=exact, z=(stats.tau_var[k]-exact)/se)
        assert abs(stats.tau_var[k]-exact) <= 5*se+.02*exact, out[f"tau_var_{k}"]
    record_property("dim_closed27", out)


def test_gull_mackay_statistics_over_both_point_sets(record_property):
    grid, plain, state, u = _setup(1.)
    config = EBConfig(n_samples=S, eb_count="gull_mackay")
    stats, record = pair_statistics(state, u, pair_id="closed27_gm", config=config, seed=SEED,
                                    backend=cs.ScriptedCertificate([True]*S), prior_seed=PRIOR_SEED)
    gm = stats.gm
    assert gm is not None and gm.usable and gm.n_prior_converged == S
    assert record["prior"]["system"]["noise_order"] == list(lp.PRIOR_NOISE_ORDER_CLOSED27)
    zero = np.zeros(grid.shape+(3,))
    K0, Pi0 = dense_closed_blocks(state, zero, ("diffusion", "logvol", "curl", "logvol_closed27"))
    A_alpha = K0["diffusion"].T@K0["diffusion"]/state.alpha
    A_nu = K0["curl"].T@K0["curl"]/state.nu
    D, c, P, tau2 = _points(state)
    Sel = selection(grid)
    Ddiv = np.concatenate([D[j]@Sel for j in range(3)], axis=1)
    w_rho = c[:, None]*P*(np.asarray(state.tau2)[None, :]/tau2[:, None])**2
    d0 = lp.sample_perturbations(state, zero, PRIOR_SEED, S, system=lp.PriorSystem(state, config=config.laplace)).deltas
    qa = np.einsum("si,ij,sj->s", d0, A_alpha, d0)
    qn = np.einsum("si,ij,sj->s", d0, A_nu, d0)
    m_s = ((d0@Ddiv.T)**2)@w_rho
    checks = dict(g_alpha=(gm.g_alpha, state.alpha*qa.mean()), g_nu=(gm.g_nu, state.nu*qn.mean()))
    for k in range(P.shape[1]):
        checks[f"m_prior_{k}"] = (gm.m_prior[k], m_s[:, k].mean())
    for name, (got, direct) in checks.items():
        assert abs(got-direct) <= 1e-12*abs(direct), name
    identity = state.alpha*qa+state.nu*qn+m_s@(1/np.asarray(state.tau2))
    pi = np.einsum("si,ij,sj->s", d0, Pi0, d0)
    record_property("identity_dense_max_rel", float(np.max(np.abs(identity-pi)/pi)))
    record_property("identity_recorded_max_rel", gm.identity_max_rel)
    assert np.max(np.abs(identity-pi)/pi) <= 1e-10 and gm.identity_max_rel <= 1e-10
    logj_hat = np.log(_J(D, u))
    deltas = lp.sample_perturbations(state, u, SEED, S, system=lp.LaplaceSystem(state, u, config=config.laplace)).deltas
    var = np.var(np.stack([np.log(_J(D, u+inject_free(d, grid))) for d in deltas]), axis=0, ddof=1)
    for k in range(P.shape[1]):
        assert abs(gm.m_hat[k]-w_rho[:, k]@logj_hat**2) <= 1e-10*(w_rho[:, k]@logj_hat**2), k
        assert abs(gm.m_var[k]-w_rho[:, k]@var) <= 1e-10*(w_rho[:, k]@var), k
    sig0 = np.linalg.inv(Pi0)
    sig0 = .5*(sig0+sig0.T)
    out = {}
    for k in range(P.shape[1]):
        Bt = Ddiv.T@(w_rho[:, k][:, None]*Ddiv)
        exact, se = _expect(Bt, sig0, S)
        out[f"m_prior_{k}"] = dict(estimate=gm.m_prior[k], exact=exact, z=(gm.m_prior[k]-exact)/se)
        assert abs(gm.m_prior[k]-exact) <= 5*se, out[f"m_prior_{k}"]
    n = 3*grid.n_free_scalar
    record_property("dense_check", out)
    record_property("count_check", gm.count_check)
    assert abs(gm.count_check-1) <= 5*np.sqrt(2/(n*S))
    # the closed27 rows matter: without them m_prior's dense value differs
    rows_g = slice(0, grid.n_cells*27)
    plain_m = ((d0@Ddiv[rows_g].T)**2)@w_rho[rows_g]
    assert np.max(np.abs(plain_m.mean(axis=0)-np.asarray(gm.m_prior))/np.asarray(gm.m_prior)) > 1e-3


def test_posterior_qc_noise_order_follows_the_state():
    """MIN1 (review IMPL27b minor 1): the EB pair record's posterior_qc.config.noise_order is the state's noise order
    (with logvol_closed27 under C05-2b, as laplace_system.noise_order) and NOISE_ORDER under gauss27 (as before MIN1;
    laplace_system then has no noise_order key, logvol_noise_record is {} for gauss27).
    FUP1 (review MIN1 minor 5 and question 5): the gauss27 half asserts that the key is absent (the `.get` default made
    it hold whatever the record said); EBConfig.record() without an argument and with logvol_points="gauss27" is the
    record before FUP1 (NOISE_ORDER, the same JSON), with "gauss27+closed27" its laplace.noise_order is the C14-7 order
    and nothing else changes."""
    grid, plain, state, u = _setup()
    config = EBConfig(n_samples=2, eb_count="dim")
    for st, expected in ((state, lp.NOISE_ORDER_CLOSED27), (plain, lp.NOISE_ORDER)):
        _, record = pair_statistics(st, u, pair_id="noise_order", config=config, seed=SEED,
                                    backend=cs.ScriptedCertificate([True]*2))
        assert record["posterior_qc"]["config"]["noise_order"] == list(expected)
        if st.logvol_closed27:
            assert record["laplace_system"]["noise_order"] == list(expected)
        else:
            assert "noise_order" not in record["laplace_system"]
    before = config.record()
    assert before["laplace"]["noise_order"] == list(lp.NOISE_ORDER)
    assert json.dumps(config.record(logvol_points="gauss27")) == json.dumps(before)
    closed = config.record(logvol_points=CLOSED)
    assert closed["laplace"]["noise_order"] == list(lp.NOISE_ORDER_CLOSED27)
    assert any("logvol_closed27" in item for item in closed["laplace"]["noise_order"])
    closed["laplace"]["noise_order"] = before["laplace"]["noise_order"]
    assert json.dumps(closed) == json.dumps(before)
