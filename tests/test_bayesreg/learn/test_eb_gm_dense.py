"""C15-9 statistics on the real path against a dense oracle: `pair_statistics` with eb_count = "gull_mackay"
(LaplaceSystem -> posterior samples -> posterior_maps; PriorSystem -> prior samples at u = 0; rho-weighted Gauss sums)
on the 6^3 case of tests/test_bayesreg/infer with every precision x 400 (3 N_int = 192), S = S0 = 64, scripted all-pass
certificates.  The oracle is tests/test_bayesreg/model/_helpers (dense_model, dense_operators, selection; nothing in it
calls bayesreg).
  (a) deterministic (the same seeds reproduce the samples): g_nu = nu mean d0^T A_nu d0, g_alpha, m_prior[k] =
      sum_q c_q rho_qk mean (D_div d0)_q^2 with the dense A, D_div and rho to 1e-12; m_hat[k] = sum_q c_q rho_qk
      (log J_q(u_hat))^2 and m_var[k] (ddof 1 over the posterior samples, the oracle's log J) to 1e-10; per prior sample
      alpha d0^T A_alpha d0 + nu d0^T A_nu d0 + sum_k d0^T B~_k d0/tau_k^2 = d0^T Pi_0 d0 (dense) to 1e-10 relative
      (C15-9 gate (iii)), and the recorded maximum of the same identity;
  (b) statistical against Sigma_0 = Pi_0^-1 (dense): |estimate - exact| <= 5 SE for g_nu, g_alpha, m_prior[k]
      (SE = sqrt(2 tr((A Sigma_0)^2)/S0)); m_var[k] against the linearized tr(B~_k^post H^-1) within 5 SE + 2 %
      (as test_eb_traces_dense); the count check (g_alpha + g_nu + sum_k m_prior_k/tau_k^2)/3N_int within
      5 sqrt(2/(3N_int S0)) of 1 (B17-3 (b));
  (c) one C15-9 update from these statistics: finite values and z values, recorded;
  (d) one driver round through the default statistics step (outcome_statistics -> pair_statistics with prior_seed).
"""
import json

import numpy as np

from jd_curl.three_d.boundary import inject_free

from bayesreg.infer import laplace as lp
from bayesreg.learn import (EBConfig, EBPair, MapOutcome, alpha_nugget, eb_update, pair_statistics, prior_round_seed,
                            run_population_eb)
from bayesreg.solve import Eta

from ..infer import _cases_gauss27 as cs  # DEF: the C05-2 model ("gauss27", the default before DEF), explicit
from ..model._helpers import dense_model, dense_operators, selection

FACTOR = 400.
S = 64
SEED = 20261030
PRIOR_SEED = 20261031


def _dense_J(D, u):
    F = np.einsum("jqn,nr->qrj", D, np.asarray(u).reshape(-1, 3))+np.eye(3)
    return np.linalg.det(F)


def test_gull_mackay_statistics_against_dense(record_property):
    grid, state = cs.cube6(factor=FACTOR)
    u = cs.u_hat_of(grid)
    config = EBConfig(n_samples=S, eb_count="gull_mackay")
    backend = cs.ScriptedCertificate([True]*S)
    stats, record = pair_statistics(state, u, pair_id="dense_gm", config=config, seed=SEED, backend=backend,
                                    prior_seed=PRIOR_SEED)
    gm = stats.gm
    assert gm is not None and gm.n_prior_samples == S and gm.n_prior_converged == S and gm.usable
    assert record["prior"]["S0_conv"] == S and record["prior"]["seed"] == PRIOR_SEED
    assert stats.record()["gull_mackay"]["S0"] == S

    zero = np.zeros(grid.shape+(3,))
    K0, _ = cs.dense_blocks(state, zero)
    Pi0 = sum(K0[name].T@K0[name] for name in ("diffusion", "logvol", "curl"))
    A_alpha = K0["diffusion"].T@K0["diffusion"]/state.alpha
    A_nu = K0["curl"].T@K0["curl"]/state.nu
    D, B, wq = dense_operators(grid)
    Sel = selection(grid)
    Ddiv = np.concatenate([D[j]@Sel for j in range(3)], axis=1)
    blocks = dense_model(state, u, (D, B, wq))
    tau_q2 = blocks["tau2"]
    P = np.stack([B@state.tissue[k].ravel() for k in range(state.tissue.shape[0])], axis=1)
    c = wq/state.omega0
    w_rho = c[:, None]*P*(np.asarray(state.tau2)[None, :]/tau_q2[:, None])**2
    n_tau = P.shape[1]

    # (a) deterministic
    d0 = lp.sample_perturbations(state, zero, PRIOR_SEED, S, system=lp.PriorSystem(state, config=config.laplace)).deltas
    qa = np.einsum("si,ij,sj->s", d0, A_alpha, d0)
    qn = np.einsum("si,ij,sj->s", d0, A_nu, d0)
    div2 = (d0@Ddiv.T)**2                                                   # (S, |Q|)
    m_s = div2@w_rho                                                        # (S, K_T + 1)
    checks = dict(g_alpha=(gm.g_alpha, state.alpha*qa.mean()), g_nu=(gm.g_nu, state.nu*qn.mean()))
    for k in range(n_tau):
        checks[f"m_prior_{k}"] = (gm.m_prior[k], m_s[:, k].mean())
    for name, (got, direct) in checks.items():
        assert abs(got-direct) <= 1e-12*abs(direct), name
    identity = state.alpha*qa+state.nu*qn+m_s@(1/np.asarray(state.tau2))
    pi = np.einsum("si,ij,sj->s", d0, Pi0, d0)
    record_property("identity_dense_max_rel", float(np.max(np.abs(identity-pi)/pi)))
    record_property("identity_recorded_max_rel", gm.identity_max_rel)
    assert np.max(np.abs(identity-pi)/pi) <= 1e-10 and gm.identity_max_rel <= 1e-10      # gate (iii)
    logj_hat = np.log(_dense_J(D, u))
    deltas = lp.sample_perturbations(state, u, SEED, S, system=lp.LaplaceSystem(state, u, config=config.laplace)).deltas
    logj = np.stack([np.log(_dense_J(D, u+inject_free(d, grid))) for d in deltas])
    var = np.var(logj, axis=0, ddof=1)
    for k in range(n_tau):
        assert abs(gm.m_hat[k]-w_rho[:, k]@logj_hat**2) <= 1e-10*(w_rho[:, k]@logj_hat**2), k
        assert abs(gm.m_var[k]-w_rho[:, k]@var) <= 1e-10*(w_rho[:, k]@var), k
        assert gm.m_var_stderr_bound[k] == gm.m_var[k]*np.sqrt(2/(S-1))

    # (b) statistical
    Sig0 = np.linalg.inv(Pi0)
    Sig0 = .5*(Sig0+Sig0.T)
    out = {}

    def expect(A, sigma, dof):
        AS = A@sigma
        return float(np.trace(AS)), float(np.sqrt(2*np.sum(AS*AS.T)/dof))
    for name, A, got, scale in (("g_alpha", A_alpha, gm.g_alpha, state.alpha), ("g_nu", A_nu, gm.g_nu, state.nu)):
        exact, se = expect(A, Sig0, S)
        out[name] = dict(estimate=got, exact=scale*exact, z=(got-scale*exact)/(scale*se))
        assert abs(got-scale*exact) <= 5*scale*se, out[name]
    _, H = cs.dense_blocks(state, u)
    Sig = np.linalg.inv(H)
    Sig = .5*(Sig+Sig.T)
    K_J = cs.dense_blocks(state, u)[0]["logvol"]
    for k in range(n_tau):
        Bt = Ddiv.T@(w_rho[:, k][:, None]*Ddiv)
        exact, se = expect(Bt, Sig0, S)
        out[f"m_prior_{k}"] = dict(estimate=gm.m_prior[k], exact=exact, z=(gm.m_prior[k]-exact)/se)
        assert abs(gm.m_prior[k]-exact) <= 5*se, out[f"m_prior_{k}"]
        B_post = K_J.T@((tau_q2*P[:, k]*(state.tau2[k]/tau_q2)**2)[:, None]*K_J)
        exact, se = expect(B_post, Sig, S-1)
        out[f"m_var_{k}"] = dict(estimate=gm.m_var[k], exact=exact, z=(gm.m_var[k]-exact)/se)
        assert abs(gm.m_var[k]-exact) <= 5*se+.02*exact, out[f"m_var_{k}"]
    n = 3*grid.n_free_scalar
    record_property("dense_check", out)
    record_property("count_check", gm.count_check)
    assert abs(gm.count_check-1) <= 5*np.sqrt(2/(n*S))

    # (c) one update from these statistics (eta = the state's)
    eta = Eta(state.alpha, state.nu, tuple(state.tau2))
    new, update = eb_update(eta, [stats], config)
    record_property("update_forms", update["forms"])
    record_property("z_nu", update["candidates"]["nu"]["z_numerator"])
    assert np.isfinite(new.nu) and all(np.isfinite(t) for t in new.tau2)
    assert np.isfinite(update["candidates"]["nu"]["z_numerator"])
    assert all(np.isfinite(update["candidates"][f"tau_{k}"]["z_denominator"]) for k in range(1, n_tau))


def test_driver_round_through_the_default_statistics_step(tmp_path):
    """run_population_eb with eb_count "gull_mackay" and the default statistics step (outcome_statistics ->
    pair_statistics with prior_seed = prior_round_seed): one round on two 6^3 'pairs' (the same u_hat under the round's
    eta, scripted all-pass certificates, S = S0 = 8): the C15-9 statistics reach the update and the records."""
    grid, state = cs.cube6(factor=FACTOR)
    u = cs.u_hat_of(grid)
    tau2 = tuple(float(t) for t in state.tau2)
    eta0 = Eta(alpha_nugget(tau2), float(state.nu), tau2)
    pairs = tuple(EBPair(f"OASIS_OAS1_09{k:02d}_MR1", f"OASIS_OAS1_09{k+50:02d}_MR1", "A", {"check": "test"})
                  for k in range(2))

    def solve_pair(pair, eta, *, round_index):
        st = state.replace(alpha=eta.alpha, nu=eta.nu, tau2=np.asarray(eta.tau2))
        return MapOutcome("converged_regularized_gradient_B", "test", True, u, st, [1e-3]*st.n_channels, {})
    config = EBConfig(n_samples=8, eb_count="gull_mackay", max_rounds=1)
    result = run_population_eb(pairs, solve_pair, eta0=eta0, config=config, run_id="gm_driver", out_dir=tmp_path,
                               backend=cs.ScriptedCertificate([True]*64))
    rnd = result.rounds[0]
    assert rnd["update"]["eb_count"] == "gull_mackay" and rnd["step_ratio"] is None
    assert rnd["update"]["pairs_used_alpha_nu"] == [p.pair_id for p in pairs]
    for row in rnd["pairs"]:
        gm = row["statistics"]["gull_mackay"]
        assert gm["S0"] == 8 and gm["S0_conv"] == 8
        assert gm["seed"] == prior_round_seed("gm_driver", 0, row["pair_id"])
    assert result.record["eb_count"] == "gull_mackay" and result.record["step_ratios"] == [None]
    stored = json.loads((tmp_path/"pairs"/"round_0"/f"{pairs[0].pair_id}.json").read_text())
    assert stored["posterior"]["prior"]["S0"] == 8
