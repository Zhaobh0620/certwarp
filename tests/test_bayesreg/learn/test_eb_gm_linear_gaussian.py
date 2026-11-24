"""Contract C15-9 gate (i) (revised after review R1): eb_count = "gull_mackay" on the linear Gaussian population of
_lg_toy.py (dense, independent oracle; 8 pairs, 3 N_int = 108; P2 with tau_0 = 1 fixed and the largest, so
alpha = alpha_nug = 1e-2 is constant near the target).  The update under test is bayesreg's `eb_update` (C15-9 path).

(a) exact traces: the fixed point of the C15-9 map equals the maximizer of the exact evidence, found WITHOUT B17's
    trace formula (BFGS + Newton on the marginal-form gradient of y ~ N(0, J Pi_0^-1 J^T + I), checked against central
    differences of l and against Nelder-Mead on l alone): max relative difference <= 1e-7 (contract); the ratio form
    (the recorded fallback values) has the same fixed point (<= 1e-9); the C15-1 (dim) fixed point lies at larger nu
    with d l/d nu < 0 there (B17 corollary 4.2).
(b) MC regression test (not evidence for B17-4): S = S0 = 16 Gaussian samples per pair and round, fixed seed, 20 rounds
    from the maximizer; the mean log deviation of each learned component (nu, tau_1, tau_2 as standard deviations)
    must be <= 4 SE_pred + v (SE_pred: the 20-round mean of the linearized iteration, innovation covariance by the
    delta method from exact traces and the Jacobian of the exact map at the fixed point by central differences;
    v: the predicted one-step variance, the order of the ratio estimator's bias); no gm_fallback.  Before the gate was
    fixed, the prediction was compared with 40 seeds (scratch, 2026-10-30): 20-round SE predicted .0099 / .0142 /
    .0152, empirical .0098 / .0130 / .0163; one-step sd predicted .0232 / .0146 / .0146, empirical .0251 / .0151 /
    .0147.
(c) trajectory regression test from (nu*/5, 2 tau*), fixed seed, 3 rounds (the C15-6 cap): each MC step is within
    4 sd_pred + v of the exact step from the same point; the exact iterates approach the maximizer; the MC trajectory
    is pinned (values of the first run, rtol 1e-6: detects any change of the update or the statistics' use).
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from bayesreg.learn import GM_FORM

from . import _lg_toy as lg

SEED_MC = 20261030
SEED_OFF = 20261031
ROUNDS = 20
# (c): the MC trajectory (z = log nu, log tau_1, log tau_2) of the first run, rounds 1-3 (2026-10-30, this machine)
PINNED_OFF_START = [[1.0094351412521192, -0.870263991520022, -0.8443933738256837],
                    [1.214370544788239, -0.9391950633361873, -0.9270482307202534],
                    [1.2709599253358392, -0.9459104719441234, -0.9218863323384872]]


def fixed_point(F, z0, plain=30, newton=6):
    z = np.asarray(z0, dtype=np.float64).copy()
    for _ in range(plain):
        z = F(z)
    for _ in range(newton):
        r = F(z)-z
        jac = np.array([(F(z+e)-F(z-e))/2e-6 for e in 1e-6*np.eye(3)]).T-np.eye(3)
        z = z-np.linalg.solve(jac, r)
    return z, float(np.abs(F(z)-z).max())


@pytest.fixture(scope="module")
def target():
    """The exact evidence maximizer from the marginal form (BFGS + Newton with a central-difference Hessian of the
    marginal gradient), the GM fixed point and the Jacobian of the exact map there."""
    toy = lg.toy()
    z_true = np.log([lg.NU_TRUE, *lg.TAU_TRUE])
    res = minimize(lambda z: (-toy.log_evidence(z), -toy.marginal_gradient(z)), z_true, jac=True, method="BFGS",
                   options=dict(gtol=1e-10, maxiter=500))
    z = res.x
    for _ in range(3):
        hess = np.array([(toy.marginal_gradient(z+e)-toy.marginal_gradient(z-e))/2e-5 for e in 1e-5*np.eye(3)])
        hess = .5*(hess+hess.T)
        z = z-np.linalg.solve(hess, toy.marginal_gradient(z))
    step = lambda x: toy.step(x)[0]                                           # noqa: E731
    z_gm, residual = fixed_point(step, z_true)
    jac = np.array([(step(z+e)-step(z-e))/2e-5 for e in 1e-5*np.eye(3)]).T
    return dict(toy=toy, z_max=z, hess=hess, z_gm=z_gm, residual=residual, jac=jac, z_true=z_true)


def test_marginal_gradient_is_the_derivative_of_the_evidence(target, record_property):
    toy = target["toy"]
    for z in (target["z_true"], target["z_true"]+np.array([.3, -.2, .25])):
        grad = toy.marginal_gradient(z)
        fd = np.array([(toy.log_evidence(z+e)-toy.log_evidence(z-e))/2e-5 for e in 1e-5*np.eye(3)])
        err = float(np.abs(grad-fd).max()/np.abs(grad).max())
        record_property("marginal_gradient_vs_fd", err)
        assert err <= 1e-6


def test_fixed_point_is_the_exact_evidence_maximizer(target, record_property):
    toy, z_max, z_gm = target["toy"], target["z_max"], target["z_gm"]
    assert np.all(np.linalg.eigvalsh(target["hess"]) < 0)                    # a maximum, not a saddle
    assert np.abs(toy.marginal_gradient(z_max)).max() <= 1e-9
    eta_max = lg.eta_of(z_max)
    assert max(eta_max.tau2[1:]) < eta_max.tau2[0]                           # tau_0 largest: alpha_nug constant here
    nm = minimize(lambda z: -toy.log_evidence(z), target["z_true"], method="Nelder-Mead",
                  options=dict(xatol=1e-9, fatol=1e-11, maxiter=3000, maxfev=3000)).x
    rel_gm = float(np.abs(np.exp(z_gm-z_max)-1).max())
    record_property("gm_fixed_point_residual", target["residual"])
    record_property("gm_vs_maximizer_rel", rel_gm)
    record_property("nelder_mead_vs_maximizer", float(np.abs(nm-z_max).max()))
    record_property("marginal_grad_at_gm_fixed_point", float(np.abs(toy.marginal_gradient(z_gm)).max()))
    assert target["residual"] <= 1e-12
    assert np.abs(nm-z_max).max() <= 1e-6                                   # the maximizer from l values alone
    assert rel_gm <= 1e-7                                                    # contract C15-9 gate (i)
    _, record = toy.step(z_gm)
    assert record["gm_fallback"] == [] and set(record["forms"].values()) == {GM_FORM}
    assert record["alpha_nug_coupling"]["flag"] is False                     # default case: no coupling (B17 4.3)
    z_ratio, res_ratio = fixed_point(toy.ratio_step, target["z_true"], plain=60)
    record_property("ratio_vs_gm", float(np.abs(z_ratio-z_gm).max()))
    assert res_ratio <= 1e-12 and np.abs(z_ratio-z_gm).max() <= 1e-9         # same fixed point (B17-4)

    def dim_step(z):
        c = toy.step(z)[1]["candidates"]
        return np.log([c["nu"]["dim"], np.sqrt(c["tau_1"]["dim"]), np.sqrt(c["tau_2"]["dim"])])
    z_dim, res_dim = fixed_point(dim_step, target["z_true"], plain=60)
    grad_dim = toy.marginal_gradient(z_dim)
    record_property("dim_fixed_point_over_maximizer", np.exp(z_dim-z_max).tolist())
    record_property("dim_dl_dlognu", float(grad_dim[0]))
    assert res_dim <= 1e-10 and z_dim[0] > z_max[0] and grad_dim[0] < 0     # B17 corollary 4.2 (recorded C15-1 values)


def test_mc_from_the_maximizer_regression(target, record_property):
    toy, z_max, jac = target["toy"], target["z_max"], target["jac"]
    innovation = toy.innovation_cov(lg.eta_of(z_max))
    sums = [sum(np.linalg.matrix_power(jac, m) for m in range(ROUNDS-i)) for i in range(ROUNDS)]
    se = np.sqrt(np.diag(sum(M@innovation@M.T for M in sums)/ROUNDS**2))
    v = np.diag(innovation)
    rng = np.random.default_rng(SEED_MC)
    z, devs, fallbacks = z_max.copy(), [], []
    for _ in range(ROUNDS):
        z, record = toy.step(z, rng=rng)
        devs.append(z-z_max)
        fallbacks += record["gm_fallback"]
    mean = np.mean(devs, axis=0)
    record_property("spectral_radius", float(np.abs(np.linalg.eigvals(jac)).max()))
    record_property("se_pred", se.tolist())
    record_property("mean_log_dev", mean.tolist())
    record_property("z", (mean/se).tolist())
    record_property("per_round_sd", np.std(devs, axis=0, ddof=1).tolist())
    assert fallbacks == []
    assert np.all(np.abs(mean) <= 4*se+v), (mean, se, v)


def test_mc_trajectory_from_an_off_start_regression(target, record_property):
    toy, z_max = target["toy"], target["z_max"]
    z0 = z_max+np.log([.2, 2., 2.])
    rng = np.random.default_rng(SEED_OFF)
    z, z_exact, trajectory = z0.copy(), z0.copy(), []
    for r in range(3):
        exact, _ = toy.step(z)
        mc, record = toy.step(z, rng=rng)
        innovation = toy.innovation_cov(lg.eta_of(z))
        sd, v = np.sqrt(np.diag(innovation)), np.diag(innovation)
        record_property(f"round{r}_mc_minus_exact_over_sd", ((mc-exact)/sd).tolist())
        assert np.all(np.abs(mc-exact) <= 4*sd+v), (r, mc-exact, sd)
        assert record["gm_fallback"] == []
        z = mc
        trajectory.append(z.tolist())
        z_exact, _ = toy.step(z_exact)
    start_err = np.abs(np.exp(z0-z_max)-1)
    exact_err = np.abs(np.exp(z_exact-z_max)-1)
    record_property("exact_rel_err_after_3", exact_err.tolist())
    record_property("mc_trajectory", trajectory)
    assert np.all(exact_err <= .5*start_err)                                 # the exact iterates approach the target
    assert PINNED_OFF_START is not None, f"pin the trajectory: {trajectory!r}"
    np.testing.assert_allclose(trajectory, PINNED_OFF_START, rtol=1e-6, atol=0)
