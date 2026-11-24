"""Contract C15-9 prior side: bayesreg.infer.laplace.PriorSystem and prior_perturbation_rhs (samples of N(0, Pi_0^-1),
Pi_0 the GN precision of the three prior blocks at u = 0) against the independent dense oracle of
tests/test_bayesreg/model/_helpers, on the 6^3 case (3 N_int = 192).
  * the right-hand side equals the dense sum_i K_{i,0}^T eps_i with the C15-9 noise order (diffusion (192,) by the DST
    square root, also under diffusion_factor "explicit"; then per tile logvol, curl);
  * PriorSystem.hvp = dense Pi_0; the preconditioner is the C09.1 metric with c_dat = lam_b = gamma = 0;
  * gate (vi): the generalized eigenvalues of omega0 Pi_0 against K_Q lie in [min(a_min, b), max(a_max, b)] (B17-5);
  * gate (ii) (slow, S = 1e5): Cov(delta0) = Pi_0^-1 by the whitening test of C14 -- (i) |v^T C v - 1| <= 5 sqrt(2/S) on
    the axes and 20 fixed random directions, (ii) ||C - I||_F <= 1.2 sqrt(d (d + 1)/S), (iii) 1.1 Pi_0 must fail (i);
    right-hand sides formed in batches with the dense K_i (the streamed rhs is tied to them above), PCG with the dense
    Pi_0 and M^-1 (tied above), contract settings rtol 1e-3, maxiter 200; and the exact covariance of a sampler that
    omits or doubles a block fails (i) and (ii) by a wide margin (power, no sampling).
"""
import numpy as np
import pytest
import scipy.linalg

from jd_curl.validation import InputValidationError

from bayesreg import model
from bayesreg.infer import laplace as lp

from . import _cases_gauss27 as cs          # DEF: the C05-2 model ("gauss27", the default before DEF), explicit
from .test_whitening import _directions
from ..model._helpers import dense_model, free_random

S_WHITEN = 100_000
CHUNK = 1000
SEED = 20261030


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-300))


def prior_dense(state, grid):
    zero = np.zeros(grid.shape+(3,))
    K, _ = cs.dense_blocks(state, zero)
    K = {name: K[name] for name in ("diffusion", "logvol", "curl")}
    return K, sum(k.T@k for k in K.values())


@pytest.mark.parametrize("tile_size", [8192, 37])
def test_prior_rhs_matches_dense_oracle_and_noise_layout(tile_size, record_property):
    grid, state = cs.cube6(tile_size=tile_size)
    zero = np.zeros(grid.shape+(3,))
    ev = model.evaluate(state, zero)
    draw = cs.RecordingDraw(5)
    b = lp.prior_perturbation_rhs(state, ev, draw)
    K, _ = prior_dense(state, grid)
    expected = [("diffusion", (192,))]
    for cells in [min(tile_size, grid.n_cells-s) for s in range(0, grid.n_cells, tile_size)]:
        expected += [("logvol", (cells, 27)), ("curl", (cells, 27, 3))]
    assert [(name, shape) for name, shape, _ in draw.calls] == expected
    lam, Q = np.linalg.eigh(K["diffusion"].T@K["diffusion"])
    dense = K["logvol"].T@draw.concatenated("logvol")+K["curl"].T@draw.concatenated("curl") + \
        Q@(np.sqrt(np.clip(lam, 0, None))*(Q.T@draw.concatenated("diffusion")))
    err = rel(b, dense)
    record_property("prior_rhs_rel_err", err)
    assert err <= 1e-10


def test_prior_rhs_refuses_nonzero_u_and_foreign_evaluation():
    grid, state = cs.cube6()
    ev = model.evaluate(state, cs.u_hat_of(grid))
    with pytest.raises(InputValidationError, match="u = 0"):
        lp.prior_perturbation_rhs(state, ev, lambda name, shape: np.zeros(shape))
    ev0 = model.evaluate(state, np.zeros(grid.shape+(3,)))
    with pytest.raises(InputValidationError):
        lp.prior_perturbation_rhs(cs.scaled_state(grid, 2.), ev0, lambda name, shape: np.zeros(shape))


def test_prior_system_is_pi0_with_the_c09_metric_at_zero(record_property):
    grid, state = cs.cube6()
    system = lp.PriorSystem(state)
    _, Pi0 = prior_dense(state, grid)
    dense = cs.dense_operator(system.hvp, 192)
    err = float(np.max(np.abs(dense-Pi0))/np.max(np.abs(Pi0)))
    record_property("Pi0_vs_dense", err)
    assert err <= 1e-12
    M = model.build_metric(state, c_dat=0., b_P=model.metric_bP(state), lam_b=0., gamma=0.)
    v = free_random(grid, np.random.default_rng(1))
    assert np.array_equal(system.precondition(v), M.solve(v))
    record = system.record()
    assert record["metric"]["a"] == 0. and record["blocks"] == ["diffusion", "logvol", "curl"]
    assert record["diffusion_factor"].startswith("dst_sqrt")
    explicit = lp.PriorSystem(state, config=lp.LaplaceConfig(diffusion_factor="explicit"))
    draw = cs.RecordingDraw(3)
    explicit.rhs(draw)
    assert draw.calls[0][:2] == ("diffusion", (192,))                      # DST square root also under "explicit"
    items = lp.sample_perturbations(state, np.zeros(grid.shape+(3,)), 11, 4, system=system)
    assert all(item.converged for item in items)
    record_property("prior_pcg_iterations", items.iterations.tolist())


def test_pi0_spectral_bounds_b17_5(record_property):
    """Gate (vi): min(a_min, b) K_Q <= omega0 Pi_0 <= max(a_max, b) K_Q, a_q = alpha + 1/tau_q^2, b = alpha + nu."""
    for factor, nu in ((1., None), (1., 40.), (30., None)):
        grid, state = cs.cube6(factor=factor)
        if nu is not None:
            state = state.replace(nu=nu)
        K, Pi0 = prior_dense(state, grid)
        KQ = state.omega0/state.alpha*(K["diffusion"].T@K["diffusion"])
        tau_q2 = dense_model(state, np.zeros(grid.shape+(3,)))["tau2"]          # the oracle's tau_q^2
        a = state.alpha+1/tau_q2
        b = state.alpha+state.nu
        lo, hi = min(a.min(), b), max(a.max(), b)
        lam = scipy.linalg.eigh(state.omega0*Pi0, KQ, eigvals_only=True)
        record_property(f"factor{factor}_nu{nu}", dict(eig=[float(lam.min()), float(lam.max())], bound=[lo, hi]))
        assert lam.min() >= lo*(1-1e-10) and lam.max() <= hi*(1+1e-10)


def test_whitening_gate_power_for_the_prior_blocks():
    """A sampler omitting or doubling one prior block has Cov = Pi^-1 (Pi -/+ K_i^T K_i) Pi^-1: (i), (ii) flag it."""
    grid, state = cs.cube6()
    K, Pi0 = prior_dense(state, grid)
    d = Pi0.shape[0]
    Linv = scipy.linalg.solve_triangular(np.linalg.cholesky(Pi0), np.eye(d), lower=True)
    V = np.concatenate([np.eye(d), _directions(d)], axis=1)
    tol_i, tol_ii = 5*np.sqrt(2/S_WHITEN), 1.2*np.sqrt(d*(d+1)/S_WHITEN)
    for name, Kb in K.items():
        for factor in (-1., 1.):
            C = Linv@(Pi0+factor*Kb.T@Kb)@Linv.T
            dev = float(np.max(np.abs(np.einsum("ik,ij,jk->k", V, C, V)-1)))
            assert dev > 4*tol_i and float(np.linalg.norm(C-np.eye(d))) > 2*tol_ii, (name, factor)


@pytest.mark.slow                   # S = 1e5 samples, about 25 s
def test_prior_whitening_gate(record_property):
    grid, state = cs.cube6()
    system = lp.PriorSystem(state)
    K, Pi0 = prior_dense(state, grid)
    d = 3*grid.n_free_scalar
    assert np.max(np.abs(cs.dense_operator(system.hvp, d)-Pi0)) <= 1e-12*np.max(np.abs(Pi0))
    Minv = cs.dense_operator(system.precondition, d)
    B_D = np.sqrt(state.alpha/state.omega0)*cs.dense_operator(lambda e: lp.stiffness_sqrt_apply(e, grid), d)
    R = np.concatenate([K["logvol"].T, K["curl"].T], axis=1)
    L = np.linalg.cholesky(Pi0)
    rng = np.random.Generator(np.random.SFC64(SEED))
    second = np.zeros((d, d))
    eps_r, eps_d = np.empty((R.shape[1], CHUNK)), np.empty((d, CHUNK))
    max_iter, max_res = 0, 0.
    for _ in range(S_WHITEN//CHUNK):
        rhs = R@rng.standard_normal(out=eps_r)+B_D@rng.standard_normal(out=eps_d)
        res = lp.pcg(lambda X: Pi0@X, lambda X: Minv@X, rhs, rtol=1e-3, maxiter=200)
        assert np.all(res.converged)
        max_iter, max_res = max(max_iter, int(res.iterations.max())), max(max_res, float(res.rel_residual.max()))
        second += res.x@res.x.T
    cov = second/S_WHITEN
    C = L.T@cov@L
    tol_i, tol_ii = 5*np.sqrt(2/S_WHITEN), 1.2*np.sqrt(d*(d+1)/S_WHITEN)
    diag_dev = float(np.max(np.abs(np.diag(C)-1)))
    directions = _directions(d)
    proj_dev = float(np.max(np.abs(np.einsum("ik,ij,jk->k", directions, C, directions)-1)))
    frob = float(np.linalg.norm(C-np.eye(d)))
    wrong = np.abs(np.diag(np.linalg.cholesky(1.1*Pi0).T@cov@np.linalg.cholesky(1.1*Pi0))-1)
    record_property("axes_max_dev", diag_dev)
    record_property("random_max_dev", proj_dev)
    record_property("frobenius", frob)
    record_property("frobenius_expected", float(np.sqrt(d*(d+1)/S_WHITEN)))
    record_property("pcg_max_iterations", max_iter)
    record_property("pcg_max_rel_residual", max_res)
    assert diag_dev <= tol_i and proj_dev <= tol_i                         # (i)
    assert frob <= tol_ii                                                  # (ii)
    assert np.min(wrong) > tol_i                                           # (iii): every axis fails
