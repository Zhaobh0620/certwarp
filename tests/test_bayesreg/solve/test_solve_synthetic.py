"""Synthetic end-to-end checks of the solver (no data/): intensity-only recovery with option (a), the
proximal term and option (b), the four-level run to B-line convergence with export.

Recovery and the aperture problem.  The data term constrains each node only along grad M at its warped
position; the components across it are fixed by the neighbours through the priors.  Recovery is therefore
measured only where the texture is rich -- structure tensor of the fixed image (Gaussian window 1.5 voxels)
with lambda_min / lambda_max >= .1 inside the mask, as contract C18 layer 1 asks -- and the deviation
that remains is a property of F_B (the MAP is not u*: F_B(u_hat) < F_B(u*), prior bias), not of the solver.
"""
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from jd_curl.three_d.boundary import inject_free, restrict_free

from bayesreg.model import ModelState, build_samplers, evaluate
from bayesreg.solve import (Caps, Commit, Eta, FixedSigma, GaussianMStep, NumpyBackend, ProximalParams, SolverConfig,
                            default_lambda, export_native, g0_of, solve_direct, solve_multilevel, solve_proximal,
                            threshold, voxels_to_displacement)
from bayesreg.solve import proximal as prox
from bayesreg.solve.multilevel import channel_mean_square, n_q
from bayesreg.solve.stopping import CONVERGED_B, state_hash

from . import _synthetic as syn


def make_state(grid, fixed, moving, mask, *, sigma2, alpha, nu=None, tau2=1.):
    return ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                      mask=mask, sigma2=np.array([sigma2]), alpha=alpha, nu=alpha if nu is None else nu,
                      tau2=np.full(4, tau2), tissue=syn.uniform_tissue(grid))


def solenoidal_problem(n, amplitude=.5):
    grid = syn.cube_grid(n)
    u_true = syn.solenoidal_displacement(grid, amplitude)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, u_true)
    return grid, fixed, moving, u_true


def start_commit(backend, grid):
    zero = np.zeros(grid.shape+(3,))
    return Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))


def texture_rich(image, mask, ratio=.1, window=1.5):
    gz, gy, gx = np.gradient(image)
    g = np.stack([gx, gy, gz])
    tensor = np.einsum("i...,j...->...ij", g, g)
    for i in range(3):
        for j in range(3):
            tensor[..., i, j] = gaussian_filter(tensor[..., i, j], window)
    eig = np.linalg.eigvalsh(tensor)
    return mask & (eig[..., 0] >= ratio*eig[..., -1])


def test_intensity_only_recovery_option_a(record_property):
    """Option (a), intensity only, 13^3 nodes (h = 1 mm), divergence-free u* (max |u*| = .5 voxel), noise
    free, sigma^2 = 1e-2, alpha = nu = 1e-3, tau_k = 1; 120 accepted steps (cap_reached: the threshold
    1e-8 + 1e-6 G0 is not reached in that budget; observed GF_B/T_B ~ 9e2 at the end).
    Gates: every accepted step certified and F_B strictly decreasing (certified stable comparison);
    GF_B down by > 100x; texture-rich nodes: mean |u - u*| < .08 voxel (u = 0: .18) and max < .3 voxel;
    the output recertified (CPU FP64) with its state_hash; F_B(u_hat) < F_B(u*)."""
    backend = NumpyBackend()
    grid, fixed, moving, u_true = solenoidal_problem(13)
    mask = syn.brain_box(grid, 1)
    state = make_state(grid, fixed, moving, mask, sigma2=1e-2, alpha=1e-3)
    start = start_commit(backend, grid)
    target = threshold(g0_of(backend, state))
    rows = []
    result = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=120), log=rows.append)
    accepted = [r for r in rows if r["kind"] == "accept"]
    candidates = [r for r in rows if r["kind"] == "candidate"]
    assert result.status == "cap_reached" and result.accepted == len(accepted) == 120
    assert all(r["cert_status"] == "certified_feasible" and r["certified"] > 0 for r in candidates if r["accepted"])
    assert all(r["energy"] is None for r in candidates if r["cert_status"] != "certified_feasible")
    energies = np.array(result.energies)
    assert np.all(np.diff(energies) < 0)
    norms = np.array(result.gradient_norms)
    assert norms[-1] < 1e-2*norms[0]
    rich = texture_rich(fixed, mask)
    assert rich.sum() >= .9*mask.sum()
    err = np.linalg.norm(result.commit.u-u_true, axis=-1)
    err0 = np.linalg.norm(u_true, axis=-1)
    record_property("recovery", dict(steps=result.accepted, gf_over_target=result.gradient_norm/target,
                                     mean_err=float(err[rich].mean()), max_err=float(err[rich].max()),
                                     mean_err_u0=float(err0[rich].mean()), F_start=energies[0], F_end=energies[-1]))
    assert err[rich].mean() < .08 and err[rich].max() < .3 and err0[rich].mean() > .15
    recert = backend.certify(result.commit.u, grid)
    assert recert.passed and recert.state_hash == result.commit.cert.state_hash == state_hash(result.commit.u)
    assert evaluate(state, u_true).energy > energies[-1]


# ---- proximal term and option (b) ------------------------------------------------------------------------

def test_proximal_term_identities_and_derivative():
    """C10 gate: P(v, v) = 0 and grad_u P(v, v) = 0 exactly; E_v(v) = F_B(v); the gradient matches a central
    difference; the stable change equals the difference of the longdouble totals."""
    grid, fixed, moving, u_true = solenoidal_problem(7, .4)
    state = make_state(grid, fixed, moving, syn.brain_box(grid, 1), sigma2=.1, alpha=.1)
    params = ProximalParams(lam=.7, mu=.3, nu2=.2)
    v = u_true
    pv = prox.evaluate(state, v, v, params)
    assert pv.total == 0 and all(value == 0 for value in pv.parts.values())
    assert not np.any(prox.gradient(state, pv, params))
    fb = evaluate(state, v)
    assert prox.combine(fb, pv).energy == fb.energy
    rng = np.random.default_rng(1)
    u = .6*v+0.02*syn.solenoidal_displacement(grid, 1.)
    w = restrict_free(syn.solenoidal_displacement(grid, 1.)*rng.uniform(.5, 1.5, grid.shape)[..., None], grid)
    pu = prox.evaluate(state, u, v, params)
    g = prox.gradient(state, pu, params)
    t = 1e-5
    plus = prox.evaluate(state, u+t*inject_free(w, grid), v, params).energy
    minus = prox.evaluate(state, u-t*inject_free(w, grid), v, params).energy
    fd = (plus-minus)/(2*t)
    assert abs(fd-g@w) <= 1e-7*max(1., abs(g@w))
    trial = prox.evaluate(state, .5*v, v, params)
    change = prox.change(state, pu, trial, params)
    total = sum(change.values(), np.longdouble(0))
    assert abs(float(total-(trial.total-pu.total))) <= 1e-13*max(1., float(pu.total))


def test_option_b_commits_and_matches_option_a(record_property):
    """C10 gate: every commit of option (b) logs both stable differences (E_v against E_{u_i}(u_i) = F_B(u_i),
    and F_B independently), both certified > 0; F_B strictly decreases over commits; on a strongly
    regularized 9^3 problem (b) and (a) both reach GF_B <= T_B and agree to 1e-5 mm."""
    backend = NumpyBackend()
    grid, fixed, moving, _ = solenoidal_problem(9, .4)
    state = make_state(grid, fixed, moving, syn.brain_box(grid, 1), sigma2=1., alpha=1.)
    start = start_commit(backend, grid)
    target = threshold(g0_of(backend, state))
    rows = []
    params = ProximalParams(lam=default_lambda(grid))
    b = solve_proximal(backend, state, start, target=target, caps=Caps(), params=params, log=rows.append)
    a = solve_direct(backend, state, start, target=target, caps=Caps())
    assert b.status == a.status == "converged_gradient"
    checks = [r for r in rows if r["kind"] == "commit_check"]
    assert len(checks) == b.commits >= 1 and all(r["committed"] for r in checks)
    for r in checks:
        for key in ("E_v", "F_B"):
            assert r[key]["certified"] > 0 and r[key]["reason"] is None
    assert np.all(np.diff(b.energies) < 0)
    assert b.gradient_norm <= target and a.gradient_norm <= target
    record_property("option_b", dict(commits=b.commits, accepted=b.accepted, a_steps=a.accepted,
                                     max_diff=float(np.max(np.abs(a.commit.u-b.commit.u)))))
    assert np.max(np.abs(a.commit.u-b.commit.u)) < 1e-5
    inner = [r for r in rows if r["kind"] == "candidate"]
    assert {r["objective"] for r in inner} == {"E_v"}
    assert max(r["iteration"] for r in inner) <= params.n_in


def test_gaussian_m_step_is_the_closed_form():
    """C11-2 (Gaussian, a = b = 0, w = 1): sigma^2 = <rho^2>_B/(N_q + 2), <rho^2>_B = sum q_a m_a rho_a^2/omega0."""
    grid, fixed, moving, u_true = solenoidal_problem(7, .4)
    mask = syn.brain_box(grid, 1)
    state = make_state(grid, fixed, moving, mask, sigma2=.37, alpha=.1)
    point = evaluate(state, .5*u_true)
    q = grid.node_weights().ravel()[state.node_index]
    direct = float(np.sum(q*point.channel.rho[0]**2, dtype=np.longdouble))/state.omega0
    assert channel_mean_square(state, point)[0] == pytest.approx(direct, rel=1e-13)
    assert n_q(state) == pytest.approx(float(np.sum(grid.node_weights()[mask]))/state.omega0, rel=1e-15)
    sigma = GaussianMStep().m_step(state, point)
    assert sigma[0] == pytest.approx(direct/(n_q(state)+2), rel=1e-13)


# ---- four levels ----------------------------------------------------------------------------------------

def test_four_levels_to_b_line_convergence(tmp_path, record_property):
    """C12/C13 on four_level_grids(26^3): u = 0 certified at level 0, every transfer accepted at k = 0,
    one round per level, F_B strictly decreasing within every round, native round GF_B <= T_B ->
    converged_regularized_gradient_B ("B 线收敛"); export: final_native.npz, disp.npz (h = 1: d -> u bitwise),
    cert.json from a fresh CPU certificate equal to the solver's."""
    native, levels, _ = syn.four_levels(26)
    backend = NumpyBackend()
    config = SolverConfig(rounds=(1, 1, 1, 1))
    rows = []
    result = solve_multilevel(backend, levels, eta=Eta(1., 1., (1., 1., 1., 1.)), sigma=FixedSigma([1.]), config=config,
                              log=rows.append)
    record_property("levels", [dict(level=r["level"], status=r["status"], accepted=r["accepted"],
                                    gf_over_t=r["GF_B_over_T_B"]) for r in result.rounds])
    assert result.status == CONVERGED_B and result.wording == "B 线收敛" and result.native
    assert [r["level"] for r in result.rounds] == [0, 1, 2, 3]
    assert all(r["status"] == "converged_gradient" and r["continuation"]["proceed"] for r in result.rounds)
    for r in result.rounds:
        assert np.all(np.diff(r["F_B_trace"]) < 0) and r["GF_B"] <= r["T_B"]
        assert r["T_B"] == 1e-8+1e-6*r["G0"]
    assert [t["accepted_k"] for t in result.transfers] == [0, 0, 0]
    assert result.commit.cert.passed
    exported = export_native(tmp_path, u=result.commit.u, grid=native, status=result.status, backend=backend,
                             solver_cert=result.commit.cert)
    cert = json.loads((tmp_path/"cert.json").read_text())
    assert cert["status"] == "certified_feasible" and cert["matches_solver_certificate"]
    assert cert["state_hash"] == exported["state_hash"] == state_hash(result.commit.u)
    with np.load(tmp_path/"final_native.npz") as z:
        assert z["u"].tobytes() == result.commit.u.tobytes() and str(z["state_hash"]) == cert["state_hash"]
    with np.load(tmp_path/"disp.npz") as z:
        assert set(z.files) == {"d", "meta"} and z["d"].shape == (3,)+native.shape and z["d"].dtype == np.float64
        assert voxels_to_displacement(z["d"], native).tobytes() == result.commit.u.tobytes()
