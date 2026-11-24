from dataclasses import replace

import numpy as np
import pytest

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, StateMismatchError
from jd_curl.three_d.problem import make_problem, make_self_reference, with_controls, Weights3D, MemoryConfig
from jd_curl.three_d.residuals import evaluate_trial, promote_trial, residual_blocks, validate_state
from jd_curl.three_d.linearization import (build_linearization, energy_gradient, barrier_ggn_hvp,
                                          barrier_ggn_diagonal, residual_jvp_blocks, residual_vjp_blocks)
from .b_support import case, pack, image


@pytest.mark.parametrize("mu", [1., .001])
def test_self_reference_exact_and_absolute_barrier(mu):
    p, s, lin = case(mu=mu)
    controls = make_self_reference(p, s)
    fresh = with_controls(p, controls)
    assert fresh.sampler is p.sampler and fresh.fixed_image is p.fixed_image
    assert fresh._gauss is p._gauss
    assert fresh.problem_id != p.problem_id and controls.controls_id != p.controls.controls_id
    t = promote_trial(fresh, evaluate_trial(fresh, s.theta))
    newlin = build_linearization(fresh, t)
    assert t.energy.proximity == t.energy.jd == t.energy.curl == 0.
    assert t.energy.total == t.energy.outer
    assert t.energy.barrier == s.energy.barrier > 0
    for b in residual_blocks(fresh, t):
        if b.name != "image":
            assert np.count_nonzero(b.values) == 0
    for name in ("proximity", "jd", "curl"):
        assert np.count_nonzero(energy_gradient(newlin, term=name)) == 0
    np.testing.assert_array_equal(energy_gradient(newlin), energy_gradient(newlin, term="outer"))
    # Strong/weak change penalties do not change F=D+B at the same map.
    other, os, _ = case(mu=.001 if mu == 1 else 1.)
    assert os.energy.outer == s.energy.outer
    with pytest.raises(StateMismatchError):
        build_linearization(fresh, s)
    with pytest.raises(StateMismatchError):
        promote_trial(other, s)
    with pytest.raises(StateMismatchError):
        residual_vjp_blocks(newlin, residual_blocks(p, s))


def test_owned_inputs_and_frozen_base():
    p, s, lin = case()
    fixed, moving, ref = p.fixed_image.copy(), image(p.grid.identity()), p.controls.u_ref.copy()
    fresh = make_problem(grid=p.grid, fixed_image=fixed, moving_image=moving, u_ref=ref, weights=p.weights)
    before = (fresh.fixed_image.copy(), fresh.controls.u_ref.copy())
    fixed[:] = 99
    moving[:] = 99
    ref[:] = 99
    np.testing.assert_array_equal(fresh.fixed_image, before[0])
    np.testing.assert_array_equal(fresh.controls.u_ref, before[1])
    theta = s.theta.copy()
    trial = evaluate_trial(fresh, theta)
    theta[:] = 999
    np.testing.assert_array_equal(trial.theta, s.theta)
    np.testing.assert_array_equal(trial.warped, s.warped)
    for a in (fresh.fixed_image, fresh.controls.u_ref, trial.theta, trial.u, trial.warped,
              s.image_grad, next(residual_blocks(p, s)).values):
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a.flat[0] = 0
    ref_before, u_before, grad_before = p.controls.u_ref.copy(), s.u.copy(), energy_gradient(lin)
    assert isinstance(evaluate_trial(p, np.full_like(s.theta, 100)), InvalidTrial)
    assert isinstance(evaluate_trial(p, np.full_like(s.theta, np.nan)), InvalidTrial)
    np.testing.assert_array_equal(p.controls.u_ref, ref_before)
    np.testing.assert_array_equal(s.u, u_before)
    np.testing.assert_array_equal(energy_gradient(lin), grad_before)
    # No full-volume Gauss cofactor or J/curl reference stored on the base/controls.
    assert set(vars(p.controls)) == {"u_ref", "certificate", "controls_id", "grid_id", "quadrature_id", "_ops"}
    assert set(vars(lin)) == {"problem", "state"}


def test_refuse_invalid_or_uncertain_before_sampling_and_barrier(monkeypatch):
    import jd_curl.three_d.residuals as r
    p, s, _ = case()
    def forbidden(*args, **kwargs):
        pytest.fail("invalid trial reached image/barrier")
    monkeypatch.setattr(r, "sample_points", forbidden)
    monkeypatch.setattr(r, "barrier_terms", forbidden)
    assert isinstance(evaluate_trial(p, np.full_like(s.theta, 100)), InvalidTrial)
    cert = replace(s.certificate, status="uncertain", reason="TEST_BUDGET", n_uncertain=1)
    monkeypatch.setattr(r, "certify_map", lambda *args, **kwargs: cert)
    result = evaluate_trial(p, s.theta)
    assert result.reason == "FEASIBILITY_UNCERTAIN"


@pytest.mark.parametrize("change", ["problem_id", "eval_id", "quadrature_id", "tile", "name", "missing", "extra", "duplicate", "order", "shape", "nan"])
def test_vjp_rejects_broken_stream(change):
    p, s, lin = case()
    blocks = list(residual_blocks(p, s))
    if change in ("problem_id", "eval_id", "quadrature_id", "name"):
        blocks[2] = replace(blocks[2], **{change: "wrong"})
    elif change == "tile":
        blocks[2] = replace(blocks[2], tile=(1, 65))
    elif change == "missing":
        blocks.pop()
    elif change == "extra":
        blocks.append(blocks[-1])
    elif change == "duplicate":
        blocks[3] = blocks[2]
    elif change == "order":
        blocks[2], blocks[3] = blocks[3], blocks[2]
    elif change == "shape":
        blocks[2] = replace(blocks[2], values=np.zeros(3))
    else:
        blocks[2] = replace(blocks[2], values=np.full_like(blocks[2].values, np.nan))
    with pytest.raises((StateMismatchError, InputValidationError)):
        residual_vjp_blocks(lin, blocks)


def test_cross_evaluation_and_controls_guard():
    p, s, lin = case()
    trial = evaluate_trial(p, s.theta*.5)
    with pytest.raises(StateMismatchError):
        residual_vjp_blocks(lin, residual_blocks(p, trial))
    with pytest.raises(InputValidationError):
        build_linearization(p, trial)
    with pytest.raises(InputValidationError):
        make_self_reference(p, trial)
    for key, value in (("grid_id", "wrong"), ("quadrature_id", "wrong"), ("u_ref", s.u)):
        with pytest.raises(StateMismatchError):
            with_controls(p, replace(p.controls, **{key: value}))
    with pytest.raises(StateMismatchError):
        validate_state(p, replace(s, certificate=replace(s.certificate, status="uncertain")))


@pytest.mark.parametrize("tile,chunk", [(1, 1), (7, 11), (64, 31)])
def test_block_vs_whole(tile, chunk):
    p, s, lin = case(tile=tile, chunk=chunk)
    whole, ws, wl = case(tile=p.grid.n_cells, chunk=int(np.prod(p.grid.shape)))
    direction = np.random.default_rng(91).normal(size=len(s.theta))
    pairs = [(list(vars(s.energy).values()), list(vars(ws.energy).values())),
             (pack(residual_blocks(p, s)), pack(residual_blocks(whole, ws))),
             (energy_gradient(lin), energy_gradient(wl)),
             (barrier_ggn_hvp(lin, direction), barrier_ggn_hvp(wl, direction)),
             (barrier_ggn_diagonal(lin), barrier_ggn_diagonal(wl))]
    for a, b in pairs:
        np.testing.assert_allclose(a, b, rtol=2e-11, atol=2e-16)


def test_generators_do_not_leak_numpy_errstate():
    p, s, lin = case()
    original = np.geterr()
    for generator in (residual_blocks(p, s), residual_jvp_blocks(lin, s.theta)):
        while True:
            try:
                next(generator)
                assert np.geterr() == original
            except StopIteration:
                break


def test_all_geometry_work_obeys_tile_bound(monkeypatch):
    import jd_curl.three_d.problem as pm
    import jd_curl.three_d.residuals as rm
    import jd_curl.three_d.linearization as lm
    p, s, lin = case(tile=7)
    calls = []
    original = pm.compute_cell_geometry
    def observe(u, ops, *, tile):
        calls.append((ops.location, tile.stop-tile.start))
        return original(u, ops, tile=tile)
    for module in (pm, rm, lm):
        monkeypatch.setattr(module, "compute_cell_geometry", observe)
    assert not isinstance(evaluate_trial(p, s.theta), InvalidTrial)
    for _ in residual_blocks(p, s):
        pass
    for _ in residual_jvp_blocks(lin, s.theta):
        pass
    residual_vjp_blocks(lin, residual_blocks(p, s))
    energy_gradient(lin)
    barrier_ggn_hvp(lin, s.theta)
    barrier_ggn_diagonal(lin)
    assert {name for name, size in calls} == {"gauss27", "corners8"}
    assert max(size for name, size in calls) <= 7 < p.grid.n_cells


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True])
@pytest.mark.parametrize("name", ["lam", "mu", "nu", "kappa"])
def test_invalid_weights(name, value):
    args = dict(lam=1., mu=1., nu=1., kappa=1.)
    args[name] = value
    with pytest.raises(InputValidationError):
        Weights3D(**args)


@pytest.mark.parametrize("value", [0, -1, 1.2, True])
def test_invalid_tile_size(value):
    with pytest.raises(InputValidationError):
        MemoryConfig(cell_tile_size=value)
