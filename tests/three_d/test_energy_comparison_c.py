"""Independent high-precision checks of T13 (not an optimizer test)."""
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest

from jd_curl.energy_comparison import EnergyReduction
from jd_curl.three_d import energy_comparison as comparison
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.geometry import compute_cell_geometry
from jd_curl.three_d.problem import make_self_reference, with_controls
from jd_curl.three_d.residuals import TrialState3D, evaluate_trial, residual_blocks
from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, StateMismatchError
from .b_support import case


def dec(value):
    return Decimal.from_float(float(value))


def decimal_barrier_difference(a, b, delta):
    """Independent direct log expression, not the production rearrangement."""
    a, b, d = dec(a), dec(b), dec(delta)
    return (b-a)/(1-d)-((b-d)/(a-d)).ln()


@pytest.mark.parametrize("a,b", [
    (1., np.nextafter(1., 2.)), (1., np.nextafter(1., 0.)),
    (1.-1e-9, 1.+2e-9), (1.0000001, 1.000000100000001),
    (.2, .201), (1.03, 1.025), (.4, .9), (2., .01001),
    (.01+1e-10, np.nextafter(.01+1e-10, 1.)),
    (np.nextafter(.01, 1.), 1.), (1., np.nextafter(.01, 1.)),
    (1e308, np.nextafter(.01, 1.)), (np.nextafter(.01, 1.), 1e308),
])
def test_scalar_barrier_difference_against_decimal(a, b):
    with localcontext() as ctx:
        ctx.prec = 110
        expected = decimal_barrier_difference(a, b, .01)
        observed = comparison._barrier_difference(np.array([a]), np.array([b]), delta=.01)[0]
        relative = abs((Decimal(str(observed))-expected)/expected)
    assert relative < Decimal("2e-15")


@pytest.mark.parametrize("a,b,delta", [(.01, 1., .01), (1., .01, .01),
    (-1., 1., .01), (np.nan, 1., .01), (1., np.inf, .01),
    (1., 1., -.01), (2., 2., 1.)])
def test_scalar_difference_rejects_nonfinite_or_infeasible(a, b, delta):
    with pytest.raises(InputValidationError):
        comparison._barrier_difference(np.array([a]), np.array([b]), delta=delta)


def decimal_reduction(problem, base, trial, outer_only):
    """Exact-real objective difference of the evaluated float64 R0/J inputs.

    Geometry and interpolation were independently tested in A/B. Here square
    energies and 110-digit direct logarithms independently check the stable
    arithmetic; no production difference or scalar-barrier routine is used.
    """
    with localcontext() as ctx:
        ctx.prec = 110
        change = Decimal(0)
        for a, b in zip(residual_blocks(problem, base), residual_blocks(problem, trial), strict=True):
            if not outer_only or a.name == "image":
                change += sum((dec(y)**2-dec(x)**2 for x, y in zip(a.values.ravel(), b.values.ravel())), Decimal(0))/2
            if outer_only:
                break
        weight = dec(problem.weights.kappa)*dec(problem.grid.omega)/8
        for tile in cell_tiles(problem.grid, problem.memory.cell_tile_size):
            a = compute_cell_geometry(base.u, problem._corners, tile=tile).J
            b = compute_cell_geometry(trial.u, problem._corners, tile=tile).J
            change += weight*sum((decimal_barrier_difference(x, y, problem.certificate_config.delta)
                                  for x, y in zip(a.ravel(), b.ravel())), Decimal(0))
    return float(-change)


@pytest.fixture(scope="module")
def data():
    problem, state, _ = case(nu=.071)
    return problem, state


@pytest.mark.parametrize("outer_only", [False, True])
@pytest.mark.parametrize("step", [.2, 1e-10])
def test_full_stable_difference_against_decimal(data, step, outer_only):
    problem, base = data
    trial = evaluate_trial(problem, (1-step)*base.theta)
    assert isinstance(trial, TrialState3D)
    actual = comparison.stable_energy_reduction(problem, base, trial, outer_only=outer_only)
    expected = decimal_reduction(problem, base, trial, outer_only)
    assert isinstance(actual, EnergyReduction)
    np.testing.assert_allclose(actual.actual, expected, rtol=2e-13, atol=1e-28)
    energies = [s.energy.outer if outer_only else s.energy.total for s in (base, trial)]
    assert actual.resolution == 16*np.finfo(float).eps*max(energies)
    assert actual.certified == actual.actual-actual.resolution
    reverse = comparison.stable_energy_reduction(problem, trial, base, outer_only=outer_only)
    np.testing.assert_allclose(reverse.actual, -actual.actual, rtol=2e-13, atol=1e-28)


@pytest.mark.parametrize("outer_only", [False, True])
def test_equal_state_is_unresolved_not_a_descent(data, outer_only):
    problem, base = data
    same = evaluate_trial(problem, base.theta.copy())
    result = comparison.stable_energy_reduction(problem, base, same, outer_only=outer_only)
    assert result.actual == 0
    assert result.resolution > 0
    assert result.certified < 0


def test_foreign_problem_reference_or_uncertified_state_is_rejected(data):
    problem, base = data
    refreshed = with_controls(problem, make_self_reference(problem, base))
    refreshed_state = evaluate_trial(refreshed, base.theta)
    assert isinstance(refreshed_state, TrialState3D)
    uncertain = replace(base, certificate=replace(base.certificate, status="uncertain"))
    for trial in (refreshed_state, uncertain, replace(base, problem_id="foreign")):
        with pytest.raises(StateMismatchError):
            comparison.stable_energy_reduction(problem, base, trial)
        with pytest.raises(StateMismatchError):
            comparison.stable_energy_reduction(problem, trial, base)
    with pytest.raises(InputValidationError):
        comparison.stable_energy_reduction(problem, base, InvalidTrial("OUT_OF_DOMAIN", "test", 1))


@pytest.mark.parametrize("outer_only", [False, True])
def test_block_equivalence_and_tile_bounds(outer_only, monkeypatch):
    reductions = []
    original = comparison.compute_cell_geometry
    for tile_size in (1, 7, 10000):
        problem, base, _ = case(tile=tile_size, chunk=13)
        trial = evaluate_trial(problem, .83*base.theta)
        seen = []

        def bounded_geometry(u, ops, *, tile):
            seen.append(tile.stop-tile.start)
            return original(u, ops, tile=tile)

        with monkeypatch.context() as patch:
            patch.setattr(comparison, "compute_cell_geometry", bounded_geometry)
            reductions.append(comparison.stable_energy_reduction(problem, base, trial, outer_only=outer_only))
        assert max(seen) <= min(tile_size, problem.grid.n_cells)
    np.testing.assert_allclose([r.actual for r in reductions], reductions[0].actual, rtol=2e-13, atol=1e-28)
    np.testing.assert_allclose([r.resolution for r in reductions], reductions[0].resolution, rtol=2e-15)


def test_outer_difference_independent_of_soft_weights(data):
    problem, base = data
    weak, weak_base, _ = case(mu=.001, nu=.001)
    trial = evaluate_trial(problem, .85*base.theta)
    weak_trial = evaluate_trial(weak, .85*weak_base.theta)
    strong_f = comparison.stable_energy_reduction(problem, base, trial, outer_only=True)
    weak_f = comparison.stable_energy_reduction(weak, weak_base, weak_trial, outer_only=True)
    assert strong_f == weak_f
    assert comparison.stable_energy_reduction(problem, base, trial).actual != comparison.stable_energy_reduction(weak, weak_base, weak_trial).actual


def test_comparison_does_not_mutate_inputs_or_change_numpy_error_state(data):
    problem, base = data
    trial = evaluate_trial(problem, .7*base.theta)
    arrays = (problem.controls.u_ref, base.theta, base.u, base.warped, trial.theta, trial.u, trial.warped)
    before = [a.tobytes() for a in arrays]
    errors = np.geterr()
    comparison.stable_energy_reduction(problem, base, trial)
    comparison.stable_energy_reduction(problem, base, trial, outer_only=True)
    assert [a.tobytes() for a in arrays] == before
    assert all(not a.flags.writeable for a in arrays)
    assert np.geterr() == errors
