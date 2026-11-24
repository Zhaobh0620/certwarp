"""T05 rational inclusion oracles, counterexamples and fail-closed certificates."""
from fractions import Fraction as Q
from itertools import product, permutations

import numpy as np
import pytest

from jd_curl.three_d import interval as iv
from jd_curl.three_d.feasibility import (
    CertificateConfig, certify_cell, certify_map, jacobian_intervals,
)
from jd_curl.validation import InputValidationError


VERTICES = np.array([(x, y, z) for z, y, x in product((0., 1.), repeat=3)])


def fraction(x):
    return Q(float(x))


def contains(interval, exact):
    exact = np.asarray(exact, dtype=object)
    lo, hi, answer = np.broadcast_arrays(interval.lo, interval.hi, exact)
    for a, b, q in zip(lo.ravel(), hi.ravel(), answer.ravel()):
        assert fraction(a) <= q <= fraction(b), (a, q, b)


def exact_jacobian(u, spacing, point):
    """Independent shape-function derivative and permutation determinant."""
    t, h = list(map(fraction, point)), list(map(fraction, spacing))
    F = [[Q(r == j) for j in range(3)] for r in range(3)]
    for node, bits in enumerate(VERTICES.astype(int)):
        for j in range(3):
            weight = Q(2*int(bits[j])-1)/h[j]
            for k in range(3):
                if k != j:
                    weight *= t[k] if bits[k] else 1-t[k]
            for r in range(3):
                F[r][j] += fraction(u[node, r])*weight
    answer = Q(0)
    for p in permutations(range(3)):
        sign = (-1)**sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        answer += sign*F[0][p[0]]*F[1][p[1]]*F[2][p[2]]
    return answer


def exact_bernstein(values):
    T = [[Q(1), Q(0), Q(0)], [Q(-1, 2), Q(2), Q(-1, 2)], [Q(0), Q(0), Q(1)]]
    result = np.empty((3, 3, 3), dtype=object)
    for z, y, x in np.ndindex(3, 3, 3):
        result[z, y, x] = sum((T[z][k]*T[y][j]*T[x][i]*values[k, j, i]
                              for k, j, i in np.ndindex(3, 3, 3)), Q(0))
    return result


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_interval_operations_fraction_inclusion(operation):
    rng = np.random.default_rng(17)
    values = list(rng.uniform(-3, 3, 15)) + [0., .1, -.1, 1e-200, 1e200, np.nextafter(0., 1.), -1e-310]
    for x in values:
        for y in (0.1, -1.7, 1., np.nextafter(1., 2.)):
            a, b = iv.Interval.point(x), iv.Interval.point(y)
            A, B = fraction(x), fraction(y)
            if operation == "add":
                actual, expected = a+b, A+B
            elif operation == "sub":
                actual, expected = a-b, A-B
            elif operation == "mul":
                actual, expected = a*b, A*B
            else:
                actual, expected = a/b, A/B
            contains(actual, expected)
    # Non-degenerate endpoint intervals and broadcasting.
    a = iv.Interval(np.array([-2., .1, 1e-310]), np.array([1., .3, 2e-310]))
    b = iv.Interval(np.array(-1.7), np.array(-.7))
    result = {"add": a.__add__, "sub": a.__sub__, "mul": a.__mul__, "div": a.__truediv__}[operation](b)
    for k in range(3):
        for x, y in product((a.lo[k], a.hi[k]), (b.lo.item(), b.hi.item())):
            A, B = fraction(x), fraction(y)
            answer = {"add": lambda: A+B, "sub": lambda: A-B, "mul": lambda: A*B, "div": lambda: A/B}[operation]()
            contains(result[k], answer)


def test_arithmetic_rejects_unsafe_operations():
    assert iv.arithmetic_supported()
    with pytest.raises(iv.ArithmeticUncertain):
        iv.Interval.point(1)/iv.Interval(-1., 1.)
    with pytest.raises(iv.ArithmeticUncertain):
        iv.Interval.point(1e308)*2
    for lo, hi in ((np.nan, 1), (0, np.inf), (2, 1)):
        with pytest.raises(iv.ArithmeticUncertain):
            iv.Interval(lo, hi)
    contains(iv.Interval.point(np.nextafter(0., 1.))*.5, fraction(np.nextafter(0., 1.))/2)
    contains(iv.Interval.point(1.)-1., Q(0))


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_q1_intervals_and_bernstein_contain_rationals(seed, record_property):
    u = np.random.default_rng(seed).normal(scale=.07, size=(8, 3))
    spacing = (.2, .3, .4)
    intervals = jacobian_intervals(u[None], spacing)[0]
    exact = np.array([exact_jacobian(u, spacing, (x, y, z)) for z, y, x in product((0., .5, 1.), repeat=3)],
                     dtype=object).reshape(3, 3, 3)
    contains(intervals, exact)
    p = intervals-.01
    poly = exact-fraction(.01)
    coeff = iv.lagrange_to_bernstein(p)
    truth = exact_bernstein(poly)
    contains(coeff, truth)
    # Check the triquadratic representation at independent non-Lagrange points.
    for point in ((.13, .37, .61), (.83, .19, .43)):
        basis = [[(1-fraction(t))**2, 2*fraction(t)*(1-fraction(t)), fraction(t)**2] for t in point]
        reconstructed = sum((truth[k, j, i]*basis[0][i]*basis[1][j]*basis[2][k]
                             for k, j, i in np.ndindex(3, 3, 3)), Q(0))
        assert reconstructed == exact_jacobian(u, spacing, point)-fraction(.01)
    record_property("max_jacobian_interval_width", float(np.max(intervals.hi-intervals.lo)))
    for axis in range(3):
        left, right = iv.split_bernstein(coeff, axis)
        a, b, c = [np.take(truth, j, axis=-1-axis) for j in range(3)]
        mid = (a+2*b+c)/4
        contains(left, np.stack([a, (a+b)/2, mid], axis=-1-axis))
        contains(right, np.stack([mid, (b+c)/2, c], axis=-1-axis))


def counterexample(d):
    x, y, z = VERTICES.T
    phi = np.stack((x, y+2*(x-.5)*z, -2*(x-.5)*y+d*z), axis=-1)
    return phi-VERTICES


def test_corner_positive_but_interior_negative_witness(record_property):
    u = counterexample(-.1)
    assert all(exact_jacobian(u, (1, 1, 1), p) > fraction(.01) for p in VERTICES)
    result = certify_cell(u, (1, 1, 1))
    assert result.status == "infeasible" and result.n_boxes == 1
    witness = exact_jacobian(u, (1, 1, 1), result.witness_reference_point)
    assert witness <= fraction(.01)
    record_property("witness_J", float(witness))
    assert result.sampled_min_jd < 0


def test_negative_coefficient_positive_cell_requires_subdivision(record_property):
    u = counterexample(.1)
    good = certify_cell(u, (1, 1, 1))
    assert good.status == "certified_feasible" and good.n_boxes == 3 and good.max_depth == 1
    assert .01 < good.min_jd_lower_bound <= .1
    record_property("refined_lower_bound", good.min_jd_lower_bound)
    for cfg in (CertificateConfig(max_depth=0), CertificateConfig(max_boxes=1), CertificateConfig(max_boxes=2)):
        limited = certify_cell(u, (1, 1, 1), config=cfg)
        assert limited.status == "uncertain" and limited.n_boxes <= cfg.max_boxes
        assert limited.reason.endswith("budget")


def test_all_27_samples_positive_but_subdivision_finds_inversion(record_property):
    x, y, z = VERTICES.T
    phi = np.stack((x, y+2*(x-.25)*z, -2*(x-.25)*y-.0625*z), axis=-1)
    u = phi-VERTICES
    samples = [exact_jacobian(u, (1, 1, 1), p) for p in product((0., .5, 1.), repeat=3)]
    assert min(samples) > fraction(.01)
    result = certify_cell(u, (1, 1, 1))
    assert result.status == "infeasible" and result.reason == "subdivision_witness"
    witness = exact_jacobian(u, (1, 1, 1), result.witness_reference_point)
    assert witness <= fraction(.01) and result.max_depth >= 4
    assert result.sampled_min_jd > .01  # Root samples are explicitly not a certificate!
    record_property("positive_root_sample_min", float(min(samples)))
    record_property("subdivision_witness_J", float(witness))
    limited = certify_cell(u, (1, 1, 1), config=CertificateConfig(max_depth=3))
    assert limited.status == "uncertain", "unvisited negative point must not become false feasible"


@pytest.mark.parametrize("scale", [2.**-40, 1., 2.**40])
def test_interval_scale_and_translation_cancellation(scale):
    # Large constant displacements must cancel in physical derivatives.
    u = counterexample(.1)*scale + np.array([17., -31., 9.])*scale
    spacing = np.array([1., 1., 1.])*scale
    intervals = jacobian_intervals(u[None], spacing)[0]
    exact = np.array([exact_jacobian(u, spacing, (x, y, z)) for z, y, x in product((0., .5, 1.), repeat=3)],
                     dtype=object).reshape(3, 3, 3)
    contains(intervals, exact)
    assert certify_cell(u, spacing).status == "certified_feasible"


@pytest.mark.parametrize("offset,expected", [(-.03125, "infeasible"), (0., "uncertain"), (.03125, "certified_feasible")])
def test_threshold_equality_is_never_accepted(offset, expected):
    # Exact dyadic delta makes mathematical equality unambiguous despite float64 storage.
    u = np.zeros((8, 3))
    u[:, 0] = (.125+offset-1)*VERTICES[:, 0]
    assert exact_jacobian(u, (1, 1, 1), (.5, .5, .5)) == fraction(.125+offset)
    result = certify_cell(u, (1, 1, 1), config=CertificateConfig(delta=.125, max_depth=2, max_boxes=7))
    assert result.status == expected


@pytest.mark.parametrize("target", [.005, .01-1e-10, .01, .01+1e-10, .02])
def test_actual_float_threshold_and_physical_units(target):
    spacing = (.125, .25, .5)
    u = np.zeros((8, 3))
    u[:, 0] = (target-1)*VERTICES[:, 0]*spacing[0]
    exact = exact_jacobian(u, spacing, (.5, .5, .5))
    result = certify_cell(u, spacing, config=CertificateConfig(max_depth=2, max_boxes=7))
    if target < .01:
        assert result.status == "infeasible"
    elif target > .01:
        assert result.status == "certified_feasible"  # reference det would be scaled by cell volume!
    else:
        assert result.status == "uncertain"
    if result.status == "certified_feasible":
        assert exact > fraction(.01) and fraction(result.min_jd_lower_bound) <= exact


def test_map_safe_tiling_hash_and_counts(grid3d, record_property):
    g = grid3d
    u = np.zeros(g.shape+(3,))
    first = certify_map(u, g, config=CertificateConfig(tile_size=7))
    assert first.status == "certified_feasible" and first.n_cells_checked == g.n_cells
    assert first.n_boxes == g.n_cells and first.n_uncertain == first.n_infeasible == 0
    assert .99 < first.min_jd_lower_bound <= 1
    iz, iy, ix = np.indices(g.interior_shape)+1
    mode = np.sin(np.pi*ix/(g.shape[2]-1))*np.sin(np.pi*iy/(g.shape[1]-1))*np.sin(np.pi*iz/(g.shape[0]-1))
    u[1:-1, 1:-1, 1:-1] = mode[..., None]*[.013, -.017, .011]
    original = u.copy()
    result = certify_map(u, g, config=CertificateConfig(tile_size=13))
    full = certify_map(u, g, config=CertificateConfig(tile_size=g.n_cells))
    assert result.status == full.status == "certified_feasible"
    assert result.min_jd_lower_bound == full.min_jd_lower_bound
    assert result.n_boxes == full.n_boxes and result.sampled_min_jd == full.sampled_min_jd
    assert result.state_hash == full.state_hash != first.state_hash
    assert result.grid_id == g.grid_id and result.delta_hex == float(.01).hex()
    assert result.min_jd_lower_bound <= result.sampled_min_jd
    record_property("safe_map_jd_lower_bound", result.min_jd_lower_bound)
    np.testing.assert_array_equal(u, original)


def test_shared_node_jacobians_not_averaged(grid3d):
    u = np.zeros(grid3d.shape+(3,))
    u[2, 2, 2, 0] = 1.2*grid3d.spacing[0]
    result = certify_map(u, grid3d)
    assert result.status == "infeasible" and result.reason == "JACOBIAN_VIOLATION"
    assert result.n_infeasible > 0 and result.witness_cell is not None
    assert result.sampled_min_jd < 0
    assert result.n_uncertain + result.n_infeasible <= result.n_cells_checked


@pytest.mark.parametrize("kind", ["boundary", "domain", "nonfinite", "unsafe"])
def test_map_initial_rejections(grid3d, kind, monkeypatch):
    u = np.zeros(grid3d.shape+(3,))
    if kind == "boundary":
        u[0, 0, 0, 2] = 1e-20
    elif kind == "domain":
        u[2, 2, 2, 0] = 100
    elif kind == "nonfinite":
        u[2, 2, 2, 0] = np.nan
    else:
        monkeypatch.setattr(iv, "arithmetic_supported", lambda: False)
    result = certify_map(u, grid3d)
    assert result.status != "certified_feasible" and result.n_cells_checked == 0
    expected = dict(boundary="NONZERO_BOUNDARY", domain="INVALID_DOMAIN", nonfinite="NONFINITE_STATE", unsafe="unsupported_arithmetic")
    assert result.reason == expected[kind]


def test_arithmetic_overflow_is_uncertain():
    u = np.full((8, 3), 1e308)
    u[0] = -1e308
    result = certify_cell(u, (1e-300, .2, .3))
    assert result.status == "uncertain" and result.reason == "arithmetic_uncertain"
    assert result.min_jd_lower_bound is None


def test_map_tile_arithmetic_failure_counts_unchecked_cells(grid3d, monkeypatch):
    import jd_curl.three_d.feasibility as feasibility
    original = feasibility.jacobian_intervals
    calls = []
    def fail_second_tile(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise iv.ArithmeticUncertain("injected arithmetic failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(feasibility, "jacobian_intervals", fail_second_tile)
    result = certify_map(np.zeros(grid3d.shape+(3,)), grid3d, config=CertificateConfig(tile_size=7))
    assert result.status == "uncertain" and result.reason == "arithmetic_uncertain"
    assert result.n_cells_checked == grid3d.n_cells and result.n_uncertain == 7
    assert result.n_boxes == grid3d.n_cells-7 and result.n_infeasible == 0
    assert result.min_jd_lower_bound is None


def test_subdivision_arithmetic_failure_preserves_visited_cost(monkeypatch):
    def fail_split(*args, **kwargs):
        raise iv.ArithmeticUncertain("injected subdivision failure")
    monkeypatch.setattr(iv, "split_bernstein", fail_split)
    result = certify_cell(counterexample(.1), (1, 1, 1))
    assert result.status == "uncertain" and result.reason == "arithmetic_uncertain"
    assert result.n_boxes == 1 and result.max_depth == 0


@pytest.mark.parametrize("kw", [dict(delta=-.01), dict(delta=1), dict(delta=True), dict(max_depth=-1),
                                dict(max_depth=1.5), dict(max_boxes=0), dict(tile_size=True)])
def test_invalid_config(kw):
    with pytest.raises(InputValidationError):
        CertificateConfig(**kw)
