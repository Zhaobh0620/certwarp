"""T07 same-spline value/physical derivative, support and frozen ownership."""
import json

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.interpolation import build_image_sampler, sample_points, sample_value_gradient, check_map_domain


def polynomial(p):
    x, y, z = np.moveaxis(p, -1, 0)
    v = 1+.2*x-.3*y+.4*z+.07*x**3*y**2*z+.03*x*y*z**3+.11*y**3
    grad = np.stack((.2+.21*x**2*y**2*z+.03*y*z**3,
                     -.3+.14*x**3*y*z+.03*x*z**3+.33*y**2,
                     .4+.07*x**3*y**2+.09*x*y*z**2), axis=-1)
    return v, grad


def test_cubic_polynomial_chunk_ownership(grid3d, record_property):
    g = grid3d
    image, _ = polynomial(g.identity())
    original = image.copy()
    sampler = build_image_sampler(image, g)
    image[:] = -100  # sampler must own coefficients
    phi = g.identity()
    phi[1:-1, 1:-1, 1:-1] += np.array(g.spacing)*[.17, -.21, .13]
    before = phi.copy()
    result = sample_value_gradient(sampler, phi, chunk_size=11)
    expected, grad = polynomial(phi)
    errors = []
    for a, b in ((result.value, expected), (result.gradient, grad)):
        error = np.max(np.abs(a-b))/max(1., np.max(np.abs(b)))
        errors.append(float(error))
        assert error < 2e-12
        assert not a.flags.writeable
    record_property("spline_polynomial_errors", json.dumps(errors))
    other = sample_value_gradient(sampler, phi)
    np.testing.assert_array_equal(other.value, result.value)
    np.testing.assert_array_equal(other.gradient, result.gradient)
    np.testing.assert_allclose(sample_value_gradient(sampler, g.identity()).value, original, atol=2e-12, rtol=2e-12)
    values_only = sample_points(sampler, phi.reshape(-1, 3), with_gradient=False)
    np.testing.assert_array_equal(values_only.value, result.value.ravel())
    assert values_only.gradient is None
    assert not sampler._spline.c.flags.writeable
    np.testing.assert_array_equal(phi, before)


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_nonpolynomial_directional_difference(grid3d, seed, record_property):
    g = grid3d
    x, y, z = np.moveaxis(g.identity(), -1, 0)
    sampler = build_image_sampler(np.sin(1.7*x+.3*y-.4*z)+.2*np.cos(.4*x-1.3*y+.7*z), g)
    phi = g.identity()
    phi[1:-1, 1:-1, 1:-1] += np.array(g.spacing)*[.17, -.21, .13]
    direction = np.zeros_like(phi)
    direction[1:-1, 1:-1, 1:-1] = np.random.default_rng(seed).uniform(-.4, .4, g.interior_shape+(3,))
    analytic = np.sum(sample_value_gradient(sampler, phi).gradient*direction, axis=-1)
    eps = [1e-2, 5e-3, 2.5e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for t in eps:
        diff = (sample_value_gradient(sampler, phi+t*direction).value-sample_value_gradient(sampler, phi-t*direction).value)/(2*t)
        errors.append(float(np.max(np.abs(diff-analytic))/max(1., np.max(np.abs(analytic)))))
    record_property("epsilons", json.dumps(eps))
    record_property("spline_direction_errors", json.dumps(errors))
    ratio = np.array(errors[:2])/errors[1:3]
    assert np.all((ratio > 3.3) & (ratio < 4.7)) and min(errors) < 1e-7


def test_random_image_against_independent_power_basis_splines(grid3d, record_property):
    g = grid3d
    rng = np.random.default_rng(91)
    image = rng.normal(size=g.shape)
    sampler = build_image_sampler(image, g)
    points = np.stack([rng.uniform(axis[0], axis[-1], 19) for axis in g.axes], axis=-1)
    basis = [CubicSpline(axis, np.eye(len(axis)), bc_type="not-a-knot") for axis in g.axes]
    weights = [s(points[:, j]) for j, s in enumerate(basis)]
    derivatives = [s(points[:, j], 1) for j, s in enumerate(basis)]
    expected = np.einsum("qi,qj,qk,ijk->q", weights[2], weights[1], weights[0], image)
    gradient = []
    for j in range(3):
        w = weights.copy()
        w[j] = derivatives[j]
        gradient.append(np.einsum("qi,qj,qk,ijk->q", w[2], w[1], w[0], image))
    actual = sample_points(sampler, points, chunk_size=3)
    grad = np.stack(gradient, axis=-1)
    error = max(np.max(np.abs(expected-actual.value))/max(1., np.max(np.abs(expected))),
                np.max(np.abs(grad-actual.gradient))/max(1., np.max(np.abs(grad))))
    record_property("independent_power_basis_error", float(error))
    assert error < 2e-12


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("location", ["in", "on", "out"])
def test_support_before_sampling(grid3d, axis, upper, boundary, location, monkeypatch):
    g = grid3d
    sampler = build_image_sampler(polynomial(g.identity())[0], g)
    bound = g.axes[axis][-1 if upper else 0]
    value = bound
    if location != "on":
        sign = 1 if upper else -1
        value = np.nextafter(bound, (sign if location == "out" else -sign)*np.inf)
    phi = g.identity()
    node = (0, 2, 2) if boundary else (2, 2, 2)
    phi[node+(axis,)] = value
    valid = location == "in" or (boundary and location == "on")
    original = type(sampler._spline).__call__
    calls = []
    def tracked(self, *args, **kw):
        assert valid, "invalid query reached the spline"
        calls.append(1)
        return original(self, *args, **kw)
    monkeypatch.setattr(type(sampler._spline), "__call__", tracked)
    result = sample_value_gradient(sampler, phi)
    assert isinstance(result, InvalidTrial) != valid
    if not valid:
        assert result.reason == "INVALID_DOMAIN" and result.invalid_count == 1 and not calls


def test_map_domain_and_nonfinite(grid3d):
    g = grid3d
    u = np.zeros(g.shape+(3,))
    assert check_map_domain(u, g) is None
    u[2, 2, 2, 2] = g.axes[2][-1]-g.axes[2][2]
    assert check_map_domain(u, g).reason == "INVALID_DOMAIN"
    u[2, 2, 2, :] = np.nan
    assert check_map_domain(u, g).reason == "NONFINITE_STATE"
    sampler = build_image_sampler(polynomial(g.identity())[0], g)
    assert sample_value_gradient(sampler, g.identity()+u).reason == "NONFINITE_STATE"


@pytest.mark.parametrize("kind", ["shape", "bool", "complex", "chunk", "strict"])
def test_invalid_queries(grid3d, kind):
    sampler = build_image_sampler(polynomial(grid3d.identity())[0], grid3d)
    p, kw = grid3d.identity(), {}
    if kind == "shape":
        p = p[..., :2]
    elif kind == "bool":
        p = p.astype(bool)
    elif kind == "complex":
        p = p.astype(complex)
    elif kind == "chunk":
        kw["chunk_size"] = True
    else:
        kw["strict"] = np.zeros(grid3d.shape, dtype=int)
    with pytest.raises(InputValidationError):
        sample_points(sampler, p, **kw)
