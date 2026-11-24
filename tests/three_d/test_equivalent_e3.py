"""E3 monotone interval specialization and shared Q1 gather regressions."""
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest

from jd_curl.three_d import interval as iv
from jd_curl.three_d import feasibility as feas
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.geometry import compute_cell_geometry, cofactor


@pytest.mark.parametrize('weight', [0., np.nextafter(0., 1.), .25, .5, 1., 2., 1e100])
def test_nonnegative_scale_exact_and_rational(weight):
    lo = np.array([-1e100, -1., -1e-310, -0., 0., 1e-310, .1, 1.])
    hi = np.nextafter(lo, np.inf)
    value = iv.Interval(lo, hi)
    expected = value*weight
    actual = value.scale_nonnegative(weight)
    assert actual.lo.tobytes() == expected.lo.tobytes()
    assert actual.hi.tobytes() == expected.hi.tobytes()
    for k in range(lo.size):
        for x in (lo[k], hi[k]):
            exact = Fraction(float(x))*Fraction(float(weight))
            assert Fraction(float(actual.lo[k])) <= exact <= Fraction(float(actual.hi[k]))


def test_nonnegative_scale_broadcast_and_fail_closed():
    value = iv.Interval([[-2.], [0.], [.1]], [[-1.], [.1], [.3]])
    weights = np.array([0., .25, .5, 1.])
    for key in ('lo', 'hi'):
        assert getattr(value.scale_nonnegative(weights), key).tobytes() == getattr(value*weights, key).tobytes()
    for weight in (-1., np.nan, np.inf):
        with pytest.raises(iv.ArithmeticUncertain):
            value.scale_nonnegative(weight)
    with pytest.raises(iv.ArithmeticUncertain):
        iv.Interval.point(1e308).scale_nonnegative(2.)


@pytest.mark.parametrize('seed', [0, 13, 71])
@pytest.mark.parametrize('scale', [2.**-40, 1., 2.**40])
def test_certificate_matches_generic_interval_path(seed, scale, grid3d, monkeypatch):
    g = grid3d
    u = np.zeros(g.shape+(3,))
    u[1:-1,1:-1,1:-1] = np.random.default_rng(seed).normal(scale=.045, size=g.interior_shape+(3,))
    verts = np.random.default_rng(seed).normal(scale=.2*scale, size=(17,8,3))
    spacing = np.array([.2,.3,.4])*scale
    specialized = feas.jacobian_intervals(verts, spacing)
    result = feas.certify_map(u, g, config=feas.CertificateConfig(tile_size=7))
    monkeypatch.setattr(iv.Interval, 'scale_nonnegative', lambda self, factor: self*factor)
    generic = feas.jacobian_intervals(verts, spacing)
    other = feas.certify_map(u, g, config=feas.CertificateConfig(tile_size=7))
    assert specialized.lo.tobytes() == generic.lo.tobytes()
    assert specialized.hi.tobytes() == generic.hi.tobytes()
    assert asdict(result) == asdict(other)


@pytest.mark.parametrize('location', ['gauss27','corners8'])
@pytest.mark.parametrize('tile_size', [1,7,120])
def test_geometry_matches_individual_scalar_operators(grid3d, location, tile_size):
    g = grid3d
    ops = CellQ1Operators3D(g, location=location)
    rng = np.random.default_rng(14)
    u = rng.normal(scale=.04, size=g.shape+(3,))
    original = u.copy()
    for tile in cell_tiles(g, tile_size):
        expected = np.stack([np.stack([ops.derivative(u[...,a], axis, tile=tile)
                             for axis in ('x','y','z')], axis=-1) for a in range(3)], axis=-2)
        expected += np.eye(3)
        actual = compute_cell_geometry(u, ops, tile=tile)
        A = cofactor(expected)
        assert actual.F.tobytes() == expected.tobytes()
        assert actual.A.tobytes() == A.tobytes()
        assert actual.J.tobytes() == np.sum(expected[...,0,:]*A[...,0,:], axis=-1).tobytes()
        assert not any(v.flags.writeable for v in (actual.F, actual.A, actual.J, actual.curl))
    np.testing.assert_array_equal(u, original)
