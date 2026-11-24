"""T01–T03 independent polynomial, cofactor and explicit transpose checks."""
from itertools import product
import json

import numpy as np
import pytest
from scipy.special import roots_legendre

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.boundary import inject_free, restrict_free, pack_initial_displacement
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles, cell_node_indices
from jd_curl.three_d.geometry import compute_cell_geometry, cofactor, determinant, cofactor_jvp
from jd_curl.validation import InputValidationError


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(1., np.linalg.norm(b)))


def test_grid_layout_mass_boundary(grid3d):
    g = grid3d
    rng = np.random.default_rng(17)
    theta = rng.normal(size=3*g.n_free_scalar)
    v = rng.normal(size=g.shape+(3,))
    u = inject_free(theta, g)
    np.testing.assert_array_equal(restrict_free(u, g), theta)
    np.testing.assert_array_equal(pack_initial_displacement(u, g), theta)
    for j in range(3):
        np.testing.assert_array_equal(u[1:-1, 1:-1, 1:-1, j].ravel(), theta[j*g.n_free_scalar:(j+1)*g.n_free_scalar])
    assert abs(np.sum(u*v) - theta@restrict_free(v, g)) < 2e-12
    np.testing.assert_allclose(restrict_free(g.node_weights()[..., None]*u, g), g.omega*theta, atol=1e-15)
    assert abs(g.node_weights().sum() - np.prod([a[-1]-a[0] for a in g.axes])) < 1e-14
    assert g.node_weights()[0, 0, 0] == g.omega/8
    phi = g.identity()
    assert phi.shape == (5, 6, 7, 3) and g.cell_shape == (4, 5, 6)
    for axis in range(3):
        for end in (0, -1):
            nodes = [slice(None)]*4
            nodes[axis] = end
            assert np.all(u[tuple(nodes)] == 0)
            bad = u.copy()
            bad[tuple(nodes)] = np.nextafter(0., 1.)
            with pytest.raises(InputValidationError):
                pack_initial_displacement(bad, g)
    for a in (*g.axes, *g.trapezoid):
        assert not a.flags.writeable


@pytest.mark.parametrize("key,value", [("nx", 4), ("ny", True), ("nz", 5.), ("hx", 0), ("hy", -1),
                                        ("hz", np.inf), ("x0", np.nan), ("x0", 1e20), ("unit", ""),
                                        ("hx", 1e308), ("hz", np.nextafter(0., 1.))])
def test_invalid_grid(key, value):
    kw = dict(nx=7, ny=6, nz=5, hx=.2, hy=.3, hz=.4)
    kw[key] = value
    with pytest.raises(InputValidationError):
        make_grid_3d(**kw)


def polynomial(p):
    x, y, z = np.moveaxis(p, -1, 0)
    f = 1+.3*x-.7*y+.2*z+.11*x*y-.13*x*z+.17*y*z+.19*x*y*z
    grad = np.stack((.3+.11*y-.13*z+.19*y*z, -.7+.11*x+.17*z+.19*x*z,
                     .2-.13*x+.17*y+.19*x*y), axis=-1)
    return f, grad


@pytest.mark.parametrize("location", ["gauss27", "corners8"])
def test_q1_polynomial_and_quadrature(grid3d, location, record_property):
    g, ops = grid3d, CellQ1Operators3D(grid3d, location=location)
    image, _ = polynomial(g.identity())
    original = image.copy()
    errors = []
    for tile in cell_tiles(g, 17):
        indices = cell_node_indices(g, tile)
        base = g.identity().reshape(-1, 3)[indices[:, 0]]
        points = base[:, None, :] + ops.points[None, :, :]*g.spacing
        expected, derivative = polynomial(points)
        errors.append(relative(ops.interpolate(image, tile=tile), expected))
        for j, axis in enumerate(("x", "y", "z")):
            errors.append(relative(ops.derivative(image, axis, tile=tile), derivative[..., j]))
    record_property("max_polynomial_error", max(errors))
    assert max(errors) < 2e-12
    assert abs(ops.weights.sum()-g.omega) < 1e-15
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("location", ["gauss27", "corners8"])
@pytest.mark.parametrize("axis", ["x", "y", "z", None])
def test_independent_dense_transpose_and_tiles(grid3d, location, axis, record_property):
    g, ops = grid3d, CellQ1Operators3D(grid3d, location=location)
    nq, N = len(ops.points), int(np.prod(g.shape))
    D = np.zeros((g.n_cells*nq, N))
    j = None if axis is None else ("x", "y", "z").index(axis)
    # Build a separate global matrix directly from analytical shape functions.
    for cid, (cz, cy, cx) in enumerate(np.ndindex(g.cell_shape)):
        for q, xi in enumerate(ops.points):
            for bz, by, bx in product((0, 1), repeat=3):
                bits = (bx, by, bz)
                factors = [(xi[k] if bits[k] else 1-xi[k]) if k != j
                           else (2*bits[k]-1)/g.spacing[k] for k in range(3)]
                node = np.ravel_multi_index((cz+bz, cy+by, cx+bx), g.shape)
                D[cid*nq+q, node] = np.prod(factors)
    rng = np.random.default_rng(91)
    f = rng.normal(size=g.shape)
    t = rng.normal(size=(g.n_cells, nq))
    full = slice(0, g.n_cells)
    action = ops.interpolate(f, tile=full) if axis is None else ops.derivative(f, axis, tile=full)
    out = np.zeros(g.shape)
    for tile in cell_tiles(g, 13):
        ops.derivative_transpose(t[tile], axis, tile=tile, accumulate_into=out)
    errs = [relative(action.ravel(), D@f.ravel()), relative(out.ravel(), D.T@t.ravel()),
            abs(np.sum(action*t)-np.sum(f*out))/max(1., abs(np.sum(action*t)))]
    whole = np.zeros(g.shape)
    ops.derivative_transpose(t, axis, tile=full, accumulate_into=whole)
    errs.append(relative(out, whole))
    record_property("transpose_errors", json.dumps(errs))
    assert max(errs) < 2e-12


@pytest.mark.parametrize("matrix", [np.eye(3), [[1.1, .2, -.1], [.3, .9, .1], [.1, -.2, 1.2]],
                                     [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                     [[1, .7, 0], [0, 1, .3], [0, 0, 1]],
                                     [[1, 2, 3], [1, 2, 3], [0, 1, 1]]])
@pytest.mark.parametrize("location", ["gauss27", "corners8"])
def test_affine_full_geometry(grid3d, matrix, location, record_property):
    F = np.asarray(matrix, dtype=float)
    g = grid3d
    u = g.identity()@(F-np.eye(3)).T + [.11, -.13, .09]
    geo = compute_cell_geometry(u, CellQ1Operators3D(g, location=location), tile=slice(0, g.n_cells))
    A = np.array([[(-1)**(i+j)*np.linalg.det(np.delete(np.delete(F, i, 0), j, 1))
                   for j in range(3)] for i in range(3)])
    C = np.array([F[2, 1]-F[1, 2], F[0, 2]-F[2, 0], F[1, 0]-F[0, 1]])
    errors = []
    for actual, expected in ((geo.F, F), (geo.A, A), (geo.J, np.linalg.det(F)), (geo.curl, C)):
        errors.append(relative(actual, np.broadcast_to(expected, actual.shape)))
        assert not actual.flags.writeable
    record_property("affine_geometry_errors", json.dumps(errors))
    assert max(errors) < 2e-12


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_det_cofactor_directions(seed, record_property):
    rng = np.random.default_rng(seed)
    F, V = rng.normal(size=(2, 3, 3)), rng.normal(size=(2, 3, 3))
    A = cofactor(F)
    expected = np.sum(A*V, axis=(-2, -1))
    eps = np.array([1e-2, 5e-3, 2.5e-3, 1e-4, 1e-5, 1e-6])
    errors = [relative((np.linalg.det(F+t*V)-np.linalg.det(F-t*V))/(2*t), expected) for t in eps]
    record_property("det_fd_errors", json.dumps(errors))
    assert np.all((np.array(errors[:2])/errors[1:3] > 3.3) & (np.array(errors[:2])/errors[1:3] < 4.7))
    assert min(errors) < 1e-7
    t = .17
    expansion = determinant(F)+t*expected+t*t*np.sum(F*cofactor(V), axis=(-2, -1))+t**3*determinant(V)
    assert relative(expansion, np.linalg.det(F+t*V)) < 2e-12
    assert relative(cofactor_jvp(F, V), (cofactor(F+1e-3*V)-cofactor(F-1e-3*V))/2e-3) < 2e-12


def test_trilinear_jd_integral_27_vs_independent_125(record_property):
    g = make_grid_3d(nx=7, ny=6, nz=5, hx=1/6, hy=1/5, hz=1/4)
    x, y, z = np.moveaxis(g.identity(), -1, 0)
    u = np.stack((x*y/3+y*z/4+x*z/5+x*y*z/7,
                  -x*z/3+y*z/5+x*y*z/7, x*y/4-x*z/6+y*z/8+x*y*z/9), axis=-1)
    ops = CellQ1Operators3D(g)
    actual = sum(float(np.sum((compute_cell_geometry(u, ops, tile=t).J-1)**2*ops.weights))
                 for t in cell_tiles(g, 19))
    p, w = roots_legendre(5)
    expected = 0.
    for i, j, k in product(range(5), repeat=3):
        x, y, z = (p[[i, j, k]]+1)/2
        F = np.array([[1+y/3+z/5+y*z/7, x/3+z/4+x*z/7, y/4+x/5+x*y/7],
                      [-z/3+y*z/7, 1+z/5+x*z/7, -x/3+y/5+x*y/7],
                      [y/4-z/6+y*z/9, x/4+z/8+x*z/9, 1-x/6+y/8+x*y/9]])
        expected += w[i]*w[j]*w[k]/8*(np.linalg.det(F)-1)**2
    record_property("jd_square_integral_error", abs(actual-expected))
    assert abs(actual-expected) < 2e-12
    assert abs(expected-1661594959541/4629441600000) < 2e-12


@pytest.mark.parametrize("location", ["gauss27", "corners8"])
def test_nonaffine_vector_curl_and_cofactor(grid3d, location, record_property):
    g, ops = grid3d, CellQ1Operators3D(grid3d, location=location)
    X = g.identity()
    p, _ = polynomial(X)
    factors = np.array([.17, -.23, .31])
    u = p[..., None]*factors
    max_error = 0.
    for tile in cell_tiles(g, 11):
        indices = cell_node_indices(g, tile)
        base = X.reshape(-1, 3)[indices[:, 0]]
        _, derivative = polynomial(base[:, None, :]+ops.points*g.spacing)
        F = np.eye(3) + factors[None, None, :, None]*derivative[..., None, :]
        curl = np.stack((F[..., 2, 1]-F[..., 1, 2], F[..., 0, 2]-F[..., 2, 0],
                         F[..., 1, 0]-F[..., 0, 1]), axis=-1)
        geo = compute_cell_geometry(u, ops, tile=tile)
        max_error = max(max_error, relative(geo.curl, curl), relative(geo.F, F), relative(geo.J, np.linalg.det(F)))
    record_property("nonaffine_geometry_error", max_error)
    assert max_error < 2e-12


@pytest.mark.parametrize("tile", [slice(None, 1), slice(0, 0), slice(-1, 2), slice(0, 2, 2), slice(0, 10000), 1])
def test_invalid_tiles(grid3d, tile):
    with pytest.raises(InputValidationError):
        cell_node_indices(grid3d, tile)


@pytest.mark.parametrize("kind", ["shape", "bool", "complex", "nan", "inf"])
def test_building_blocks_reject_invalid_arrays(grid3d, kind):
    from jd_curl.three_d.poisson import build_poisson_inverse
    g = grid3d
    def corrupt(a):
        if kind == "shape":
            return a[:-1]
        if kind in ("bool", "complex"):
            return a.astype(bool if kind == "bool" else complex)
        a.flat[0] = float(kind)
        return a
    inv = build_poisson_inverse(g, a=.1, b=1.)
    ops = CellQ1Operators3D(g)
    for action, data in ((inject_free, np.zeros(3*g.n_free_scalar)),
                          (restrict_free, np.zeros(g.shape+(3,))),
                          (lambda u, grid: compute_cell_geometry(u, ops, tile=slice(0, g.n_cells)), np.zeros(g.shape+(3,))),
                          (lambda v, grid: inv.apply(v), np.zeros(3*g.n_free_scalar))):
        with pytest.raises(InputValidationError):
            action(corrupt(data), g)
