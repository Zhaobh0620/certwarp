"""T10 inverse and matrix action against separate sparse assembly."""
import numpy as np
import pytest
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.poisson import build_poisson_inverse, apply_poisson_operator
from jd_curl.validation import InputValidationError


def assembled(grid, metric):
    K, M, I = [], [], []
    for n, h in zip(grid.interior_shape[::-1], grid.spacing):
        K.append(diags([-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1])/h**2)
        M.append(diags([np.ones(n-1), 4*np.ones(n), np.ones(n-1)], [-1, 0, 1])/6)
        I.append(eye(n))
    mass = M if metric == "q1" else I
    return (kron(mass[2], kron(mass[1], K[0])) + kron(mass[2], kron(K[1], mass[0]))
            + kron(K[2], kron(mass[1], mass[0]))).tocsr()


@pytest.mark.parametrize("spec", [(7, 6, 5, .2, .3, .4), (5, 7, 6, .013, .7, .29)])
@pytest.mark.parametrize("metric", ["seven_point", "q1"])
@pytest.mark.parametrize("a", [0., .7])
@pytest.mark.parametrize("seed", [0, 17, 91])
def test_inverse_sparse_direct(spec, metric, a, seed, record_property):
    nx, ny, nz, hx, hy, hz = spec
    g = make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz)
    L = assembled(g, metric)
    M = g.omega*(a*eye(g.n_free_scalar)+.85*L)
    inv = build_poisson_inverse(g, a=a, b=.85, metric=metric)
    rhs = np.random.default_rng(seed).normal(size=(3, g.n_free_scalar))
    actual = inv.apply(rhs.ravel()).reshape(rhs.shape)
    expected = np.stack([spsolve(M.tocsr(), b) for b in rhs])
    residual = np.stack([M@v-b for v, b in zip(actual, rhs)])
    action = inv.matvec(rhs.ravel()).reshape(rhs.shape)
    err = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    res = np.linalg.norm(residual)/np.linalg.norm(rhs)
    act = np.linalg.norm(action-np.stack([M@b for b in rhs]))/np.linalg.norm(action)
    for name, v in (("inverse_error", err), ("true_residual", res), ("action_error", act)):
        record_property(name, float(v))
    assert max(err, res) < 1e-10 and act < 1e-11
    assert rhs.ravel()@actual.ravel() > 0
    assert not inv.denominator.flags.writeable


@pytest.mark.parametrize("metric", ["seven_point", "q1"])
def test_high_frequency_component_isolation(grid3d, metric):
    g = grid3d
    iz, iy, ix = np.indices(g.interior_shape)+1
    kz, ky, kx = g.interior_shape
    mode = (np.sin(np.pi*kz*iz/(kz+1))*np.sin(np.pi*ky*iy/(ky+1))*np.sin(np.pi*kx*ix/(kx+1)))
    rhs = np.zeros((3,)+g.interior_shape)
    rhs[1] = mode
    inv = build_poisson_inverse(g, a=.7, b=.85, metric=metric)
    out = inv.apply(rhs.ravel()).reshape(rhs.shape)
    np.testing.assert_array_equal(out[[0, 2]], 0)
    np.testing.assert_allclose(out[1], mode/inv.denominator[-1, -1, -1], atol=1e-12, rtol=1e-12)


def test_q1_not_seven_point(grid3d):
    f = np.random.default_rng(17).normal(size=grid3d.interior_shape)
    a = apply_poisson_operator(f, grid3d, metric="q1")
    b = apply_poisson_operator(f, grid3d, metric="seven_point")
    assert np.linalg.norm(a-b)/np.linalg.norm(b) > .1


@pytest.mark.parametrize("a,b,metric", [(-1, 1, "q1"), (1, 0, "q1"), (True, 1, "q1"),
                                         (1, np.inf, "q1"), (1, 1, "bad"), (1e308, 1e308, "q1")])
def test_invalid_inverse(grid3d, a, b, metric):
    with pytest.raises(InputValidationError):
        build_poisson_inverse(grid3d, a=a, b=b, metric=metric)
