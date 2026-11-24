"""bayesreg.geom.q1 against contract C02, C05 and C06 (geometric gates), theory B7-1, B11, B13-1/2 and
review F1 (2026-10-24: validate once, tile-local kernels, required tile, zero-boundary jvp).

Small grids only (<= 8 nodes per axis, apart from one 42^3 grid that is only used to trigger the
tile=None limit).  Independent references: affine closed forms, a dense shape-function oracle built
from numpy's Gauss-Legendre rule (not from CellQ1Operators3D) with the cofactor from 2x2 minors, the
Kronecker assembly and analytic DST spectrum of T10, and the mainline functions at eb808be (verbatim
copies in bayesreg/geom/_mainline_copy.py, private symbols locked by source sha256 below).

Lives in the package tests/test_bayesreg/geom/ (with __init__.py), imported as
test_bayesreg.geom.test_geom_q1, so it never shadows the real bayesreg package.  The B11 test writes
docs/bayesreg/checks/NB11_curl_rank.json only when the environment variable BAYESREG_WRITE_CHECKS=1 is
set (and then only when its content changes); an ordinary test run never touches tracked files.
"""
from functools import lru_cache
import hashlib
import inspect
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh

import jd_curl.three_d.geometry as mainline_geometry
import jd_curl.three_d.linearization as mainline
from jd_curl.three_d.boundary import inject_free, restrict_free
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.validation import InputValidationError

import bayesreg.geom.q1 as q1
from bayesreg.geom import (MAX_WHOLE_CELLS, finish, gauss27, gauss_geometry, gauss_gradient, gauss_gradient_energy,
                           iter_geometry, jvp, node_field, scalar_field, stiffness_apply, tile_geometry,
                           tile_interpolate, tile_jvp, tile_vjp_accumulate, vjp, vjp_stream)
from bayesreg.geom import _mainline_copy as mcopy

MAINLINE_SHA256 = "e725e95cb6f85278c183b6da1e909926a270191e3785e5878b7a431c0b932387"   # linearization.py @ eb808be
FILE_SHA256 = {"cell_q1.py": "35d0bfae26f2d8c743141148be29bec01852b4f1a3334a89afd035575bf9c668",   # @ eb808be
               "geometry.py": "163e173598a4ebe6d98c755e508ea3401dd96f57a0027618470f40f107b99951"}
PRIVATE_SOURCE_SHA256 = {   # sha256 of inspect.getsource at eb808be; see bayesreg/geom/_mainline_copy.py
    "CellQ1Operators3D._transpose_add": "28e3ccf388b3b17bc53f1793f36d894b8fd56b8fc62b8533306dc836f04c10b6",
    "CellQ1Operators3D._apply": "fda422a4468889dd9c1f6a8b2ea1a8aa87481f517a3729e55b25dc40e2127315",
    "geometry._compute_cell_geometry": "4f7c493677d9d1ea632483ef5295e786e80c00216e78012fb81f7ce74a5674d2",
    "geometry.cofactor": "533e46ae7062e7b5f03985751a7f1914e60472d502ab1973b3f67b2b560901b8"}
NB11_JSON = Path(__file__).resolve().parents[3]/"docs"/"bayesreg"/"checks"/"NB11_curl_rank.json"
WRITE_CHECKS_ENV = "BAYESREG_WRITE_CHECKS"   # "1": the B11 test (re)writes NB11_JSON; otherwise read-only


def aniso():
    """The mainline tests/three_d grid3d fixture (7x6x5 nodes, anisotropic, off-origin)."""
    return make_grid_3d(nx=7, ny=6, nz=5, hx=.2, hy=.3, hz=.4, x0=-.3, y0=.2, z0=-.7, unit="mm")


def centered(nx=7, ny=6, nz=5, hx=.2, hy=.3, hz=.4):
    return make_grid_3d(nx=nx, ny=ny, nz=nz, hx=hx, hy=hy, hz=hz, x0=-(nx-1)*hx/2, y0=-(ny-1)*hy/2,
                        z0=-(nz-1)*hz/2, unit="mm")


B_GRID = dict(nx=6, ny=7, nz=8, hx=.9, hy=1., hz=1.2)   # 6x7x8 nodes, N_int = 120 (B11, B13)


def free_random(grid, rng, scale=1.):
    return scale*rng.normal(size=3*grid.n_free_scalar)


def gauss_points(grid, ops, rng):
    """Random Gauss-point arrays for one all-cell tile."""
    n = (grid.n_cells, len(ops.points))
    return rng.normal(size=n), rng.normal(size=n+(3,)), rng.normal(size=n+(3, 3))


def stacked_derivative(ops, v, tile):
    """V[..., r, j] from the mainline public ops.derivative, as in _geometry_jvp."""
    return np.stack([np.stack([ops.derivative(v[..., r], a, tile=tile) for a in ("x", "y", "z")], axis=-1)
                     for r in range(3)], axis=-2)


def oracle_derivatives(grid, npts):
    """Global shape-derivative matrices D[j] (|Q| x N) from the scalar Q1 formula, numpy's
    Gauss-Legendre rule on [0,1], point order z,y,x (x fastest); independent of CellQ1Operators3D."""
    xi, wi = np.polynomial.legendre.leggauss(npts)
    nodes, weights = (xi+1)/2, wi/2
    pz, py, px = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wz, wy, wx = np.meshgrid(weights, weights, weights, indexing="ij")
    pts = np.stack((px.ravel(), py.ravel(), pz.ravel()), axis=-1)
    P = len(pts)
    cz, cy, cx = np.indices(grid.cell_shape).reshape(3, -1)
    rows = np.arange(grid.n_cells)[:, None]*P+np.arange(P)[None, :]
    D = np.zeros((3, grid.n_cells*P, int(np.prod(grid.shape))))
    for vx, vy, vz in product((0, 1), repeat=3):
        vertex = (vx, vy, vz)
        node = np.ravel_multi_index((cz+vz, cy+vy, cx+vx), grid.shape)
        for j in range(3):
            value = np.full(P, (2*vertex[j]-1)/grid.spacing[j])
            for k in range(3):
                if k != j:
                    value = value*(pts[:, k] if vertex[k] else 1-pts[:, k])
            D[j, rows, node[:, None]] += value[None, :]
    return D, np.tile(grid.omega*(wz*wy*wx).ravel(), grid.n_cells)


def oracle_cofactor(F):
    """cof F from signed 2x2 minors (tests/three_d/b_support.dense_geometry), independent of geometry.cofactor."""
    A = np.empty_like(F)
    for r, j in product(range(3), repeat=2):
        rr, jj = [k for k in range(3) if k != r], [k for k in range(3) if k != j]
        A[:, r, j] = (-1.)**(r+j)*(F[:, rr[0], jj[0]]*F[:, rr[1], jj[1]]-F[:, rr[0], jj[1]]*F[:, rr[1], jj[0]])
    return A


def interior_rows(grid):
    return np.arange(int(np.prod(grid.shape))).reshape(grid.shape)[1:-1, 1:-1, 1:-1].ravel()


def relmax(a, b):
    return float(np.max(np.abs(a-b))/max(1., np.max(np.abs(b))))


def tridiag(n, diagonal, off):
    return diagonal*np.eye(n)+off*(np.eye(n, k=1)+np.eye(n, k=-1))


def t10_eigenvalues(grid):
    """All eigenvalues omega (l_x m_y m_z + m_x l_y m_z + m_x m_y l_z) of the scalar K_Q (T10)."""
    factors = []
    for n, h in zip(grid.interior_shape[::-1], grid.spacing):   # x, y, z
        theta = np.pi*np.arange(1, n+1)/(n+1)
        factors.append(((2*np.sin(theta/2)/h)**2, (2+np.cos(theta))/3))
    (lx, mx), (ly, my), (lz, mz) = factors
    lam = (lx[None, None, :]*my[None, :, None]*mz[:, None, None]+mx[None, None, :]*ly[None, :, None]*mz[:, None, None]
           + mx[None, None, :]*my[None, :, None]*lz[:, None, None])
    return grid.omega*lam.ravel()


# ----------------------------------------------------------------------------------------------
# C02: point set, affine exactness, consistency with the mainline geometry
# ----------------------------------------------------------------------------------------------

def test_points_order_and_weights():
    """C02 gate: points are the z,y,x product (x fastest); sum of weights = omega to 1e-15."""
    g = aniso()
    ops = gauss27(g)
    nodes = (.5-np.sqrt(15.)/10, .5, .5+np.sqrt(15.)/10)
    for k, (iz, iy, ix) in enumerate(product(range(3), repeat=3)):
        assert tuple(ops.points[k]) == (nodes[ix], nodes[iy], nodes[iz])
    assert abs(ops.weights.sum()-g.omega) <= 1e-15*g.omega
    assert len(ops.points) == 27 and ops.location == "gauss27"


MATRICES = [np.eye(3), [[1.1, .2, -.1], [.3, .9, .1], [.1, -.2, 1.2]], [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            [[1, .7, 0], [0, 1, .3], [0, 0, 1]], [[.8, -.3, .25], [.15, 1.3, -.4], [-.2, .35, .7]]]


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("make", [aniso, centered], ids=["aniso", "centered"])
def test_affine_exact(matrix, make, record_property):
    """C02 gate: u = (M - I)x + b (nonzero faces, explicit allow_nonzero_faces): |J - det M| <= 1e-14
    max(1,|det M|), |A - cof M|_inf <= 1e-14, |curl - curl_M| <= 1e-14; also grad = M - I, div = tr M - 3."""
    M = np.asarray(matrix, dtype=float)
    g = make()
    u = g.identity()@(M-np.eye(3)).T+[.11, -.13, .09]
    geo = gauss_geometry(u, gauss27(g), allow_nonzero_faces=True)
    cof = np.array([[(-1)**(i+j)*np.linalg.det(np.delete(np.delete(M, i, 0), j, 1)) for j in range(3)]
                    for i in range(3)])
    curl = np.array([M[2, 1]-M[1, 2], M[0, 2]-M[2, 0], M[1, 0]-M[0, 1]])
    det = np.linalg.det(M)
    errors = dict(J=float(np.max(np.abs(geo.J-det))/max(1., abs(det))), A=float(np.max(np.abs(geo.A-cof))),
                  curl=float(np.max(np.abs(geo.curl-curl))), grad=float(np.max(np.abs(geo.grad-(M-np.eye(3))))),
                  div=float(np.max(np.abs(geo.div-(np.trace(M)-3)))))
    record_property("affine_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-14, errors
    for name in ("F", "A", "J", "curl", "grad", "div"):
        assert not getattr(geo, name).flags.writeable


@pytest.mark.parametrize("faces", ["random", "zero"])
def test_geometry_matches_mainline_bitwise(faces):
    """Review F1 items 1, 4: tile_geometry computes grad once, F = grad + I, cofactor, the mainline J and
    curl expressions; F, A, J, curl equal geometry._compute_cell_geometry and the public
    compute_cell_geometry bitwise on every tile and on all cells; grad equals the stacked ops.derivative
    bitwise; tiles reproduce the all-cell arrays; tile_interpolate equals ops.interpolate bitwise (C02-4)."""
    g = aniso()
    ops = gauss27(g)
    rng = np.random.default_rng(5)
    if faces == "random":
        u = rng.normal(size=g.shape+(3,))*.03
        field = node_field(u, g, allow_nonzero_faces=True)
    else:
        u = inject_free(free_random(g, rng, .03), g)
        field = node_field(u, g)
    assert field is u   # already a C-order float64 node field: validated, not copied
    whole = slice(0, g.n_cells)
    tiles = list(cell_tiles(g, 17))
    s = rng.normal(size=g.shape)
    sf = scalar_field(s, g)
    for tile in [whole]+tiles:
        geo = tile_geometry(field, ops, tile=tile)
        for ref in (mainline_geometry._compute_cell_geometry(u, ops, tile=tile),
                    mainline_geometry.compute_cell_geometry(u, ops, tile=tile)):
            for name in ("F", "A", "J", "curl"):
                assert np.array_equal(getattr(geo, name), getattr(ref, name)), name
        assert np.array_equal(geo.F, geo.grad+np.eye(3))
        assert np.array_equal(geo.grad, stacked_derivative(ops, u, tile))
        assert np.array_equal(tile_interpolate(sf, ops, tile=tile), ops.interpolate(s, tile=tile))
    full = gauss_geometry(u, ops, allow_nonzero_faces=True)
    parts = [tile_geometry(field, ops, tile=t) for t in tiles]
    for name in ("F", "A", "J", "curl", "grad", "div"):
        assert np.array_equal(getattr(full, name), getattr(tile_geometry(field, ops, tile=whole), name))
        np.testing.assert_allclose(np.concatenate([getattr(p, name) for p in parts]), getattr(full, name),
                                   rtol=1e-15, atol=1e-15)
    theta = free_random(g, rng, .03)
    assert np.array_equal(gauss_gradient(theta, ops), gauss_gradient(inject_free(theta, g), ops))


def test_dense_oracle_gradient_and_quadrature(record_property):
    """C02 gate: Gauss-point gradients equal the independent oracle; sum_q omega w_g |grad u_h|^2
    equals the 5x5x5 Gauss value to 1e-12 (polynomial exactness, B7-1)."""
    g = aniso()
    ops = gauss27(g)
    u = inject_free(free_random(g, np.random.default_rng(11), .05), g)
    D3, w3 = oracle_derivatives(g, 3)
    D5, w5 = oracle_derivatives(g, 5)
    flat = u.reshape(-1, 3)
    G3 = np.einsum("jqn,nr->qrj", D3, flat)
    G5 = np.einsum("jqn,nr->qrj", D5, flat)
    grad = gauss_geometry(u, ops).grad.reshape(-1, 3, 3)
    e27 = gauss_gradient_energy(u, ops)
    e125 = float(np.sum(w5[:, None, None]*G5**2))
    errors = dict(grad=relmax(grad, G3), weights=relmax(np.tile(ops.weights, g.n_cells), w3),
                  energy_125=abs(e27-e125)/e125)
    record_property("oracle_errors", json.dumps(errors))
    assert errors["grad"] <= 1e-13 and errors["weights"] <= 1e-15 and errors["energy_125"] <= 1e-12


# ----------------------------------------------------------------------------------------------
# Mainline at eb808be: verbatim copies, locked private dependencies, bitwise new paths
# ----------------------------------------------------------------------------------------------

def test_mainline_copy_source_and_hash():
    path = Path(mainline.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MAINLINE_SHA256
    lines = path.read_text().splitlines()
    for name, (first, last) in dict(_geometry_jvp=(40, 45), _geometry_flux=(48, 57), _scatter_geometry=(60, 66),
                                    _finish=(84, 88)).items():
        source = inspect.getsource(getattr(mcopy, name))
        assert source == inspect.getsource(getattr(mainline, name))
        assert source.rstrip("\n").splitlines() == lines[first-1:last]
    assert finish is mcopy._finish


def test_private_dependencies_locked():
    """Review F1 item 1: the mainline private symbols bayesreg.geom relies on (listed in _mainline_copy.py)
    and the files holding them are those of eb808be (source sha256)."""
    folder = Path(mainline_geometry.__file__).parent
    for name, digest in FILE_SHA256.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    objects = {"CellQ1Operators3D._transpose_add": CellQ1Operators3D._transpose_add,
               "CellQ1Operators3D._apply": CellQ1Operators3D._apply,
               "geometry._compute_cell_geometry": mainline_geometry._compute_cell_geometry,
               "geometry.cofactor": mainline_geometry.cofactor}
    for name, obj in objects.items():
        assert hashlib.sha256(inspect.getsource(obj).encode()).hexdigest() == PRIVATE_SOURCE_SHA256[name], name


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_mainline_copy_bitwise(seed):
    """C06 gate: copies equal the eb808be originals bitwise; the tile-local paths equal both bitwise:
    tile_jvp's (dJ, dcurl) = _geometry_jvp and dgrad = its V; tile_vjp_accumulate without ggrad =
    _scatter_geometry (one flux, the same nine scatters); finish = _finish = restrict_free."""
    g = aniso()
    ops = gauss27(g)
    rng = np.random.default_rng(seed)
    v = rng.normal(size=g.shape+(3,))
    base = rng.normal(size=(3,)+g.shape)
    outs = [base.copy() for _ in range(5)]
    for tile in cell_tiles(g, 37):
        n = (tile.stop-tile.start, 27)
        A, jd, curl = rng.normal(size=n+(3, 3)), rng.normal(size=n), rng.normal(size=n+(3,))
        new = tile_jvp(v, ops, tile=tile, A=A)
        for a, b, c in zip(mcopy._geometry_jvp(ops, A, v, tile), mainline._geometry_jvp(ops, A, v, tile), new):
            assert np.array_equal(a, b) and np.array_equal(a, c)
        assert np.array_equal(new[2], stacked_derivative(ops, v, tile))
        for c in (curl, None):
            assert np.array_equal(mcopy._geometry_flux(A, jd, c), mainline._geometry_flux(A, jd, c))
        mcopy._scatter_geometry(ops, A, jd, curl, tile, outs[0], also_into=outs[2])
        mainline._scatter_geometry(ops, A, jd, curl, tile, outs[1], also_into=outs[3])
        tile_vjp_accumulate(outs[4], ops, tile=tile, gJ=jd, gcurl=curl, A=A)
        mcopy._scatter_geometry(ops, A, jd, None, tile, outs[0])
        mainline._scatter_geometry(ops, A, jd, None, tile, outs[1])
        tile_vjp_accumulate(outs[4], ops, tile=tile, gJ=jd, A=A)
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[2], outs[3])
    assert np.array_equal(outs[4], outs[1])
    assert np.array_equal(mcopy._finish(outs[0]), mainline._finish(outs[1]))
    assert np.array_equal(finish(outs[4]), mainline._finish(outs[1]))
    assert np.array_equal(mcopy._finish(outs[0]), restrict_free(np.moveaxis(outs[0], 0, -1), g))


# ----------------------------------------------------------------------------------------------
# C06: jvp central differences, jvp/vjp adjoint, dense oracle, tiles and streams
# ----------------------------------------------------------------------------------------------

def feasible_state(g, seed):
    rng = np.random.default_rng(seed)
    scale = .08*min(g.spacing)
    u = inject_free(free_random(g, rng, scale), g)
    v = inject_free(free_random(g, rng, scale), g)
    return u, v


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_jvp_central_difference(seed, record_property):
    """C06 gate: |(r(u+tv) - r(u-tv))/2t - K v|_inf <= 1e-7 max(1, |K v|_inf) for J, curl, grad."""
    g = aniso()
    ops = gauss27(g)
    u, v = feasible_state(g, seed)
    geo = gauss_geometry(u, ops)
    assert geo.J.min() > .5
    dJ, dcurl, dgrad = jvp(u, v, ops)
    t = 1e-4
    plus, minus = gauss_geometry(u+t*v, ops), gauss_geometry(u-t*v, ops)
    errors = {}
    for name, d in (("J", dJ), ("curl", dcurl), ("grad", dgrad)):
        fd = (getattr(plus, name)-getattr(minus, name))/(2*t)
        errors[name] = float(np.max(np.abs(fd-d))/max(1., np.max(np.abs(d))))
    record_property("fd_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-7, errors
    assert np.max(np.abs(dJ)) > 1e-2   # nontrivial direction


@pytest.mark.parametrize("seed", [0, 17, 91])
def test_jvp_vjp_adjoint(seed, record_property):
    """C06 gate: |<K v, r> - <v, K^T r>| <= 1e-12 |K v| |r| per output and combined, on U0."""
    g = aniso()
    ops = gauss27(g)
    u, _ = feasible_state(g, seed)
    rng = np.random.default_rng(seed+1)
    theta = free_random(g, rng)
    outs = jvp(u, theta, ops)
    grads = gauss_points(g, ops, rng)
    errors = {}
    for k, name in enumerate(("J", "curl", "grad")):
        args = [None, None, None]
        args[k] = grads[k]
        back = vjp(u, *args, ops)
        lhs, rhs = float(np.sum(outs[k]*grads[k])), float(theta@back)
        errors[name] = abs(lhs-rhs)/(np.linalg.norm(outs[k])*np.linalg.norm(grads[k]))
    total = sum(float(np.sum(o*r)) for o, r in zip(outs, grads))
    errors["all"] = abs(total-float(theta@vjp(u, *grads, ops)))/(
        np.sqrt(sum(np.sum(o**2) for o in outs))*np.sqrt(sum(np.sum(r**2) for r in grads)))
    record_property("adjoint_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors


def test_dense_oracle_jvp_vjp(record_property):
    """C06 gate: jvp and vjp equal the dense oracle (A : D v, curl, D v and their transposes) to 1e-12,
    with F = D u + I and A = cof F from 2x2 minors, independent of geometry.cofactor (review F1 item 6);
    the module's A and J match that oracle too."""
    g = aniso()
    ops = gauss27(g)
    u, v = feasible_state(g, 3)
    D, _ = oracle_derivatives(g, 3)
    F = np.einsum("jqn,nr->qrj", D, u.reshape(-1, 3))+np.eye(3)
    A = oracle_cofactor(F)
    geo = gauss_geometry(u, ops)
    V = np.einsum("jqn,nr->qrj", D, v.reshape(-1, 3))
    expected = (np.sum(A*V, axis=(-2, -1)),
                np.stack((V[:, 2, 1]-V[:, 1, 2], V[:, 0, 2]-V[:, 2, 0], V[:, 1, 0]-V[:, 0, 1]), axis=-1), V)
    actual = [a.reshape(e.shape) for a, e in zip(jvp(u, v, ops), expected)]
    gJ, gcurl, ggrad = gauss_points(g, ops, np.random.default_rng(4))
    gc, gg = gcurl.reshape(-1, 3), ggrad.reshape(-1, 3, 3)
    flux = A*gJ.reshape(-1)[:, None, None]+gg
    flux[:, 2, 1] += gc[:, 0]
    flux[:, 1, 2] -= gc[:, 0]
    flux[:, 0, 2] += gc[:, 1]
    flux[:, 2, 0] -= gc[:, 1]
    flux[:, 1, 0] += gc[:, 2]
    flux[:, 0, 1] -= gc[:, 2]
    nodes = np.einsum("jqn,qrj->nr", D, flux)
    back_expected = np.moveaxis(nodes.reshape(g.shape+(3,))[1:-1, 1:-1, 1:-1], -1, 0).ravel()
    errors = dict(A=relmax(geo.A.reshape(-1, 3, 3), A), J=relmax(geo.J.ravel(), np.linalg.det(F)),
                  dJ=relmax(actual[0], expected[0]), dcurl=relmax(actual[1], expected[1]),
                  dgrad=relmax(actual[2], expected[2]), vjp=relmax(vjp(u, gJ, gcurl, ggrad, ops), back_expected))
    record_property("dense_oracle_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors


def test_tiles_and_streams():
    """Tiled jvp/vjp equal the all-cell versions; iter_geometry yields cell_tiles in ascending order with
    the tile kernels' geometry; tile_jvp on fields validated once equals the per-call jvp bitwise; the
    tile loop of tile_vjp_accumulate + finish equals vjp, and vjp_stream equals that loop bitwise; vjp
    with a given A equals vjp computing it; free-vector and node-field inputs agree bitwise."""
    g = aniso()
    ops = gauss27(g)
    u, v = feasible_state(g, 7)
    gJ, gcurl, ggrad = gauss_points(g, ops, np.random.default_rng(8))
    full = jvp(u, v, ops)
    tiles = list(cell_tiles(g, 17))
    parts = [jvp(u, v, ops, tile=t) for t in tiles]
    for k in range(3):
        np.testing.assert_allclose(np.concatenate([p[k] for p in parts]), full[k], rtol=1e-15, atol=1e-15)
    for a, b in zip(jvp(restrict_free(u, g), restrict_free(v, g), ops), full):
        assert np.array_equal(a, b)
    uf, vf = node_field(restrict_free(u, g), g), node_field(v, g)
    streamed = list(iter_geometry(u, ops, tile_size=17))
    assert [t for t, _ in streamed] == tiles
    out = np.zeros((3,)+g.shape)
    for (tile, geo), part in zip(streamed, parts):
        reference = tile_geometry(uf, ops, tile=tile)
        for name in ("F", "A", "J", "curl", "grad", "div"):
            assert np.array_equal(getattr(geo, name), getattr(reference, name))
        for a, b in zip(tile_jvp(vf, ops, tile=tile, A=geo.A), part):
            assert np.array_equal(a, b)
        tile_vjp_accumulate(out, ops, tile=tile, gJ=gJ[tile], gcurl=gcurl[tile], ggrad=ggrad[tile], A=geo.A)
    whole = vjp(u, gJ, gcurl, ggrad, ops)
    np.testing.assert_allclose(finish(out), whole, rtol=1e-13, atol=1e-13)
    streamed_vjp = vjp_stream(u, lambda tile, geo: (gJ[tile], gcurl[tile], ggrad[tile]), ops, tile_size=17)
    assert np.array_equal(streamed_vjp, finish(out))
    assert np.array_equal(vjp(None, gJ, gcurl, ggrad, ops, A=gauss_geometry(u, ops).A), whole)
    assert np.array_equal(vjp(u, None, None, None, ops), np.zeros(3*g.n_free_scalar))


def test_layouts_and_errors():
    """Review F1 item 3: jvp (and every entry except gauss_geometry with allow_nonzero_faces=True) takes
    only U0 fields (zero faces) or free vectors; tile kernels take only node_field-validated layouts;
    gJ needs A; the finiteness check of a tile kernel is tile-local."""
    g = aniso()
    ops = gauss27(g)
    u, v = feasible_state(g, 7)
    gJ, _, _ = gauss_points(g, ops, np.random.default_rng(8))
    bumped = v.copy()
    bumped[0, 2, 3, 1] = 1e-3   # one nonzero face value
    for call in (lambda: jvp(u, bumped, ops), lambda: jvp(bumped, v, ops), lambda: gauss_geometry(bumped, ops),
                 lambda: gauss_gradient(bumped, ops), lambda: gauss_gradient_energy(bumped, ops),
                 lambda: iter_geometry(bumped, ops), lambda: vjp_stream(bumped, None, ops),
                 lambda: vjp(bumped, gJ, None, None, ops), lambda: node_field(bumped, g),
                 lambda: stiffness_apply(bumped, g)):
        with pytest.raises(InputValidationError, match="faces"):
            call()
    assert gauss_geometry(bumped, ops, allow_nonzero_faces=True).J.shape == (g.n_cells, 27)
    tile = slice(0, 4)
    with pytest.raises(InputValidationError, match="out"):
        tile_vjp_accumulate(np.zeros(g.shape+(3,)), ops, tile=tile, gJ=gJ[tile], A=np.ones((4, 27, 3, 3)))
    with pytest.raises(InputValidationError, match="A"):
        tile_vjp_accumulate(np.zeros((3,)+g.shape), ops, tile=tile, gJ=gJ[tile])
    for bad in (np.asfortranarray(u), u.astype(np.float32), restrict_free(u, g)):
        with pytest.raises(InputValidationError, match="node_field"):
            tile_geometry(bad, ops, tile=tile)
        with pytest.raises(InputValidationError, match="node_field"):
            tile_jvp(bad, ops, tile=tile, A=np.ones((4, 27, 3, 3)))
    with pytest.raises(InputValidationError, match="scalar_field"):
        tile_interpolate(np.asfortranarray(u[..., 0]), ops, tile=tile)
    with pytest.raises(InputValidationError):
        jvp(u, v[..., :2], ops)
    holed = u.copy()
    holed[2, 2, 2, 0] = np.nan   # node (z, y, x) = (2, 2, 2): cells with indices in {1, 2}^3 only
    assert np.isfinite(tile_geometry(holed, ops, tile=slice(0, 1)).J).all()   # cell (0, 0, 0)
    with pytest.raises(InputValidationError, match="tile values"):
        tile_geometry(holed, ops, tile=slice(37, 38))   # cell (1, 1, 1) = (1*5+1)*6+1
    with pytest.raises(InputValidationError):
        node_field(holed, g)


def test_whole_volume_limit():
    """Review F1 item 2: tile=None (one all-cell tile) raises above MAX_WHOLE_CELLS cells, before any
    allocation; explicit tiles still work."""
    g = make_grid_3d(nx=42, ny=42, nz=42, hx=1., hy=1., hz=1., unit="mm")
    assert g.n_cells > MAX_WHOLE_CELLS
    ops = gauss27(g)
    zero = np.zeros(g.shape+(3,))
    for call in (lambda: gauss_geometry(zero, ops), lambda: gauss_gradient(zero, ops), lambda: jvp(zero, zero, ops),
                 lambda: vjp(zero, None, None, None, ops)):
        with pytest.raises(InputValidationError, match="MAX_WHOLE_CELLS"):
            call()
    assert gauss_geometry(zero, ops, tile=slice(0, 8)).J.shape == (8, 27)


def test_streams_check_whole_fields_once(monkeypatch):
    """Review F1 item 1: in the streams, np.isfinite sees whole-field-sized arrays a fixed number of times
    (the one validation, plus finish's accumulator and result), independent of the 105 tiles;
    inject_free runs once per field; CellQ1Operators3D._apply (ops.derivative / ops.interpolate) is never
    called.  Negative control: the mainline per-tile ops.derivative checks the whole field on every tile."""
    g = make_grid_3d(nx=8, ny=7, nz=6, hx=.2, hy=.3, hz=.4, unit="mm")
    ops = gauss27(g)
    rng = np.random.default_rng(2)
    theta, phi = free_random(g, rng, .02), free_random(g, rng, .02)
    s = rng.normal(size=g.shape)
    field = inject_free(theta, g)
    nodes = int(np.prod(g.shape))
    whole_sizes = {nodes, 3*nodes, 3*g.n_free_scalar}   # 336, 1008, 360
    tiles = list(cell_tiles(g, 2))
    assert len(tiles) == 105   # per-tile array sizes are 8, 24, 27, 81, 243 times 1 or 2 cells: never whole
    sizes, injected = [], []
    real_isfinite, real_inject = np.isfinite, q1.inject_free

    def counting_isfinite(x, *args, **kwargs):
        sizes.append(np.size(x))
        return real_isfinite(x, *args, **kwargs)

    def counting_inject(*args, **kwargs):
        injected.append(1)
        return real_inject(*args, **kwargs)

    def whole_checks(run):
        sizes.clear()
        injected.clear()
        run()
        return sum(n in whole_sizes for n in sizes), len(injected)

    monkeypatch.setattr(np, "isfinite", counting_isfinite)
    monkeypatch.setattr(q1, "inject_free", counting_inject)
    control, _ = whole_checks(lambda: [ops.derivative(field[..., 0], "x", tile=t) for t in tiles])
    assert control >= len(tiles)

    def forbidden(*args, **kwargs):
        raise AssertionError("whole-field CellQ1Operators3D._apply called inside a stream")

    monkeypatch.setattr(CellQ1Operators3D, "_apply", forbidden)
    v = node_field(phi, g)

    def hvp():   # GN-type composition: A once per tile for both the jvp and the adjoint
        return vjp_stream(theta, lambda t, geo: tile_jvp(v, ops, tile=t, A=geo.A), ops, tile_size=2)

    def interpolate():
        sf = scalar_field(s, g)
        return [tile_interpolate(sf, ops, tile=t) for t in tiles]

    # expected (whole-sized isfinite calls, inject_free calls): theta once; hvp adds finish's two checks
    runs = dict(iter_geometry=(lambda: list(iter_geometry(theta, ops, tile_size=2)), (1, 1)),
                energy=(lambda: gauss_gradient_energy(theta, ops, tile_size=2), (1, 1)),
                hvp=(hvp, (3, 1)), interpolate=(interpolate, (1, 0)))
    for name, (run, expected) in runs.items():
        assert whole_checks(run) == expected, name


def test_single_flux_scatter(monkeypatch):
    """Review F1 item 5: with gJ, gcurl and ggrad the adjoint forms one flux and scatters it once (nine
    _transpose_add calls per tile, not eighteen); the result equals the two separate scatters to 1e-13."""
    g = aniso()
    ops = gauss27(g)
    u, _ = feasible_state(g, 5)
    gJ, gcurl, ggrad = gauss_points(g, ops, np.random.default_rng(6))
    separate = vjp(u, gJ, gcurl, None, ops)+vjp(u, None, None, ggrad, ops)
    calls = []
    real = CellQ1Operators3D._transpose_add

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(CellQ1Operators3D, "_transpose_add", staticmethod(counting))
    merged = vjp(u, gJ, gcurl, ggrad, ops)
    assert len(calls) == 9
    np.testing.assert_allclose(merged, separate, rtol=1e-13, atol=1e-13*np.max(np.abs(separate)))
    calls.clear()
    vjp_stream(u, lambda tile, geo: (gJ[tile], gcurl[tile], ggrad[tile]), ops, tile_size=17)
    assert len(calls) == 9*len(list(cell_tiles(g, 17)))


# ----------------------------------------------------------------------------------------------
# C05-1 / B7-1: K_Q
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make", [aniso, lambda: make_grid_3d(unit="mm", **B_GRID)], ids=["aniso", "b_grid"])
@pytest.mark.parametrize("seed", [0, 17])
def test_stiffness_energy_identity(make, seed, record_property):
    """B7-1 / C05 gate: u^T K_Q u equals sum_q omega w_g |grad u_h|^2 to 1e-12 (any tile size)."""
    g = make()
    ops = gauss27(g)
    theta = free_random(g, np.random.default_rng(seed))
    quad = theta@stiffness_apply(theta, g)
    energies = [gauss_gradient_energy(theta, ops, tile_size=s) for s in (8192, 13, 1)]
    errors = [abs(quad-e)/quad for e in energies]
    record_property("energy_identity_errors", json.dumps(errors))
    assert max(errors) <= 1e-12
    assert np.array_equal(stiffness_apply(theta, g), stiffness_apply(inject_free(theta, g), g))


def test_stiffness_dense_oracle_and_spectrum(record_property):
    """C05 gate: K_Q action equals the independent dense Gauss assembly sum_j D_j^T W D_j on the
    interior (each component) to 1e-12, and its DST-I modes have the T10 eigenvalues to 1e-12."""
    g = aniso()
    D, w = oracle_derivatives(g, 3)
    S = interior_rows(g)
    K1 = sum(D[j][:, S].T@(w[:, None]*D[j][:, S]) for j in range(3))
    K = np.kron(np.eye(3), K1)
    n = 3*g.n_free_scalar
    dense = np.stack([stiffness_apply(e, g) for e in np.eye(n)], axis=1)
    errors = dict(oracle=float(np.max(np.abs(dense-K))/np.max(np.abs(K))))
    kz, ky, kx = g.interior_shape
    iz, iy, ix = np.indices(g.interior_shape)+1
    spectral = []
    for mz, my, mx in [(1, 1, 1), (2, 3, 1), (kz, ky, kx), (1, ky, 2)]:
        mode = np.sin(np.pi*mz*iz/(kz+1))*np.sin(np.pi*my*iy/(ky+1))*np.sin(np.pi*mx*ix/(kx+1))
        ell, mass = [], []
        for m, nn, h in ((mx, kx, g.spacing[0]), (my, ky, g.spacing[1]), (mz, kz, g.spacing[2])):
            th = np.pi*m/(nn+1)
            ell.append((2*np.sin(th/2)/h)**2)
            mass.append((2+np.cos(th))/3)
        lam = g.omega*(ell[0]*mass[1]*mass[2]+mass[0]*ell[1]*mass[2]+mass[0]*mass[1]*ell[2])
        theta = np.concatenate((np.zeros(mode.size), mode.ravel(), np.zeros(mode.size)))
        spectral.append(float(np.max(np.abs(stiffness_apply(theta, g)-lam*theta))/(lam*np.max(np.abs(mode)))))
    errors["spectrum"] = max(spectral)
    record_property("stiffness_errors", json.dumps(errors))
    assert errors["oracle"] <= 1e-12 and errors["spectrum"] <= 1e-12


@pytest.mark.parametrize("make", [aniso, lambda: make_grid_3d(nx=5, ny=5, nz=5, hx=.25, hy=.25, hz=.25, unit="mm")],
                         ids=["aniso", "unit5"])
def test_stiffness_kron_full_spectrum(make, record_property):
    """C05-1 gate (review F1 item 8): the K_Q action equals, per component, the direct Kronecker assembly
    M_z(x)M_y(x)K_x + M_z(x)K_y(x)M_x + K_z(x)M_y(x)M_x (M_s = h_s tridiag(1,4,1)/6, K_s =
    tridiag(-1,2,-1)/h_s on the interior) to 1e-12, and the full spectra (eigvalsh) of both equal all
    T10 eigenvalues (each three times) to 1e-12 relative to the largest; all positive."""
    g = make()
    kz, ky, kx = g.interior_shape
    hx, hy, hz = g.spacing
    Mx, My, Mz = (h*tridiag(n, 4., 1.)/6 for n, h in ((kx, hx), (ky, hy), (kz, hz)))
    Kx, Ky, Kz = (tridiag(n, 2., -1.)/h for n, h in ((kx, hx), (ky, hy), (kz, hz)))
    K1 = np.kron(Mz, np.kron(My, Kx))+np.kron(Mz, np.kron(Ky, Mx))+np.kron(Kz, np.kron(My, Mx))
    KQ = np.kron(np.eye(3), K1)
    n = 3*g.n_free_scalar
    dense = np.stack([stiffness_apply(e, g) for e in np.eye(n)], axis=1)
    analytic = np.sort(np.tile(t10_eigenvalues(g), 3))
    eig_dense, eig_kron = np.linalg.eigvalsh(dense), np.linalg.eigvalsh(KQ)
    scale, top = np.max(np.abs(KQ)), analytic[-1]
    errors = dict(kron=float(np.max(np.abs(dense-KQ))/scale), symmetry=float(np.max(np.abs(dense-dense.T))/scale),
                  spectrum_dense=float(np.max(np.abs(eig_dense-analytic))/top),
                  spectrum_kron=float(np.max(np.abs(eig_kron-analytic))/top))
    record_property("kron_spectrum_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors
    assert analytic[0] > 0 and eig_dense[0] > 0


def test_checkerboard_energies_positive(record_property):
    """C05 gate (B8 unit part): alternating +-eps interior displacement has positive Gauss-point
    gradient and curl energies, positive Gauss J, and J not identically 1 (max |log J_q| > 0)."""
    g = aniso()
    ops = gauss27(g)
    sign = (-1.)**np.indices(g.shape).sum(axis=0)
    u = np.zeros(g.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = (.01*min(g.spacing)*sign[1:-1, 1:-1, 1:-1])[..., None]*[1., -.5, .25]
    geo = gauss_geometry(u, ops)
    log_j = float(np.max(np.abs(np.log(geo.J))))
    record_property("checkerboard_max_abs_log_J", log_j)
    assert gauss_gradient_energy(u, ops) > 0
    assert np.sum(ops.weights[None, :, None]*geo.curl**2) > 0
    assert geo.J.min() > 0
    assert log_j > 0


# ----------------------------------------------------------------------------------------------
# B13-1/2 and B11: explicit matrices from column-wise jvp
# ----------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def assembled(spec):
    """K_Q (3N_int square, columns K_Q e_i), D_div (|Q| x 3N_int) and C (3|Q| x 3N_int, rows (q, k))
    from the jvp at u = 0 of every unit vector e_i, W = omega w_g per Gauss point; also max|dJ - tr dgrad|
    over all columns (B13-2).  The unit vectors are probed 24 at a time: interior nodes with equal
    index parity per axis have disjoint cell supports, so every Gauss-point output of a probe belongs to
    exactly one node (each cell holds one node of each parity) and the columns are bitwise those of one
    jvp per unit vector (spot-checked below)."""
    g = make_grid_3d(unit="mm", **dict(spec))
    ops = gauss27(g)
    m, n = g.n_free_scalar, 3*g.n_free_scalar
    zero = np.zeros(g.shape+(3,))
    div = np.zeros((g.n_cells*27, n))
    curl = np.zeros((g.n_cells*27*3, n))
    free = np.full(g.shape, -1)
    free[1:-1, 1:-1, 1:-1] = np.arange(m).reshape(g.interior_shape)
    parity = np.indices(g.shape) % 2
    cz, cy, cx = np.indices(g.cell_shape).reshape(3, -1)
    rows = np.arange(g.n_cells)[:, None]*27+np.arange(27)[None, :]
    b13_2 = 0.
    for d in range(3):
        for pz, py, px in product((0, 1), repeat=3):
            probe = zero.copy()
            probe[(parity[0] == pz) & (parity[1] == py) & (parity[2] == px) & (free >= 0), d] = 1.
            dJ, dcurl, dgrad = jvp(zero, probe, ops)
            tr = np.trace(dgrad, axis1=-2, axis2=-1)
            b13_2 = max(b13_2, float(np.max(np.abs(dJ-tr))))
            node = free[cz+(pz-cz) % 2, cy+(py-cy) % 2, cx+(px-cx) % 2]
            hit = node >= 0
            cols = (d*m+node[hit])[:, None]
            div[rows[hit], cols] = tr[hit]
            for k in range(3):
                curl[3*rows[hit]+k, cols] = dcurl[hit][..., k]
    for i in (0, m+1, n-1):
        _, dcurl, dgrad = jvp(zero, np.eye(n)[i], ops)
        assert np.array_equal(div[:, i], np.trace(dgrad, axis1=-2, axis2=-1).ravel())
        assert np.array_equal(curl[:, i], dcurl.ravel())
    K = np.stack([stiffness_apply(e, g) for e in np.eye(n)], axis=1)
    w = np.tile(ops.weights, g.n_cells)
    return g, K, div, curl, w, b13_2


@pytest.mark.parametrize("spec", [tuple(B_GRID.items()), (("nx", 5), ("ny", 6), ("nz", 7), ("hx", .2), ("hy", .3),
                                                           ("hz", .4))], ids=["6x7x8", "5x6x7"])
def test_b13_helmholtz_identity(spec, record_property):
    """B13-1: K_Q - D_div^T W D_div - C^T W C has max entry <= 1e-12 max|K_Q| (Q1, zero faces,
    Gauss27 exact); B13-2: at u = 0, dJ = div (A = I)."""
    g, K, div, curl, w, b13_2 = assembled(spec)
    residual = K-div.T@(w[:, None]*div)-curl.T@(np.repeat(w, 3)[:, None]*curl)
    error = float(np.max(np.abs(residual))/np.max(np.abs(K)))
    record_property("b13_1_relative_max_residual", error)
    record_property("b13_2_max_abs", b13_2)
    assert error <= 1e-12
    assert b13_2 <= 1e-15*max(1., np.max(np.abs(div)))
    assert np.allclose(K, K.T, rtol=0, atol=1e-13*np.max(np.abs(K)))


B11_SPECS = {"5x5x5": (("nx", 5), ("ny", 5), ("nz", 5), ("hx", .25), ("hy", .25), ("hz", .25)),
             "6x6x6": (("nx", 6), ("ny", 6), ("nz", 6), ("hx", .2), ("hy", .2), ("hz", .2)),
             "7x7x7": (("nx", 7), ("ny", 7), ("nz", 7), ("hx", 1/6), ("hy", 1/6), ("hz", 1/6)),
             "6x7x8": tuple(B_GRID.items())}


def test_b11_curl_injective():
    """B11 (待复核), check NB11: the Gauss27 curl operator C on U0 has full column rank 3N_int on every
    grid, with sigma_min(C), sigma_min(W^1/2 C) and lambda_min(C^T W C, K_Q) > 0 (no threshold).  The
    values and, on the unit cube, the decay ratios of lambda_min against (h_prev/h)^2 are recorded in
    docs/bayesreg/checks/NB11_curl_rank.json (review F1 item 10; written only with BAYESREG_WRITE_CHECKS=1):
    injective, not uniformly well conditioned (B11-3)."""
    records = []
    for name, spec in B11_SPECS.items():
        g, K, div, curl, w, _ = assembled(spec)
        n = 3*g.n_free_scalar
        rank = int(np.linalg.matrix_rank(curl))
        plain = np.linalg.eigvalsh(curl.T@curl)
        weighted = np.linalg.eigvalsh(curl.T@(np.repeat(w, 3)[:, None]*curl))
        lam = float(eigh(curl.T@(np.repeat(w, 3)[:, None]*curl), K, eigvals_only=True, subset_by_index=[0, 0])[0])
        assert rank == n, name
        assert plain[0] > 0 and weighted[0] > 0 and lam > 0, name
        records.append(dict(grid=name, h_xyz=list(g.spacing), n_columns=n, rank=rank,
                            sigma_min=float(np.sqrt(plain[0])), sigma_max=float(np.sqrt(plain[-1])),
                            sigma_min_weighted=float(np.sqrt(weighted[0])), lambda_min_curl_over_KQ=lam))
    unit = records[:3]
    decay = [dict(grids=f"{a['grid']} -> {b['grid']}", lambda_min_ratio=a["lambda_min_curl_over_KQ"]/b[
        "lambda_min_curl_over_KQ"], h_ratio_squared=(a["h_xyz"][0]/b["h_xyz"][0])**2) for a, b in zip(unit, unit[1:])]
    payload = dict(
        check="NB11", theory="docs/bayesreg/02_理论.md B11 (待复核)",
        generated_by="tests/test_bayesreg/geom/test_geom_q1.py::test_b11_curl_injective",
        operator="C: Gauss27 Q1 curl on U0 (rows (q, k), 3|Q| x 3N_int) assembled from bayesreg.geom.jvp at u = 0; "
                 "W = omega w_g; K_Q from bayesreg.geom.stiffness_apply",
        numpy=np.__version__, grids=records, unit_cube_lambda_min_decay=decay,
        reading="rank = 3N_int and all minima > 0 are asserted; lambda_min(C^T W C, K_Q) decays like h^2 on the "
                "unit cube (ratio vs (h_prev/h)^2), sigma_min(C) in the Euclidean nodal norm only slowly")
    text = json.dumps(payload, indent=1, ensure_ascii=False)+"\n"
    if os.environ.get(WRITE_CHECKS_ENV) == "1" and (not NB11_JSON.exists()
                                                     or NB11_JSON.read_text(encoding="utf-8") != text):
        NB11_JSON.write_text(text, encoding="utf-8")
