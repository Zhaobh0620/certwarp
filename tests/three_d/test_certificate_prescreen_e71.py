"""E71 T75: rigorous float Bernstein prescreen. References use exact rational arithmetic (Fraction):
the float inputs (u, spacing) are exact rationals, so beta_exact is computed without any rounding."""
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from jd_curl.three_d.cell_q1 import CellQ1Operators3D, VERTICES
from jd_curl.three_d.certificate_prescreen import prescreen_lower_bounds
from jd_curl.three_d.feasibility import (CompiledFallbackCertificateConfig, ParallelEnclosureCertificateConfig,
                                         PrescreenCertificateConfig, PrescreenFeasibilityResult, certify_map)
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.validation import InputValidationError

HALF = (Fraction(0), Fraction(1, 2), Fraction(1))


def exact_beta(u, grid, cell):
    """Exact Bernstein coefficients of J on one cell (rational arithmetic only)."""
    cz, cy, cx = np.unravel_index(cell, grid.cell_shape)
    inv = [1/Fraction(float(h)) for h in grid.spacing]
    verts = []
    for x, y, z in VERTICES:
        verts.append([Fraction(float(v)) for v in u[cz+z, cy+y, cx+x]])
    J = {}
    for iz, iy, ix in product(range(3), repeat=3):
        xi = (HALF[ix], HALF[iy], HALF[iz])
        F = [[Fraction(int(r == j)) for j in range(3)] for r in range(3)]
        for a, vertex in enumerate(VERTICES):
            for j in range(3):
                g = inv[j]*(1 if vertex[j] else -1)
                for k in range(3):
                    if k != j:
                        g *= xi[k] if vertex[k] else 1-xi[k]
                for r in range(3):
                    F[r][j] += verts[a][r]*g
        J[(iz, iy, ix)] = (F[0][0]*(F[1][1]*F[2][2]-F[1][2]*F[2][1]) - F[0][1]*(F[1][0]*F[2][2]-F[1][2]*F[2][0])
                           + F[0][2]*(F[1][0]*F[2][1]-F[1][1]*F[2][0]))
    T = [[1, 0, 0], [Fraction(-1, 2), 2, Fraction(-1, 2)], [0, 0, 1]]
    beta = []
    for kz, ky, kx in product(range(3), repeat=3):
        beta.append(sum(T[kz][a]*T[ky][b]*T[kx][c]*J[(a, b, c)] for a in range(3) for b in range(3) for c in range(3)))
    return beta


def lower_bounds(u, grid, cells):
    return prescreen_lower_bounds(u, grid, CellQ1Operators3D(grid, location='closed27_uniform').coefficients, cells)


def random_field(grid, amplitude, seed, scale=None):
    r = np.random.default_rng(seed)
    u = np.zeros(grid.shape+(3,))
    sp = np.array(grid.spacing) if scale is None else scale
    u[1:-1, 1:-1, 1:-1] = amplitude*sp*r.uniform(-1, 1, u[1:-1, 1:-1, 1:-1].shape)
    return u


GRIDS = [dict(nx=6, ny=5, nz=7, hx=.7, hy=1.1, hz=.9), dict(nx=5, ny=6, nz=5, hx=1.9875, hy=1.9910714285714286, hz=1.9895833333333333),
         dict(nx=5, ny=5, nz=6, hx=1., hy=1., hz=1.), dict(nx=6, ny=5, nz=5, hx=.3, hy=7.7, hz=.013)]


@pytest.mark.parametrize('gi', range(len(GRIDS)))
@pytest.mark.parametrize('amplitude', [.05, .3, .45, 3.])
def test_lower_bound_is_rigorous_against_exact_rationals(gi, amplitude):
    g = make_grid_3d(unit='mm', **GRIDS[gi])
    u = random_field(g, amplitude, gi*10+int(amplitude*100))
    cells = np.arange(g.n_cells)
    lb = lower_bounds(u, g, cells)
    for c in cells[::max(1, len(cells)//25)]:
        beta = exact_beta(u, g, int(c))
        mn = min(beta)
        assert Fraction(float(lb[c])) <= mn, (c, float(lb[c]), float(mn))     # conservative, exactly
        # not vacuous: on well-conditioned grids the bound only removes a rounding-size slack. The
        # extreme grid (h = .3/7.7/.013) makes |F| ~ 30, where the a-priori bound (~ phi^3) is loose
        # by design; such cells simply go to the interval fallback.
        if gi != 3:
            assert float(mn)-float(lb[c]) <= 1e-9*max(1., max(abs(float(b)) for b in beta))


def test_rigorous_at_the_degeneracy_edge():
    """Scale a real field so that the exact min beta sits within ~1e-13 of 0 (both signs)."""
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=.7, hy=1.1, hz=.9, unit='mm')
    base = random_field(g, .6, 3)
    cells = np.arange(g.n_cells)
    lo, hi = 0., 4.                       # bisection on the exact min beta of the worst cell
    def worst(s):
        return min(min(exact_beta(s*base, g, int(c))) for c in cells)
    assert worst(lo) > 0 > worst(hi)
    for _ in range(45):
        mid = .5*(lo+hi)
        lo, hi = (mid, hi) if worst(mid) > 0 else (lo, mid)
    for s in (lo, hi, lo*(1-1e-12), hi*(1+1e-12)):
        u = s*base
        lb = lower_bounds(u, g, cells)
        for c in cells:
            assert Fraction(float(lb[c])) <= min(exact_beta(u, g, int(c)))


def test_nonfinite_is_never_proven_and_validation():
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=1., hy=1., hz=1., unit='mm')
    u = random_field(g, .2, 1); u[2, 2, 2, 0] = np.inf
    lb = lower_bounds(u, g, np.arange(g.n_cells))
    assert np.isnan(lb).any() and not np.any(lb[np.isnan(lb)] > 0)
    with pytest.raises(InputValidationError):
        lower_bounds(u[:-1], g, np.arange(3))
    with pytest.raises(InputValidationError):
        prescreen_lower_bounds(u, g, np.zeros((3, 8, 27)), np.arange(3))


def configs(delta=0.):
    from dataclasses import asdict
    from jd_curl.three_d.feasibility import EnclosureCertificateConfig
    base = ParallelEnclosureCertificateConfig(**asdict(EnclosureCertificateConfig(delta=delta, tile_size=7)), fallback_workers=2)
    return CompiledFallbackCertificateConfig(**asdict(base)), PrescreenCertificateConfig(**asdict(base))


@pytest.mark.parametrize('amplitude, seed', [(.2, 0), (.45, 1), (.6, 2), (1.5, 3)])
def test_certificate_decisions_match_the_interval_path(amplitude, seed):
    g = make_grid_3d(nx=7, ny=6, nz=8, hx=.7, hy=1.1, hz=.9, unit='mm')
    u = random_field(g, amplitude, seed)
    old_cfg, new_cfg = configs()
    old, new = certify_map(u, g, config=old_cfg), certify_map(u, g, config=new_cfg)
    assert isinstance(new, PrescreenFeasibilityResult)
    assert (new.status, new.reason == old.reason or new.status == 'certified_feasible') == (old.status, True)
    assert (new.witness_cell, new.witness_reference_point, new.n_infeasible) == (old.witness_cell, old.witness_reference_point, old.n_infeasible)
    assert new.n_cells_checked == old.n_cells_checked and new.n_fast_cells == old.n_fast_cells
    assert new.n_prescreen_cells+new.n_fallback_cells == old.n_fallback_cells
    if new.status == 'certified_feasible':
        assert new.min_jd_lower_bound > 0
        # the reported bound is a true lower bound: never above the exact minimum of any checked cell
        exact_min = min(min(exact_beta(u, g, int(c))) for c in range(g.n_cells))   # every cell (review E71)
        assert Fraction(new.min_jd_lower_bound) <= exact_min


def test_prescreen_proves_cells_and_old_versions_are_unchanged():
    g = make_grid_3d(nx=7, ny=6, nz=8, hx=.7, hy=1.1, hz=.9, unit='mm')
    u = random_field(g, .45, 1)
    old_cfg, new_cfg = configs()
    new = certify_map(u, g, config=new_cfg)
    old = certify_map(u, g, config=old_cfg)
    assert new.n_prescreen_cells > 0 and new.certificate_version.endswith('prescreen-compiled-bernstein-v1')
    assert not hasattr(old, 'n_prescreen_cells') and old.certificate_version == 'q1-3d-parallel-hoisted-edge-hull-compiled-bernstein-v1'


# ---------------------------------------------------------------- review E71 additions

def affine_field(grid, A, shift=0.):
    """u = (A - I) x + shift: J is det(A) everywhere; |F| ~ |A|; a large shift makes sigma >> phi."""
    z, y, x = np.meshgrid(*(np.arange(n)*h for n, h in zip(grid.shape, grid.spacing[::-1])), indexing='ij')
    X = np.stack([x, y, z], axis=-1)
    return np.einsum('rj,...j->...r', A-np.eye(3), X)+shift


@pytest.mark.parametrize('det, big', [(1e-6, 1e3), (-1e-9, 1e2), (3e-4, 1e4), (2., 30.)])
def test_rigorous_for_large_F_near_singular(det, big):
    g = make_grid_3d(nx=5, ny=6, nz=5, hx=.7, hy=1.1, hz=.9, unit='mm')
    r = np.random.default_rng(int(big))
    Q1, _ = np.linalg.qr(r.standard_normal((3, 3))); Q2, _ = np.linalg.qr(r.standard_normal((3, 3)))
    A = Q1@np.diag([big, 1/big, 1.])@Q2
    A *= np.cbrt(det/np.linalg.det(A)) if det > 0 else -np.cbrt(-det/np.linalg.det(A))
    u = affine_field(g, A)
    cells = np.arange(g.n_cells)
    lb = lower_bounds(u, g, cells)
    for c in cells[::5]:
        assert Fraction(float(lb[c])) <= min(exact_beta(u, g, int(c)))


@pytest.mark.parametrize('shift', [1e3, 1e8, 1e12])
def test_rigorous_when_sigma_dominates_phi(shift):
    """A huge common translation cancels exactly in F but inflates sigma (the summands)."""
    g = make_grid_3d(nx=5, ny=5, nz=6, hx=.7, hy=1.1, hz=.9, unit='mm')
    u = random_field(g, .4, 5)+shift*np.array([.7, -1.1, .9])
    cells = np.arange(g.n_cells)
    lb = lower_bounds(u, g, cells)
    for c in cells[::4]:
        assert Fraction(float(lb[c])) <= min(exact_beta(u, g, int(c)))
    if shift == 1e3:
        assert np.mean(lb > 0) > .5                  # not vacuous: the bound still proves cells


@pytest.mark.parametrize('delta', [.01, .3])
@pytest.mark.parametrize('amplitude, seed', [(.2, 0), (.6, 2), (1.5, 3)])
def test_decisions_match_with_positive_delta(delta, amplitude, seed):
    g = make_grid_3d(nx=7, ny=6, nz=8, hx=.7, hy=1.1, hz=.9, unit='mm')
    u = random_field(g, amplitude, seed)
    old_cfg, new_cfg = configs(delta)
    old, new = certify_map(u, g, config=old_cfg), certify_map(u, g, config=new_cfg)
    assert (new.status, new.witness_cell, new.n_infeasible) == (old.status, old.witness_cell, old.n_infeasible)
    if new.status == 'certified_feasible':
        assert new.min_jd_lower_bound > delta
        assert Fraction(new.min_jd_lower_bound) <= min(min(exact_beta(u, g, int(c))) for c in range(g.n_cells))


def test_subnormal_table_gives_no_proof_and_extreme_spacing_does_not_raise():
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=4.4e307, hy=1e-150, hz=1e-150, unit='mm')     # f/hx subnormal
    D = CellQ1Operators3D(g, location='closed27_uniform').coefficients
    assert np.abs(D[D != 0]).min() < np.finfo(float).tiny
    lb = prescreen_lower_bounds(np.zeros(g.shape+(3,)), g, D, np.arange(g.n_cells))
    assert np.isnan(lb).all()                                   # J = 1 exactly, but outside the bound's assumptions
    g = make_grid_3d(nx=5, ny=5, nz=5, hx=1e-310, hy=1e150, hz=1e150, unit='mm')        # 1/h overflows
    u = np.zeros(g.shape+(3,))                                  # zero boundary: reach the prescreen block
    u[1:-1, 1:-1, 1:-1, 0] = .95e-310*np.random.default_rng(0).uniform(-1, 1, u[1:-1, 1:-1, 1:-1, 0].shape)
    old_cfg, new_cfg = configs()
    old = certify_map(u, g, config=old_cfg)
    assert old.reason != 'NONZERO_BOUNDARY' and old.n_fallback_cells > 0
    new = certify_map(u, g, config=new_cfg)                     # previously raised FloatingPointError
    assert (new.status, new.reason, new.witness_cell, new.n_prescreen_cells) == (old.status, old.reason, old.witness_cell, 0)
