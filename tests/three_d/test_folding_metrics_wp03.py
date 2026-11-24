"""WP-0.3: tetrahedral proxies and Learn2Reg-style indicators (diagnostics next to the certificate)."""
import numpy as np

from jd_curl.three_d.bernstein_barrier import to_bernstein
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.geometry import _compute_cell_geometry
from jd_curl.three_d.grid import make_grid_3d
from scripts import folding_metrics as fm

SPACING = (.9, 1.1, 1.3)


def grid():
    return make_grid_3d(nx=7, ny=6, nz=5, hx=SPACING[0], hy=SPACING[1], hz=SPACING[2], x0=0., y0=0., z0=0., unit='mm')


def test_identity_and_affine_maps():
    g = grid()
    u = np.zeros(g.shape+(3,))
    for s, r in fm.tet_ratio_summary(u, SPACING).items():
        assert np.isclose(r['min_ratio'], 1., rtol=0, atol=1e-14) and r['cells_nonpositive'] == 0
    A = np.array([[1.2, .1, 0.], [-.05, .9, .2], [.1, 0., 1.1]])            # y = A x (columns xyz)
    X = g.identity()
    ua = X@A.T-X
    for name, r in fm.tet_ratio_summary(ua, SPACING).items():
        assert np.isclose(r['min_ratio'], np.linalg.det(A), rtol=1e-12), name
    ind = fm.l2r_indicators(np.zeros((12, 11, 10, 3)), (1., 1., 1.))
    assert ind['l2r_sdlogj'] == 0. and ind['central_det_nonpositive_fraction'] == 0.


def test_mean_of_24_ratios_is_the_cell_average_J_and_the_bernstein_mean():
    g = grid()
    rng = np.random.default_rng(0)
    u = np.zeros(g.shape+(3,)); u[1:-1, 1:-1, 1:-1] = .15*rng.standard_normal(u[1:-1, 1:-1, 1:-1].shape)
    mean24 = fm.cell_mean_ratio24(u, SPACING).ravel()
    ops = CellQ1Operators3D(g, location='closed27_uniform')
    J = np.concatenate([_compute_cell_geometry(u, ops, tile=t).J for t in cell_tiles(g, 64)])
    bern = to_bernstein(J).mean(axis=1)
    assert np.allclose(mean24, bern, rtol=1e-12, atol=1e-13)


def test_corner_squeeze_counterexample_is_a_proxy_gap():
    """Plan's counterexample (a): corner 1 moved inward by (t, t, t), t = .5: the corner J is -.5 but the minimum
    of the 24 ratios is .125, so the Heldmann-24 proxy passes a cell whose J is negative (unit cube)."""
    u = np.zeros((2, 2, 2, 3)); u[0, 0, 0] = (.5, .5, .5)
    r = fm.tet_ratio_summary(u, (1., 1., 1.))
    assert np.isclose(r['tet24']['min_ratio'], .125)
    assert np.isclose(fm.cell_mean_ratio24(u, (1., 1., 1.))[0, 0, 0], .625)
