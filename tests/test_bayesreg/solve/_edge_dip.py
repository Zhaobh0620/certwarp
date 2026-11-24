"""Cells whose Q1 J minimum lies inside an edge (work packages JDC / JDC1, C09.1-3a jd_cell_estimate; the mode of
runs/20261030-diag-g3).  No data/ access.

Moved here from _synthetic.py in JDC1 so that _synthetic.py stays byte-identical to commit 481e33b: the golden
generator tests/test_bayesreg/_golden/make_golden.py --ref 481e33b refuses to run when its helpers differ.

Both constructions: vertex displacements (mm, components xyz, in the order of jd_curl cell_q1.VERTICES) of one cell on a
unit grid (h = 1), every other node zero, found by a penalty search and rounded to 1e-3.

MID  (JDC, 38e63ab): the 8 corners J >= .15 and the 27 Gauss points J >= .18 (m_K = .1508), but J = .00202 at the
     midpoint of the z edge at local (x, y) = (0, 0) -- a closed27 point, so closed27 sees it -- and the true minimum
     (about .0019) on that edge; the root Bernstein minimum of the cell is negative, so the certificate proves it by
     subdivision; the 26 neighbouring cells have m_K >= .0996.
OFF  (JDC1, review JDC_复核_20261030 important item 3: a counterexample off the edge midpoint): the minimum of J on the
     same edge at zeta = .25, J = .00096 there (the minimum over the cell: a 17^3 sampling finds nothing lower);
     on that edge J(0) = .1185, J(1/2) = .1195, J(1) = 1.065 (Q1 J restricted to an edge is a quadratic in the edge
     parameter, so a dip near zeta = 1/4 leaves J(1/2) about equal to J(0)); J >= .1185 at all 53 points of Gauss27 and
     closed27, so closed27 gives the same m_K as the default (.1185); the 26 neighbouring cells have m_K >= .0998; the
     certificate proves the cell in its interval fallback with b_K = 7.1e-4.  Search: scipy least_squares on the 24
     vertex values (penalties: J = .002 and a minimum at (0, 0, .25) on the edge, J >= .12 at the 53 points,
     J >= .0015 on a 5^3 sampling of the cell and on the edge, J >= .1 on a 5^3 sampling of every neighbouring cell),
     started from MID, six starts (MID and five random perturbations); the deepest admissible result, rounded.
"""
import numpy as np

EDGE_DIP_VERTICES = ((-.161, 1.171, -.053), (-.007, -.9, .001), (-.162, .362, -.122), (.167, .141, -.134),
                     (.036, .113, .232), (-.722, .351, .042), (-.488, .078, -.402), (-.754, -.079, -.202))
EDGE_DIP_POINT = (0., 0., .5)          # reference (xi, eta, zeta) of the dip on the cell's z edge at (x, y) = (0, 0)

EDGE_DIP_OFF_VERTICES = ((.217, 1.218, -.405), (.008, -.899, .008), (.22, .326, -.466), (.745, .116, .04),
                         (.474, .151, .225), (-.01, .21, .202), (-.213, .196, .456), (-.898, .17, .483))
EDGE_DIP_OFF_POINT = (0., 0., .25)     # off the midpoint: between the closed27 points (0, 0, 0) and (0, 0, 1/2)

KINDS = dict(mid=(EDGE_DIP_VERTICES, EDGE_DIP_POINT), off=(EDGE_DIP_OFF_VERTICES, EDGE_DIP_OFF_POINT))


def edge_dip_field(grid, cell, kind="mid"):
    """u (nz, ny, nx, 3) mm: zero except the 8 vertices of `cell` = (cz, cy, cx), which get the vertex values of `kind`
    ("mid" or "off"; unit spacing assumed by the construction; the cell must not touch the faces)."""
    from jd_curl.three_d.cell_q1 import VERTICES
    vertices, _ = KINDS[kind]
    cz, cy, cx = cell
    if min(cell) < 1 or cz+2 >= grid.shape[0] or cy+2 >= grid.shape[1] or cx+2 >= grid.shape[2]:
        raise ValueError("edge_dip_field: the cell must be at least one cell away from the faces")
    u = np.zeros(grid.shape+(3,))
    for v, (x, y, z) in enumerate(VERTICES):
        u[cz+z, cy+y, cx+x] = vertices[v]
    return u


def dip_point(kind="mid"):
    return KINDS[kind][1]


def cell_point_jacobian(u, grid, cell, point):
    """J of the Q1 field u in `cell` (cz, cy, cx) at the reference point (xi, eta, zeta), and dJ/du at its 8 vertices
    ((8, 3), cell_q1.VERTICES order): cof(F) applied to the shape-function gradients (exact Q1 formulas)."""
    from jd_curl.three_d.cell_q1 import VERTICES
    cz, cy, cx = cell
    h = np.asarray(grid.spacing, dtype=np.float64)
    dN = np.empty((8, 3))
    for v, bits in enumerate(VERTICES):
        f = [point[j] if bits[j] else 1-point[j] for j in range(3)]
        d = [1. if bits[j] else -1. for j in range(3)]
        dN[v] = (d[0]*f[1]*f[2], f[0]*d[1]*f[2], f[0]*f[1]*d[2])
    dN = dN/h[None, :]
    verts = np.array([u[cz+z, cy+y, cx+x] for x, y, z in VERTICES])
    F = np.eye(3)+verts.T@dN                       # F[r, j] = delta_rj + sum_v u_v[r] dN_v/dx_j
    J = float(np.linalg.det(F))
    cof = J*np.linalg.inv(F).T
    return J, dN@cof.T
