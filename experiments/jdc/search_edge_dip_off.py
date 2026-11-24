"""JDC1: the penalty search that produced EDGE_DIP_OFF_VERTICES of tests/test_bayesreg/solve/_edge_dip.py (a cell whose
Q1 J minimum lies inside its z edge at zeta = .25, between closed27 points; review JDC_复核_20261030 important item 3).

Synthetic geometry only (one unit cell and its 26 neighbours, every other node zero); no data access, no GPU.
Variables: the 24 displacement values of the cell's 8 vertices (jd_curl cell_q1.VERTICES order, components xyz).
Residuals (scipy.optimize.least_squares, trf, 2-point Jacobian, max_nfev 150):
  200 (J(0, 0, z*) - .002), 200 max(0, J(z* -+ .001) below J(z*)) (a minimum at z* on the edge),
  20 max(0, .12 - J) at the 53 points (Gauss27 and closed27) of the cell,
  20 max(0, .0015 - J) on a 5^3 sampling of the cell and on 201 points of the edge,
  20 max(0, .1 - J) on a 5^3 sampling of each of the 26 neighbouring cells,
  .01 |x| (keeps the displacements small).
Starts: the "mid" construction (EDGE_DIP_VERTICES) and five perturbations (default_rng(0), sigma .3).  Each result is
rounded to 1e-3 and kept if its edge minimum is positive, within .05 of z*, its 41^3 cell sampling positive, its 53-point
minimum > .1 and its neighbours' 5^3 minimum > .05; the deepest such edge minimum is printed (with z* = .25 this gave
EDGE_DIP_OFF_VERTICES; the test file checks the field itself, so the constants do not depend on rerunning this).

    python -B experiments/jdc/search_edge_dip_off.py --zstar .25
"""
import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jd_curl.three_d.cell_q1 import VERTICES  # noqa: E402

from tests.test_bayesreg.solve import _edge_dip as dip  # noqa: E402

VERT = np.array(VERTICES, dtype=float)                        # (8, 3) bits (x, y, z)
GAUSS = (.5-np.sqrt(15.)/10, .5, .5+np.sqrt(15.)/10)
P53 = np.array(sorted(set(product(GAUSS, repeat=3)) | set(product((0., .5, 1.), repeat=3))))
DENSE = np.array(list(product(np.linspace(0, 1, 5), repeat=3)))
EDGE = np.array([(0., 0., t) for t in np.linspace(0, 1, 201)])
TARGET, FLOOR53 = .002, .12


def shape_gradients(points):
    """(n, 8, 3) gradients of the Q1 shape functions at reference points (unit spacing)."""
    p = np.asarray(points, dtype=float)
    out = np.empty((len(p), 8, 3))
    for v, bits in enumerate(VERT):
        f = np.where(bits[None, :] == 1, p, 1-p)
        d = np.where(bits == 1, 1., -1.)
        out[:, v, 0] = d[0]*f[:, 1]*f[:, 2]
        out[:, v, 1] = f[:, 0]*d[1]*f[:, 2]
        out[:, v, 2] = f[:, 0]*f[:, 1]*d[2]
    return out


def jacobians(verts, gradients):
    return np.linalg.det(np.eye(3)[None]+np.einsum("vr,nvj->nrj", verts, gradients))


def neighbourhood(x):
    """{offset: (8, 3) vertex values} of the 27 cells around the cell whose vertex values are x (others zero)."""
    field = {tuple(bits): x[v] for v, bits in enumerate(VERT.astype(int))}
    out = {}
    for offset in product((-1, 0, 1), repeat=3):
        verts = np.zeros((8, 3))
        for v, bits in enumerate(VERT.astype(int)):
            key = tuple(np.array(offset)+bits)
            if key in field:
                verts[v] = field[key]
        out[offset] = verts
    return out


def make_residuals(zstar):
    D53, DD, DE = shape_gradients(P53), shape_gradients(DENSE), shape_gradients(EDGE)
    DS = shape_gradients(np.array([(0., 0., zstar), (0., 0., zstar-1e-3), (0., 0., zstar+1e-3)]))

    def residuals(flat):
        cells = neighbourhood(flat.reshape(8, 3))
        K = cells[(0, 0, 0)]
        js = jacobians(K, DS)
        r = [200*(js[0]-TARGET), 200*(js[1]-js[0])*(js[1] < js[0]), 200*(js[2]-js[0])*(js[2] < js[0])]
        r.extend(20*np.maximum(0., FLOOR53-jacobians(K, D53)))
        r.extend(20*np.maximum(0., .0015-jacobians(K, DD)))
        r.extend(20*np.maximum(0., .0015-jacobians(K, DE)))
        for offset, verts in cells.items():
            if offset != (0, 0, 0):
                r.extend(20*np.maximum(0., .1-jacobians(verts, DD)))
        r.append(.01*np.sum(flat**2)**.5)
        return np.array(r)
    return residuals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--zstar", type=float, default=.25)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--max-nfev", type=int, default=150)
    args = parser.parse_args(argv)
    residuals = make_residuals(args.zstar)
    x0 = np.array(dip.EDGE_DIP_VERTICES, dtype=float).ravel()
    rng = np.random.default_rng(0)
    fine = shape_gradients(np.array(list(product(np.linspace(0, 1, 41), repeat=3))))
    best = None
    for trial in range(args.starts):
        start = x0+(0 if trial == 0 else .3*rng.standard_normal(24))
        sol = least_squares(residuals, start, method="trf", max_nfev=args.max_nfev)
        x = np.round(sol.x, 3).reshape(8, 3)
        edge = jacobians(x, shape_gradients(EDGE))
        m53 = jacobians(x, shape_gradients(P53)).min()
        neighbours = min(jacobians(v, shape_gradients(DENSE)).min() for o, v in neighbourhood(x).items()
                         if o != (0, 0, 0))
        cell = jacobians(x, fine).min()
        where = np.argmin(edge)/200
        print(f"{trial} cost={sol.cost:.3g} edge_min={edge.min():.5f} at={where:.3f} m53={m53:.4f} "
              f"cell_min={cell:.5f} neighbour_min={neighbours:.4f}", flush=True)
        if edge.min() > 0 and cell > 0 and m53 > .1 and neighbours > .05 and abs(where-args.zstar) < .05:
            if best is None or edge.min() < best[0]:
                best = (edge.min(), x)
    if best is not None:
        print("BEST", best[0], tuple(tuple(float(v) for v in row) for row in best[1]))
    return best


if __name__ == "__main__":
    main()
