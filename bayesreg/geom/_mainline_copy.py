"""Verbatim copy of four private helpers of the frozen mainline (contract C06-3; plan R10 "copy").

Source: jd_curl/three_d/linearization.py at commit eb808be401fde7ed6fee73c2d86212c8c463a513
(tag freeze-20261022 lineage), file sha256
e725e95cb6f85278c183b6da1e909926a270191e3785e5878b7a431c0b932387.

    _geometry_jvp       lines 40-45   test reference: q1.tile_jvp must equal it bitwise
    _geometry_flux      lines 48-57   used by q1.tile_vjp_accumulate
    _scatter_geometry   lines 60-66   test reference: q1._scatter_flux of _geometry_flux equals it bitwise
    _finish             lines 84-88   exported as q1.finish

Function bodies are copied character for character; only the imports below are rewritten as
absolute imports of the same jd_curl objects (relative imports in the original).  Do not edit
the bodies: tests/test_bayesreg/geom/test_geom_q1.py checks source text and bitwise output against
the original on random inputs.

Mainline private symbols that bayesreg.geom relies on without copying (review F1, 2026-10-24); the
same test file locks their source sha256 and the sha256 of the files holding them at eb808be (for the two
CPUPAR entries the source sha256 is locked in tests/test_bayesreg/model/test_parallel.py, cell_q1.py's file
sha256 in test_geom_q1.py):

    jd_curl/three_d/cell_q1.py   CellQ1Operators3D._transpose_add   called by q1._scatter_flux and by
                                                                    the copied _scatter_geometry
    jd_curl/three_d/cell_q1.py   CellQ1Operators3D._transpose_local called by q1.tile_vjp_local (E108:
                                                                    _transpose_add's check, error state
                                                                    and product; CPUPAR)
    jd_curl/three_d/cell_q1.py   CellQ1Operators3D._transpose_scatter  called by q1.tile_vjp_scatter (E108:
                                                                    _transpose_add's accumulation; CPUPAR)
    jd_curl/three_d/cell_q1.py   CellQ1Operators3D._apply           not called; q1._gradient and
                                                                    q1.tile_interpolate repeat its
                                                                    gather and product (bitwise tests)
    jd_curl/three_d/geometry.py  _compute_cell_geometry             not called; reference of the
                                                                    bitwise test of q1.tile_geometry

The public `geometry.cofactor` (same file, also locked) computes A in q1.tile_geometry.
"""
import numpy as np

from jd_curl.validation import require_finite_result
from jd_curl.three_d.cell_q1 import cell_node_indices


def _geometry_jvp(ops, A, v, tile):
    V = np.stack([np.stack([ops.derivative(v[..., r], axis, tile=tile) for axis in ("x", "y", "z")], axis=-1)
                  for r in range(3)], axis=-2)
    jd = np.sum(A*V, axis=(-2, -1))
    curl = np.stack((V[..., 2, 1]-V[..., 1, 2], V[..., 0, 2]-V[..., 2, 0], V[..., 1, 0]-V[..., 0, 1]), axis=-1)
    return jd, curl


def _geometry_flux(A, jd, curl):
    flux = A*jd[..., None, None]
    if curl is not None:
        flux[..., 2, 1] += curl[..., 0]
        flux[..., 1, 2] -= curl[..., 0]
        flux[..., 0, 2] += curl[..., 1]
        flux[..., 2, 0] -= curl[..., 1]
        flux[..., 1, 0] += curl[..., 2]
        flux[..., 0, 1] -= curl[..., 2]
    return flux


def _scatter_geometry(ops, A, jd, curl, tile, out, *, also_into=None):
    flux = _geometry_flux(A, jd, curl)
    indices = cell_node_indices(ops.grid, tile)
    for r in range(3):
        for j in range(3):
            ops._transpose_add(flux[..., r, j], ops.coefficients[j], indices, out[r],
                               also_into=None if also_into is None else also_into[r])


def _finish(out):
    require_finite_result(out, "complete derivative accumulator")
    result = out[:, 1:-1, 1:-1, 1:-1].copy().ravel(order="C")
    require_finite_result(result, "free derivative result")
    return result
