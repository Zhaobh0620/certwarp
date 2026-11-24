"""Q1 Gauss27 kernels, K_Q and DST-I on the device, FP64 (contract C02, C05-1, C06-3, C09.1).

Layouts (device):
- node arrays are component-major: (k, nz, ny, nx), k = 3 for a vector field (components x, y, z), 1 for a
  scalar field; the free vector (3 N_int,) is `field[:, 1:-1, 1:-1, 1:-1].reshape(-1)`, i.e. the layout of
  `boundary.inject_free` / `geom.finish`;
- a chunk is a run of whole z-slabs of cells [z0, z1) (cells (cz, cy, cx) in C order, x fastest, as
  `cell_q1.cell_tiles`); Gauss-point arrays of a chunk are (c, 27) with points in the order of
  `ops.points`; gradients (3, c, 3, 27) indexed [r, cell, j, point] = d_j u_r.
- the eight vertices of a cell follow `cell_q1.VERTICES` (x fastest).

Gather and scatter use eight shifted strided slabs of the node array, never index lists: the adjoint adds
each vertex's contribution with one strided in-place add, every destination once per add (no atomics), so
the result is deterministic.  The scatter is the mainline `TorchModel.scatter` (archive/seq_vp_3d/jd_curl/
three_d/backends/torch_ops.py lines 459-475 at eb808be: flux = A jd + curl-flux, then D^T flux per vertex)
with the vertex loop written on slabs instead of index lists; the curl-flux signs are verbatim.

`stiffness_apply` is `bayesreg.geom.stiffness_apply` on the device: the q1 branch of the frozen
`poisson.apply_poisson_operator` (jd_curl/three_d/poisson.py lines 59-85, `_tridiagonal` included) with
the same constants and operation order, batched over the three components.

`dst1` is verbatim the mainline function (torch_ops.py lines 98-104 at eb808be; file sha256
b314fdb0207d2707cfa5ace4f2a3cab27990666817680b6839794aba43472088): orthonormal DST-I by the FFT of the
odd extension.  `poisson_solve` is the `TorchPoisson.apply` sequence (torch_ops.py lines 113-121) one
component at a time (peak memory a third).
"""
from itertools import product

import numpy as np
import torch

VERTICES = tuple((x, y, z) for z, y, x in product((0, 1), repeat=3))     # = jd_curl cell_q1.VERTICES
CHUNK_CELLS = 131072          # target cells per device chunk (whole z-slabs, at least one)


class DeviceQ1:
    """Gauss27 operators of one grid on one device; chunks of whole z-slabs of cells.  The same class serves the
    corner operators of C05-2a (`geom.corners8`, P = 8 points in VERTICES order): every array below is then
    (..., 8) instead of (..., 27); the chunks depend only on the grid and chunk_cells, so both share them.  It also
    serves the closed27 operators of C05-2b (`geom.closed27`, P = 27 points {0, 1/2, 1}^3 in ops.points order); their
    `weights` attribute is None there (the uniform ops.weights are not the prior's weights, contract C05-2b)."""

    def __init__(self, grid, ops, device, chunk_cells=CHUNK_CELLS):
        if (ops.location, len(ops.points)) not in (("gauss27", 27), ("corners8", 8), ("closed27_uniform", 27)):
            raise ValueError("DeviceQ1: Gauss27 operators required (C02-3; corners8 for the C05-2a corner term, "
                             "closed27_uniform for the C05-2b closed27 part)")
        self.grid = grid
        self.shape = grid.shape
        self.cell_shape = grid.cell_shape
        self.n_points = P = len(ops.points)
        nz, ny, nx = grid.shape
        self.slab_cells = (ny-1)*(nx-1)
        slabs = max(1, int(chunk_cells)//self.slab_cells)
        self.slabs_per_chunk = slabs
        self.chunks = tuple((z0, min(z0+slabs, nz-1)) for z0 in range(0, nz-1, slabs))
        up = device.upload
        D = np.asarray(ops.coefficients)                                   # (3, P, 8) [j, p, a]
        self.Dcat = up(np.ascontiguousarray(D.transpose(2, 0, 1).reshape(8, 3*P)))   # [a, j*P + p]
        self.DcatT = self.Dcat.T.contiguous()                              # (3P, 8)
        self.basisT = up(np.asarray(ops.basis).T)                          # (8, P)
        self.weights = (None if ops.location == "closed27_uniform"        # C05-2b: weights from the ModelState
                        else up(ops.weights))                              # (P,) omega w_g (corners: omega/8)

    def cells(self, chunk):
        z0, z1 = chunk
        return (z1-z0)*self.slab_cells

    def cell_range(self, chunk):
        z0, z1 = chunk
        return z0*self.slab_cells, z1*self.slab_cells

    def _slab(self, field, chunk, vertex):
        z0, z1 = chunk
        vx, vy, vz = vertex
        nz, ny, nx = self.shape
        return field[:, z0+vz:z1+vz, vy:vy+ny-1, vx:vx+nx-1]

    def gather(self, field, chunk):
        """(k, nz, ny, nx) -> (k, c, 8): the eight vertex values of every cell of the chunk."""
        local = torch.stack([self._slab(field, chunk, v) for v in VERTICES], dim=-1)
        return local.reshape(field.shape[0], -1, 8)

    def gradient(self, field, chunk):
        """(3, nz, ny, nx) -> (3, c, 3, P) [r, cell, j, p] = d_j field_r at the Gauss points (P = 27)."""
        local = self.gather(field, chunk)
        k, c, _ = local.shape
        return (local.reshape(k*c, 8) @ self.Dcat).view(k, c, 3, self.n_points)

    def interpolate(self, field, chunk):
        """(1, nz, ny, nx) scalar node field -> (c, P): Q1 interpolation at the Gauss points (C02-4)."""
        local = self.gather(field, chunk)
        return local.reshape(-1, 8) @ self.basisT

    def scatter_add(self, out, flux, chunk):
        """out (3, nz, ny, nx) += node-wise adjoint of `gradient` applied to flux (3, c, 3, P)."""
        k, c = flux.shape[:2]
        local = (flux.reshape(k*c, 3*self.n_points) @ self.DcatT).view(k, c, 8)
        z0, z1 = chunk
        nz, ny, nx = self.shape
        cells = (k, z1-z0, ny-1, nx-1)
        for a, vertex in enumerate(VERTICES):
            self._slab(out, chunk, vertex).add_(local[:, :, a].reshape(cells))
        return out


def cofactor(F):
    """A = cof F on (3, c, 3, 27) [r, cell, col, p]: the formula and loop order of `geometry.cofactor`."""
    A = torch.empty_like(F)
    for r, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for c, l, m in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            A[r, :, c] = F[j, :, l]*F[k, :, m]-F[j, :, m]*F[k, :, l]
    return A


def determinant_row0(F, A):
    """J = sum_c F_0c A_0c, summed in the order of np.sum over the last axis of length 3."""
    return F[0, :, 0]*A[0, :, 0]+F[0, :, 1]*A[0, :, 1]+F[0, :, 2]*A[0, :, 2]


def curl_of(G):
    """(c_0, c_1, c_2) = (G_21 - G_12, G_02 - G_20, G_10 - G_01), each (c, 27)."""
    return G[2, :, 1]-G[1, :, 2], G[0, :, 2]-G[2, :, 0], G[1, :, 0]-G[0, :, 1]


def flux_of(A, gJ, gcurl, like):
    """The mainline flux A gJ + curl-flux(gcurl) (TorchModel.scatter / linearization._geometry_flux), on
    (3, c, 3, 27); A and gJ may be None (= zero), gcurl None or a (c0, c1, c2) triple."""
    flux = torch.zeros_like(like) if gJ is None else A*gJ[None, :, None, :]
    if gcurl is not None:
        c0, c1, c2 = gcurl
        flux[2, :, 1] += c0
        flux[1, :, 2] -= c0
        flux[0, :, 2] += c1
        flux[2, :, 0] -= c1
        flux[1, :, 0] += c2
        flux[0, :, 1] -= c2
    return flux


# ----------------------------------------------------------------------------------------------
# K_Q (C05-1) and the DST-I metric solve (C09.1)
# ----------------------------------------------------------------------------------------------

def _tridiagonal(field, axis, diagonal, off_diagonal):
    result = diagonal*field
    n = field.shape[axis]
    result.narrow(axis, 1, n-1).add_(off_diagonal*field.narrow(axis, 0, n-1))
    result.narrow(axis, 0, n-1).add_(off_diagonal*field.narrow(axis, 1, n-1))
    return result


def poisson_operator(fields, spacing):
    """apply_poisson_operator(metric="q1") of each component of (3, mz, my, mx) interior arrays."""
    out = torch.zeros_like(fields)
    for axis, h in enumerate(spacing[::-1]):
        factor = (1/np.float64(h))**2
        term = _tridiagonal(fields, axis+1, float(2*factor), float(-factor))
        for other in range(3):
            if other != axis:
                term = _tridiagonal(term, other+1, 2/3, 1/6)
        out += term
    return out


def stiffness_apply(field_cm, grid):
    """C05-1: K_Q u as the free vector, u a (3, nz, ny, nx) component-major node field with zero faces."""
    interior = field_cm[:, 1:-1, 1:-1, 1:-1]
    return (grid.omega*poisson_operator(interior, grid.spacing)).reshape(-1)


def dst1(x, axis):
    n = x.shape[axis]
    shape = list(x.shape)
    shape[axis] = 1
    zero = torch.zeros(shape, dtype=x.dtype, device=x.device)
    extension = torch.cat((zero, x, zero, -x.flip((axis,))), dim=axis)
    return -torch.fft.fft(extension, dim=axis).imag.narrow(axis, 1, n)/(2*(n+1))**.5


def poisson_solve(rhs, denominator, interior_shape):
    """idstn(dstn(rhs)/denominator) (orthonormal DST-I), rhs the free vector; one component at a time."""
    x = rhs.reshape((3,)+tuple(interior_shape))
    out = torch.empty_like(x)
    for d in range(3):
        y = x[d]
        for axis in (0, 1, 2):
            y = dst1(y, axis)
        y = y/denominator
        for axis in (0, 1, 2):
            y = dst1(y, axis)
        out[d] = y
    return out.reshape(-1)
