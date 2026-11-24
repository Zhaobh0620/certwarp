"""Q1 geometry at the Gauss27 points, its linearization and the consistent stiffness K_Q.

Contract docs/bayesreg/01_合同.md: C01 (grid, U0, free vector), C02 (Q1 geometry, Gauss27; C02-4
interpolation), C05-1 (K_Q), C06-2..4 (geometric parts of the residual-block Jacobians and adjoints).

Layout (all inherited from jd_curl, eb808be):
- node fields u, v: shape (nz, ny, nx, 3), mm, components (x, y, z); scalar node fields (nz, ny, nx);
- free vector: (3 N_int,), component-major, interior C-order (`boundary.inject_free` / `restrict_free`);
- Gauss-point arrays of a cell tile: shape (cells, 27[, 3[, 3]]); cells in C-order of grid.cell_shape
  (x fastest) starting at tile.start, points in the order of `ops.points` (z, y, x product, x fastest),
  weights `ops.weights` = omega * w_g; with the corner operators of C05-2a (`corners8`) the same kernels give
  (cells, 8[, 3[, 3]]) arrays at the cell corners, with the closed27 operators of C05-2b (`closed27`) (cells, 27[, 3[,
  3]]) arrays at the points {0, 1/2, 1}^3 (their weights: `simpson27_weights`, never `ops.weights`);
- grad[..., r, j] = d u_r / d x_j (F = I + grad, C02-2); curl = (F32-F23, F13-F31, F21-F12);
- the adjoint returns the free vector, i.e. `restrict_free` of the node-wise accumulator (`finish`):
  the adjoint of the jvp restricted to U0 in the Euclidean inner products of both spaces.

Three layers (review F1, 2026-10-24):
1. Validate once.  `node_field` (free vector or node field -> C-order float64 (nz,ny,nx,3); faces
   exactly zero unless the caller passes allow_nonzero_faces=True, which only gauss_geometry exposes,
   for the affine tests of C02) and `scalar_field`.  Whole-field cost: once per field and evaluation.
2. Tile kernels.  `tile_geometry`, `tile_jvp`, `tile_vjp_accumulate`, `tile_interpolate`, `finish`:
   O(1) layout checks of the whole field (dtype, shape, C order) and finiteness checks of the gathered
   tile values and tile outputs only.  They never call `ops.derivative`, `ops.interpolate` or the
   public `geometry.compute_cell_geometry`, each of which checks (and, for a strided component view,
   copies) the whole node field on every call.  Their inputs must come from layer 1; tile_jvp's v must
   lie in U0 (zero faces), the precondition of the adjoint identity with tile_vjp_accumulate + finish.
3. Streams and conveniences.  `iter_geometry`, `vjp_stream`, `gauss_gradient_energy` validate once and
   visit cell_tiles(grid, tile_size) in ascending order (C06-7).  `gauss_geometry`, `gauss_gradient`,
   `jvp`, `vjp` validate on every call and serve one explicit tile or, with tile=None, all cells of a
   small grid: tile=None raises above MAX_WHOLE_CELLS cells (a native volume has ~6.8e6 cells, i.e.
   ~13 GB per (cells, 27, 3, 3) array) -- native volumes stream.

Bitwise identities with the frozen mainline (tests/test_bayesreg/geom/test_geom_q1.py): tile_geometry's
F, A, J, curl equal `geometry._compute_cell_geometry` (grad computed once, F = grad + I, the public
`geometry.cofactor`, the mainline J and curl expressions); grad equals the stacked `ops.derivative`;
tile_jvp's dJ, dcurl equal `linearization._geometry_jvp`; tile_vjp_accumulate without ggrad equals
`_scatter_geometry`; tile_interpolate equals `ops.interpolate`.  Mainline private symbols relied on are
listed in _mainline_copy.py.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np

from jd_curl.validation import InputValidationError, as_float64_array, require_finite_result
from jd_curl.three_d.boundary import inject_free, pack_initial_displacement
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_node_indices, cell_tiles
from jd_curl.three_d.geometry import cofactor
from jd_curl.three_d.poisson import apply_poisson_operator
from jd_curl.three_d.validation import zero_boundary

from ._mainline_copy import _finish, _geometry_flux

TILE_SIZE = 8192          # C06-7 default tile (cells) for longdouble accumulation of Gauss-point sums
MAX_WHOLE_CELLS = 65536   # tile=None limit (8 default tiles; one (cells, 27, 3, 3) float64 array ~ 127 MB)


POINT_SETS = ("gauss27", "corners8", "closed27_uniform")   # C02-3 Gauss27; C05-2a cell corners; C05-2b closed27
                                                          # (the last two: log-volume options only)
SIMPSON_1D = (1/6, 2/3, 1/6)       # C05-2b / D48: closed Newton-Cotes (Simpson) weights of [0, 1] at {0, 1/2, 1}


def gauss27(grid):
    """C02-3: the point set of every Gauss-point block, `CellQ1Operators3D(grid, location="gauss27")`."""
    return CellQ1Operators3D(grid, location="gauss27")


def corners8(grid):
    """C05-2a: the eight corners of every cell, `CellQ1Operators3D(grid, location="corners8")`: points in the
    order of cell_q1.VERTICES (= the columns of cell_node_indices), weights omega/8; every value at a corner is
    that cell's own Q1 map there (shared vertices are not averaged).  Used only by the log-volume prior with
    logvol_points = "gauss27+corners"; the tile kernels below accept it unchanged."""
    return CellQ1Operators3D(grid, location="corners8")


def closed27(grid):
    """C05-2b: the closed27 points {0, 1/2, 1}^3 of every cell, `CellQ1Operators3D(grid, location="closed27_uniform")`:
    points in the order of `ops.points` (z, y, x product, x fastest), Q1 basis and derivative coefficients of that
    object (the contract's tables; the corner rows of the basis are one-hot).  Its `.weights` (uniform omega/27) are
    NOT the weights of the log-volume prior: those are `simpson27_weights` (or, for the uniform ablation, omega/27)
    times the share, computed in bayesreg.model.state.  Every value at a point is that cell's own Q1 map there (shared
    points are not averaged).  Used only by the log-volume prior with logvol_points = "gauss27+closed27"; the tile
    kernels below accept it unchanged."""
    return CellQ1Operators3D(grid, location="closed27_uniform")


def simpson27_weights(grid):
    """C05-2b / D48: omega s_p, s_p = sigma_i sigma_j sigma_k with sigma = (1/6, 2/3, 1/6), in the point order of
    `closed27(grid).points`; the floating-point expression and order of jd_curl's Gauss27 weights
    (omega * x * y * z for z, y, x in product(sigma, repeat=3)).  Sum omega up to rounding; corners omega/216, edge
    midpoints 4 omega/216, face centres 16 omega/216, the body centre 64 omega/216.  Read-only (27,)."""
    weights = np.array([grid.omega * x * y * z for z, y, x in product(SIMPSON_1D, repeat=3)], dtype=np.float64)
    require_finite_result(weights, "Simpson27 weights")
    weights.flags.writeable = False
    return weights


def _check_ops(ops):
    if not isinstance(ops, CellQ1Operators3D) or ops.location not in POINT_SETS:
        raise InputValidationError("ops: CellQ1Operators3D(grid, location='gauss27') required (C02-3; "
                                   "location='corners8' for the C05-2a corner term, 'closed27_uniform' for the C05-2b "
                                   "closed27 part)")


def _whole(ops, tile):
    """An explicit tile, or (tile=None) one tile of all cells on a grid with <= MAX_WHOLE_CELLS cells."""
    if tile is not None:
        return tile
    n = ops.grid.n_cells
    if n > MAX_WHOLE_CELLS:
        raise InputValidationError(
            f"tile=None means one tile of all {n} cells, above MAX_WHOLE_CELLS = {MAX_WHOLE_CELLS}; pass an "
            "explicit tile from cell_tiles(grid, TILE_SIZE) or stream with iter_geometry / vjp_stream / "
            "gauss_gradient_energy")
    return slice(0, n)


# ----------------------------------------------------------------------------------------------
# Layer 1: validate once
# ----------------------------------------------------------------------------------------------

def node_field(u, grid, *, name="u", allow_nonzero_faces=False):
    """Validate a vector field once (C01-2) for the tile kernels.

    A 1-D argument is a free vector (3 N_int,) and is injected with zero faces (`inject_free`).  A node
    field (nz,ny,nx,3) is checked for shape and finiteness and, unless allow_nonzero_faces, for exactly
    zero faces (U0).  Returns a C-order float64 (nz,ny,nx,3) array: the input itself when it already is
    one (never modified or frozen), otherwise a new array.
    """
    array = np.asarray(u)
    if array.ndim == 1:
        return inject_free(array, grid)
    field = np.ascontiguousarray(as_float64_array(array, name, grid.shape + (3,)))
    if not allow_nonzero_faces and not zero_boundary(field):
        raise InputValidationError(f"{name}: all three components on all six faces must be exactly zero (U0); "
                                   "pass a free vector or an inject_free field")
    return field


def scalar_field(field, grid, *, name="field"):
    """Validate a scalar node field (nz,ny,nx) once for tile_interpolate (C02-4)."""
    return np.ascontiguousarray(as_float64_array(field, name, grid.shape))


# ----------------------------------------------------------------------------------------------
# Layer 2: tile kernels (tile-local checks only)
# ----------------------------------------------------------------------------------------------

def _require_validated(field, grid, name, tail):
    if (not isinstance(field, np.ndarray) or field.dtype != np.float64 or field.shape != grid.shape + tail
            or not field.flags.c_contiguous):
        raise InputValidationError(f"{name}: C-order float64 array of shape {grid.shape + tail} required "
                                   "(validate once with node_field / scalar_field)")


def _gradient(u, ops, tile, name):
    """grad[c, g, r, j] = d_j u_r at the Gauss points of one tile.

    The gather and (cells,8) @ (8,27) products of `geometry._compute_cell_geometry` and `ops.derivative`
    (bitwise), with finiteness checked on the gathered values and the output only.
    """
    _require_validated(u, ops.grid, name, (3,))
    indices = cell_node_indices(ops.grid, tile)
    gathered = as_float64_array(u.reshape(-1, 3)[indices], name + " (tile values)")
    local = [gathered[..., r].copy(order="C") for r in range(3)]
    out = np.empty((len(indices), len(ops.points), 3, 3), dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        for r in range(3):
            for j, coefficients in enumerate(ops.coefficients):
                out[..., r, j] = local[r] @ coefficients.T
    require_finite_result(out, "Gauss-point gradient")
    return out


@dataclass(frozen=True)
class GaussGeometry:
    """C02-2 at the Gauss points of one tile; every array read-only.

    grad = grad u_h is computed once; F = grad + I, A = cof F (T02 explicit formula, `geometry.cofactor`),
    J = sum_c F_0c A_0c and curl with the mainline expressions (bitwise those of
    `geometry._compute_cell_geometry`); div = tr grad.  J is never clipped.
    """
    F: np.ndarray       # (cells, 27, 3, 3)
    A: np.ndarray       # (cells, 27, 3, 3)
    J: np.ndarray       # (cells, 27)
    curl: np.ndarray    # (cells, 27, 3)
    grad: np.ndarray    # (cells, 27, 3, 3)
    div: np.ndarray     # (cells, 27)


def tile_geometry(u, ops, *, tile):
    """C02-2/C02-3 at the Gauss27 points of `tile` from a node_field-validated u."""
    _check_ops(ops)
    grad = _gradient(u, ops, tile, "u")
    with np.errstate(over="raise", invalid="raise"):
        F = grad + np.eye(3)
        A = cofactor(F)
        J = np.sum(F[..., 0, :] * A[..., 0, :], axis=-1)
        curl = np.stack((F[..., 2, 1] - F[..., 1, 2], F[..., 0, 2] - F[..., 2, 0],
                         F[..., 1, 0] - F[..., 0, 1]), axis=-1)
        div = grad[..., 0, 0] + grad[..., 1, 1] + grad[..., 2, 2]
    for name, value in (("F", F), ("A", A), ("J", J), ("curl", curl), ("div", div)):
        require_finite_result(value, name)
    for value in (F, A, J, curl, grad, div):
        value.flags.writeable = False   # all newly allocated here; no caller-owned array is frozen
    return GaussGeometry(F, A, J, curl, grad, div)


def tile_jvp(v, ops, *, tile, A):
    """C06-3 on one tile: (dJ, dcurl, dgrad) = (A : V, curl V, V), V = grad v_h at the Gauss points.

    v: node_field-validated, in U0 (zero faces; not re-checked per tile).  A: cof F(u) of the same tile
    (tile_geometry(u).A), shape (cells, 27, 3, 3).  dJ, dcurl use the expressions of the mainline
    `_geometry_jvp` (Jacobi dJ = A : grad v_h, C02-2) on V from _gradient.  Returns
    (dJ (cells,27), dcurl (cells,27,3), dgrad (cells,27,3,3)), new arrays owned by the caller.
    """
    _check_ops(ops)
    V = _gradient(v, ops, tile, "v")
    A = as_float64_array(A, "A", V.shape)
    with np.errstate(over="raise", invalid="raise"):
        jd = np.sum(A*V, axis=(-2, -1))
        curl = np.stack((V[..., 2, 1]-V[..., 1, 2], V[..., 0, 2]-V[..., 2, 0], V[..., 1, 0]-V[..., 0, 1]), axis=-1)
    require_finite_result(jd, "JVP of J")
    require_finite_result(curl, "JVP of curl")
    return jd, curl, V


def _scatter_flux(ops, flux, indices, out):
    """The scatter loop of the mainline `_scatter_geometry` for a precomputed flux (cells, 27, 3, 3):
    `_scatter_flux(ops, _geometry_flux(A, jd, curl), indices, out)` is bitwise `_scatter_geometry(ops,
    A, jd, curl, tile, out)`.  Uses the mainline private `CellQ1Operators3D._transpose_add`."""
    for r in range(3):
        for j in range(3):
            ops._transpose_add(flux[..., r, j], ops.coefficients[j], indices, out[r])


def tile_vjp_accumulate(out, ops, *, tile, gJ=None, gcurl=None, ggrad=None, A=None):
    """Add the node-wise adjoint of tile_jvp on one tile into out, shape (3, nz, ny, nx).

    out: caller-owned, writable, C-order float64, component first (the mainline accumulator); only its
    layout is checked here, its finiteness once by finish.  gJ (cells,27), gcurl (cells,27,3) and
    ggrad (cells,27,3,3) may each be None (= zero); A = cof F(u) of the tile is required with gJ.  One
    flux, one scatter: flux = _geometry_flux(A, gJ, gcurl) (verbatim mainline: A gJ plus the curl flux)
    + ggrad.  Boundary rows of out receive contributions too; finish drops them.  Returns out.
    """
    _check_ops(ops)
    grid = ops.grid
    if (not isinstance(out, np.ndarray) or out.dtype != np.float64 or out.shape != (3,) + grid.shape
            or not out.flags.c_contiguous or not out.flags.writeable):
        raise InputValidationError("out: writable C-order float64 array of shape (3, nz, ny, nx) required")
    indices = cell_node_indices(grid, tile)
    shape = (len(indices), len(ops.points))
    curl = None if gcurl is None else as_float64_array(gcurl, "gcurl", shape + (3,))
    with np.errstate(over="raise", invalid="raise"):
        flux = None
        if gJ is not None:
            if A is None:
                raise InputValidationError("A: cof F(u) of the tile (tile_geometry(u).A) is required with gJ")
            flux = _geometry_flux(as_float64_array(A, "A", shape + (3, 3)), as_float64_array(gJ, "gJ", shape), curl)
        elif curl is not None:
            flux = _geometry_flux(np.zeros((3, 3)), np.zeros(shape), curl)
        if ggrad is not None:
            ggrad = as_float64_array(ggrad, "ggrad", shape + (3, 3))
            if flux is None:
                flux = ggrad      # read only below
            else:
                flux += ggrad     # flux is newly allocated by _geometry_flux
        if flux is not None:
            _scatter_flux(ops, flux, indices, out)
    return out


@dataclass(frozen=True)
class TileAdjoint:
    """The adjoint of one tile before its scatter (tile_vjp_local): the (cells, 8) node indices of the tile and
    the nine (cells, 8) local products local[3 r + j] = flux[..., r, j] @ ops.coefficients[j]."""
    indices: np.ndarray
    local: tuple


def tile_vjp_local(ops, *, tile, gJ=None, gcurl=None, ggrad=None, A=None):
    """The part of tile_vjp_accumulate that does not touch the accumulator (safe in a worker thread, CPUPAR).

    Same arguments, checks and flux as tile_vjp_accumulate; the nine products are the mainline
    `CellQ1Operators3D._transpose_local` (E108: _transpose_add's check, error state and product), in the (r, j)
    order of `_scatter_flux`.  Returns a TileAdjoint, or None when every cotangent is None (nothing to add).
    tile_vjp_scatter(out, ops, tile_vjp_local(...)) is bitwise tile_vjp_accumulate(out, ...).
    """
    _check_ops(ops)
    indices = cell_node_indices(ops.grid, tile)
    shape = (len(indices), len(ops.points))
    curl = None if gcurl is None else as_float64_array(gcurl, "gcurl", shape + (3,))
    with np.errstate(over="raise", invalid="raise"):
        flux = None
        if gJ is not None:
            if A is None:
                raise InputValidationError("A: cof F(u) of the tile (tile_geometry(u).A) is required with gJ")
            flux = _geometry_flux(as_float64_array(A, "A", shape + (3, 3)), as_float64_array(gJ, "gJ", shape), curl)
        elif curl is not None:
            flux = _geometry_flux(np.zeros((3, 3)), np.zeros(shape), curl)
        if ggrad is not None:
            ggrad = as_float64_array(ggrad, "ggrad", shape + (3, 3))
            if flux is None:
                flux = ggrad      # read only below
            else:
                flux += ggrad     # flux is newly allocated by _geometry_flux
    if flux is None:
        return None
    local = tuple(ops._transpose_local(flux[..., r, j], ops.coefficients[j]) for r in range(3) for j in range(3))
    return TileAdjoint(indices, local)


def tile_vjp_scatter(out, ops, part):
    """Add a TileAdjoint into out (3, nz, ny, nx) in the order of `_scatter_flux`: components r, then j, then the
    eight vertices (the mainline `CellQ1Operators3D._transpose_scatter`, E108: _transpose_add's accumulation).
    Called by one thread, tiles in ascending order, this gives every node its additions in the serial order."""
    _check_ops(ops)
    grid = ops.grid
    if (not isinstance(out, np.ndarray) or out.dtype != np.float64 or out.shape != (3,) + grid.shape
            or not out.flags.c_contiguous or not out.flags.writeable):
        raise InputValidationError("out: writable C-order float64 array of shape (3, nz, ny, nx) required")
    if part is None:
        return out
    for r in range(3):
        for j in range(3):
            ops._transpose_scatter(part.local[3*r+j], part.indices, out[r])
    return out


finish = _finish   # C06-3: check the whole accumulator once, return its interior as the free vector


def tile_interpolate(field, ops, *, tile):
    """C02-4: Q1 interpolation of a scalar_field-validated node field at the Gauss points of one tile,
    shape (cells, 27); the gather and product of `ops.interpolate` (bitwise), tile-local checks."""
    _check_ops(ops)
    _require_validated(field, ops.grid, "field", ())
    indices = cell_node_indices(ops.grid, tile)
    local = as_float64_array(field.ravel()[indices], "field (tile values)")
    with np.errstate(over="raise", invalid="raise"):
        out = local @ ops.basis.T
    require_finite_result(out, "Q1 interpolation")
    return out


# ----------------------------------------------------------------------------------------------
# Layer 3: streams (validate once, ascending tiles) and per-call conveniences
# ----------------------------------------------------------------------------------------------

def iter_geometry(u, ops, *, tile_size=TILE_SIZE):
    """Validate u once (node_field, U0) now; return an iterator of (tile, tile_geometry(u, tile))
    over cell_tiles(grid, tile_size) in ascending order."""
    _check_ops(ops)
    u = node_field(u, ops.grid)
    tiles = list(cell_tiles(ops.grid, tile_size))
    return ((tile, tile_geometry(u, ops, tile=tile)) for tile in tiles)


def vjp_stream(u, cotangents, ops, *, tile_size=TILE_SIZE):
    """Adjoint summed over all tiles in ascending order with u validated once; returns the free vector.

    Per tile: geo = tile_geometry(u), (gJ, gcurl, ggrad) = cotangents(tile, geo) (each may be None),
    one tile_vjp_accumulate with A = geo.A; finish at the end.  Example (GN HVP of a block): validate
    the direction once, v = node_field(p, grid), and let cotangents weight tile_jvp(v, ops, tile=tile,
    A=geo.A) -- A is computed once per tile for both the jvp and the adjoint.
    """
    _check_ops(ops)
    u = node_field(u, ops.grid)
    out = np.zeros((3,) + ops.grid.shape)
    for tile in cell_tiles(ops.grid, tile_size):
        geo = tile_geometry(u, ops, tile=tile)
        gJ, gcurl, ggrad = cotangents(tile, geo)
        tile_vjp_accumulate(out, ops, tile=tile, gJ=gJ, gcurl=gcurl, ggrad=ggrad, A=geo.A)
    return finish(out)


def gauss_gradient_energy(u, ops, *, tile_size=TILE_SIZE):
    """B7-1: sum_q omega w_g |grad u_h(x_q)|^2, u in U0 validated once, longdouble per tile in ascending
    tile order (C06-7)."""
    _check_ops(ops)
    u = node_field(u, ops.grid)
    weights = ops.weights[None, :, None, None]
    total = np.longdouble(0)
    with np.errstate(over="raise", invalid="raise"):
        for tile in cell_tiles(ops.grid, tile_size):
            total += np.sum(weights * _gradient(u, ops, tile, "u")**2, dtype=np.longdouble)
    result = float(total)
    require_finite_result(result, "Gauss gradient energy")
    return result


def gauss_geometry(u, ops, *, tile=None, allow_nonzero_faces=False):
    """C02-2/C02-3 on one tile (all cells of a small grid when tile=None), u validated on this call.

    allow_nonzero_faces=True accepts u outside U0: only for the affine tests of the C02 gate.
    """
    _check_ops(ops)
    tile = _whole(ops, tile)
    return tile_geometry(node_field(u, ops.grid, allow_nonzero_faces=allow_nonzero_faces), ops, tile=tile)


def gauss_gradient(v, ops, *, tile=None):
    """grad[c, g, r, j] = d_j v_{h,r}(x_{c,g}) (C06-3 V_q), shape (cells, 27, 3, 3); v in U0."""
    _check_ops(ops)
    tile = _whole(ops, tile)
    return _gradient(node_field(v, ops.grid, name="v"), ops, tile, "v")


def jvp(u, v, ops, *, tile=None):
    """C06-3 at u along v (both in U0, validated on this call) on one tile: (dJ, dcurl, dgrad)."""
    _check_ops(ops)
    tile = _whole(ops, tile)
    A = tile_geometry(node_field(u, ops.grid), ops, tile=tile).A
    return tile_jvp(node_field(v, ops.grid, name="v"), ops, tile=tile, A=A)


def vjp(u, gJ, gcurl, ggrad, ops, *, tile=None, A=None):
    """C06-3 adjoint of `jvp` on one tile restricted to U0: the free vector (3 N_int,) (finish).

    u (in U0) is read only when gJ is given and A is not.
    """
    _check_ops(ops)
    tile = _whole(ops, tile)
    if gJ is not None and A is None:
        A = tile_geometry(node_field(u, ops.grid), ops, tile=tile).A
    out = np.zeros((3,) + ops.grid.shape)
    tile_vjp_accumulate(out, ops, tile=tile, gJ=gJ, gcurl=gcurl, ggrad=ggrad, A=A)
    return finish(out)


def stiffness_apply(u, grid):
    """C05-1: K_Q u on U0, per component omega * apply_poisson_operator(u_d|int, metric="q1").

    u: free vector (3 N_int,) or (nz,ny,nx,3) field with exactly zero faces; returns the free
    vector K_Q u (same layout as `vjp`).  u^T K_Q u = gauss_gradient_energy(u) (B7-1).
    """
    array = np.asarray(u)
    if array.ndim == 1:
        theta = as_float64_array(array, "u", (3 * grid.n_free_scalar,))
    else:
        theta = pack_initial_displacement(array, grid)
    fields = theta.reshape((3,) + grid.interior_shape)
    with np.errstate(over="raise", invalid="raise"):
        out = np.stack([grid.omega * apply_poisson_operator(s, grid, metric="q1") for s in fields])
    require_finite_result(out, "K_Q action")
    return out.ravel(order="C")
