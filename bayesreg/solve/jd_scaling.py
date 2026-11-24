"""C09.1-3a JD affine-scaling metric of the PG direction (work package JDS; path only).

Contract docs/bayesreg/01_合同.md C09.1-3a (mainline T62/T70, docs/reference/04_..._ZH.md).  F_B, the certificate,
the C07 acceptance order, the C08 stable comparison and the C13 stopping rules are unchanged; only the symmetric
positive definite metric of the PG direction changes:

    p = -S~ M^-1 S~ g,   S~ = diag(r_a (x) 1_3),   r_a = min(1, max(J_a, J_floor) / J_act),

J_act = .25, J_floor = 1e-12, and J_a the node estimate: the minimum over the (eight) cells K adjacent to the
interior node a of m_K = min(J at the 8 corners of K, J at the 27 Gauss points of K), J = det(I + grad u_h) of the
Q1 field.  The estimate is a heuristic (not a bound, not the certificate) and enters no decision.  S~ > 0, so
S~ M^-1 S~ is symmetric positive definite and g.p = -||M^-1/2 S~ g||^2 < 0; BB2 uses S~ M^-1 S~ in place of M^-1
(its identity M^-1 y = p_{k-1} - p_k holds for the frozen operator).

Freezing (C09.1-3/3a): S~ is computed with M at the freeze point (option (a): the (level, EM round) start; option
(b): every reference refresh) and kept for the inner solve, except for the refresh rule: when, within one inner
solve, 3 consecutive candidates were rejected (any rejection reason; a line_search_failed ends the solve and is
never followed by a refresh), or more than half of the most recent 26 candidates (i.e. > 13) were rejected
JACOBIAN_VIOLATION, S~ is recomputed (M unchanged) at the next accepted state, before the next direction (run_pg
does it at the start of the following iteration, after the stop checks: no refresh when the solve ends there), and
counted (`jd_scaling_refresh`).  Both trigger statistics count the candidates since the last computation of S~.
After a refresh BB2 keeps its secant pair and uses the refreshed operator: the stored previous direction is
recomputed as -S~' M^-1 S~' g_{k-1} (one extra metric solve), so that p_{k-1} - p_k = S~' M^-1 S~' y exactly as
the rule asks.  Every computation of S~ records min J_a, the fraction of nodes with r_a < 1, min r_a, its time.

Implementation: the corner and Gauss-point gradients of the Q1 field per slab of cell layers (z), separably: the
x-derivative of a Q1 cell is independent of xi and bilinear in (eta, zeta) on the cell's four x-edge differences
(likewise y, z); for 0 and 1 the interpolation is a slice.  numpy only, float64; the J values equal those of
`bayesreg.geom.tile_geometry` (Gauss) and of `CellQ1Operators3D(location="corners8")` (corners) to rounding.
For the torch64 backend S~ is computed on the host from the host u (the certificate's input) and applied to the
host vectors around the device metric solve.

Cell estimate option (C09.1-3a, last item; `JDScalingConfig.cell_estimate`, CLI --jd-cell-estimate; path only,
D47: contract d7231eb, implementation 38e63ab, review docs/bayesreg/reviews/JDC_复核_20261030.md, fixes JDC1; basis
runs/20261030-diag-g3, where the only cell near J = 0 had its minimum inside an edge).  Default of JDScalingConfig:
"certificate" since work package DEF (2026-10-31; the lead's decision after runs/20261031-jdc-timing, contract C09.1-3a
last item, D47); "gauss27+corners" (the default before DEF) is an explicit option, bitwise the code before DEF.  With
metric_jd_scaling "off" the option has no effect: PGConfig then holds no JDScalingConfig and records "none" (NONE);
an explicit one is refused (bayesreg.solve.pg.PGConfig).  The low-level functions `cell_estimate` and
`scaling_factors` keep estimate = "gauss27+corners" when none is given (they need cert / cert_config for "certificate"
and "certificate_bernstein").
  "gauss27+corners"  (explicit option; the default before DEF) m_K as above, bit for bit;
  "closed27"         m_K over Gauss27 and the closed27 points {0, 1/2, 1}^3 (corners, 12 edge midpoints, 6 face
                     centres, centre): 53 points by the same separable slab computation (J equal to
                     CellQ1Operators3D(location="closed27_uniform") to rounding).  A heuristic, not a bound: Q1 J on an
                     edge is a quadratic in the edge parameter, so a dip off the edge midpoint can stay invisible at all
                     53 points (tests: the off-midpoint edge dip, where closed27 equals the default and is blocked);
  "certificate"      m'_K = min(m_K, b_K), b_K the per-cell certified lower bound of the accepted, certified state at
                     which S~ is computed (a rigorous lower bound of J over the whole cell): the per-cell path of the
                     C07-2 certify_map (edge hull with the whole-tile rule -> float Bernstein prescreen -> interval
                     Bernstein fallback and subdivision), replicated read-only by `bayesreg.eval.cert_eval.cell_pass`
                     with the same configuration.  Provenance of the replicated path: jd_curl/three_d/feasibility.py
                     `_certify_map_parallel` (jd_curl frozen at eb808be, file unchanged, sha256 403f090c...); jd_curl
                     gives no per-cell output.  The state's CertOutcome (the certify_map call that accepted it) is
                     passed in, never a new certify_map call.  Checked: its certificate_version equals that of the
                     configuration, sha256(u) = its state_hash, every cell proven, min_K b_K == its min_jd_lower_bound
                     bit for bit; anything else raises backend.CertificateContractError (implementation error).  Cells
                     without a bound keep m_K.  One cached entry keyed by (state_hash, grid id, certificate version,
                     tile size): the same state again (e.g. the next round starting where a line search failed)
                     reuses its bounds.
  "certificate_bernstein"  (work package JDTI; contract C09.1-3a item "certificate_bernstein", draft JDT 2026-10-31,
                     D53; NOT the default, path only) m''_K = min(m_K, b~_K): b~_K = max(b_K, l_K) for K in S = {cells
                     proven by the edge hull (cell_pass code PROVEN_HULL) with b_K < J_act} and l_K finite, b~_K = b_K
                     elsewhere; l_K = jd_curl certificate_prescreen.prescreen_lower_bounds(u, grid, D, S) with D =
                     CellQ1Operators3D(grid, location="closed27_uniform").coefficients (the certify_map prescreen
                     function and table, read-only; min beta_fl - B of T75, a rigorous lower bound of J over K).  b~ is
                     a rigorous lower bound but, on hull cells, not the bound the certificate used.  Checks: (a) every
                     check of "certificate" on b_K (raises CertificateContractError); (b) for every raised cell (K in
                     S, l_K > b_K) l_K <= min of J at its 53 points (Gauss27 and closed27, from the same
                     closed27_cell_estimate slab pass that gives m_K), no tolerance -- a violating cell falls back to
                     b_K and is counted, recorded and warned (`j53_warning_row`); the run total is
                     `j53_violations_total`.  Coefficient table failing with FloatingPointError: l all NaN, b~ = b.
                     Cache: the "certificate" entry (which also keeps the per-cell codes) plus one entry of (S, l_S)
                     keyed by (that key, "certificate_bernstein", J_act.hex()); the check (b) runs on every call.
The S~ records of "gauss27+corners" are those of the code before the option; the other options (the default
"certificate" among them) add the fields listed in `scaling_factors`.
"""
from collections import deque
from dataclasses import dataclass
from itertools import product
import time

import numpy as np

from jd_curl.validation import InputValidationError, finite_real
from jd_curl.three_d.validation import integer_at_least
from jd_curl.three_d.cell_q1 import CellQ1Operators3D
from jd_curl.three_d.certificate_prescreen import PRESCREEN_KERNEL, prescreen_lower_bounds

from ..geom import node_field

AFFINE_V1 = "affine_v1"
OFF = "off"
MODES = (AFFINE_V1, OFF)
J_ACT = .25                        # mainline T62 jd_affine_activation
J_FLOOR = 1e-12                    # mainline T62 J_floor
REFRESH_CONSECUTIVE = 3            # C09.1-3a
REFRESH_WINDOW = 26                # C09.1-3a: the most recent 26 candidates (one full line search)
SLAB_CELLS = 65536                 # cells per slab of z layers (at least one layer)
GAUSS_NODES = (.5 - np.sqrt(15.) / 10, .5, .5 + np.sqrt(15.) / 10)     # as CellQ1Operators3D(location="gauss27")
CORNER_NODES = (0., 1.)
CLOSED_NODES = (0., .5, 1.)        # as CellQ1Operators3D(location="closed27_uniform")
JACOBIAN_VIOLATION = "JACOBIAN_VIOLATION"
GAUSS_CORNERS = "gauss27+corners"  # C09.1-3a jd_cell_estimate (D47): the default before DEF (explicit option since)
CLOSED27 = "closed27"
CERTIFICATE = "certificate"
CERTIFICATE_BERNSTEIN = "certificate_bernstein"   # JDTI: C09.1-3a, D53 (explicit option, not the default)
CERTIFICATE_ESTIMATES = (CERTIFICATE, CERTIFICATE_BERNSTEIN)  # the options that read cert / cert_config
CELL_ESTIMATES = (GAUSS_CORNERS, CLOSED27, CERTIFICATE, CERTIFICATE_BERNSTEIN)
DEFAULT_CELL_ESTIMATE = CERTIFICATE  # DEF (2026-10-31): C09.1-3a last item, D47
J53_VIOLATION_CELLS = 10           # C09.1-3a certificate_bernstein check (b): at most 10 violating cell ids recorded
NONE = "none"                      # recorded jd_cell_estimate with metric_jd_scaling "off" (no effect; C09.1-3a, DEF)
ESTIMATE = ("J_a = min over the cells adjacent to the interior node a of m_K = min(J at the 8 corners, J at the 27 "
            "Gauss points) of the Q1 field (heuristic, not a bound; no decision uses it)")
ESTIMATES = {
    GAUSS_CORNERS: ESTIMATE,
    CLOSED27: ("J_a = min over the cells adjacent to the interior node a of m_K = min(J at the 27 Gauss points, J at "
               "the closed27 points {0, 1/2, 1}^3: corners, edge midpoints, face centres, centre) of the Q1 field "
               "(heuristic, not a bound; no decision uses it)"),
    CERTIFICATE: ("J_a = min over the cells adjacent to the interior node a of m'_K = min(m_K, b_K): m_K = min(J at the "
                  "8 corners, J at the 27 Gauss points), b_K the per-cell certified lower bound of the accepted, "
                  "certified state (per-cell path of the C07-2 certify_map, replicated read-only; min_K b_K equal to "
                  "the reported min_jd_lower_bound bit for bit); no decision uses it"),
    CERTIFICATE_BERNSTEIN: (
        "J_a = min over the cells adjacent to the interior node a of m''_K = min(m_K, b~_K): m_K = min(J at the 8 "
        "corners, J at the 27 Gauss points); b~_K = max(b_K, l_K) on S = {edge-hull-proven cells with b_K < J_act}, "
        "b_K elsewhere; b_K the per-cell certified lower bound of the accepted, certified state (as certificate), l_K "
        "the float Bernstein prescreen bound of jd_curl (T75, min beta_fl - B; the certify_map prescreen function and "
        "table, a rigorous lower bound, but on hull cells not the one the certificate used); a raised cell whose l_K "
        "exceeds J at its 53 points falls back to b_K (counted, warned); no decision uses it"),
}


@dataclass(frozen=True)
class JDScalingConfig:
    """C09.1-3a parameters (algorithm parameters, C09.4-15); cell_estimate: the jd_cell_estimate option (D47; default
    "certificate" since DEF, "gauss27+corners" the explicit option bitwise the code before)."""
    activation: float = J_ACT
    floor: float = J_FLOOR
    refresh_consecutive: int = REFRESH_CONSECUTIVE
    refresh_window: int = REFRESH_WINDOW
    cell_estimate: str = DEFAULT_CELL_ESTIMATE

    def __post_init__(self):
        object.__setattr__(self, "activation", finite_real(self.activation, "jd activation", positive=True))
        object.__setattr__(self, "floor", finite_real(self.floor, "jd floor", positive=True))
        if not self.floor < self.activation <= 1:
            raise InputValidationError("JD scaling: J_floor < J_act <= 1 required (mainline T62)")
        object.__setattr__(self, "refresh_consecutive",
                           integer_at_least(self.refresh_consecutive, "jd refresh_consecutive"))
        object.__setattr__(self, "refresh_window", integer_at_least(self.refresh_window, "jd refresh_window"))
        if self.cell_estimate not in CELL_ESTIMATES:
            raise InputValidationError(f"jd_cell_estimate: one of {CELL_ESTIMATES}")

    def record(self):
        return dict(mode=AFFINE_V1, contract="C09.1-3a", direction="p = -S~ M^-1 S~ g", activation=self.activation,
                    floor=self.floor, factor="r_a = min(1, max(J_a, floor)/activation), same for the 3 components",
                    estimate=ESTIMATES[self.cell_estimate], cell_estimate=self.cell_estimate,
                    freeze="with M (option (a): (level, EM round); option (b): reference refresh)",
                    refresh=dict(consecutive_rejections=self.refresh_consecutive,
                                 window=self.refresh_window, jacobian_violation_share="> 1/2 of the window",
                                 when="after the next accepted step, M unchanged; statistics restart at every "
                                      "computation of S~; BB2 previous direction recomputed with the new operator"))


# ---- the J estimate -----------------------------------------------------------------------------------

def _end(a, axis, hi):
    index = [slice(None)] * a.ndim
    index[axis] = slice(1, None) if hi else slice(None, -1)
    return a[tuple(index)]


def _lerp(a, t, axis):
    """(1 - t) a[lo] + t a[hi] along `axis` (exact slices for t = 0, 1)."""
    if t == 0.:
        return _end(a, axis, False)
    if t == 1.:
        return _end(a, axis, True)
    return (1. - t) * _end(a, axis, False) + t * _end(a, axis, True)


def _pairs(closed=False):
    other = CLOSED_NODES if closed else CORNER_NODES
    return tuple(product(GAUSS_NODES, GAUSS_NODES)) + tuple(product(other, other))


def _points(closed=False):
    """(xi, eta, zeta) of the 27 Gauss points and the 8 corners (closed: the 27 points {0, 1/2, 1}^3 instead of the
    corners; the centre is also a Gauss point)."""
    other = CLOSED_NODES if closed else CORNER_NODES
    return tuple(product(GAUSS_NODES, repeat=3)) + tuple(product(other, repeat=3))


DEFAULT_POINTS = frozenset(_points())                          # Gauss27 and the 8 corners (35 points)


def _det(dx, dy, dz):
    """det(I + [d_x u, d_y u, d_z u]) with F[r, j] = delta_rj + d_j u_r; dx[r] = d_x u_r (component first)."""
    a00, a01, a02 = 1. + dx[0], dy[0], dz[0]
    a10, a11, a12 = dx[1], 1. + dy[1], dz[1]
    a20, a21, a22 = dx[2], dy[2], 1. + dz[2]
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)


def slab_point_jacobians(u, grid, k0, k1, *, closed=False):
    """{(xi, eta, zeta): J (k1 - k0, ny - 1, nx - 1)} at the 27 Gauss points and the 8 corners of the cells of the
    z layers k0..k1-1 (u validated by node_field); closed=True: at the 27 Gauss points and the 27 closed27 points
    {0, 1/2, 1}^3 (53 keys: the centre is shared).  For tests and `cell_jd_estimate`."""
    hx, hy, hz = grid.spacing
    U = np.moveaxis(u[k0:k1 + 1], -1, 0)                         # (3, Z+1, ny, nx), components x, y, z
    dx = (U[..., 1:] - U[..., :-1]) / hx                         # x edges: (3, Z+1, ny, nx-1)
    dy = (U[..., 1:, :] - U[..., :-1, :]) / hy                   # y edges: (3, Z+1, ny-1, nx)
    dz = (U[:, 1:] - U[:, :-1]) / hz                             # z edges: (3, Z, ny, nx)
    nodes = GAUSS_NODES + (CLOSED_NODES if closed else CORNER_NODES)
    # d_x u at (eta, zeta): along y (axis 2) then z (axis 1); d_y u at (xi, zeta): along x (axis 3) then z;
    # d_z u at (xi, eta): along x then y
    x_y = {t: _lerp(dx, t, 2) for t in nodes}
    y_x = {t: _lerp(dy, t, 3) for t in nodes}
    z_x = {t: _lerp(dz, t, 3) for t in nodes}
    DX = {(e, z): _lerp(x_y[e], z, 1) for e, z in _pairs(closed)}
    DY = {(x, z): _lerp(y_x[x], z, 1) for x, z in _pairs(closed)}
    DZ = {(x, e): _lerp(z_x[x], e, 2) for x, e in _pairs(closed)}
    return {(x, e, z): _det(DX[e, z], DY[x, z], DZ[x, e]) for x, e, z in _points(closed)}


def _slabs(grid, slab_cells):
    nz, ny, nx = grid.shape
    step = max(1, integer_at_least(slab_cells, "slab_cells") // ((ny - 1) * (nx - 1)))
    for k0 in range(0, nz - 1, step):
        yield k0, min(nz - 1, k0 + step)


def _minimum_into(block, values):
    values = iter(values)
    np.copyto(block, next(values))
    for value in values:
        np.minimum(block, value, out=block)


def cell_jd_estimate(u, grid, *, slab_cells=SLAB_CELLS):
    """m_K for every cell (grid.cell_shape): min of J over the 8 corners and the 27 Gauss points."""
    u = node_field(u, grid)
    out = np.empty(grid.cell_shape)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for k0, k1 in _slabs(grid, slab_cells):
            _minimum_into(out[k0:k1], slab_point_jacobians(u, grid, k0, k1).values())
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("JD scaling: nonfinite cell J estimate")
    return out


def closed27_cell_estimate(u, grid, *, slab_cells=SLAB_CELLS):
    """(m_K over Gauss27 and closed27, m_K of "gauss27+corners" over Gauss27 and the corners), both grid.cell_shape;
    one slab pass (jd_cell_estimate "closed27")."""
    u = node_field(u, grid)
    out, default = np.empty(grid.cell_shape), np.empty(grid.cell_shape)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for k0, k1 in _slabs(grid, slab_cells):
            values = slab_point_jacobians(u, grid, k0, k1, closed=True)
            _minimum_into(default[k0:k1], (v for p, v in values.items() if p in DEFAULT_POINTS))
            block = out[k0:k1]
            np.copyto(block, default[k0:k1])
            for p, v in values.items():
                if p not in DEFAULT_POINTS:
                    np.minimum(block, v, out=block)
    if not (np.all(np.isfinite(out)) and np.all(np.isfinite(default))):
        raise FloatingPointError("JD scaling: nonfinite cell J estimate")
    return out, default


def node_minimum(m, grid):
    """min of a per-cell array m (grid.cell_shape) over the 8 cells adjacent to every interior node."""
    nz, ny, nx = grid.interior_shape
    out = m[:nz, :ny, :nx].copy()
    for dz, dy, dx in product((0, 1), repeat=3):
        if dz or dy or dx:
            np.minimum(out, m[dz:dz + nz, dy:dy + ny, dx:dx + nx], out=out)
    return out


def node_jd_estimate(u, grid, *, slab_cells=SLAB_CELLS):
    """J_a on the interior nodes (grid.interior_shape): the min of m_K over the 8 cells adjacent to a."""
    return node_minimum(cell_jd_estimate(u, grid, slab_cells=slab_cells), grid)


# ---- jd_cell_estimate "certificate": per-cell certified lower bounds of an accepted, certified state ----------

_CELL_BOUNDS_CACHE = {}            # one entry: key (state_hash, grid id, certificate version, tile size), bounds and
                                   # the per-cell codes of cell_pass (JDTI: the codes give S of certificate_bernstein)
_BERNSTEIN_CACHE = {}              # JDTI: one entry: key (the entry's key above, "certificate_bernstein", J_act.hex()),
                                   # S (cell ids) and l_S (C09.1-3a certificate_bernstein, two-level cache)


def clear_cell_bounds_cache():
    _CELL_BOUNDS_CACHE.clear()
    _BERNSTEIN_CACHE.clear()


def _certificate_cells(u, grid, cert, cert_config, name=CERTIFICATE):
    """(b (grid.cell_shape, read-only), per-cell codes of cell_pass (flat, read-only), record, cache key, u) of
    `certificate_cell_bounds`, with its checks; `name` only labels the messages."""
    from jd_curl.three_d.feasibility import certificate_version
    from .backend import CertificateContractError, check_certificate_config
    from .stopping import state_hash
    tick = time.perf_counter()
    if cert_config is None:
        raise InputValidationError(f"jd_cell_estimate '{name}': the C07-2 certificate configuration is required")
    check_certificate_config(cert_config, cert_config.tile_size)
    if cert is None or not getattr(cert, "passed", False) or cert.min_jd_lower_bound is None:
        raise InputValidationError(f"jd_cell_estimate '{name}': the CertOutcome of an accepted, certified state is "
                                   "required")
    version = certificate_version(cert_config)
    if getattr(cert, "certificate_version", None) != version:
        raise CertificateContractError(f"jd_cell_estimate '{name}': the certificate passed with u was not made "
                                       "with the C07-2 configuration given (certificate_version differs)")
    u = node_field(u, grid)
    digest = state_hash(u)
    if digest != cert.state_hash:
        raise CertificateContractError(f"jd_cell_estimate '{name}': sha256(u) differs from the state_hash of the "
                                       "certificate passed with it")
    key = (digest, grid.grid_id, version, int(cert_config.tile_size))
    cached = _CELL_BOUNDS_CACHE.get("key") == key
    if cached:
        bounds, code = _CELL_BOUNDS_CACHE["bounds"], _CELL_BOUNDS_CACHE["code"]
    else:
        from ..eval.cert_eval import PROVEN_FALLBACK, cell_pass
        code, lower = cell_pass(u, grid, cert_config)
        if np.any(code > PROVEN_FALLBACK):
            raise CertificateContractError(f"jd_cell_estimate '{name}': the per-cell path leaves a cell unproven "
                                           "although the certificate passed")
        bounds = lower.reshape(grid.cell_shape)
        bounds.flags.writeable = False
        code.flags.writeable = False
    finite = np.isfinite(bounds)
    lowest = float(np.min(bounds, where=finite, initial=np.inf)) if np.any(finite) else None
    reported = float(cert.min_jd_lower_bound)
    if lowest is None or lowest.hex() != reported.hex():
        raise CertificateContractError(
            f"jd_cell_estimate '{name}': min of the per-cell bounds {None if lowest is None else lowest.hex()} "
            f"differs from the certificate's min_jd_lower_bound {reported.hex()}")
    if not cached:
        _CELL_BOUNDS_CACHE.clear()
        _CELL_BOUNDS_CACHE.update(key=key, bounds=bounds, code=code)
    record = dict(min_cert_bound=lowest, cert_state_hash=digest, cert_min_matches=True,
                  cell_bounds_cached=bool(cached), cell_bounds_seconds=time.perf_counter() - tick)
    return bounds, code, record, key, u


def certificate_cell_bounds(u, grid, cert, cert_config):
    """(b (grid.cell_shape, read-only; NaN where the per-cell path gives no bound), record) for the accepted,
    certified state u whose CertOutcome `cert` came from the C07-2 certify_map call with `cert_config`.

    b_K: the lower bound certify_map's per-cell path gets for cell K (bayesreg.eval.cert_eval.cell_pass, the same
    decisions and values as jd_curl feasibility._certify_map_parallel at eb808be; never a new certify_map call).  Raises
    InputValidationError without a passed outcome or with another configuration, backend.CertificateContractError when
    cert.certificate_version is not that of cert_config, sha256(u) differs from cert.state_hash, a cell is not proven,
    or min_K b_K differs from cert.min_jd_lower_bound in any bit (implementation error, C07-3)."""
    bounds, _, record, _, _ = _certificate_cells(u, grid, cert, cert_config)
    return bounds, record


# ---- jd_cell_estimate "certificate_bernstein" (JDTI; C09.1-3a, D53; path only, not the default) ---------------------

def _prescreen_on(u, grid, cells):
    """l on `cells` (flat ids, increasing): jd_curl's prescreen_lower_bounds with the closed27 coefficient table of the
    grid (the certify_map prescreen function and table, read-only); all NaN when building the table raises
    FloatingPointError (as certify_map / cell_pass); a non-finite or subnormal table gives all NaN inside jd_curl."""
    if not len(cells):
        return np.empty(0)
    try:
        table = CellQ1Operators3D(grid, location="closed27_uniform").coefficients
    except FloatingPointError:
        return np.full(len(cells), np.nan)
    return np.asarray(prescreen_lower_bounds(u, grid, table, cells), dtype=np.float64)


def bernstein_cell_bounds(u, grid, cert, cert_config, activation=J_ACT, *, j53=None, slab_cells=SLAB_CELLS):
    """C09.1-3a certificate_bernstein: (b~ (flat, n_cells), b (grid.cell_shape, read-only), record) at the accepted,
    certified state u with its CertOutcome `cert` (C07-2 configuration `cert_config`).

    b: the per-cell certified bounds of `certificate_cell_bounds` with all its checks (check (a): violations raise
    CertificateContractError).  S = {K: cell_pass code PROVEN_HULL, b_K < activation}; l_S = jd_curl prescreen on S;
    a cell K of S is raised when l_K is finite and l_K > b_K; check (b): a raised cell with l_K > J53_K (min of J at
    its 53 points, `j53` (flat) or from closed27_cell_estimate) is not raised (b~_K = b_K), counted and listed; no
    tolerance.  b~ = b except b~_K = l_K on the raised cells.  The (S, l_S) entry is cached by (certificate key,
    "certificate_bernstein", activation.hex()); check (b) runs on every call (J53 recomputed by the caller)."""
    from ..eval.cert_eval import PROVEN_HULL
    activation = finite_real(activation, "jd activation", positive=True)
    bounds, code, record, key, u = _certificate_cells(u, grid, cert, cert_config, CERTIFICATE_BERNSTEIN)
    tick = time.perf_counter()
    b = bounds.ravel()
    bkey = (key, CERTIFICATE_BERNSTEIN, float(activation).hex())
    cached = _BERNSTEIN_CACHE.get("key") == bkey
    if cached:
        S, ell = _BERNSTEIN_CACHE["S"], _BERNSTEIN_CACHE["ell"]
    else:
        with np.errstate(invalid="ignore"):
            S = np.flatnonzero((code == PROVEN_HULL) & (b < activation)).astype(np.int64)
        ell = _prescreen_on(u, grid, S)
        if ell.shape != S.shape:
            raise InputValidationError("certificate_bernstein: the prescreen returned a wrong number of bounds")
        S.flags.writeable = False
        ell.flags.writeable = False
        _BERNSTEIN_CACHE.clear()
        _BERNSTEIN_CACHE.update(key=bkey, S=S, ell=ell)
    seconds = time.perf_counter() - tick
    if j53 is None:
        j53 = closed27_cell_estimate(u, grid, slab_cells=slab_cells)[0].ravel()
    bS, jS = b[S], np.asarray(j53).ravel()[S]
    with np.errstate(invalid="ignore"):
        raised = np.isfinite(ell) & (ell > bS)
        violation = raised & (ell > jS)                       # check (b): no tolerance
    ok = raised & ~violation
    bt = b.copy()
    bt[S[ok]] = ell[ok]
    finite = np.isfinite(bt)
    n_violations = int(np.count_nonzero(violation))
    record = dict(record, n_cells_hull=int(np.count_nonzero(code == PROVEN_HULL)), n_cells_bernstein=int(S.size),
                  n_cells_raised=int(np.count_nonzero(ok)),
                  min_bernstein_bound=float(np.min(bt, where=finite, initial=np.inf)) if np.any(finite) else None,
                  bernstein_cached=bool(cached), bernstein_seconds=seconds, bernstein_activation=float(activation),
                  prescreen_kernel=PRESCREEN_KERNEL,
                  bernstein_j53_min_margin=float(np.min(jS[ok] - ell[ok])) if np.any(ok) else None,
                  n_bernstein_j53_violations=n_violations,
                  bernstein_j53_max_excess=float(np.max(ell[violation] - jS[violation])) if n_violations else None,
                  bernstein_j53_violation_cells=[int(k) for k in S[violation][:J53_VIOLATION_CELLS]])
    return bt, bounds, record


J53_WARNING = "certificate_bernstein_j53_violation"


def j53_warning_row(record):
    """The log.jsonl warning row fields (kind "warning") of an S~ record whose check (b) found violations
    (C09.1-3a certificate_bernstein), else None."""
    count = record.get("n_bernstein_j53_violations") if isinstance(record, dict) else None
    if not count:
        return None
    return dict(kind="warning", warning=J53_WARNING, contract="C09.1-3a certificate_bernstein check (b)",
                message=(f"{count} raised cell(s) with the prescreen bound l_K above J at its 53 points: those cells "
                         "fell back to b_K; the run's audit recertifies its final state with a configuration "
                         "without the prescreen, and its numbers enter no conclusion before that audit"),
                at=record.get("at"), n_bernstein_j53_violations=count,
                bernstein_j53_max_excess=record.get("bernstein_j53_max_excess"),
                bernstein_j53_violation_cells=record.get("bernstein_j53_violation_cells"),
                cert_state_hash=record.get("cert_state_hash"))


def j53_violations_logged(rows):
    """The run total of check (b) violations from log rows (the "warning" rows of `j53_warning_row`): for a run that an
    exception ended outside every round, where no round record exists (run_pair's status "exception"; JDTI review
    minor 1)."""
    return sum(int(row.get("n_bernstein_j53_violations") or 0) for row in rows
               if row.get("kind") == "warning" and row.get("warning") == J53_WARNING)


def j53_violations_total(rounds):
    """The run total of check (b) violations (C09.1-3a certificate_bernstein, result `bernstein_j53_violations_total`):
    the sum of n_bernstein_j53_violations over every S~ computation in the round records -- `metric.jd_scaling` of a
    finished round, `jd_scaling` of a round an exception ended (multilevel's exception row, JDTI review minor 1)."""
    total = 0
    for row in rounds or ():
        scaling = (row.get("metric") or {}).get("jd_scaling") or row.get("jd_scaling") or {}
        for computation in scaling.get("computations") or ():
            total += int(computation.get("n_bernstein_j53_violations") or 0)
    return total


def _bernstein_estimate(u, grid, cert, cert_config, activation, slab_cells):
    """(m'' (grid.cell_shape), base m_K, m' of "certificate" (grid.cell_shape), extra record fields) of
    certificate_bernstein: one closed27 slab pass gives J53 and the base m_K (bitwise cell_jd_estimate)."""
    j53, default = closed27_cell_estimate(u, grid, slab_cells=slab_cells)
    bt, bounds, record = bernstein_cell_bounds(u, grid, cert, cert_config, activation, j53=j53.ravel())
    del j53
    bt = bt.reshape(grid.cell_shape)
    has = np.isfinite(bounds)                     # S cells have finite bounds, so b~ is finite exactly where b is
    lowered_cert = has & (bounds < default)
    m_cert = np.where(lowered_cert, bounds, default)
    lowered = has & (bt < default)
    m = np.where(lowered, bt, default)
    extra = dict(cell_estimate=CERTIFICATE_BERNSTEIN, n_cells_lowered=int(np.count_nonzero(lowered)),
                 n_cells_without_bound=int(np.count_nonzero(~has)), **record)
    return m, default, m_cert, extra


def cell_estimate(u, grid, estimate=GAUSS_CORNERS, *, slab_cells=SLAB_CELLS, cert=None, cert_config=None,
                  activation=J_ACT):
    """Default `estimate`: the base estimate "gauss27+corners", NOT the method default "certificate" (which needs cert
    and cert_config; JDScalingConfig gives it, MIN1 / review DEF minor 2).

    Returns (m per cell (grid.cell_shape), the base m_K of gauss27+corners -- the "default" of the record names
    n_r_lt_1_default, min_m_K_default -- or None when `estimate` is the base itself, extra record fields) for a
    jd_cell_estimate option.  `activation` (J_act) is read by "certificate_bernstein" only (S depends on it)."""
    if estimate == CERTIFICATE_BERNSTEIN:
        m, default, _, extra = _bernstein_estimate(u, grid, cert, cert_config, activation, slab_cells)
        return m, default, extra
    if estimate == GAUSS_CORNERS:
        return cell_jd_estimate(u, grid, slab_cells=slab_cells), None, {}
    if estimate == CLOSED27:
        m, default = closed27_cell_estimate(u, grid, slab_cells=slab_cells)
        return m, default, dict(cell_estimate=CLOSED27, n_cells_lowered=int(np.count_nonzero(m < default)),
                                min_m_K_default=float(default.min()))
    if estimate == CERTIFICATE:
        default = cell_jd_estimate(u, grid, slab_cells=slab_cells)
        bounds, record = certificate_cell_bounds(u, grid, cert, cert_config)
        has = np.isfinite(bounds)
        lowered = has & (bounds < default)
        m = np.where(lowered, bounds, default)
        return m, default, dict(cell_estimate=CERTIFICATE, n_cells_lowered=int(np.count_nonzero(lowered)),
                                n_cells_without_bound=int(np.count_nonzero(~has)), **record)
    raise InputValidationError(f"jd_cell_estimate: one of {CELL_ESTIMATES}")


def _factors(J, activation, floor):
    return np.minimum(1., np.maximum(J, floor) / activation)


def scaling_factors(u, grid, *, activation=J_ACT, floor=J_FLOOR, slab_cells=SLAB_CELLS, estimate=GAUSS_CORNERS,
                    cert=None, cert_config=None):
    """Default `estimate`: the base estimate "gauss27+corners", NOT the method default "certificate" (which needs cert
    and cert_config; JDScalingConfig gives it, MIN1 / review DEF minor 2).

    Returns (r on the interior nodes, record): r_a = min(1, max(J_a, floor)/activation).
    record: min_J_a, fraction_r_lt_1, n_r_lt_1, n_nodes, min_r, seconds (the base estimate, unchanged); another
    `estimate` (jd_cell_estimate) adds cell_estimate, n_nodes_r_lowered (nodes whose r_a is below the base
    estimate's), n_r_lt_1_default (of the base estimate), and -- closed27: n_cells_lowered (cells whose m_K the
    closed27 points lowered), min_m_K_default; certificate: n_cells_lowered (cells with b_K < m_K: nearly all, a
    valid lower bound is below the sampled values), n_cells_without_bound, min_cert_bound, cert_state_hash,
    cert_min_matches, cell_bounds_cached, cell_bounds_seconds.  `cert` (the CertOutcome of u) and `cert_config`
    (C07-2) are read by "certificate" and "certificate_bernstein" only.

    certificate_bernstein (JDTI; C09.1-3a "记录"): the certificate fields (n_cells_lowered here: cells with b~_K < m_K)
    and n_cells_hull, n_cells_bernstein (|S|), n_cells_raised, min_bernstein_bound (min b~_K), bernstein_cached,
    bernstein_seconds, bernstein_activation, prescreen_kernel, bernstein_j53_min_margin (min J53_K - b~_K over the
    raised cells; None without one), n_bernstein_j53_violations, bernstein_j53_max_excess (None without a violation),
    bernstein_j53_violation_cells (at most 10 flat cell ids); nodes against certificate: n_r_lt_1_certificate,
    n_nodes_r_lowered_certificate, n_nodes_r_raised_vs_certificate; frozen nodes (r_a < .01): n_r_lt_0p01,
    n_r_lt_0p01_certificate, n_r_lt_0p01_default."""
    tick = time.perf_counter()
    m_cert = None
    if estimate == CERTIFICATE_BERNSTEIN:
        m, default, m_cert, extra = _bernstein_estimate(u, grid, cert, cert_config, activation, slab_cells)
    else:
        m, default, extra = cell_estimate(u, grid, estimate, slab_cells=slab_cells, cert=cert, cert_config=cert_config)
    J = node_minimum(m, grid)
    r = _factors(J, activation, floor)
    if not (np.all(np.isfinite(r)) and np.all(r > 0)):
        raise FloatingPointError("JD scaling: factors must be finite and positive")
    if default is not None:
        r_default = _factors(node_minimum(default, grid), activation, floor)
        extra = dict(extra, n_nodes_r_lowered=int(np.count_nonzero(r < r_default)),
                     n_r_lt_1_default=int(np.count_nonzero(r_default < 1)))
    if m_cert is not None:
        r_cert = _factors(node_minimum(m_cert, grid), activation, floor)
        del m_cert
        extra = dict(extra, n_r_lt_1_certificate=int(np.count_nonzero(r_cert < 1)),
                     n_nodes_r_lowered_certificate=int(np.count_nonzero(r_cert < r_default)),
                     n_nodes_r_raised_vs_certificate=int(np.count_nonzero(r > r_cert)),
                     n_r_lt_0p01=int(np.count_nonzero(r < .01)),
                     n_r_lt_0p01_certificate=int(np.count_nonzero(r_cert < .01)),
                     n_r_lt_0p01_default=int(np.count_nonzero(r_default < .01)))
    scaled = int(np.count_nonzero(r < 1))
    record = dict(min_J_a=float(J.min()), fraction_r_lt_1=scaled / r.size, n_r_lt_1=scaled, n_nodes=int(r.size),
                  min_r=float(r.min()), seconds=time.perf_counter() - tick)
    record.update(extra)
    return r, record


def free_scaling(r):
    """S~ as a free vector (3 N_int,): component-major, the same factor for the three components."""
    out = np.tile(np.asarray(r, dtype=np.float64).ravel(order="C"), 3)
    out.flags.writeable = False
    return out


# ---- the metric ---------------------------------------------------------------------------------------

class JDScaledMetric:
    """S~ M^-1 S~ for the PG direction (C09.1-3a) around a frozen metric M (`solve`, `matvec`).

    solve(g) = S~ M^-1 (S~ g); matvec(v) = S~^-1 M S~^-1 v (its inverse).  `observe(reason)` takes every candidate's
    rejection reason (None = accepted) for the refresh rule; `refresh_due` tells run_pg to call `refresh(u, ...)` at
    the accepted state.  `computations` lists the record of every computation of S~ (`at`: freeze / refresh).
    jd_cell_estimate "certificate" and "certificate_bernstein" (config.cell_estimate): `cert` is the CertOutcome of u
    at the freeze and of the accepted state at every refresh (run_pg passes the base's), `cert_config` the C07-2
    configuration; the other estimates ignore both."""
    mode = AFFINE_V1

    def __init__(self, metric, grid, u, *, config=JDScalingConfig(), at="freeze", slab_cells=SLAB_CELLS, cert=None,
                 cert_config=None, **context):
        if not isinstance(config, JDScalingConfig):
            raise InputValidationError("JDScaledMetric: a JDScalingConfig required")
        if config.cell_estimate in CERTIFICATE_ESTIMATES and cert_config is None:
            raise InputValidationError(f"jd_cell_estimate '{config.cell_estimate}': the C07-2 certificate "
                                       "configuration is required")
        self.base, self.grid, self.config, self.slab_cells = metric, grid, config, slab_cells
        self._cert_config, self._cert = cert_config, cert
        self.computations, self.refreshes = [], 0
        self._consecutive, self._window, self._due = 0, deque(maxlen=config.refresh_window), None
        self._compute(u, at=at, **context)

    def _compute(self, u, **context):
        r, record = scaling_factors(u, self.grid, activation=self.config.activation, floor=self.config.floor,
                                    slab_cells=self.slab_cells, estimate=self.config.cell_estimate, cert=self._cert,
                                    cert_config=self._cert_config)
        self.r = r
        self.scale = free_scaling(r)
        record = dict(context, **record)
        self.computations.append(record)
        self._consecutive, self._due = 0, None
        self._window.clear()
        return record

    # ---- metric protocol (the base metric's attributes for the records) ----
    @property
    def a(self):
        return self.base.a

    @property
    def shift(self):
        return self.base.shift

    def solve(self, g):
        with np.errstate(over="raise", invalid="raise"):
            return self.scale * self.base.solve(self.scale * g)

    def matvec(self, v):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return self.base.matvec(v / self.scale) / self.scale

    # ---- refresh rule ----
    def observe(self, reason):
        """One candidate's rejection reason (None when accepted)."""
        self._consecutive = 0 if reason is None else self._consecutive + 1
        self._window.append(reason == JACOBIAN_VIOLATION)
        if self._due is None:
            share = sum(self._window)
            if self._consecutive >= self.config.refresh_consecutive:
                self._due = f"{self._consecutive} consecutive rejected candidates"
            elif 2 * share > self.config.refresh_window:
                self._due = (f"{share} of the most recent {len(self._window)} candidates rejected "
                             f"{JACOBIAN_VIOLATION}")

    @property
    def refresh_due(self):
        return self._due

    def refresh(self, u, *, cert=None, **context):
        """Recompute S~ at the accepted state u (M unchanged; `cert` its CertOutcome, read by the "certificate" and
        "certificate_bernstein" estimates only); returns the record."""
        trigger = self._due
        self.refreshes += 1
        self._cert = cert
        return self._compute(u, at="refresh", trigger=trigger, refresh=self.refreshes, **context)

    def record(self):
        return dict(self.config.record(), jd_scaling_refresh=self.refreshes, computations=list(self.computations))


def scaled_metric(metric, grid, u, mode, config=JDScalingConfig(), *, cert=None, cert_config=None, **context):
    """The PG metric for `mode`: M itself ("off": bitwise the C09.1 path) or JDScaledMetric ("affine_v1"); `cert`
    (the CertOutcome of u) and `cert_config` (C07-2) serve jd_cell_estimate "certificate" and
    "certificate_bernstein"."""
    if mode == OFF:
        return metric
    if mode == AFFINE_V1:
        return JDScaledMetric(metric, grid, u, config=config, cert=cert, cert_config=cert_config, **context)
    raise InputValidationError(f"metric_jd_scaling: one of {MODES}")
