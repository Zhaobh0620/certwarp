"""The four residual blocks of F_B = 1/2 ||r||^2 and their Jacobians (contract C04, C05, C06-2/3/6; B2).

Channel block (node layout).  r_{c,a} = beta_{c,a} rho_c(u)_a, beta = sqrt(q_a m_a w_c(a)/(omega0
sigma_c^2)), rho_c(u)_a = G_c(x_a + u_a) - g_c(a), c = 0..C-1 stacked on the first axis.  Only the
mask nodes (state.node_index, C order) carry nonzero rows, so every channel array is (C, n) on them;
`channel_expand` gives the (C, nz, ny, nx) layout of the contract (zero rows outside the mask).
(K_c v)_a = beta_{c,a} grad G_c(x_a + u_a) . v_a; K_c^T r is node-wise (no scatter).  Out of the
sampling domain the block returns the InvalidTrial of `check_map_domain` / `sample_points`
(INVALID_DOMAIN; no extrapolation).  The spline sampling of the C channels runs in state.cpu_threads
threads by whole sample_points chunks (bitwise the serial loop; `_sample_channels`).

Gauss-point blocks (Diffusion, LogVolume, Curl).  Per-tile kernels on the geometry of
`bayesreg.geom.tile_geometry` (A, J, curl, grad of u at the Gauss27 points of one tile):
  scale(state, tile)          the per-point square-root weight,
  residual(scale, geo)        r_i on the tile,
  jvp(scale, geo, dJ, dcurl, V)  K_i v on the tile from tile_jvp(v, A=geo.A),
  cotangent(scale, geo, r)    (gJ, gcurl, ggrad) with K_i^T r = tile_vjp_accumulate(...) + finish.
    Diffusion  r_D = sqrt(alpha omega w_g/omega0) grad u_h(x_q)        (cells, 27, 3, 3), rows (q, r, j)
    LogVolume  r_J = sqrt(omega w_g/(omega0 tau_q^2)) log J_q           (cells, 27)
    Curl       r_C = sqrt(nu omega w_g/omega0) c_q                      (cells, 27, 3)
LogVolume refuses J_q <= 0 (no large number, C05-2/C07-5): its kernels raise; `gauss_value` returns
InvalidTrial NONPOSITIVE_GAUSS_JD.  `gauss_value`, `gauss_jvp`, `gauss_vjp` serve one tile, or all
cells of a small grid when tile=None (above geom.MAX_WHOLE_CELLS they raise: native volumes stream,
see energy.py).

Cell corners (C05-2a, logvol_points = "gauss27+corners"; work package LVC).  LogVolumeCorners is the second part of
the same block r_J: on the geometry of the corner operators state.corner_ops (`geom.corners8`, the same tile
kernels), r_{J,K,v} = sqrt(beta_c omega/(8 omega0 tau_{K,v}^2)) log J_{K,v}, (cells, 8) with the corners in the
order of cell_q1.VERTICES; J_{K,v} is the J of cell K's own Q1 map at v and tau_{K,v}^2 the node value of
state.tau2_nodes at v (gathered, exactly the node value).  Residual, jvp and cotangent are those of LogVolume.
Every block names its point set by `operators(state)`; the conveniences below accept LogVolumeCorners too.  A
floating J <= 0 at a corner is the failure of C07-5 (iii) as at a Gauss point (reason NONPOSITIVE_GAUSS_JD, the
message names the corners; `nonpositive_logvol_jd`).

Closed27 points (C05-2b, logvol_points = "gauss27+closed27"; work package IMPL27).  The block r_J = [r_{J,G};
r_{J,S}]: LogVolume (Gauss27) with the weights state.logvol_gauss_weights = omega w_g theta_G, and LogVolumeClosed27
on the closed27 operators state.closed27_ops (`geom.closed27`, the same tile kernels):
r_{J,S,K,p} = sqrt(omega s_p theta_S/(omega0 tau_{K,p}^2)) log J_{K,p}, (cells, 27) in the point order of
closed27_ops.points; J_{K,p} of cell K's own Q1 map at p; tau_{K,p}^2 the Q1 interpolation of state.tau2_nodes at p
(tile_interpolate with the closed27 basis: at the corners the basis rows are one-hot, so the node values exactly).
Residual, jvp and cotangent are those of LogVolume; `point_weights(state)` gives the omega-scaled point weights of a
second part (C08-2 weights of `energy.stable_reduction`).  A floating J <= 0 at a closed27 point is the failure of
C07-5 (iii) (reason NONPOSITIVE_GAUSS_JD, the message names closed27; count Gauss + closed27 points).
`logvol_extra_block(state)` is LogVolumeCorners, LogVolumeClosed27 or None.  With logvol_points = "gauss27" LogVolume
reads state.logvol_gauss_weights, which is state.ops.weights itself: every value bitwise that of the code before.
"""
from contextlib import closing
from dataclasses import dataclass

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, as_float64_array, readonly_copy, require_finite_result
from jd_curl.three_d.cell_q1 import cell_node_indices
from jd_curl.three_d.interpolation import check_map_domain, sample_points

from ..readonly import freeze_new
from ..geom import (MAX_WHOLE_CELLS, finish, node_field, ordered_map, tile_geometry, tile_interpolate, tile_jvp,
                    tile_vjp_accumulate)

_RAISE = dict(over="raise", invalid="raise", divide="raise")
NONPOSITIVE_GAUSS_JD = "NONPOSITIVE_GAUSS_JD"
SPLINE_CHUNK = 65536              # = the default chunk_size of jd_curl sample_points (checked by the CPUPAR tests)


# ----------------------------------------------------------------------------------------------
# Channel block (nodes)
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelPoint:
    """The channel block at one u, on the mask nodes (state.node_index), channels first; read-only."""
    rho: np.ndarray      # (C, n) G_c(x_a + u_a) - g_c(a)
    grad: np.ndarray     # (C, n, 3) physical xyz gradient of G_c at x_a + u_a
    r: np.ndarray        # (C, n) beta * rho


def channel_point(state, u):
    """C04-1 at u (free vector or U0 node field): ChannelPoint, or the InvalidTrial of the sampling
    domain check (all nodes: interior strictly inside, boundary in the closed box)."""
    u = node_field(u, state.grid)
    invalid = check_map_domain(u, state.grid)
    if invalid is not None:
        return invalid
    with np.errstate(**_RAISE):
        points = state.points0+u.reshape(-1, 3)[state.node_index]
    sampled = _sample_channels(state, points)
    if isinstance(sampled, InvalidTrial):
        return sampled
    values, grads = sampled
    del sampled, points
    with np.errstate(**_RAISE):
        rho = values-state.fixed_masked
        r = state.beta*rho
    del values
    require_finite_result(r, "channel residual")
    # MEM (memory only): rho, grads and r are new arrays of this call -- read-only in place instead of copied (the
    # second copy of rho, r (C, n) and grads (C, n, 3) is 1.2 GB at the native size with 14 channels)
    return ChannelPoint(freeze_new(rho), freeze_new(grads), freeze_new(r))


def _sample_serial(state, points):
    values = np.empty(state.fixed_masked.shape)
    grads = np.empty(values.shape+(3,))
    for c, sampler in enumerate(state.samplers):
        sample = sample_points(sampler, points, strict=state.strict)
        if isinstance(sample, InvalidTrial):
            return sample
        values[c], grads[c] = sample.value, sample.gradient
    return values, grads


def _sample_channels(state, points):
    """(values (C, n), grads (C, n, 3)) of sample_points(sampler_c, points, strict=state.strict) for c = 0..C-1, or
    the first InvalidTrial in channel order (the serial loop `_sample_serial`).

    With state.cpu_threads > 1 the (channel, chunk) pieces run in threads: chunks of SPLINE_CHUNK points, the
    chunk size of sample_points, starting at multiples of it, so every NdBSpline call receives the very query
    array of the serial call and every value is bitwise the serial one.  A piece that returns an InvalidTrial
    hands the whole decision back to the serial loop (its count and reason refer to all points)."""
    n = points.shape[0]
    if state.cpu_threads == 1:
        return _sample_serial(state, points)
    items = [(c, lo, min(lo+SPLINE_CHUNK, n)) for c in range(state.n_channels) for lo in range(0, n, SPLINE_CHUNK)]

    def piece(item):
        c, lo, hi = item
        return sample_points(state.samplers[c], points[lo:hi], strict=state.strict[lo:hi])

    values = np.empty(state.fixed_masked.shape)
    grads = np.empty(values.shape+(3,))
    with closing(ordered_map(piece, items, state.cpu_threads)) as results:
        for (c, lo, hi), sample in zip(items, results):
            if isinstance(sample, InvalidTrial):
                break
            values[c, lo:hi], grads[c, lo:hi] = sample.value, sample.gradient
        else:
            return values, grads
    return _sample_serial(state, points)


def channel_jvp(state, point, v):
    """(K_c v)_a = beta_{c,a} grad G_c . v_a on the mask nodes, (C, n); v a free vector or U0 field."""
    v = node_field(v, state.grid, name="v").reshape(-1, 3)[state.node_index]
    g = point.grad
    with np.errstate(**_RAISE):
        out = state.beta*(g[..., 0]*v[:, 0]+g[..., 1]*v[:, 1]+g[..., 2]*v[:, 2])
    require_finite_result(out, "channel JVP")
    return out


def _channel_node_add(state, coefficient, grad, out):
    """out[:, a] += sum_c coefficient[c, a] grad[c, a, :] at the mask nodes (channels in ascending order)."""
    with np.errstate(**_RAISE):
        total = coefficient[0][:, None]*grad[0]
        for c in range(1, coefficient.shape[0]):
            total = total+coefficient[c][:, None]*grad[c]
    flat = out.reshape(3, -1)
    flat[:, state.node_index] += total.T
    return out


def channel_vjp(state, point, r):
    """K_c^T r summed over channels: sum_c beta_c r_c grad G_c at the mask nodes, as the free vector."""
    r = as_float64_array(r, "r", point.r.shape)
    out = np.zeros((3,)+state.grid.shape)
    with np.errstate(**_RAISE):
        coefficient = state.beta*r
    return finish(_channel_node_add(state, coefficient, point.grad, out))


def channel_gn_diagonal(state, point):
    """C06-6: diag(sum_c K_c^T K_c)_{a,d} = sum_c beta_{c,a}^2 (d_d G_c)^2, as the free vector (zero
    outside the mask)."""
    out = np.zeros((3,)+state.grid.shape)
    with np.errstate(**_RAISE):
        total = (state.beta[0]**2)[:, None]*point.grad[0]**2
        for c in range(1, state.n_channels):
            total = total+(state.beta[c]**2)[:, None]*point.grad[c]**2
    out.reshape(3, -1)[:, state.node_index] = total.T
    return finish(out)


def channel_cdat_weight(state):
    """The (C, n) c_dat weight w_c(a)/sigma_c^2 on the mask nodes (a new array); under sigma_domain support /
    support_shared (C04-6, C09.1 appended note) rows c >= 1 times the frozen support weight v_c(a), in place (exact: v in
    {0, 1}).  Brain: exactly state.weights_masked/state.sigma2[:, None], as before."""
    with np.errstate(**_RAISE):
        weight = state.weights_masked/state.sigma2[:, None]
        if state.support_v is not None:
            weight[1:] *= state.support_v
    return weight


def channel_c_dat(state, point):
    """C09.1: c_dat = (1/(3 N_B)) sum_{a in mask} sum_d sum_c (w_c(a)/sigma_c^2) (d_d G_c(x_a + u_a))^2,
    the per-node mean of the channel GN diagonal without q_a/omega0 (longdouble sum, float result).  Under sigma_domain
    support / support_shared the summand is multiplied by v_c(a); the normalization 1/(3 N_B) stays the mask node count
    (C09.1 appended note)."""
    with np.errstate(**_RAISE):
        terms = channel_cdat_weight(state)[..., None]*point.grad**2
        total = np.sum(terms, dtype=np.longdouble)/np.longdouble(3*state.node_index.size)
    result = float(total)
    require_finite_result(result, "c_dat")
    return result


def channel_expand(state, values):
    """(C, n) values on the mask nodes -> (C, nz, ny, nx), exactly zero outside the mask."""
    values = as_float64_array(values, "values", state.fixed_masked.shape)
    out = np.zeros((values.shape[0], int(np.prod(state.grid.shape))))
    out[:, state.node_index] = values
    return out.reshape((values.shape[0],)+state.grid.shape)


# ----------------------------------------------------------------------------------------------
# Gauss-point blocks: per-tile kernels
# ----------------------------------------------------------------------------------------------

def _positive_jacobian(geo):
    if not np.all(geo.J > 0):
        raise InputValidationError("log-volume block: J_q <= 0 at a Gauss point (C05-2: no value, no derivative)")


class Diffusion:
    """C05-1, C06-2/3: r_D = sqrt(alpha omega w_g/omega0) grad u_h(x_q), K_D = sqrt(alpha/omega0) W^1/2 grad_h."""
    name = "diffusion"

    @staticmethod
    def operators(state):
        return state.ops

    @staticmethod
    def scale(state, tile):
        with np.errstate(**_RAISE):
            return np.sqrt(state.alpha*state.ops.weights/state.omega0)[None, :, None, None]

    @staticmethod
    def residual(scale, geo):
        return scale*geo.grad

    @staticmethod
    def jvp(scale, geo, dJ, dcurl, V):
        return scale*V

    @staticmethod
    def cotangent(scale, geo, r):
        return None, None, scale*r


class LogVolume:
    """C05-2, C06-2/3: r_J = sqrt(omega w_g/(omega0 tau_q^2)) log J_q, (K_J v)_q = scale/J_q A_q : V_q."""
    name = "logvol"

    @staticmethod
    def operators(state):
        return state.ops

    @staticmethod
    def tau2(state, tile):
        """tau_q^2 on the tile: Q1 interpolation of state.tau2_nodes (= sum_k tau_k^2 Ptilde_T^k(x_q))."""
        return tile_interpolate(state.tau2_nodes, state.ops, tile=tile)

    @staticmethod
    def point_weights(state):
        """omega w_g (C05-2; the same array as state.ops.weights), times theta_G under C05-2b."""
        return state.logvol_gauss_weights

    @staticmethod
    def scale(state, tile):
        tau2 = LogVolume.tau2(state, tile)
        with np.errstate(**_RAISE):
            return np.sqrt(state.logvol_gauss_weights/(state.omega0*tau2))

    @staticmethod
    def residual(scale, geo):
        _positive_jacobian(geo)
        with np.errstate(**_RAISE):
            return scale*np.log(geo.J)

    @staticmethod
    def jvp(scale, geo, dJ, dcurl, V):
        _positive_jacobian(geo)
        with np.errstate(**_RAISE):
            return scale*dJ/geo.J

    @staticmethod
    def cotangent(scale, geo, r):
        _positive_jacobian(geo)
        with np.errstate(**_RAISE):
            return scale*r/geo.J, None, None


class Curl:
    """C05-3, C06-2/3: r_C = sqrt(nu omega w_g/omega0) c_q, linear."""
    name = "curl"

    @staticmethod
    def operators(state):
        return state.ops

    @staticmethod
    def scale(state, tile):
        with np.errstate(**_RAISE):
            return np.sqrt(state.nu*state.ops.weights/state.omega0)[None, :, None]

    @staticmethod
    def residual(scale, geo):
        return scale*geo.curl

    @staticmethod
    def jvp(scale, geo, dJ, dcurl, V):
        return scale*dcurl

    @staticmethod
    def cotangent(scale, geo, r):
        return None, scale*r, None


class LogVolumeCorners(LogVolume):
    """C05-2a: the corner part of r_J, r_{J,K,v} = sqrt(beta_c omega/(8 omega0 tau_{K,v}^2)) log J_{K,v}, (cells, 8);
    (K_J v)_{K,v} = scale/J_{K,v} A_{K,v} : V_{K,v}; residual, jvp and cotangent inherited from LogVolume.  Only for a
    state with logvol_points = "gauss27+corners"."""
    name = "logvol"

    @staticmethod
    def operators(state):
        if state.corner_ops is None:
            raise InputValidationError("LogVolumeCorners: logvol_points = 'gauss27+corners' required (C05-2a)")
        return state.corner_ops

    @staticmethod
    def tau2(state, tile):
        """tau_{K,v}^2 on the tile: state.tau2_nodes at the eight vertices of every cell, (cells, 8) in VERTICES order
        (gathered: the node values exactly, = the Q1 interpolation at the corners)."""
        return state.tau2_nodes.ravel()[cell_node_indices(state.grid, tile)]

    @staticmethod
    def point_weights(state):
        """beta_c omega/8 per corner (C05-2a)."""
        LogVolumeCorners.operators(state)
        return state.corner_weights

    @staticmethod
    def scale(state, tile):
        LogVolumeCorners.operators(state)
        tau2 = LogVolumeCorners.tau2(state, tile)
        with np.errstate(**_RAISE):
            return np.sqrt(state.corner_weights/(state.omega0*tau2))


class LogVolumeClosed27(LogVolume):
    """C05-2b: the closed27 part of r_J, r_{J,S,K,p} = sqrt(omega s_p theta_S/(omega0 tau_{K,p}^2)) log J_{K,p},
    (cells, 27); (K_{J,S} v)_{K,p} = scale/J_{K,p} A_{K,p} : V_{K,p}; residual, jvp and cotangent inherited from
    LogVolume.  Only for a state with logvol_points = "gauss27+closed27"."""
    name = "logvol"

    @staticmethod
    def operators(state):
        if state.closed27_ops is None:
            raise InputValidationError("LogVolumeClosed27: logvol_points = 'gauss27+closed27' required (C05-2b)")
        return state.closed27_ops

    @staticmethod
    def tau2(state, tile):
        """tau_{K,p}^2 on the tile: the Q1 interpolation of state.tau2_nodes at the closed27 points, (cells, 27)
        (D48; at the corners the node values exactly, the basis rows being one-hot)."""
        return tile_interpolate(state.tau2_nodes, LogVolumeClosed27.operators(state), tile=tile)

    @staticmethod
    def point_weights(state):
        """omega s_p theta_S per closed27 point (C05-2b, D48; state.closed27_point_weights)."""
        LogVolumeClosed27.operators(state)
        return state.closed27_point_weights

    @staticmethod
    def scale(state, tile):
        tau2 = LogVolumeClosed27.tau2(state, tile)
        with np.errstate(**_RAISE):
            return np.sqrt(state.closed27_point_weights/(state.omega0*tau2))


GAUSS_BLOCKS = (Diffusion, LogVolume, Curl)
EXTRA_LOGVOL_BLOCKS = dict(corners=LogVolumeCorners, closed27=LogVolumeClosed27)


def logvol_extra_block(state):
    """The second part of the log-volume block of `state`: LogVolumeCorners (C05-2a), LogVolumeClosed27 (C05-2b) or
    None (C05-2, "gauss27")."""
    extra = state.logvol_extra
    return None if extra is None else EXTRA_LOGVOL_BLOCKS[extra]


def merge_cotangents(parts):
    """Sum (gJ, gcurl, ggrad) triples entrywise (None = zero) for one tile_vjp_accumulate."""
    merged = [None, None, None]
    with np.errstate(**_RAISE):
        for part in parts:
            for k, value in enumerate(part):
                if value is not None:
                    merged[k] = value if merged[k] is None else merged[k]+value
    return tuple(merged)


# ----------------------------------------------------------------------------------------------
# Gauss-point blocks: one tile / small-grid conveniences
# ----------------------------------------------------------------------------------------------

def _whole(state, tile):
    if tile is not None:
        return tile
    n = state.grid.n_cells
    if n > MAX_WHOLE_CELLS:
        raise InputValidationError(f"tile=None means all {n} cells, above MAX_WHOLE_CELLS = {MAX_WHOLE_CELLS}; "
                                   "pass a tile or use the streamed energy / gradient / hvp")
    return slice(0, n)


def nonpositive_gauss_jd(count):
    return InvalidTrial(NONPOSITIVE_GAUSS_JD, "floating-point J_q <= 0 at a Gauss point (C07-5 iii)", int(count))


def nonpositive_logvol_jd(gauss, corners=0, *, closed27=0):
    """C07-5 (iii) for the log-volume points of one tile: `nonpositive_gauss_jd(gauss)` when no corner or closed27 point
    fails; with C05-2a corners (C05-2b closed27 points) the same reason, the count Gauss points + corners (closed27
    points) and a message that names both."""
    if closed27:
        return InvalidTrial(NONPOSITIVE_GAUSS_JD, f"floating-point J <= 0 at {int(gauss)} Gauss point(s) and "
                            f"{int(closed27)} closed27 point(s) (C07-5 iii; closed27 part of C05-2b)",
                            int(gauss)+int(closed27))
    if not corners:
        return nonpositive_gauss_jd(gauss)
    return InvalidTrial(NONPOSITIVE_GAUSS_JD, f"floating-point J <= 0 at {int(gauss)} Gauss point(s) and "
                        f"{int(corners)} cell corner(s) (C07-5 iii; corner term of C05-2a)", int(gauss)+int(corners))


def nonpositive_extra_jd(state, gauss, extra):
    """`nonpositive_logvol_jd` for the second part of state's log-volume block (corners or closed27 points)."""
    if state.logvol_extra == "closed27":
        return nonpositive_logvol_jd(gauss, closed27=extra)
    return nonpositive_logvol_jd(gauss, extra)


def gauss_value(block, state, u, *, tile=None):
    """r_i of a Gauss-point block (or of LogVolumeCorners / LogVolumeClosed27) on one tile (all cells when tile=None);
    the log-volume blocks return InvalidTrial NONPOSITIVE_GAUSS_JD when some J <= 0 at their points."""
    tile = _whole(state, tile)
    ops = block.operators(state)
    geo = tile_geometry(node_field(u, state.grid), ops, tile=tile)
    if issubclass(block, LogVolume):
        count = int(np.count_nonzero(geo.J <= 0))
        if count:
            if block is LogVolume:
                return nonpositive_gauss_jd(count)
            if block is LogVolumeClosed27:
                return nonpositive_logvol_jd(0, closed27=count)
            return nonpositive_logvol_jd(0, count)
    return block.residual(block.scale(state, tile), geo)


def gauss_jvp(block, state, u, v, *, tile=None):
    """K_i v of a Gauss-point block (or of LogVolumeCorners) at u (both in U0) on one tile."""
    tile = _whole(state, tile)
    ops = block.operators(state)
    geo = tile_geometry(node_field(u, state.grid), ops, tile=tile)
    dJ, dcurl, V = tile_jvp(node_field(v, state.grid, name="v"), ops, tile=tile, A=geo.A)
    return block.jvp(block.scale(state, tile), geo, dJ, dcurl, V)


def gauss_vjp(block, state, u, r, *, tile=None):
    """K_i^T r of a Gauss-point block (or of LogVolumeCorners) at u on one tile, restricted to U0: the free vector."""
    tile = _whole(state, tile)
    ops = block.operators(state)
    geo = tile_geometry(node_field(u, state.grid), ops, tile=tile)
    gJ, gcurl, ggrad = block.cotangent(block.scale(state, tile), geo, np.asarray(r, dtype=np.float64))
    out = np.zeros((3,)+state.grid.shape)
    tile_vjp_accumulate(out, ops, tile=tile, gJ=gJ, gcurl=gcurl, ggrad=ggrad, A=geo.A)
    return finish(out)
