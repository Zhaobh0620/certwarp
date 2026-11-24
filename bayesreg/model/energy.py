"""F_B, its gradient, GN Hessian action, GN diagonal, metric and stable comparison (contract C06-C09.1).

Block order everywhere: channels c = 0..C-1, diffusion, logvol, curl.

- `evaluate(state, u)` (C06-1/2/7): F_B = 1/2 ||r||^2.  Node blocks: per channel np.sum(r_c^2,
  dtype=longdouble) over the C-order flattened (nz, ny, nx) array (zero rows outside the mask);
  Gauss-point blocks: per tile sum of r_i^2 in longdouble, tiles of state.tile_size cells in
  ascending order (cell_tiles); F_B (float64) = float(longdouble total).  One tile_geometry per tile
  serves the three Gauss-point blocks.  Failures are InvalidTrial (no energy value): INVALID_DOMAIN /
  NONFINITE_STATE of the sampling check, NONPOSITIVE_GAUSS_JD when a floating J_q <= 0 (C07-5 iii).
- `gradient(state, ev)` (C06-4): sum_i K_i^T r_i; diffusion as (alpha/omega0) K_Q u with
  K_Q = omega_ell apply_poisson_operator(metric="q1") (geom.stiffness_apply), logvol and curl in one
  scatter per tile, channels node-wise.
- `hvp(state, ev, v)` (C06-5): H v = sum_i K_i^T (K_i v), never assembled; K_D^T K_D = (alpha/omega0)
  K_Q (B2(b)) is applied through stiffness_apply; the geometry of u is recomputed per tile (A, J of a
  native volume do not fit in memory).
- `gn_diagonal`, `c_dat` (C06-6, C09.1): channel block only, as the contract defines them.
- `Metric` / `build_metric` / `metric_bP` (C09.1): M = (1/omega0)[omega_ell (c_dat + lam_b + gamma) I
  + b_P K_Q]; M^-1 = omega0 * build_poisson_inverse(grid, a = c_dat + lam_b + gamma, b = b_P,
  metric="q1").apply (DST-I).  In the methodology's form (b K_Q/omega0 + c I)^-1 (methodology 3.1
  "(alpha L + c I)^-1"): b = b_P (= alpha for mode "methodology") and c = omega_ell a/omega0
  (`Metric.shift`), i.e. the data shift is the per-node c_dat scaled to the level cell volume.
- `stable_reduction(state, base, trial)` (C08): ared = -sum_i Delta E_i, guard s_E = 16 eps64
  max(E_b, E_t), certified = ared - s_E; Delta E_i = 1/2 sum (t - b)(t + b) of the weighted residual
  values (longdouble elementwise and summed, the order of C06-7); the log-volume block by
  1/2 sum w_q l_q (log J_t + log J_b) with l_q = log1p((J_t - J_b)/J_b) (x >= -1/2) else
  log(J_t/J_b), w_q = omega w_g/(omega0 tau_q^2).  A resolution guard, not a rigorous error bound.

Cell corners (C05-2a, ModelState.logvol_points = "gauss27+corners"; work package LVC).  The log-volume block r_J
gets the corner part LogVolumeCorners ((cells, 8) per tile, on state.corner_ops); "logvol" in energies, gradient
and HVP selections and the stable comparison always means the whole block (Gauss points and corners).  Per tile:
the corner geometry follows the Gauss geometry; J <= 0 at a Gauss point or a corner fails the tile (C07-5 iii,
`nonpositive_logvol_jd`, count Gauss + corners); the tile's corner sum of r^2 (longdouble) is added to the
log-volume accumulator right after the tile's Gauss sum (C06-7 order: tiles ascending, in each tile Gauss points
then corners); the adjoint's corner local products are scattered right after the tile's Gauss-point products; in
the stable comparison the corner term 1/2 sum w_{K,v} l (log J_t + log J_b), w_{K,v} = beta_c omega/(8 omega0
tau_{K,v}^2), in the log1p form of C08-2 (longdouble), is added right after the tile's Gauss log-volume term.  With
"gauss27" (the default before DEF, an explicit option since) none of this runs and every value is bitwise that of
the code before C05-2a (481e33b).

Closed27 points (C05-2b, ModelState.logvol_points = "gauss27+closed27"; work package IMPL27).  The same machinery with
the second part LogVolumeClosed27 ((cells, 27) per tile, on state.closed27_ops) in the place of the corners (one code
path for both, `logvol_extra_block`): the Gauss part uses the weights omega w_g theta_G (LogVolume reads
state.logvol_gauss_weights), the closed27 part omega s_p theta_S; C06-7 order per tile Gauss sum then closed27 sum; the
adjoint scatters the closed27 products right after the tile's Gauss-point products; C08-2: the closed27 term in the
log1p form with w = omega s_p theta_S/(omega0 tau_{K,p}^2), right after the tile's Gauss log-volume term; a floating J
<= 0 at a closed27 point fails the tile (C07-5 iii, the message names closed27).  The contract's record: "logvol" in
energies is the whole block; evaluate adds energies["logvol_parts"] = dict(gauss, closed27 (each 1/2 its longdouble
sum of r^2, as floats), closed27_min_J, closed27_min_cell, closed27_min_point (the first minimum in tile, cell, point
order)) -- only under C05-2b (the energies of the other options are unchanged).

Threads (work package CPUPAR).  The per-tile work of evaluate, gradient, hvp and stable_reduction (geometry,
block values, the tile's longdouble sums, the adjoint's local products `tile_vjp_local`) is a function of the
tile alone (`_*_tile` below), run by `bayesreg.geom.ordered_map` in state.cpu_threads threads; the consumer adds
the tile sums in ascending tile order and scatters the local products in ascending tile order
(`tile_vjp_scatter`), so every value is bitwise that of the serial loop (cpu_threads = 1) for every thread
count.  evaluate still returns at the first tile (ascending) with a floating J_q <= 0, with that tile's count.
"""
from contextlib import closing
from dataclasses import dataclass
from functools import partial

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, as_float64_array, finite_real, require_finite_result
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.poisson import PoissonInverse3D, build_poisson_inverse

from ..geom import (finish, node_field, ordered_map, stiffness_apply, tile_geometry, tile_jvp, tile_vjp_local,
                    tile_vjp_scatter)
from .blocks import (GAUSS_BLOCKS, ChannelPoint, Curl, Diffusion, LogVolume, _channel_node_add, channel_c_dat,
                     channel_gn_diagonal, channel_jvp, channel_point, logvol_extra_block, merge_cotangents,
                     nonpositive_extra_jd, nonpositive_gauss_jd)
from ..readonly import adopt_readonly
from .state import ModelState, require_extended_longdouble

_RAISE = dict(over="raise", invalid="raise", divide="raise")
LD = np.longdouble
BLOCK_NAMES = ("channel", "diffusion", "logvol", "curl")
GUARD_FACTOR = 16                 # C08-3: s_E = 16 eps64 max(E_b, E_t)
METRIC_MODES = ("principal", "methodology")


@dataclass(frozen=True, eq=False)
class Evaluation:
    """F_B at one u of one ModelState; u is a read-only (nz, ny, nx, 3) node field in U0."""
    state: ModelState
    u: np.ndarray
    channel: ChannelPoint
    total: np.longdouble          # longdouble F_B
    energy: float                 # float(total)
    energies: dict                # "channel": tuple of C floats, "diffusion" / "logvol" / "curl": float


def _readonly(array):
    array = np.array(array, dtype=np.float64, copy=True, order="C")
    array.flags.writeable = False
    return array


def _node_sum(state, values):
    """np.sum(dtype=longdouble) of a node block over the C-order flattened (nz, ny, nx) array whose
    entries are `values` on the mask nodes and exactly zero elsewhere (C06-7)."""
    full = np.zeros(int(np.prod(state.grid.shape)), dtype=values.dtype)
    full[state.node_index] = values
    return np.sum(full, dtype=LD)


def _tile_map(state, function):
    """function(tile) over cell_tiles(state.grid, state.tile_size), results in ascending tile order (CPUPAR)."""
    return ordered_map(function, cell_tiles(state.grid, state.tile_size), state.cpu_threads)


def _evaluate_tile(state, u, tile):
    """(InvalidTrial NONPOSITIVE_GAUSS_JD, None) or (None, [np.sum(r_i^2, dtype=longdouble) for the Gauss-point
    blocks] + [the sum of the second log-volume part under C05-2a / C05-2b] + [(min J, flat (cell, point) index from
    tile.start) of the closed27 points under C05-2b]) of one tile."""
    geo = tile_geometry(u, state.ops, tile=tile)
    count = int(np.count_nonzero(geo.J <= 0))
    extra = logvol_extra_block(state)
    if extra is None:
        if count:
            return nonpositive_gauss_jd(count), None
    else:
        extra_geo = tile_geometry(u, extra.operators(state), tile=tile)
        count_extra = int(np.count_nonzero(extra_geo.J <= 0))
        if count or count_extra:
            return nonpositive_extra_jd(state, count, count_extra), None
    sums = []
    for block in GAUSS_BLOCKS:
        r = block.residual(block.scale(state, tile), geo)
        sums.append(np.sum(r*r, dtype=LD))
    if extra is not None:
        r = extra.residual(extra.scale(state, tile), extra_geo)
        sums.append(np.sum(r*r, dtype=LD))
        if state.logvol_closed27:                                  # C05-2b record: first minimum of the tile
            flat = int(np.argmin(extra_geo.J))
            sums.append((float(extra_geo.J.ravel()[flat]), flat))
    return None, sums


def evaluate(state, u):
    """C06-1/7: Evaluation of F_B at u (free vector or U0 node field), or an InvalidTrial."""
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    u = adopt_readonly(node_field(u, state.grid))           # MEM: no copy of an owned read-only u (same values)
    point = channel_point(state, u)
    if isinstance(point, InvalidTrial):
        return point
    extra = logvol_extra_block(state) is not None
    closed27 = state.logvol_closed27
    n_points = len(state.closed27_ops.points) if closed27 else None
    with np.errstate(**_RAISE):
        channel = [_node_sum(state, point.r[c]*point.r[c]) for c in range(state.n_channels)]
        sums = {block.name: LD(0) for block in GAUSS_BLOCKS}
        part_gauss, part_closed27, minimum = LD(0), LD(0), None
        tiles_seen = cell_tiles(state.grid, state.tile_size)
        with closing(_tile_map(state, partial(_evaluate_tile, state, u))) as tiles:
            for tile, (failure, tile_sums) in zip(tiles_seen, tiles):
                if failure is not None:
                    return failure
                for block, value in zip(GAUSS_BLOCKS, tile_sums):
                    sums[block.name] += value
                if extra:                         # C05-2a / C05-2b: the second part right after the Gauss sum
                    sums[LogVolume.name] += tile_sums[len(GAUSS_BLOCKS)]
                if closed27:
                    part_gauss += tile_sums[1]
                    part_closed27 += tile_sums[3]
                    value, flat = tile_sums[4]
                    if minimum is None or value < minimum[0]:          # the first minimum in tile order
                        minimum = (value, tile.start+flat//n_points, flat % n_points)
        half = LD(.5)
        parts = [half*s for s in channel]+[half*sums[block.name] for block in GAUSS_BLOCKS]
        total = LD(0)
        for part in parts:
            total += part
    energy = float(total)
    require_finite_result(energy, "F_B")
    energies = dict(channel=tuple(float(p) for p in parts[:state.n_channels]),
                    **{block.name: float(p) for block, p in zip(GAUSS_BLOCKS, parts[state.n_channels:])})
    if closed27:
        energies["logvol_parts"] = logvol_parts_record(state, float(half*part_gauss), float(half*part_closed27),
                                                       *minimum)
    return Evaluation(state, u, point, total, energy, energies)


def logvol_parts_record(state, gauss, closed27, min_J, cell, point):
    """C05-2b evaluation record: the Gauss and closed27 parts of the log-volume energy (each 1/2 its longdouble sum of
    r^2) and the minimum J over the closed27 points with its cell (C order) and point (index into closed27_ops.points,
    reference coordinates xyz)."""
    return dict(gauss=gauss, closed27=closed27, closed27_min_J=float(min_J), closed27_min_cell=int(cell),
                closed27_min_point=int(point), closed27_min_xyz=[float(t) for t in state.closed27_ops.points[point]])


def _check(state, ev):
    if not isinstance(ev, Evaluation) or ev.state is not state:
        raise InputValidationError("evaluation: an Evaluation of this ModelState required (sigma, w, eta frozen)")


def _names(blocks):
    names = tuple(blocks)
    if not names or any(name not in BLOCK_NAMES for name in names) or len(set(names)) != len(names):
        raise InputValidationError(f"blocks: a nonempty subset of {BLOCK_NAMES} required")
    return names


def _gauss_selected(names):
    return [block for block in (LogVolume, Curl) if block.name in names]


def _extra_selected(state, selected):
    """The second log-volume part (LogVolumeCorners / LogVolumeClosed27) when the log-volume block is selected, else
    None."""
    return logvol_extra_block(state) if LogVolume in selected else None


def _extra_local(state, extra, u, tile, v=None):
    """C05-2a / C05-2b: the local products of K_Jx^T r_Jx (v None) or of K_Jx^T K_Jx v of the second log-volume part
    `extra` (corners or closed27 points) on one tile."""
    ops = extra.operators(state)
    geo = tile_geometry(u, ops, tile=tile)
    scale = extra.scale(state, tile)
    if v is None:
        r = extra.residual(scale, geo)
    else:
        dJ, _, _ = tile_jvp(v, ops, tile=tile, A=geo.A)
        r = extra.jvp(scale, geo, dJ, None, None)
    gJ, _, _ = extra.cotangent(scale, geo, r)
    return tile_vjp_local(ops, tile=tile, gJ=gJ, A=geo.A)


def _scatter_tiles(state, out, tiles, extra):
    """Scatter the per-tile local products in ascending tile order; under C05-2a / C05-2b (extra not None) each
    tile's second-part products right after its Gauss-point products."""
    for part in tiles:
        if extra is not None:
            part, extra_part = part
            tile_vjp_scatter(out, state.ops, part)
            tile_vjp_scatter(out, extra.operators(state), extra_part)
        else:
            tile_vjp_scatter(out, state.ops, part)


def _gradient_tile(state, u, selected, tile):
    """The local products of sum_i K_i^T r_i of the selected Gauss-point blocks on one tile (tile_vjp_local); with
    the C05-2a corners / C05-2b closed27 points and the log-volume block selected, the pair (Gauss-point products,
    second-part products)."""
    geo = tile_geometry(u, state.ops, tile=tile)
    parts = []
    for block in selected:
        scale = block.scale(state, tile)
        parts.append(block.cotangent(scale, geo, block.residual(scale, geo)))
    gJ, gcurl, _ = merge_cotangents(parts)
    local = tile_vjp_local(state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo.A)
    extra = _extra_selected(state, selected)
    if extra is None:
        return local
    return local, _extra_local(state, extra, u, tile)


def gradient(state, ev, *, blocks=BLOCK_NAMES):
    """C06-4: grad F_B = sum_i K_i^T r_i at ev.u as the free vector (3 N_int,); `blocks` selects terms."""
    _check(state, ev)
    names = _names(blocks)
    out = np.zeros((3,)+state.grid.shape)
    if "channel" in names:
        with np.errstate(**_RAISE):
            coefficient = state.beta*ev.channel.r
        _channel_node_add(state, coefficient, ev.channel.grad, out)
    selected = _gauss_selected(names)
    if selected:
        with closing(_tile_map(state, partial(_gradient_tile, state, ev.u, selected))) as tiles:
            _scatter_tiles(state, out, tiles, _extra_selected(state, selected))
    g = finish(out)
    if "diffusion" in names:
        with np.errstate(**_RAISE):
            g = g+(state.alpha/state.omega0)*stiffness_apply(ev.u, state.grid)
    require_finite_result(g, "gradient of F_B")
    return g


def _hvp_tile(state, u, v, selected, tile):
    """The local products of sum_i K_i^T (K_i v) of the selected Gauss-point blocks on one tile (with the C05-2a
    corners / C05-2b closed27 points and the log-volume block selected, the pair (Gauss-point products, second-part
    products))."""
    geo = tile_geometry(u, state.ops, tile=tile)
    dJ, dcurl, V = tile_jvp(v, state.ops, tile=tile, A=geo.A)
    parts = []
    for block in selected:
        scale = block.scale(state, tile)
        parts.append(block.cotangent(scale, geo, block.jvp(scale, geo, dJ, dcurl, V)))
    gJ, gcurl, _ = merge_cotangents(parts)
    local = tile_vjp_local(state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo.A)
    extra = _extra_selected(state, selected)
    if extra is None:
        return local
    return local, _extra_local(state, extra, u, tile, v)


def hvp(state, ev, v, *, blocks=BLOCK_NAMES):
    """C06-5: H v = sum_i K_i^T (K_i v) at ev.u (GN; exact Hessian not used), v a free vector or U0 field."""
    _check(state, ev)
    names = _names(blocks)
    v = node_field(v, state.grid, name="v")
    out = np.zeros((3,)+state.grid.shape)
    if "channel" in names:
        kv = channel_jvp(state, ev.channel, v)
        with np.errstate(**_RAISE):
            coefficient = state.beta*kv
        _channel_node_add(state, coefficient, ev.channel.grad, out)
    selected = _gauss_selected(names)
    if selected:
        with closing(_tile_map(state, partial(_hvp_tile, state, ev.u, v, selected))) as tiles:
            _scatter_tiles(state, out, tiles, _extra_selected(state, selected))
    h = finish(out)
    if "diffusion" in names:
        with np.errstate(**_RAISE):
            h = h+(state.alpha/state.omega0)*stiffness_apply(v, state.grid)
    require_finite_result(h, "GN Hessian action")
    return h


def gn_diagonal(state, ev):
    """C06-6: the channel-block GN diagonal at ev.u (free vector)."""
    _check(state, ev)
    return channel_gn_diagonal(state, ev.channel)


def c_dat(state, ev):
    """C09.1: the per-node mean channel GN diagonal at the metric freeze point ev.u."""
    _check(state, ev)
    return channel_c_dat(state, ev.channel)


# ----------------------------------------------------------------------------------------------
# C09.1 metric
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Metric:
    """C09.1: M = (1/omega0) [omega_ell a I + b K_Q] on U0, the scalar operator on each component;
    a = c_dat + lam_b + gamma >= 0, b = b_P > 0.  Equivalently M = b K_Q/omega0 + shift I with
    shift = omega_ell a / omega0.  Frozen for one inner solve (C09.1-3)."""
    omega0: float
    a: float
    b: float
    inverse: PoissonInverse3D

    @property
    def shift(self):
        return self.inverse.grid.omega*self.a/self.omega0

    def solve(self, g):
        """M^-1 g = omega0 (omega_ell a I + b K_Q)^-1 g (DST-I), free vector in, free vector out."""
        with np.errstate(**_RAISE):
            out = self.omega0*self.inverse.apply(g)
        require_finite_result(out, "metric solve")
        return out

    def matvec(self, v):
        """M v, computed independently of the spectral inverse (stencil)."""
        with np.errstate(**_RAISE):
            out = self.inverse.matvec(v)/self.omega0
        require_finite_result(out, "metric action")
        return out


def build_metric(state, *, c_dat, b_P, lam_b=0., gamma=0.):
    """C09.1: the metric with a = c_dat + lam_b + gamma (each >= 0) and b = b_P > 0 on state.grid."""
    values = {}
    for name, value in (("c_dat", c_dat), ("lam_b", lam_b), ("gamma", gamma)):
        values[name] = finite_real(value, name)
        if values[name] < 0:
            raise InputValidationError(f"{name}: must be nonnegative (C09.1-2)")
    b = finite_real(b_P, "b_P", positive=True)
    a = values["c_dat"]+values["lam_b"]+values["gamma"]
    return Metric(state.omega0, a, b, build_poisson_inverse(state.grid, a=a, b=b, metric="q1"))


def brain_mean_inverse_tau2(state):
    """<1/tau^2>_B / <1>_B over the mask nodes with weights q_a m_a, tau_a^2 = sum_k Ptilde_T^k tau_k^2."""
    q = state.grid.node_weights().ravel()[state.node_index]
    tau2 = state.tau2_nodes.ravel()[state.node_index]
    with np.errstate(**_RAISE):
        mean = np.sum(q/tau2, dtype=LD)/np.sum(q, dtype=LD)
    return float(mean)


def metric_bP(state, *, mode="principal", mu=None, nu2=None):
    """C09.1 b_P.  "principal" (default, C20 D11): alpha + 1/2 (<1/tau^2>_B/<1>_B + nu), plus
    1/2 (mu + nu2) for outer option (b) when both are given; "methodology": alpha."""
    if mode not in METRIC_MODES:
        raise InputValidationError(f"mode: one of {METRIC_MODES}")
    if (mu is None) != (nu2 is None):
        raise InputValidationError("mu and nu2: give both (outer option (b)) or neither")
    if mode == "methodology":
        if mu is not None:
            raise InputValidationError("mode 'methodology' has no proximal term")
        return state.alpha
    value = state.alpha+.5*(brain_mean_inverse_tau2(state)+state.nu)
    if mu is not None:
        value += .5*(finite_real(mu, "mu")+finite_real(nu2, "nu2"))
    return value


# ----------------------------------------------------------------------------------------------
# C08 stable comparison
# ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Reduction:
    """C08-3/4: ared = float(-Delta E), guard = s_E; blocks holds the longdouble Delta E_i (E_t - E_b):
    "channel" a tuple of C values, then "diffusion", "logvol", "curl"; delta their longdouble sum."""
    ared: float
    guard: float
    delta: np.longdouble
    blocks: dict

    @property
    def certified(self):
        return self.ared-self.guard

    @property
    def reason(self):
        """C08-4: ENERGY_RESOLUTION if |ared| <= s_E, NO_STRICT_DECREASE if certified <= 0, else None."""
        if abs(self.ared) <= self.guard:
            return "ENERGY_RESOLUTION"
        if self.certified <= 0:
            return "NO_STRICT_DECREASE"
        return None


def _square_change_sum(base, trial):
    with np.errstate(**_RAISE):
        a, b = base.astype(LD), trial.astype(LD)
        return np.sum((b-a)*(b+a), dtype=LD)


def half_square_change(base, trial):
    """C08-1: 1/2 sum_k (t_k - b_k)(t_k + b_k) of two residual arrays, longdouble elementwise and summed
    in C order."""
    base = as_float64_array(base, "base").ravel(order="C")
    trial = as_float64_array(trial, "trial", None).ravel(order="C")
    if trial.shape != base.shape:
        raise InputValidationError("trial: same shape as base required")
    return _square_change_sum(base, trial)/2


def _logvol_change_sum(J_b, J_t, weight):
    if np.any(J_b <= 0) or np.any(J_t <= 0):
        raise InputValidationError("C08: J <= 0 at a Gauss point of a compared state")
    with np.errstate(**_RAISE):
        a, b = J_b.astype(LD), J_t.astype(LD)
        x = (b-a)/a
        ell = np.empty_like(x)
        near = x >= -.5
        ell[near] = np.log1p(x[near])
        ell[~near] = np.log(b[~near]/a[~near])
        return np.sum(weight*ell*(np.log(b)+np.log(a)), dtype=LD)


def logvol_change(J_b, J_t, weight):
    """C08-2: 1/2 sum_q weight_q l_q (log J_t + log J_b), l_q = log1p((J_t - J_b)/J_b) for x >= -1/2,
    else log(J_t/J_b); all longdouble.  weight is omega w_g/(omega0 tau_q^2) (float64 or longdouble)."""
    J_b = as_float64_array(J_b, "J_b")
    J_t = as_float64_array(J_t, "J_t", J_b.shape)
    weight = np.asarray(weight)
    if weight.dtype not in (np.float64, LD) or weight.shape != J_b.shape or not np.all(np.isfinite(weight)):
        raise InputValidationError("weight: finite float64 or longdouble array of the J shape required")
    return _logvol_change_sum(J_b, J_t, weight.astype(LD))/2


def _node_change_sum(state, base, trial):
    """sum (t - b)(t + b) of a node block over its C-order flattened (nz, ny, nx) array (zero rows
    outside the mask contribute exact zeros in place)."""
    with np.errstate(**_RAISE):
        a, b = base.astype(LD), trial.astype(LD)
        full = np.zeros(int(np.prod(state.grid.shape)), dtype=LD)
        full[state.node_index] = (b-a)*(b+a)
        return np.sum(full, dtype=LD)


def _reduction_tile(state, u_b, u_t, weights, omega0, tile):
    """[(block, longdouble sum of the tile)] of C08-1/2 for diffusion, curl and log-volume on one tile; under C05-2a
    (C05-2b) a second log-volume entry, the corner (closed27) term, right after the Gauss-point one."""
    geo_b = tile_geometry(u_b, state.ops, tile=tile)
    geo_t = tile_geometry(u_t, state.ops, tile=tile)
    out = []
    for block in (Diffusion, Curl):
        scale = block.scale(state, tile)
        out.append((block.name, _square_change_sum(block.residual(scale, geo_b), block.residual(scale, geo_t))))
    weight = weights/(omega0*LogVolume.tau2(state, tile).astype(LD))
    out.append((LogVolume.name, _logvol_change_sum(geo_b.J, geo_t.J, weight)))
    extra = logvol_extra_block(state)
    if extra is not None:
        ops = extra.operators(state)
        extra_b = tile_geometry(u_b, ops, tile=tile)
        extra_t = tile_geometry(u_t, ops, tile=tile)
        weight = extra.point_weights(state).astype(LD)/(omega0*extra.tau2(state, tile).astype(LD))
        out.append((LogVolume.name, _logvol_change_sum(extra_b.J, extra_t.J, weight)))
    return out


def stable_reduction(state, base, trial):
    """C08: the stable energy difference of two Evaluations of the same ModelState (both certified, so
    every J_q > 0, and every corner J under C05-2a, every closed27 J under C05-2b; otherwise InputValidationError).
    The Gauss log-volume weights are omega w_g (C05-2; times theta_G under C05-2b: state.logvol_gauss_weights)."""
    require_extended_longdouble()
    _check(state, base)
    _check(state, trial)
    channel = [_node_change_sum(state, base.channel.r[c], trial.channel.r[c]) for c in range(state.n_channels)]
    sums = {block.name: LD(0) for block in GAUSS_BLOCKS}
    weights = state.logvol_gauss_weights.astype(LD)
    omega0 = LD(state.omega0)
    with np.errstate(**_RAISE):
        with closing(_tile_map(state, partial(_reduction_tile, state, base.u, trial.u, weights, omega0))) as tiles:
            for tile_sums in tiles:
                for name, value in tile_sums:
                    sums[name] += value
        half = LD(.5)
        changes = [half*s for s in channel]+[half*sums[block.name] for block in GAUSS_BLOCKS]
        delta = LD(0)
        for change in changes:
            delta += change
    ared = float(-delta)
    guard = float(GUARD_FACTOR*np.finfo(float).eps*max(base.energy, trial.energy))
    require_finite_result(np.array([ared, guard]), "stable energy reduction")
    blocks = dict(channel=tuple(changes[:state.n_channels]),
                  **{block.name: change for block, change in zip(GAUSS_BLOCKS, changes[state.n_channels:])})
    return Reduction(ared, guard, delta, blocks)
