"""Proximal term of outer option (b) (contract C10-2; theory B9), numpy reference on bayesreg.geom.

    P(u, v) = lam/2 sum_a (q_a/omega0) |u_a - v_a|^2 + mu/2 <(J_q(u) - J_q(v))^2>_Q + nu2/2 <|c_q(u - v)|^2>_Q

(the lam term over the whole field, no mask; <f>_Q = sum_q omega w_g f_q / omega0).  Residual blocks
    r_lam = sqrt(lam q_a/omega0) (u - v)_a         node layout (nz, ny, nx, 3), C order
    r_mu  = sqrt(mu omega w_g/omega0) (J_q(u) - J_q(v))   (cells, 27) per tile
    r_nu2 = sqrt(nu2 omega w_g/omega0) c_q(u - v)          (cells, 27, 3) per tile
P = 1/2 ||r||^2.  Sums and differences follow C06-7 / C08-1: longdouble elementwise and summed, node
block over the C-order flattened array, Gauss-point blocks per tile of state.tile_size cells in ascending
order.  Gradient: K_lam^T r_lam = (lam q_a/omega0)(u - v)_a restricted to U0; the mu and nu2 blocks in one
tile_vjp_accumulate per tile with gJ = mu omega w_g/omega0 (J(u) - J(v)) and A = cof F(u), gcurl =
nu2 omega w_g/omega0 c_q(u - v).  P(v, v) = 0 and grad_u P(v, v) = 0 exactly (B9-1).

J_q(v) of the reference is recomputed per tile (never cached: a native volume has ~1.8e8 Gauss points).
The per-tile geometry and residuals (and the gradient's local products) run in state.cpu_threads threads
(bayesreg.geom.ordered_map, CPUPAR); the residuals are yielded, and the local products scattered, in ascending
tile order, so every value is bitwise that of the serial loop.
(lam, mu, nu2) are algorithm parameters (C10-3): they never enter the posterior or the Laplace H.
"""
from contextlib import closing
from dataclasses import dataclass
from functools import partial

import numpy as np

from jd_curl.validation import InputValidationError, finite_real, require_finite_result
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.validation import integer_at_least

from ..geom import finish, node_field, ordered_map, tile_geometry, tile_jvp, tile_vjp_local, tile_vjp_scatter
from ..model import Reduction
from ..model.energy import GUARD_FACTOR

_RAISE = dict(over="raise", invalid="raise", divide="raise")
LD = np.longdouble
PROX_BLOCKS = ("prox_lambda", "prox_mu", "prox_nu2")
DEFAULT_MU = .03                  # C10-3 starting value (configuration C native level)
DEFAULT_NU2 = .03
DEFAULT_N_IN = 5                  # C09.2-9 / D12
LAMBDA_FACTOR = .1                # lam = .1 / L^2, L the geometric mean FOV edge (mm)


def default_lambda(grid):
    """C10-3: lam = .1/L^2, L = geometric mean of the FOV edge lengths (n_j - 1) h_j of `grid` (mm)."""
    edges = [axis[-1]-axis[0] for axis in grid.axes]
    length = float(np.prod(edges))**(1/3)
    return LAMBDA_FACTOR/length**2


@dataclass(frozen=True)
class ProximalParams:
    """(lam, mu, nu2) and the inner accepted-step limit n_in of option (b)."""
    lam: float
    mu: float = DEFAULT_MU
    nu2: float = DEFAULT_NU2
    n_in: int = DEFAULT_N_IN

    def __post_init__(self):
        object.__setattr__(self, "lam", finite_real(self.lam, "lam", positive=True))   # C10 gate: lam > 0
        for name in ("mu", "nu2"):
            value = finite_real(getattr(self, name), name)
            if value < 0:
                raise InputValidationError(f"{name}: must be nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "n_in", integer_at_least(self.n_in, "n_in"))

    def record(self):
        return dict(lam=self.lam, mu=self.mu, nu2=self.nu2, n_in=self.n_in, role="algorithm parameters (C10-3)")


@dataclass(frozen=True, eq=False)
class ProximalValue:
    """P(u, v) at one u: the three halves of the longdouble sums and the node-field difference."""
    u: np.ndarray
    v: np.ndarray
    parts: dict                   # block name -> longdouble 1/2 sum r^2
    total: np.longdouble

    @property
    def energy(self):
        return float(self.total)


def _lam_scale(state, params):
    q = state.grid.node_weights()
    with np.errstate(**_RAISE):
        return np.sqrt(params.lam*q/state.omega0)[..., None]


def _gauss_scales(state, params):
    w = state.ops.weights
    with np.errstate(**_RAISE):
        return np.sqrt(params.mu*w/state.omega0)[None, :], np.sqrt(params.nu2*w/state.omega0)[None, :, None]


def _residual_tile(state, s_mu, s_nu2, u, v, d, tile):
    """(r_mu, r_nu2) of one tile."""
    geo_u = tile_geometry(u, state.ops, tile=tile)
    geo_v = tile_geometry(v, state.ops, tile=tile)
    geo_d = tile_geometry(d, state.ops, tile=tile)
    with np.errstate(**_RAISE):
        return s_mu*(geo_u.J-geo_v.J), s_nu2*geo_d.curl


def _residuals(state, params, u, v, d):
    """Yield (block, residual array) in the fixed order: lam (whole node field), then per tile mu, nu2."""
    with np.errstate(**_RAISE):
        yield "prox_lambda", _lam_scale(state, params)*d
    s_mu, s_nu2 = _gauss_scales(state, params)
    tiles = ordered_map(partial(_residual_tile, state, s_mu, s_nu2, u, v, d), cell_tiles(state.grid, state.tile_size),
                        state.cpu_threads)
    with closing(tiles):
        for r_mu, r_nu2 in tiles:
            yield "prox_mu", r_mu
            yield "prox_nu2", r_nu2


def _fields(state, u, v):
    u = node_field(u, state.grid)
    v = node_field(v, state.grid, name="v")
    with np.errstate(**_RAISE):
        d = np.ascontiguousarray(u-v)
    return u, v, d


def evaluate(state, u, v, params):
    """P(u, v) (C10-2) for the ModelState's grid, omega0 and tile size."""
    u, v, d = _fields(state, u, v)
    sums = {name: LD(0) for name in PROX_BLOCKS}
    with np.errstate(**_RAISE):
        for name, r in _residuals(state, params, u, v, d):
            sums[name] += np.sum(r*r, dtype=LD)
        half = LD(.5)
        parts = {name: half*sums[name] for name in PROX_BLOCKS}
        total = LD(0)
        for name in PROX_BLOCKS:
            total += parts[name]
    require_finite_result(float(total), "proximal term")
    return ProximalValue(u, v, parts, total)


def _gradient_tile(state, params, w, u, v, d, tile):
    """The local products (tile_vjp_local) of the mu and nu2 blocks' adjoint on one tile."""
    geo_u = tile_geometry(u, state.ops, tile=tile)
    geo_v = tile_geometry(v, state.ops, tile=tile)
    geo_d = tile_geometry(d, state.ops, tile=tile)
    with np.errstate(**_RAISE):
        gJ = (params.mu*w/state.omega0)*(geo_u.J-geo_v.J)
        gcurl = (params.nu2*w/state.omega0)[..., None]*geo_d.curl
    return tile_vjp_local(state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo_u.A)


def gradient(state, value, params):
    """grad_u P(u, v) at value.u as the free vector (3 N_int,)."""
    u, v = value.u, value.v
    with np.errstate(**_RAISE):
        d = np.ascontiguousarray(u-v)
        lam = restrict_free((params.lam*state.grid.node_weights()/state.omega0)[..., None]*d, state.grid)
    w = state.ops.weights[None, :]
    out = np.zeros((3,)+state.grid.shape)
    tiles = ordered_map(partial(_gradient_tile, state, params, w, u, v, d), cell_tiles(state.grid, state.tile_size),
                        state.cpu_threads)
    with closing(tiles):
        for part in tiles:
            tile_vjp_scatter(out, state.ops, part)
    with np.errstate(**_RAISE):
        g = finish(out)+lam
    require_finite_result(g, "proximal gradient")
    return g


def _hvp_tile(state, params, w, u, V, tile):
    """The local products of K_mu^T K_mu V + K_nu2^T K_nu2 V on one tile (A = cof F(u) of the base u)."""
    geo_u = tile_geometry(u, state.ops, tile=tile)
    dJ, dcurl, _ = tile_jvp(V, state.ops, tile=tile, A=geo_u.A)
    with np.errstate(**_RAISE):
        gJ = (params.mu*w/state.omega0)*dJ
        gcurl = (params.nu2*w/state.omega0)[..., None]*dcurl
    return tile_vjp_local(state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo_u.A)


def hvp(state, value, v, params):
    """C09.3-10 option (b) (work package LMI): (K_lam^T K_lam + K_mu^T K_mu + K_nu2^T K_nu2) v at value.u, the GN of the
    three proximal blocks (exact for lam and nu2, which are linear in u - v): K_lam^T K_lam v = (lam q_a/omega0) v_a over
    the whole field (no mask); (K_mu v)_q = sqrt(mu omega w_g/omega0) A_q(u) : grad v_q with A_q = cof F at the BASE u
    (not at the reference); (K_nu2 v)_q = sqrt(nu2 omega w_g/omega0) curl v_q.  v a free vector or U0 field; returns the
    free vector.  Tiles in ascending order, the local products in threads (ordered_map, as `gradient`)."""
    u = value.u
    V = node_field(v, state.grid, name="v")
    with np.errstate(**_RAISE):
        lam = restrict_free((params.lam*state.grid.node_weights()/state.omega0)[..., None]*V, state.grid)
    w = state.ops.weights[None, :]
    out = np.zeros((3,)+state.grid.shape)
    tiles = ordered_map(partial(_hvp_tile, state, params, w, u, V), cell_tiles(state.grid, state.tile_size),
                        state.cpu_threads)
    with closing(tiles):
        for part in tiles:
            tile_vjp_scatter(out, state.ops, part)
    with np.errstate(**_RAISE):
        h = finish(out)+lam
    require_finite_result(h, "proximal GN Hessian action")
    return h


def change(state, base, trial, params):
    """C08-1 for the three proximal blocks: {block: longdouble 1/2 sum (t - b)(t + b)} of two values with
    the same reference v."""
    if not np.array_equal(base.v, trial.v):
        raise InputValidationError("proximal change: both values need the same reference v")
    sums = {name: LD(0) for name in PROX_BLOCKS}
    b_iter = _residuals(state, params, base.u, base.v, np.ascontiguousarray(base.u-base.v))
    t_iter = _residuals(state, params, trial.u, trial.v, np.ascontiguousarray(trial.u-trial.v))
    with np.errstate(**_RAISE):
        for (name, rb), (name_t, rt) in zip(b_iter, t_iter):
            assert name == name_t
            a, b = rb.astype(LD), rt.astype(LD)
            sums[name] += np.sum((b-a)*(b+a), dtype=LD)
        half = LD(.5)
        return {name: half*sums[name] for name in PROX_BLOCKS}


@dataclass(frozen=True, eq=False)
class ProxPoint:
    """E_v(u) = F_B(u) + P(u, v): the F_B Evaluation, the proximal value, the longdouble and float total."""
    fb: object
    prox: ProximalValue
    total: np.longdouble
    energy: float

    @property
    def u(self):
        return self.fb.u


def combine(fb, prox):
    with np.errstate(**_RAISE):
        total = LD(fb.total)+prox.total
    energy = float(total)
    require_finite_result(energy, "E_v")
    return ProxPoint(fb, prox, total, energy)


def combined_reduction(fb_reduction, prox_changes, base, trial):
    """C08 for E_v: Delta E = Delta F_B (longdouble) + the proximal block changes; s_E = 16 eps max(E_b, E_t)
    with the E_v values; the blocks of both kept."""
    with np.errstate(**_RAISE):
        delta = LD(fb_reduction.delta)
        for name in PROX_BLOCKS:
            delta += prox_changes[name]
    ared = float(-delta)
    guard = float(GUARD_FACTOR*np.finfo(float).eps*max(base.energy, trial.energy))
    require_finite_result(np.array([ared, guard]), "stable E_v reduction")
    blocks = dict(fb_reduction.blocks, **prox_changes)
    return Reduction(ared, guard, delta, blocks)
