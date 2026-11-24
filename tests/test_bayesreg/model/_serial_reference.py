"""The serial CPU reference as it was before work package CPUPAR (commit 8d624c8), for the bitwise tests.

Function bodies are copied from bayesreg/model/blocks.py (channel_point), bayesreg/model/energy.py (evaluate,
gradient, hvp, stable_reduction) and bayesreg/solve/proximal.py (_residuals, evaluate, gradient, change) at
8d624c8, with only the module-level names they use imported below and the names prefixed `ref_`.  They loop over
the tiles in one thread and accumulate with tile_vjp_accumulate (_transpose_add), exactly as before; the helpers
they call (_node_sum, _square_change_sum, _logvol_change_sum, _node_change_sum, the block kernels) are unchanged
by CPUPAR.  Do not edit the bodies: they are the "single-thread implementation" every thread count must equal.
"""
import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, readonly_copy, require_finite_result
from jd_curl.three_d.cell_q1 import cell_tiles
from jd_curl.three_d.interpolation import check_map_domain, sample_points

from bayesreg.geom import finish, node_field, stiffness_apply, tile_geometry, tile_jvp, tile_vjp_accumulate
from bayesreg.model.blocks import (GAUSS_BLOCKS, ChannelPoint, Curl, Diffusion, LogVolume, _channel_node_add,
                                   channel_jvp, merge_cotangents, nonpositive_gauss_jd)
from bayesreg.model.energy import (BLOCK_NAMES, GUARD_FACTOR, Evaluation, Reduction, _check, _gauss_selected,
                                   _logvol_change_sum, _names, _node_change_sum, _node_sum, _readonly,
                                   _square_change_sum)
from bayesreg.model.state import ModelState, require_extended_longdouble
from bayesreg.solve.proximal import PROX_BLOCKS, ProximalValue, _fields, _gauss_scales, _lam_scale

_RAISE = dict(over="raise", invalid="raise", divide="raise")
LD = np.longdouble


# ---- bayesreg/model/blocks.py @ 8d624c8 -------------------------------------------------------------------

def ref_channel_point(state, u):
    u = node_field(u, state.grid)
    invalid = check_map_domain(u, state.grid)
    if invalid is not None:
        return invalid
    with np.errstate(**_RAISE):
        points = state.points0+u.reshape(-1, 3)[state.node_index]
    values = np.empty(state.fixed_masked.shape)
    grads = np.empty(values.shape+(3,))
    for c, sampler in enumerate(state.samplers):
        sample = sample_points(sampler, points, strict=state.strict)
        if isinstance(sample, InvalidTrial):
            return sample
        values[c], grads[c] = sample.value, sample.gradient
    with np.errstate(**_RAISE):
        rho = values-state.fixed_masked
        r = state.beta*rho
    require_finite_result(r, "channel residual")
    return ChannelPoint(readonly_copy(rho), readonly_copy(grads), readonly_copy(r))


# ---- bayesreg/model/energy.py @ 8d624c8 -------------------------------------------------------------------

def ref_evaluate(state, u):
    if not isinstance(state, ModelState):
        raise InputValidationError("state: ModelState required")
    u = _readonly(node_field(u, state.grid))
    point = ref_channel_point(state, u)
    if isinstance(point, InvalidTrial):
        return point
    with np.errstate(**_RAISE):
        channel = [_node_sum(state, point.r[c]*point.r[c]) for c in range(state.n_channels)]
        sums = {block.name: LD(0) for block in GAUSS_BLOCKS}
        for tile in cell_tiles(state.grid, state.tile_size):
            geo = tile_geometry(u, state.ops, tile=tile)
            count = int(np.count_nonzero(geo.J <= 0))
            if count:
                return nonpositive_gauss_jd(count)
            for block in GAUSS_BLOCKS:
                r = block.residual(block.scale(state, tile), geo)
                sums[block.name] += np.sum(r*r, dtype=LD)
        half = LD(.5)
        parts = [half*s for s in channel]+[half*sums[block.name] for block in GAUSS_BLOCKS]
        total = LD(0)
        for part in parts:
            total += part
    energy = float(total)
    require_finite_result(energy, "F_B")
    energies = dict(channel=tuple(float(p) for p in parts[:state.n_channels]),
                    **{block.name: float(p) for block, p in zip(GAUSS_BLOCKS, parts[state.n_channels:])})
    return Evaluation(state, u, point, total, energy, energies)


def ref_gradient(state, ev, *, blocks=BLOCK_NAMES):
    _check(state, ev)
    names = _names(blocks)
    out = np.zeros((3,)+state.grid.shape)
    if "channel" in names:
        with np.errstate(**_RAISE):
            coefficient = state.beta*ev.channel.r
        _channel_node_add(state, coefficient, ev.channel.grad, out)
    selected = _gauss_selected(names)
    if selected:
        for tile in cell_tiles(state.grid, state.tile_size):
            geo = tile_geometry(ev.u, state.ops, tile=tile)
            parts = []
            for block in selected:
                scale = block.scale(state, tile)
                parts.append(block.cotangent(scale, geo, block.residual(scale, geo)))
            gJ, gcurl, _ = merge_cotangents(parts)
            tile_vjp_accumulate(out, state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo.A)
    g = finish(out)
    if "diffusion" in names:
        with np.errstate(**_RAISE):
            g = g+(state.alpha/state.omega0)*stiffness_apply(ev.u, state.grid)
    require_finite_result(g, "gradient of F_B")
    return g


def ref_hvp(state, ev, v, *, blocks=BLOCK_NAMES):
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
        for tile in cell_tiles(state.grid, state.tile_size):
            geo = tile_geometry(ev.u, state.ops, tile=tile)
            dJ, dcurl, V = tile_jvp(v, state.ops, tile=tile, A=geo.A)
            parts = []
            for block in selected:
                scale = block.scale(state, tile)
                parts.append(block.cotangent(scale, geo, block.jvp(scale, geo, dJ, dcurl, V)))
            gJ, gcurl, _ = merge_cotangents(parts)
            tile_vjp_accumulate(out, state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo.A)
    h = finish(out)
    if "diffusion" in names:
        with np.errstate(**_RAISE):
            h = h+(state.alpha/state.omega0)*stiffness_apply(v, state.grid)
    require_finite_result(h, "GN Hessian action")
    return h


def ref_stable_reduction(state, base, trial):
    require_extended_longdouble()
    _check(state, base)
    _check(state, trial)
    channel = [_node_change_sum(state, base.channel.r[c], trial.channel.r[c]) for c in range(state.n_channels)]
    sums = {block.name: LD(0) for block in GAUSS_BLOCKS}
    weights = state.ops.weights.astype(LD)
    omega0 = LD(state.omega0)
    with np.errstate(**_RAISE):
        for tile in cell_tiles(state.grid, state.tile_size):
            geo_b = tile_geometry(base.u, state.ops, tile=tile)
            geo_t = tile_geometry(trial.u, state.ops, tile=tile)
            for block in (Diffusion, Curl):
                scale = block.scale(state, tile)
                sums[block.name] += _square_change_sum(block.residual(scale, geo_b), block.residual(scale, geo_t))
            weight = weights/(omega0*LogVolume.tau2(state, tile).astype(LD))
            sums[LogVolume.name] += _logvol_change_sum(geo_b.J, geo_t.J, weight)
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


# ---- bayesreg/solve/proximal.py @ 8d624c8 -----------------------------------------------------------------

def ref_prox_residuals(state, params, u, v, d):
    with np.errstate(**_RAISE):
        yield "prox_lambda", _lam_scale(state, params)*d
    s_mu, s_nu2 = _gauss_scales(state, params)
    for tile in cell_tiles(state.grid, state.tile_size):
        geo_u = tile_geometry(u, state.ops, tile=tile)
        geo_v = tile_geometry(v, state.ops, tile=tile)
        geo_d = tile_geometry(d, state.ops, tile=tile)
        with np.errstate(**_RAISE):
            yield "prox_mu", s_mu*(geo_u.J-geo_v.J)
            yield "prox_nu2", s_nu2*geo_d.curl


def ref_prox_evaluate(state, u, v, params):
    u, v, d = _fields(state, u, v)
    sums = {name: LD(0) for name in PROX_BLOCKS}
    with np.errstate(**_RAISE):
        for name, r in ref_prox_residuals(state, params, u, v, d):
            sums[name] += np.sum(r*r, dtype=LD)
        half = LD(.5)
        parts = {name: half*sums[name] for name in PROX_BLOCKS}
        total = LD(0)
        for name in PROX_BLOCKS:
            total += parts[name]
    require_finite_result(float(total), "proximal term")
    return ProximalValue(u, v, parts, total)


def ref_prox_gradient(state, value, params):
    from jd_curl.three_d.boundary import restrict_free
    u, v = value.u, value.v
    with np.errstate(**_RAISE):
        d = np.ascontiguousarray(u-v)
        lam = restrict_free((params.lam*state.grid.node_weights()/state.omega0)[..., None]*d, state.grid)
    w = state.ops.weights[None, :]
    out = np.zeros((3,)+state.grid.shape)
    for tile in cell_tiles(state.grid, state.tile_size):
        geo_u = tile_geometry(u, state.ops, tile=tile)
        geo_v = tile_geometry(v, state.ops, tile=tile)
        geo_d = tile_geometry(d, state.ops, tile=tile)
        with np.errstate(**_RAISE):
            gJ = (params.mu*w/state.omega0)*(geo_u.J-geo_v.J)
            gcurl = (params.nu2*w/state.omega0)[..., None]*geo_d.curl
        tile_vjp_accumulate(out, state.ops, tile=tile, gJ=gJ, gcurl=gcurl, A=geo_u.A)
    with np.errstate(**_RAISE):
        g = finish(out)+lam
    require_finite_result(g, "proximal gradient")
    return g


def ref_prox_change(state, base, trial, params):
    if not np.array_equal(base.v, trial.v):
        raise InputValidationError("proximal change: both values need the same reference v")
    sums = {name: LD(0) for name in PROX_BLOCKS}
    b_iter = ref_prox_residuals(state, params, base.u, base.v, np.ascontiguousarray(base.u-base.v))
    t_iter = ref_prox_residuals(state, params, trial.u, trial.v, np.ascontiguousarray(trial.u-trial.v))
    with np.errstate(**_RAISE):
        for (name, rb), (name_t, rt) in zip(b_iter, t_iter):
            assert name == name_t
            a, b = rb.astype(LD), rt.astype(LD)
            sums[name] += np.sum((b-a)*(b+a), dtype=LD)
        half = LD(.5)
        return {name: half*sums[name] for name in PROX_BLOCKS}
