"""bayesreg.model.blocks and ModelState against contract C04, C05, C06-2/3/6 (gates) and theory B2.

Small grids only (<= 7x6x5 nodes); no data/ access, no certificate here (see test_energy.py).
DEF (2026-10-31): these tests are about the C05-2 model (the ModelState default before DEF); their states are
built with logvol_points "gauss27" explicitly (_gauss27.make_state).
"""
import json

import numpy as np
import pytest

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import inject_free

from bayesreg.geom import gauss_geometry, tile_interpolate
from bayesreg.model import (Curl, Diffusion, LogVolume, NONPOSITIVE_GAUSS_JD, build_samplers, c_dat, channel_expand,
                            channel_jvp, channel_point, channel_vjp, evaluate, gauss_jvp, gauss_value, gauss_vjp,
                            gn_diagonal, gradient, hvp, require_extended_longdouble)

from ._gauss27 import make_state
from ._helpers import TAU2, aniso, dense_model, feasible_field, free_random, poly_images, relmax, theta_of

GAUSS = {"diffusion": Diffusion, "logvol": LogVolume, "curl": Curl}


def block_value(state, name, u):
    if name == "channel":
        return channel_expand(state, channel_point(state, u).r)
    return gauss_value(GAUSS[name], state, u)


def block_jvp(state, name, u, v):
    if name == "channel":
        return channel_expand(state, channel_jvp(state, channel_point(state, u), v))
    return gauss_jvp(GAUSS[name], state, u, v)


def block_vjp(state, name, u, r):
    """K_i^T r with r in the layout of block_value."""
    if name == "channel":
        return channel_vjp(state, channel_point(state, u), r.reshape(state.n_channels, -1)[:, state.node_index])
    return gauss_vjp(GAUSS[name], state, u, r)


NAMES = ("channel", "diffusion", "logvol", "curl")


# ----------------------------------------------------------------------------------------------
# ModelState
# ----------------------------------------------------------------------------------------------

def test_model_state_validation_and_replace():
    """Inputs are validated once; derived beta = sqrt(q m w/(omega0 sigma^2)) on the mask nodes; replace
    keeps grid, channels and samplers and recomputes the derived arrays; C08-7 platform guard."""
    g = aniso()
    state = make_state(g, student_t=True)
    q = g.node_weights().ravel()[state.node_index]
    w = state.weights.reshape(3, -1)[:, state.node_index]
    np.testing.assert_array_equal(state.beta, np.sqrt(q*w/(state.omega0*state.sigma2[:, None])))
    assert np.array_equal(state.node_index, np.flatnonzero(state.mask.ravel()))
    for name in ("fixed", "mask", "beta", "tau2_nodes", "points0", "fixed_masked", "node_index"):
        assert not getattr(state, name).flags.writeable, name
    other = state.replace(sigma2=np.array([.5, .5, .5]), weights=None)
    assert all(a is b for a, b in zip(other.samplers, state.samplers, strict=True))   # splines not rebuilt
    np.testing.assert_array_equal(other.beta, np.sqrt(q*1./(state.omega0*.5))[None].repeat(3, axis=0))
    record = state.provenance()
    assert record["tile_size"] == 8192 and record["omega0"] == g.omega/3 and record["n_channels"] == 3
    bad = dict(mask=np.zeros(g.shape, bool), sigma2=np.array([.1, 0., .1]), alpha=-1., tau2=np.array([1., -1, 1, 1]),
               tissue=np.full((4,)+g.shape, .3), weights=-np.ones((3,)+g.shape), tile_size=0, omega0=0.,
               samplers=state.samplers[:2], fixed=np.zeros((3, 4, 4, 4)))
    for key, value in bad.items():
        with pytest.raises(InputValidationError):
            state.replace(**{key: value})
    with pytest.raises(InputValidationError):
        state.replace(samplers=build_samplers(np.zeros((3,)+aniso(nx=6).shape), aniso(nx=6)))
    require_extended_longdouble()
    with pytest.raises(InputValidationError, match="C08-7"):
        require_extended_longdouble(nmant=52)


# ----------------------------------------------------------------------------------------------
# C06: dense oracle, central differences, adjoints
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("student_t", [False, True])
def test_dense_oracle_values_jvp_vjp(student_t, record_property):
    """C06 gate: on 7x6x5 nodes every block's r_i, K_i v and K_i^T r equal an independent dense assembly to
    1e-12 (omega0 != omega_ell; random mask, tissue and, with Student-t, weights)."""
    g = aniso()
    state = make_state(g, seed=4, student_t=student_t)
    rng = np.random.default_rng(5)
    u = feasible_field(g, rng)
    v = free_random(g, rng)
    dense = dense_model(state, u)
    errors = {}
    for name in NAMES:
        r_dense, K = dense[name]
        value = block_value(state, name, u)
        kv = block_jvp(state, name, u, v)
        cot = rng.normal(size=np.shape(value))
        back = block_vjp(state, name, u, cot)
        if name == "channel":
            errors[name] = dict(value=relmax(value.reshape(3, -1), r_dense), jvp=relmax(kv.reshape(3, -1), K@v),
                                vjp=relmax(back, np.einsum("cnk,cn->k", K, cot.reshape(3, -1)*state.mask.ravel())))
        else:
            errors[name] = dict(value=relmax(np.ravel(value), r_dense), jvp=relmax(np.ravel(kv), K@v),
                                vjp=relmax(back, K.T@np.ravel(cot)))
    record_property("dense_oracle_errors", json.dumps(errors))
    assert max(e for block in errors.values() for e in block.values()) <= 1e-12, errors
    # the linear blocks' residuals are K theta (r_D = K_D theta, r_C = K_C theta)
    theta = theta_of(u, g)
    for name in ("diffusion", "curl"):
        assert relmax(np.ravel(block_value(state, name, u)), dense[name][1]@theta) <= 1e-12


@pytest.mark.parametrize("seed", [0, 7])
def test_block_jvp_central_difference(seed, record_property):
    """C06 gate: |(r_i(u + t v) - r_i(u - t v))/2t - K_i v|_inf <= 1e-7 max(1, |K_i v|_inf), |t v| ~ 1e-4 |u|."""
    g = aniso()
    state = make_state(g, seed=seed, student_t=bool(seed))
    rng = np.random.default_rng(seed+10)
    u = feasible_field(g, rng)
    v = inject_free(free_random(g, rng), g)
    v *= np.max(np.abs(u))/np.max(np.abs(v))
    t = 1e-4
    errors = {}
    for name in NAMES:
        kv = block_jvp(state, name, u, v)
        fd = (block_value(state, name, u+t*v)-block_value(state, name, u-t*v))/(2*t)
        errors[name] = float(np.max(np.abs(fd-kv))/max(1., np.max(np.abs(kv))))
        assert np.max(np.abs(kv)) > 0, name
    record_property("fd_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-7, errors


@pytest.mark.parametrize("seed", [0, 3])
def test_block_adjoint(seed, record_property):
    """C06 gate: |<K_i v, r> - <v, K_i^T r>| <= 1e-12 |K_i v| |r| for every block, on U0."""
    g = aniso()
    state = make_state(g, seed=seed, student_t=True)
    rng = np.random.default_rng(seed+20)
    u = feasible_field(g, rng)
    theta = free_random(g, rng)
    errors = {}
    for name in NAMES:
        kv = block_jvp(state, name, u, theta)
        r = rng.normal(size=np.shape(kv))
        if name == "channel":
            r = r*state.mask[None]          # the block's rows outside the mask are structurally zero
        lhs, rhs = float(np.sum(kv*r)), float(theta@block_vjp(state, name, u, r))
        errors[name] = abs(lhs-rhs)/(np.linalg.norm(kv)*np.linalg.norm(r))
    record_property("adjoint_errors", json.dumps(errors))
    assert max(errors.values()) <= 1e-12, errors


# ----------------------------------------------------------------------------------------------
# C04 channel gates
# ----------------------------------------------------------------------------------------------

def test_channel_self_pair_and_mask():
    """C04 gates: M = F self-pair at u = 0 gives |rho_I| <= 1e-12 at the nodes (spline interpolation);
    rho_c(0) = G_c(x_a) - g_c(a); nodes outside the mask contribute exactly 0 to the residual, the gradient
    and the GN diagonal, and changing g_c there changes nothing (bitwise)."""
    g = aniso()
    rng = np.random.default_rng(1)
    image = rng.normal(size=(1,)+g.shape)
    state = make_state(g, moving=image, fixed=image, sigma2=(.1,))
    point = channel_point(state, np.zeros(3*g.n_free_scalar))
    assert np.max(np.abs(point.rho)) <= 1e-12
    state = make_state(g, seed=2)
    point = channel_point(state, np.zeros(g.shape+(3,)))
    np.testing.assert_allclose(point.rho, poly_images(state.points0)-state.fixed_masked, rtol=0, atol=1e-14)

    u = feasible_field(g, rng)
    ev = evaluate(state, u)
    outside = ~state.mask
    assert np.all(channel_expand(state, ev.channel.r)[:, outside] == 0)
    interior_outside = outside[1:-1, 1:-1, 1:-1].ravel()
    assert interior_outside.any()
    grad_c = gradient(state, ev, blocks=("channel",)).reshape(3, -1)
    diag = gn_diagonal(state, ev).reshape(3, -1)
    assert np.all(grad_c[:, interior_outside] == 0) and np.all(diag[:, interior_outside] == 0)
    assert np.all(diag[:, ~interior_outside] > 0)
    fixed = state.fixed.copy()
    fixed[:, outside] += rng.normal(size=fixed[:, outside].shape)
    changed = state.replace(fixed=fixed)
    ev2 = evaluate(changed, u)
    v = free_random(g, rng)
    assert ev2.total == ev.total and ev2.energy == ev.energy
    assert np.array_equal(gradient(changed, ev2), gradient(state, ev))
    assert np.array_equal(hvp(changed, ev2, v), hvp(state, ev, v))
    assert np.array_equal(gn_diagonal(changed, ev2), gn_diagonal(state, ev))


def test_gaussian_path_equals_unit_weights_bitwise():
    """C04 gate: the Gaussian path (weights=None) equals Student-t weights w = 1 bitwise (energy, gradient,
    HVP, GN diagonal, c_dat)."""
    g = aniso()
    gauss = make_state(g, seed=3)
    ones = gauss.replace(weights=np.ones((3,)+g.shape))
    rng = np.random.default_rng(3)
    u, v = feasible_field(g, rng), free_random(g, rng)
    a, b = evaluate(gauss, u), evaluate(ones, u)
    assert np.array_equal(gauss.beta, ones.beta)
    assert a.total == b.total and a.energies == b.energies
    assert np.array_equal(gradient(gauss, a), gradient(ones, b))
    assert np.array_equal(hvp(gauss, a, v), hvp(ones, b, v))
    assert np.array_equal(gn_diagonal(gauss, a), gn_diagonal(ones, b))
    assert c_dat(gauss, a) == c_dat(ones, b)


def test_sampling_domain_and_nonpositive_gauss_jd():
    """Out of the sampling domain the channel block (and evaluate) returns INVALID_DOMAIN, never an
    extrapolated value; a floating J_q <= 0 makes the log-volume block return NONPOSITIVE_GAUSS_JD (no large
    number) and its kernels refuse."""
    g = aniso()
    state = make_state(g)
    u = np.zeros(g.shape+(3,))
    u[2, 2, 1, 0] = -1.5*g.spacing[0]          # interior node at x index 1 pushed past x0
    for result in (channel_point(state, u), evaluate(state, u)):
        assert isinstance(result, InvalidTrial) and result.reason == "INVALID_DOMAIN"
    big = aniso(nx=7, ny=7, nz=7)
    state = make_state(big)
    u = np.zeros(big.shape+(3,))
    u[3, 2, 3, 2] = u[3, 3, 3, 2] = -1.5*big.spacing[2]     # B3-2 construction with beta = 1.5 > 1.127
    for result in (gauss_value(LogVolume, state, u), evaluate(state, u)):
        assert isinstance(result, InvalidTrial) and result.reason == NONPOSITIVE_GAUSS_JD and result.invalid_count > 0
    geo = gauss_geometry(u, state.ops)
    scale = LogVolume.scale(state, slice(0, big.n_cells))
    for call in (lambda: LogVolume.residual(scale, geo), lambda: LogVolume.cotangent(scale, geo, geo.J),
                 lambda: LogVolume.jvp(scale, geo, geo.J, None, None)):
        with pytest.raises(InputValidationError, match="J_q <= 0"):
            call()


# ----------------------------------------------------------------------------------------------
# C05 prior gates
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("matrix", [((1.1, .05, -.02), (.03, .95, .04), (-.01, .02, 1.2)),
                                    ((.8, .1, 0.), (-.05, .9, .02), (.01, -.03, .85))])
def test_affine_logvol_energy(matrix, record_property):
    """C05 gate: affine u = (A - I) x + b (faces nonzero, geometry via gauss_geometry(allow_nonzero_faces))
    gives J = det A and 1/2 ||r_J||^2 = 1/2 <1/tau^2>_Q (log det A)^2 to 1e-13 relative, with <1/tau^2>_Q from
    a dense Q1 basis."""
    g = aniso()
    state = make_state(g, seed=6)
    A = np.array(matrix)
    X = g.identity()
    u = np.einsum("rj,...j->...r", A-np.eye(3), X)+np.array([.01, -.02, .015])
    tile = slice(0, g.n_cells)
    geo = gauss_geometry(u, state.ops, tile=tile, allow_nonzero_faces=True)
    r = LogVolume.residual(LogVolume.scale(state, tile), geo)
    energy = .5*float(np.sum(r*r, dtype=np.longdouble))
    assert relmax(geo.J, np.full(geo.J.shape, np.linalg.det(A))) <= 1e-14
    dense = dense_model(state, np.zeros(g.shape+(3,)))
    inverse_tau2 = float(np.sum(dense["wq"]/(state.omega0*dense["tau2"]), dtype=np.longdouble))
    expected = .5*inverse_tau2*np.log(np.linalg.det(A))**2
    error = abs(energy-expected)/expected
    record_property("affine_logvol_relative_error", error)
    assert error <= 1e-13


def test_checkerboard_priors_positive():
    """C05 gate (B8 unit part): an alternating +-eps interior displacement has Pi_J > 0 and Pi_D > 0."""
    g = aniso()
    state = make_state(g)
    sign = (-1.)**np.indices(g.shape).sum(axis=0)
    u = np.zeros(g.shape+(3,))
    u[1:-1, 1:-1, 1:-1] = (.01*min(g.spacing)*sign[1:-1, 1:-1, 1:-1])[..., None]*[1., -.5, .25]
    ev = evaluate(state, u)
    assert ev.energies["logvol"] > 0 and ev.energies["diffusion"] > 0


def test_tau2_at_gauss_points():
    """C05 gate: sum_k Ptilde_T^k(x_q) = 1 to 1e-12; min tau_k^2 <= tau_q^2 <= max tau_k^2; tau_q^2 (one
    interpolation of sum_k Ptilde^k tau_k^2) equals sum_k tau_k^2 Ptilde^k(x_q) of C05-2 to 1e-15 relative;
    tau_q^2 reads only the tissue layer: other likelihood channels leave it bitwise unchanged."""
    g = aniso()
    state = make_state(g, seed=8)
    tile = slice(0, g.n_cells)
    ops = state.ops
    classes = [tile_interpolate(np.ascontiguousarray(p), ops, tile=tile) for p in state.tissue]
    assert np.max(np.abs(sum(classes)-1)) <= 1e-12
    tau2 = LogVolume.tau2(state, tile)
    contract = sum(t*p for t, p in zip(state.tau2, classes))
    assert relmax(tau2, contract) <= 1e-15
    lo, hi = min(TAU2), max(TAU2)
    assert tau2.min() >= lo*(1-1e-15) and tau2.max() <= hi*(1+1e-15)
    rng = np.random.default_rng(0)
    other_moving = rng.normal(size=(2,)+g.shape)
    other = make_state(g, seed=8, moving=other_moving, fixed=other_moving+1, sigma2=(.3, .7), tissue=state.tissue)
    assert np.array_equal(LogVolume.scale(other, tile), LogVolume.scale(state, tile))
    assert np.array_equal(other.tau2_nodes, state.tau2_nodes)
