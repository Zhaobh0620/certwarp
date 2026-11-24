"""sigma_domain in the model (work package SWLI; contract C04-6 gates 1 (ModelState part), 2 (numpy), 6, 7 (model part),
9 (iii), 15 (theta bound with K_T); draft SWL 2026-10-31).  Synthetic data only."""
import hashlib

import numpy as np
import pytest

from jd_curl.three_d.boundary import inject_free
from jd_curl.validation import InputValidationError

from bayesreg import model as M
from bayesreg.model import ModelState, channel_point, support_level, theta_lower_bound
from bayesreg.model.blocks import SPLINE_CHUNK, channel_cdat_weight
from bayesreg.model.support import SUPPORT_RULE, check_theta, check_theta_coarse, resolve_sigma_domain, v_sha256
from bayesreg.solve.stopping import gf_b

from . import _swl_cases as cases


def field(grid, seed=4, scale=.08):
    rng = np.random.default_rng(seed)
    return inject_free(scale*min(grid.spacing)*rng.normal(size=3*grid.n_free_scalar), grid)


# ---- gate 1 (ModelState part) and the field rules ------------------------------------------------------------

def test_brain_state_holds_no_v_and_keeps_the_broadcast_view():
    _, state, _, _ = cases.template()
    assert state.sigma_domain == "brain" and state.support_v is None and state.support_threshold is None
    assert state.support_empty == () and state.support_v_sha256 is None and not state.has_support
    assert state.weights is None and state.weights_masked.strides == (0, 0)
    assert "sigma_domain" not in state.provenance() and state.sigma2_record() == state.sigma2.tolist()
    with pytest.raises(InputValidationError):
        state.replace(support_threshold=.05)
    with pytest.raises(InputValidationError):
        state.replace(support_v=np.zeros((3, state.node_index.size), dtype=np.uint8))


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_support_state_is_gaussian_with_v_as_its_own_field(domain):
    grid, state, _, _ = cases.template()
    level, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    assert st.weights is None and st.weights_masked.strides == (0, 0)
    prov = st.provenance()
    assert prov["noise"] == "gaussian" and prov["sigma_domain"] == domain and prov["support_rule"] == SUPPORT_RULE
    assert st.support_v.dtype == np.uint8 and st.support_v.shape == (3, st.node_index.size)
    assert not st.support_v.flags.writeable and st.support_v_sha256 == level.v_sha256 == v_sha256(level.v)
    assert st.support_empty == (3,) and st.sigma2_record()[3] is None
    # v enters beta (exact): beta^2 of a structure channel is the brain beta^2 times v
    brain = state.replace(sigma2=st.sigma2)
    for c in (1, 2, 3):
        assert np.array_equal(st.beta[c], brain.beta[c]*st.support_v[c-1])
    assert np.array_equal(st.beta[0], brain.beta[0])


def test_v_field_validation():
    grid, state, _, _ = cases.template()
    n = state.node_index.size
    good = np.zeros((3, n), dtype=np.uint8)
    for bad in (np.zeros((3, n)), np.zeros((2, n), dtype=np.uint8), np.full((3, n), 2, dtype=np.uint8), None):
        with pytest.raises(InputValidationError):
            state.replace(sigma_domain="support", support_threshold=.05, support_v=bad)
    with pytest.raises(InputValidationError):
        state.replace(sigma_domain="other")
    st = state.replace(sigma_domain="support", support_threshold=.05, support_v=good)
    assert st.support_empty == (1, 2, 3)


def test_theta_checks_two_steps_and_k():
    for bad in (0., 1., -.1, 1.5, np.nan, np.inf, "x", True):
        with pytest.raises(InputValidationError):
            check_theta_coarse(bad)
    assert check_theta_coarse(.05) == .05
    # gate 15: the floor bound uses K (K_T under channel_source tissue); eps/(1 + (K+1) eps) decreases with K
    low3, low13 = theta_lower_bound(3), theta_lower_bound(13)
    assert low3 > low13
    mid = (low3+low13)/2
    with pytest.raises(InputValidationError):
        check_theta(mid, 3)
    assert check_theta(mid, 13) == mid
    with pytest.raises(InputValidationError):
        check_theta(low3, 3)                               # at the bound: refused (the support would be the mask)
    assert check_theta(np.nextafter(low3, 1), 3) > low3
    assert resolve_sigma_domain() == ("brain", None)
    assert resolve_sigma_domain("support") == ("support", .05)
    with pytest.raises(InputValidationError):
        resolve_sigma_domain("brain", .05)
    # the model checks with K = n_channels - 1: a 4-channel state (K = 3) refuses mid
    grid, state, _, _ = cases.template()
    with pytest.raises(InputValidationError):
        state.replace(sigma_domain="support", support_threshold=mid,
                      support_v=np.zeros((3, state.node_index.size), dtype=np.uint8))


# ---- gate 2 (numpy): "v as a weight" reference ------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_v_path_equals_the_weight_reference_bitwise(domain):
    grid, state, _, _ = cases.template()
    _, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    ref = cases.weight_reference(st)
    assert np.array_equal(st.beta, ref.beta)
    for u in (np.zeros(grid.shape+(3,)), field(grid)):
        a, b = M.evaluate(st, u), M.evaluate(ref, u)
        assert a.total == b.total and a.energy == b.energy and a.energies == b.energies
        assert np.array_equal(M.gradient(st, a), M.gradient(ref, b))
        v = field(grid, seed=9)
        assert np.array_equal(M.hvp(st, a, v), M.hvp(ref, b, v))
        assert np.array_equal(M.gn_diagonal(st, a), M.gn_diagonal(ref, b))
        assert M.c_dat(st, a) == M.c_dat(ref, b)
        assert np.array_equal(channel_cdat_weight(st), channel_cdat_weight(ref))
    g0 = gf_b(M.gradient(st, M.evaluate(st, np.zeros(grid.shape+(3,)))), grid.omega, st.omega0)
    g0_ref = gf_b(M.gradient(ref, M.evaluate(ref, np.zeros(grid.shape+(3,)))), grid.omega, ref.omega0)
    assert g0 == g0_ref


def test_brain_c_dat_weight_is_the_old_expression():
    grid, state, _, _ = cases.template()
    st = state.replace(sigma2=np.array([.04, .09, .02, .5]))
    assert np.array_equal(channel_cdat_weight(st), st.weights_masked/st.sigma2[:, None])


# ---- gate 6: threshold boundary -----------------------------------------------------------------------------------

def test_threshold_boundary():
    grid, state, _, _ = cases.template()
    theta = .05
    t = float(np.sqrt(np.float64(theta)))
    n = state.node_index.size
    fixed = np.array(state.fixed)
    nodes = state.node_index[:6]
    values = [t, np.nextafter(t, 2.), 0., 0., 0., 0.]
    flat = fixed.reshape(fixed.shape[0], -1)
    flat[1, :] = 0.
    for node, value in zip(nodes, values):
        flat[1, node] = value
    outside = np.flatnonzero(~state.mask.ravel())[0]
    flat[1, outside] = .9                                # a mask-free node above t: never in the support
    st = ModelState(grid=grid, omega0=state.omega0, fixed=fixed, samplers=state.samplers, mask=state.mask,
                    sigma2=state.sigma2, alpha=state.alpha, nu=state.nu, tau2=state.tau2, tissue=state.tissue,
                    cpu_threads=1)
    rho_1 = np.zeros(n)
    rho_1[:6] = [0., -.5, t, np.nextafter(t, 2.), -.5, 0.]  # Ghat = t (out), nextafter (in), negative (out)
    rows = {c: np.zeros(n) for c in range(4)}
    rows[1] = rho_1
    level = support_level(st, np.zeros(grid.shape+(3,)), theta, sigma_domain="support",
                          rows=lambda c: rows[c].copy())
    assert level.v[0, :6].tolist() == [0, 1, 0, 1, 0, 0]
    assert level.support_nodes[0] == 2 and level.fixed_only[0] == 1 and level.moving_only[0] == 1
    assert level.near_threshold[0] == 3                     # |Ghat - t| <= 1e-12 at nodes 0, 2, 3
    full = np.zeros(int(np.prod(grid.shape)), dtype=np.uint8)
    full[st.node_index] = level.v[0]
    assert full[outside] == 0 and int(full.sum()) == 2
    assert level.t_theta == t


# ---- gate 7 (model part): empty support -------------------------------------------------------------------------

@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_empty_support_channel_contributes_exactly_zero(domain):
    grid, state, _, _ = cases.template()
    level, st = cases.with_support(state, np.zeros(grid.shape+(3,)), sigma_domain=domain)
    assert level.empty == (3,) and level.support_nodes[2] == 0 and level.n_c[2] == 0.
    assert level.record()["empty_support_channels"] == [3]
    assert np.all(st.beta[3] == 0.)
    u = field(grid)
    ev = M.evaluate(st, u)
    assert ev.energies["channel"][3] == 0. and np.isfinite(ev.energy)
    assert np.all(ev.channel.r[3] == 0.)
    g = M.gradient(st, ev, blocks=("channel",))
    assert np.all(np.isfinite(g))
    # the same state without channel 3 (3 channels, same v rows) gives the same channel gradient
    keep = [0, 1, 2]
    three = ModelState(grid=grid, omega0=st.omega0, fixed=st.fixed[keep], samplers=tuple(st.samplers[c] for c in keep),
                       mask=st.mask, sigma2=st.sigma2[keep], alpha=st.alpha, nu=st.nu, tau2=st.tau2, tissue=st.tissue,
                       cpu_threads=1, sigma_domain=domain, support_threshold=.05, support_v=np.array(st.support_v[:2]))
    ev3 = M.evaluate(three, u)
    assert np.array_equal(M.gradient(three, ev3, blocks=("channel",)), g)
    assert np.array_equal(M.gn_diagonal(three, ev3), M.gn_diagonal(st, ev))


# ---- gate 9 (iii): the chunked value-only CPU path equals channel_point's rho ------------------------------------

@pytest.mark.parametrize("threads", [1, 4])
def test_chunked_cpu_pass_equals_channel_point(threads):
    grid, state, _, _ = cases.template(n=43, h=1., cpu_threads=threads)
    assert state.node_index.size > SPLINE_CHUNK                  # several chunks
    u = field(grid, scale=.05)
    point = channel_point(state, u)
    a = support_level(state, u, .05, sigma_domain="support_shared")
    b = support_level(state, u, .05, sigma_domain="support_shared", rows=lambda c: np.array(point.rho[c]))
    assert a.v_sha256 == b.v_sha256 and np.array_equal(a.v, b.v)
    assert a.n_c_ld == b.n_c_ld and a.s_ld == b.s_ld and a.n_q_ld == b.n_q_ld
    assert a.support_nodes == b.support_nodes and a.near_threshold == b.near_threshold
    assert hashlib.sha256(a.v.tobytes()).hexdigest() == a.v_sha256


def test_masked_node_weights_are_bitwise_the_grid_weights():
    from bayesreg.model.support import node_weights_masked
    for n, h in ((12, 2.), (43, 1.)):
        grid, state, _, _ = cases.template(n=n, h=h)
        assert np.array_equal(node_weights_masked(state), grid.node_weights().ravel()[state.node_index])
    from ._helpers import aniso, make_state
    state = make_state(aniso())
    assert np.array_equal(node_weights_masked(state), state.grid.node_weights().ravel()[state.node_index])


def test_support_pass_threads_are_bitwise_neutral():
    """Review SWLI minor 6: the chunks of a channel run in state.cpu_threads threads; 1 and 4 threads give the same
    SupportLevel bit for bit (several chunks per channel); v is hashed through its buffer (minor 5)."""
    levels = {}
    for threads in (1, 4):
        grid, state, _, _ = cases.template(n=43, h=1., cpu_threads=threads)
        assert state.node_index.size > SPLINE_CHUNK
        levels[threads] = support_level(state, field(grid, scale=.05), .05, sigma_domain="support")
    a, b = levels[1], levels[4]
    assert a.v.tobytes() == b.v.tobytes() and a.v_sha256 == b.v_sha256
    assert a.n_q_ld == b.n_q_ld and a.n_c_ld == b.n_c_ld and a.s_ld == b.s_ld
    assert (a.support_nodes, a.fixed_only, a.moving_only, a.near_threshold) == (
        b.support_nodes, b.fixed_only, b.moving_only, b.near_threshold)
    assert a.v_sha256 == hashlib.sha256(a.v.tobytes()).hexdigest() == v_sha256(np.array(a.v))
    from bayesreg.model.support import cpu_channel_rho
    u = field(grid, scale=.05)
    for c in range(state.n_channels):
        assert np.array_equal(cpu_channel_rho(state, u, c, threads=1), cpu_channel_rho(state, u, c, threads=4))
