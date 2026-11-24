"""Contract C14-7 and the prior side of C15-10 (logvol_points = "gauss27+closed27", Simpson; work package IMPL27b)
against the independent dense oracles: tests/test_bayesreg/model/_helpers (Gauss27 blocks) and the closed27 rows of
tests/test_bayesreg/model/test_logvol_closed27 (dense_closed27: scalar Q1 formula at {0, 1/2, 1}^3, Simpson weights,
shares theta_G, theta_S; nothing in either calls bayesreg's geometry).  The gates of tests/test_bayesreg/infer
(test_laplace, test_prior_side) applied to the closed27 block on the 6^3 case (3 N_int = 192):
  * perturbation_rhs = dense sum_i K_i^T eps_i with the C14-7 noise order (per tile [diffusion explicit], logvol, curl,
    logvol_closed27; the Gauss log-volume rows carry sqrt(theta_G)) -- 1e-12 (explicit), 1e-10 (DST square root);
    beta_S = 1 and 3, one tile and tiles of 37 cells;
  * LaplaceSystem.hvp = dense H with K_{J,S}^T K_{J,S} (1e-12); the system record names the point set and noise order;
  * prior_perturbation_rhs (noise order per tile logvol, curl, logvol_closed27) and PriorSystem.hvp = dense Pi_0 with
    the closed27 block (C15-10); the B17-5 spectral bound with a_x = alpha + 1/tau_x^2 over the 53 sample points
    (Simpson weights);
  * posterior_maps: Var_s(log J) at the closed27 points (logj_var_closed27, ddof 1) equals the dense log J of the
    passed samples (1e-12); sigma_logJ stays on the Gauss points (C14-5); a passed sample with a floating J <= 0 only
    at a closed27 point is excluded under C05-2b (NONPOSITIVE_GAUSS_JD, closed27) and included under gauss27;
  * the noise-order records (MIN1, review IMPL27b minor 1): qc.json config.noise_order and laplace_rng.noise_order are
    the state's (lp.noise_order_of, the source of laplace_system.noise_order), with logvol_closed27 under C05-2b and
    NOISE_ORDER, as before, under gauss27; the prior side's sample records the prior order.
Certificates are scripted (all pass unless stated) with the real C07 configuration.  No data/, no GPU.
"""
import json

import numpy as np
import pytest
import scipy.linalg

from jd_curl.three_d.boundary import inject_free, restrict_free

from bayesreg import model
from bayesreg.infer import laplace as lp

from . import _cases_gauss27 as cs          # DEF: the base states are C05-2 ("gauss27", the default before DEF)
from ..model._gauss27 import make_state
from ..model._helpers import dense_model, free_random, relmax
from ..model.test_logvol_closed27 import dense_closed27, thetas, w1_field, w_grid

CLOSED = "gauss27+closed27"


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-300))


def closed_cube(share=1., tile_size=8192, factor=1., **kwargs):
    grid, state = cs.cube6(tile_size=tile_size, factor=factor, **kwargs)
    return grid, state.replace(logvol_points=CLOSED, logvol_closed27_share=share)


def dense_closed_blocks(state, u, names=None):
    """Dense K_i (rows x 3 N_int) of the C05-2b model at u: channel, diffusion, logvol (Gauss, times sqrt(theta_G)),
    curl, logvol_closed27; and H = sum K_i^T K_i over `names` (all by default)."""
    K, _ = cs.dense_blocks(state, u)
    K = dict(K, logvol=np.sqrt(thetas(state)[0])*K["logvol"])
    K["logvol_closed27"] = dense_closed27(state, u)[1]
    names = tuple(K) if names is None else names
    return {n: K[n] for n in names}, sum(K[n].T@K[n] for n in names)


def expected_layout(state, grid, tile_size, *, diffusion="dst_sqrt", prior=False):
    out = [] if prior else [("channel", (state.n_channels, state.node_index.size))]
    if diffusion == "dst_sqrt":
        out.append(("diffusion", (192,)))
    for cells in [min(tile_size, grid.n_cells-s) for s in range(0, grid.n_cells, tile_size)]:
        if diffusion == "explicit":
            out.append(("diffusion", (cells, 27, 3, 3)))
        out += [("logvol", (cells, 27)), ("curl", (cells, 27, 3)), ("logvol_closed27", (cells, 27))]
    return out


def dst_part(K, eps):
    """The PSD square root of K_D^T K_D applied to eps (dense eigh), as test_laplace."""
    lam, Q = np.linalg.eigh(K["diffusion"].T@K["diffusion"])
    return Q@(np.sqrt(np.clip(lam, 0, None))*(Q.T@eps))


@pytest.mark.parametrize("diffusion", ["dst_sqrt", "explicit"])
@pytest.mark.parametrize("tile_size, share", [(8192, 1.), (37, 3.)])
def test_rhs_matches_dense_oracle_and_noise_layout(diffusion, tile_size, share, record_property):
    grid, state = closed_cube(share, tile_size, student_t=tile_size == 37)
    u = cs.u_hat_of(grid)
    ev = model.evaluate(state, u)
    draw = cs.RecordingDraw(5)
    b = lp.perturbation_rhs(state, ev, draw, diffusion=diffusion)
    assert [(name, shape) for name, shape, _ in draw.calls] == expected_layout(state, grid, tile_size,
                                                                             diffusion=diffusion)
    K, _ = dense_closed_blocks(state, u)
    n_q = grid.n_cells*27
    assert draw.concatenated("logvol_closed27").size == n_q == K["logvol_closed27"].shape[0]
    dense = sum(K[name].T@draw.concatenated(name) for name in ("channel", "logvol", "curl", "logvol_closed27"))
    if diffusion == "explicit":
        dense = dense+K["diffusion"].T@draw.concatenated("diffusion")
    else:
        dense = dense+dst_part(K, draw.concatenated("diffusion"))
    err = rel(b, dense)
    record_property("rhs_rel_err", err)
    assert err <= (1e-12 if diffusion == "explicit" else 1e-10)
    # power: the closed27 part is a sizeable share of b
    assert rel(b, dense-K["logvol_closed27"].T@draw.concatenated("logvol_closed27")) > 1e-3


def test_system_h_contains_the_closed27_block(record_property):
    """C14-7: H = sum_i K_i^T K_i with K_{J,S}^T K_{J,S} (dense, 1e-12); without the closed27 block the dense H differs."""
    grid, state = closed_cube(3.)
    u = cs.u_hat_of(grid)
    system = lp.LaplaceSystem(state, u)
    _, H = dense_closed_blocks(state, u)
    H_sys = cs.dense_operator(system.hvp, 192)
    err = float(np.max(np.abs(H_sys-H))/np.max(np.abs(H)))
    record_property("H_vs_dense", err)
    assert err <= 1e-12
    _, H_no = dense_closed_blocks(state, u, ("channel", "diffusion", "logvol", "curl"))
    assert float(np.max(np.abs(H_sys-H_no))/np.max(np.abs(H))) > 1e-3
    record = system.record()
    assert record["logvol_points"] == CLOSED and record["logvol_closed27_share"] == 3.
    assert record["noise_order"] == list(lp.NOISE_ORDER_CLOSED27)
    assert record["logvol_theta"] == [1.0/(1.0+3.), 3./(1.0+3.)]
    plain_grid, plain = cs.cube6()
    assert "logvol_points" not in lp.LaplaceSystem(plain, cs.u_hat_of(plain_grid)).record()    # gauss27 unchanged


@pytest.mark.parametrize("tile_size", [8192, 37])
def test_prior_rhs_and_pi0_with_the_closed27_block(tile_size, record_property):
    """C15-10 prior side: prior_perturbation_rhs = dense sum_i K_{i,0}^T eps_i (noise per tile logvol, curl,
    logvol_closed27; 1e-10 with the DST root); PriorSystem.hvp = dense Pi_0 including K_{J,S,0}^T K_{J,S,0} (1e-12)."""
    grid, state = closed_cube(1., tile_size)
    zero = np.zeros(grid.shape+(3,))
    ev = model.evaluate(state, zero)
    draw = cs.RecordingDraw(5)
    b = lp.prior_perturbation_rhs(state, ev, draw)
    assert [(name, shape) for name, shape, _ in draw.calls] == expected_layout(state, grid, tile_size, prior=True)
    K, Pi0 = dense_closed_blocks(state, zero, ("diffusion", "logvol", "curl", "logvol_closed27"))
    dense = sum(K[name].T@draw.concatenated(name) for name in ("logvol", "curl", "logvol_closed27")) + \
        dst_part(K, draw.concatenated("diffusion"))
    err = rel(b, dense)
    system = lp.PriorSystem(state)
    err_pi0 = float(np.max(np.abs(cs.dense_operator(system.hvp, 192)-Pi0))/np.max(np.abs(Pi0)))
    record_property("prior_rhs_rel_err", err)
    record_property("Pi0_vs_dense", err_pi0)
    assert err <= 1e-10 and err_pi0 <= 1e-12
    assert system.record()["noise_order"] == list(lp.PRIOR_NOISE_ORDER_CLOSED27)
    items = lp.sample_perturbations(state, zero, 11, 4, system=system)
    assert all(item.converged for item in items)


def test_pi0_spectral_bounds_b17_5_over_53_points(record_property):
    """B17-5 under C05-2b (Simpson): min(a_min, b) K_Q <= omega0 Pi_0 <= max(a_max, b) K_Q with a_x = alpha + 1/tau_x^2
    over the Gauss and the closed27 points, b = alpha + nu (02_理论 B17')."""
    for factor, nu, share in ((1., None, 1.), (1., 40., 3.), (30., None, 1.)):
        grid, state = closed_cube(share, factor=factor)
        if nu is not None:
            state = state.replace(nu=nu)
        zero = np.zeros(grid.shape+(3,))
        K, Pi0 = dense_closed_blocks(state, zero, ("diffusion", "logvol", "curl", "logvol_closed27"))
        KQ = state.omega0/state.alpha*(K["diffusion"].T@K["diffusion"])
        tau2 = np.concatenate((dense_model(state, zero)["tau2"], dense_closed27(state, zero)[3]))
        a = state.alpha+1/tau2
        b = state.alpha+state.nu
        lo, hi = min(a.min(), b), max(a.max(), b)
        lam = scipy.linalg.eigh(state.omega0*Pi0, KQ, eigvals_only=True)
        record_property(f"factor{factor}_nu{nu}_share{share}", dict(eig=[float(lam.min()), float(lam.max())],
                                                                     bound=[lo, hi]))
        assert lam.min() >= lo*(1-1e-10) and lam.max() <= hi*(1+1e-10)


def test_posterior_maps_closed27_variance_and_gauss_maps(record_property):
    """keep_gauss_variance under C05-2b: logj_var_closed27 = ddof-1 variance of the dense closed27 log J over the
    passed samples (1e-12); the Gauss variance and sigma_logJ are those of the Gauss points (C14-5 unchanged); the
    qc carries the point set; a gauss27 state has logj_var_closed27 None and no new qc key."""
    grid, state = closed_cube(1., factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 31, 6))
    assert all(s.converged for s in samples)
    passes = [True, False, True, True, True, True]
    maps = lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate(passes), keep_gauss_variance=True)
    used = [s for s, ok in zip(samples, passes) if ok]
    assert maps.n_passed == len(used) == 5
    logj_s = np.stack([np.log(dense_closed27(state, u+inject_free(s.delta, grid))[2]) for s in used])
    var_s = np.var(logj_s, axis=0, ddof=1).reshape(grid.n_cells, 27)
    errors = dict(closed27=relmax(maps.logj_var_closed27, var_s))
    from .test_laplace import _manual_logj
    logj = _manual_logj(u, used, grid, state)
    std = np.std(logj, axis=0, ddof=1)
    errors["gauss_var"] = relmax(maps.logj_var_gauss, std**2)
    errors["sigma_logj"] = relmax(maps.sigma_logj, (std@(state.ops.weights/grid.omega)).reshape(grid.cell_shape))
    record_property("errors", errors)
    assert max(errors.values()) <= 1e-12, errors
    assert maps.qc["logvol_points"] == CLOSED and "logvol_closed27_note" in maps.qc
    plain_grid, plain = cs.cube6(factor=30.)
    plain_maps = lp.posterior_maps(plain, u, samples, backend=cs.ScriptedCertificate(passes), keep_gauss_variance=True)
    assert plain_maps.logj_var_closed27 is None and "logvol_points" not in plain_maps.qc
    no_var = lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate(passes))
    assert no_var.logj_var_closed27 is None


def test_closed27_nonpositive_sample_is_excluded():
    """C07-5 (iii) under C05-2b: a sample that passes the (scripted) certificate with every Gauss J > 0 but a floating
    J <= 0 at a closed27 point (W1(17/20)) is excluded from sigma_logJ with NONPOSITIVE_GAUSS_JD naming closed27; the
    gauss27 state includes it."""
    grid = w_grid((1., 1., 1.))
    base = make_state(grid, seed=11)
    state = base.replace(logvol_points=CLOSED)
    zero = np.zeros(grid.shape+(3,))
    delta = restrict_free(w1_field(grid, 17/20), grid)
    items = [lp.Perturbation(0, delta, 1, 0., True, 1., 0.)]
    out = lp.posterior_maps(state, zero, items, backend=cs.ScriptedCertificate([True]))
    assert out.n_cert_passed == 1 and out.n_passed == 0
    assert out.samples[0]["excluded"] == "logj: NONPOSITIVE_GAUSS_JD (closed27 points, C05-2b)"
    plain = lp.posterior_maps(base, zero, items, backend=cs.ScriptedCertificate([True]))
    assert plain.n_passed == 1 and plain.samples[0]["excluded"] is None


def test_uniform_and_corners_refused_everywhere():
    grid, state = closed_cube()
    u = cs.u_hat_of(grid)
    uniform = state.replace(closed27_weights="uniform")
    for call in (lambda: lp.LaplaceSystem(uniform, u), lambda: lp.PriorSystem(uniform),
                 lambda: lp.posterior_maps(uniform, u, [], backend=cs.ScriptedCertificate([])),
                 lambda: lp.prior_perturbation_rhs(uniform, model.evaluate(uniform, np.zeros(grid.shape+(3,))),
                                                   lambda name, shape: np.zeros(shape))):
        with pytest.raises(Exception, match="scope limit"):
            call()
    corners = cs.cube6()[1].replace(logvol_points="gauss27+corners")
    with pytest.raises(Exception, match="corners"):
        lp.LaplaceSystem(corners, u)
    v = free_random(grid, np.random.default_rng(0))
    assert np.all(np.isfinite(lp.LaplaceSystem(state, u).hvp(v)))


@pytest.mark.parametrize("points", ["gauss27", CLOSED])
def test_noise_order_records_follow_the_state(points, tmp_path):
    """MIN1 (review IMPL27b minor 1): run_laplace's qc.json records the noise order of the state's point set in
    config.noise_order and laplace_rng.noise_order, the same as laplace_system.noise_order (C14-7: logvol_closed27 last
    in every tile); under gauss27 both are NOISE_ORDER, the records before MIN1, and laplace_system has no noise_order
    (unchanged).  Without a state (EBConfig.record) the records keep NOISE_ORDER.  sample_perturbations with a
    PriorSystem records the prior order (C15-9 / C15-10)."""
    grid, state = cs.cube6(factor=30.)
    if points == CLOSED:
        state = state.replace(logvol_points=CLOSED)
    closed = points == CLOSED
    expected = list(lp.NOISE_ORDER_CLOSED27 if closed else lp.NOISE_ORDER)
    assert lp.noise_order_of(state) == expected
    u = cs.u_hat_of(grid)
    config = lp.LaplaceConfig(n_samples=2)
    seed = lp.seed_from_run_id("min1/noise_order")
    lp.run_laplace(tmp_path, state, u, seed=seed, config=config, backend=cs.ScriptedCertificate([True]*2))
    qc = json.loads((tmp_path/"qc.json").read_text())
    assert qc["config"]["noise_order"] == expected and qc["laplace_rng"]["noise_order"] == expected
    if closed:
        assert qc["laplace_system"]["noise_order"] == expected and "logvol_closed27" in expected[-1]
        assert "logvol_closed27" not in lp.NOISE_ORDER[-1]
    else:
        assert "noise_order" not in qc["laplace_system"]
        assert config.record(state) == config.record() and lp.rng_record(seed, state) == lp.rng_record(seed)
    assert config.record()["noise_order"] == list(lp.NOISE_ORDER) == lp.rng_record(seed)["noise_order"]
    prior_expected = list(lp.PRIOR_NOISE_ORDER_CLOSED27 if closed else lp.PRIOR_NOISE_ORDER)
    zero = np.zeros(grid.shape+(3,))
    system = lp.PriorSystem(state, config=config)
    items = lp.sample_perturbations(state, zero, seed, 2, system=system)
    assert items.rng["noise_order"] == prior_expected == system.record()["noise_order"]
    assert lp.sample_perturbations(state, u, seed, 2, system=lp.LaplaceSystem(state, u, config=config)).rng[
        "noise_order"] == expected


def test_sample_perturbations_records_the_systems_state():
    """FUP1 (review MIN1 minor 3): sample_perturbations takes the rng record (noise order, tile size) from
    system.host, the state the system's rhs draws with, not from its `state` argument: a PriorSystem built on the
    closed27 state records the closed27 prior order and that state's tile size even when called with the gauss27 state
    (before FUP1 the record named the gauss27 order while the draws followed closed27).  The draws themselves are
    the system's either way (same deltas)."""
    grid, base = cs.cube6(factor=30.)
    plain, closed = base.replace(tile_size=64), base.replace(logvol_points=CLOSED)
    zero = np.zeros(grid.shape+(3,))
    config = lp.LaplaceConfig(n_samples=2)
    system = lp.PriorSystem(closed, config=config)
    own = lp.sample_perturbations(closed, zero, 11, 2, system=system)
    other = lp.sample_perturbations(plain, zero, 11, 2, system=system)
    assert other.rng == own.rng
    assert other.rng["noise_order"] == list(lp.PRIOR_NOISE_ORDER_CLOSED27) == system.record()["noise_order"]
    assert other.rng["tile_size"] == closed.tile_size == system.record()["tile_size"] != plain.tile_size
    assert all(np.array_equal(a.delta, b.delta) for a, b in zip(own, other))
