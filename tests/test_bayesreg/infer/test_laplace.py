"""bayesreg.infer.laplace against contract C14 (perturb-and-MAP, PCG, posterior maps, pass rate, outputs)
and theory B5/B6 on small grids.  The statistical covariance gate (S = 1e5) is test_whitening.py.

The real-certificate tests trigger the numba compilation of certify_map (about 10 s, once per process); they
are marked `slow` (registered in conftest.py; deselect with -m "not slow").
"""
import json

import numpy as np
import pytest

from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.cell_q1 import cell_node_indices

from bayesreg import model
from bayesreg.geom import gauss_geometry, stiffness_apply
from bayesreg.infer import laplace as lp
from bayesreg.solve.backend import NumpyBackend

from . import _cases_gauss27 as cs          # DEF: the C05-2 model ("gauss27", the default before DEF), explicit
from ..model._helpers import aniso, free_random, relmax


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-300))


# ----------------------------------------------------------------------------------------------
# Configuration, seeds
# ----------------------------------------------------------------------------------------------

def test_config_defaults_are_contract_c14():
    config = lp.LaplaceConfig()
    assert (config.n_samples, config.rtol, config.maxiter) == (32, 1e-3, 200)
    assert (lp.S_NATIVE, lp.S_GROUP, lp.PASS_RATE_THRESHOLD) == (32, 16, .8)
    assert config.diffusion_factor == "dst_sqrt" and config.metric_mode == "principal"
    record = config.record()
    assert record["pcg"]["atol"] == 0. and "no proximal" in record["hessian"]
    for bad in (dict(rtol=0.), dict(rtol=1.), dict(maxiter=0), dict(diffusion_factor="chol"),
                dict(metric_mode="x"), dict(pass_rate_threshold=0.), dict(n_samples=0)):
        with pytest.raises(InputValidationError):
            lp.LaplaceConfig(**bad)


def test_seed_from_run_id_and_rng_record():
    a, b = lp.seed_from_run_id("L2R_val/0001_0002/bayesreg"), lp.seed_from_run_id("L2R_val/0001_0003/bayesreg")
    assert a == lp.seed_from_run_id("L2R_val/0001_0002/bayesreg") and a != b and 0 <= a < 2**63
    assert lp.rng_record(a)["seed"] == a and lp.rng_record(np.random.default_rng(0))["seed"] is None
    for record in (lp.rng_record(a), lp.rng_record(np.random.Generator(np.random.SFC64(0)))):
        assert record["numpy_version"] == np.__version__
    assert lp.rng_record(a)["bit_generator"] == "PCG64"
    assert lp.rng_record(np.random.Generator(np.random.SFC64(0)))["bit_generator"] == "SFC64"
    h = lp.hutchinson_seed_from_run_id("L2R_val/0001_0002/bayesreg")
    assert h == lp.seed_from_run_id("L2R_val/0001_0002/bayesreg/hutchinson") and h != a and 0 <= h < 2**63
    for bad in (-1, 1.5, True, "7"):
        with pytest.raises(InputValidationError):
            lp.rng_record(bad)
    with pytest.raises(InputValidationError):
        lp.hutchinson_seed_from_run_id("")


# ----------------------------------------------------------------------------------------------
# Square-root factor of the diffusion block, trace operators
# ----------------------------------------------------------------------------------------------

def test_stiffness_sqrt_squares_to_kq_and_is_symmetric(record_property):
    grid = aniso()
    rng = np.random.default_rng(0)
    v, w = free_random(grid, rng), free_random(grid, rng)
    half = lp.stiffness_sqrt_apply(lp.stiffness_sqrt_apply(v, grid), grid)
    err = rel(half, stiffness_apply(v, grid))
    sym = abs(w @ lp.stiffness_sqrt_apply(v, grid)-v @ lp.stiffness_sqrt_apply(w, grid))
    record_property("sqrt_square_rel_err", err)
    assert err <= 1e-12
    assert sym <= 1e-12*np.linalg.norm(v)*np.linalg.norm(w)*np.max(lp.stiffness_sqrt_spectrum(grid))


def test_dst_factor_gram_equals_dense_diffusion_block(record_property):
    """B_D B_D^T = K_D^T K_D (C14-2: the DST square root has the covariance of the explicit block)."""
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    K, _ = cs.dense_blocks(state, u)
    B = np.sqrt(state.alpha/state.omega0)*cs.dense_operator(lambda e: lp.stiffness_sqrt_apply(e, grid), 192)
    gram = K["diffusion"].T@K["diffusion"]
    err = float(np.max(np.abs(B@B.T-gram))/np.max(np.abs(gram)))
    record_property("BBt_vs_KDtKD", err)
    assert err <= 1e-12


def test_trace_operators_are_the_prior_blocks(record_property):
    grid = aniso()
    state = cs.scaled_state(grid)
    u = cs.u_hat_of(grid)
    ev = model.evaluate(state, u)
    v = free_random(grid, np.random.default_rng(4))
    curl = state.nu*lp.curl_gram_apply(state, v)
    diff = state.alpha*lp.stiffness_apply_scaled(state, v)
    e1 = relmax(curl, model.hvp(state, ev, v, blocks=("curl",)))
    e2 = relmax(diff, model.hvp(state, ev, v, blocks=("diffusion",)))
    record_property("curl_gram_err", e1)
    record_property("kq_err", e2)
    assert e1 <= 1e-12 and e2 <= 1e-12
    ops = lp.default_trace_operators(state)
    assert set(ops) == {"KQ_over_omega0", "CtWC_over_omega0"}
    assert np.array_equal(ops["CtWC_over_omega0"](v), lp.curl_gram_apply(state, v))


# ----------------------------------------------------------------------------------------------
# Right-hand side b = sum_i B_i eps_i
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("diffusion", ["dst_sqrt", "explicit"])
@pytest.mark.parametrize("tile_size, student_t", [(8192, False), (37, False), (37, True)])
def test_rhs_matches_dense_oracle_and_noise_layout(diffusion, tile_size, student_t, record_property):
    """Gaussian and Student-t (w_used) channel weights; the dense oracle carries the same weights."""
    grid, state = cs.cube6(tile_size=tile_size, student_t=student_t)
    assert (state.weights is not None) == student_t
    u = cs.u_hat_of(grid)
    ev = model.evaluate(state, u)
    draw = cs.RecordingDraw(5)
    b = lp.perturbation_rhs(state, ev, draw, diffusion=diffusion)
    K, _ = cs.dense_blocks(state, u)
    n_q = grid.n_cells*27
    tiles = [min(tile_size, grid.n_cells-s) for s in range(0, grid.n_cells, tile_size)]
    expected = [("channel", (state.n_channels, state.node_index.size))]
    if diffusion == "dst_sqrt":
        expected.append(("diffusion", (192,)))
    for cells in tiles:
        if diffusion == "explicit":
            expected.append(("diffusion", (cells, 27, 3, 3)))
        expected += [("logvol", (cells, 27)), ("curl", (cells, 27, 3))]
    assert [(name, shape) for name, shape, _ in draw.calls] == expected
    dims = {name: draw.concatenated(name).size for name in ("channel", "diffusion", "logvol", "curl")}
    assert dims == dict(channel=state.n_channels*state.node_index.size,
                        diffusion=192 if diffusion == "dst_sqrt" else 9*n_q, logvol=n_q, curl=3*n_q)
    dense = sum(K[name].T@draw.concatenated(name) for name in ("channel", "logvol", "curl"))
    if diffusion == "explicit":
        dense = dense+K["diffusion"].T@draw.concatenated("diffusion")
    else:
        # the PSD square root of K_D^T K_D = (alpha/omega0) K_Q, from the dense oracle (eigh)
        lam, Q = np.linalg.eigh(K["diffusion"].T@K["diffusion"])
        dense = dense+Q@(np.sqrt(np.clip(lam, 0, None))*(Q.T@draw.concatenated("diffusion")))
    err = rel(b, dense)
    record_property("rhs_rel_err", err)
    assert err <= (1e-12 if diffusion == "explicit" else 1e-10)


def test_rhs_rejects_bad_noise_and_foreign_evaluation():
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    ev = model.evaluate(state, u)
    with pytest.raises(InputValidationError):
        lp.perturbation_rhs(state, ev, lambda name, shape: np.zeros(shape[:-1] if len(shape) > 1 else (1,)))
    with pytest.raises(InputValidationError):
        lp.perturbation_rhs(state, ev, lambda name, shape: np.full(shape, np.nan))
    other = cs.scaled_state(grid, 2.)
    with pytest.raises(InputValidationError):
        lp.perturbation_rhs(other, ev, lambda name, shape: np.zeros(shape))


def test_system_has_no_proximal_terms_and_contract_metric(record_property):
    """C14-1: H = sum_i K_i^T K_i (dense oracle); C14-3: M with gamma = lam_b = 0, c_dat at u_hat."""
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    system = lp.LaplaceSystem(state, u)
    _, H = cs.dense_blocks(state, u)
    H_sys = cs.dense_operator(system.hvp, 192)
    err = float(np.max(np.abs(H_sys-H))/np.max(np.abs(H)))
    record_property("H_vs_dense", err)
    assert err <= 1e-12
    ev = model.evaluate(state, u)
    c = model.c_dat(state, ev)
    M = model.build_metric(state, c_dat=c, b_P=model.metric_bP(state), lam_b=0., gamma=0.)
    v = free_random(grid, np.random.default_rng(1))
    assert np.array_equal(system.precondition(v), M.solve(v))
    record = system.record()
    assert record["metric"]["gamma"] == 0. and record["metric"]["lam_b"] == 0. and record["c_dat"] == c
    with pytest.raises(InputValidationError):
        bad = u.copy()
        bad[2, 2, 2] = [0., 0., 50.]
        lp.LaplaceSystem(state, bad)


# ----------------------------------------------------------------------------------------------
# PCG
# ----------------------------------------------------------------------------------------------

def test_pcg_converges_to_contract_tolerance_with_b5_3_bound(record_property):
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    system = lp.LaplaceSystem(state, u)
    _, H = cs.dense_blocks(state, u)
    b = system.rhs(cs.RecordingDraw(8))
    result = system.solve(b)
    exact = np.linalg.solve(H, b)
    true = np.linalg.norm(b-H@result.x)/np.linalg.norm(b)
    bound = 1e-3*np.linalg.norm(b)/np.linalg.eigvalsh(H)[0]
    record_property("pcg_iterations", result.iterations)
    record_property("pcg_rel_residual", result.rel_residual)
    assert result.converged and 1 <= result.iterations <= 200
    assert result.rel_residual <= 1e-3 and abs(true-result.rel_residual) <= 1e-12
    assert np.linalg.norm(result.x-exact) <= bound
    tight = system.solve(b, rtol=1e-12, maxiter=500)
    assert tight.converged and rel(tight.x, exact) <= 1e-9


def test_pcg_is_odd_in_b_bitwise():
    """B5-3 (EAM): x(-b) = -x(b), so E delta = 0."""
    grid, state = cs.cube6()
    system = lp.LaplaceSystem(state, cs.u_hat_of(grid))
    b = system.rhs(cs.RecordingDraw(9))
    plus, minus = system.solve(b), system.solve(-b)
    assert np.array_equal(minus.x, -plus.x) and minus.iterations == plus.iterations


def test_pcg_not_converged_zero_rhs_columns_and_non_spd():
    grid, state = cs.cube6()
    system = lp.LaplaceSystem(state, cs.u_hat_of(grid))
    b = system.rhs(cs.RecordingDraw(10))
    stuck = system.solve(b, rtol=1e-14, maxiter=3)
    assert not stuck.converged and stuck.iterations == 3 and stuck.rel_residual > 1e-14
    zero = system.solve(np.zeros(192))
    assert zero.converged and zero.iterations == 0 and zero.rel_residual == 0. and not np.any(zero.x)
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 30))
    A = A@A.T+30*np.eye(30)
    Minv = np.diag(1/np.diag(A))
    Bm = rng.normal(size=(30, 4))
    Bm[:, 2] = 0.
    many = lp.pcg(lambda X: A@X, lambda X: Minv@X, Bm, rtol=1e-10, maxiter=100)
    for j in range(4):
        one = lp.pcg(lambda x: A@x, lambda x: Minv@x, Bm[:, j], rtol=1e-10, maxiter=100)
        assert one.converged and bool(many.converged[j])
        assert np.allclose(many.x[:, j], one.x, rtol=0, atol=1e-12) and many.iterations[j] == one.iterations
    assert many.iterations[2] == 0
    with pytest.raises(FloatingPointError):
        lp.pcg(lambda x: -x, lambda x: x, np.ones(5))


# ----------------------------------------------------------------------------------------------
# Samples
# ----------------------------------------------------------------------------------------------

def test_samples_reproducible_per_child_stream():
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    system = lp.LaplaceSystem(state, u)
    five = lp.sample_perturbations(state, u, 21, 5, system=system)
    three = lp.sample_perturbations(state, u, 21, 3, system=system)
    other = lp.sample_perturbations(state, u, 22, 3, system=system)
    assert np.array_equal(five.deltas[:3], three.deltas)
    assert not np.array_equal(three.deltas, other.deltas)
    record = five.record()
    assert record["n_samples"] == 5 and record["rng"]["seed"] == 21 and record["system"]["n_free"] == 192
    assert all(item.converged and item.rel_residual <= 1e-3 for item in five)
    assert not five.items[0].delta.flags.writeable


# ----------------------------------------------------------------------------------------------
# Posterior maps
# ----------------------------------------------------------------------------------------------

def _manual_logj(u_hat, items, grid, state):
    ops = state.ops
    out = []
    for item in items:
        geo = gauss_geometry(u_hat+inject_free(item.delta, grid), ops)
        out.append(np.log(geo.J))
    return np.stack(out)


def test_posterior_maps_statistics_with_scripted_certificate(record_property):
    """sigma_u over PCG-converged samples; sigma_logJ only over converged AND certified samples; pass rate
    S_cert / S_conv, flag, exclusions, cell log J and sample traces (C14-3, C14-5, C15-3)."""
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 31, 7))
    # sample 1: PCG not converged (not certified, in no statistic); samples 3, 5 fail the certificate
    samples[1] = lp.Perturbation(1, samples[1].delta, 200, 2e-3, False, 1., 0.)
    passes = [True, True, True, False, True, False, True]
    backend = cs.ScriptedCertificate([ok for s, ok in zip(samples, passes) if s.converged])
    rows = []
    ops = lp.default_trace_operators(state)
    maps = lp.posterior_maps(state, u, samples, backend=backend, trace_operators=ops, log=rows.append,
                             keep_gauss_variance=True)
    conv = [s for s in samples if s.converged]
    used = [s for s, ok in zip(samples, passes) if s.converged and ok]
    assert (maps.n_samples, maps.n_converged, maps.n_pcg_unconverged, maps.n_cert_passed, maps.n_passed) == \
        (7, 6, 1, 4, 4)
    assert maps.pass_rate == 4/6 and maps.pass_rate_low and maps.qc["posterior_cert_pass_rate_low"]
    assert (maps.qc["pcg_unconverged"], maps.qc["pcg_not_converged"], maps.qc["S_conv"], maps.qc["S_cert"]) == (1, 1, 6, 4)
    assert maps.qc["usable_fraction"] == 4/7 and maps.qc["n_nonpositive_gauss_jd"] == 0
    assert maps.u_hat_state_hash == maps.qc["u_hat_state_hash"] == lp.state_hash(u)
    var = np.mean([s.delta**2 for s in conv], axis=0)
    assert relmax(maps.sigma_u, np.sqrt(inject_free(var, grid))) <= 1e-14
    assert relmax(maps.sigma_u_mag, np.sqrt(inject_free(var, grid).sum(axis=-1))) <= 1e-14
    logj = _manual_logj(u, used, grid, state)
    std = np.std(logj, axis=0, ddof=1)
    w_g = state.ops.weights/grid.omega
    err = relmax(maps.sigma_logj, (std@w_g).reshape(grid.cell_shape))
    record_property("sigma_logj_err", err)
    assert err <= 1e-12
    assert relmax(maps.logj_var_gauss, std**2) <= 1e-12
    assert np.array_equal(maps.logj_sample_index, [0, 2, 4, 6])
    brain = maps.logj_cell_index
    vertices = cell_node_indices(grid, slice(0, grid.n_cells))
    assert np.array_equal(brain, np.flatnonzero(state.mask.ravel()[vertices].any(axis=1)))
    single = np.zeros(grid.shape, dtype=bool)
    single[2, 2, 2] = True        # one mask node: exactly the 8 cells around it
    cells = np.flatnonzero(lp._brain_cells(single))
    assert np.array_equal(cells, np.flatnonzero(single.ravel()[vertices].any(axis=1))) and cells.size == 8
    assert maps.logj_cells.dtype == np.float32 and maps.logj_cells.shape == (4, brain.size)
    assert np.array_equal(maps.logj_cells, (logj@w_g)[:, brain].astype(np.float32))
    excluded = [r["excluded"] for r in maps.samples]
    assert excluded[1] == lp.PCG_NOT_CONVERGED and excluded[3].startswith("logj: certificate infeasible")
    assert maps.samples[1]["certificate_skipped"] == lp.PCG_NOT_CONVERGED and maps.samples[1]["cert_passed"] is None
    assert all(r["certificate_skipped"] is None for k, r in enumerate(maps.samples) if k != 1)
    assert [r["in_sigma_u"] for r in maps.samples] == [True, False, True, True, True, True, True]
    assert [r["in_logj"] for r in maps.samples] == [True, False, True, False, True, False, True]
    assert len(rows) == 7 and rows[0]["event"] == "posterior_sample"
    for name, apply_A in ops.items():
        values = [s.delta @ apply_A(s.delta) for s in conv]
        assert abs(maps.traces[name].mean-np.mean(values)) <= 1e-14*abs(np.mean(values))
        assert maps.traces[name].n_used == 6
        direct = lp.sample_trace(samples, apply_A, name=name)
        assert direct.mean == maps.traces[name].mean and direct.n_used == 6
    assert backend.calls == 6       # the PCG-unconverged sample is not certified (C14-3)


def test_pcg_unconverged_samples_are_not_certificate_failures():
    """Review M4 item 1: PCG failures alone never set posterior_cert_pass_rate_low; with S_conv = 0 the pass
    rate is undefined (NaN, flag None), not 0."""
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 34, 7))
    for k in (0, 3, 5):                       # 3 of 7 unconverged: the old S_pass / S would be 4/7 < .8
        samples[k] = lp.Perturbation(k, samples[k].delta, 200, 2e-3, False, 1., 0.)
    backend = cs.ScriptedCertificate([True]*4)
    maps = lp.posterior_maps(state, u, samples, backend=backend)
    assert backend.calls == 4 and (maps.n_converged, maps.n_pcg_unconverged, maps.n_cert_passed) == (4, 3, 4)
    assert maps.pass_rate == 1. and maps.pass_rate_low is False and maps.qc["posterior_cert_pass_rate_low"] is False
    assert maps.qc["usable_fraction"] == 4/7 and list(maps.logj_sample_index) == [1, 2, 4, 6]
    conv = [s for s in samples if s.converged]
    var = np.mean([s.delta**2 for s in conv], axis=0)
    assert relmax(maps.sigma_u, np.sqrt(inject_free(var, grid))) <= 1e-14


def test_posterior_maps_checks_the_state_hash_and_the_recorded_tile_size():
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 35, 2))
    good = lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate([True]*2),
                             expected_state_hash=lp.state_hash(u))
    assert good.u_hat_state_hash == lp.state_hash(u)
    with pytest.raises(InputValidationError, match="state_hash"):
        lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate([True]*2), expected_state_hash="0"*64)
    other = NumpyBackend(cert_tile_size=4096)
    with pytest.raises(InputValidationError):
        lp.posterior_maps(state, u, samples, backend=other)          # recorded T = 8192 by default
    maps = lp.posterior_maps(state, u, samples, backend=other, cert_tile_size=4096)
    assert maps.qc["certificate"]["tile_size"] == 4096


def test_posterior_maps_degenerate_counts():
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 32, 3))
    one = lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate([True, False, False]))
    assert one.n_passed == 1 and np.all(np.isnan(one.sigma_logj)) and np.all(np.isfinite(one.sigma_u))
    none = [lp.Perturbation(s.index, s.delta, 200, 1., False, 1., 0.) for s in samples]
    backend = cs.ScriptedCertificate([True]*3)
    empty = lp.posterior_maps(state, u, none, backend=backend)
    assert empty.n_converged == 0 and np.isnan(empty.pass_rate) and empty.pass_rate_low is None
    assert backend.calls == 0 and empty.qc["pcg_unconverged"] == 3 and empty.qc["usable_fraction"] == 0.
    assert np.all(np.isnan(empty.sigma_u)) and np.all(np.isnan(empty.sigma_logj))
    high = lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate([True]*3))
    assert high.pass_rate == 1. and not high.pass_rate_low
    with pytest.raises(InputValidationError):
        lp.posterior_maps(state, u, [])


def test_posterior_maps_refuses_a_wrong_certificate_configuration():
    from jd_curl.three_d.feasibility import PrescreenCertificateConfig
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 33, 1))
    backend = NumpyBackend()
    backend.cert_config = PrescreenCertificateConfig(delta=.01)
    with pytest.raises(InputValidationError):
        lp.posterior_maps(state, u, samples, backend=backend)


@pytest.mark.slow                   # numba compilation of certify_map on first use (about 10 s)
def test_real_certificate_pass_rate_and_logj_only_from_passed(record_property):
    """C07 certificate per sample: a PASS sample has every Gauss J_q > 0, a sample with some J_q <= 0
    never passes; pass rate S_pass / S and its .8 flag; sigma_logJ from the passed samples only."""
    grid, state = cs.cube6(factor=6.)
    u = cs.u_hat_of(grid)
    samples = lp.sample_perturbations(state, u, 11, 16)
    maps = lp.posterior_maps(state, u, samples)
    record_property("pass_rate", maps.pass_rate)
    record_property("reasons", sorted({r["cert_reason"] for r in maps.samples}))
    assert 0 < maps.n_passed < 16 and maps.n_converged+maps.n_pcg_unconverged == 16
    assert maps.pass_rate == maps.n_cert_passed/maps.n_converged and maps.pass_rate_low == (maps.pass_rate < .8)
    assert maps.n_cert_passed == maps.n_passed
    passed = []
    for item, row in zip(samples, maps.samples):
        J = gauss_geometry(u+inject_free(item.delta, grid), state.ops).J
        if row["cert_passed"]:
            assert np.all(J > 0) and row["min_jd_lower_bound"] > 0 and row["cert_status"] == "certified_feasible"
        if np.any(J <= 0):
            assert not row["cert_passed"]
        assert row["in_logj"] == (bool(row["cert_passed"]) and item.converged)
        assert (row["cert_passed"] is None) == (not item.converged)        # unconverged: not certified (C14-3)
        if row["in_logj"]:
            passed.append(item)
    logj = _manual_logj(u, passed, grid, state)
    w_g = state.ops.weights/grid.omega
    assert relmax(maps.sigma_logj, (np.std(logj, axis=0, ddof=1)@w_g).reshape(grid.cell_shape)) <= 1e-12
    assert maps.qc["certificate"]["delta_hex"] == "0x0.0p+0" and maps.qc["certificate"]["max_depth"] == 48


# ----------------------------------------------------------------------------------------------
# Traces: Hutchinson against the dense truth (B6)
# ----------------------------------------------------------------------------------------------

def test_hutchinson_matches_dense_probes_and_truth(record_property):
    grid, state = cs.cube6()
    u = cs.u_hat_of(grid)
    system = lp.LaplaceSystem(state, u)
    _, H = cs.dense_blocks(state, u)
    Hinv = np.linalg.inv(H)
    A = cs.dense_operator(lambda v: lp.stiffness_apply_scaled(state, v), 192)
    n = 48
    est = lp.hutchinson_trace(system, lambda v: lp.stiffness_apply_scaled(state, v), 41, n, rtol=1e-10,
                              maxiter=500)
    probes = [2.*g.integers(0, 2, size=192).astype(float)-1. for g in np.random.default_rng(41).spawn(n)]
    dense = np.array([z@A@Hinv@z for z in probes])
    assert est.n_used == n and est.method == "hutchinson_rademacher"
    assert est.record()["pcg"] == dict(rtol=1e-10, maxiter=500, n_not_converged=0)
    assert lp.hutchinson_trace(system, lambda v: v, 41, 1).pcg["rtol"] == 1e-3      # the system's rtol by default
    assert np.max(np.abs(np.array(est.values)-dense)/np.abs(dense)) <= 1e-8
    exact = np.trace(A@Hinv)
    Bs = (A@Hinv+(A@Hinv).T)/2
    se = np.sqrt(2*(np.sum(Bs**2)-np.sum(np.diag(Bs)**2))/n)
    record_property("hutchinson_rel_diff", (est.mean-exact)/exact)
    record_property("hutchinson_rel_se", se/exact)
    assert abs(est.mean-exact) <= 5*se


# ----------------------------------------------------------------------------------------------
# Output files (C16) through run_laplace
# ----------------------------------------------------------------------------------------------

def test_run_laplace_writes_c16_files(tmp_path):
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    rows = []
    config = lp.LaplaceConfig(n_samples=4)
    digest = lp.state_hash(u)
    maps = lp.run_laplace(tmp_path, state, u, seed=lp.seed_from_run_id("test/run"), config=config,
                          trace_operators=lp.default_trace_operators(state), log=rows.append,
                          extra_qc=dict(posterior_skipped=None), expected_state_hash=digest)
    with np.load(tmp_path/"sigma_u.npz") as f:
        assert f["sigma_u"].shape == grid.shape+(3,) and f["sigma_u"].dtype == np.float64
        assert np.array_equal(f["sigma_u"], maps.sigma_u) and int(f["S_conv"]) == 4
        assert json.loads(str(f["grid"]))["shape_zyx"] == list(grid.shape)
        assert np.all(f["sigma_u"][0] == 0) and np.all(f["sigma_u"][:, :, -1] == 0)
        assert str(f["u_hat_state_hash"]) == digest
    with np.load(tmp_path/"sigma_logj.npz") as f:
        assert f["sigma_logj"].shape == grid.cell_shape and np.all(np.isfinite(f["sigma_logj"]))
        assert str(f["u_hat_state_hash"]) == digest
    with np.load(tmp_path/"posterior_logj_cells.npz") as f:
        assert f["logj"].dtype == np.float32 and f["logj"].shape[0] == maps.n_passed
        assert np.array_equal(f["cell_index"], maps.logj_cell_index) and str(f["u_hat_state_hash"]) == digest
    qc = json.loads((tmp_path/"qc.json").read_text())
    for key in ("pass_rate", "pcg_not_converged", "pcg_unconverged", "S_conv", "S_cert", "S_pass", "usable_fraction",
                "posterior_skipped", "posterior_cert_pass_rate_low", "u_hat_state_hash",
                "sigma_u_mag_brain_median", "sigma_u_mag_brain_q99", "logj_hat_brain_q01", "logj_hat_brain_q99",
                "pcg_bias_note", "certificate", "config", "samples", "laplace_system", "laplace_rng", "traces"):
        assert key in qc, key
    assert qc["laplace_rng"]["seed"] == lp.seed_from_run_id("test/run") and len(qc["samples"]) == 4
    assert qc["laplace_rng"]["bit_generator"] == "PCG64" and qc["laplace_rng"]["numpy_version"] == np.__version__
    assert qc["pass_rate"] == 1. and qc["posterior_cert_pass_rate_low"] is False and qc["u_hat_state_hash"] == digest
    assert len(rows) == 4
    with pytest.raises(FileExistsError):
        lp.write_posterior_outputs(tmp_path, maps, grid)
    with pytest.raises(FileExistsError):
        lp.run_laplace(tmp_path, state, u, seed=1, config=config)          # refused before any sampling
    with pytest.raises(InputValidationError):
        lp.run_laplace(tmp_path/"x", state, u, seed=np.random.default_rng(0), config=config)
    with pytest.raises(InputValidationError, match="state_hash"):
        lp.run_laplace(tmp_path/"y", state, u, seed=1, config=config, expected_state_hash="0"*64)


def test_write_outputs_checks_every_path_before_writing(tmp_path):
    """Review M4 item 7: one existing output stops the call before any file is written."""
    grid, state = cs.cube6(factor=30.)
    u = cs.u_hat_of(grid)
    samples = list(lp.sample_perturbations(state, u, 36, 2))
    maps = lp.posterior_maps(state, u, samples, backend=cs.ScriptedCertificate([True]*2))
    (tmp_path/"qc.json").write_text("{}")
    with pytest.raises(FileExistsError, match="qc.json"):
        lp.write_posterior_outputs(tmp_path, maps, grid)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["qc.json"] and (tmp_path/"qc.json").read_text() == "{}"
