"""gpu-port procedure L0 of bayesreg.backends.torch64 against the numpy reference bayesreg.model (contract C06, C08, C09.1,
C17): same input, FP64, relative error <= 1e-12 on every function, on cpu and (when available) cuda; the same
failure semantics (InvalidTrial reason and count, InputValidationError); HVP symmetry; determinism.

Relative error of an array = max |device - reference| / max |reference|; of a scalar, |d - r| / |r|.  Energies
are sums of squares (condition number 1); for gradients and HVPs an assertion failure also prints the
cancellation factor max(sum_i |g_i|) / max|g| over the blocks i.

Set BAYESREG_L0_REPORT=<path> to write the maximum errors per (function, case, device) as JSON (off by
default: a test run never writes into the repository).
"""
import json
import os

import numpy as np
import pytest
import torch

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.interpolation import sample_points

from bayesreg.model import (BLOCK_NAMES, build_metric, c_dat, evaluate, gn_diagonal, gradient, hvp, metric_bP,
                            stable_reduction)
import bayesreg.backends.torch64 as T
from bayesreg.backends.torch64.spline import TorchSpline, sample_channels

from ._cases_gauss27 import case, medium_state, smooth_field  # DEF: "gauss27" (the default before DEF) explicit
from ..model._helpers import free_random

DEVICES = ["cpu"]+(["cuda"] if torch.cuda.is_available() else [])
CASES = ["small", "box12", "medium"]
L0 = 1e-12
REPORT = {}


@pytest.fixture(scope="module", autouse=True)
def l0_report():
    yield
    path = os.environ.get("BAYESREG_L0_REPORT")
    if path:
        with open(path, "w") as handle:
            json.dump(REPORT, handle, indent=1, sort_keys=True)


def record(function, name, device, error):
    key = f"{function}/{name}/{device}"
    REPORT[key] = max(REPORT.get(key, 0.), float(error))
    return error


def rel(a, b):
    a = T.to_numpy(a) if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a-b))/scale) if scale > 0 else float(np.max(np.abs(a)))


_REFERENCE = {}


def reference(name):
    """Numpy reference results of one case (computed once per test session)."""
    if name not in _REFERENCE:
        state, u, options = case(name)
        rng = np.random.default_rng(7)
        ev = evaluate(state, u)
        v = free_random(state.grid, rng)
        w = free_random(state.grid, rng)
        out = dict(state=state, u=u, options=options, ev=ev, v=v, w=w,
                   gradient={b: gradient(state, ev, blocks=(b,)) for b in BLOCK_NAMES},
                   hvp={b: hvp(state, ev, v, blocks=(b,)) for b in BLOCK_NAMES})
        out["gradient"]["total"] = gradient(state, ev)
        out["hvp"]["total"] = hvp(state, ev, v)
        out["gn"] = gn_diagonal(state, ev)
        out["c_dat"] = c_dat(state, ev)
        out["b_P"] = metric_bP(state)
        _REFERENCE[name] = out
    return _REFERENCE[name]


def device_state(ref, device, **extra):
    return T.TorchModelState(ref["state"], device, **{**ref["options"], **extra})


def cancellation(parts, total):
    return float(np.max(sum(np.abs(p) for p in parts))/np.max(np.abs(total)))


# ----------------------------------------------------------------------------------------------
# Spline
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_spline_matches_sample_points(device, record_property):
    """C03.5-10: TorchSpline (copied mainline torch path) equals interpolation.sample_points in value and xyz
    gradient (relative 1e-12), at random interior points, on the faces and on the nodes; sample_channels is
    bitwise the per-channel TorchSpline.sample."""
    state = medium_state()
    g = state.grid
    rng = np.random.default_rng(3)
    lo = np.array([a[0] for a in g.axes])
    hi = np.array([a[-1] for a in g.axes])
    inside = lo+(hi-lo)*rng.random((4000, 3))
    faces = lo+(hi-lo)*rng.random((600, 3))
    axis = rng.integers(0, 3, 600)
    faces[np.arange(600), axis] = np.where(rng.random(600) < .5, lo[axis], hi[axis])
    nodes = g.identity().reshape(-1, 3)[rng.choice(np.prod(g.shape), 500, replace=False)]
    points = np.concatenate((inside, faces, nodes))
    dev = T.Device(device)
    splines = tuple(TorchSpline(s, dev.upload) for s in state.samplers)
    pts = dev.upload(points)
    errors = {}
    for c, (sampler, spline) in enumerate(zip(state.samplers, splines)):
        expected = sample_points(sampler, points)
        value, grad = spline.sample(pts, chunk_size=1000)
        errors[f"value{c}"] = rel(value, expected.value)
        errors[f"gradient{c}"] = rel(grad, expected.gradient)
    values, grads = sample_channels(splines, pts, chunk_size=777)
    for c, spline in enumerate(splines):
        value, grad = spline.sample(pts, chunk_size=777)
        assert torch.equal(values[c], value) and torch.equal(grads[c], grad)
    record_property("spline_errors", json.dumps(errors))
    record("spline", "medium", device, max(errors.values()))
    assert max(errors.values()) <= L0, errors


# ----------------------------------------------------------------------------------------------
# F_B, gradient, HVP, GN diagonal, c_dat, metric
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_evaluate_l0(name, device, record_property):
    """C06-1/2/7: F_B and every block energy, and the channel arrays rho, grad G, r, equal the reference to
    1e-12; F_B = float of the longdouble total; the per-tile sums cover ceil(n_cells / tile_size) tiles."""
    ref = reference(name)
    state, ev = ref["state"], ref["ev"]
    ds = device_state(ref, device)
    tev = T.evaluate(ds, ref["u"])
    assert isinstance(tev, T.TorchEvaluation)
    errors = dict(F_B=abs(tev.energy-ev.energy)/ev.energy)
    for name_, value in ev.energies.items():
        if name_ == "channel":
            for c, (a, b) in enumerate(zip(tev.energies["channel"], value)):
                errors[f"channel{c}"] = abs(a-b)/b
        else:
            errors[name_] = abs(tev.energies[name_]-value)/value
    errors["rho"] = rel(tev.channel.rho, ev.channel.rho)
    errors["grad_G"] = rel(tev.channel.grad, ev.channel.grad)
    errors["r"] = rel(tev.channel.r, ev.channel.r)
    record_property("evaluate_errors", json.dumps(errors))
    record("evaluate", name, device, max(errors.values()))
    assert max(errors.values()) <= L0, errors
    assert tev.energy == float(tev.total)
    assert tev.tile_sums.shape == (-(-state.grid.n_cells//state.tile_size), 3)
    assert np.array_equal(tev.u_host(), ev.u)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_gradient_l0(name, device, record_property):
    """C06-4: the gradient and each block's term equal the reference to 1e-12."""
    ref = reference(name)
    ds = device_state(ref, device)
    tev = T.evaluate(ds, ref["u"])
    errors = {b: rel(T.gradient(ds, tev, blocks=(b,)), ref["gradient"][b]) for b in BLOCK_NAMES}
    errors["total"] = rel(T.gradient(ds, tev), ref["gradient"]["total"])
    kappa = cancellation([ref["gradient"][b] for b in BLOCK_NAMES], ref["gradient"]["total"])
    record_property("gradient_errors", json.dumps(errors))
    record("gradient", name, device, max(errors.values()))
    assert max(errors.values()) <= L0, (errors, f"cancellation factor {kappa:.3g}")


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_hvp_l0_cache_and_symmetry(name, device, record_property):
    """C06-5: H v and each block's term equal the reference to 1e-12, with the geometry cache full, empty and
    (medium) partial -- the cached and recomputed HVP are bitwise equal; |<Hv, w> - <v, Hw>| <= 1e-12 |Hv||w|;
    <v, Hv> >= -1e-12 |v|^2."""
    ref = reference(name)
    v, w = ref["v"], ref["w"]
    errors = {}
    results = {}
    full = 72*27*ref["state"].grid.n_cells
    budgets = dict(full=full, none=0, partial=full//3)
    for label, budget in budgets.items():
        ds = device_state(ref, device, geometry_cache_bytes=budget)
        tev = T.evaluate(ds, ref["u"])
        first = T.hvp(ds, tev, v)                     # builds the cache
        second = T.hvp(ds, tev, v)                    # uses it
        assert torch.equal(first, second)
        results[label] = second
        assert tev.geometry_cache_nbytes <= budget
        if label == "full":
            assert tev.geometry_cache_nbytes == full
            for b in BLOCK_NAMES:
                errors[b] = rel(T.hvp(ds, tev, v, blocks=(b,)), ref["hvp"][b])
            Hv, Hw = second, T.hvp(ds, tev, w)
            vt, wt = torch.as_tensor(v, device=Hv.device), torch.as_tensor(w, device=Hv.device)
            symmetry = abs(float(Hv@wt)-float(vt@Hw))/(float(torch.linalg.norm(Hv))*float(torch.linalg.norm(wt)))
            errors["symmetry"] = symmetry
            assert float(vt@Hv) >= -1e-12*float(vt@vt)
            tev.release_geometry_cache()
            assert tev.geometry_cache_nbytes == 0
        if label == "none":
            assert tev.geometry_cache_nbytes == 0
    for label in ("none", "partial"):
        assert torch.equal(results[label], results["full"]), label
    errors["total"] = rel(results["full"], ref["hvp"]["total"])
    kappa = cancellation([ref["hvp"][b] for b in BLOCK_NAMES], ref["hvp"]["total"])
    record_property("hvp_errors", json.dumps(errors))
    record("hvp", name, device, max(v_ for k, v_ in errors.items() if k != "symmetry"))
    record("hvp_symmetry", name, device, errors["symmetry"])
    assert max(errors.values()) <= L0, (errors, f"cancellation factor {kappa:.3g}")


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_gn_diagonal_c_dat_metric_l0(name, device, record_property):
    """C06-6, C09.1: GN diagonal, c_dat, M^-1 g (DST-I) and M v (stencil) equal the reference to 1e-12."""
    ref = reference(name)
    state = ref["state"]
    ds = device_state(ref, device)
    tev = T.evaluate(ds, ref["u"])
    errors = dict(gn_diagonal=rel(T.gn_diagonal(ds, tev), ref["gn"]))
    errors["c_dat"] = abs(T.c_dat(ds, tev)-ref["c_dat"])/ref["c_dat"]
    for gamma in (0., 3.7):
        M = build_metric(state, c_dat=ref["c_dat"], b_P=ref["b_P"], gamma=gamma)
        TM = T.build_metric(ds, c_dat=ref["c_dat"], b_P=ref["b_P"], gamma=gamma)
        assert (TM.a, TM.b, TM.shift) == (M.a, M.b, M.shift)
        g = ref["gradient"]["total"]
        errors[f"metric_solve_{gamma}"] = rel(TM.solve(g), M.solve(g))
        errors[f"metric_matvec_{gamma}"] = rel(TM.matvec(g), M.matvec(g))
        errors[f"metric_inverse_{gamma}"] = rel(TM.matvec(TM.solve(g)), g)
    record_property("metric_errors", json.dumps(errors))
    for key in ("gn_diagonal", "c_dat"):
        record(key, name, device, errors[key])
    record("metric", name, device, max(v for k, v in errors.items() if k.startswith("metric")))
    assert max(errors.values()) <= L0, errors


# ----------------------------------------------------------------------------------------------
# Failure semantics, binding, determinism, interop
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_failure_paths_match_reference(device):
    """C06-1/C07-5: INVALID_DOMAIN and NONPOSITIVE_GAUSS_JD give the reference's reason and count (the
    NONPOSITIVE count is that of the first failing tile of state.tile_size cells); nonzero faces, nonfinite
    values and wrong shapes raise InputValidationError; an evaluation is bound to its device state."""
    ref = reference("small")
    state, u = ref["state"], ref["u"]
    g = state.grid
    ds = device_state(ref, device)
    h = g.spacing
    out_of_box = u.copy()
    out_of_box[2, 3, 3, 0] += 10*h[0]
    out_of_box[1, 1, 1, 1] = g.axes[1][0]-g.axes[1][1]            # interior node exactly on the face y = y0
    folded = np.zeros_like(u)
    folded[2, 2, 2, 0] = 1.6*h[0]                                  # Gauss J < 0 in the cells around (2, 2, 2)
    folded[2, 3, 4, 0] = -1.6*h[0]
    for trial in (out_of_box, folded):
        expected = evaluate(state, trial)
        got = T.evaluate(ds, trial)
        assert isinstance(expected, InvalidTrial) and isinstance(got, InvalidTrial)
        assert (got.reason, got.invalid_count) == (expected.reason, expected.invalid_count)
    assert T.evaluate(ds, folded).reason == "NONPOSITIVE_GAUSS_JD"
    assert T.evaluate(ds, out_of_box).reason == "INVALID_DOMAIN"
    big_tile = state.replace(tile_size=8192)
    assert (T.evaluate(T.TorchModelState(big_tile, device), folded).invalid_count
            == evaluate(big_tile, folded).invalid_count)
    with pytest.raises(InputValidationError):
        T.evaluate(ds, u+1.)                                       # faces must be zero (U0)
    bad = u.copy()
    bad[2, 2, 2, 1] = np.nan
    with pytest.raises(InputValidationError):
        T.evaluate(ds, bad)
    with pytest.raises(InputValidationError):
        T.evaluate(ds, np.zeros(5))
    with pytest.raises(InputValidationError):
        T.evaluate(ds, torch.zeros(3*g.n_free_scalar, dtype=torch.float32))
    tev = T.evaluate(ds, u)
    with pytest.raises(InputValidationError):
        T.gradient(device_state(ref, device), tev)
    with pytest.raises(InputValidationError):
        T.hvp(ds, tev, ref["v"], blocks=("channel", "barrier"))
    with pytest.raises(InputValidationError):
        T.hvp(ds, tev, u+1.)


@pytest.mark.parametrize("device", DEVICES)
def test_determinism_inputs_and_with_state(device):
    """C17-4 / gpu-port procedure 3: two evaluations of the same u give bitwise the same energies, tile sums, gradient
    and HVP; free vector, node field and device tensor inputs agree bitwise; with_state reuses the uploaded
    splines and equals a fresh device state."""
    ref = reference("box12")
    state, u, v = ref["state"], ref["u"], ref["v"]
    ds = device_state(ref, device)
    a, b = T.evaluate(ds, u), T.evaluate(ds, np.moveaxis(u[1:-1, 1:-1, 1:-1], -1, 0).ravel())
    c = T.evaluate(ds, torch.as_tensor(u, device=ds.device.device))
    for other in (b, c):
        assert other.total == a.total and other.energies == a.energies
        assert np.array_equal(other.tile_sums, a.tile_sums)
    assert torch.equal(T.gradient(ds, a), T.gradient(ds, b))
    assert torch.equal(T.hvp(ds, a, v), T.hvp(ds, b, torch.as_tensor(v, device=ds.device.device)))
    nxt = state.replace(sigma2=state.sigma2*2., alpha=.2)
    moved = ds.with_state(nxt)
    assert moved.splines is ds.splines
    fresh = T.TorchModelState(nxt, device, **ref["options"])
    x, y = T.evaluate(moved, u), T.evaluate(fresh, u)
    assert x.total == y.total and torch.equal(T.gradient(moved, x), T.gradient(fresh, y))
    with pytest.raises(InputValidationError):
        ds.with_state(reference("small")["state"])


@pytest.mark.parametrize("device", DEVICES)
def test_as_evaluation_feeds_cpu_stable_reduction(device):
    """C08-6: as_evaluation() gives a numpy Evaluation; the CPU stable_reduction on two device evaluations
    agrees with the one on the reference evaluations to 1e-12 max(E_b, E_t) per block and in ared."""
    ref = reference("medium")
    state = ref["state"]
    ds = device_state(ref, device)
    u_t = ref["u"]+smooth_field(state.grid, .05, 9)
    cpu = stable_reduction(state, ref["ev"], evaluate(state, u_t))
    base, trial = T.evaluate(ds, ref["u"]).as_evaluation(), T.evaluate(ds, u_t).as_evaluation()
    dev = stable_reduction(state, base, trial)
    scale = max(base.energy, trial.energy)
    assert abs(dev.ared-cpu.ared) <= 1e-12*scale
    for name in ("diffusion", "logvol", "curl"):
        assert abs(float(dev.blocks[name]-cpu.blocks[name])) <= 1e-12*scale
    assert dev.reason == cpu.reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
@pytest.mark.parametrize("config", ["one_chunk_no_cache", "chunks_full_cache"])
def test_cuda_peak_memory_within_estimate(config, record_property):
    """Memory estimate: the measured cuda peak of a device state + evaluate + gradient + 2 HVP on the medium
    case stays below estimate_peak_bytes for its shape (the formula of the native-size budget).  The
    one-chunk configuration (all 50 505 cells in one chunk, no cache) bounds the per-cell chunk temporaries
    CHUNK_BYTES_PER_CELL that dominate the transient term at native size."""
    ref = reference("medium")
    state = ref["state"]
    cells = state.grid.n_cells
    options = (dict(chunk_cells=cells, geometry_cache_bytes=0) if config == "one_chunk_no_cache"
               else dict(chunk_cells=4096, geometry_cache_bytes=72*27*cells))
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ds = T.TorchModelState(state, "cuda", **options)
    tev = T.evaluate(ds, ref["u"])
    T.gradient(ds, tev)
    T.hvp(ds, tev, ref["v"])
    T.hvp(ds, tev, ref["v"])
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()-base
    estimate = T.estimate_peak_bytes(state.grid.shape, state.node_index.size, state.n_channels,
                                     chunk_cells=ds.chunk_cells, geometry_cache_bytes=ds.geometry_cache_bytes,
                                     live_evaluations=1)
    record_property("cuda_peak", json.dumps(dict(peak=peak, estimate=estimate["total"])))
    REPORT[f"memory/medium/cuda/{config}"] = dict(peak_bytes=int(peak), estimate_bytes=int(estimate["total"]),
                                                  chunk_cells=estimate["chunk_cells"],
                                                  peak_bytes_per_chunk_cell=peak/estimate["chunk_cells"])
    assert len(ds.q1.chunks) == (1 if config == "one_chunk_no_cache" else 18)
    assert peak <= estimate["total"], (peak, estimate)
