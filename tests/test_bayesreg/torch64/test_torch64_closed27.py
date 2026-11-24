"""C05-2b on the torch64 backend (work package IMPL27): gpu-port procedure L0 against the numpy reference with
logvol_points = "gauss27+closed27" (beta_S = 1, 3; Simpson and the uniform ablation), on the cpu device.

(i)   F_B and every block, the logvol_parts record (Gauss and closed27 parts, minimum J over the closed27 points and
      its cell), gradient and HVP (total and per block), GN diagonal: relative error <= 1e-12 (L0); the per-tile sums
      have the fourth (closed27) column; the HVP cache (closed27 rows under ("closed27", chunk)) is bitwise the
      uncached HVP; the speculative gradient (OPT1 C3) is bitwise the recomputed gradient.
(ii)  C08-8(a)(b) device comparison with the closed27 column against the CPU stable_reduction (|A_hat - A| <= s_hat,
      log-volume block 1e-12); OPT1 C2a: the closed27 J are cached under ("closed27", chunk) and the comparison with
      the cache is bitwise the one without; reduction provenance and model identity.
(iii) W1(17/20) (J <= 0 only at a closed27 point): the device evaluation gives the reference's InvalidTrial (reason,
      count, message); a forged compared state with that J is a DeviceAnomaly naming closed27 points.
(iv)  device_residuals' closed27 rows equal the reference; estimate_peak_bytes(closed27=True).
cuda: minimal tests, skipped without cuda (CUDA_VISIBLE_DEVICES is empty in this package; the lead runs them when the
GPU is free).
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from jd_curl.types import InvalidTrial

from bayesreg import model as M
from bayesreg.geom import gauss_geometry
import bayesreg.backends.torch64 as T
from bayesreg.backends.torch64 import stable as S
from bayesreg.backends.torch64.model import node_field_cm
from bayesreg.model import BLOCK_NAMES, LogVolumeClosed27

from ._cases_gauss27 import case  # DEF: "gauss27" (the default before DEF) explicit
from ..model import _logvol_digest as D
from ..model._gauss27 import make_state
from ..model._helpers import free_random
from ..model.test_logvol_closed27 import w1_field, w_grid

CLOSED = "gauss27+closed27"
L0 = 1e-12
VARIANTS = [(1., "simpson"), (3., "simpson"), (1., "uniform")]


def rel(a, b):
    a = T.to_numpy(a) if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a-b))/scale) if scale > 0 else float(np.max(np.abs(a)))


def closed_case(name, share=1., rule="simpson"):
    state, u, options = case(name)
    return state.replace(logvol_points=CLOSED, logvol_closed27_share=share, closed27_weights=rule), u, options


@pytest.mark.parametrize("share,rule", VARIANTS)
@pytest.mark.parametrize("name", ["small", "box12"])
def test_closed27_l0_on_cpu(name, share, rule, record_property):
    state, u, options = closed_case(name, share, rule)
    rng = np.random.default_rng(7)
    v = free_random(state.grid, rng)
    ev = M.evaluate(state, u)
    ds = T.TorchModelState(state, "cpu", **options)
    tev = T.evaluate(ds, u)
    errors = dict(energy=abs(tev.energy-ev.energy)/ev.energy)
    for b in ("diffusion", "logvol", "curl"):
        errors[f"energy_{b}"] = abs(tev.energies[b]-ev.energies[b])/ev.energies[b]
    errors["energy_channel"] = rel(tev.energies["channel"], ev.energies["channel"])
    parts, ref = tev.energies["logvol_parts"], ev.energies["logvol_parts"]
    for key in ("gauss", "closed27", "closed27_min_J"):
        errors[f"parts_{key}"] = abs(parts[key]-ref[key])/abs(ref[key])
    assert (parts["closed27_min_cell"], parts["closed27_min_point"], parts["closed27_min_xyz"]) == \
        (ref["closed27_min_cell"], ref["closed27_min_point"], ref["closed27_min_xyz"])
    assert tev.tile_sums.shape == (ds.n_tiles, 4)
    errors["gradient"] = rel(T.gradient(ds, tev), M.gradient(state, ev))
    errors["hvp"] = rel(T.hvp(ds, tev, v), M.hvp(state, ev, v))
    for b in BLOCK_NAMES:
        errors[f"gradient_{b}"] = rel(T.gradient(ds, tev, blocks=(b,)), M.gradient(state, ev, blocks=(b,)))
        errors[f"hvp_{b}"] = rel(T.hvp(ds, tev, v, blocks=(b,)), M.hvp(state, ev, v, blocks=(b,)))
    errors["gn"] = rel(T.gn_diagonal(ds, tev), M.gn_diagonal(state, ev))
    record_property("l0_closed27", errors)
    assert max(errors.values()) <= L0, errors
    cached = T.hvp(ds, tev, v)                                  # the cache was built by the first hvp above
    assert any(isinstance(k, tuple) and k[0] == "closed27" for k in tev._cache)
    fresh = T.TorchModelState(state, "cpu", geometry_cache_bytes=0, speculative_gradient=False, **options)
    fev = T.evaluate(fresh, u)
    assert torch.equal(T.hvp(fresh, fev, v), cached) and not fev._cache
    assert torch.equal(T.gradient(fresh, fev), T.gradient(ds, tev))     # OPT1 C3 speculative field, bitwise
    assert ds.provenance()["logvol_points"] == CLOSED and ds.provenance()["logvol_closed27_share"] == share
    assert ds.q1_closed27.weights is None                        # the uniform ops.weights are never the prior's


@pytest.mark.parametrize("share,rule", VARIANTS)
@pytest.mark.parametrize("name", ["small", "box12"])
def test_closed27_device_comparison(name, share, rule, record_property):
    """Delta F_B of the GPU path (cpu device) against the CPU stable_reduction: |A_hat - A| <= s_hat, log-volume block
    1e-12; with the Jacobian cache (closed27 J under ("closed27", chunk)) bitwise the same as without."""
    state, u, options = closed_case(name, share, rule)
    _, u_trial = D.directions(state, u, 7)
    ev, trial = M.evaluate(state, u), M.evaluate(state, u_trial)
    red = M.stable_reduction(state, ev, trial)
    ds = T.TorchModelState(state, "cpu", **options)
    tb, tt = T.evaluate(ds, u), T.evaluate(ds, u_trial)
    assert any(k == ("closed27", 0) for k in tb._jacobians["entries"])
    tiles = S.device_gauss_changes(ds, tb, tt)
    assert tiles.shape == (ds.n_tiles, 4)
    off = T.TorchModelState(state, "cpu", jacobian_cache_slots=0, **options)
    ob, ot = T.evaluate(off, u), T.evaluate(off, u_trial)
    assert not ob._jacobians and np.array_equal(S.device_gauss_changes(off, ob, ot), tiles)
    delta, blocks = S.gpu_fb_change(ds, tb, tt)
    A_hat, s_hat = float(-delta), S.gpu_guard(tb.energy, tt.energy)
    errors = dict(A=abs(A_hat-red.ared)/s_hat, logvol=float(abs(blocks["logvol"]-red.blocks["logvol"])
                                                             / abs(red.blocks["logvol"])))
    record_property("comparison", errors)
    assert errors["A"] <= 1. and errors["logvol"] <= L0, errors
    provenance = S.reduction_provenance(8192, CLOSED)
    assert "logvol_closed27" in provenance["per_cell"] and "logvol_corners" not in provenance["per_cell"]
    assert provenance["depth_bound"] == S.reduction_depth(8192) == 8225
    assert S.model_identity(state) != S.model_identity(state.replace(logvol_points="gauss27",
                                                                     logvol_closed27_share=1.,
                                                                     closed27_weights="simpson"))
    residuals = S.device_residuals(ds, tb, 0, state.grid.n_cells)
    tile = slice(0, state.grid.n_cells)
    assert set(residuals) == {"diffusion", "logvol", "curl", "logvol_closed27"}
    assert rel(residuals["logvol_closed27"], M.gauss_value(LogVolumeClosed27, state, u, tile=tile)) <= L0
    assert rel(residuals["logvol"], M.gauss_value(M.LogVolume, state, u, tile=tile)) <= L0


def test_closed27_failure_path_matches_reference():
    """W1(17/20): every Gauss J > 0 and corner J > 0, one closed27 point (K0's edge midpoint) with J < 0: the device
    evaluation gives the reference's InvalidTrial (NONPOSITIVE_GAUSS_JD, same count and message naming closed27);
    the default evaluates; a forged compared state is a DeviceAnomaly naming closed27 points."""
    grid = w_grid((.9, 1.1, 1.3))
    u = w1_field(grid, 17/20)
    state = make_state(grid, seed=11, tile_size=40)
    assert not isinstance(T.evaluate(T.TorchModelState(state, "cpu", chunk_cells=72), u), InvalidTrial)
    closed = state.replace(logvol_points=CLOSED)
    expected = M.evaluate(closed, u)
    got = T.evaluate(T.TorchModelState(closed, "cpu", chunk_cells=72), u)
    assert isinstance(expected, InvalidTrial) and isinstance(got, InvalidTrial)
    assert (got.reason, got.invalid_count, got.message) == (expected.reason, expected.invalid_count, expected.message)
    assert "closed27" in got.message
    ds = T.TorchModelState(closed, "cpu", chunk_cells=72)
    base = T.evaluate(ds, np.zeros_like(u))
    forged = replace(base, u_cm=node_field_cm(u, ds), _cache={}, _jacobians={})
    n_bad = int(np.count_nonzero(gauss_geometry(u, closed.closed27_ops).J <= 0))
    with pytest.raises(S.DeviceAnomaly, match=f"J_q <= 0 at {n_bad} Gauss points and closed27 points"):
        S.device_gauss_changes(ds, base, forged)


@pytest.mark.parametrize("chunk_cells", [36, 72, 100000], ids=["six_chunks", "three_chunks", "one_chunk"])
def test_closed27_minimum_merged_across_chunks(chunk_cells):
    """The minimum J over the closed27 points across several device chunks (review IMPL27 suggestion: the existing
    cases put it in the first chunk): W1(.8) puts it at K0's edge midpoint (J = .04, chunk 2 of 6 one-slab chunks),
    u = 0 makes every J exactly 1 (the first point of the first chunk must win the tie, as in numpy's tile order);
    cell, point, xyz and value equal the numpy reference's."""
    grid = w_grid((1., 1., 1.))
    state = make_state(grid, seed=11).replace(logvol_points=CLOSED)
    ds = T.TorchModelState(state, "cpu", chunk_cells=chunk_cells)
    assert len(ds.q1_closed27.chunks) == {36: 6, 72: 3, 100000: 1}[chunk_cells]
    k0 = int(np.ravel_multi_index((2, 2, 2), grid.cell_shape))
    for u, cell in ((w1_field(grid, .8), k0), (np.zeros(grid.shape+(3,)), 0)):
        ref = M.evaluate(state, u).energies["logvol_parts"]
        got = T.evaluate(ds, u).energies["logvol_parts"]
        assert ref["closed27_min_cell"] == cell
        assert (got["closed27_min_cell"], got["closed27_min_point"], got["closed27_min_xyz"]) == \
            (ref["closed27_min_cell"], ref["closed27_min_point"], ref["closed27_min_xyz"])
        assert abs(got["closed27_min_J"]-ref["closed27_min_J"]) <= 1e-14
    assert M.evaluate(state, np.zeros(grid.shape+(3,))).energies["logvol_parts"]["closed27_min_point"] == 0


def test_peak_estimate_counts_the_closed27_terms():
    """estimate_peak_bytes(closed27=True): 27 extra points per cell (cache rows 72 B x 27, chunk temporaries of a
    Gauss chunk, Jacobian cache 8 B x 27), the fourth per-cell row; closed27=False is the estimate before C05-2b;
    exclusive with corners."""
    from bayesreg.backends.torch64 import model as TM
    from jd_curl.validation import InputValidationError
    shape, n_mask, C = (160, 192, 224), 2_500_000, 4
    nz, ny, nx = shape
    cells, slab = (nz-1)*(ny-1)*(nx-1), (ny-1)*(nx-1)
    chunk = max(1, TM.CHUNK_CELLS//slab)*slab
    default = T.estimate_peak_bytes(shape, n_mask, C)
    assert default == T.estimate_peak_bytes(shape, n_mask, C, closed27=False) and "closed27" not in default
    closed = T.estimate_peak_bytes(shape, n_mask, C, closed27=True)
    assert closed["closed27"] and not closed["corners"]
    assert closed["full_cache"] == 72*54*cells
    assert closed["transient"]["chunk"] == 2*TM.CHUNK_BYTES_PER_CELL*chunk
    assert closed["transient"]["evaluate_cell_sums"] == (4*8+8+8+1)*cells
    assert closed["jacobian_cache"] == 2*8*54*cells
    corner = T.estimate_peak_bytes(shape, n_mask, C, corners=True)
    assert closed["total"] > corner["total"] > default["total"]
    with pytest.raises(InputValidationError):
        T.estimate_peak_bytes(shape, n_mask, C, corners=True, closed27=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only (run by the lead when the GPU is free)")
@pytest.mark.parametrize("share", [1., 3.])
def test_cuda_closed27_minimal(share):
    """cuda L0: F_B, the closed27 part, gradient and HVP against the numpy reference; the device comparison against
    the CPU stable_reduction (|A_hat - A| <= s_hat)."""
    state, u, options = closed_case("small", share)
    ev = M.evaluate(state, u)
    ds = T.TorchModelState(state, "cuda", **options)
    tev = T.evaluate(ds, u)
    assert abs(tev.energy-ev.energy) <= L0*ev.energy
    assert abs(tev.energies["logvol_parts"]["closed27"]-ev.energies["logvol_parts"]["closed27"]) <= \
        L0*ev.energies["logvol_parts"]["closed27"]
    assert rel(T.gradient(ds, tev), M.gradient(state, ev)) <= L0
    v = free_random(state.grid, np.random.default_rng(7))
    assert rel(T.hvp(ds, tev, v), M.hvp(state, ev, v)) <= L0
    _, u_trial = D.directions(state, u, 7)
    trial = M.evaluate(state, u_trial)
    red = M.stable_reduction(state, ev, trial)
    delta, _ = S.gpu_fb_change(ds, tev, T.evaluate(ds, u_trial))
    assert abs(float(-delta)-red.ared) <= S.gpu_guard(ev.energy, trial.energy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only (run by the lead when the GPU is free)")
def test_cuda_peak_memory_with_closed27_within_estimate():
    """The cuda peak of a closed27 device state + evaluate + gradient + 2 HVP on the medium case stays below
    estimate_peak_bytes(closed27=True)."""
    state, u, _ = case("medium")
    state = state.replace(logvol_points=CLOSED)
    cells = state.grid.n_cells
    v = free_random(state.grid, np.random.default_rng(7))
    options = dict(chunk_cells=4096, geometry_cache_bytes=72*54*cells)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ds = T.TorchModelState(state, "cuda", **options)
    tev = T.evaluate(ds, u)
    T.gradient(ds, tev)
    T.hvp(ds, tev, v)
    T.hvp(ds, tev, v)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()-base
    estimate = T.estimate_peak_bytes(state.grid.shape, state.node_index.size, state.n_channels,
                                     chunk_cells=ds.chunk_cells, geometry_cache_bytes=ds.geometry_cache_bytes,
                                     live_evaluations=1, closed27=True)
    assert peak <= estimate["total"], (peak, estimate)
