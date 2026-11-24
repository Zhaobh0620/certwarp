"""C05-2a on the torch64 backend (work package LVC): gpu-port procedure L0 against the numpy reference with
logvol_points = "gauss27+corners", and the default "gauss27" bitwise the code before C05-2a.

(i)  "gauss27" (the default before DEF; explicit since DEF, 2026-10-31): golden digests (see
     tests/test_bayesreg/model/_logvol_digest.py) of evaluate (with the per-tile sums),
     gradient, HVP and the C08-8(a)(b) device comparison on the cpu device, commit 481e33b, numpy 2.0.2,
     torch 2.5.1, single-threaded.  Source (FB1): tests/test_bayesreg/_golden/golden_481e33b.json, generated on the
     code of 481e33b by `python tests/test_bayesreg/_golden/make_golden.py --ref 481e33b --out
     tests/test_bayesreg/_golden/golden_481e33b.json` (checked against the constants by
     model/test_logvol_corners.py::test_golden_constants_match_the_generation_record).
     FB1 also: estimate_peak_bytes(corners=True) counts the corner terms (CPU test; the cuda peak test is skipped
     without cuda).
(iv) corners on the cpu device: F_B and every block, gradient and HVP (total and per block), GN pieces unchanged,
     relative error <= 1e-12; the HVP cache (corner rows included) bitwise equal to the uncached HVP; the device
     comparison (corner column) against the CPU stable_reduction (|A_hat - A| <= s_hat, logvol block 1e-12); the
     B3-2 counterexample gives the reference's reason and count; device_residuals' corner rows equal the reference.
cuda: one minimal test, skipped without cuda (the D0 timing queue shares the GPU; not run in this package).
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from jd_curl.types import InvalidTrial
from jd_curl.three_d.grid import make_grid_3d

from bayesreg import model as M
from bayesreg.geom import gauss_geometry
import bayesreg.backends.torch64 as T
from bayesreg.backends.torch64 import stable as S
from bayesreg.backends.torch64.model import node_field_cm
from bayesreg.model import BLOCK_NAMES, LogVolumeCorners

from ._cases_gauss27 import case  # DEF: "gauss27" (the default before DEF) explicit
from ..model import _logvol_digest as D
from ..model._gauss27 import make_state
from ..model._helpers import free_random

CORNERS = "gauss27+corners"
L0 = 1e-12

GOLDEN_481E33B = dict(small="5269d5a4053512525c96705b9ef29051f4a555f463fbde8fafe0db9e82831d9b",
                      box12="3dbe619e5681f4a5e084f191e8ba70039156fa6602f1390f7530315062496c7e")
# DEF (2026-10-31): digests of the defaults since DEF (gauss27+closed27, beta_S 1, Simpson; jd_cell_estimate
# certificate) GIVEN EXPLICITLY on the code before DEF (commit 8a4d060, where those options existed), numpy 2.0.2, torch
# 2.5.1, single-threaded.  Source: tests/test_bayesreg/_golden/golden_8a4d060_def_explicit.json, generated once by
#   python tests/test_bayesreg/_golden/make_golden.py --path def_explicit --ref 8a4d060 \
#       --out tests/test_bayesreg/_golden/golden_8a4d060_def_explicit.json
# The code's defaults (no option) must give them.
GOLDEN_DEF = dict(small="9ada8399d83fe300ecc23c37db4d92a592d10f8f0dda6d20ef24c126b3d556c0",
                  box12="55bd503f2e6b7eead57e14412b4d483263884858ae9376453214be383b7f520b")


def rel(a, b):
    a = T.to_numpy(a) if isinstance(a, torch.Tensor) else np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.max(np.abs(b))
    return float(np.max(np.abs(a-b))/scale) if scale > 0 else float(np.max(np.abs(a)))


@pytest.mark.skipif(np.__version__ != "2.0.2" or not torch.__version__.startswith("2.5.1"),
                    reason="bitwise golden digests recorded with numpy 2.0.2 and torch 2.5.1")
@pytest.mark.parametrize("name", ["small", "box12"])
def test_default_is_bitwise_the_pre_change_code(name):
    """DEF: "gauss27" -- the default before DEF -- is given explicitly (D.torch_case(name, "pre_def"))."""
    state, u, options = D.torch_case(name, "pre_def")
    assert state.logvol_points == "gauss27" and not state.logvol_corners
    assert D.torch_digest(state, u, **options) == GOLDEN_481E33B[name]


@pytest.mark.skipif(np.__version__ != "2.0.2" or not torch.__version__.startswith("2.5.1"),
                    reason="bitwise golden digests recorded with numpy 2.0.2 and torch 2.5.1")
@pytest.mark.parametrize("name", ["small", "box12"])
def test_def_default_is_bitwise_the_explicit_options_before_def(name):
    """DEF: the torch64 digest (cpu device) of a state built without the option (gauss27+closed27, beta_S 1, Simpson)
    equals that of these options given explicitly on the code before DEF (8a4d060)."""
    state, u, options = D.torch_case(name, "default")
    assert (state.logvol_points, state.logvol_closed27_share, state.closed27_weights) == ("gauss27+closed27", 1.,
                                                                                         "simpson")
    assert D.torch_digest(state, u, **options) == GOLDEN_DEF[name]


@pytest.mark.parametrize("name", ["small", "box12"])
def test_corners_l0_on_cpu(name, record_property):
    state, u, options = case(name)
    state = state.replace(logvol_points=CORNERS)
    rng = np.random.default_rng(7)
    v = free_random(state.grid, rng)
    ev = M.evaluate(state, u)
    ds = T.TorchModelState(state, "cpu", **options)
    tev = T.evaluate(ds, u)
    errors = dict(energy=abs(tev.energy-ev.energy)/ev.energy)
    for b in ("diffusion", "logvol", "curl"):
        errors[f"energy_{b}"] = abs(tev.energies[b]-ev.energies[b])/ev.energies[b]
    errors["energy_channel"] = rel(tev.energies["channel"], ev.energies["channel"])
    assert tev.tile_sums.shape == (ds.n_tiles, 4)
    errors["gradient"] = rel(T.gradient(ds, tev), M.gradient(state, ev))
    errors["hvp"] = rel(T.hvp(ds, tev, v), M.hvp(state, ev, v))
    for b in BLOCK_NAMES:
        errors[f"gradient_{b}"] = rel(T.gradient(ds, tev, blocks=(b,)), M.gradient(state, ev, blocks=(b,)))
        errors[f"hvp_{b}"] = rel(T.hvp(ds, tev, v, blocks=(b,)), M.hvp(state, ev, v, blocks=(b,)))
    errors["gn"] = rel(T.gn_diagonal(ds, tev), M.gn_diagonal(state, ev))
    record_property("l0_corners", errors)
    assert max(errors.values()) <= L0, errors
    cached = T.hvp(ds, tev, v)                                  # the cache was built by the first hvp above
    assert any(isinstance(k, tuple) and k[0] == "corners" for k in tev._cache)
    fresh = T.TorchModelState(state, "cpu", geometry_cache_bytes=0, **options)
    fev = T.evaluate(fresh, u)
    assert torch.equal(T.hvp(fresh, fev, v), cached) and not fev._cache
    assert ds.provenance()["logvol_points"] == CORNERS


@pytest.mark.parametrize("name", ["small", "box12"])
def test_corners_device_comparison(name, record_property):
    """C08-8(a)(b) with the corner column: Delta F_B of the GPU path (cpu device) against the CPU stable_reduction:
    |A_hat - A| <= s_hat, the log-volume block to 1e-12 relative; the reduction provenance names the corner sums;
    the model identity differs from the default state's."""
    state, u, options = case(name)
    state = state.replace(logvol_points=CORNERS)
    _, u_trial = D.directions(state, u, 7)
    ev, trial = M.evaluate(state, u), M.evaluate(state, u_trial)
    red = M.stable_reduction(state, ev, trial)
    ds = T.TorchModelState(state, "cpu", **options)
    tb, tt = T.evaluate(ds, u), T.evaluate(ds, u_trial)
    assert S.device_gauss_changes(ds, tb, tt).shape == (ds.n_tiles, 4)
    delta, blocks = S.gpu_fb_change(ds, tb, tt)
    A_hat, s_hat = float(-delta), S.gpu_guard(tb.energy, tt.energy)
    errors = dict(A=abs(A_hat-red.ared)/s_hat, logvol=float(abs(blocks["logvol"]-red.blocks["logvol"])
                                                             / abs(red.blocks["logvol"])))
    record_property("comparison", errors)
    assert errors["A"] <= 1. and errors["logvol"] <= L0, errors
    assert "logvol_corners" in S.reduction_provenance(8192, CORNERS)["per_cell"]
    assert "logvol_corners" not in S.reduction_provenance(8192)["per_cell"]
    assert S.model_identity(state) != S.model_identity(state.replace(logvol_points="gauss27"))
    residuals = S.device_residuals(ds, tb, 0, state.grid.n_cells)
    tile = slice(0, state.grid.n_cells)
    assert rel(residuals["logvol_corners"], M.gauss_value(LogVolumeCorners, state, u, tile=tile)) <= L0


def test_corners_failure_path_matches_reference():
    """The B3-2 construction (all Gauss J_q > 0, corners with J < 0): with the corners the device evaluation gives
    the reference's InvalidTrial (NONPOSITIVE_GAUSS_JD, same count and message); the default evaluates."""
    grid = make_grid_3d(nx=7, ny=7, nz=7, hx=.9, hy=1.1, hz=1.3, x0=-1., y0=.5, z0=-2., unit="mm")
    u = np.zeros(grid.shape+(3,))
    u[3, 2, 3, 2] = u[3, 3, 3, 2] = -1.05*1.3
    state = make_state(grid, seed=11, tile_size=40)
    assert not isinstance(T.evaluate(T.TorchModelState(state, "cpu", chunk_cells=72), u), InvalidTrial)
    corners = state.replace(logvol_points=CORNERS)
    expected = M.evaluate(corners, u)
    got = T.evaluate(T.TorchModelState(corners, "cpu", chunk_cells=72), u)
    assert isinstance(expected, InvalidTrial) and isinstance(got, InvalidTrial)
    assert (got.reason, got.invalid_count, got.message) == (expected.reason, expected.invalid_count, expected.message)
    ds = T.TorchModelState(corners, "cpu", chunk_cells=72)
    base = T.evaluate(ds, np.zeros_like(u))
    forged = replace(base, u_cm=node_field_cm(u, ds), _cache={})     # a corner J < 0 never evaluate()d
    n_bad = int(np.count_nonzero(gauss_geometry(u, corners.corner_ops).J <= 0))     # Gauss J all > 0 here
    with pytest.raises(S.DeviceAnomaly, match=f"J_q <= 0 at {n_bad} Gauss points and cell corners"):
        S.device_gauss_changes(ds, base, forged)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
def test_cuda_corners_minimal():
    state, u, options = case("small")
    state = state.replace(logvol_points=CORNERS)
    ev = M.evaluate(state, u)
    ds = T.TorchModelState(state, "cuda", **options)
    tev = T.evaluate(ds, u)
    assert abs(tev.energy-ev.energy) <= L0*ev.energy
    assert rel(T.gradient(ds, tev), M.gradient(state, ev)) <= L0


def test_peak_estimate_counts_the_corner_terms():
    """FB1 (review LVC_CPUPAR item 2): estimate_peak_bytes(corners=True) adds the corner rows of the HVP cache (72 B x 8
    per cell, same budget), the corner chunk temporaries (ceil(8/27 of the Gauss chunk's per cell)) and the fourth
    per-cell sum row with its J <= 0 counter; corners=False is the estimate before C05-2a (formula restated here)."""
    from bayesreg.backends.torch64 import model as TM
    from bayesreg.backends.torch64.model import CHUNK_BYTES_PER_CELL, CORNER_CHUNK_BYTES_PER_CELL
    shape, n_mask, C = (160, 192, 224), 2_500_000, 4
    nz, ny, nx = shape
    N, cells, slab = nz*ny*nx, (nz-1)*(ny-1)*(nx-1), (ny-1)*(nx-1)
    chunk = max(1, TM.CHUNK_CELLS//slab)*slab
    old = dict(persistent=8*C*N+8*N+8*(3*C*n_mask+3*n_mask)+9*n_mask, evaluation=8*3*N+8*C*n_mask*5,
               cache=min(TM.GEOMETRY_CACHE_BYTES, 72*27*cells),
               transient=max(CHUNK_BYTES_PER_CELL*chunk+8*3*N*6,
                             8*64*TM.SPLINE_CHUNK*6+(3*8+8+1)*cells+CHUNK_BYTES_PER_CELL*chunk))
    default = T.estimate_peak_bytes(shape, n_mask, C)
    assert default == T.estimate_peak_bytes(shape, n_mask, C, corners=False) and not default["corners"]
    pre_opt1 = T.estimate_peak_bytes(shape, n_mask, C, jacobian_cache_slots=0, speculative_gradient=False)
    assert pre_opt1["total"] == old["persistent"]+2*old["evaluation"]+old["cache"]+old["transient"]
    # OPT1: two Jacobian caches (J only by default since OPT1b, 8 B x 27 x cells each) and a gradient node field per
    # live evaluation; log J doubles the caches
    assert default["jacobian_cache"] == 2*8*27*cells and default["speculative_gradient"] == 2*8*3*N
    assert default["total"] == pre_opt1["total"]+2*8*27*cells+2*8*3*N
    assert T.estimate_peak_bytes(shape, n_mask, C, jacobian_cache_log=True)["jacobian_cache"] == 2*2*8*27*cells
    assert default == T.estimate_peak_bytes(shape, n_mask, C, jacobian_cache_log=False)
    assert default["full_cache"] == 72*27*cells
    corner = T.estimate_peak_bytes(shape, n_mask, C, corners=True)
    assert CORNER_CHUNK_BYTES_PER_CELL == -(-CHUNK_BYTES_PER_CELL*8//27) == 3641
    assert corner["full_cache"] == 72*35*cells and corner["corners"]
    assert corner["transient"]["chunk"] == (CHUNK_BYTES_PER_CELL+CORNER_CHUNK_BYTES_PER_CELL)*chunk
    assert corner["transient"]["evaluate_cell_sums"] == (4*8+8+8+1)*cells
    assert corner["cache"] == default["cache"] == TM.GEOMETRY_CACHE_BYTES      # the budget binds at native size
    assert corner["cached_fraction"] < default["cached_fraction"]
    assert corner["total"] > default["total"]
    assert corner["jacobian_cache"] == 2*8*35*cells                         # J only by default (OPT1b)
    assert T.estimate_peak_bytes(shape, n_mask, C, corners=True, jacobian_cache_log=True)["jacobian_cache"] == \
        2*2*8*35*cells
    full = 72*35*cells
    unbounded = T.estimate_peak_bytes(shape, n_mask, C, corners=True, geometry_cache_bytes=10*full)
    assert unbounded["cache"] == full and unbounded["cached_fraction"] == 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda only")
@pytest.mark.parametrize("config", ["one_chunk_no_cache", "chunks_full_cache"])
def test_cuda_peak_memory_with_corners_within_estimate(config):
    """The cuda peak of a corner-enabled device state + evaluate + gradient + 2 HVP on the medium case stays below
    estimate_peak_bytes(corners=True) (the counterpart of test_torch64.test_cuda_peak_memory_within_estimate)."""
    state, u, _ = case("medium")
    state = state.replace(logvol_points=CORNERS)
    cells = state.grid.n_cells
    v = free_random(state.grid, np.random.default_rng(7))
    options = (dict(chunk_cells=cells, geometry_cache_bytes=0) if config == "one_chunk_no_cache"
               else dict(chunk_cells=4096, geometry_cache_bytes=72*35*cells))
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
                                     live_evaluations=1, corners=True)
    assert peak <= estimate["total"], (peak, estimate)
