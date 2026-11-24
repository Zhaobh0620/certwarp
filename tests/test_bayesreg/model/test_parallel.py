"""Work package CPUPAR: the thread-parallel CPU reference is bitwise the serial one.

Reference: the serial loops of commit 8d624c8 (`_serial_reference`, verbatim bodies).  For every case and thread
count, evaluate (energy, longdouble total, block energies, channel arrays), gradient and hvp (all blocks and
subsets), gn_diagonal, c_dat, stable_reduction (ared, guard, longdouble delta and blocks) and the proximal
value / gradient / change of option (b) must equal the reference bit for bit (float64 arrays by their bytes,
longdouble scalars by ==).

Cases (tile counts include a last tile that the tile size does not divide):
  small   7 x 6 x 5 nodes, 120 cells, tile 7 -> 18 tiles (last 1 cell), C = 3, Student-t
  box     12 x 11 x 10 nodes, 990 cells, tile 97 -> 11 tiles (last 20 cells), C = 3, Gaussian
  medium  40 x 38 x 36 nodes, 50 505 cells, tile 8192 -> 7 tiles (last 1 353 cells), C = 4, Student-t
  spline  42 x 40 x 40 nodes, all 67 200 in the mask (two sample_points chunks per channel), C = 2, tile 5003 ->
          13 tiles (last 2 325 cells)
Also: the failure semantics (first tile with J_q <= 0 in ascending order and its count; the InvalidTrial of the
spline sampling; the exception of stable_reduction), `ordered_map` itself (order, exceptions, early exit, NumPy
error state and buffer size in the workers, nested calls), tile_vjp_local + tile_vjp_scatter against
tile_vjp_accumulate, the recorded thread count, and two short solver runs (options (a) and (b)) whose
trajectories are bitwise the same for 1 and 4 threads.
DEF (2026-10-31): the serial reference is the C05-2 model (the ModelState default before DEF), so the four cases are
built with logvol_points "gauss27" explicitly (_gauss27); the two solver runs use the defaults of the code (since DEF
gauss27+closed27 and the certificate cell estimate).
"""
from contextlib import closing
from functools import lru_cache
import hashlib
import inspect
import threading
import time

import numpy as np
import pytest

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError
from jd_curl.three_d.boundary import restrict_free
from jd_curl.three_d.cell_q1 import CellQ1Operators3D, cell_tiles
from jd_curl.three_d.grid import make_grid_3d
from jd_curl.three_d.interpolation import sample_points

from bayesreg import model as M
from bayesreg.geom import (gauss27, node_field, ordered_map, tile_geometry, tile_vjp_accumulate, tile_vjp_local,
                           tile_vjp_scatter)
from bayesreg.geom.parallel import DEFAULT_CPU_THREADS, cpu_threads
from bayesreg.model.blocks import SPLINE_CHUNK, _sample_channels, _sample_serial
from bayesreg.solve import proximal as prox

from . import _serial_reference as R
from ._gauss27 import gauss27 as gauss27_model, make_state       # (bayesreg.geom.gauss27 are the operators)
from ._helpers import aniso, feasible_field, free_random
from ..torch64._cases import medium_state, smooth_field

THREADS = (1, 2, 3, 8)
SUBSETS = (M.BLOCK_NAMES, ("logvol",), ("curl",), ("diffusion", "curl"), ("channel", "logvol"))
TRANSPOSE_SOURCE_SHA256 = {   # E108 helpers of jd_curl/three_d/cell_q1.py @ eb808be (file locked in test_geom_q1.py)
    "_transpose_local": "f8329d3c344ee21118fa30815a211029cefefe5d5c03f17b3d07d4dfe457a7c3",
    "_transpose_scatter": "2ffe4a9a1432f5d7afa75e86809685513045233ad06e265a3ad830cf31f9c614"}


def same(a, b):
    """Bitwise equality of two float64 arrays (bytes), or of two (long)double scalars."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    return type(a) is type(b) and (a == b or (a != a and b != b)) and np.signbit(a) == np.signbit(b)


@lru_cache(maxsize=None)
def case(name):
    """(serial ModelState, base u, trial u, direction v) of a named case; cached for the module."""
    if name == "small":
        g = aniso()
        state = make_state(g, seed=0, student_t=True, tile_size=7)
        rng = np.random.default_rng(1)
        u = feasible_field(g, rng)
        return state, u, feasible_field(g, rng), free_random(g, rng)
    if name == "box":
        g = make_grid_3d(nx=12, ny=11, nz=10, hx=.5, hy=.45, hz=.6, x0=.1, y0=-.4, z0=.3, unit="mm")
        state = make_state(g, seed=3, student_t=False, tile_size=97)
        rng = np.random.default_rng(4)
        return state, feasible_field(g, rng, scale=.12), feasible_field(g, rng, scale=.1), free_random(g, rng)
    if name == "medium":
        state = gauss27_model(medium_state())
        g = state.grid
        return state, smooth_field(g, .4, 5), smooth_field(g, .35, 6), free_random(g, np.random.default_rng(7))
    if name == "spline":
        g = make_grid_3d(nx=42, ny=40, nz=40, hx=.25, hy=.3, hz=.28, x0=-1., y0=.5, z0=-.2, unit="mm")
        state = make_state(g, seed=8, n_channels=2, tile_size=5003, mask=np.ones(g.shape, bool))
        rng = np.random.default_rng(9)
        return state, smooth_field(g, .3, 10), smooth_field(g, .25, 11), free_random(g, rng)
    raise KeyError(name)


CASE_SUBSETS = dict(small=SUBSETS, box=SUBSETS, medium=SUBSETS[:2], spline=SUBSETS[:1])


@lru_cache(maxsize=None)
def reference(name):
    """The 8d624c8 serial results of a case (the serial ModelState, cpu_threads = 1)."""
    state, u, w, v = case(name)
    state = state.replace(cpu_threads=1)
    base, trial = R.ref_evaluate(state, u), R.ref_evaluate(state, w)
    out = dict(state=state, base=base, trial=trial,
               gradient={s: R.ref_gradient(state, base, blocks=s) for s in CASE_SUBSETS[name]},
               hvp={s: R.ref_hvp(state, base, v, blocks=s) for s in CASE_SUBSETS[name]},
               gn=M.gn_diagonal(state, base), c_dat=M.c_dat(state, base),
               reduction=R.ref_stable_reduction(state, base, trial))
    if name != "spline":
        params = prox.ProximalParams(lam=prox.default_lambda(state.grid), mu=.07, nu2=.05)
        pb, pt = R.ref_prox_evaluate(state, u, u, params), R.ref_prox_evaluate(state, w, u, params)
        out.update(params=params, prox_trial=pt, prox_gradient=R.ref_prox_gradient(state, pt, params),
                   prox_change=R.ref_prox_change(state, pb, pt, params))
    return out


def assert_same_evaluation(ev, ref):
    assert isinstance(ev, M.Evaluation)
    assert same(ev.total, ref.total) and same(ev.energy, ref.energy)
    assert ev.energies == ref.energies
    assert all(same(a, b) for a, b in zip(ev.energies["channel"], ref.energies["channel"]))
    assert same(ev.u, ref.u)
    for name in ("rho", "grad", "r"):
        assert same(getattr(ev.channel, name), getattr(ref.channel, name)), name


def assert_same_reduction(red, ref):
    assert same(red.ared, ref.ared) and same(red.guard, ref.guard) and same(red.delta, ref.delta)
    assert red.blocks.keys() == ref.blocks.keys()
    for key in ref.blocks:
        a, b = red.blocks[key], ref.blocks[key]
        pairs = zip(a, b) if key == "channel" else [(a, b)]
        assert all(same(x, y) for x, y in pairs), key


def tile_count(state):
    return len(list(cell_tiles(state.grid, state.tile_size)))


# ----------------------------------------------------------------------------------------------
# Bitwise equality with the serial reference
# ----------------------------------------------------------------------------------------------

def test_cases_cover_partial_tiles_and_spline_chunks():
    counts = {}
    for name in ("small", "box", "medium", "spline"):
        state = case(name)[0]
        counts[name] = (tile_count(state), state.grid.n_cells % state.tile_size, state.node_index.size)
    assert counts["small"][:2] == (18, 1) and counts["box"][:2] == (11, 20) and counts["medium"][:2] == (7, 1353)
    assert counts["spline"][:2] == (13, 2325)
    assert SPLINE_CHUNK < counts["spline"][2] <= 2*SPLINE_CHUNK                  # two chunks per channel
    assert SPLINE_CHUNK == inspect.signature(sample_points).parameters["chunk_size"].default


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize("name", ["small", "box"])
def test_bitwise_small_grids(name, threads):
    """Every model function and the proximal term, all block subsets, against the 8d624c8 serial loops."""
    ref = reference(name)
    state0, u, w, v = case(name)
    state = state0.replace(cpu_threads=threads)
    assert state.cpu_threads == threads
    base, trial = M.evaluate(state, u), M.evaluate(state, w)
    assert_same_evaluation(base, ref["base"])
    assert_same_evaluation(trial, ref["trial"])
    for subset in SUBSETS:
        assert same(M.gradient(state, base, blocks=subset), ref["gradient"][subset]), subset
        assert same(M.hvp(state, base, v, blocks=subset), ref["hvp"][subset]), subset
    assert same(M.gn_diagonal(state, base), ref["gn"]) and same(M.c_dat(state, base), ref["c_dat"])
    assert_same_reduction(M.stable_reduction(state, base, trial), ref["reduction"])
    params = ref["params"]
    pb, pt = prox.evaluate(state, u, u, params), prox.evaluate(state, w, u, params)
    assert same(pt.total, ref["prox_trial"].total)
    assert all(same(pt.parts[k], ref["prox_trial"].parts[k]) for k in prox.PROX_BLOCKS)
    assert same(prox.gradient(state, pt, params), ref["prox_gradient"])
    change = prox.change(state, pb, pt, params)
    assert all(same(change[k], ref["prox_change"][k]) for k in prox.PROX_BLOCKS)


@pytest.mark.parametrize("threads", [8])
def test_bitwise_medium_grid(threads):
    """The 7-tile medium case (default tile 8192, C = 4, Student-t): same checks, the default thread count."""
    ref = reference("medium")
    state0, u, w, v = case("medium")
    state = state0.replace(cpu_threads=threads)
    base, trial = M.evaluate(state, u), M.evaluate(state, w)
    assert_same_evaluation(base, ref["base"])
    assert_same_evaluation(trial, ref["trial"])
    for subset in CASE_SUBSETS["medium"]:
        assert same(M.gradient(state, base, blocks=subset), ref["gradient"][subset]), subset
        assert same(M.hvp(state, base, v, blocks=subset), ref["hvp"][subset]), subset
    assert same(M.gn_diagonal(state, base), ref["gn"]) and same(M.c_dat(state, base), ref["c_dat"])
    assert_same_reduction(M.stable_reduction(state, base, trial), ref["reduction"])
    params = ref["params"]
    pb, pt = prox.evaluate(state, u, u, params), prox.evaluate(state, w, u, params)
    assert same(pt.total, ref["prox_trial"].total)
    assert same(prox.gradient(state, pt, params), ref["prox_gradient"])
    change = prox.change(state, pb, pt, params)
    assert all(same(change[k], ref["prox_change"][k]) for k in prox.PROX_BLOCKS)


@pytest.mark.parametrize("threads", [3, 8])
def test_bitwise_spline_chunks(threads):
    """More mask nodes than one sample_points chunk: the (channel, chunk) pieces give the serial channel arrays;
    13 tiles of 5003 cells."""
    ref = reference("spline")
    state0, u, w, v = case("spline")
    state = state0.replace(cpu_threads=threads)
    base, trial = M.evaluate(state, u), M.evaluate(state, w)
    assert_same_evaluation(base, ref["base"])
    assert_same_evaluation(trial, ref["trial"])
    assert same(M.gradient(state, base), ref["gradient"][M.BLOCK_NAMES])
    assert same(M.hvp(state, base, v), ref["hvp"][M.BLOCK_NAMES])
    assert_same_reduction(M.stable_reduction(state, base, trial), ref["reduction"])


# ----------------------------------------------------------------------------------------------
# Failure semantics
# ----------------------------------------------------------------------------------------------

def folded(state, nodes, shift=1.6):
    """U0 field (mostly 0) with interior nodes pushed by shift * h_x along x: J_q <= 0 in the cells around them."""
    u = np.zeros(state.grid.shape+(3,))
    for node in nodes:
        u[node+(0,)] = shift*state.grid.spacing[0]
    return u


@pytest.mark.parametrize("threads", THREADS)
def test_first_nonpositive_tile_and_its_count(threads):
    """Two folded regions in different tiles: every thread count returns the InvalidTrial of the first such tile
    in ascending order with that tile's count, as the serial loop."""
    state0 = case("box")[0]
    u = folded(state0, [(2, 3, 4), (7, 8, 6)])
    ref = R.ref_evaluate(state0.replace(cpu_threads=1), u)
    state = state0.replace(cpu_threads=threads)
    bad = []
    for k, tile in enumerate(cell_tiles(state.grid, state.tile_size)):
        count = int(np.count_nonzero(tile_geometry(node_field(u, state.grid), state.ops, tile=tile).J <= 0))
        if count:
            bad.append((k, count))
    assert len(bad) >= 2 and bad[0][0] > 0                               # not the first tile; later ones folded too
    got = M.evaluate(state, u)
    assert isinstance(ref, InvalidTrial) and isinstance(got, InvalidTrial)
    assert (got.reason, got.message, got.invalid_count) == (ref.reason, ref.message, ref.invalid_count)
    assert got.reason == M.NONPOSITIVE_GAUSS_JD and got.invalid_count == bad[0][1]


@pytest.mark.parametrize("where", ["domain", "nonfinite"])
@pytest.mark.parametrize("threads", [2, 8])
def test_spline_invalid_trial_is_the_serial_one(where, threads):
    """A sampling failure in the second chunk of the second channel's points: the pieces hand the decision to the
    serial loop, whose InvalidTrial (reason, message, count over all points) is returned."""
    state0 = case("spline")[0]
    points = np.array(state0.points0)
    k = SPLINE_CHUNK+11
    points[k, 0] = 1e3 if where == "domain" else np.nan
    points[k+5, 1] = -1e3 if where == "domain" else np.inf
    ref = _sample_serial(state0.replace(cpu_threads=1), points)
    got = _sample_channels(state0.replace(cpu_threads=threads), points)
    assert isinstance(ref, InvalidTrial) and isinstance(got, InvalidTrial)
    assert (got.reason, got.message, got.invalid_count) == (ref.reason, ref.message, ref.invalid_count)
    ok = _sample_channels(state0.replace(cpu_threads=threads), np.array(state0.points0))
    serial = _sample_serial(state0.replace(cpu_threads=1), np.array(state0.points0))
    assert same(ok[0], serial[0]) and same(ok[1], serial[1])


@pytest.mark.parametrize("threads", [1, 3])
def test_stable_reduction_exception_is_the_serial_one(threads):
    """A compared state with J_q <= 0 (built by hand; certified states never have one): the same exception."""
    state0, u, w, _ = case("box")
    errors = []
    for function, st in ((R.ref_stable_reduction, state0.replace(cpu_threads=1)),
                         (M.stable_reduction, state0.replace(cpu_threads=threads))):
        good = M.evaluate(st, w)
        bad_u = folded(st, [(7, 8, 6)])
        bad_u.flags.writeable = False
        bad = M.Evaluation(st, bad_u, good.channel, good.total, good.energy, good.energies)
        with pytest.raises(InputValidationError) as info:
            function(st, good, bad)
        errors.append(str(info.value))
    assert errors[0] == errors[1] and "J <= 0" in errors[0]


# ----------------------------------------------------------------------------------------------
# ordered_map
# ----------------------------------------------------------------------------------------------

def test_ordered_map_order_exceptions_and_early_exit():
    rng = np.random.default_rng(0)
    delays = rng.uniform(0, 3e-3, 40)
    done = []

    def slow(item):
        time.sleep(delays[item])
        done.append(item)
        return item*item

    with closing(ordered_map(slow, range(40), 4)) as results:
        assert list(results) == [k*k for k in range(40)]

    def failing(item):
        time.sleep(.004 if item == 5 else 0.)
        if item in (5, 9):
            raise ValueError(item)
        return item

    seen = []
    with pytest.raises(ValueError) as info, closing(ordered_map(failing, range(20), 4)) as results:
        for value in results:
            seen.append(value)
    assert seen == [0, 1, 2, 3, 4] and info.value.args == (5,)

    done.clear()
    with closing(ordered_map(slow, range(40), 3)) as results:
        for k, value in enumerate(results):
            if k == 2:
                break
    count = len(done)
    time.sleep(.05)
    assert count == len(done) <= 3+2*3                                    # window 2 threads; nothing after close
    with pytest.raises(InputValidationError):
        list(ordered_map(slow, range(3), 0))
    for bad in (True, 1.5, -2):
        with pytest.raises(InputValidationError):
            cpu_threads(bad)
    assert DEFAULT_CPU_THREADS == 4                                       # FB1 (was 8)


def test_ordered_map_numpy_state_and_nesting():
    """Workers run under the caller's NumPy error state and buffer size (NumPy 2: per context, not inherited by
    threads); a nested ordered_map inside a worker runs serially (no pool waits on itself)."""
    def overflow(_):
        return np.float64(1e308)*np.float64(10.)

    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            list(ordered_map(overflow, range(4), 2))
    with np.errstate(over="ignore"):
        assert all(np.isinf(x) for x in ordered_map(overflow, range(4), 2))
    previous = np.setbufsize(4096)
    try:
        sizes = list(ordered_map(lambda _: np.getbufsize(), range(6), 3))
    finally:
        np.setbufsize(previous)
    assert sizes == [4096]*6

    def nested(item):
        inner = list(ordered_map(lambda x: (x, threading.current_thread().name), range(3), 2))
        assert all(name == threading.current_thread().name for _, name in inner)
        return item+sum(x for x, _ in inner)

    assert list(ordered_map(nested, range(6), 2)) == [k+3 for k in range(6)]


# ----------------------------------------------------------------------------------------------
# Adjoint split, mainline helpers, provenance
# ----------------------------------------------------------------------------------------------

def test_transpose_helpers_locked():
    for name, digest in TRANSPOSE_SOURCE_SHA256.items():
        source = inspect.getsource(getattr(CellQ1Operators3D, name))
        assert hashlib.sha256(source.encode()).hexdigest() == digest, name


@pytest.mark.parametrize("which", ["J", "curl", "grad", "J+curl", "all", "none"])
def test_vjp_local_scatter_is_accumulate(which):
    """tile_vjp_scatter(out, tile_vjp_local(...)) adds bitwise what tile_vjp_accumulate adds, tile after tile into
    one accumulator (shared nodes across tiles included)."""
    g = make_grid_3d(nx=9, ny=8, nz=7, hx=.3, hy=.25, hz=.4, unit="mm")
    ops = gauss27(g)
    rng = np.random.default_rng(3)
    u = node_field(feasible_field(g, rng), g)
    a = rng.normal(size=(3,)+g.shape)
    b = a.copy()
    for tile in cell_tiles(g, 29):
        geo = tile_geometry(u, ops, tile=tile)
        n = tile.stop-tile.start
        gJ = rng.normal(size=(n, 27)) if which in ("J", "J+curl", "all") else None
        gcurl = rng.normal(size=(n, 27, 3)) if which in ("curl", "J+curl", "all") else None
        ggrad = rng.normal(size=(n, 27, 3, 3)) if which in ("grad", "all") else None
        tile_vjp_accumulate(a, ops, tile=tile, gJ=gJ, gcurl=gcurl, ggrad=ggrad, A=geo.A)
        part = tile_vjp_local(ops, tile=tile, gJ=gJ, gcurl=gcurl, ggrad=ggrad, A=geo.A)
        assert (part is None) == (which == "none")
        tile_vjp_scatter(b, ops, part)
    assert same(a, b)
    with pytest.raises(InputValidationError):
        tile_vjp_scatter(np.zeros((3,)+g.shape, dtype=np.float32), ops, None)


def test_thread_count_is_recorded_and_not_in_the_model_identity():
    pytest.importorskip("torch")
    from bayesreg.backends.torch64.stable import model_identity
    from bayesreg.solve import SolverConfig
    state = case("small")[0]
    assert state.cpu_threads == DEFAULT_CPU_THREADS == 4          # FB1: 4 (was 8), room for a concurrent run
    other = state.replace(cpu_threads=3)
    record = other.provenance()
    assert record["cpu_threads"] == 3 and "bitwise identical" in record["cpu_parallel"]
    assert model_identity(other) == model_identity(state)
    for bad in (0, True, 2.5):
        with pytest.raises(InputValidationError):
            state.replace(cpu_threads=bad)
    assert SolverConfig().record()["cpu_threads"] == 4 and SolverConfig(cpu_threads=8).record()["cpu_threads"] == 8
    from bayesreg.cli.run_pair import parse_args
    assert parse_args(["--pair", "s/p", "--channel-source", "none", "--out", "o"]).cpu_threads == 4
    with pytest.raises(InputValidationError):
        SolverConfig(cpu_threads=0)


# ----------------------------------------------------------------------------------------------
# Solver trajectories (options (a) and (b))
# ----------------------------------------------------------------------------------------------

def _trajectory(option, threads):
    from bayesreg.solve import Caps, Commit, NumpyBackend, ProximalParams, default_lambda, g0_of, threshold
    from bayesreg.solve import solve_direct, solve_proximal
    from ..solve import _synthetic as syn
    grid = syn.cube_grid(9)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, syn.solenoidal_displacement(grid, .4))
    state = M.ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=M.build_samplers(moving[None], grid),
                         mask=syn.brain_box(grid, 1), sigma2=np.array([1e-2]), alpha=1e-3, nu=1e-3,
                         tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid), tile_size=61, cpu_threads=threads)
    backend = NumpyBackend()
    zero = np.zeros(grid.shape+(3,))
    start = Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))
    target = threshold(g0_of(backend, state))
    rows = []
    if option == "a":
        result = solve_direct(backend, state, start, target=target, caps=Caps(max_accepted=5), log=rows.append)
    else:
        params = ProximalParams(lam=default_lambda(grid), n_in=3)
        result = solve_proximal(backend, state, start, target=target, caps=Caps(max_accepted=6, max_outer=2),
                                params=params, log=rows.append)
    return result, [r for r in rows if r.get("kind") in ("candidate", "accept", "commit", "commit_check")]


@pytest.mark.parametrize("option", ["a", "b"])
def test_solver_trajectory_is_bitwise_the_serial_one(option):
    """9^3 nodes, tile 61 (9 tiles): accepted steps, energies, GF_B and the final state are the same bits for 1 and 4
    threads (the logged rows too, timings removed)."""
    one, rows_one = _trajectory(option, 1)
    four, rows_four = _trajectory(option, 4)
    assert one.accepted == four.accepted > 0 and one.status == four.status
    assert [same(a, b) for a, b in zip(one.energies, four.energies)] == [True]*len(one.energies)
    assert [same(a, b) for a, b in zip(one.gradient_norms, four.gradient_norms)] == [True]*len(one.gradient_norms)
    assert same(one.commit.u, four.commit.u)

    def strip(row):
        return {k: v for k, v in row.items() if "second" not in k and "time" not in k}

    assert [strip(r) for r in rows_one] == [strip(r) for r in rows_four]
