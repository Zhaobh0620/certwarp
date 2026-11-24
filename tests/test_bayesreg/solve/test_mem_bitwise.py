"""MEM gate: the memory-only changes leave every value unchanged (work package MEM; runs/20261031-mem/plan.md).

- golden trajectories: run_pair end to end (14 structure channels, gauss27+closed27, certificate cell estimate, rounds
  1,1,1,2, 3 accepted steps per round) and solve_multilevel on four synthetic levels (3 channels), numpy and torch64 on
  the cpu device, give the digests of tests/test_bayesreg/_golden/golden_9747553_mem.json -- computed by
  make_golden_mem.py on the code before the memory changes (9747553: facf4b8 plus the read-only memtrace marks; the
  generation on 6fb91bd = 8a4d060 plus the marks gave the same digests) --
  row for row (log rows without timing fields), state_hash for state_hash, and the exported state, cert.json and
  disp.npz; solve_multilevel with release_inputs=True (levels handed over as a list) gives the same digests;
- the mechanisms: ModelState shares owned read-only inputs and, under `replace`, the derived node arrays (beta
  recomputed); adopt_readonly / freeze_new copy whatever they cannot share; release_inputs swaps each LevelInput for
  its grid only and needs a list; Handoff gives run_pg the point and keeps none; the zero-copy hashes equal the
  tobytes() digests; the torch64 backend drops the device states of other grids;
- memtrace: off by default (mark does nothing), on writes one JSON line per mark with the caller's context.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg import memtrace
from bayesreg.model import ModelState, build_samplers
from bayesreg.readonly import adopt_readonly, freeze_new
from bayesreg.solve import FixedSigma, SolverConfig, make_backend, solve_multilevel
from bayesreg.solve.multilevel import grid_only
from bayesreg.solve.pg import Handoff
from bayesreg.solve.stopping import c_order_bytes, state_hash

from . import _mem_digest as D
from .test_run_pair_structure import use_synthetic_folds
from . import _synthetic as syn

GOLDEN = Path(__file__).resolve().parents[1]/"_golden"/"golden_9747553_mem.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())["digests"]


def _same(digest, reference):
    """Compare two digests; on a difference name the first differing row."""
    if digest == reference:
        return
    rows = [(k, a, b) for k, (a, b) in enumerate(zip(digest.get("row_digests", []), reference.get("row_digests", [])))
            if a != b]
    keys = sorted(k for k in set(digest) | set(reference) if digest.get(k) != reference.get(k))
    raise AssertionError(f"digest differs in {keys}; first differing row {rows[:1]}")


@pytest.mark.parametrize("backend", ["numpy", "torch64"])
def test_run_pair_trajectory_is_bitwise_the_golden(golden, backend, monkeypatch):
    use_synthetic_folds(monkeypatch)                                            # C23.9-13 (8), revision 6f: the synthetic training draw; the golden is not changed
    digest = D.run_pair_digest(backend)
    assert digest["exit_code"] == 0 and digest["status"] == "cap_reached"
    assert len(digest["accepted_state_hashes"]) == 3*5                          # 3 steps x 5 rounds
    _same(digest, golden["run_pair"][backend])


@pytest.mark.parametrize("backend", ["numpy", "torch64"])
@pytest.mark.parametrize("release", [False, True])
def test_solve_multilevel_trajectory_is_bitwise_the_golden(golden, backend, release):
    digest = D.solve_digest(backend, release=release)
    assert digest["native"] and len(digest["accepted_state_hashes"]) == 3*5
    _same(digest, golden["solve"][backend])


# ---- mechanisms -------------------------------------------------------------------------------------------

def _state(n=9, channels=2):
    grid = syn.cube_grid(n)
    moving = syn.moving_image(grid.identity())
    fixed = np.stack([moving*(1+.1*c) for c in range(channels)])
    fixed = adopt_readonly(fixed)
    samplers = build_samplers(np.stack([moving]*channels), grid)
    return ModelState(grid=grid, omega0=1., fixed=fixed, samplers=samplers, mask=syn.brain_box(grid, 1),
                      sigma2=np.ones(channels), alpha=.1, nu=.2, tau2=np.array([1., .25, .25, .25]),
                      tissue=syn.uniform_tissue(grid))


def test_model_state_shares_inputs_and_replace_reuses_node_arrays():
    state = _state()
    assert not state.fixed.flags.writeable and state.fixed.base is None
    shared = state.replace(sigma2=np.array([.5, 2.]))
    for name in ("fixed", "mask", "tissue", "node_index", "strict", "points0", "fixed_masked", "weights_masked",
                 "tau2_nodes"):
        assert getattr(shared, name) is getattr(state, name), name
    assert shared.beta is not state.beta
    fresh = ModelState(grid=state.grid, omega0=1., fixed=np.array(state.fixed), samplers=state.samplers,
                       mask=np.array(state.mask), sigma2=np.array([.5, 2.]), alpha=.1, nu=.2, tau2=state.tau2,
                       tissue=np.array(state.tissue))
    for name in ("fixed", "mask", "tissue", "node_index", "strict", "points0", "fixed_masked", "weights_masked",
                 "tau2_nodes", "beta"):
        a, b = getattr(shared, name), getattr(fresh, name)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name          # bitwise the fresh construction
        assert not a.flags.writeable
    other_eta = state.replace(tau2=np.array([1., .5, .25, .25]))
    assert other_eta.tau2_nodes is not state.tau2_nodes
    assert not np.array_equal(other_eta.tau2_nodes, state.tau2_nodes)
    writeable = np.array(state.fixed)                                          # a writeable input is copied
    assert ModelState(grid=state.grid, omega0=1., fixed=writeable, samplers=state.samplers, mask=state.mask,
                      sigma2=np.ones(2), alpha=.1, nu=.2, tau2=state.tau2, tissue=state.tissue).fixed is not writeable


def test_adopt_and_freeze():
    owned = np.arange(6.)
    owned.flags.writeable = False
    assert adopt_readonly(owned) is owned
    view = owned[1:]
    copied = adopt_readonly(view)
    assert copied is not view and copied.base is None and not copied.flags.writeable
    writeable = np.arange(6.)
    assert adopt_readonly(writeable) is not writeable and writeable.flags.writeable
    assert adopt_readonly(np.ones(3, dtype=np.float32)).dtype == np.float64
    mask = np.ones(4, dtype=bool)
    assert adopt_readonly(mask, bool) is not mask and adopt_readonly(adopt_readonly(mask, bool), bool).dtype == bool
    new = np.arange(4.)+1
    assert freeze_new(new) is new and not new.flags.writeable
    strided = np.arange(8.)[::2]
    frozen = freeze_new(strided)
    assert frozen is not strided and not frozen.flags.writeable and np.array_equal(frozen, strided)


def test_zero_copy_hashes():
    rng = np.random.default_rng(3)
    u = rng.standard_normal((4, 5, 6, 3))
    assert state_hash(u) == hashlib.sha256(u.tobytes(order="C")).hexdigest()
    sliced = u[:, ::2]
    assert state_hash(sliced) == hashlib.sha256(np.ascontiguousarray(sliced).tobytes(order="C")).hexdigest()
    assert bytes(c_order_bytes(np.ones((2, 3), dtype=bool))) == np.ones((2, 3), dtype=bool).tobytes()
    from bayesreg.backends.torch64.stable import array_sha256
    for array in (u, np.ones((3, 2), dtype=bool), np.array([1., 2.]), np.zeros(0)):
        assert array_sha256(array) == hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def test_release_inputs_needs_a_list_and_keeps_grids():
    _, levels, _ = syn.four_levels(26)
    with pytest.raises(InputValidationError):
        solve_multilevel(make_backend("numpy"), tuple(levels), eta=None, sigma=FixedSigma([1.]), release_inputs=True)
    stub = grid_only(levels[0])
    assert stub.grid is levels[0].grid and stub.level == 0 and stub.fixed is None and stub.moving is None
    owned = list(levels)
    result = solve_multilevel(make_backend("numpy"), owned, eta=D_eta(), sigma=FixedSigma([1.]),
                              config=SolverConfig(rounds=(1, 1, 1, 1), max_levels=2), release_inputs=True)
    assert result.status == "stopped_before_native"
    assert [lv.fixed is None for lv in owned] == [True, True, False, False]      # released as the levels were built
    assert [lv.grid for lv in owned] == [lv.grid for lv in levels]


def D_eta():
    from bayesreg.solve import Eta
    return Eta(1., 1., (1., 1., 1., 1.))


def test_handoff_passes_the_point_once():
    point = object()
    handoff = Handoff(point)
    assert handoff.take() is point
    with pytest.raises(InputValidationError):
        handoff.take()


def test_torch64_backend_drops_device_states_of_other_grids():
    pytest.importorskip("torch")
    backend = make_backend("torch64", device="cpu")
    coarse, fine = _state(7, 1), _state(9, 1)
    backend.prepare(coarse)
    backend.prepare(coarse.replace(sigma2=np.array([.5])))
    assert len(backend._states) == 2
    backend.prepare(fine)
    assert [host for host, _, _ in backend._states.values()] == [fine]
    assert backend._lookup(coarse)[0].state is coarse                         # prepared again on demand


def test_memtrace_off_by_default_and_on(tmp_path):
    assert not memtrace.enabled()
    memtrace.mark("nothing", context=dict(level=1))
    path = tmp_path/"marks.jsonl"
    try:
        memtrace.configure(path)
        memtrace.mark("a", context=dict(level=3, iteration=2), attempt=1)
        memtrace.mark("b")
    finally:
        memtrace.configure(None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["label"] for r in rows] == ["a", "b"] and [r["seq"] for r in rows] == [0, 1]
    assert rows[0]["level"] == 3 and rows[0]["iteration"] == 2 and rows[0]["attempt"] == 1
    assert rows[0]["rss_mb"] > 0 and "hwm_lifetime_mb" in rows[0]
    memtrace.mark("after")                                                     # off again: nothing written
    assert len(path.read_text().splitlines()) == 2


def test_comparator_forgets_cpu_evaluations_of_other_states():
    """The C08-8(d) cache serves an Evaluation only to its own ModelState object, so a new state's prepare drops the
    others' entries (memory only); InvalidTrial entries stay; the backend calls it when it creates a device state."""
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from jd_curl.types import InvalidTrial
    from bayesreg.backends.torch64.stable import StableComparator
    comparator = StableComparator(cache_size=4)
    a, b = object(), object()
    invalid = InvalidTrial("INVALID_DOMAIN", "x", 1)
    comparator.cache.update({"ka": SimpleNamespace(state=a), "kb": SimpleNamespace(state=b), "ki": invalid})
    comparator.base = ("ka", comparator.cache["ka"])
    assert comparator.forget_other_states(b) == 2
    assert list(comparator.cache) == ["kb", "ki"] and comparator.base is None
    backend = make_backend("torch64", device="cpu")
    state = _state(7, 1)
    backend.comparator.cache["old"] = SimpleNamespace(state=object())
    backend.prepare(state)
    assert "old" not in backend.comparator.cache


def test_release_free_heap_returns_a_status():
    from bayesreg.heap import release_free_heap
    junk = [bytearray(1 << 20) for _ in range(8)]
    del junk
    assert release_free_heap() in (0, 1, None)
