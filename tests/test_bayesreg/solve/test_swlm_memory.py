"""SWLM (memory only; contract C04-6 gate 12): what support_shared keeps in host memory beyond brain.  Synthetic data
only (no data/), numpy backend, CPU.  Review: docs/bayesreg/reviews/SWLM_复核_20261101.md.

What tracemalloc sees: the buffers NumPy allocates (and Python objects), not arrays that numba-compiled code allocates
through its own runtime (NRT; e.g. np.empty inside an njit function), and not the free memory glibc keeps in its
arenas (that is what the releases of part 3 hand back).  The native first run's mallinfo2 covers the first blind spot
for the held part: there the allocated heap differed by v plus a constant 2.4 MiB.

1. Fixed-state sequence (`measure_arm` of experiments/swl/fixed_state_memory.py: template, seed evaluation, the
   commit's theta, level start, round state, G0, one evaluation with gradient and c_dat), one CPU thread, K_S = 13
   structure channels (as the native OASIS run), at two grid sizes; the live value is read after gc.collect (reachable
   memory: cyclic garbage freed at GC time made it vary by ~20 KB).
   - live(support_shared) - live(brain) = v.nbytes (K_S n bytes, n mask nodes) + a constant independent of n, at
     swl.level_start and after (swl.seed_evaluated is left out: support_shared has dropped the seed evaluation there,
     brain still holds it for its level start), and ~0 at swl.prepare;
   - the segments that run the same code in both arms have the same peak up to the resident v: swl.prepare and
     swl.seed_evaluated |peak difference| <= CONSTANT, swl.state, swl.g0, swl.eval |peak difference - v.nbytes| <=
     CONSTANT.  This catches an extra transient array in one arm's segment only when it is alive at that segment's
     peak (review R2-4: injected during the seed evaluation it fails, injected after it passes; in swl.eval the peak is
     in the gradient, so a transient during the evaluation is not seen).  The level-start segment differs by design
     (support pass against GaussianMStep.level_start); part 2 bounds the support pass.
2. The support pass alone (bayesreg.model.support.support_level) under tracemalloc: peak - live before <= v.nbytes +
   16 n + PASS_CONSTANT, 16 n bytes = the two length-n float64 temporaries the contract allows for the level start
   (gate 12: v 26 MiB + 32 MiB at the native n ~ 2.1e6), and the excess over v + 16 n does not grow with n.  The
   spline chunk is set to CHUNK points (SPLINE_CHUNK is 65536, larger than these grids' n: the per-chunk temporaries,
   outside the contract's 32 MiB, would otherwise scale with n here) and the map-domain check takes its numba path
   (as at the native size, where u has more than 2^20 entries).  An injected transient of 8 bytes per node fails it.
3. Releases (bayesreg.heap.release_free_heap, malloc_trim): solve_multilevel under support / support_shared calls it
   after support_level, before the level's first round state; under brain the calls are those of the code before SWLM
   (after every G0 only).  The script does the same, and drops the LevelInput once the template is built
   (solve_multilevel's release_inputs).
"""
import gc
import importlib.util
from pathlib import Path
import tracemalloc
import weakref

import numpy as np
import pytest

from bayesreg.geom.q1 import TILE_SIZE
from bayesreg.solve import Caps, GaussianMStep, NumpyBackend, SolverConfig, eta_temporary, solve_multilevel
from bayesreg.solve import multilevel as ML

from . import _swl_synthetic as S
from ._synthetic import cube_grid, true_displacement

ROOT = Path(__file__).resolve().parents[3]
K_S = 13
THETA = .05
CONSTANT = 64*1024                 # bytes: Python objects of the support records, independent of n
SIZES = (16, 32)                   # cube grids (h = 2 mm); mask nodes (n - 2)^3 = 2744 and 27000
SLOPE_TOL = 12*1024                # bytes; between runs <= ~5 KB seen.  12288 / (27000 - 2744) = 0.51: an extra
#                                    array of more than ~0.51 byte per mask node is caught, a bool / uint8 array (1 byte
#                                    per node, 24 KB here) always
LABELS = ("swl.prepare", "swl.seed_evaluated", "swl.level_start", "swl.state", "swl.g0", "swl.eval")
HELD = ("swl.level_start", "swl.state", "swl.g0", "swl.eval")          # v is held from the level start on
CHUNK = 1024                       # spline chunk of part 2 (points)
PASS_SIZES = (24, 40)              # mask nodes 10648 and 54872
PASS_CONSTANT = 256*1024           # bytes: one CHUNK-point spline chunk's temporaries and the pass's Python objects


def script():
    spec = importlib.util.spec_from_file_location("swlm_fixed_state_memory", ROOT/"experiments/swl/fixed_state_memory.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def level_input(n):
    """(LevelInput, eta) of a cube grid with K_S structure blobs (intensity + K_S channels, square-root coordinates)."""
    grid = cube_grid(n, 2.)
    length = (n-1)*2.
    rng = np.random.default_rng(20261031)
    blobs = tuple((tuple(rng.uniform(.25*length, .75*length, 3)), float(rng.uniform(.12*length, .2*length)), .5)
                  for _ in range(K_S))
    level = S.structure_level(3, grid, true_displacement(grid, (.5, -.4, .3)), blobs, 1)
    return level, eta_temporary(level.tissue, level.mask, grid)


# ---- 1. the fixed-state sequence ------------------------------------------------------------------------------------

def traced_arm(measure_arm, n, arm):
    """{label: (live bytes, peak bytes of the segment ending at label)} from tracemalloc's start, the level-start
    record (None under brain), mask nodes."""
    level, eta = level_input(n)
    seed = np.zeros(level.grid.shape+(3,))
    nodes = int(level.mask.sum())
    inputs = [level]
    del level
    marks = {}
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()

        def mark(label):
            current, peak = tracemalloc.get_traced_memory()      # the segment peak, before the collection below
            gc.collect()                                          # live = reachable (cyclic garbage freed at GC time
            marks[label] = (tracemalloc.get_traced_memory()[0]-base, peak-base)   # would make the live value vary)
            tracemalloc.reset_peak()
        *values, record = measure_arm(arm, inputs, eta, NumpyBackend(), seed, theta=THETA, cpu_threads=1, mark=mark)
    finally:
        tracemalloc.stop()
    assert all(np.isfinite(values))
    return marks, record, nodes


@pytest.fixture(scope="module")
def measured():
    measure_arm = script().measure_arm
    # a first call at the largest size, not compared: lazy imports and numba compilation (module state, not arrays:
    # ~14 MB live after a process's first evaluation; some paths start only above a size).  support_shared runs every
    # step brain runs except GaussianMStep.level_start (plain NumPy on the seed evaluation)
    traced_arm(measure_arm, SIZES[-1], "support_shared")
    out = {}
    for n in SIZES:
        brain, _, nodes = traced_arm(measure_arm, n, "brain")
        shared, record, _ = traced_arm(measure_arm, n, "support_shared")
        out[n] = dict(brain=brain, shared=shared, record=record, v_bytes=int(np.prod(record["v_shape"])), nodes=nodes)
    return out


def test_marks_of_the_script(measured):
    for m in measured.values():
        assert tuple(m["brain"]) == tuple(m["shared"]) == ("swl.start",)+LABELS


def test_v_is_one_uint8_row_per_structure_channel(measured):
    for n, m in measured.items():
        assert m["record"]["v_dtype"] == "uint8" and m["record"]["v_shape"] == [K_S, m["nodes"]], n
        assert m["v_bytes"] == K_S*m["nodes"] > 0, n


def test_live_difference_is_v_plus_a_constant(measured):
    residual = {}
    for n, m in measured.items():
        for label in HELD:
            extra = m["shared"][label][0]-m["brain"][label][0]
            residual[n, label] = extra-m["v_bytes"]
            assert abs(residual[n, label]) <= CONSTANT, (n, label, extra, m["v_bytes"])
        assert abs(m["shared"]["swl.prepare"][0]-m["brain"]["swl.prepare"][0]) <= CONSTANT, n   # before the arms part
    # the constant does not grow with n: no other O(n) array is kept
    for label in HELD:
        assert abs(residual[SIZES[1], label]-residual[SIZES[0], label]) <= SLOPE_TOL, (label, residual)


def test_same_code_segments_have_the_same_peak(measured):
    for n, m in measured.items():
        for label in LABELS:
            if label == "swl.level_start":
                continue                                  # support pass against GaussianMStep: part 2
            held = m["v_bytes"] if label in HELD else 0
            extra = m["shared"][label][1]-m["brain"][label][1]
            assert abs(extra-held) <= CONSTANT, (n, label, extra, held)


# ---- 2. the support pass alone ---------------------------------------------------------------------------------------

def support_pass(monkeypatch, n, inject=0):
    """(peak - live before, v bytes, mask nodes) of support_level on the template of level_input(n) under tracemalloc,
    with CHUNK-point spline chunks and the numba map-domain check; inject: bytes per mask node of a transient array
    alive while every channel's rho is computed (a mutation check of the bound)."""
    from jd_curl.three_d import fast_checks
    import bayesreg.model.support as SUP
    level, eta = level_input(n)
    host = ML._template(level, eta, level.grid.omega, TILE_SIZE, K_S+1, 1)
    del level
    seed = np.zeros(host.grid.shape+(3,))
    nodes = int(host.node_index.size)
    monkeypatch.setattr(SUP, "SPLINE_CHUNK", CHUNK)
    monkeypatch.setattr(fast_checks, "MIN_SIZE", 0)
    if inject:
        rho_of = SUP.cpu_channel_rho

        def injected(state, u, c, **kwargs):
            extra = np.ones(inject*nodes, dtype=np.uint8)            # touched: a real allocation
            out = rho_of(state, u, c, **kwargs)
            assert extra[-1] == 1
            return out
        monkeypatch.setattr(SUP, "cpu_channel_rho", injected)
    SUP.support_level(host, seed, THETA, sigma_domain="support_shared")   # first call: numba compilation, not compared
    gc.collect()
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        level = SUP.support_level(host, seed, THETA, sigma_domain="support_shared")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    monkeypatch.undo()
    return peak-before, level.v.nbytes, nodes


def test_support_pass_temporaries_within_two_length_n_float64(monkeypatch):
    excess = {}
    for n in PASS_SIZES:
        rise, v_bytes, nodes = support_pass(monkeypatch, n)
        assert v_bytes == K_S*nodes
        excess[n] = (rise-v_bytes-16*nodes, nodes)
        assert excess[n][0] <= PASS_CONSTANT, (n, rise, v_bytes, nodes)
    (e1, n1), (e2, n2) = excess[PASS_SIZES[0]], excess[PASS_SIZES[1]]
    assert abs(e2-e1) <= 2*(n2-n1), excess                  # nothing of 2 or more bytes per node beyond v + 16 n


@pytest.mark.parametrize("inject", [8, 24])
def test_support_pass_bound_catches_an_injected_transient(monkeypatch, inject):
    n = PASS_SIZES[-1]
    rise, v_bytes, nodes = support_pass(monkeypatch, n, inject=inject)
    assert rise-v_bytes-16*nodes > PASS_CONSTANT, (inject, rise, v_bytes, nodes)


# ---- 3. releases and holdings ---------------------------------------------------------------------------------------

def _solver_events(monkeypatch, domain):
    events = []
    support, g0 = ML.support_level, ML.g0_of

    def support_wrapper(*args, **kwargs):
        level = support(*args, **kwargs)
        events.append("support")
        return level

    def g0_wrapper(*args, **kwargs):
        events.append("g0")
        return g0(*args, **kwargs)
    monkeypatch.setattr(ML, "support_level", support_wrapper)
    monkeypatch.setattr(ML, "g0_of", g0_wrapper)
    monkeypatch.setattr(ML, "release_free_heap", lambda: events.append("release"))
    _, levels, _ = S.four_levels_structure()
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)
    config = SolverConfig(rounds=(2, 1, 1, 1), caps=Caps(1, 1), sigma_domain=domain, max_levels=2)
    solve_multilevel(NumpyBackend(), list(levels), eta=eta, sigma=GaussianMStep(), config=config)
    return events


@pytest.mark.parametrize("domain", ["support", "support_shared"])
def test_solver_releases_after_the_support_pass(monkeypatch, domain):
    # level 0 (two rounds), level 1 (one round): one support pass and one release per level, before the first G0
    assert _solver_events(monkeypatch, domain) == ["support", "release"]+["g0", "release"]*2+["support", "release",
                                                                                              "g0", "release"]


def test_solver_brain_releases_unchanged(monkeypatch):
    assert _solver_events(monkeypatch, "brain") == ["g0", "release"]*3


@pytest.mark.parametrize("arm", ["brain", "support_shared"])
def test_script_releases_and_drops_the_level_input(monkeypatch, arm):
    import bayesreg.heap as heap
    events = []
    monkeypatch.setattr(heap, "release_free_heap", lambda: events.append("release"))
    level, eta = level_input(12)
    moving = weakref.ref(level.moving)
    inputs = [level]
    del level

    def mark(label):
        gc.collect()
        events.append((label, moving() is not None))
    script().measure_arm(arm, inputs, eta, NumpyBackend(), np.zeros((12, 12, 12, 3)), theta=THETA, cpu_threads=1,
                         mark=mark)
    support = ["release"] if arm == "support_shared" else []
    assert events == [("swl.start", True), ("swl.prepare", False), ("swl.seed_evaluated", False)]+support+[
        ("swl.level_start", False), ("swl.state", False), "release", ("swl.g0", False), ("swl.eval", False)]
    assert inputs == [None]


def test_script_records_page_settings_and_smaps():
    module = script()
    settings = module.page_settings()
    assert set(settings) == {"numpy_madvise_hugepage", "thp_enabled", "thp_defrag", "khugepaged_max_ptes_none"}
    assert isinstance(settings["numpy_madvise_hugepage"], bool)       # the NumPy 2 attribute path works here
    smaps = module.smaps_rollup()
    assert {"smaps_rss_mb", "smaps_anon_mb", "smaps_anon_huge_mb", "smaps_locked_mb"} <= set(smaps)
    assert 0 < smaps["smaps_anon_mb"] <= smaps["smaps_rss_mb"]


# ---- 4. the gate 12 summary of an ABBA series (experiments/swl/gate12_summary.py) ----------------------------------

def summary_module():
    spec = importlib.util.spec_from_file_location("swlm_gate12_summary", ROOT/"experiments/swl/gate12_summary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V_BYTES = 13*2_098_661                                  # the native v (26.0 MiB)
BRAIN = [7000., 7040., 6990., 7060., 7020.]


def fake_run(arm, host, device=7908.3, held=0., threads=4, channels_hwm=5400.):
    """A fixed_state_memory.py result: host / device peaks (MiB) in the swl.g0 segment, a channel-preparation segment
    of channels_hwm MiB before swl.start; allocated heap 4000 MiB, plus v + 2.4 + held MiB from the level start on under
    support_shared."""
    shared = arm == "support_shared"
    extra = V_BYTES/2**20+2.4+held if shared else 0.
    live = {"swl.prepare": 4000., "swl.seed_evaluated": 6000.-(500. if shared else 0.)}   # the dropped seed point
    live.update({k: 4000.+extra for k in HELD})
    memtrace = [dict(label="channels.level_inputs", hwm_mb=channels_hwm)]
    memtrace += [dict(label=label, hwm_mb=host if label == "swl.g0" else host-500., smaps_anon_mb=3000.,
                      cuda_peak_mb=device if label == "swl.eval" else 1000.) for label in ("swl.start",)+LABELS]
    return dict(arm=arm, cpu_threads=threads, peaks=dict(host_hwm_mb=max(host, channels_hwm), device_peak_mb=device),
                malloc_live_mb=live, v_bytes=V_BYTES if shared else None, memtrace=memtrace, page_settings={})


def series(brain_hosts, shared_hosts, options=None, **common):
    runs = {}
    for i, (b, s) in enumerate(zip(brain_hosts, shared_hosts), start=1):
        runs["brain", i] = fake_run("brain", b, **common)
        runs["support_shared", i] = fake_run("support_shared", s, **common, **(options or {}).get(i, {}))
    return runs


BRAIN6 = BRAIN+[7010.]


def judge(runs, pairs, failures=None, warmup="ok"):
    """summarize with a finished warm-up run unless given (None = no warm-up record)."""
    if warmup == "ok":
        warmup = fake_run("support_shared", 7100.)
    return summary_module().summarize(runs, pairs, failures, warmup)


@pytest.mark.parametrize("brain, shared, options, expected", [
    (BRAIN, [h+30 for h in BRAIN], None, "通过"),
    (BRAIN, [h+30 for h in BRAIN], {3: dict(held=16.)}, "不通过（持有）"),         # one more length-n float64 held
    (BRAIN, [h+30 for h in BRAIN], {2: dict(device=7950.)}, "不通过（持有）"),
    (BRAIN, [h+70 for h in BRAIN], None, "不通过（分配器滞留）"),
    ([7000., 7140., 6990., 7060., 7020.], [h+30 for h in BRAIN], None, "未决（c）"),   # brain range 150 MiB
    ([7000., 7140., 6990., 7060., 7020.], [h+90 for h in BRAIN], None, "未决（c）"),   # 未决 over 分配器滞留
])
def test_gate12_summary_verdicts(brain, shared, options, expected):
    summary = judge(series(brain, shared, options), 5)
    assert summary["verdict"] == expected, summary
    assert len(summary["per_pair"]) == 5


def test_gate12_summary_host_peak_from_swl_segments_and_descriptive_items():
    # the channel preparation (9000 MiB) would be every run's overall peak and dilute (c) to 0: only swl.* counts
    shared = [h+30 for h in BRAIN]
    shared[1] += 60                                                  # pair 2: +90 MiB
    summary = judge(series(BRAIN, shared, channels_hwm=9000.), 5)
    assert summary["verdict"] == "通过"
    c, d = summary["criterion_c"], summary["descriptive"]
    assert c["median_difference_mb"] == pytest.approx(30.) and c["medians_mb"]["brain"] == 7020.
    assert d["host_peak_max_mb"] == {"brain": 7060., "support_shared": 7130.}
    assert d["pairs_host_difference_over_64_mib"] == 1
    assert summary["warmup"] == dict(failed=None, arm="support_shared", host_peak_mb=7100., device_peak_mb=7908.3)


def test_gate12_summary_prepare_warning():
    runs = series(BRAIN, [h+30 for h in BRAIN])
    runs["support_shared", 2]["malloc_live_mb"]["swl.prepare"] += 9.
    summary = judge(runs, 5)
    assert summary["verdict"] == "通过"                          # (a) names swl.level_start, state, g0, eval only
    assert [row["prepare_warning"] for row in summary["per_pair"]] == [False, True, False, False, False]


def test_gate12_summary_one_thread_series_is_descriptive():
    summary = judge(series(BRAIN[:2], [h+30 for h in BRAIN[:2]], threads=1), 2)
    assert summary["verdict"].startswith("只作描述") and summary["cpu_threads"] == [1]


def test_gate12_summary_decision_order_with_few_complete_pairs():
    # contract order (577b49b): holding, then failed runs, then "fewer than 5 complete pairs -> descriptive only"
    summary = judge(series(BRAIN[:3], [h+30 for h in BRAIN[:3]], {1: dict(held=16.)}), 3)
    assert summary["verdict"] == "不通过（持有）" and summary["criterion_a"]["failed_pairs"] == [1]
    runs = series(BRAIN, [h+30 for h in BRAIN])
    runs["brain", 2] = None                                          # 5 pairs, brain-2 failed: 4 complete pairs
    summary = judge(runs, 5)
    assert summary["verdict"] == "未决（运行失败）" and summary["failed_runs"] == {"brain-2": "no JSON"}
    assert summary["complete_pairs"] == [1, 3, 4, 5]
    summary = judge(series(BRAIN[:3], [h+30 for h in BRAIN[:3]]), 3)   # 3 complete pairs, no failure
    assert summary["verdict"].startswith("只作描述") and summary["complete_pairs"] == [1, 2, 3]


@pytest.mark.parametrize("pairs", [5, 6])
def test_gate12_summary_failed_runs(tmp_path, pairs):
    import json
    module = summary_module()
    hosts = BRAIN6[:pairs]
    runs = series(hosts, [h+30 for h in hosts])
    for (arm, i), run in runs.items():
        if (arm, i) != ("support_shared", 4):                    # a run killed before its JSON
            (tmp_path/f"{arm}-{i}.json").write_text(json.dumps(run))
    (tmp_path/"warmup.json").write_text(json.dumps(fake_run("support_shared", 7100.)))
    rows = ["tag\tarm\texit\toom_new", "warmup\tsupport_shared\t0\t0"]+[f"{arm}-{i}\t{arm}\t0\t0"
                                                                       for (arm, i) in runs]
    rows[rows.index("support_shared-4\tsupport_shared\t0\t0")] = "support_shared-4\tsupport_shared\t137\t1"
    (tmp_path/"runs.tsv").write_text("\n".join(rows)+"\n")
    assert module.main([str(tmp_path), "--pairs", str(pairs)]) == 0
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["verdict"] == "未决（运行失败）"
    assert summary["failed_runs"] == {"support_shared-4": "exit 137; 1 new OOM lines; no JSON"}
    assert summary["failure_difference"] == {"brain": [], "support_shared": ["support_shared-4"]}
    assert summary["complete_pairs"] == [i for i in range(1, pairs+1) if i != 4]
    # an OOM line without a failed exit counts too; a holding failure in the complete pairs still decides
    runs = series(hosts, [h+30 for h in hosts], {1: dict(held=16.)})
    failures = {f"{arm}-{i}": dict(exit="0", oom_new="1" if (arm, i) == ("brain", 3) else "0") for arm, i in runs}
    summary = judge(runs, pairs, failures)
    assert summary["failed_runs"] == {"brain-3": "1 new OOM lines"} and summary["verdict"] == "不通过（持有）"


def test_gate12_summary_review_probe_oom_in_support_shared():
    # the round-4 reviewer's probe: 5 pairs, support_shared-4 OOM -> 未决（运行失败）; with pair 1 holding one more
    # length-n float64 (16 MiB at the native n) as well -> 不通过（持有）
    runs = series(BRAIN, [h+30 for h in BRAIN])
    failures = {f"{arm}-{i}": dict(exit="0", oom_new="0") for arm, i in runs}
    failures["support_shared-4"] = dict(exit="137", oom_new="1")
    summary = judge(runs, 5, failures)
    assert summary["verdict"] == "未决（运行失败）"
    assert summary["failed_runs"] == {"support_shared-4": "exit 137; 1 new OOM lines"}
    held = series(BRAIN, [h+30 for h in BRAIN], {1: dict(held=16.)})
    assert judge(held, 5, failures)["verdict"] == "不通过（持有）"


def test_gate12_summary_failed_warmup(tmp_path):
    # R3-2: the warm-up's failure counts (contract: "运行失败……含预热"); at most 未决（运行失败） unless (a)(b) hold
    import json
    module = summary_module()
    runs = series(BRAIN, [h+30 for h in BRAIN])
    for (arm, i), run in runs.items():
        (tmp_path/f"{arm}-{i}.json").write_text(json.dumps(run))
    (tmp_path/"warmup.json").write_text(json.dumps(fake_run("support_shared", 7100.)))
    rows = ["tag\tarm\texit\toom_new", "warmup\tsupport_shared\t137\t1"]+[f"{arm}-{i}\t{arm}\t0\t0"
                                                                         for (arm, i) in runs]
    (tmp_path/"runs.tsv").write_text("\n".join(rows)+"\n")
    assert module.main([str(tmp_path), "--pairs", "5"]) == 0
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["verdict"] == "未决（运行失败）"                  # every pair passes, but the warm-up failed
    assert summary["failed_runs"] == {"warmup": "exit 137; 1 new OOM lines"}
    assert summary["warmup"]["failed"] == "exit 137; 1 new OOM lines"
    assert summary["complete_pairs"] == [1, 2, 3, 4, 5]
    # no warm-up record at all is a failure as well; a holding failure still decides
    assert judge(runs, 5, warmup=None)["failed_runs"] == {"warmup": "no JSON"}
    assert judge(runs, 5, warmup=None)["verdict"] == "未决（运行失败）"
    held = series(BRAIN, [h+30 for h in BRAIN], {2: dict(held=16.)})
    assert judge(held, 5, warmup=None)["verdict"] == "不通过（持有）"
