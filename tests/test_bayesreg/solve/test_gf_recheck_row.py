"""The C08-8(f) gf_recheck log row (work package GFR; a bug found outside the scope of review MEM_复核_20261031).

The torch64 backend's `confirm_gradient_norm` returns a record with its own key `context` ("pg" / "outer").  run_pg
logged it as `_row(row_context, kind="gf_recheck", **check)`, whose first parameter is also named `context`, so the
first GF_B inside [T_B/1.01, 1.01 T_B] raised TypeError ("got multiple values for argument 'context'"); TypeError is
not a ROUND_EXCEPTION, so run_pair ended with status exception (no hyper.json, no export).

Here the band is widened by monkeypatching bayesreg.backends.torch64.stable.GF_BAND (the module constant is read at
call time; the source value is not changed), so every GF_B of a small synthetic problem (torch64 on the CPU device)
goes through the recheck:
- `_row` keeps the record's `context` as an ordinary key and every other row unchanged;
- option (a) (solve_direct -> run_pg): one gf_recheck row per gradient norm, each the backend record (all its keys,
  context "pg", energy "F_B") after the run_pg row context, kind and iteration -- the key order of the outer rows
  (context, kind, counter, record; FUP1, review GFR minor 1); the norms the solver used are the CPU values G_cpu
  (C08-8(f)), and at least one G_gpu differs from its G_cpu, so that check can tell the two apart (FUP1, minor 2);
- option (b) (solve_proximal): the outer rows (context "outer", the commit count) and the inner run_pg rows (context
  "pg", energy "E_v"); the outer norms the solver used are the outer rows' G_cpu (FUP1, minor 3);
- run_pair end to end (torch64, --device cpu): exit 0, not status exception, hyper.json written, gf_recheck rows in
  log.jsonl.
Before the GFR fix all four tests failed with that TypeError (the run_pair one through status exception, exit 1): on
2830d48, the base of the GFR branch (review GFR), and on its ancestor 9de1f94 (FUP1, review GFR minor 4).  The key
order of the pg row changed in FUP1 (iteration after kind); the row had never been written before GFR.
"""
import json

import numpy as np
import pytest

from jd_curl.three_d.boundary import restrict_free

from bayesreg.backends.torch64 import stable as S
from bayesreg.model import ModelState, build_samplers
from bayesreg.solve import (Caps, Commit, ProximalParams, default_lambda, g0_of, make_backend, solve_direct,
                            solve_proximal, threshold)
from bayesreg.solve.pg import _row

from . import _synthetic as syn

CHECK_KEYS = ("context", "energy", "G_gpu", "G_cpu", "target", "changed_decision", "seconds")


@pytest.fixture
def wide_band(monkeypatch):
    """Every GF_B lies in [T_B/GF_BAND, GF_BAND T_B]: each one is rechecked on the CPU."""
    monkeypatch.setattr(S, "GF_BAND", 1e30)


def _problem(n=9):
    grid = syn.cube_grid(n)
    u_true = syn.solenoidal_displacement(grid, .4)
    moving = syn.moving_image(grid.identity())
    fixed = syn.warp_with_spline(moving, grid, u_true)
    state = ModelState(grid=grid, omega0=grid.omega, fixed=fixed[None], samplers=build_samplers(moving[None], grid),
                       mask=syn.brain_box(grid, 1), sigma2=np.array([1.]), alpha=1., nu=1.,
                       tau2=np.full(4, 1.), tissue=syn.uniform_tissue(grid))
    return grid, state


def _start(backend, grid):
    zero = np.zeros(grid.shape+(3,))
    return Commit(zero, restrict_free(zero, grid), backend.certify(zero, grid))


def _torch64(state):
    backend = make_backend("torch64", device="cpu")
    return backend, backend.prepare(state)


def test_row_keeps_a_context_key_and_other_rows_unchanged():
    """`_row(context, **values)`: a value named `context` is an ordinary key of the row (it was a TypeError); rows
    without one are the same dicts, in the same key order, as before.  (_row itself; the gf_recheck row of run_pg
    passes kind before iteration since FUP1, see test_option_a_recheck_rows.)"""
    ctx = dict(level=2, round=0, iteration=3)
    row = _row(ctx, kind="gf_recheck", context="pg", energy="F_B", G_gpu=1.)
    assert row == dict(level=2, round=0, iteration=3, kind="gf_recheck", context="pg", energy="F_B", G_gpu=1.)
    assert list(row) == ["level", "round", "iteration", "kind", "context", "energy", "G_gpu"]
    assert ctx == dict(level=2, round=0, iteration=3)                       # the caller's dict is not changed
    other = _row(ctx, kind="accept", iteration=4, tau=.5)
    assert list(other) == ["level", "round", "iteration", "kind", "tau"] and other["iteration"] == 4


def test_option_a_recheck_rows(wide_band):
    """solve_direct (run_pg) on torch64: the recheck at iteration 0 and after every accepted step is logged as
    dict(row context, kind="gf_recheck", iteration, **record) -- the outer rows' order (context, kind, counter,
    record); the solver's norms are the CPU values, and G_gpu != G_cpu somewhere, so the check tells them apart."""
    grid, state = _problem()
    backend, st = _torch64(state)
    target = threshold(g0_of(backend, st))
    rows = []
    context = dict(level=0, round=0)
    result = solve_direct(backend, st, _start(backend, grid), target=target, caps=Caps(max_accepted=3),
                          log=rows.append, context=context)
    assert result.status in ("cap_reached", "converged_gradient") and result.accepted >= 1
    rechecks = [r for r in rows if r["kind"] == "gf_recheck"]
    records = backend.comparator.stats.gf_checks
    assert len(rechecks) == len(records) == len(result.gradient_norms) == result.accepted+1
    accepts = [r["iteration"] for r in rows if r["kind"] == "accept"]
    for row, record, iteration in zip(rechecks, records, [0]+accepts):
        assert tuple(record) == CHECK_KEYS
        expected = dict(context, kind="gf_recheck", iteration=iteration, **record)
        assert row == expected and list(row) == list(expected)
        assert list(row) == ["level", "round", "kind", "iteration", *CHECK_KEYS]       # the outer rows' order
        assert row["context"] == "pg" and row["energy"] == "F_B" and row["target"] == target
    # C08-8(f): the value that enters the stopping decision is the CPU one
    assert list(result.gradient_norms) == [r["G_cpu"] for r in rechecks]
    assert result.gradient_norm == rechecks[-1]["G_cpu"]
    # the check above can tell the CPU value from the device one only where they differ (FUP1, review GFR minor 2)
    assert any(r["G_gpu"] != r["G_cpu"] for r in rechecks)
    json.dumps(rechecks)                                                     # log.jsonl-ready


def test_option_b_recheck_rows(wide_band):
    """solve_proximal on torch64: the outer rows (context "outer", F_B, the commit count) and the inner run_pg rows
    (context "pg", E_v, the outer index of the run_pg context and the iteration).  Key order: the outer rows
    (context, kind, outer, record) and the inner rows (run_pg context with outer, kind, iteration, record).  The outer
    gradient norms the solver used are the outer rows' G_cpu (C08-8(f); FUP1, review GFR minor 3)."""
    grid, state = _problem()
    backend, st = _torch64(state)
    target = threshold(g0_of(backend, st))
    rows = []
    result = solve_proximal(backend, st, _start(backend, grid), target=target, caps=Caps(max_outer=2, max_accepted=4),
                            params=ProximalParams(lam=default_lambda(grid), n_in=2), log=rows.append,
                            context=dict(level=0, round=0))
    assert result.commits >= 1
    rechecks = [r for r in rows if r["kind"] == "gf_recheck"]
    outer =[r for r in rechecks if r["context"] == "outer"]
    inner = [r for r in rechecks if r["context"] == "pg"]
    assert len(outer)+len(inner) == len(rechecks) == len(backend.comparator.stats.gf_checks)
    assert outer and all(r["energy"] == "F_B" for r in outer)
    assert inner and all(r["energy"] == "E_v" and "iteration" in r and "outer" in r for r in inner)
    for row, record in zip(rechecks, backend.comparator.stats.gf_checks):
        assert {k: row[k] for k in CHECK_KEYS} == record
        assert row["level"] == 0 and row["round"] == 0
    assert [r["outer"] for r in outer] == list(range(len(outer)))
    assert all(list(r) == ["level", "round", "kind", "outer", *CHECK_KEYS] for r in outer)
    assert all(list(r) == ["level", "round", "outer", "kind", "iteration", *CHECK_KEYS] for r in inner)
    # C08-8(f): the outer norms are the CPU values of the outer rows (one row per norm here)
    assert list(result.gradient_norms) == [r["G_cpu"] for r in outer]
    assert any(r["G_gpu"] != r["G_cpu"] for r in outer)
    json.dumps(rechecks)


def test_run_pair_torch64_with_rechecks(wide_band, tmp_path):
    """run_pair.main --backend torch64 --device cpu on the synthetic 26^3 pair of test_run_pair: with every GF_B
    rechecked the run ends normally (exit 0, cap_reached, hyper.json and the export written); log.jsonl carries the
    gf_recheck rows of run_pg."""
    from bayesreg.cli import run_pair
    from ..io_seg._helpers import DEV, allowlist_document, write_json
    from .test_run_pair import write_synthetic_pair, write_tissue
    allowlist = write_json(tmp_path/"dev_subjects.json", allowlist_document())
    grid, pair = write_synthetic_pair(tmp_path, DEV[0], DEV[1])
    write_tissue(tmp_path, grid, DEV[0])
    write_tissue(tmp_path, grid, DEV[1], shift=(.7, -.5, .4))
    out = tmp_path/"run_torch64"
    code = run_pair.main(["--pair", pair, "--channel-source", "tissue", "--out", str(out), "--data-root",
                          str(tmp_path/"data"), "--allowlist", str(allowlist), "--tissue-root", str(tmp_path/"seg"),
                          "--rounds", "1,1,1,1", "--max-accepted", "2", "--backend", "torch64", "--device", "cpu",
                          "--cpu-threads", "2"])
    status = json.loads((out/"status.json").read_text())
    assert status["status"] != "exception", status
    assert code == 0 and status["status"] == "cap_reached" and status["export"]["certificate_passed"]
    hyper = json.loads((out/"hyper.json").read_text())
    assert hyper["gpu_comparison"]["gf_rechecks"]["count"] >= 1 and hyper["GF_B_cpu_final"] is not None
    rows = [json.loads(line) for line in (out/"log.jsonl").read_text().splitlines()]
    rechecks = [r for r in rows if r["kind"] == "gf_recheck"]
    assert rechecks and all(set(CHECK_KEYS) <= set(r) and r["context"] == "pg" and r["energy"] == "F_B"
                            for r in rechecks)
    assert (out/"disp.npz").is_file()
