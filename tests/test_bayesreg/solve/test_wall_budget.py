"""Development wall-clock budget (--wall-seconds; contract C13-3 appendix, C20 D44), synthetic data only.

A fake clock (WallBudget(clock=...)) is moved past the budget by the solver's own log rows, so each case stops at a
known safe point:
- WallBudget: validation, the first expiry is recorded once, budget_expired(None) is False; status vocabulary
  (budget_exhausted in the inner / round / run sets, wording 预算收尾, never 收敛, not a C13-7 continue row);
- run_pg: at the top of an iteration (the last accepted point is kept; a natural ending checked first wins) and
  inside a line search (the base is not moved, the rejection of the last candidate recorded);
- solve_multilevel, option (a): native level -> budget_exhausted with the last formal commit, exported with a fresh
  CPU certificate PASS and disp.npz; coarse level -> stopped_before_native with that level's last commit; before a
  transfer -> stopped_before_native at the previous level, no transfer row;
- option (b): an outer commit is kept, uncommitted inner progress is dropped;
- run_pair: a tiny --wall-seconds ends stopped_before_native (final_level0.npz, no disp.npz, exit 2) with the
  records in config.json / provenance.json / hyper.json / status.json / timing.json; a budget used up on the native
  level ends budget_exhausted (exit 0, disp.npz, cert.json PASS) and the evaluator's run description reads it as
  native_formal; invalid values are refused by the parser.
GS review item 5 (added by GS1): the default path -- budget None and a budget that never expires give bitwise the same
state_hash, commit, round records and log rows (options (a) and (b)); the native level reaching T_B exactly when the
budget is used up ends converged_regularized_gradient_B (the natural ending wins); a native level with zero commits
used up -> budget_exhausted with the certified native seed, which the evaluator reads as handover_only; option (b)
used up on the native level (after an outer commit / with uncommitted inner progress); a budget ending whose final
certificate fails -> final_certificate_rejected (exit 2, no disp.npz). GS review item 6: the budget check before a
transfer comes before the level's template is built.
"""
import hashlib

import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.cli import run_pair
from bayesreg.eval import fields as eval_fields
from bayesreg.solve import (Caps, Eta, FixedSigma, NumpyBackend, ProximalParams, SolverConfig, default_lambda,
                            export_native, run_pg, solve_multilevel)
from bayesreg.solve import jd_scaling as jds
from bayesreg.solve import multilevel as multilevel_module
from bayesreg.solve import stopping as st
from bayesreg.solve.stopping import (BUDGET_EXHAUSTED, CONVERGED_B, FINAL_CERTIFICATE_REJECTED, STOPPED_BEFORE_NATIVE,
                                     WallBudget, budget_expired, state_hash)

from . import _synthetic as syn
from ..io_seg._helpers import DEV
from .test_pg import FakeObjective, Identity, Red, passing, small_state, start
from .test_run_pair import read, setup, write_synthetic_pair, write_tissue  # noqa: F401  (fixture)
from .test_s1_fixes import FailingCertificate


class Clock:
    """A settable clock; `jump()` moves it far past any budget."""

    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t

    def jump(self):
        self.t = 1e9


def jumping_log(rows, clock, when):
    """log callback: keeps every row and jumps the clock after the first row for which when(row) is true."""
    def log(row):
        rows.append(row)
        if when(row):
            clock.jump()
    return log


# ---- WallBudget and the status vocabulary ------------------------------------------------------------------

def test_wall_budget_records_the_first_expiry_once():
    clock = Clock()
    budget = WallBudget(10., clock=clock)
    assert not budget.expired("a") and budget.hit is None
    clock.t = 9.999
    assert not budget.expired("b")
    clock.t = 10.
    assert budget.expired("c")
    clock.t = 50.
    assert budget.expired("d")
    record = budget.record()
    assert record["exhausted"] and record["hit"] == dict(where="c", elapsed_seconds=10., check=3)
    assert record["checks"] == 4 and record["wall_seconds"] == 10. and record["clock"] == "injected"
    assert record["elapsed_seconds"] == 50.
    assert not budget_expired(None, "anywhere")
    for bad in (0., -1., float("nan"), float("inf")):
        with pytest.raises(InputValidationError):
            WallBudget(bad)
    real = WallBudget(3600.)
    assert not real.expired() and 0 <= real.elapsed() < 60 and real.record()["clock"] == "time.monotonic"


def test_budget_status_vocabulary():
    assert BUDGET_EXHAUSTED in st.INNER_STATUSES and BUDGET_EXHAUSTED in st.ROUND_STATUSES
    assert BUDGET_EXHAUSTED in st.RUN_STATUSES and st.RUN_WORDING[BUDGET_EXHAUSTED] == "预算收尾"
    assert "收敛" not in st.RUN_WORDING[BUDGET_EXHAUSTED]
    assert BUDGET_EXHAUSTED not in st.FINAL_STATUS and BUDGET_EXHAUSTED not in st.CONTINUE_ALWAYS+st.CONTINUE_IF_SAFE
    assert not st.continuation(BUDGET_EXHAUSTED).proceed           # the level loop ends the run before the table
    assert BUDGET_EXHAUSTED in run_pair.ENDINGS and STOPPED_BEFORE_NATIVE not in run_pair.ENDINGS


# ---- run_pg ------------------------------------------------------------------------------------------------

def test_pg_budget_at_the_iteration_top_keeps_the_last_accepted_point():
    state = small_state()
    clock = Clock()
    budget = WallBudget(5., clock=clock)
    objective = FakeObjective(state, [Red(1., 0.)]*3)
    base, theta, cert = start(state)
    accepted = []
    result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, target=0.,
                    max_steps=10, budget=budget,
                    on_accept=lambda th, point, c: accepted.append((th, point, c)) or clock.jump())
    assert result.status == BUDGET_EXHAUSTED and result.accepted == 1 and len(accepted) == 1
    assert result.theta is accepted[0][0] and result.point is accepted[0][1] and result.cert is accepted[0][2]
    assert "used up before iteration 2" in result.reason
    assert budget.hit["where"]["iteration"] == 2 and budget.hit["where"]["attempt"] == 0
    # a natural ending found at the same check wins: max_steps = 1 ends max_iterations, the budget is not consulted
    clock2 = Clock()
    budget2 = WallBudget(5., clock=clock2)
    result = run_pg(FakeObjective(state, [Red(1., 0.)]), base, theta=theta, cert=cert, metric=Identity(),
                    certify=passing, target=0., max_steps=1, budget=budget2,
                    on_accept=lambda *a: clock2.jump())
    assert result.status == "max_iterations" and budget2.hit is None


def test_pg_budget_inside_a_line_search_does_not_move_the_base():
    state = small_state()
    clock = Clock()
    budget = WallBudget(5., clock=clock)

    class Jumping(FakeObjective):
        def reduction(self, base, trial):
            clock.jump()
            return super().reduction(base, trial)
    objective = Jumping(state, [Red(-1., 1e-12)])
    base, theta, cert = start(state)
    rows = []
    result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, target=0.,
                    max_steps=10, budget=budget, log=rows.append)
    assert result.status == BUDGET_EXHAUSTED and result.accepted == 0
    assert result.theta.tobytes() == theta.tobytes() and result.point is base and result.cert is cert
    assert result.last_iteration == [("NO_STRICT_DECREASE", None)] and result.counters.candidates == 1
    assert [r["reason"] for r in rows if r["kind"] == "candidate"] == ["NO_STRICT_DECREASE"]
    assert budget.hit["where"]["iteration"] == 1 and budget.hit["where"]["attempt"] == 1
    assert "line search" in result.reason


# ---- solve_multilevel --------------------------------------------------------------------------------------

ETA = Eta(1., 1., (1., 1., 1., 1.))


def multilevel(when, *, levels, config, backend=None):
    clock = Clock()
    budget = WallBudget(100., clock=clock)
    rows = []
    backend = NumpyBackend() if backend is None else backend
    result = solve_multilevel(backend, levels, eta=ETA, sigma=FixedSigma([1.]), config=config,
                              log=jumping_log(rows, clock, when), budget=budget)
    return result, rows, budget, backend


def test_native_budget_exports_the_last_formal_commit(tmp_path):
    """Option (a): the clock jumps after the first native accepted step; the round ends budget_exhausted at the next
    iteration top, the run budget_exhausted (预算收尾) on the native level with that step as its commit; the export
    re-certifies it on the CPU (PASS) and writes disp.npz; final_native.npz carries the status."""
    native, levels, _ = syn.four_levels(26)
    result, rows, budget, backend = multilevel(
        lambda r: r["kind"] == "accept" and r["level"] == 3 and r["accepted_steps"] == 1, levels=levels,
        config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3)))
    native_accepts = [r for r in rows if r["kind"] == "accept" and r["level"] == 3]
    assert result.status == BUDGET_EXHAUSTED and result.wording == "预算收尾" and result.native and result.level == 3
    assert len(native_accepts) == 1
    assert result.commit.cert.state_hash == native_accepts[0]["state_hash"] == state_hash(result.commit.u)
    row = result.rounds[-1]
    assert row["level"] == 3 and row["status"] == BUDGET_EXHAUSTED and row["final"] and row["accepted"] == 1
    assert row["commits"] == 1 and row["continuation"]["proceed"] is False and row["GF_B_cpu"] is not None
    assert row["sigma2_post"] is None and row["budget"]["exhausted"]
    assert [r["status"] for r in result.rounds[:3]] == ["cap_reached"]*3
    assert result.sigma_post is None and result.budget["exhausted"]
    assert result.budget["hit"]["where"]["level"] == 3 and result.budget["hit"]["where"]["iteration"] == 2
    out = export_native(tmp_path, u=result.commit.u, grid=native, status=result.status, backend=backend,
                        solver_cert=result.commit.cert)
    assert out["certificate_passed"] and (tmp_path/"disp.npz").is_file()
    with np.load(tmp_path/"final_native.npz") as z:
        assert str(z["status"]) == BUDGET_EXHAUSTED


def test_coarse_budget_stops_before_native():
    native, levels, _ = syn.four_levels(26)
    result, rows, budget, _ = multilevel(
        lambda r: r["kind"] == "accept" and r["level"] == 1 and r["accepted_steps"] == 1, levels=levels,
        config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3)))
    level1 = [r for r in rows if r["kind"] == "accept" and r["level"] == 1]
    assert result.status == STOPPED_BEFORE_NATIVE and not result.native and result.level == 1
    assert "--wall-seconds" in result.reason and BUDGET_EXHAUSTED in result.reason
    assert result.commit.cert.state_hash == level1[-1]["state_hash"] and len(level1) == 1
    assert [r["level"] for r in result.rounds] == [0, 1] and result.rounds[-1]["status"] == BUDGET_EXHAUSTED
    assert not any(r.get("level", 0) >= 2 for r in rows)
    assert [t["to_level"] for t in result.transfers] == [1]


class CountingPrepare(NumpyBackend):
    """Records the grid shape of every prepared ModelState (the level templates and the round states)."""

    def __init__(self):
        super().__init__()
        self.prepared = []

    def prepare(self, state):
        self.prepared.append(tuple(state.grid.shape))
        return super().prepare(state)


def test_budget_before_a_transfer_stops_at_the_previous_level():
    native, levels, _ = syn.four_levels(26)
    backend = CountingPrepare()
    result, rows, budget, _ = multilevel(lambda r: r["kind"] == "round" and r["level"] == 0, levels=levels,
                                         config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=2)),
                                         backend=backend)
    # GS review item 6: the check comes before the level-1 template is built
    assert set(backend.prepared) == {tuple(levels[0].grid.shape)} and levels[1].grid.shape != levels[0].grid.shape
    assert result.status == STOPPED_BEFORE_NATIVE and result.level == 0 and not result.native
    assert result.transfers == [] and result.rounds[-1]["status"] == "cap_reached"
    assert result.commit.cert.state_hash == result.rounds[-1]["state_hash"]
    assert budget.hit["where"] == dict(level=1, at="before_transfer")
    assert [r for r in rows if r["kind"] == "budget"][0]["level"] == 1
    assert "before the transfer 0 -> 1" in result.reason


def proximal_config(levels):
    return SolverConfig(outer="b", rounds=(1, 1, 1, 1), caps=Caps(max_accepted=40, max_outer=5), max_levels=1,
                        proximal=ProximalParams(lam=default_lambda(levels[-1].grid)))


def test_option_b_keeps_the_outer_commit():
    native, levels, _ = syn.four_levels(26)
    result, rows, budget, _ = multilevel(lambda r: r["kind"] == "commit" and r["outer"] == 1, levels=levels,
                                         config=proximal_config(levels))
    commits = [r for r in rows if r["kind"] == "commit"]
    assert len(commits) == 1 and result.status == STOPPED_BEFORE_NATIVE
    row = result.rounds[-1]
    assert row["status"] == BUDGET_EXHAUSTED and row["commits"] == 1 and "outer iteration 2" in row["reason"]
    assert result.commit.cert.state_hash == commits[0]["state_hash"] == state_hash(result.commit.u)
    assert budget.hit["where"]["at"] == "outer_iteration"


def test_option_b_drops_uncommitted_inner_progress():
    native, levels, _ = syn.four_levels(26)
    result, rows, budget, _ = multilevel(lambda r: r["kind"] == "accept" and r["outer"] == 0, levels=levels,
                                         config=proximal_config(levels))
    row = result.rounds[-1]
    assert row["status"] == BUDGET_EXHAUSTED and row["commits"] == 0 and row["accepted"] == 1
    assert row["inner_status"] == BUDGET_EXHAUSTED and "uncommitted" in row["reason"]
    assert not any(r["kind"] == "commit" for r in rows)
    assert not np.any(result.commit.u)                     # the round's start state (u = 0), certified
    assert result.commit.cert.passed and result.status == STOPPED_BEFORE_NATIVE


# ---- GS review item 5: the default path, the natural ending at the budget point, native-level endings ------------

def _clock_key(key):
    return key in ("seconds", "timing") or str(key).endswith("_seconds")


def canonical(value):
    """A bitwise-comparable image of a record: floats by their hex form, arrays by dtype, shape and the sha256 of their
    bytes; wall-clock fields (seconds, *_seconds, timing) dropped -- they differ between any two runs."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if not _clock_key(k)}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return ("float", float(value).hex())
    if isinstance(value, np.integer):
        return int(value)
    return value


def four_level_option_b(levels, **caps):
    return SolverConfig(outer="b", rounds=(1, 1, 1, 1), caps=Caps(**caps),
                        proximal=ProximalParams(lam=default_lambda(levels[-1].grid)))


@pytest.mark.parametrize("option", ["a", "b"])
def test_never_expiring_budget_is_bitwise_the_default_path(option):
    """budget=None against WallBudget(1e9) on the real clock (never expires): the same status, commit, state_hash,
    round and transfer records and log rows, bit for bit (wall-clock fields excepted); the budget was consulted."""
    native, levels, _ = syn.four_levels(26)
    config = (SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3)) if option == "a"
              else proximal_config(levels))
    runs = []
    for budget in (None, WallBudget(1e9)):
        rows = []
        # DEF: the default cell estimate is "certificate", whose per-cell bounds are cached across calls by state hash
        # (C09.1-3a, JDC); both runs start with an empty cache, so that the records (cell_bounds_cached) compare
        jds.clear_cell_bounds_cache()
        result = solve_multilevel(NumpyBackend(), levels, eta=ETA, sigma=FixedSigma([1.]), config=config,
                                  log=rows.append, budget=budget)
        runs.append((result, rows, budget))
    (plain, plain_rows, _), (timed, timed_rows, budget) = runs
    assert budget.checks > 0 and budget.hit is None and plain.budget is None and timed.budget["exhausted"] is False
    assert (plain.status, plain.reason, plain.wording, plain.level, plain.native) == (
        timed.status, timed.reason, timed.wording, timed.level, timed.native)
    assert plain.commit.u.tobytes() == timed.commit.u.tobytes()
    assert plain.commit.cert.state_hash == timed.commit.cert.state_hash == state_hash(timed.commit.u)
    assert canonical(plain.rounds) == canonical(timed.rounds)
    assert canonical(plain.transfers) == canonical(timed.transfers)
    assert canonical(plain.sigma) == canonical(timed.sigma) and canonical(plain.sigma_post) == canonical(timed.sigma_post)
    assert len(plain_rows) == len(timed_rows) and canonical(plain_rows) == canonical(timed_rows)
    assert not any(r.get("kind") == "budget" for r in timed_rows)


def test_native_threshold_reached_at_the_budget_point_is_convergence(monkeypatch):
    """The native T_B is set to the GF_B of the first native accepted step (taken from a run without a budget), and the
    clock jumps past the budget at that same step: at the next safe point the threshold is checked first, so the round
    ends converged_gradient and the run converged_regularized_gradient_B ("B 线收敛"), the budget never recorded as
    hit although it is used up."""
    native, levels, _ = syn.four_levels(26)
    config = SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3))
    first = solve_multilevel(NumpyBackend(), levels, eta=ETA, sigma=FixedSigma([1.]), config=config)
    trace = first.rounds[-1]["GF_B_trace"]
    assert first.rounds[-1]["level"] == 3 and first.rounds[-1]["accepted"] >= 1 and trace[1] < trace[0]
    target = trace[1]
    real, calls = multilevel_module.threshold, []

    def native_target(g0, **kwargs):
        calls.append(g0)
        return target if len(calls) == 4 else real(g0, **kwargs)      # rounds (1, 1, 1, 1): the fourth is native
    monkeypatch.setattr(multilevel_module, "threshold", native_target)
    result, rows, budget, _ = multilevel(lambda r: r["kind"] == "accept" and r["level"] == 3, levels=levels,
                                         config=config)
    assert result.status == CONVERGED_B and result.wording == "B 线收敛" and result.native and result.level == 3
    row = result.rounds[-1]
    assert row["level"] == 3 and row["status"] == "converged_gradient" and row["accepted"] == 1 and row["final"]
    assert row["T_B"] == target and row["GF_B"] <= row["T_B"] and row["sigma2_post"] is not None
    assert budget.hit is None and budget.elapsed() >= budget.seconds and not result.budget["exhausted"]
    assert [r["status"] for r in result.rounds[:3]] == ["cap_reached"]*3 and len(calls) == 4


def test_native_budget_with_zero_commits_keeps_the_native_seed(tmp_path):
    """The clock jumps at the transfer into the native level: the first native safe point (iteration 1) finds the
    budget used up; the round has zero commits and the run ends budget_exhausted with the certified native seed, which
    is exported (fresh CPU PASS)."""
    native, levels, _ = syn.four_levels(26)
    result, rows, budget, backend = multilevel(lambda r: r["kind"] == "transfer" and r["to_level"] == 3, levels=levels,
                                               config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3)))
    assert result.status == BUDGET_EXHAUSTED and result.native and result.level == 3
    row = result.rounds[-1]
    assert row["level"] == 3 and row["status"] == BUDGET_EXHAUSTED and row["accepted"] == 0 and row["commits"] == 0
    assert budget.hit["where"]["level"] == 3 and budget.hit["where"]["iteration"] == 1
    seed_row = result.transfers[-1]["rows"][result.transfers[-1]["accepted_k"]]
    assert result.commit.cert.state_hash == seed_row["cert"]["state_hash"] == state_hash(result.commit.u)
    assert not any(r["kind"] == "accept" and r.get("level") == 3 for r in rows)
    out = export_native(tmp_path, u=result.commit.u, grid=native, status=result.status, backend=backend,
                        solver_cert=result.commit.cert)
    assert out["certificate_passed"] and (tmp_path/"disp.npz").is_file()


@pytest.mark.parametrize("trigger", ["after_commit", "inner_uncommitted"])
def test_option_b_budget_on_the_native_level(tmp_path, trigger):
    """Option (b) through four levels; the clock jumps on the native level either at the first outer commit (the round
    keeps it: budget found at the top of outer iteration 2) or at the first inner accepted step before any commit (the
    inner ends budget_exhausted, its progress is dropped, the native seed stays the commit). Both end the run
    budget_exhausted on the native level and export a certified state."""
    native, levels, _ = syn.four_levels(26)
    when = ((lambda r: r["kind"] == "commit" and r.get("level") == 3 and r["outer"] == 1) if trigger == "after_commit"
            else (lambda r: r["kind"] == "accept" and r.get("level") == 3 and r["outer"] == 0))
    result, rows, budget, backend = multilevel(when, levels=levels,
                                               config=four_level_option_b(levels, max_accepted=40, max_outer=2))
    assert result.status == BUDGET_EXHAUSTED and result.native and result.level == 3
    row = result.rounds[-1]
    assert row["level"] == 3 and row["status"] == BUDGET_EXHAUSTED and row["final"]
    native_commits = [r for r in rows if r["kind"] == "commit" and r.get("level") == 3]
    if trigger == "after_commit":
        assert row["commits"] == 1 and len(native_commits) == 1 and "outer iteration 2" in row["reason"]
        assert budget.hit["where"]["at"] == "outer_iteration" and budget.hit["where"]["level"] == 3
        assert result.commit.cert.state_hash == native_commits[0]["state_hash"]
    else:
        assert row["commits"] == 0 and not native_commits and row["accepted"] >= 1
        assert row["inner_status"] == BUDGET_EXHAUSTED and "uncommitted" in row["reason"]
        seed_row = result.transfers[-1]["rows"][result.transfers[-1]["accepted_k"]]
        assert result.commit.cert.state_hash == seed_row["cert"]["state_hash"]
    assert result.commit.cert.state_hash == state_hash(result.commit.u)
    out = export_native(tmp_path, u=result.commit.u, grid=native, status=result.status, backend=backend,
                        solver_cert=result.commit.cert)
    assert out["certificate_passed"] and (tmp_path/"disp.npz").is_file()


# ---- run_pair ----------------------------------------------------------------------------------------------

def test_run_pair_tiny_budget_stops_before_native(setup):
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    out = root/"run_tiny_budget"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--wall-seconds", "1e-6")) == 2
    status = read(out/"status.json")
    assert status["status"] == STOPPED_BEFORE_NATIVE and status["exit_code"] == 2 and not status["native"]
    assert "--wall-seconds" in status["reason"] and status["wall_budget"]["exhausted"]
    assert status["wall_budget"]["hit"]["where"]["level"] == 0
    assert (out/"final_level0.npz").is_file()
    for name in ("disp.npz", "final_native.npz", "cert.json"):
        assert not (out/name).exists()
    hyper = read(out/"hyper.json")
    assert [(r["level"], r["status"], r["accepted"]) for r in hyper["rounds"]] == [(0, BUDGET_EXHAUSTED, 0)]
    assert hyper["wall_budget"]["exhausted"]
    config = read(out/"config.json")
    assert config["args"]["wall_seconds"] == 1e-6 and config["wall_budget"]["wall_seconds"] == 1e-6
    assert read(out/"provenance.json")["switches"]["wall_seconds"] == 1e-6
    assert read(out/"timing.json")["wall_budget"]["exhausted"]
    with np.load(out/"final_level0.npz") as z:
        assert not np.any(z["u"]) and str(z["status"]) == STOPPED_BEFORE_NATIVE


@pytest.mark.parametrize("backend", [(), ("--backend", "torch64", "--device", "cpu", "--cpu-threads", "1")],
                         ids=["numpy", "torch64-cpu"])
def test_run_pair_native_budget_is_a_budget_ending(setup, monkeypatch, backend):
    """The clock jumps after the first native accepted step: budget_exhausted (预算收尾, exit 0), disp.npz and a fresh
    CPU PASS in cert.json; the evaluator describes the run as native_formal with its run status."""
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    clock = Clock()
    monkeypatch.setattr(run_pair, "make_budget", lambda seconds: WallBudget(seconds, clock=clock))
    real = run_pair.solve_multilevel

    def spying(backend, inputs, *, log, **kwargs):
        def spy(row):
            log(row)
            if row.get("kind") == "accept" and row.get("level") == 3:
                clock.jump()
        return real(backend, inputs, log=spy, **kwargs)
    monkeypatch.setattr(run_pair, "solve_multilevel", spying)
    out = root/"run_native_budget"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "2", "--wall-seconds", "2700",
                              *backend)) == 0
    status = read(out/"status.json")
    assert status["status"] == BUDGET_EXHAUSTED and status["wording"] == "预算收尾" and status["native"]
    assert status["export"]["certificate_passed"] and status["wall_budget"]["hit"]["where"]["level"] == 3
    assert (out/"disp.npz").is_file() and read(out/"cert.json")["status"] == "certified_feasible"
    with np.load(out/"final_native.npz") as z:
        assert str(z["status"]) == BUDGET_EXHAUSTED
    hyper = read(out/"hyper.json")
    assert hyper["rounds"][-1]["status"] == BUDGET_EXHAUSTED and hyper["rounds"][-1]["commits"] == 1
    assert hyper["GF_B_cpu_final"] is not None
    info = eval_fields.describe_run(out)
    assert info.kind == "bayesreg" and info.run_status == BUDGET_EXHAUSTED
    assert info.field_role == "native_formal" and eval_fields.field_present(info)


def jump_in_run_pair(monkeypatch, when):
    """run_pair with an injected budget clock that jumps after the first log row for which when(row) is true."""
    clock = Clock()
    monkeypatch.setattr(run_pair, "make_budget", lambda seconds: WallBudget(seconds, clock=clock))
    real = run_pair.solve_multilevel

    def spying(backend, inputs, *, log, **kwargs):
        def spy(row):
            log(row)
            if when(row):
                clock.jump()
        return real(backend, inputs, log=spy, **kwargs)
    monkeypatch.setattr(run_pair, "solve_multilevel", spying)


def test_run_pair_native_budget_with_zero_commits_is_handover_only(setup, monkeypatch):
    """The budget is used up at the transfer into the native level: budget_exhausted (exit 0) with the certified native
    seed exported (disp.npz, fresh CPU PASS) and zero native commits, so the evaluator's field_role is handover_only --
    the case the smoke comparison lists apart from A0 (GS review item 4)."""
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    jump_in_run_pair(monkeypatch, lambda row: row.get("kind") == "transfer" and row.get("to_level") == 3)
    out = root/"run_native_zero_commits"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "2", "--wall-seconds", "2700")) == 0
    status = read(out/"status.json")
    assert status["status"] == BUDGET_EXHAUSTED and status["native"] and status["export"]["certificate_passed"]
    assert status["wall_budget"]["hit"]["where"]["level"] == 3 and status["wall_budget"]["hit"]["where"]["iteration"] == 1
    hyper = read(out/"hyper.json")
    assert hyper["rounds"][-1]["level"] == 3 and hyper["rounds"][-1]["commits"] == 0
    assert (out/"disp.npz").is_file() and read(out/"cert.json")["status"] == "certified_feasible"
    info = eval_fields.describe_run(out)
    assert info.run_status == BUDGET_EXHAUSTED and info.field_role == eval_fields.ROLE_HANDOVER
    assert eval_fields.field_present(info) and info.field_role not in eval_fields.COMPARABLE_ROLES


def test_run_pair_budget_ending_with_a_failing_final_certificate(setup, monkeypatch):
    """A native budget ending whose fresh CPU certificate does not pass: final_certificate_rejected (exit 2), solver
    status budget_exhausted kept in status.json / hyper.json, no disp.npz, final_native.npz and cert.json (rejected)
    kept as failure evidence; the evaluator reads no native result."""
    root, args = setup
    grid, pair = write_synthetic_pair(root, DEV[0], DEV[1])
    write_tissue(root, grid, DEV[0])
    write_tissue(root, grid, DEV[1], shift=(.7, -.5, .4))
    jump_in_run_pair(monkeypatch, lambda row: row.get("kind") == "accept" and row.get("level") == 3)
    real_export = run_pair.export_native

    def failing_export(out, *, backend, **kwargs):
        wrapped = FailingCertificate()
        wrapped.cert_config = backend.cert_config
        return real_export(out, backend=wrapped, **kwargs)
    monkeypatch.setattr(run_pair, "export_native", failing_export)
    out = root/"run_budget_rejected_cert"
    assert run_pair.main(args(pair, out, "--rounds", "1,1,1,1", "--max-accepted", "2", "--wall-seconds", "2700")) == 2
    status = read(out/"status.json")
    assert status["status"] == FINAL_CERTIFICATE_REJECTED and status["exit_code"] == 2
    assert status["solver_status"] == BUDGET_EXHAUSTED and status["wall_budget"]["exhausted"]
    assert status["export"]["certificate_passed"] is False
    assert not (out/"disp.npz").exists() and read(out/"cert.json")["rejected"] is True
    with np.load(out/"final_native.npz") as z:
        assert str(z["status"]) == FINAL_CERTIFICATE_REJECTED
    hyper = read(out/"hyper.json")
    assert hyper["status"] == FINAL_CERTIFICATE_REJECTED and hyper["solver_status"] == BUDGET_EXHAUSTED
    assert hyper["rounds"][-1]["status"] == BUDGET_EXHAUSTED and hyper["rounds"][-1]["commits"] == 1
    info = eval_fields.describe_run(out)
    assert info.field_role == eval_fields.ROLE_NONE and not eval_fields.field_present(info)


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "soon"])
def test_run_pair_refuses_invalid_wall_seconds(value):
    with pytest.raises(SystemExit):
        run_pair.parse_args(["--pair", "s/p", "--channel-source", "none", "--out", "x", "--wall-seconds", value])
