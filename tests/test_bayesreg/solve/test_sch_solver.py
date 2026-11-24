"""Per-level caps and coarse-level shares in the solver (work package SCHI, 2026-10-31; contract C13-4a, C13-5a, C12-3a SCH
item, C13-7 SCH row, C16 SCH item; the C13 SCH gates 5-9 and 11, the solver halves of 3, 7 and 14).  Synthetic data
only (the 26^3 four-level synthetic case of solve/_synthetic.py and its structure variant), numpy backend, injected
clocks.

The clock of a WallBudget is moved by the solver's own log rows (`Events`), so each case triggers at a known safe point:
W = 100 s, shares (.1, .1, .1): D_0 = 10, D_1 = 20, D_2 = 30.
- gate 5: the share ends the round at the PG iteration top (option (a)), the option (b) outer iteration top and the
  option (b) inner PG iteration top; the round's state_hash is that of the last formal commit, the transfer seed is
  prolonged from it (source_state_hash), uncommitted inner progress appears in no commit; a clock that crosses D_ell in
  a line search does not stop it (run_pg: the next iteration top does; a line search that fails is line_search_failed);
- gate 6: on the native level a clock past every D_ell never gives level_budget_reached; W ends it (budget_exhausted);
- gate 7: a natural ending (cap, convergence, stagnation) at the same check wins; W and D_ell at the same check give
  budget_exhausted (stopped_before_native); with an incrementing clock every safe point reads the clock once, the share
  compares the budget's reading and level_share_hit.check is the budget's count;
- gate 8: a share in round 0 of a level with two rounds skips round 1 (rounds_skipped 1); the M step is made and marked
  not used; a FloatingPointError in that M step ends the run level_failed;
- gate 9: a deadline already passed at the level top (L0 from u = 0, L1 from the transfer seed): 0 accepted steps,
  level_budget_reached, the transfer goes on;
- gate 11: with sigma_domain support_shared the support branch's M step is marked not used, the level's support record is
  that of the untriggered run, the next level recomputes v from its seed;
- gate 3 / 14 (solver side): per-level caps act on every (level, round); accepted_committed of option (b).
"""
import numpy as np
import pytest

from jd_curl.validation import InputValidationError

from bayesreg.solve import (Caps, Eta, FixedSigma, GaussianMStep, NumpyBackend, ProximalParams, SolverConfig,
                            default_lambda, eta_temporary, run_pg, solve_multilevel)
from bayesreg.solve import jd_scaling as jds
from bayesreg.solve import stopping as st
from bayesreg.solve.stopping import (BUDGET_EXHAUSTED, CAP_REACHED, LEVEL_BUDGET_REACHED, LEVEL_FAILED,
                                     STOPPED_BEFORE_NATIVE, LevelDeadline, LevelShares, WallBudget, state_hash)

from . import _swl_synthetic as S
from . import _synthetic as syn
from .test_pg import FakeObjective, Identity, Red, passing, small_state, start

ETA = Eta(1., 1., (1., 1., 1., 1.))
W = 100.
SHARES = (.1, .1, .1)                       # D = (10, 20, 30)
SCH_ROUND_KEYS = ("cap_accepted", "accepted_committed", "level_time_share", "level_cumulative_share",
                  "level_deadline_seconds", "level_elapsed_seconds", "run_elapsed_seconds", "ended_by_level_share",
                  "level_share_hit", "rounds_skipped")


class Clock:
    """A settable clock counting its readings."""

    def __init__(self, t=0.):
        self.t, self.calls = t, 0

    def __call__(self):
        self.calls += 1
        return self.t


class Events:
    """log callback: keeps every row; the first row matching each pending (predicate, t) sets the clock to t."""

    def __init__(self, clock, *events):
        self.clock, self.pending, self.rows = clock, list(events), []

    def __call__(self, row):
        self.rows.append(row)
        for k, (when, t) in enumerate(self.pending):
            if when(row):
                self.clock.t = t
                del self.pending[k]
                break


def accept_at(level, *, outer=None, steps=1, round_=0):
    return lambda r: (r["kind"] == "accept" and r.get("level") == level and r.get("round") == round_
                      and (outer is None or r.get("outer") == outer) and r["accepted_steps"] == steps)


def solve(levels, *, config, events=(), start_t=0., sigma=None, backend=None, wall=W):
    jds.clear_cell_bounds_cache()
    clock = Clock(start_t)
    budget = WallBudget(wall, clock=clock, started=0.)
    log = Events(clock, *events)
    result = solve_multilevel(NumpyBackend() if backend is None else backend, list(levels), eta=ETA,
                              sigma=FixedSigma([1.]) if sigma is None else sigma, config=config, log=log,
                              budget=budget)
    return result, log.rows, budget, clock


def shares_config(levels=None, *, outer="a", caps=None, rounds=(1, 1, 1, 1), max_levels=4, **extra):
    shares = LevelShares(SHARES, W)
    if outer == "b":
        caps = Caps(max_accepted=40, max_outer=5) if caps is None else caps
        return SolverConfig(outer="b", rounds=rounds, caps=caps, max_levels=max_levels, level_time_shares=shares,
                            proximal=ProximalParams(lam=default_lambda(levels[-1].grid)), **extra)
    return SolverConfig(rounds=rounds, caps=Caps(max_accepted=3) if caps is None else caps, max_levels=max_levels,
                        level_time_shares=shares, **extra)


@pytest.fixture(scope="module")
def levels():
    return syn.four_levels(26)[1]


def kinds(rows, kind, **match):
    return [r for r in rows if r["kind"] == kind and all(r.get(k) == v for k, v in match.items())]


def zero_hash(grid):
    return state_hash(np.zeros(grid.shape+(3,)))


# ---- LevelShares / LevelDeadline / Caps units -------------------------------------------------------------------------

def test_level_shares_deadlines_and_validation():
    shares = LevelShares((.1, .2, .3), 1000.)
    assert shares.cumulative == (.1, .1+.2, (.1+.2)+.3)
    assert shares.deadlines == tuple(c*1000. for c in shares.cumulative)
    for bad in [(.1, .2), (.1, .2, .3, .1), (0., .1, .1), (-.1, .1, .1), (1., .1, .1), (.5, .3, .2), (.4, .4, .3),
                (float("nan"), .1, .1), (float("inf"), .1, .1)]:
        with pytest.raises(InputValidationError):
            LevelShares(bad, 1000.)
    with pytest.raises(InputValidationError):
        LevelShares((.1, .1, .1), 0.)
    with pytest.raises(InputValidationError):
        shares.deadline(None, 0)                                        # shares need the run's WallBudget
    with pytest.raises(InputValidationError):
        shares.deadline(WallBudget(999.), 0)


def test_level_deadline_reads_the_budgets_reading():
    clock = Clock()
    budget = WallBudget(100., clock=clock, started=0.)
    deadline = LevelShares(SHARES, 100.).deadline(budget, 1)
    assert isinstance(deadline, LevelDeadline) and deadline.deadline == (.1+.1)*100.
    clock.t = 19.999
    assert not budget.expired("a") and not deadline.reached({"at": "a"})
    calls = clock.calls
    clock.t = 20.
    assert not budget.expired("b") and clock.calls == calls+1
    assert deadline.reached({"at": "b"}) and clock.calls == calls+1          # no reading of its own
    assert deadline.hit == dict(where={"at": "b"}, elapsed_seconds=20., check=2) and budget.checks == 2
    assert budget.record()["checks"] == 2 and budget.hit is None              # the WallBudget record is unchanged


def test_caps_per_level_units():
    caps = Caps.per_level((1, 2, 3, 4), max_outer=7)
    assert caps.max_accepted is None and caps.max_accepted_per_level == (1, 2, 3, 4)
    assert [caps.for_level(k) for k in range(4)] == [Caps(1, 7), Caps(2, 7), Caps(3, 7), Caps(4, 7)]
    assert caps.record() == dict(max_accepted_per_round=None, max_outer_commits_per_round=7)
    uniform = Caps.per_level((5, 5, 5, 5))
    assert uniform.max_accepted == 5 and uniform.record() == Caps(5).record()
    plain = Caps(3, 2)
    assert plain.for_level(2) is plain and plain.max_accepted_per_level is None
    with pytest.raises(InputValidationError):
        Caps(max_accepted=500, max_accepted_per_level=(1, 2, 3, 4))           # inconsistent
    with pytest.raises(InputValidationError):
        Caps.per_level((1, 2, 3))
    with pytest.raises(InputValidationError):
        Caps.per_level((1, -2, 3, 4))


def test_level_budget_status_vocabulary():
    """Gate 13 / 14: level_budget_reached is in the inner and round sets only; the run sets and wording are unchanged;
    C13-7 continues on it unconditionally; final_status never maps it."""
    assert st.INNER_STATUSES == ("converged_gradient", "max_iterations", "stagnated", "line_search_failed",
                                 "nonfinite_evaluation", "non_descent_direction", "budget_exhausted",
                                 "level_budget_reached")
    assert st.ROUND_STATUSES == ("converged_gradient", "cap_reached", "stagnated", "line_search_failed",
                                 "trial_budget_exhausted", "outer_stagnated", "outer_no_progress", "outer_inner_failed",
                                 "nonfinite_evaluation", "non_descent_direction", "budget_exhausted",
                                 "level_budget_reached")
    assert st.RUN_STATUSES == ("converged_regularized_gradient_B", "cap_reached", "stagnated", "outer_stagnated",
                               "line_search_failed", "trial_budget_exhausted", "outer_no_progress",
                               "outer_inner_failed", "budget_exhausted", "level_failed", "transfer_rejected",
                               "initialization_failed", "stopped_before_native", "subject_sealed",
                               "provenance_rejected", "final_certificate_rejected")
    assert LEVEL_BUDGET_REACHED not in st.RUN_STATUSES and LEVEL_BUDGET_REACHED not in st.RUN_WORDING
    assert LEVEL_BUDGET_REACHED not in st.FINAL_STATUS and LEVEL_BUDGET_REACHED not in st.CONTINUE_ALWAYS
    assert st.LEVEL_BUDGET_WORDING == "粗层预算转层" and "收敛" not in st.LEVEL_BUDGET_WORDING
    decision = st.continuation(LEVEL_BUDGET_REACHED, last_iteration=[("NONFINITE_STATE", None)])
    assert decision.proceed                                                   # unconditional, as cap_reached
    with pytest.raises(AssertionError):
        st.final_status(LEVEL_BUDGET_REACHED, decision)


# ---- gate 5: where the share ends a round ----------------------------------------------------------------------------

def check_share_round(row, *, level, deadline):
    assert row["status"] == LEVEL_BUDGET_REACHED and row["ended_by_level_share"] is True
    assert row["level_deadline_seconds"] == deadline and row["level_share_hit"]["elapsed_seconds"] >= deadline
    assert row["continuation"]["proceed"] is True and row["level"] == level


def test_share_at_the_pg_iteration_top_option_a(levels):
    result, rows, budget, _ = solve(levels, config=shares_config(), events=[(accept_at(0), 15.)])
    level0 = [r for r in result.rounds if r["level"] == 0]
    assert len(level0) == 1
    row = level0[0]
    check_share_round(row, level=0, deadline=10.)
    assert row["accepted"] == row["commits"] == row["accepted_committed"] == 1
    hit = row["level_share_hit"]
    assert hit["where"]["at"] == "pg_iteration_top" and hit["where"]["iteration"] == 2
    accept = kinds(rows, "accept", level=0)
    assert len(accept) == 1 and row["state_hash"] == accept[0]["state_hash"]
    # the transfer seed is prolonged from that commit (before the .9^k back-off)
    assert result.transfers[0]["source_state_hash"] == row["state_hash"]
    assert result.transfers[0]["rows"][0]["scale"] == 1.
    budget_rows = kinds(rows, "level_budget")
    assert len(budget_rows) == 1 and budget_rows[0]["level"] == 0 and budget_rows[0]["safe_point"] == "pg_iteration_top"
    assert budget_rows[0]["level_deadline_seconds"] == 10. and budget_rows[0]["check"] == hit["check"]
    # the run goes on: level 1 (15 < D_1 = 20) and the others to their caps
    assert [r["status"] for r in result.rounds[1:]] == [CAP_REACHED]*3 and result.status == CAP_REACHED
    assert budget.hit is None


def test_share_at_the_option_b_outer_iteration_top(levels):
    result, rows, budget, _ = solve(levels, config=shares_config(levels, outer="b", max_levels=2),
                                    events=[(lambda r: r["kind"] == "commit" and r.get("level") == 0
                                             and r["outer"] == 1, 15.)])
    row = result.rounds[0]
    check_share_round(row, level=0, deadline=10.)
    assert row["level_share_hit"]["where"]["at"] == "outer_iteration" and row["level_share_hit"]["where"]["outer"] == 1
    commits = kinds(rows, "commit", level=0)
    assert row["commits"] == len(commits) == 1 and row["state_hash"] == commits[0]["state_hash"]
    assert row["accepted"] == row["accepted_committed"]                       # nothing dropped at an outer top
    assert result.transfers[0]["source_state_hash"] == row["state_hash"]
    assert result.status == STOPPED_BEFORE_NATIVE and result.rounds[-1]["level"] == 1


def test_share_at_the_option_b_inner_iteration_top(levels):
    """The clock moves at the first inner accepted step of outer iteration 1 (after one commit): the inner PG ends at its
    next iteration top; its accepted steps are dropped (accepted - accepted_committed = their number) and appear in no
    commit; the round keeps the outer commit."""
    result, rows, budget, _ = solve(levels, config=shares_config(levels, outer="b", max_levels=2),
                                    events=[(accept_at(0, outer=1), 15.)])
    row = result.rounds[0]
    check_share_round(row, level=0, deadline=10.)
    assert row["inner_status"] == LEVEL_BUDGET_REACHED and "uncommitted" in row["reason"]
    hit = row["level_share_hit"]["where"]
    assert hit["at"] == "pg_iteration_top" and hit["outer"] == 1 and hit["objective"] == "E_v" and hit["iteration"] == 2
    commits = kinds(rows, "commit", level=0)
    assert len(commits) == row["commits"] == 1 and row["state_hash"] == commits[0]["state_hash"]
    committed = len(kinds(rows, "accept", level=0, outer=0))
    dropped = kinds(rows, "accept", level=0, outer=1)
    assert len(dropped) == 1
    assert row["accepted_committed"] == committed and row["accepted"] - row["accepted_committed"] == len(dropped)
    assert dropped[0]["state_hash"] not in {c["state_hash"] for c in kinds(rows, "commit")}
    assert result.transfers[0]["source_state_hash"] == row["state_hash"] == result.transfers[0]["source_state_hash"]


def pg_deadline():
    clock = Clock()
    budget = WallBudget(W, clock=clock, started=0.)
    deadline = LevelShares(SHARES, W).deadline(budget, 0)
    return clock, budget, deadline


def test_clock_crossing_the_deadline_in_a_line_search():
    """The clock passes D_0 at the first (rejected) candidate: the line search goes on to the accepted second candidate;
    the share ends the solve at the next iteration top."""
    state = small_state()
    clock, budget, deadline = pg_deadline()
    rows = []

    def log(row):
        rows.append(row)
        if row.get("kind") == "candidate" and row["attempt"] == 0 and row["iteration"] == 1:
            clock.t = 15.
    objective = FakeObjective(state, [Red(-1., 1e-12), Red(1., 0.)])
    base, theta, cert = start(state)
    result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, target=0.,
                    max_steps=10, budget=budget, level_deadline=deadline, log=log)
    assert result.status == LEVEL_BUDGET_REACHED and result.accepted == 1
    assert [(r["iteration"], r["attempt"], r["reason"]) for r in rows if r["kind"] == "candidate"] == [
        (1, 0, "NO_STRICT_DECREASE"), (1, 1, None)]
    assert deadline.hit["where"]["iteration"] == 2 and deadline.hit["where"]["attempt"] == 0
    assert budget.hit is None


def test_line_search_failing_after_the_deadline_is_line_search_failed():
    state = small_state()
    clock, budget, deadline = pg_deadline()
    rows = []

    def log(row):
        rows.append(row)
        clock.t = 15.
    objective = FakeObjective(state, [Red(-1., 1e-12)]*26)
    base, theta, cert = start(state)
    result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, target=0.,
                    max_steps=10, budget=budget, level_deadline=deadline, log=log)
    assert result.status == "line_search_failed" and result.accepted == 0 and deadline.hit is None
    assert len([r for r in rows if r["kind"] == "candidate"]) == 26


# ---- gate 6: never on the native level -----------------------------------------------------------------------------

def test_native_level_has_no_share(levels):
    """The clock passes every D_ell at the transfer into the native level (50 s < W): the native round runs on to its
    cap; at its first accepted step the clock passes W: budget_exhausted ends it."""
    result, rows, budget, _ = solve(levels, config=shares_config(rounds=(1, 1, 1, 2)), events=[
        (lambda r: r["kind"] == "transfer" and r["to_level"] == 3, 50.),
        (accept_at(3, round_=1), 150.)])
    statuses = [(r["level"], r["round"], r["status"]) for r in result.rounds]
    assert statuses == [(0, 0, CAP_REACHED), (1, 0, CAP_REACHED), (2, 0, CAP_REACHED), (3, 0, CAP_REACHED),
                        (3, 1, BUDGET_EXHAUSTED)]
    assert not kinds(rows, "level_budget")
    native = [r for r in result.rounds if r["level"] == 3]
    assert all(r["level_time_share"] is None and r["level_deadline_seconds"] is None and
               r["ended_by_level_share"] is False and r["level_share_hit"] is None for r in native)
    assert result.status == BUDGET_EXHAUSTED and result.native and budget.hit["where"]["level"] == 3


# ---- gate 7: priority and one clock reading --------------------------------------------------------------------------

def test_natural_endings_win_over_the_share(levels):
    # cap: a_0 = 1, the clock passes D_0 at that accepted step; the next check finds the cap first
    config = shares_config(caps=Caps.per_level((1, 3, 3, 3)), max_levels=1)
    result, rows, budget, _ = solve(levels, config=config, events=[(accept_at(0), 15.)])
    assert result.rounds[0]["status"] == CAP_REACHED and result.rounds[0]["level_share_hit"] is None
    assert not kinds(rows, "level_budget") and result.status == STOPPED_BEFORE_NATIVE
    # convergence and stagnation in run_pg with the deadline already passed
    state = small_state()
    for kwargs, status in [(dict(target=1e9), "converged_gradient"),
                           (dict(target=0., config=st_config(stall=1)), "stagnated")]:
        clock, budget, deadline = pg_deadline()
        clock.t = 15. if status == "converged_gradient" else 0.
        objective = FakeObjective(state, [Red(1., 0.)]*3)
        base, theta, cert = start(state)
        result = run_pg(objective, base, theta=theta, cert=cert, metric=Identity(), certify=passing, max_steps=10,
                        budget=budget, level_deadline=deadline,
                        on_accept=lambda *a, c=clock: setattr(c, "t", 15.), **kwargs)
        assert result.status == status and deadline.hit is None


def st_config(stall):
    from bayesreg.solve import PGConfig
    return PGConfig(step_tol=1e300, stall_steps=stall)


def test_budget_and_share_at_the_same_check_is_budget_exhausted(levels):
    result, rows, budget, _ = solve(levels, config=shares_config(max_levels=2), events=[(accept_at(0), 150.)])
    row = result.rounds[0]
    assert row["status"] == BUDGET_EXHAUSTED and row["ended_by_level_share"] is False and row["level_share_hit"] is None
    assert result.status == STOPPED_BEFORE_NATIVE and budget.hit["where"]["iteration"] == 2
    assert not kinds(rows, "level_budget")


class IncrementingClock(Clock):
    """Every reading is one second later than the previous one."""

    def __call__(self):
        self.calls += 1
        self.t += 1.
        return self.t


def test_one_clock_reading_per_safe_point(levels, monkeypatch):
    clock = IncrementingClock()
    trace = []

    class TracingBudget(WallBudget):
        def expired(self, where=None):
            before = clock.calls
            out = super().expired(where)
            trace.append(("budget", before, clock.calls, self.checks, self.last_elapsed, out))
            return out
    real_reached = LevelDeadline.reached

    def reached(self, where):
        before = clock.calls
        out = real_reached(self, where)
        trace.append(("share", before, clock.calls, self.budget.checks, self.budget.last_elapsed, out))
        return out
    monkeypatch.setattr(LevelDeadline, "reached", reached)
    budget = TracingBudget(1000., clock=clock, started=0.)
    config = SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(50), max_levels=1,
                          level_time_shares=LevelShares((.005, .3, .3), 1000.))
    jds.clear_cell_bounds_cache()
    rows = []
    result = solve_multilevel(NumpyBackend(), list(levels), eta=ETA, sigma=FixedSigma([1.]), config=config,
                              log=rows.append, budget=budget)
    row = result.rounds[0]
    assert row["status"] == LEVEL_BUDGET_REACHED
    # the last level run (max_levels = 1) ended on its share: the run ends stopped_before_native (C13-5a)
    assert result.status == STOPPED_BEFORE_NATIVE and "level_budget_reached" in result.reason and row["final"]
    assert row["sigma2_post"] is not None and result.timing["levels"][0]["ended_by_level_share"] is True
    shares_checked = [t for t in trace if t[0] == "share"]
    assert shares_checked and shares_checked[-1][-1] is True
    for k, entry in enumerate(trace):
        kind, before, after, checks, last, out = entry
        if kind == "budget":
            assert after == before+1 and not out                    # one reading per budget check, never expired
        else:
            previous = trace[k-1]
            assert previous[0] == "budget" and after == before == previous[2]     # no reading of its own
            assert last == previous[4] and checks == previous[3]                  # the budget's reading and count
    hit = row["level_share_hit"]
    assert hit["check"] == shares_checked[-1][3] and hit["elapsed_seconds"] == shares_checked[-1][4] >= 5.
    assert row["accepted"] >= 1 and len(shares_checked) >= 2
    # readings (started given: none at construction): one per check, the level top and round end records (SCH), the
    # run's budget record
    assert clock.calls == budget.checks+1+1+1


# ---- gate 8: skipped rounds and the M step ---------------------------------------------------------------------------

class TracingSigma(FixedSigma):
    def __init__(self, sigma2, fail_at=None):
        super().__init__(sigma2)
        self.fail_at, self.m_steps = fail_at, 0

    def m_step(self, state, point):
        self.m_steps += 1
        if self.fail_at == self.m_steps:
            raise FloatingPointError("injected M step failure")
        return super().m_step(state, point)


def test_share_skips_the_remaining_rounds_and_marks_the_m_step_unused(levels):
    sigma = TracingSigma([1.])
    result, rows, budget, _ = solve(levels, config=shares_config(rounds=(2, 1, 1, 1), max_levels=2),
                                    events=[(accept_at(0), 15.)], sigma=sigma)
    level0 = [r for r in result.rounds if r["level"] == 0]
    assert [r["round"] for r in level0] == [0] and level0[0]["rounds_skipped"] == 1
    check_share_round(level0[0], level=0, deadline=10.)
    assert level0[0]["sigma2_post"] is not None and sigma.m_steps >= 1
    m_steps = [t for t in result.sigma["trace"] if t["kind"] == "m_step" and t["level"] == 0]
    assert len(m_steps) == 1 and m_steps[0]["used"] is False and m_steps[0]["collapse"] is False
    assert not kinds(rows, "accept", level=0, round=1)
    assert [r["level"] for r in result.rounds] == [0, 1] and result.status == STOPPED_BEFORE_NATIVE


def test_m_step_failure_after_a_share_is_level_failed(levels):
    result, rows, budget, _ = solve(levels, config=shares_config(rounds=(2, 1, 1, 1), max_levels=2),
                                    events=[(accept_at(0), 15.)], sigma=TracingSigma([1.], fail_at=1))
    assert result.status == LEVEL_FAILED and "M step" in result.reason and "injected" in result.reason
    assert result.rounds[-1]["status"] == LEVEL_BUDGET_REACHED and result.level == 0


# ---- gate 9: the deadline has passed before the level top -------------------------------------------------------------

def test_deadline_passed_before_level_0(levels):
    result, rows, budget, _ = solve(levels, config=shares_config(max_levels=2), start_t=15.)
    row = result.rounds[0]
    check_share_round(row, level=0, deadline=10.)
    assert row["accepted"] == 0 and row["commits"] == 0 and row["state_hash"] == zero_hash(levels[0].grid)
    assert row["G0"] > 0 and row["sigma2_used"] == [1.]                  # the level start preparation was made
    assert result.transfers[0]["source_state_hash"] == row["state_hash"]
    assert result.rounds[1]["status"] == CAP_REACHED and result.status == STOPPED_BEFORE_NATIVE


def test_deadline_passed_before_level_1(levels):
    result, rows, budget, _ = solve(levels, config=shares_config(max_levels=3),
                                    events=[(lambda r: r["kind"] == "transfer" and r["to_level"] == 1, 25.)])
    assert result.rounds[0]["status"] == CAP_REACHED
    row = result.rounds[1]
    check_share_round(row, level=1, deadline=20.)
    seed = result.transfers[0]["rows"][result.transfers[0]["accepted_k"]]["cert"]["state_hash"]
    assert row["accepted"] == 0 and row["state_hash"] == seed
    assert result.transfers[1]["source_state_hash"] == seed                  # the transfer goes on from the seed
    assert result.rounds[2]["level"] == 2 and result.rounds[2]["status"] == CAP_REACHED


# ---- gate 10 (solver side): a share that never comes is the path without shares -----------------------------------------

def strip(rows):
    out = []
    for row in rows:
        out.append({k: v for k, v in row.items() if k not in SCH_ROUND_KEYS})
    return out


def test_share_never_due_is_the_path_without_shares(levels):
    from .test_wall_budget import canonical
    with_shares, rows_shares, _, _ = solve(levels, config=shares_config())
    without, rows_plain, _, _ = solve(levels, config=SolverConfig(rounds=(1, 1, 1, 1), caps=Caps(max_accepted=3)))
    assert with_shares.status == without.status and with_shares.commit.u.tobytes() == without.commit.u.tobytes()
    assert canonical(strip(with_shares.rounds)) == canonical(without.rounds)
    assert canonical(strip(rows_shares)) == canonical(rows_plain)
    assert canonical(with_shares.transfers) == canonical(without.transfers)
    assert all(r["ended_by_level_share"] is False and r["rounds_skipped"] == 0 for r in with_shares.rounds)


# ---- gate 11: with sigma_domain support_shared ------------------------------------------------------------------------

def test_share_with_support_shared():
    _, levels, _ = S.four_levels_structure()
    eta = eta_temporary(levels[-1].tissue, levels[-1].mask, levels[-1].grid)

    def run(events):
        jds.clear_cell_bounds_cache()
        clock = Clock()
        budget = WallBudget(W, clock=clock, started=0.)
        log = Events(clock, *events)
        config = SolverConfig(rounds=(2, 1, 1, 1), caps=Caps(3, 2), sigma_domain="support_shared", max_levels=2,
                              level_time_shares=LevelShares(SHARES, W))
        return solve_multilevel(NumpyBackend(), list(levels), eta=eta, sigma=GaussianMStep(), config=config, log=log,
                                budget=budget), log.rows
    triggered, rows = run([(accept_at(0), 15.)])
    plain, _ = run([])
    row = triggered.rounds[0]
    assert row["status"] == LEVEL_BUDGET_REACHED and row["rounds_skipped"] == 1
    assert row["sigma2_post"] is not None and row["sigma2_S_post"] is not None
    m_steps = [t for t in triggered.sigma["trace"] if t["kind"] == "m_step" and t["level"] == 0]
    assert len(m_steps) == 1 and m_steps[0]["used"] is False and "sigma2_S" in m_steps[0]
    # level 0's support record is that of the run without a share (it is made at the level top)
    from .test_wall_budget import canonical
    assert canonical(triggered.support[0]) == canonical(plain.support[0])
    # the next level recomputes v from its own seed
    seed = triggered.transfers[0]["rows"][triggered.transfers[0]["accepted_k"]]["cert"]["state_hash"]
    assert len(triggered.support) == 2 and triggered.support[1]["seed_state_hash"] == seed
    assert triggered.support[1]["v_sha256"] != triggered.support[0]["v_sha256"]
    assert [r["level"] for r in plain.rounds] == [0, 0, 1]


# ---- gate 3 (solver side): per-level caps on every (level, round) ------------------------------------------------------

def test_per_level_caps_act_on_every_level_and_round(levels):
    # (1, 2, 3, 4) lets the second native round converge on this case (FixedSigma, eta 1); (4, 3, 2, 1) keeps the
    # premise of gate 3 (every (level, round) ends cap_reached); run_pair's synthetic pair uses 1,2,3,4
    config = SolverConfig(rounds=(1, 1, 1, 2), caps=Caps.per_level((4, 3, 2, 1)))
    result, rows, _, _ = solve(levels, config=config)
    assert all(r["status"] == CAP_REACHED for r in result.rounds)              # the premise
    assert [(r["level"], r["round"], r["accepted"], r["cap_accepted"]) for r in result.rounds] == [
        (0, 0, 4, 4), (1, 0, 3, 3), (2, 0, 2, 2), (3, 0, 1, 1), (3, 1, 1, 1)]
    assert all(r["cap"] == "max_accepted" and r["level_time_share"] is None and r["level_elapsed_seconds"] is not None
               for r in result.rounds)
    record = config.record()
    assert record["max_accepted_per_level"] == [4, 3, 2, 1] and record["max_accepted_source"] == "max_accepted_per_level"
    assert record["level_time_shares"] is None and record["level_deadlines_seconds"] is None
    assert record["caps"]["max_accepted_per_round"] is None


def test_solver_refusals():
    _, levels, _ = syn.four_levels(26)
    with pytest.raises(InputValidationError):                                  # shares need the run's WallBudget
        solve_multilevel(NumpyBackend(), list(levels), eta=ETA, sigma=FixedSigma([1.]),
                         config=shares_config(max_levels=1))
    with pytest.raises(InputValidationError):
        solve_multilevel(NumpyBackend(), list(levels), eta=ETA, sigma=FixedSigma([1.]),
                         config=shares_config(max_levels=1), budget=WallBudget(99.))
    with pytest.raises(InputValidationError):
        SolverConfig(caps=Caps.per_level((1, 2, 3, 4)), max_accepted_source="max_accepted")
    with pytest.raises(InputValidationError):
        SolverConfig(max_accepted_source="max_accepted_per_level")
    from bayesreg.solve.outer import solve_round
    with pytest.raises(InputValidationError):
        solve_round(None, None, None, option="a", target=1., caps=Caps.per_level((1, 2, 3, 4)))


def test_exception_row_carries_the_sch_keys(levels, monkeypatch):
    """C16 SCH: every round row, also the one of a round ended by an exception (level_failed, hard constraint 4)."""
    from bayesreg.solve import multilevel as ML

    def failing(*args, **kwargs):
        raise RuntimeError("injected round failure")
    monkeypatch.setattr(ML, "solve_round", failing)
    result, rows, budget, _ = solve(levels, config=shares_config(max_levels=1))
    row = result.rounds[0]
    assert result.status == LEVEL_FAILED and row["status"] == "exception"
    assert all(key in row for key in SCH_ROUND_KEYS) and row["accepted_committed"] is None
    assert row["ended_by_level_share"] is False and row["run_elapsed_seconds"] == 0.
    assert [r for r in rows if r["kind"] == "round"][0]["cap_accepted"] == 3


def test_default_config_record_has_no_sch_keys():
    record = SolverConfig().record()
    assert not {"max_accepted_per_level", "max_accepted_source", "level_time_shares",
                "level_deadlines_seconds"} & set(record)
    assert SolverConfig().sch_records is False and SolverConfig(caps=Caps(7)).sch_records is False
