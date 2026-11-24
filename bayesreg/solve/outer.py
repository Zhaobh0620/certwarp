"""C10 outer options for one (level, EM round), sigma / w / eta frozen.

(a) `solve_direct`: the PG inner solve on E = F_B runs to the C13 stop; every accepted step is a formal
    commit (certified, F_B decreased by the certified stable comparison).  Inner status -> round status:
    converged_gradient -> converged_gradient, max_iterations (the C13-4 cap of 500 accepted steps) ->
    cap_reached, stagnated / line_search_failed / numerical failures unchanged.  The metric is frozen at
    the round's start state (C09.1-3, D11).
(b) `solve_proximal`: self-referenced proximal outer loop (C10-2..5, B9).  With v = u_i: stop if
    GF_B(u_i) <= T_B (converged_gradient: "outer reached the threshold") or the caps are reached
    (cap_reached: 60 commits or 500 accepted inner steps per (level, round)); else an inner PG solve on
    E_v with at most n_in accepted steps, metric rebuilt at v (lam_b = lam, b_P with 1/2 (mu + nu2)) and
    BB history cleared.  The inner may be committed only if it ended converged_gradient, max_iterations
    or stagnated (forcing is off, C10-6); the commit further needs (ii) the stable comparison of
    E_{u_i}(u+) against E_{u_i}(u_i) = F_B(u_i) certified > 0, (iii) the independent comparison of
    F_B(u+) against F_B(u_i) certified > 0 (each with its own s), (iv) u+ != u_i.  Failure of (ii)-(iv)
    -> outer_no_progress; an inner status outside that set -> outer_inner_failed; neither recovers the
    uncommitted inner progress.  A committed inner that ended stagnated ends the round outer_stagnated.
    Both differences of every commit attempt are logged (C10 gate).
Wall-clock budget (development switch --wall-seconds, D44; `budget`, a stopping.WallBudget or None): (a) the PG solve
ends budget_exhausted at its last accepted step (every accepted step is a formal commit), and so does the round; (b) the
outer loop checks the budget at the top of every outer iteration (after the threshold and the caps) and passes it to the
inner solve; an inner that ended budget_exhausted ends the round budget_exhausted at the last OUTER commit (the
uncommitted inner progress is dropped, never exported).
Coarse-level shares (C13-5a, SCH; `level_deadline`, a stopping.LevelDeadline or None): checked at the same two kinds of
safe point right after the budget, with its clock reading -- (a) the top of every PG iteration: the round ends
level_budget_reached at its last accepted step (a formal commit); (b) the top of every outer iteration (after the
threshold, the caps and the budget) and the top of every inner PG iteration: the round ends level_budget_reached at the
last OUTER commit, uncommitted inner progress dropped (C10-4).  Per-level caps (C13-4a): the level loop passes
`Caps.for_level(ell)`; a Caps with per-level values is refused here.  RoundResult.accepted_committed: the accepted steps
contained in the formal commits ((a): accepted; (b): the inner steps of the committed inner solves) -- recorded only with
the SCH switches (C13-4a "选项 (b) 的 accepted 口径").
JD affine scaling (C09.1-3a, `PGConfig.metric_jd_scaling`): with "affine_v1" S~ is computed with M at the same
freeze point (option (a): the round's start state; option (b): every reference v = u_i) and the PG metric is
S~ M^-1 S~ (`jd_scaling.JDScaledMetric`; refreshes inside the inner solve by its rule).  Every computation of S~
is logged (kind "jd_scaling") and the round's metric record carries `jd_scaling` (configuration, every
computation with min J_a and the fraction r_a < 1, and `jd_scaling_refresh`, the number of refreshes).  With
"off" the metric is M itself (bitwise the C09.1 path) and the record says only {"mode": "off"}.  The freeze point's
CertOutcome (start.cert / commit.cert) and backend.cert_config go to the metric for jd_cell_estimate "certificate"
(C09.1-3a last item, D47) and "certificate_bernstein" (D53); the other cell estimates ignore them.  A
certificate_bernstein S~ computation with check-(b) violations is followed by a log row of kind "warning"
(jd_scaling.j53_warning_row); under certificate_bernstein the round's ledger is also kept on the CommitTracker while
the round runs, so that a round ended by an exception still reports its S~ computations (multilevel's exception row,
JDTI review minor 1); solve_round drops it at a normal return (no JDScaledMetric outlives its round).
LM inner (work package LMI, 2026-11-01; contract C09.3; `lm`, an LMConfig, or None = PG, the code before LMI): (a) one
run_lm per (level, round) on F_B from the round start (gamma0 there), the metric record that of the gamma-0 C09.1 metric
with gamma null, jd_scaling {"mode": "none"} (S~ is never computed) and `lm` (gamma0, damping and PCG summary); (b) one
run_lm per outer iteration on E_v (H_E and pred with the proximal blocks, gamma0 = gamma0(lam) at every reference
refresh), the commit rule unchanged; trial_budget_exhausted is an inner status outside COMMITTABLE_INNER
(outer_inner_failed, C13-7 safe-set rule).  The LMMetric rebuilds the C09.1 metric at the current gamma (c_dat, b_P,
lam_b of the freeze point; its record() gives the round's a and shift without gamma from the frozen values, so no
gamma-0 metric is built for the record, and option (b) keeps no S~ ledger under LM).  The budget and the shares are
checked by run_lm at its iteration tops only (C13-5a LM item) and by the outer loop of (b) as for PG.  solve_round
refuses the LM inner with a backend that has the GPU-path comparison (stable_compare: the LM class has not passed the
C08-9 L1 gate).
"""
from dataclasses import dataclass

import numpy as np

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError

from .. import memtrace
from .jd_scaling import AFFINE_V1, CERTIFICATE_BERNSTEIN, OFF, JDScaledMetric, j53_warning_row, scaled_metric
from .pg import DirectObjective, Handoff, PGConfig, ProximalObjective, run_pg
from .stopping import (BUDGET_EXHAUSTED, CAP_REACHED, CONVERGED_GRADIENT, LEVEL_BUDGET_REACHED, MAX_ITERATIONS,
                       OUTER_INNER_FAILED, OUTER_NO_PROGRESS, OUTER_STAGNATED, STAGNATED, budget_expired, gf_b)

OUTER_OPTIONS = ("a", "b")
COMMITTABLE_INNER = (CONVERGED_GRADIENT, MAX_ITERATIONS, STAGNATED)


@dataclass(frozen=True)
class Commit:
    """A formal state: certified (`cert`), u the read-only (nz, ny, nx, 3) field, theta its free vector."""
    u: np.ndarray
    theta: np.ndarray
    cert: object


@dataclass
class CommitTracker:
    """The last formal commit of the running (level, round), updated at every formal commit (option (a): every
    accepted step; option (b): every outer commit), so that the level loop keeps it when an exception ends the
    round (hard constraint 4: formal commits are never dropped).  `ledger`: the round's C09.1-3a S~ ledger, set by
    solve_direct / solve_proximal under certificate_bernstein only and dropped by solve_round at a normal return, so
    that the level loop can still record the S~ computations of a round that an exception ended (JDTI review minor 1:
    their check-(b) violations count in the run total) and no JDScaledMetric outlives a finished round."""
    commit: Commit
    commits: int = 0
    ledger: object = None

    def update(self, commit):
        self.commit = commit
        self.commits += 1

    def observe(self, point):
        """An accepted step's evaluation (option (a), PG): nothing here; bayesreg.solve.checkpoints.ProfileTracker records it."""


@dataclass
class RoundResult:
    status: str
    reason: str
    commit: Commit                # the round's last formal commit (its start state if none)
    fb_point: object              # F_B evaluation at commit.u with this round's state
    gradient_norm: float          # GF_B at the commit
    target: float
    commits: int
    accepted: int
    inner_status: str | None
    last_iteration: list
    energies: list                # F_B at the start and after every formal commit
    gradient_norms: list          # GF_B at the start and after every formal commit
    counters: object
    metric: dict
    cap: str | None = None
    commit_refusal: dict | None = None    # option (b), outer_no_progress: the two commit-check differences
    accepted_committed: int | None = None  # SCH (C13-4a): accepted steps contained in the formal commits


def _metric_record(metric, c_dat, b_P, lam_b):
    return dict(c_dat=c_dat, b_P=b_P, lam_b=lam_b, gamma=0., a=metric.a, shift=metric.shift)


def _keep_ledger(tracker, ledger, pg):
    """Keep the round's S~ ledger on the CommitTracker for the exception row of multilevel (JDTI review minor 1) --
    certificate_bernstein only, the only estimate whose exception row reads it; solve_round drops it again at a normal
    return (JDTI minors review, important 1: no JDScaledMetric outlives its round)."""
    if tracker is not None and pg.jd_cell_estimate == CERTIFICATE_BERNSTEIN:
        tracker.ledger = ledger


def _pg_metric(metric, state, u, pg, log, context, *, cert=None, cert_config=None, **where):
    """C09.1-3a: the PG metric frozen with M at u (M itself when metric_jd_scaling is "off"); `cert` (the CertOutcome
    of u) and `cert_config` (the backend's C07-2 configuration) serve jd_cell_estimate "certificate" and
    "certificate_bernstein"; a check-(b) violation of the latter adds a "warning" row after the jd_scaling row."""
    out = scaled_metric(metric, state.grid, u, pg.metric_jd_scaling, pg.jd, cert=cert, cert_config=cert_config,
                        **where)
    if out is not metric and log is not None:
        log(dict(context, kind="jd_scaling", **out.computations[-1]))
        warning = j53_warning_row(out.computations[-1])      # C09.1-3a certificate_bernstein check (b) (JDTI)
        if warning is not None:
            log(dict(context, **warning))
    return out


class _ScalingLedger:
    """The round's C09.1-3a record: every computation of S~ (freeze points and refreshes) and the refresh count."""

    def __init__(self, pg):
        self.pg, self.metrics = pg, []

    def add(self, metric):
        if isinstance(metric, JDScaledMetric):
            self.metrics.append(metric)

    def record(self):
        if self.pg.metric_jd_scaling == OFF:
            return dict(mode=OFF)
        computations = [c for m in self.metrics for c in m.computations]
        out = dict(self.pg.jd.record(), mode=AFFINE_V1, jd_scaling_refresh=sum(m.refreshes for m in self.metrics),
                   computations=computations)
        if computations:
            out.update(min_J_a=min(c["min_J_a"] for c in computations),
                       max_fraction_r_lt_1=max(c["fraction_r_lt_1"] for c in computations),
                       seconds=sum(c["seconds"] for c in computations))
        return out


def _comparison_row(reduction):
    row = dict(ared=reduction.ared, guard=reduction.guard, certified=reduction.certified, reason=reduction.reason)
    if hasattr(reduction, "record"):
        row["comparison"] = reduction.record()
    return row


def _lm_metric_record(lm_metric, summary):
    """The round metric record under the LM inner (C09.3): LMMetric.record() (c_dat, b_P, lam_b, gamma null -- it varies
    per direction attempt, log.jsonl -- and a, shift without gamma, computed from the frozen values: no extra metric is
    built for the record), no JD scaling, and the LM summary."""
    return dict(lm_metric.record(), jd_scaling=dict(mode="none"), lm=summary)


def _lm_round_summary(records):
    """Option (b): the LM summaries of the round's inner solves, aggregated (one gamma0 per reference refresh)."""
    if not records:
        return dict(inner_solves=0)
    used_min = [r["gamma_min_used"] for r in records if r["gamma_min_used"] is not None]
    used_max = [r["gamma_max_used"] for r in records if r["gamma_max_used"] is not None]
    its = [r["pcg_iterations_max"] for r in records if r["pcg_iterations_max"] is not None]
    res = [r["true_residual_max"] for r in records if r["true_residual_max"] is not None]
    return dict(inner_solves=len(records), gamma0=sorted({r["gamma0"] for r in records}),
                lam_b=records[-1]["lam_b"], gamma_end_last=records[-1]["gamma_end"],
                directions=sum(r["directions"] for r in records),
                gamma_min_used=min(used_min) if used_min else None, gamma_max_used=max(used_max) if used_max else None,
                pcg_iterations_total=sum(r["pcg_iterations_total"] for r in records),
                pcg_iterations_max=max(its) if its else None, true_residual_max=max(res) if res else None,
                true_residual_failed=sum(r["true_residual_failed"] for r in records),
                metric_builds=sum(r["metric_builds"] for r in records))


def _solve_direct_lm(backend, state, start, objective, start_point, c_dat, b_P, *, target, caps, lm, log, context,
                     tracker, budget, level_deadline):
    """C10 option (a) with the LM inner (C09.3): every accepted step a formal commit; gamma0 at the round start.
    start_point: a Handoff of the round start's evaluation (MEM: run_lm holds it alone)."""
    from .lm import LMMetric, run_lm

    def on_accept(theta, accepted_point, cert):
        if tracker is not None:
            tracker.update(Commit(accepted_point.u, theta, cert))
    lm_metric = LMMetric(backend, state, c_dat=c_dat, b_P=b_P, lam_b=0.)
    inner = run_lm(objective, start_point, theta=start.theta, cert=start.cert, metric=lm_metric,
                   certify=backend.certify, target=target, max_steps=caps.max_accepted, config=lm, log=log,
                   context=context, on_accept=on_accept, budget=budget, level_deadline=level_deadline)
    status = {CONVERGED_GRADIENT: CONVERGED_GRADIENT, MAX_ITERATIONS: CAP_REACHED}.get(inner.status, inner.status)
    commit = Commit(inner.point.u, inner.theta, inner.cert)
    return RoundResult(status=status, reason=inner.reason, commit=commit, fb_point=inner.point,
                       gradient_norm=inner.gradient_norm, target=target, commits=inner.accepted,
                       accepted=inner.accepted, inner_status=inner.status, last_iteration=inner.last_iteration,
                       energies=list(inner.energies), gradient_norms=list(inner.gradient_norms),
                       counters=inner.counters, metric=_lm_metric_record(lm_metric, inner.lm),
                       cap="max_accepted" if status == CAP_REACHED else None, accepted_committed=inner.accepted)


def solve_direct(backend, state, start, *, target, caps, pg=PGConfig(), metric_mode="principal", log=None,
                 context=None, tracker=None, budget=None, level_deadline=None, lm=None):
    """C10 option (a) for one (level, round) from the certified start Commit; `lm` (an LMConfig) selects the LM inner
    (C09.3, work package LMI), None the PG inner (the code before LMI)."""
    objective = DirectObjective(backend, state)
    point = objective.evaluate(start.u)
    if isinstance(point, InvalidTrial):
        raise FloatingPointError(f"F_B at the certified round start is not defined ({point.reason})")
    memtrace.mark("direct.start_evaluated", context=context)
    c_dat = backend.c_dat(state, point)
    b_P = backend.metric_bP(state, mode=metric_mode)
    if lm is not None:
        start_point = Handoff(point)      # MEM: as below, run_lm holds the round start point alone
        del point
        return _solve_direct_lm(backend, state, start, objective, start_point, c_dat, b_P, target=target, caps=caps,
                                lm=lm, log=log, context=context, tracker=tracker, budget=budget,
                                level_deadline=level_deadline)
    metric = backend.build_metric(state, c_dat=c_dat, b_P=b_P)
    memtrace.mark("direct.metric", context=context)
    ledger = _ScalingLedger(pg)
    _keep_ledger(tracker, ledger, pg)
    pg_metric = _pg_metric(metric, state, start.u, pg, log, dict(context or {}), cert=start.cert,
                           cert_config=getattr(backend, "cert_config", None), at="round_start")
    ledger.add(pg_metric)
    memtrace.mark("direct.jd_freeze", context=context)

    def on_accept(theta, accepted_point, cert):
        if tracker is not None:
            tracker.update(Commit(accepted_point.u, theta, cert))
            tracker.observe(accepted_point)
    start_point = Handoff(point)          # MEM: run_pg holds the round start point alone (freed at the first step)
    del point
    inner = run_pg(objective, start_point, theta=start.theta, cert=start.cert, metric=pg_metric,
                   certify=backend.certify, target=target, max_steps=caps.max_accepted, config=pg, log=log,
                   context=context, on_accept=on_accept, budget=budget, level_deadline=level_deadline)
    status = {CONVERGED_GRADIENT: CONVERGED_GRADIENT, MAX_ITERATIONS: CAP_REACHED}.get(inner.status, inner.status)
    commit = Commit(inner.point.u, inner.theta, inner.cert)
    return RoundResult(status=status, reason=inner.reason, commit=commit, fb_point=inner.point,
                       gradient_norm=inner.gradient_norm, target=target, commits=inner.accepted,
                       accepted=inner.accepted, inner_status=inner.status, last_iteration=inner.last_iteration,
                       energies=list(inner.energies), gradient_norms=list(inner.gradient_norms),
                       counters=inner.counters,
                       metric=dict(_metric_record(metric, c_dat, b_P, 0.), jd_scaling=ledger.record()),
                       cap="max_accepted" if status == CAP_REACHED else None, accepted_committed=inner.accepted)


def solve_proximal(backend, state, start, *, target, caps, params, pg=PGConfig(), metric_mode="principal", log=None,
                   context=None, tracker=None, budget=None, level_deadline=None, lm=None):
    """C10 option (b) for one (level, round) from the certified start Commit; `lm` (an LMConfig) selects the LM inner
    on E_v (C09.3-10/11: H_E and pred with the proximal blocks; gamma0 at every reference refresh), None the PG inner
    (the code before LMI)."""
    if params is None:
        raise InputValidationError("option (b) needs ProximalParams (lam > 0)")
    if metric_mode != "principal":
        raise InputValidationError("option (b): the metric mode 'methodology' has no proximal term (C09.1)")
    log = (lambda row: None) if log is None else log
    context = dict(context or {})
    commit = start
    fb_point = backend.evaluate(state, commit.u)
    if isinstance(fb_point, InvalidTrial):
        raise FloatingPointError(f"F_B at the certified round start is not defined ({fb_point.reason})")
    from .pg import Counters
    if lm is None:
        counters = Counters()
    else:
        from .lm import LMCounters, LMMetric, run_lm
        counters = LMCounters()
        lm_records = []
    commits = accepted = committed_steps = 0
    energies, norms = [fb_point.energy], []
    inner_status, last_iteration, metric_info = None, [], None
    status = reason = cap = refusal = None
    direct = DirectObjective(backend, state)
    ledger = None
    if lm is None:                        # the C09.1-3a S~ ledger: PG only (LMI review minor 2)
        ledger = _ScalingLedger(pg)
        _keep_ledger(tracker, ledger, pg)
    while True:
        g_fb = backend.gradient(state, fb_point)
        counters.gradients += 1
        G = gf_b(g_fb, state.grid.omega, state.omega0)
        G, check = direct.confirm_gradient_norm(fb_point, G, target, context="outer")
        if check is not None:
            log(dict(context, kind="gf_recheck", outer=commits, **check))
        if len(norms) < len(energies):
            norms.append(G)
        if G <= target:
            status, reason = CONVERGED_GRADIENT, f"GF_B {G:.6e} <= T_B {target:.6e} at commit {commits}"
            break
        if commits >= caps.max_outer or accepted >= caps.max_accepted:
            cap = "max_outer" if commits >= caps.max_outer else "max_accepted"
            status, reason = CAP_REACHED, f"{cap}: {commits} commits, {accepted} accepted inner steps"
            break
        where = dict(context, outer=commits, at="outer_iteration")
        if budget_expired(budget, where):
            status, reason = BUDGET_EXHAUSTED, (f"wall-clock budget {budget.seconds:g} s used up before outer iteration "
                                                f"{commits+1} ({commits} commits; the last commit is kept)")
            break
        if level_deadline is not None and level_deadline.reached(dict(where)):
            # C13-5a: the same safe point and clock reading as the budget check above
            status, reason = LEVEL_BUDGET_REACHED, (f"level {level_deadline.level} deadline D = "
                                                    f"{level_deadline.deadline:g} s passed before outer iteration "
                                                    f"{commits+1} ({commits} commits; the last commit is kept; C13-5a)")
            break
        objective = ProximalObjective(backend, state, commit.u, params)
        base = objective.evaluate(commit.u)
        c_dat = backend.c_dat(state, fb_point)
        b_P = backend.metric_bP(state, mode=metric_mode, mu=params.mu, nu2=params.nu2)
        if lm is None:
            metric = backend.build_metric(state, c_dat=c_dat, b_P=b_P, lam_b=params.lam)
            metric_info = _metric_record(metric, c_dat, b_P, params.lam)
            pg_metric = _pg_metric(metric, state, commit.u, pg, log, dict(context, outer=commits), cert=commit.cert,
                                   cert_config=getattr(backend, "cert_config", None), at="reference_refresh",
                                   outer=commits)
            ledger.add(pg_metric)
            steps = min(params.n_in, caps.max_accepted-accepted)
            inner = run_pg(objective, base, theta=commit.theta, cert=commit.cert, metric=pg_metric,
                           certify=backend.certify, target=target, max_steps=steps, config=pg, log=log,
                           context=dict(context, outer=commits), budget=budget, level_deadline=level_deadline)
        else:
            # C09.3-12: a new run_lm per reference refresh, so gamma starts at gamma0 (lam_b = lam) every time
            # (no gamma-0 metric is built for the record: LMMetric.record(), LMI review minor 3)
            lm_metric = LMMetric(backend, state, c_dat=c_dat, b_P=b_P, lam_b=params.lam)
            metric_info = lm_metric.record()
            steps = min(params.n_in, caps.max_accepted-accepted)
            inner = run_lm(objective, base, theta=commit.theta, cert=commit.cert, metric=lm_metric,
                           certify=backend.certify, target=target, max_steps=steps, config=lm, log=log,
                           context=dict(context, outer=commits), budget=budget, level_deadline=level_deadline)
            lm_records.append(inner.lm)
        counters.add(inner.counters)
        accepted += inner.accepted
        inner_status, last_iteration = inner.status, inner.last_iteration
        if inner.status == BUDGET_EXHAUSTED:
            status, reason = BUDGET_EXHAUSTED, (f"inner {inner.reason}; its {inner.accepted} uncommitted accepted steps "
                                                f"are dropped, the last outer commit ({commits}) is kept")
            break
        if inner.status == LEVEL_BUDGET_REACHED:
            status, reason = LEVEL_BUDGET_REACHED, (f"inner {inner.reason}; its {inner.accepted} uncommitted accepted "
                                                    f"steps are dropped, the last outer commit ({commits}) is kept")
            break
        if inner.status not in COMMITTABLE_INNER:
            status, reason = OUTER_INNER_FAILED, f"inner {inner.status}: {inner.reason}"
            break
        candidate = inner.point
        same = np.array_equal(inner.theta, commit.theta)
        ratio = G/target if target > 0 else None
        e_red = None if same else objective.compare(base, candidate, required=0., context="outer_E", gf_ratio=ratio)
        f_red = None if same else direct.compare(fb_point, candidate.fb, required=0., context="outer_F",
                                                 gf_ratio=ratio)
        ok = (not same and e_red.reason is None and e_red.certified > 0 and f_red.reason is None
              and f_red.certified > 0)
        check_row = dict(same_state=bool(same), E_v=None if same else _comparison_row(e_red),
                         F_B=None if same else _comparison_row(f_red))
        log(dict(context, kind="commit_check", outer=commits, inner_status=inner.status, inner_accepted=inner.accepted,
                 committed=bool(ok), **check_row))
        if not ok:
            why = "u+ == u_i" if same else f"E_v: {e_red.reason or 'ok'}, F_B: {f_red.reason or 'ok'}"
            status, reason = OUTER_NO_PROGRESS, f"commit refused ({why})"
            refusal = dict(check_row, inner_last_iteration=[list(t) for t in inner.last_iteration])
            break
        commit = Commit(candidate.u, inner.theta, inner.cert)
        if tracker is not None:
            tracker.update(commit)
        fb_point = candidate.fb
        commits += 1
        committed_steps += inner.accepted
        energies.append(fb_point.energy)
        log(dict(context, kind="commit", outer=commits, energy=fb_point.energy, state_hash=inner.cert.state_hash,
                 min_jd_lower_bound=inner.cert.min_jd_lower_bound))
        if inner.status == STAGNATED:
            g_fb = backend.gradient(state, fb_point)
            G = gf_b(g_fb, state.grid.omega, state.omega0)
            G, check = direct.confirm_gradient_norm(fb_point, G, target, context="outer")
            if check is not None:
                log(dict(context, kind="gf_recheck", outer=commits, **check))
            norms.append(G)
            status, reason = OUTER_STAGNATED, "the committed inner solve stagnated"
            break
    if lm is None:
        metric_out = dict(metric_info or {}, jd_scaling=ledger.record())
    else:
        metric_out = dict(metric_info or {}, jd_scaling=dict(mode="none"), lm=_lm_round_summary(lm_records))
    return RoundResult(status=status, reason=reason, commit=commit, fb_point=fb_point, gradient_norm=G, target=target,
                       commits=commits, accepted=accepted, inner_status=inner_status, last_iteration=last_iteration,
                       energies=energies, gradient_norms=norms, counters=counters, metric=metric_out, cap=cap,
                       commit_refusal=refusal, accepted_committed=committed_steps)


def solve_round(backend, state, start, *, option, target, caps, params=None, pg=PGConfig(), metric_mode="principal",
                log=None, context=None, tracker=None, budget=None, level_deadline=None, lm=None):
    """One (level, round); `tracker` (CommitTracker) follows the formal commits as they are made; `budget` the
    development wall-clock budget (stopping.WallBudget) or None; `level_deadline` the coarse level's C13-5a deadline
    (stopping.LevelDeadline on `budget`) or None; `caps` the level's caps (Caps.for_level); `lm` an LMConfig for the
    LM inner (C09.3, LMI; numpy backend only: the GPU-path comparison of C08-8 has not passed the C08-9 L1 gate for the
    LM class), None for PG."""
    if lm is not None and callable(getattr(backend, "stable_compare", None)):
        raise InputValidationError("solve_round: the LM inner runs on the numpy backend only (C08-9: the LM class of the "
                                   "GPU-path comparison has not passed the L1 gate; work package LMI)")
    if caps.max_accepted_per_level is not None:
        raise InputValidationError("solve_round: pass the level's caps, Caps.for_level(ell) (C13-4a)")
    if level_deadline is not None and level_deadline.budget is not budget:
        raise InputValidationError("solve_round: the level deadline must read the run's WallBudget (C13-5a)")
    extra = {} if lm is None else dict(lm=lm)
    if option == "a":
        result = solve_direct(backend, state, start, target=target, caps=caps, pg=pg, metric_mode=metric_mode,
                              log=log, context=context, tracker=tracker, budget=budget, level_deadline=level_deadline,
                              **extra)
    elif option == "b":
        result = solve_proximal(backend, state, start, target=target, caps=caps, params=params, pg=pg,
                                metric_mode=metric_mode, log=log, context=context, tracker=tracker, budget=budget,
                                level_deadline=level_deadline, **extra)
    else:
        raise InputValidationError(f"outer option: one of {OUTER_OPTIONS}")
    if tracker is not None:
        tracker.ledger = None          # normal return: the round's record holds the ledger's record; the ledger and its
    return result                      # JDScaledMetric go (the ledger is kept only for an exception, JDTI review)
