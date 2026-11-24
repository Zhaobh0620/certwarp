"""C09.3 LM-PCG inner solver (work package LMI, 2026-11-01): GN Hessian, damping, PCG, model ratio.  A path option, off
by default (`SolverConfig.inner = "pg"`); `inner = "lm"` replaces run_pg by run_lm inside the same outer loops (C10
options (a) and (b)); F_B, the certificate, the C07-5 acceptance order, the C08 comparison and the C13 stopping rules
are those of PG.

One inner solve works on theta in R^{3 N_int} with a frozen objective E (F_B, option (a); E_v = F_B + P(., v), option
(b)), frozen sigma, w, eta, reference v and, at one base point, a frozen H_E (C09.3-10, mainline T12):
  H_E: the GN Hessian of E at the base -- objective.hvp, i.e. backend.hvp (C06-5) [+ the three proximal blocks
    K_lam^T K_lam + K_mu^T K_mu + K_nu2^T K_nu2 at the base u, bayesreg.solve.proximal.hvp, option (b)];
  direction: (H_E + gamma (omega_ell/omega0) I) p = -g by PCG from x0 = 0, atol 0, rtol 1e-3, maxiter 200, the
    preconditioner the C09.1 metric M^-1 with the current gamma (LMMetric: backend.build_metric(..., gamma=gamma));
    the PCG stops on its recursive residual ||r|| <= rtol ||g|| or at maxiter; then ONE true HVP gives H_E p and the
    true residual ||H_E p + gamma (omega_ell/omega0) p + g|| (Euclidean, both sides): > rtol ||g|| ->
    TRUE_RESIDUAL_FAILED, the direction fails (`lm_pcg`; the recursion of bayesreg.infer.laplace.pcg for one right-hand
    side, which however keeps iterating after a failed true check -- C09.3-10 asks for one check after the solve);
  pred = -g.p - 1/2 p.H_E p (undamped; option (b) with the proximal blocks); pred <= 0 -> NONPOSITIVE_PREDICTION;
  candidate theta + p (the full step), in the fixed order of C07-5 / C09.4-14: (i) finite -> (ii) certificate (CPU
    FP64; no energy without a pass) -> (iii) E (InvalidTrial reasons) -> (iv) the C08 comparison with the required
    decrease r = 1e-4 pred (C08-8(d); ENERGY_RESOLUTION, NO_STRICT_DECREASE) -> model ratio = certified/pred >= 1e-4,
    else MODEL_RATIO_REJECTED;
  damping (C09.3-12): accepted and ratio > .75 -> gamma <- max(gamma_min, gamma/2); accepted and .25 <= ratio <= .75
    -> unchanged; otherwise (accepted with ratio < .25, or any rejection / failed direction) -> gamma_max if gamma >=
    gamma_max/4, else 4 gamma.  gamma0 = min(gamma_max, max(gamma_min, 1e-3 max(1, lam_b))), gamma_min 1e-12,
    gamma_max 1e12; gamma is reset to gamma0 at the start of every run_lm call, i.e. at every (level, EM round) start
    (option (a)) and every reference refresh (option (b): one inner solve per outer iteration); never carried over;
  at most 12 direction attempts per iteration; all rejected -> `trial_budget_exhausted` (the solve ends at the last
    accepted point; a rejection never moves the base, the gradient or H_E).  C09.3-13a (e): an attempt made with gamma
    = gamma_max that is rejected ends the iteration at once with `trial_budget_exhausted` (every later attempt would
    repeat it bit for bit).  C09.3-13a (f): the direction is judged by the true residual only (a PCG stopped at maxiter
    whose true residual is <= rtol is used; its row says pcg_converged false).
End of the solve (C09.3-13 = C09.2-8/9, in this priority at the top of every iteration): converged_gradient
  (||g|| / sqrt(omega_ell/omega0) <= target), max_iterations (accepted == max_steps), stagnated (step_ratio <= 1e-12 on 5
  consecutive accepted steps); then the development wall-clock budget (budget_exhausted) and the C13-5a coarse-level
  deadline (level_budget_reached, at "lm_iteration_top", with the budget check's clock reading).  C13-5a "LM-PCG 的安全
  点": the budget and the shares are checked ONLY there -- never between direction attempts.  trial_budget_exhausted ends
  the iteration's attempts.  A FloatingPointError in the gradient, an HVP, the PCG, the true residual, the prediction
  or the comparison ends the solve with nonfinite_evaluation at the last accepted point (C13-7: terminate).  C09.3-13a
  (d): a breakdown of the linear solve -- p.(H_E + gamma I)p <= 0 or r.M^-1 r <= 0 with finite values (impossible in
  exact arithmetic: the system and M are SPD) or a nonfinite value in the PCG, the true residual or pred -- is logged as
  one attempt with reason NONFINITE_LINEAR_SOLVE and the subfield `detail` "indefinite" or "nonfinite"; it is never
  retried with a larger gamma.  C09.3-13a (h), narrowed: a TRUE_RESIDUAL_FAILED (always with a finite true residual
  here) carries in the second slot of its last_iteration entry (and the row field true_residual_case) "pcg_maxiter" when
  the PCG stopped at maxiter -- in the C13-7 safe set under the LM inner (stopping.continuation(..., inner="lm")) -- or
  "pcg_rtol_met" when the recursive residual had met rtol -- not safe, terminates; NONFINITE_LINEAR_SOLVE terminates.
JD affine scaling (C09.1-3a) changes only the PG direction; it is not used here (S~ is never computed) and the record
says metric_jd_scaling "none", jd_cell_estimate "none" (bayesreg.solve.multilevel.SolverConfig).
Every direction attempt gives one log row (kind "candidate"; the C09 gate fields with gamma for tau, plus gamma_next,
the PCG iteration count, its recursive and true relative residuals, pred, ratio), every accepted step a row of kind
"accept" (tau 1.0, the full step, for the tools that read it), every solve start a row of kind "lm_start" (gamma0).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import norm

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, finite_real, require_finite_result
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.validation import integer_at_least

from .. import memtrace
from ..readonly import adopt_readonly, freeze_new
from .pg import (COMPARISON_EXCEPTION, NONFINITE_STATE, Counters, Handoff, InnerResult, _compare, _confirm,
                 _logvol_parts, _ratio, _row)
from .stopping import (BUDGET_EXHAUSTED, CONVERGED_GRADIENT, INNER_STATUSES, LEVEL_BUDGET_REACHED, MAX_ITERATIONS,
                       NONFINITE_EVALUATION, STAGNATED, TRIAL_BUDGET_EXHAUSTED, budget_expired, gf_b, step_ratio)
from .witness import null_fields

INNER_PG = "pg"
INNER_LM = "lm"
INNER_SOLVERS = (INNER_PG, INNER_LM)          # SolverConfig.inner / run_pair --inner (default pg)
JD_NONE = "none"                              # metric_jd_scaling / jd_cell_estimate under the LM inner (not used)

TRUE_RESIDUAL_FAILED = "TRUE_RESIDUAL_FAILED"          # C09.3-10 (a failed direction)
NONPOSITIVE_PREDICTION = "NONPOSITIVE_PREDICTION"      # C09.3-11
MODEL_RATIO_REJECTED = "MODEL_RATIO_REJECTED"          # C09.3-11 (in the C13-7 safe set)
NONFINITE_LINEAR_SOLVE = "NONFINITE_LINEAR_SOLVE"      # C09.3-13a (d): ends the solve nonfinite_evaluation; detail:
PCG_MAXITER = "pcg_maxiter"           # TRUE_RESIDUAL_FAILED after the PCG used maxiter (finite): C13-7 safe under LM
PCG_RTOL_MET = "pcg_rtol_met"         # TRUE_RESIDUAL_FAILED although the recursive residual met rtol: C13-7 terminates
BREAKDOWN_INDEFINITE = "indefinite"                    # p.Ap <= 0 or r.M^-1 r <= 0 with finite values
BREAKDOWN_NONFINITE = "nonfinite"                      # a nonfinite value (PCG scalar or vector, HVP, true residual, pred)
# The inner statuses of run_lm: PG's (stopping.INNER_STATUSES; line_search_failed is never produced) plus
# trial_budget_exhausted (already a round and run status, C13-3 / C13-7).
LM_INNER_STATUSES = INNER_STATUSES+(TRIAL_BUDGET_EXHAUSTED,)


@dataclass(frozen=True)
class LMConfig:
    """C09.3 algorithm parameters (C09.4-15, C20 D13), all recorded; the defaults are the contract's."""
    rtol: float = 1e-3
    maxiter: int = 200
    gamma_min: float = 1e-12
    gamma_max: float = 1e12
    gamma0_factor: float = 1e-3          # gamma0 = min(gamma_max, max(gamma_min, 1e-3 max(1, lam_b)))
    increase: float = 4.
    decrease: float = .5
    ratio_accept: float = 1e-4
    ratio_low: float = .25
    ratio_high: float = .75
    max_directions: int = 12
    step_tol: float = 1e-12
    stall_steps: int = 5

    def __post_init__(self):
        for name in ("rtol", "gamma_min", "gamma_max", "gamma0_factor", "increase", "decrease", "ratio_accept",
                     "ratio_low", "ratio_high"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        object.__setattr__(self, "step_tol", finite_real(self.step_tol, "step_tol"))
        for name in ("maxiter", "max_directions", "stall_steps"):
            object.__setattr__(self, name, integer_at_least(getattr(self, name), name))
        if (self.rtol >= 1 or self.gamma_min > self.gamma_max or self.increase <= 1 or self.decrease >= 1
                or not self.ratio_accept <= self.ratio_low <= self.ratio_high or self.step_tol < 0):
            raise InputValidationError("LM: rtol, damping bounds or factors, ratio thresholds or step_tol invalid")

    def gamma0(self, lam_b=0.):
        """C09.3-12: gamma0 = min(gamma_max, max(gamma_min, 1e-3 max(1, lam_b)))."""
        return min(self.gamma_max, max(self.gamma_min, self.gamma0_factor*max(1., float(lam_b))))

    def increased(self, gamma):
        """C09.3-12: gamma_max if gamma >= gamma_max/4, else 4 gamma."""
        return self.gamma_max if gamma >= self.gamma_max/self.increase else self.increase*gamma

    def updated(self, gamma, *, accepted, ratio):
        """C09.3-12: the damping after one direction attempt."""
        if accepted and ratio > self.ratio_high:
            return max(self.gamma_min, self.decrease*gamma)
        if accepted and ratio >= self.ratio_low:
            return gamma
        return self.increased(gamma)

    def record(self):
        return dict(contract="C09.3", system="(H_E + gamma (omega_ell/omega0) I) p = -g at the base point",
                    hessian="GN (C06-5); option (b): + K_lam^T K_lam + K_mu^T K_mu + K_nu2^T K_nu2 at the base u",
                    pcg=dict(rtol=self.rtol, atol=0., maxiter=self.maxiter, x0="zero",
                             preconditioner="C09.1 M^-1 with the current gamma (a = c_dat + lam_b + gamma)",
                             stop="recursive residual ||r|| <= rtol ||g|| or maxiter",
                             true_residual="one HVP after the solve: ||H_E p + gamma (omega_ell/omega0) p + g|| <= "
                                           "rtol ||g||, else TRUE_RESIDUAL_FAILED (the direction fails)"),
                    prediction="pred = -g.p - 1/2 p.H_E p (undamped); pred <= 0 -> NONPOSITIVE_PREDICTION",
                    candidate="theta + p (full step)",
                    acceptance_order="finite -> certificate -> energy -> stable comparison (r = 1e-4 pred) -> "
                                     "model ratio certified/pred >= 1e-4",
                    gamma0=f"min(gamma_max, max(gamma_min, {self.gamma0_factor:g} max(1, lam_b)))",
                    gamma_min=self.gamma_min, gamma_max=self.gamma_max, increase=self.increase, decrease=self.decrease,
                    gamma_update="ratio > .75 (accepted): max(gamma_min, gamma/2); .25 <= ratio <= .75 (accepted): "
                                 "unchanged; otherwise gamma_max if gamma >= gamma_max/4, else 4 gamma",
                    gamma_reset="gamma0 at every (level, EM round) start and every reference refresh (option (b))",
                    gamma_max_rule="an attempt rejected at gamma = gamma_max ends the iteration trial_budget_exhausted "
                                   "(C09.3-13a (e))",
                    breakdown="p.Ap <= 0, r.M^-1 r <= 0 or a nonfinite value: NONFINITE_LINEAR_SOLVE (detail "
                              "indefinite | nonfinite), the solve ends nonfinite_evaluation (C09.3-13a (d))",
                    continuation="C13-7 with TRUE_RESIDUAL_FAILED after a PCG stopped at maxiter (finite true "
                                 "residual) in the safe set (C09.3-13a (h), narrowed)",
                    model_ratio=dict(accept=self.ratio_accept, low=self.ratio_low, high=self.ratio_high),
                    max_directions=self.max_directions, step_tol=self.step_tol, stall_steps=self.stall_steps,
                    safe_points="LM iteration top only (C13-5a LM item): wall budget and coarse-level shares; never "
                                "between direction attempts",
                    metric_jd_scaling=JD_NONE, jd_cell_estimate=JD_NONE)


class LMMetric:
    """The C09.1 metric at the LM damping gamma (C09.3-10: the PCG preconditioner "含当前 gamma"):
    backend.build_metric(state, c_dat, b_P, lam_b, gamma), with c_dat and b_P frozen at the freeze point (C09.1-3);
    rebuilt when gamma changes, the last one kept.  `lam_b` also fixes gamma0 (C09.3-12)."""

    def __init__(self, backend, state, *, c_dat, b_P, lam_b=0.):
        self.backend, self.state = backend, state
        self.c_dat, self.b_P, self.lam_b = c_dat, b_P, lam_b
        self._gamma = self._metric = None
        self.builds = 0

    def record(self):
        """The round metric record without gamma (C09.1): c_dat, b_P, lam_b, gamma null, a = c_dat + lam_b + 0 and shift
        = omega_ell a / omega0 -- the values build_metric(gamma=0) would give (Metric.a, Metric.shift), computed from the
        frozen values without building that metric (LMI review minor 3)."""
        a = self.c_dat+self.lam_b+0.
        return dict(c_dat=self.c_dat, b_P=self.b_P, lam_b=self.lam_b, gamma=None, a=a,
                    shift=self.state.grid.omega*a/self.state.omega0)

    def at(self, gamma):
        if self._metric is None or gamma != self._gamma:
            self._metric = None
            self._metric = self.backend.build_metric(self.state, c_dat=self.c_dat, b_P=self.b_P, lam_b=self.lam_b,
                                                     gamma=gamma)
            self._gamma = gamma
            self.builds += 1
        return self._metric


@dataclass
class LMCounters(Counters):
    """pg.Counters plus the LM work: HVPs (PCG and true-residual products), PCG iterations, direction attempts, failed
    true residuals and metric builds."""
    hvps: int = 0
    pcg_iterations: int = 0
    directions: int = 0
    true_residual_failed: int = 0
    metric_builds: int = 0

    def add(self, other):
        super().add(other)
        for name in ("hvps", "pcg_iterations", "directions", "true_residual_failed", "metric_builds"):
            setattr(self, name, getattr(self, name)+getattr(other, name, 0))

    def record(self):
        return dict(super().record(), hvps=self.hvps, pcg_iterations=self.pcg_iterations, directions=self.directions,
                    true_residual_failed=self.true_residual_failed, metric_builds=self.metric_builds)


@dataclass
class LMInnerResult(InnerResult):
    """InnerResult of run_lm plus its damping and PCG summary (`lm`)."""
    lm: dict = field(default_factory=dict)


class LinearSolveBreakdown(FloatingPointError):
    """lm_pcg: the PCG broke down; `detail` is BREAKDOWN_INDEFINITE or BREAKDOWN_NONFINITE (C09.3-13a (d))."""

    def __init__(self, detail, message):
        super().__init__(message)
        self.detail = detail


def _check_curvature(value, what):
    if not np.isfinite(value):
        raise LinearSolveBreakdown(BREAKDOWN_NONFINITE, f"PCG: {what} is not finite")
    if value <= 0:
        raise LinearSolveBreakdown(BREAKDOWN_INDEFINITE, f"PCG: {what} = {value!r} <= 0 (SPD required, C09.3-10)")


@dataclass(frozen=True)
class PCGOutcome:
    """lm_pcg: p, the recursion's iteration count (operator applications, the true-residual HVP excluded), the
    recursive relative residual ||r_k||/||b|| at the stop, and whether it met rtol."""
    p: np.ndarray
    iterations: int
    recursive_residual: float
    converged: bool


def lm_pcg(apply_A, apply_Minv, b, *, rtol, maxiter):
    """C09.3-10: preconditioned CG for A p = b from x0 = 0, atol 0, stopping when the recursive residual ||r_k|| <= rtol
    ||b|| or after maxiter iterations (the true-residual check is the caller's, one HVP).  A and Minv must be SPD:
    p.Ap <= 0 or r.z <= 0 raises LinearSolveBreakdown("indefinite"), a nonfinite scalar LinearSolveBreakdown("nonfinite")
    (a FloatingPointError; a FloatingPointError of apply_A / apply_Minv themselves passes through)."""
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        bnorm = float(norm(b, check_finite=False))
        x = np.zeros_like(b)
        if bnorm == 0:
            return PCGOutcome(x, 0, 0., True)
        r = np.array(b, dtype=np.float64, copy=True)
        z = np.asarray(apply_Minv(r), dtype=np.float64)
        p = z.copy()
        rz = float(r@z)
        _check_curvature(rz, "r.M^-1 r")
        iterations, relative, converged = 0, 1., False
        while iterations < maxiter:
            Ap = np.asarray(apply_A(p), dtype=np.float64)
            pAp = float(p@Ap)
            _check_curvature(pAp, "p.(H_E + gamma I)p")
            alpha = rz/pAp
            x += alpha*p
            r -= alpha*Ap
            iterations += 1
            relative = float(norm(r, check_finite=False))/bnorm
            if not np.isfinite(relative):
                raise LinearSolveBreakdown(BREAKDOWN_NONFINITE, "PCG: nonfinite residual")
            if relative <= rtol:
                converged = True
                break
            if iterations >= maxiter:
                break
            z = np.asarray(apply_Minv(r), dtype=np.float64)
            rz_new = float(r@z)
            _check_curvature(rz_new, "r.M^-1 r")
            p = z+(rz_new/rz)*p
            rz = rz_new
    if not np.all(np.isfinite(x)):
        raise LinearSolveBreakdown(BREAKDOWN_NONFINITE, "PCG direction: nonfinite numerical result")
    return PCGOutcome(x, iterations, relative, converged)


def run_lm(objective, base, *, theta, cert, metric, certify, target, max_steps, config=LMConfig(), log=None,
           context=None, gradient=None, on_accept=None, budget=None, level_deadline=None):
    """One LM-PCG inner solve (C09.3) from the certified base point.

    objective: DirectObjective / ProximalObjective (needs evaluate, gradient, hvp(point, v), compare or reduction);
    base: its point at theta (certified, `cert`); metric: an LMMetric (at(gamma).solve(r) = M_gamma^-1 r; lam_b);
    certify(u, grid) -> CertOutcome (CPU FP64); target: the normalized gradient threshold; max_steps: accepted-step
    limit (n_in, or the C13-4 / C13-4a cap); log(row), gradient, on_accept(theta, point, cert), budget (WallBudget),
    level_deadline (LevelDeadline): as run_pg (module docstring for the safe points).  Returns an LMInnerResult.
    """
    if isinstance(base, Handoff):
        base = base.take()
    state = objective.state
    grid = state.grid
    context = dict(context or {})
    log = (lambda row: None) if log is None else log
    counters = LMCounters()
    theta = adopt_readonly(theta)
    energies, norms = [base.energy], []
    small_steps, accepted, last_iteration = 0, 0, []
    iteration = 0
    G = None
    damping = finite_real(grid.omega, "omega_ell", positive=True)/finite_real(state.omega0, "omega0", positive=True)
    lam_b = getattr(metric, "lam_b", 0.)
    gamma0 = config.gamma0(lam_b)
    gamma = gamma0
    stats = dict(gamma0=gamma0, lam_b=lam_b, gammas_used=[], pcg_iterations=[], true_residuals=[])

    def summary():
        used = stats["gammas_used"]
        its = stats["pcg_iterations"]
        res = [r for r in stats["true_residuals"] if r is not None]
        return dict(gamma0=gamma0, lam_b=lam_b, gamma_end=gamma, directions=len(used),
                    gamma_min_used=min(used) if used else None, gamma_max_used=max(used) if used else None,
                    pcg_iterations_total=int(sum(its)), pcg_iterations_max=max(its) if its else None,
                    true_residual_max=max(res) if res else None, true_residual_failed=counters.true_residual_failed,
                    metric_builds=counters.metric_builds)

    def result(status, reason):
        return LMInnerResult(status, reason, base, theta, cert, gradient, G, accepted, energies, norms, last_iteration,
                             counters, lm=summary())

    log(_row(context, kind="lm_start", objective=objective.name, gamma0=gamma0, lam_b=lam_b, max_steps=max_steps,
             reset="gamma0 at this inner solve's start (C09.3-12)"))
    try:
        if gradient is None:
            gradient = objective.gradient(base)
            counters.gradients += 1
        G = gf_b(gradient, grid.omega, state.omega0)
        G = _confirm(objective, base, G, target, log, context, 0)
    except FloatingPointError as exc:
        gradient, G = None, None
        norms.append(None)
        return result(NONFINITE_EVALUATION, f"gradient at the start: {exc}")
    norms.append(G)

    while True:
        if G <= target:
            return result(CONVERGED_GRADIENT, f"gradient norm {G:.6e} <= target {target:.6e}")
        if accepted >= max_steps:
            return result(MAX_ITERATIONS, f"{accepted} accepted steps (limit {max_steps})")
        if small_steps >= config.stall_steps:
            return result(STAGNATED, f"step ratio <= {config.step_tol:g} on {small_steps} consecutive steps")
        where = dict(context, objective=objective.name, iteration=iteration+1, attempt=0)
        if budget_expired(budget, where):
            return result(BUDGET_EXHAUSTED, f"wall-clock budget {budget.seconds:g} s used up before LM iteration "
                          f"{iteration+1} ({accepted} accepted steps; the last accepted point is kept)")
        if level_deadline is not None and level_deadline.reached(dict(where, at="lm_iteration_top")):
            # C13-5a LM item: the same safe point and clock reading as the budget check above
            return result(LEVEL_BUDGET_REACHED, f"level {level_deadline.level} deadline D = "
                          f"{level_deadline.deadline:g} s passed before LM iteration {iteration+1} ({accepted} "
                          "accepted steps; the last accepted point is kept; C13-5a)")
        iteration += 1
        last_iteration = []
        accepted_point = None
        g_norm = float(norm(gradient, check_finite=False))
        rhs = -gradient

        def apply_hvp(v):
            counters.hvps += 1
            return objective.hvp(base, v)

        for attempt in range(config.max_directions):
            counters.candidates += 1
            counters.directions += 1
            stats["gammas_used"].append(gamma)
            shift = gamma*damping
            values = dict(kind="candidate", objective=objective.name, iteration=iteration, attempt=attempt, tau=1.,
                          gamma=gamma, gamma_next=None, damping_shift=shift, pcg_iterations=None, pcg_converged=None,
                          pcg_residual=None, true_residual=None, slope=None, pred=None, cert_status=None,
                          cert_reason=None, min_jd_lower_bound=None, energy=None, ared=None, guard=None,
                          certified=None, required=None, ratio=None, accepted=False, reason=None, **null_fields())
            reason = cert_reason = None
            ratio = None
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    M = metric.at(gamma)
                    builds = getattr(metric, "builds", None)
                    if builds is not None:
                        counters.metric_builds = builds

                    def apply_A(v):
                        return apply_hvp(v)+shift*v

                    def apply_Minv(r):
                        counters.metric_solves += 1
                        return M.solve(r)
                    outcome_pcg = lm_pcg(apply_A, apply_Minv, rhs, rtol=config.rtol, maxiter=config.maxiter)
                    p = outcome_pcg.p
                    counters.pcg_iterations += outcome_pcg.iterations
                    Hp = np.asarray(apply_hvp(p), dtype=np.float64)          # the one true HVP (C09.3-10)
                    true = float(norm(Hp+shift*p+gradient, check_finite=False))/g_norm
                    if not np.isfinite(true):
                        raise LinearSolveBreakdown(BREAKDOWN_NONFINITE, "true residual is not finite")
            except FloatingPointError as exc:
                values.update(reason=NONFINITE_LINEAR_SOLVE, detail=getattr(exc, "detail", BREAKDOWN_NONFINITE),
                              exception=str(exc))
                log(_row(context, **values))
                counters.reject(NONFINITE_LINEAR_SOLVE)
                last_iteration.append((NONFINITE_LINEAR_SOLVE, None))
                return result(NONFINITE_EVALUATION, f"LM direction (PCG / HVP) at iteration {iteration}: {exc}")
            stats["pcg_iterations"].append(outcome_pcg.iterations)
            stats["true_residuals"].append(true)
            values.update(pcg_iterations=outcome_pcg.iterations, pcg_converged=outcome_pcg.converged,
                          pcg_residual=outcome_pcg.recursive_residual, true_residual=true)
            memtrace.mark("lm.direction", iteration=iteration, attempt=attempt, context=context)
            if not true <= config.rtol:
                reason = TRUE_RESIDUAL_FAILED
                counters.true_residual_failed += 1
                # C09.3-13a (h): the second slot of the last_iteration entry says which case (stopping.continuation)
                cert_reason = PCG_RTOL_MET if outcome_pcg.converged else PCG_MAXITER
                values["true_residual_case"] = cert_reason
            else:
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        slope = float(gradient@p)
                        pred = float(-slope-.5*float(p@Hp))
                    require_finite_result(pred, "LM prediction")
                except FloatingPointError as exc:
                    values.update(reason=NONFINITE_LINEAR_SOLVE, detail=BREAKDOWN_NONFINITE, exception=str(exc))
                    log(_row(context, **values))
                    counters.reject(NONFINITE_LINEAR_SOLVE)
                    last_iteration.append((NONFINITE_LINEAR_SOLVE, None))
                    return result(NONFINITE_EVALUATION, f"LM prediction at iteration {iteration}: {exc}")
                values.update(slope=slope, pred=pred)
                if not pred > 0:
                    reason = NONPOSITIVE_PREDICTION
                else:
                    with np.errstate(over="ignore", invalid="ignore"):
                        candidate = theta+p
                    if not np.all(np.isfinite(candidate)):
                        reason = NONFINITE_STATE
                    else:
                        u = freeze_new(inject_free(candidate, grid))
                        outcome = certify(u, grid)
                        counters.certificate_calls += 1
                        memtrace.mark("lm.certify", iteration=iteration, attempt=attempt, passed=bool(outcome.passed),
                                      context=context)
                        values.update(cert_status=outcome.status, cert_reason=outcome.cert_reason,
                                      min_jd_lower_bound=outcome.min_jd_lower_bound)
                        fields = getattr(outcome, "witness_fields", None)
                        if callable(fields):
                            values.update(fields())
                        estimate = getattr(outcome, "witness_estimate", None)
                        if estimate is not None:
                            values["witness_estimate"] = estimate
                        if not outcome.passed:
                            reason, cert_reason = outcome.rejection, outcome.cert_reason
                        else:
                            try:
                                trial = objective.evaluate(u)
                            except FloatingPointError:
                                trial = InvalidTrial(NONFINITE_STATE, "floating-point exception in the evaluation", 1)
                            counters.energy_evaluations += 1
                            if isinstance(trial, InvalidTrial):
                                reason = trial.reason
                            else:
                                values["energy"] = trial.energy
                                parts = _logvol_parts(trial)
                                if parts is not None:
                                    values["logvol_parts"] = parts
                                required = config.ratio_accept*pred
                                values["required"] = required
                                try:
                                    reduction = _compare(objective, base, trial, required=required, context="lm",
                                                         gf_ratio=_ratio(G, target))
                                    ratio = float(reduction.certified)/pred
                                    require_finite_result(ratio, "LM model ratio")
                                except FloatingPointError as exc:
                                    values.update(reason=COMPARISON_EXCEPTION, exception=str(exc))
                                    log(_row(context, **values))
                                    counters.reject(COMPARISON_EXCEPTION)
                                    last_iteration.append((COMPARISON_EXCEPTION, None))
                                    return result(NONFINITE_EVALUATION, f"stable comparison: {exc}")
                                values.update(ared=reduction.ared, guard=reduction.guard,
                                              certified=reduction.certified, ratio=ratio)
                                if hasattr(reduction, "record"):
                                    values["comparison"] = reduction.record()
                                reason = reduction.reason
                                if reason is None and not ratio >= config.ratio_accept:
                                    reason = MODEL_RATIO_REJECTED
                                if reason is None:
                                    accepted_point = (candidate, u, trial, outcome, ratio, pred, outcome_pcg)
            gamma_next = config.updated(gamma, accepted=reason is None, ratio=ratio)
            values.update(gamma_next=gamma_next, accepted=reason is None, reason=reason)
            log(_row(context, **values))
            if reason is None:
                gamma_used, gamma = gamma, gamma_next
                break
            counters.reject(reason)
            last_iteration.append((reason, cert_reason))
            candidate = u = trial = p = Hp = None
            at_top = gamma >= config.gamma_max
            gamma = gamma_next
            if at_top:
                # C09.3-13a (e): every later attempt of this iteration would repeat this one bit for bit
                return result(TRIAL_BUDGET_EXHAUSTED, f"gamma_max: the attempt {attempt+1} of iteration {iteration} with "
                              f"gamma = gamma_max = {config.gamma_max:g} was rejected ({reason}); the iteration ends "
                              "(C09.3-13a (e))")
        if accepted_point is None:
            return result(TRIAL_BUDGET_EXHAUSTED, f"{config.max_directions} direction attempts rejected at iteration "
                          f"{iteration} (gamma now {gamma:.6g})")
        candidate, u, trial, outcome, ratio, pred, outcome_pcg = accepted_point
        accepted_point = None
        candidate = freeze_new(candidate)
        relative = step_ratio(candidate-theta, theta, grid)
        small_steps = small_steps+1 if relative <= config.step_tol else 0
        base, theta, cert = trial, candidate, outcome
        accepted += 1
        energies.append(base.energy)
        if on_accept is not None:
            on_accept(theta, base, cert)
        try:
            gradient = objective.gradient(base)
            counters.gradients += 1
            G = gf_b(gradient, grid.omega, state.omega0)
            G = _confirm(objective, base, G, target, log, context, iteration)
        except FloatingPointError as exc:
            gradient, G = None, None
            norms.append(None)
            return result(NONFINITE_EVALUATION, f"gradient at the accepted state: {exc}")
        norms.append(G)
        memtrace.mark("lm.accepted", iteration=iteration, context=context)
        log(_row(context, kind="accept", objective=objective.name, iteration=iteration, tau=1., gamma=gamma_used,
                 gamma_next=gamma, ratio=ratio, pred=pred, pcg_iterations=outcome_pcg.iterations, energy=base.energy,
                 gradient_norm=G, step_ratio=relative, accepted_steps=accepted, state_hash=cert.state_hash,
                 min_jd_lower_bound=cert.min_jd_lower_bound))
