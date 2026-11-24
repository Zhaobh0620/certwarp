"""C09.1-C09.2 PG inner solver: Poisson-preconditioned gradient, BB2, certificate-gated stable Armijo.

One inner solve works on the free vector theta in R^{3 N_int} with a frozen objective E (F_B for outer
option (a), E_v = F_B + P(., v) for option (b)) and a frozen metric M (C09.1-3); sigma, w, eta are frozen.

Per iteration (C09.2):
  p = -M^-1 g (finite, g.p < 0 required, C09.2-4);
  proposal: tau = 1 on the first step of the solve, then BB2 tau = s.y / (y.M^-1 y) with
    s = theta_k - theta_{k-1}, y = g_k - g_{k-1}, M^-1 y = p_{k-1} - p_k, only from accepted steps; taken
    when s.y > 0, finite and cos(s, y) > 1e-12, else the protected 2 tau_prev; clipped to [1e-8, 10]
    (the rules of the mainline `_step_proposal`, eb808be jd_curl/three_d/solvers/poisson_gradient.py);
  domain cap: tau <= .9 tau_dom, tau_dom the exact linear bound that keeps every interior node x_a +
    u_a + tau p_a in the open box (the numpy branch of the mainline `sampling_step_bound`);
  candidate theta + tau p, in the fixed acceptance order (C07-5, C09.4-14):
    (i) finite -> (ii) certificate (CPU FP64; a candidate that does not pass is never evaluated) ->
    (iii) E (InvalidTrial: INVALID_DOMAIN, NONPOSITIVE_GAUSS_JD, NONFINITE_STATE) -> (iv) stable
    comparison (C08: ENERGY_RESOLUTION if |ared| <= s_E, NO_STRICT_DECREASE if ared - s_E <= 0) ->
    Armijo ared - s_E >= c1 tau |g.p| (ARMIJO_REJECTED);
  rejection halves tau, at most 25 halvings (26 candidates), then `line_search_failed`; a rejection
  never moves the base point, the gradient or the BB history.
End of the solve (C09.2-9, in this priority after every accepted step and at the start):
  converged_gradient (||g|| / sqrt(omega_ell/omega0) <= target), max_iterations (accepted == max_steps),
  stagnated (step_ratio <= 1e-12 on 5 consecutive accepted steps, C09.2-8), line_search_failed.
  nonfinite_evaluation / non_descent_direction end the solve as numerical failures (C13-7 terminate).
  budget_exhausted (development switch --wall-seconds, D44): a `stopping.WallBudget` found used up at the top of an
  iteration (after the three endings above) or before a further candidate of the line search (attempt >= 1); the
  solve ends at its last accepted point, a rejected candidate never moves it.
  level_budget_reached (C13-5a, SCH; `level_deadline`, a stopping.LevelDeadline or None): at the top of an iteration,
  right after the budget check and with its clock reading, the coarse level's cumulative deadline D_ell has passed; the
  solve ends at its last accepted point.  Never checked inside a line search (only the budget is).  Without a deadline
  the path is the code before SCH.
Every candidate gives one log row (C09 gate): iteration, attempt, tau, certificate status and map
reason, energy (None unless the certificate passed and E was evaluated), ared, s_E, certified, the
Armijo requirement, accepted, reason.  Additive records (IMPL27, IMPL27b): every candidate row carries the C16 witness
fields witness_cell, witness_xi, witness_type, witness_cell_min_j53 (bayesreg.solve.witness; values for a
JACOBIAN_VIOLATION with a witness point, null otherwise -- also for candidates that passed or never reached the
certificate) and, when the backend computed it, witness_estimate; an evaluated candidate under C05-2b carries
logvol_parts (the Gauss and closed27 parts of the log-volume energy and the minimum J over the closed27 points with
its cell).  No decision reads them.
JD affine scaling (C09.1-3a, `PGConfig.metric_jd_scaling`, default "affine_v1"): the caller (bayesreg.solve.outer)
wraps the frozen M into `jd_scaling.JDScaledMetric`, so that p = -S~ M^-1 S~ g and BB2 uses S~ M^-1 S~; run_pg
feeds every candidate's rejection reason to its refresh rule and, when the rule fired, recomputes S~ at the next
accepted state (M unchanged; a `jd_scaling` log row) -- at the start of the following iteration, so only when the
solve goes on -- and recomputes the stored previous direction with the new operator for BB2.  With "off" the metric is M itself and run_pg takes exactly the C09.1/C09.2 path (bitwise).
The refresh passes the accepted base's CertOutcome (`cert`, from the certify_map call that accepted it) to the metric;
only the jd_cell_estimate options "certificate" (C09.1-3a last item, D47; `PGConfig.jd.cell_estimate`) and
"certificate_bernstein" (D53) read it.  Every refresh is logged (kind "jd_scaling"), followed by a "warning" row when
its certificate_bernstein check (b) found violations (jd_scaling.j53_warning_row); when the refresh succeeded but the
BB2 previous direction is not finite (the solve ends nonfinite_evaluation), certificate_bernstein still logs both
rows before ending (JDTI review minor 1), the other estimates keep the log of the code before (no row there).
jd_cell_estimate (work package DEF, 2026-10-31; contract C09.1-3a last item, D47): `PGConfig.jd` left unset gives
JDScalingConfig() under "affine_v1" (cell estimate "certificate", the default since DEF) and None under "off", where
the option has no effect and `PGConfig.jd_cell_estimate` (also in the record) is "none"; an explicit `jd` with "off" is
refused (before DEF: only the default cell estimate was allowed with "off").
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import norm

from jd_curl.types import InvalidTrial
from jd_curl.validation import InputValidationError, finite_real, readonly_copy
from jd_curl.three_d.boundary import inject_free
from jd_curl.three_d.validation import integer_at_least

from .. import memtrace
from ..readonly import adopt_readonly, freeze_new
from . import proximal as _prox
from .jd_scaling import (AFFINE_V1, CERTIFICATE_BERNSTEIN, MODES as JD_SCALING_MODES, NONE as JD_NONE, OFF,
                         JDScaledMetric, JDScalingConfig, j53_warning_row)
from .witness import null_fields
from .stopping import (BUDGET_EXHAUSTED, CONVERGED_GRADIENT, LEVEL_BUDGET_REACHED, LINE_SEARCH_FAILED, MAX_ITERATIONS,
                       NON_DESCENT_DIRECTION, NONFINITE_EVALUATION, STAGNATED, budget_expired, gf_b, step_ratio)

NONFINITE_STATE = "NONFINITE_STATE"
ARMIJO_REJECTED = "ARMIJO_REJECTED"
COMPARISON_EXCEPTION = "COMPARISON_EXCEPTION"          # FloatingPointError in the stable comparison (terminates)
NO_POSITIVE_DOMAIN_STEP = "NO_POSITIVE_DOMAIN_STEP"


@dataclass(frozen=True)
class PGConfig:
    """C09.2 algorithm parameters (C09.4-15), all recorded.

    `jd` left unset is filled in __post_init__ (JDScalingConfig() under "affine_v1"), so
    dataclasses.replace(PGConfig(), metric_jd_scaling="off") is refused: it carries the filled jd into "off", where an
    explicit jd is refused (C09.1-3a, DEF).  Construct PGConfig(metric_jd_scaling="off") directly (or replace with
    jd=None as well).  (MIN1, review DEF minor 5; behaviour unchanged.)"""
    initial_step: float = 1.
    step_min: float = 1e-8
    step_max: float = 10.
    domain_fraction: float = .9
    c1: float = 1e-4
    max_backtracks: int = 25
    step_tol: float = 1e-12
    stall_steps: int = 5
    cosine_min: float = 1e-12
    metric_jd_scaling: str = AFFINE_V1                       # C09.1-3a: "affine_v1" (default) or "off"
    jd: JDScalingConfig | None = None    # unset: JDScalingConfig() under "affine_v1", None under "off" (DEF)

    def __post_init__(self):
        for name in ("initial_step", "step_min", "step_max", "domain_fraction", "c1", "cosine_min"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name, positive=True))
        object.__setattr__(self, "step_tol", finite_real(self.step_tol, "step_tol"))
        object.__setattr__(self, "max_backtracks", integer_at_least(self.max_backtracks, "max_backtracks", 0))
        object.__setattr__(self, "stall_steps", integer_at_least(self.stall_steps, "stall_steps"))
        if self.step_min > self.step_max or self.c1 >= 1 or self.domain_fraction >= 1 or self.step_tol < 0:
            raise InputValidationError("PG: step bounds, c1, domain fraction or step_tol invalid")
        if self.metric_jd_scaling not in JD_SCALING_MODES:
            raise InputValidationError(f"metric_jd_scaling: one of {JD_SCALING_MODES}")
        if self.metric_jd_scaling == OFF:
            if self.jd is not None:
                raise InputValidationError("jd_cell_estimate: no effect with metric_jd_scaling 'off' (C09.1-3a, DEF): "
                                           "leave jd unset (recorded as 'none'); an explicit value is refused")
        elif self.jd is None:
            object.__setattr__(self, "jd", JDScalingConfig())
        elif not isinstance(self.jd, JDScalingConfig):
            raise InputValidationError("PG: jd must be a JDScalingConfig")

    @property
    def jd_cell_estimate(self):
        """The jd_cell_estimate in effect: the JDScalingConfig's under "affine_v1", "none" under "off" (C09.1-3a,
        DEF)."""
        return JD_NONE if self.jd is None else self.jd.cell_estimate

    def record(self):
        scaled = self.metric_jd_scaling == AFFINE_V1
        return dict(direction="p = -S~ M^-1 S~ g (C09.1-3a)" if scaled else "p = -M^-1 g",
                    step="BB2 (accepted steps only)", initial_step=self.initial_step,
                    bb2_interval=[self.step_min, self.step_max], bb2_cosine_min=self.cosine_min,
                    fallback="protected 2 tau_prev", domain_fraction=self.domain_fraction, c1=self.c1,
                    max_backtracks=self.max_backtracks, step_tol=self.step_tol, stall_steps=self.stall_steps,
                    acceptance_order="finite -> certificate -> energy -> stable comparison -> Armijo",
                    metric_jd_scaling=self.metric_jd_scaling, jd_cell_estimate=self.jd_cell_estimate,
                    jd_scaling=self.jd.record() if scaled else None)


# ---- objectives -----------------------------------------------------------------------------------

class DirectObjective:
    """E = F_B (outer option (a)).

    `compare(base, trial, required=..., context=..., gf_ratio=...)` is the comparison the solver uses: the
    backend's `stable_compare` (GPU path of C08-8 with its fallback rule, which needs the required decrease r)
    when the backend has one, else `reduction` (the CPU reference, C08-1..7, unchanged).
    `confirm_gradient_norm(point, G, target)` is the backend's C08-8(f) recheck when it has one, else G."""
    name = "F_B"

    def __init__(self, backend, state):
        self.backend, self.state = backend, state

    def evaluate(self, u):
        return self.backend.evaluate(self.state, u)

    def gradient(self, point):
        return self.backend.gradient(self.state, point)

    def hvp(self, point, v):
        """H v (GN of F_B at point, C06-5); used by the LM inner only (C09.3-10, bayesreg.solve.lm)."""
        return self.backend.hvp(self.state, point, v)

    def reduction(self, base, trial):
        return self.backend.stable_reduction(self.state, base, trial)

    def compare(self, base, trial, *, required=0., context="pg", gf_ratio=None):
        compare = getattr(self.backend, "stable_compare", None)
        if compare is None:
            return self.reduction(base, trial)
        return compare(self.state, base, trial, required=required, energy="F_B", context=context, gf_ratio=gf_ratio)

    def confirm_gradient_norm(self, point, G, target, *, context="pg"):
        confirm = getattr(self.backend, "confirm_gradient_norm", None)
        if confirm is None:
            return G, None
        return confirm(self.state, point, G, target, energy="F_B", context=context)

    @staticmethod
    def fb(point):
        return point


class ProximalObjective:
    """E_v = F_B + P(., v) with the frozen reference v (outer option (b), C10-2).  With a backend that has
    `stable_compare`, the proximal blocks stay the exact host computation of bayesreg.solve.proximal and enter
    the comparison as ExtraBlocks; the F_B part takes the GPU path (C08-8), the decision is made on E_v."""
    name = "E_v"

    def __init__(self, backend, state, v, params):
        self.backend, self.state, self.params = backend, state, params
        self.v = readonly_copy(v)
        self._prox = getattr(backend, "proximal", _prox)
        self._v_hash = None

    @property
    def v_hash(self):
        if self._v_hash is None:
            from .stopping import state_hash
            self._v_hash = state_hash(self.v)
        return self._v_hash

    def evaluate(self, u):
        fb = self.backend.evaluate(self.state, u)
        if isinstance(fb, InvalidTrial):
            return fb
        return _prox.combine(fb, self._prox.evaluate(self.state, fb.u, self.v, self.params))

    def gradient(self, point):
        with np.errstate(over="raise", invalid="raise"):
            return (self.backend.gradient(self.state, point.fb)
                    + self._prox.gradient(self.state, point.prox, self.params))

    def hvp(self, point, v):
        """H_E v of option (b): the GN of F_B plus the three proximal blocks at the base u (C09.3-10; LM inner only).
        The proximal part is always the host computation of bayesreg.solve.proximal."""
        with np.errstate(over="raise", invalid="raise"):
            return (np.asarray(self.backend.hvp(self.state, point.fb, v), dtype=np.float64)
                    + _prox.hvp(self.state, point.prox, v, self.params))

    def reduction(self, base, trial):
        fb = self.backend.stable_reduction(self.state, base.fb, trial.fb)
        return _prox.combined_reduction(fb, self._prox.change(self.state, base.prox, trial.prox, self.params),
                                        base, trial)

    def compare(self, base, trial, *, required=0., context="pg", gf_ratio=None):
        compare = getattr(self.backend, "stable_compare", None)
        if compare is None:
            return self.reduction(base, trial)
        from ..backends.torch64.stable import ExtraBlocks
        changes = self._prox.change(self.state, base.prox, trial.prox, self.params)
        extra = ExtraBlocks({name: changes[name] for name in _prox.PROX_BLOCKS}, base.prox.total, trial.prox.total)
        return compare(self.state, base.fb, trial.fb, required=required, energy="E_v", reference=self.v_hash,
                       extra=extra, context=context, gf_ratio=gf_ratio)

    def confirm_gradient_norm(self, point, G, target, *, context="pg"):
        confirm = getattr(self.backend, "confirm_gradient_norm", None)
        if confirm is None:
            return G, None
        return confirm(self.state, point.fb, G, target, energy="E_v", reference=self.v_hash,
                       extra_gradient=lambda: self._prox.gradient(self.state, point.prox, self.params),
                       context=context)

    @staticmethod
    def fb(point):
        return point.fb


class Handoff:
    """MEM (memory only): a point handed to run_pg so that the caller keeps no reference to it -- an argument stays
    referenced by the caller's frame for the whole call in CPython, a Handoff only references the point until run_pg
    takes it out.  run_pg(objective, Handoff(point), ...) is run_pg(objective, point, ...) in every value; the round
    start point is then freed as soon as the first step is accepted (host u and channel residuals, the device channel
    point)."""
    __slots__ = ("_point",)

    def __init__(self, point):
        self._point = point

    def take(self):
        point, self._point = self._point, None
        if point is None:
            raise InputValidationError("Handoff: the point was already taken")
        return point


# ---- step proposal and domain bound -------------------------------------------------------------

def bb2_proposal(config, theta, gradient, direction, previous):
    """C09.2-5: (tau, rule, s.y, y.M^-1 y); previous = (theta, gradient, direction, tau) of the last
    accepted step or None.  The rules of the mainline `_step_proposal` (eb808be)."""
    def protect(value):
        return min(config.step_max, max(config.step_min, value))

    if previous is None:
        return protect(config.initial_step), "initial", None, None
    old_theta, old_gradient, old_direction, old_tau = previous
    fallback = protect(2*min(old_tau, config.step_max/2))
    sy = yy = None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            s, y, inverse_y = theta-old_theta, gradient-old_gradient, old_direction-direction
            for value in (s, y, inverse_y):
                if not np.all(np.isfinite(value)):
                    raise FloatingPointError("BB2 secants")
            sy = float(s@y)
            if not np.isfinite(sy):
                raise FloatingPointError("BB2 secant curvature")
            if sy <= 0:
                return fallback, "fallback_nonpositive_curvature", sy, yy
            cosine = float((s/norm(s))@(y/norm(y)))
            if not np.isfinite(cosine):
                raise FloatingPointError("BB2 cosine")
            if cosine <= config.cosine_min:
                return fallback, "fallback_small_curvature", sy, yy
            yy = float(y@inverse_y)
            if not np.isfinite(yy):
                raise FloatingPointError("BB2 metric curvature")
            if yy <= 0:
                return fallback, "fallback_nonpositive_metric_curvature", sy, yy
            proposal = sy/yy
            if not np.isfinite(proposal) or proposal <= 0:
                return fallback, "fallback_nonpositive_step", sy, yy
    except FloatingPointError:
        return fallback, "fallback_nonfinite_secant", sy, yy
    return protect(proposal), "bb2", sy, yy


def sampling_step_bound(grid, u, direction):
    """C09.2-6: the exact linear bound tau_dom keeping every interior node x_a + u_a + tau p_a inside the
    open box (inf when p = 0); the numpy branch of the mainline `sampling_step_bound` (eb808be)."""
    v = np.asarray(direction).reshape((3,)+grid.interior_shape)
    points = (grid.identity()+u)[1:-1, 1:-1, 1:-1]
    bound = np.inf
    with np.errstate(over="ignore", invalid="raise", divide="raise"):
        for r, axis in enumerate(grid.axes):
            pos, neg = v[r] > 0, v[r] < 0
            if np.any(pos):
                bound = min(bound, float(np.min((axis[-1]-points[..., r][pos])/v[r][pos])))
            if np.any(neg):
                bound = min(bound, float(np.min((axis[0]-points[..., r][neg])/v[r][neg])))
    return bound


# ---- result ---------------------------------------------------------------------------------------

@dataclass
class Counters:
    candidates: int = 0
    certificate_calls: int = 0
    energy_evaluations: int = 0
    gradients: int = 0
    metric_solves: int = 0
    rejections: dict = field(default_factory=dict)

    def reject(self, reason):
        self.rejections[reason] = self.rejections.get(reason, 0)+1

    def add(self, other):
        for name in ("candidates", "certificate_calls", "energy_evaluations", "gradients", "metric_solves"):
            setattr(self, name, getattr(self, name)+getattr(other, name))
        for reason, count in other.rejections.items():
            self.rejections[reason] = self.rejections.get(reason, 0)+count

    def record(self):
        return dict(candidates=self.candidates, certificate_calls=self.certificate_calls,
                    energy_evaluations=self.energy_evaluations, gradients=self.gradients,
                    metric_solves=self.metric_solves, rejections=dict(sorted(self.rejections.items())))


@dataclass
class InnerResult:
    status: str
    reason: str
    point: object                 # last accepted point of E (the base when nothing was accepted)
    theta: np.ndarray
    cert: object                  # CertOutcome of `point`
    gradient: np.ndarray | None   # grad E at `point` (None after a nonfinite gradient)
    gradient_norm: float | None   # ||grad E|| / sqrt(omega_ell/omega0)
    accepted: int
    energies: list                # E after every accepted step (E of the base first)
    gradient_norms: list          # the normalized gradient norm at the base and after every accepted step
    last_iteration: list          # (reason, certificate map reason) of every rejected candidate of the last iteration
    counters: Counters


def _row(context, /, **values):
    """dict(context) updated with `values`.  `context` is positional-only (GFR): the C08-8(f) recheck record has its
    own key `context` ("pg"), an ordinary key of the row here (it was a TypeError); every other row is unchanged."""
    row = dict(context)
    row.update(values)
    return row


def _compare(objective, base, trial, *, required, context, gf_ratio):
    """The solver's comparison: objective.compare (C08-8 when the backend has a GPU path) or the reference
    objective.reduction (objectives without compare, e.g. test doubles)."""
    compare = getattr(objective, "compare", None)
    if compare is None:
        return objective.reduction(base, trial)
    return compare(base, trial, required=required, context=context, gf_ratio=gf_ratio)


def _confirm(objective, point, G, target, log, context, iteration):
    """C08-8(f): the backend's double-sided 1.01 recheck of a device gradient norm (identity without one).
    The gf_recheck row: run_pg's context, kind, the counter (iteration), then the backend's record -- the key order of
    solve_proximal's outer rows (context, kind, outer, record; FUP1, review GFR minor 1)."""
    confirm = getattr(objective, "confirm_gradient_norm", None)
    if confirm is None or G is None:
        return G
    value, check = confirm(point, G, target)
    if check is not None:
        log(_row(context, kind="gf_recheck", iteration=iteration, **check))
    return value


def _ratio(G, target):
    return G/target if G is not None and target > 0 else None


def _logvol_parts(point):
    """energies["logvol_parts"] of an evaluated point (its F_B Evaluation for E_v), or None (C05-2b record only)."""
    energies = getattr(point, "energies", None)
    if energies is None:
        energies = getattr(getattr(point, "fb", None), "energies", None)
    return None if not isinstance(energies, dict) else energies.get("logvol_parts")


def run_pg(objective, base, *, theta, cert, metric, certify, target, max_steps, config=PGConfig(), log=None,
           context=None, gradient=None, on_accept=None, budget=None, level_deadline=None):
    """One PG inner solve (C09.2) from the certified base point.

    objective: DirectObjective / ProximalObjective; base: its point at theta (already certified, `cert`);
    metric: frozen, with solve(g) = M^-1 g; certify(u, grid) -> CertOutcome (CPU FP64); target: the
    normalized gradient threshold; max_steps: accepted-step limit (n_in, or the remaining C13-4 cap);
    log(row): called with every candidate and accepted-step row; gradient: grad E at base if known;
    on_accept(theta, point, cert): called after every accepted step (option (a) records its formal commits);
    budget: a `stopping.WallBudget` or None (module docstring: budget_exhausted); level_deadline: a
    `stopping.LevelDeadline` on that budget or None (module docstring: level_budget_reached).
    A FloatingPointError in the gradient, the direction, the stable comparison or a JD scaling refresh ends the
    solve with nonfinite_evaluation and the last accepted point (hard constraint 4: accepted states are never
    dropped).  A `jd_scaling.JDScaledMetric` metric gets the C09.1-3a refresh rule (module docstring).
    """
    if isinstance(base, Handoff):
        base = base.take()                # MEM: this frame holds the only reference from here on
    state = objective.state
    grid = state.grid
    context = dict(context or {})
    log = (lambda row: None) if log is None else log
    counters = Counters()
    theta = adopt_readonly(theta)
    energies, norms = [base.energy], []
    previous, small_steps, accepted, last_iteration = None, 0, 0, []
    iteration = 0
    G = None
    scaling = metric if isinstance(metric, JDScaledMetric) else None      # C09.1-3a refresh rule

    def result(status, reason):
        return InnerResult(status, reason, base, theta, cert, gradient, G, accepted, energies, norms,
                           last_iteration, counters)

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
            return result(BUDGET_EXHAUSTED, f"wall-clock budget {budget.seconds:g} s used up before iteration "
                          f"{iteration+1} ({accepted} accepted steps; the last accepted point is kept)")
        if level_deadline is not None and level_deadline.reached(dict(where, at="pg_iteration_top")):
            # C13-5a: the same safe point and clock reading as the budget check above; before the JD refresh
            return result(LEVEL_BUDGET_REACHED, f"level {level_deadline.level} deadline D = "
                          f"{level_deadline.deadline:g} s passed before iteration {iteration+1} ({accepted} accepted "
                          "steps; the last accepted point is kept; C13-5a)")
        iteration += 1
        if scaling is not None and scaling.refresh_due is not None:
            # C09.1-3a: the rule fired during the last line search, which ended with an accepted step (a failed line
            # search ends the solve): S~ recomputed at that accepted state (M unchanged) before this direction, and
            # only when the solve goes on.  BB2 keeps its secant pair with the new operator: p_{k-1} :=
            # -S~' M^-1 S~' g_{k-1}, so that p_{k-1} - p_k = S~' M^-1 S~' y.
            record = None

            def log_scaling(record):
                log(_row(context, kind="jd_scaling", objective=objective.name, **record))
                warning = j53_warning_row(record)            # C09.1-3a certificate_bernstein check (b) (JDTI)
                if warning is not None:
                    log(_row(context, objective=objective.name, iteration=iteration, **warning))
            try:
                record = scaling.refresh(inject_free(theta, grid), cert=cert, iteration=iteration,
                                         accepted_steps=accepted)
                if previous is not None:
                    old_theta, old_gradient, _, old_tau = previous
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        old_direction = -metric.solve(old_gradient)
                    counters.metric_solves += 1
                    if not np.all(np.isfinite(old_direction)):
                        raise FloatingPointError("BB2 previous direction under the refreshed metric is not finite")
                    previous = (old_theta, old_gradient, old_direction, old_tau)
            except FloatingPointError as exc:
                if record is not None and record.get("cell_estimate") == CERTIFICATE_BERNSTEIN:
                    # JDTI review minor 1: the refresh was computed (it is in the round record), only BB2 failed; its
                    # jd_scaling row and check-(b) warning go to the log before the solve ends.  The other estimates
                    # keep the log of the code before (no row on this path).
                    log_scaling(record)
                return result(NONFINITE_EVALUATION, f"JD scaling refresh: {exc}")
            log_scaling(record)
            memtrace.mark("pg.jd_refresh", iteration=iteration, context=context)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                direction = -metric.solve(gradient)
                counters.metric_solves += 1
                if not np.all(np.isfinite(direction)):
                    raise FloatingPointError("PG direction is not finite")
                slope = float(gradient@direction)
        except FloatingPointError as exc:
            return result(NONFINITE_EVALUATION, f"direction: {exc}")
        if not np.isfinite(slope) or slope >= 0:
            return result(NON_DESCENT_DIRECTION, f"g.p = {slope!r} is not negative")
        proposal, rule, sy, yy = bb2_proposal(config, theta, gradient, direction, previous)
        try:
            bound = sampling_step_bound(grid, base.u, direction)
        except FloatingPointError as exc:
            return result(NONFINITE_EVALUATION, f"domain bound: {exc}")
        if not bound > 0:
            last_iteration = [(NO_POSITIVE_DOMAIN_STEP, None)]       # not in the C13-7 safe set: terminate
            return result(LINE_SEARCH_FAILED, NO_POSITIVE_DOMAIN_STEP)
        tau = min(proposal, config.domain_fraction*bound)
        memtrace.mark("pg.direction", iteration=iteration, context=context)
        last_iteration = []
        accepted_point = None
        for attempt in range(config.max_backtracks+1):
            if attempt > 0 and budget_expired(budget, dict(context, objective=objective.name, iteration=iteration,
                                                           attempt=attempt)):
                return result(BUDGET_EXHAUSTED, f"wall-clock budget {budget.seconds:g} s used up in the line search "
                              f"of iteration {iteration} before candidate {attempt} ({accepted} accepted steps; the "
                              "last accepted point is kept)")
            counters.candidates += 1
            values = dict(kind="candidate", objective=objective.name, iteration=iteration, attempt=attempt, tau=tau,
                          rule=rule if attempt == 0 else "halving", proposal=proposal, domain_bound=bound,
                          bb2_sy=sy if attempt == 0 else None, bb2_yy=yy if attempt == 0 else None,
                          cert_status=None, cert_reason=None, min_jd_lower_bound=None, energy=None, ared=None,
                          guard=None, certified=None, required=None, accepted=False, reason=None, **null_fields())
            with np.errstate(over="ignore", invalid="ignore"):
                candidate = theta+tau*direction
            reason = cert_reason = None
            if not np.all(np.isfinite(candidate)):
                reason = NONFINITE_STATE
            else:
                u = freeze_new(inject_free(candidate, grid))   # MEM: read-only in place, so the evaluation shares it
                outcome = certify(u, grid)
                counters.certificate_calls += 1
                memtrace.mark("pg.certify", iteration=iteration, attempt=attempt, passed=bool(outcome.passed),
                              context=context)
                values.update(cert_status=outcome.status, cert_reason=outcome.cert_reason,
                              min_jd_lower_bound=outcome.min_jd_lower_bound)
                fields = getattr(outcome, "witness_fields", None)
                if callable(fields):                             # C16 witness fields (bayesreg.solve.witness)
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
                    memtrace.mark("pg.evaluate", iteration=iteration, attempt=attempt, context=context)
                    if isinstance(trial, InvalidTrial):
                        reason = trial.reason
                    else:
                        values["energy"] = trial.energy
                        parts = _logvol_parts(trial)
                        if parts is not None:                    # C05-2b record (Gauss / closed27 parts, min J)
                            values["logvol_parts"] = parts
                        required = config.c1*tau*abs(slope)
                        values["required"] = required
                        try:
                            reduction = _compare(objective, base, trial, required=required, context="pg",
                                                 gf_ratio=_ratio(G, target))
                        except FloatingPointError as exc:
                            values.update(reason=COMPARISON_EXCEPTION, exception=str(exc))
                            log(_row(context, **values))
                            counters.reject(COMPARISON_EXCEPTION)
                            last_iteration.append((COMPARISON_EXCEPTION, None))
                            return result(NONFINITE_EVALUATION, f"stable comparison: {exc}")
                        memtrace.mark("pg.compare", iteration=iteration, attempt=attempt, context=context)
                        values.update(ared=reduction.ared, guard=reduction.guard, certified=reduction.certified)
                        if hasattr(reduction, "record"):
                            values["comparison"] = reduction.record()
                        reason = reduction.reason
                        if reason is None and not reduction.certified >= required:
                            reason = ARMIJO_REJECTED
                        if reason is None:
                            accepted_point = (candidate, u, trial, outcome)
            values.update(accepted=reason is None, reason=reason)
            log(_row(context, **values))
            if scaling is not None:
                scaling.observe(reason)
            if reason is None:
                break
            counters.reject(reason)
            last_iteration.append((reason, cert_reason))
            candidate = u = trial = None      # MEM: a rejected candidate's arrays and evaluation go before the next one
            tau = tau/2
        if accepted_point is None:
            return result(LINE_SEARCH_FAILED, f"{config.max_backtracks+1} candidates rejected")
        candidate, u, trial, outcome = accepted_point
        accepted_point = None
        candidate = freeze_new(candidate)      # MEM: theta + tau p is new and owned here: read-only in place
        relative = step_ratio(candidate-theta, theta, grid)
        small_steps = small_steps+1 if relative <= config.step_tol else 0
        previous_entry = (theta, gradient, direction, tau)
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
        previous = previous_entry
        memtrace.mark("pg.accepted", iteration=iteration, context=context)
        log(_row(context, kind="accept", objective=objective.name, iteration=iteration, tau=tau, energy=base.energy,
                 gradient_norm=G, step_ratio=relative, accepted_steps=accepted, state_hash=cert.state_hash,
                 min_jd_lower_bound=cert.min_jd_lower_bound))

