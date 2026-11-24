"""C13 stopping: GF_B, the per-(level, round) threshold, hard caps, status strings, continuation table.

GF_B(u) = ||grad_theta F_B(u)||_2 / sqrt(omega_ell/omega0) (C13-1, C20 D32), computed by the copied
`gradient_norm_mass` (BLAS nrm2, scale before squaring; C13-6).  Threshold (C13-2)
    T_B^(ell, r) = 1e-8 + 1e-6 G0^(ell, r),  G0^(ell, r) = GF_B(u = 0; sigma^(ell, r), w^(ell, r)),
recomputed for every (level, EM round) with that round's frozen sigma and w (never carried over).
Hard caps per (level, round) (C13-4, C20 D19): 500 accepted steps; 60 outer commits (option (b)).

Copied helpers (contract header, "路径补充"): from seq_vp_3d tag freeze-20261022, commit eb808be,
  - gradient_norm_mass: jd_curl/solvers/common.py lines 182-191 (blob sha256 GNM_SOURCE_SHA256), verbatim
    except the annotation FloatArray -> np.ndarray;
  - step_ratio: jd_curl/three_d/solvers/common.py lines 70-81 (blob sha256 STEP_SOURCE_SHA256), the
    numpy branch verbatim; the E51 fast_checks branch (an exact parallel max of the same values, same
    float division) is dropped.
tests/test_bayesreg/solve/test_stopping.py executes the eb808be originals and compares bit for bit.

Status vocabulary (C13-3, C09.2-9, C10-4, C12-2, C07-6).  Inner (one PG solve): INNER_STATUSES.  Round
(one (level, round) of option (a) or (b)): ROUND_STATUSES.  Run: RUN_STATUSES, with the wording of
RUN_WORDING; only `converged_regularized_gradient_B` may be called convergence ("B 线收敛").

Wall-clock budget (development switch `--wall-seconds`, contract C13-3 appendix and C20 D44, pending review; C13-5
keeps "no time budget" for every formal run): `WallBudget` is checked only at safe points -- the top of every PG
iteration, before every further candidate of a line search, the top of every option (b) outer iteration and before
every transfer -- where the state in hand is the last formal commit.  It ends the inner solve, the round and the run
with `budget_exhausted` ("预算收尾"): on the native level the last formal commit is exported and re-certified as for
any ending; before the native level the run ends `stopped_before_native` (no disp.npz).  Uncommitted inner progress
(option (b)) is never exported.  A natural ending detected at the same check wins (the budget has the lowest priority).

Per-level caps and per-level wall-clock shares (work package SCHI; contract C13-4a, C13-5a, C20 D51 / D52, draft SCH
2026-10-31, path only, defaults unchanged):
- `Caps.max_accepted_per_level = (a_0, a_1, a_2, a_3)` (`Caps.per_level`): a_ell replaces the 500 of C13-4 on every
  (level, round) of level ell (`Caps.for_level`); hitting it is still `cap_reached` with cap "max_accepted".  Without it
  `for_level` returns the Caps itself and `record()` is unchanged (C13 SCH gate 12).
- `LevelShares(shares=(s_0, s_1, s_2), wall_seconds=W)`: the cumulative deadlines D_ell = (s_0 + ... + s_ell) W of the
  three coarse levels (partial sums left to right in float64, then times W).  `LevelDeadline` checks D_ell only at the
  top of a PG iteration (option (b)'s inner PG included) and at the top of an option (b) outer iteration, right after
  the WallBudget check of the same point and with that check's clock reading (`WallBudget.last_elapsed`; one reading per
  safe point, no extra count); never inside a line search or before a transfer.  Priority at one point: natural ending >
  budget_exhausted > level_budget_reached.  `level_budget_reached` ("粗层预算转层") is an inner and round status only,
  never a run status and never on the native level; the C13-7 table continues on it unconditionally (as cap_reached).
"""
from dataclasses import dataclass
import hashlib
import time

import numpy as np
from scipy.linalg import norm

from jd_curl.validation import InputValidationError, as_float64_array, finite_real, require_finite_result
from jd_curl.three_d.validation import integer_at_least

GNM_SOURCE = ("eb808be", "jd_curl/solvers/common.py", 182, 191)
GNM_SOURCE_SHA256 = "571f77aa9ab4e4bacc8ba0e51f5d6a5e7306593080f0827ad40dbf8f98d05e79"
STEP_SOURCE = ("eb808be", "jd_curl/three_d/solvers/common.py", 70, 81)
STEP_SOURCE_SHA256 = "9ee12583455f175afe7b0c4f20e790f935094db00f4a985be4143df27e0f1765"

ABS_TOL = 1e-8                    # C13-2
REL_TOL = 1e-6                    # C13-2
MAX_ACCEPTED = 500                # C13-4 / D19
MAX_OUTER = 60                    # C13-4 / D19

# ---- status strings -----------------------------------------------------------------------------
CONVERGED_GRADIENT = "converged_gradient"            # GF_B (or the E_v norm in (b)'s inner) <= T_B
MAX_ITERATIONS = "max_iterations"                    # inner accepted-step limit (n_in, or the C13-4 cap)
STAGNATED = "stagnated"                              # C09.2-8
LINE_SEARCH_FAILED = "line_search_failed"            # C09.2-7
NONFINITE_EVALUATION = "nonfinite_evaluation"        # gradient / direction of an accepted state not finite
NON_DESCENT_DIRECTION = "non_descent_direction"      # g.p >= 0 (C09.2-4)
BUDGET_EXHAUSTED = "budget_exhausted"                # --wall-seconds reached at a safe point (development switch, D44)
LEVEL_BUDGET_REACHED = "level_budget_reached"        # C13-5a coarse-level share deadline D_ell (SCH; inner / round only)
INNER_STATUSES = (CONVERGED_GRADIENT, MAX_ITERATIONS, STAGNATED, LINE_SEARCH_FAILED, NONFINITE_EVALUATION,
                  NON_DESCENT_DIRECTION, BUDGET_EXHAUSTED, LEVEL_BUDGET_REACHED)

CAP_REACHED = "cap_reached"
OUTER_STAGNATED = "outer_stagnated"
OUTER_NO_PROGRESS = "outer_no_progress"
OUTER_INNER_FAILED = "outer_inner_failed"
TRIAL_BUDGET_EXHAUSTED = "trial_budget_exhausted"    # LM only (C09.3); never produced by PG
ROUND_STATUSES = (CONVERGED_GRADIENT, CAP_REACHED, STAGNATED, LINE_SEARCH_FAILED, TRIAL_BUDGET_EXHAUSTED,
                  OUTER_STAGNATED, OUTER_NO_PROGRESS, OUTER_INNER_FAILED, NONFINITE_EVALUATION,
                  NON_DESCENT_DIRECTION, BUDGET_EXHAUSTED, LEVEL_BUDGET_REACHED)
LEVEL_BUDGET_WORDING = "粗层预算转层"                   # C13-3 SCH item: "L ell 粗层预算转层", never 收敛 / 完成

CONVERGED_B = "converged_regularized_gradient_B"
LEVEL_FAILED = "level_failed"
TRANSFER_REJECTED = "transfer_rejected"
INITIALIZATION_FAILED = "initialization_failed"
STOPPED_BEFORE_NATIVE = "stopped_before_native"      # development switch --max-levels < 4 (not in C13)
SUBJECT_SEALED = "subject_sealed"
PROVENANCE_REJECTED = "provenance_rejected"
# Export (hard constraint 1): the CPU certificate of the state to be exported did not pass; final_native.npz and
# cert.json are kept as failure evidence, no disp.npz (S1 review item 1; the contract wording is the lead's).
FINAL_CERTIFICATE_REJECTED = "final_certificate_rejected"
RUN_STATUSES = (CONVERGED_B, CAP_REACHED, STAGNATED, OUTER_STAGNATED, LINE_SEARCH_FAILED, TRIAL_BUDGET_EXHAUSTED,
                OUTER_NO_PROGRESS, OUTER_INNER_FAILED, BUDGET_EXHAUSTED, LEVEL_FAILED, TRANSFER_REJECTED,
                INITIALIZATION_FAILED, STOPPED_BEFORE_NATIVE, SUBJECT_SEALED, PROVENANCE_REJECTED,
                FINAL_CERTIFICATE_REJECTED)
RUN_WORDING = {
    CONVERGED_B: "B 线收敛",
    CAP_REACHED: "上限收尾",
    BUDGET_EXHAUSTED: "预算收尾",
    STAGNATED: "停滞收尾", OUTER_STAGNATED: "停滞收尾",
    LINE_SEARCH_FAILED: "受阻收尾", TRIAL_BUDGET_EXHAUSTED: "受阻收尾", OUTER_NO_PROGRESS: "受阻收尾",
    OUTER_INNER_FAILED: "受阻收尾",
    LEVEL_FAILED: "运行失败", TRANSFER_REJECTED: "运行失败", INITIALIZATION_FAILED: "运行失败",
    STOPPED_BEFORE_NATIVE: "原生层之前按开发开关停止（不是收尾，无 disp.npz）",
    SUBJECT_SEALED: "运行被拒（受试者未在白名单）", PROVENANCE_REJECTED: "运行被拒（分割 provenance 不合格）",
    FINAL_CERTIFICATE_REJECTED: "运行失败（输出态的 CPU 证书未通过；无 disp.npz）",
}
assert set(RUN_WORDING) == set(RUN_STATUSES)

# C13-7 continuation table
CONTINUE_ALWAYS = (CONVERGED_GRADIENT, CAP_REACHED)
# C13-7 row of C13-5a (SCH): continue unconditionally, as cap_reached; a coarse round status only (not in FINAL_STATUS:
# the native level never ends on a share), so it is kept apart from CONTINUE_ALWAYS
CONTINUE_LEVEL_SHARE = (LEVEL_BUDGET_REACHED,)
CONTINUE_IF_SAFE = (STAGNATED, OUTER_STAGNATED, LINE_SEARCH_FAILED, TRIAL_BUDGET_EXHAUSTED, OUTER_NO_PROGRESS,
                    OUTER_INNER_FAILED)
SAFE_INNER_FOR_OUTER_FAILED = (LINE_SEARCH_FAILED, TRIAL_BUDGET_EXHAUSTED)
SAFE_REJECTIONS = ("INVALID_DOMAIN", "JACOBIAN_VIOLATION", "FEASIBILITY_UNCERTAIN", "ENERGY_RESOLUTION",
                   "NO_STRICT_DECREASE", "ARMIJO_REJECTED", "MODEL_RATIO_REJECTED")
SAFE_UNCERTAIN_CERT_REASON = "subdivision_budget"
# C09.3-13a (h) (LMI; narrowed after the second review round, 2026-11-01): under the LM inner only, a TRUE_RESIDUAL_FAILED
# is in the safe set ("条件见下" rows) only when the PCG stopped at maxiter with a finite true residual -- the second slot
# of its last_iteration entry is "pcg_maxiter" (as FEASIBILITY_UNCERTAIN is safe only with subdivision_budget).  A
# TRUE_RESIDUAL_FAILED whose recursive residual met rtol ("pcg_rtol_met") still terminates, and so does a nonfinite true
# residual (run_lm records it NONFINITE_LINEAR_SOLVE).  PG and the default path never see either.
TRUE_RESIDUAL_FAILED = "TRUE_RESIDUAL_FAILED"
LM_SAFE_TRUE_RESIDUAL_CASE = "pcg_maxiter"
# Final (native, last round) status of a round that ends without termination (C13-3).
FINAL_STATUS = {CONVERGED_GRADIENT: CONVERGED_B, CAP_REACHED: CAP_REACHED, STAGNATED: STAGNATED,
                OUTER_STAGNATED: OUTER_STAGNATED, LINE_SEARCH_FAILED: LINE_SEARCH_FAILED,
                TRIAL_BUDGET_EXHAUSTED: TRIAL_BUDGET_EXHAUSTED, OUTER_NO_PROGRESS: OUTER_NO_PROGRESS,
                OUTER_INNER_FAILED: OUTER_INNER_FAILED}


# ---- copied helpers (eb808be) ---------------------------------------------------------------------

def gradient_norm_mass(gradient: np.ndarray, omega: float) -> float:
    gradient = as_float64_array(gradient, "gradient", check_finite=False)
    require_finite_result(gradient, "gradient")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        # BLAS nrm2 scales before squaring; numpy's direct dot can underflow.
        value = float(norm(gradient.ravel(), check_finite=False)/np.sqrt(omega))
    require_finite_result(value, "mass gradient norm")
    if value == 0. and np.any(gradient != 0):
        raise FloatingPointError("mass gradient norm: nonzero result underflowed")
    return value


def step_ratio(delta, theta, grid):
    delta = as_float64_array(delta, "step", (3*grid.n_free_scalar,))
    theta = as_float64_array(theta, "base theta", delta.shape)
    length = norm([axis[-1]-axis[0] for axis in grid.axes])
    value = float(np.max(np.abs(delta))/max(length, np.max(np.abs(theta))))
    require_finite_result(value, "relative step")
    return value


# ---- GF_B, threshold, caps ----------------------------------------------------------------------

def gf_b(gradient, omega_level, omega0):
    """C13-1: ||g|| / sqrt(omega_ell/omega0) of a free-vector gradient."""
    ratio = finite_real(omega_level, "omega_level", positive=True)/finite_real(omega0, "omega0", positive=True)
    return gradient_norm_mass(gradient, ratio)


def threshold(g0, *, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    """C13-2: T_B = abs_tol + rel_tol G0 (1e-8 + 1e-6 G0)."""
    g0 = finite_real(g0, "G0")
    if g0 < 0:
        raise InputValidationError("G0: must be nonnegative")
    return abs_tol+rel_tol*g0


N_LEVELS = 4                      # C12-1: levels 0..3, 3 the native level
N_COARSE = N_LEVELS-1             # C13-5a: shares s_0, s_1, s_2 of the three coarse levels


@dataclass(frozen=True)
class Caps:
    """C13-4 hard caps per (level, round).

    max_accepted_per_level (C13-4a, D51; SCH): (a_0, a_1, a_2, a_3), the accepted-step cap of every (level, round) of
    level ell; build it with `Caps.per_level`.  max_accepted is then the common value when the four agree, None otherwise
    (C16 SCH: max_accepted_per_round records null).  The solver takes `for_level(ell)` for each level: the Caps itself
    without per-level caps (the code before SCH), else Caps(a_ell, max_outer).  record() is unchanged."""
    max_accepted: int | None = MAX_ACCEPTED
    max_outer: int = MAX_OUTER
    max_accepted_per_level: tuple | None = None

    def __post_init__(self):
        if self.max_accepted_per_level is None:
            object.__setattr__(self, "max_accepted", integer_at_least(self.max_accepted, "max_accepted", 0))
        object.__setattr__(self, "max_outer", integer_at_least(self.max_outer, "max_outer", 0))
        if self.max_accepted_per_level is not None:
            per = tuple(integer_at_least(a, "max_accepted_per_level", 0) for a in self.max_accepted_per_level)
            if len(per) != N_LEVELS:
                raise InputValidationError(f"max_accepted_per_level: {N_LEVELS} values (a_0..a_3, C13-4a)")
            if self.max_accepted != _uniform(per):
                raise InputValidationError("max_accepted with max_accepted_per_level: the common value of the four "
                                           "levels or None (C13-4a; use Caps.per_level)")
            object.__setattr__(self, "max_accepted_per_level", per)

    @classmethod
    def per_level(cls, caps, max_outer=MAX_OUTER):
        """C13-4a: the Caps of the per-level accepted caps (a_0, a_1, a_2, a_3)."""
        per = tuple(integer_at_least(a, "max_accepted_per_level", 0) for a in caps)
        return cls(max_accepted=_uniform(per), max_outer=max_outer, max_accepted_per_level=per)

    def for_level(self, level):
        """The caps of every (level, round) of `level`: self without per-level caps, else Caps(a_level, max_outer)."""
        if self.max_accepted_per_level is None:
            return self
        return Caps(max_accepted=self.max_accepted_per_level[level], max_outer=self.max_outer)

    def accepted_cap(self, level):
        """a_level (C16 SCH round key cap_accepted)."""
        return self.for_level(level).max_accepted

    def record(self):
        return dict(max_accepted_per_round=self.max_accepted, max_outer_commits_per_round=self.max_outer)


def _uniform(values):
    return values[0] if len(set(values)) == 1 else None


@dataclass(frozen=True)
class LevelShares:
    """C13-5a (D52; SCH, development switch): shares (s_0, s_1, s_2) of the wall-clock budget W of `--wall-seconds`.

    Each s finite with 0 < s < 1 and s_0 + s_1 + s_2 < 1 (float64, left to right).  cumulative[ell] = s_0 + ... + s_ell
    (left to right) and deadlines[ell] = cumulative[ell] * W, D_ell in seconds from the start of run_pair's main()."""
    shares: tuple
    wall_seconds: float

    def __post_init__(self):
        shares = tuple(self.shares)
        if len(shares) != N_COARSE:
            raise InputValidationError(f"level_time_shares: {N_COARSE} values (s_0, s_1, s_2; C13-5a)")
        values = []
        for s in shares:
            value = finite_real(s, "level_time_shares")
            if not 0 < value < 1:
                raise InputValidationError(f"level_time_shares: every share in (0, 1), got {value!r} (C13-5a)")
            values.append(value)
        if not self._partial(values)[-1] < 1:
            raise InputValidationError("level_time_shares: s_0 + s_1 + s_2 < 1 required (C13-5a)")
        object.__setattr__(self, "shares", tuple(values))
        object.__setattr__(self, "wall_seconds", finite_real(self.wall_seconds, "wall_seconds", positive=True))

    @staticmethod
    def _partial(values):
        out, total = [], 0.
        for k, value in enumerate(values):
            total = value if k == 0 else total+value
            out.append(total)
        return tuple(out)

    @property
    def cumulative(self):
        return self._partial(self.shares)

    @property
    def deadlines(self):
        return tuple(c*self.wall_seconds for c in self.cumulative)

    def deadline(self, budget, level):
        """The LevelDeadline of coarse `level` on `budget` (the run's WallBudget with W = wall_seconds)."""
        if budget is None or budget.seconds != self.wall_seconds:
            raise InputValidationError("level_time_shares: the run's WallBudget with W = wall_seconds is required "
                                       "(C13-5a: only with --wall-seconds)")
        return LevelDeadline(budget, level=level, share=self.shares[level], cumulative=self.cumulative[level],
                             deadline=self.deadlines[level])


class LevelDeadline:
    """C13-5a: the cumulative deadline D_ell of one coarse level, compared with the WallBudget's reading of the same safe
    point (`WallBudget.last_elapsed`; no clock reading and no check count of its own).  `reached(where)` is called only
    right after `budget.expired(...)` returned False at that point; the first True records `hit` (where, elapsed_seconds,
    check = the WallBudget's checks count)."""

    def __init__(self, budget, *, level, share, cumulative, deadline):
        self.budget, self.level, self.share, self.cumulative, self.deadline = budget, level, share, cumulative, deadline
        self.hit = None

    def reached(self, where):
        if self.hit is not None:
            return True
        elapsed = self.budget.last_elapsed
        if elapsed is None or elapsed < self.deadline:        # the comparison of WallBudget.expired
            return False
        self.hit = dict(where=where, elapsed_seconds=elapsed, check=self.budget.checks)
        return True


class WallBudget:
    """Wall-clock budget of a development run (`--wall-seconds`; C13-3 appendix, D44 -- not part of the method).

    seconds: positive budget; clock: a monotonic clock in seconds (default time.monotonic, looked up at every call);
    started: the clock value the budget counts from (default: now).  `expired(where)` is called only at the solver's
    safe points; the first call that finds the budget used up records where and when (`hit`), and every later call
    returns True."""

    def __init__(self, seconds, *, clock=None, started=None):
        self.seconds = finite_real(seconds, "wall_seconds", positive=True)
        self._clock = clock
        self.started = self.now() if started is None else finite_real(started, "started")
        self.checks = 0
        self.hit = None
        self.last_elapsed = None          # SCH: the reading of the last check (C13-5a shares compare with it)

    def now(self):
        return float(time.monotonic() if self._clock is None else self._clock())

    def elapsed(self):
        return self.now()-self.started

    def expired(self, where=None):
        self.checks += 1
        if self.hit is not None:
            return True
        elapsed = self.elapsed()
        self.last_elapsed = elapsed
        if elapsed < self.seconds:
            return False
        self.hit = dict(where=where, elapsed_seconds=elapsed, check=self.checks)
        return True

    def record(self):
        return dict(wall_seconds=self.seconds, exhausted=self.hit is not None, hit=self.hit, checks=self.checks,
                    elapsed_seconds=self.elapsed(), clock="time.monotonic" if self._clock is None else "injected",
                    counts_from="run_pair main() start (before the inputs are read)",
                    safe_points="PG iteration top; before every further line-search candidate; option (b) outer "
                                "iteration top; before every transfer",
                    contract="C13-3 appendix / C20 D44 (development switch, pending review; C13-5: formal runs use "
                             "no time budget)")


def budget_expired(budget, where):
    """True when a WallBudget is given and used up (None: no budget)."""
    return budget is not None and budget.expired(where)


def identity_gradient(backend, state, *, blocks=None):
    """grad F_B at u = 0 for this state (C13-2 G0; blocks for the C13 gate)."""
    zero = np.zeros(state.grid.shape+(3,))
    ev = backend.evaluate(state, zero)
    if not hasattr(ev, "energy"):
        raise FloatingPointError(f"F_B at u = 0 is not defined ({ev.reason}: {ev.message})")
    return backend.gradient(state, ev) if blocks is None else backend.gradient(state, ev, blocks=blocks)


def g0_of(backend, state):
    """C13-2: G0^(ell, r) = GF_B(u = 0) with the state's frozen sigma and w; on the backend's CPU reference
    when it has one (C08-8(f): G0 is a CPU value)."""
    backend = getattr(backend, "reference", None) or backend
    return gf_b(identity_gradient(backend, state), state.grid.omega, state.omega0)


# ---- C13-7 continuation table -------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    proceed: bool
    reason: str

    def record(self):
        return dict(proceed=self.proceed, reason=self.reason)


def c_order_bytes(array):
    """MEM (memory only): the C-order bytes of `array` as a buffer -- array.tobytes(order="C") without the copy when
    the array is C-contiguous (np.ascontiguousarray copies only otherwise); the same bytes, hence the same digest."""
    array = np.ascontiguousarray(array)
    return memoryview(array.reshape(-1).view(np.uint8))


def state_hash(u):
    """sha256 of the C-order float64 bytes (C07-7, the digest of certify_map)."""
    return hashlib.sha256(c_order_bytes(np.asarray(u, dtype=np.float64))).hexdigest()


def last_iteration_safe(last_iteration, *, lm=False):
    """Every rejection of the round's last iteration is in the C13-7 safe set (FEASIBILITY_UNCERTAIN only
    with the map-level reason subdivision_budget; with lm=True -- the LM inner, C09.3-13a (h) -- also TRUE_RESIDUAL_FAILED
    with the second slot "pcg_maxiter": the PCG used maxiter and the true residual is finite).  Returns (safe,
    offending)."""
    offending = []
    for reason, cert_reason in last_iteration:
        if lm and reason == TRUE_RESIDUAL_FAILED:
            if cert_reason != LM_SAFE_TRUE_RESIDUAL_CASE:
                offending.append(f"{reason}/{cert_reason}")
        elif reason not in SAFE_REJECTIONS:
            offending.append(reason)
        elif reason == "FEASIBILITY_UNCERTAIN" and cert_reason != SAFE_UNCERTAIN_CERT_REASON:
            offending.append(f"{reason}/{cert_reason}")
    return not offending, offending


def continuation(status, *, last_iteration=(), inner_status=None, commit_u=None, commit_cert=None, inner="pg"):
    """C13-7: continue (M step / next round / transfer) or terminate the run (level_failed).

    status: the round status; last_iteration: (rejection reason, map-level certificate reason) of every
    candidate of the round's last iteration; inner_status: the inner status behind outer_inner_failed;
    commit_u / commit_cert: the round's last formal commit (or its legal initial state) and its certificate;
    inner: "pg" (default, the table as before LMI) or "lm" (C09.3-13a (h), narrowed: TRUE_RESIDUAL_FAILED of run_lm is in
    the safe set only with the second slot "pcg_maxiter"; NONFINITE_LINEAR_SOLVE and every other reason as before).
    """
    if inner not in ("pg", "lm"):
        raise InputValidationError("continuation: inner 'pg' or 'lm'")
    if status in CONTINUE_ALWAYS:
        return Decision(True, f"{status}: continue (C13-7)")
    if status in CONTINUE_LEVEL_SHARE:
        return Decision(True, f"{status}: continue (C13-7, as cap_reached; C13-5a coarse-level share)")
    if status not in CONTINUE_IF_SAFE:
        return Decision(False, f"{status}: terminate (C13-7 'other')")
    if status == OUTER_INNER_FAILED and inner_status not in SAFE_INNER_FOR_OUTER_FAILED:
        return Decision(False, f"{status} with inner status {inner_status}: terminate (C13-7)")
    safe, offending = last_iteration_safe(last_iteration, lm=inner == "lm")
    if not safe:
        return Decision(False, f"{status}: last-iteration rejections outside the safe set {sorted(set(offending))}")
    if commit_cert is None or commit_u is None:
        return Decision(False, f"{status}: no certified commit to continue from")
    if commit_cert.min_jd_lower_bound is None or not commit_cert.min_jd_lower_bound > 0:
        return Decision(False, f"{status}: last commit's certificate min J lower bound is not > 0")
    if commit_cert.state_hash != state_hash(commit_u):
        return Decision(False, f"{status}: last commit's state_hash differs from the stored array")
    return Decision(True, f"{status}: safe-set rejections only, certified commit (C13-7)")


def final_status(status, decision):
    """C13-3 for the native level's last round: the round status mapped to the run status, or
    level_failed for the 'terminate' rows of the table (decision from `continuation`)."""
    if not decision.proceed:
        return LEVEL_FAILED
    if status in CONTINUE_LEVEL_SHARE:
        raise AssertionError(f"{status}: a coarse-level round status, never the native level's ending (C13-5a)")
    return FINAL_STATUS[status]
